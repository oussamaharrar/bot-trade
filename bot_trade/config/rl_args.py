import argparse, os, sys
from typing import Optional, Dict, Any
from .rl_paths import (
    DEFAULT_AGENTS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_REPORTS_DIR,
    DEFAULT_MEMORY_FILE, DEFAULT_KB_FILE
)

def _parse_list(s: str):
    s = (s or "").strip().strip("[]")
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_net_arch(s: str) -> Dict[str, Any]:
    pi, vf = [1024, 1024], [1024, 1024]
    s = (s or "").strip()
    if s:
        for part in s.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                k, v = k.strip().lower(), v.strip()
                if k in ("pi", "vf"):
                    lst = _parse_list(v)
                    if lst:
                        if k == "pi": pi = lst
                        else: vf = lst
    return dict(net_arch=dict(pi=pi, vf=vf))

def build_policy_kwargs(net_arch_str: str, activation: str, ortho_init: bool) -> Dict[str, Any]:
    """Build policy kwargs lazily importing torch."""
    from torch import nn  # type: ignore

    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leakyrelu": nn.LeakyReLU,
    }
    kw = parse_net_arch(net_arch_str)
    kw["activation_fn"] = mapping.get((activation or "relu").strip().lower(), nn.ReLU)
    kw["ortho_init"] = bool(ortho_init)
    return kw


# ---------------------------------------------------------------------------
# Validation helpers (thin shims for train_rl orchestrator)
# ---------------------------------------------------------------------------

def validate_args(args):
    """Light validation of numeric arguments."""
    for k in ("n_envs", "n_steps", "batch_size", "total_steps"):
        if getattr(args, k, 0) <= 0:
            raise ValueError(f"[ARGS] {k} يجب أن يكون > 0.")
    return args


def clamp_batch(args):
    """Ensure batch_size respects rollout and divides it cleanly."""
    rollout = int(args.n_envs) * int(args.n_steps)
    if int(args.batch_size) > rollout:
        import logging

        logging.warning(
            "[PPO] batch_size (%d) > n_envs*n_steps (%d) — سيتم تقليمه",
            args.batch_size,
            rollout,
        )
    batch = min(int(args.batch_size), rollout)
    batch = (batch // int(args.n_envs)) * int(args.n_envs)
    if rollout % batch != 0:
        import math as _math

        new_bs = _math.gcd(rollout, batch)
        logging.info(
            "[PPO] batch_size=%d not divisor of rollout=%d → %d",
            batch,
            rollout,
            new_bs,
        )
        batch = max(int(args.n_envs), new_bs)
    args.batch_size = max(int(args.n_envs), int(batch))
    return args


def auto_shape_resources(args):
    """Auto-derive n_envs/n_steps/batch_size based on hardware when unset."""
    import os
    import logging
    import psutil  # type: ignore
    import torch  # type: ignore

    cpu_cores = os.cpu_count() or 2
    gpu = torch.cuda.is_available()
    user_set = getattr(args, "n_envs", None)
    if user_set is None or int(user_set) <= 0:
        if gpu:
            args.n_envs = 4
        else:
            args.n_envs = min(max(2, cpu_cores // 2), 16)
        if gpu and args.n_envs < 4:
            args.n_envs = 4
        auto = True
    else:
        logging.info("[AUTO] disabled (user provided n_envs=%s)", args.n_envs)
        auto = False
    if getattr(args, "batch_size", 0) > args.n_envs * args.n_steps:
        logging.warning(
            "[PPO] batch_size (%d) > n_envs*n_steps (%d) — clipping",
            args.batch_size,
            args.n_envs * args.n_steps,
        )
        args.batch_size = args.n_envs * args.n_steps
    clamp_batch(args)
    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        ram_gb = float("nan")
    if auto:
        logging.info(
            "[AUTO] cpu_cores=%s ram_gb=%.1f n_envs=%s n_steps=%s batch_size=%s",
            cpu_cores,
            ram_gb,
            args.n_envs,
            args.n_steps,
            args.batch_size,
        )
    else:
        logging.info(
            "cpu_cores=%s ram_gb=%.1f n_envs=%s n_steps=%s batch_size=%s",
            cpu_cores,
            ram_gb,
            args.n_envs,
            args.n_steps,
            args.batch_size,
        )
    return args

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train reinforcement learning agent",
        epilog=(
            "Example: python -m bot_trade.train_rl "
            "--symbol BTCUSDT --frame 1m --device cpu --allow-synth"
        ),
    )
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--frame", type=str, default="1m")
    ap.add_argument("--policy", type=str, default="MlpPolicy")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--n-envs", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=3_000_000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=str, default="0.0")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--sde", action="store_true")
    ap.add_argument("--net-arch", type=str, default="pi=[1024,1024];vf=[1024,1024]")
    ap.add_argument("--activation", type=str, default="relu")
    ap.add_argument("--ortho-init", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-every", type=int, default=200_000)
    ap.add_argument("--resume-auto", action="store_true")
    ap.add_argument("--resume-best", action="store_true", help="Resume from best checkpoint if available")
    ap.add_argument("--resume", nargs="?", const="latest", default=None,
                    help="Resume from memory snapshot (optionally specify run_id)")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--safe", action="store_true")
    ap.add_argument("--use-indicators", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--eval-every-steps", type=int, default=0)
    ap.add_argument("--log-every-steps", type=int, default=10_000)
    ap.add_argument("--print-every-sec", type=int, default=10)
    ap.add_argument("--benchmark-every-steps", type=int, default=50_000)
    ap.add_argument("--artifact-every-steps", type=int, default=100_000,
                    help="Dump periodic artefacts every N steps (default 100k)")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--tb-logdir", type=str, default=os.path.join("logs", "tb"))
    ap.add_argument("--quiet-device-report", action="store_true")
    ap.add_argument("--log-level", type=str, default="INFO",
                    help="Root logging level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--omp-threads", type=int, default=1)
    ap.add_argument("--mkl-threads", type=int, default=1)
    ap.add_argument("--cuda-tf32", action="store_true")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--sb3-verbose", type=int, default=1, choices=[0,1,2])
    ap.add_argument("--vecnorm", action="store_true")
    ap.add_argument("--norm-obs", action="store_true")
    ap.add_argument("--norm-reward", action="store_true")
    ap.add_argument("--clip-obs", type=float, default=10.0)
    ap.add_argument("--clip-reward", type=float, default=10.0)
    ap.add_argument("--agents-dir", type=str, default=DEFAULT_AGENTS_DIR)
    ap.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--reports-dir", type=str, default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--memory-file", type=str, default=DEFAULT_MEMORY_FILE)
    ap.add_argument("--kb-file", type=str, default=DEFAULT_KB_FILE)
    ap.add_argument(
        "--data-dir",
        dest="data_root",
        type=str,
        default=None,
        help="Dataset root directory (overrides config)"
    )
    ap.add_argument("--data-root", dest="data_root", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--playlist", type=str, default=None)
    ap.add_argument("--mp-start", type=str, default="spawn", choices=["spawn", "forkserver", "fork"])
    ap.add_argument(
        "--monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch monitor manager during training",
    )
    ap.add_argument("--monitor-refresh", type=int, default=10)
    ap.add_argument("--monitor-images-out", type=str,
                    default="exports/{symbol}/{frame}/live")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run training without GUI; monitor exports charts to reports.",
    )
    ap.add_argument("--export-min-images", type=int, default=5)
    ap.add_argument("--debug-export", action="store_true")
    ap.add_argument("--allow-synth", action="store_true")
    ap.add_argument("--latency-ms", type=int, default=None)
    ap.add_argument("--max-spread-bp", type=float, default=None)
    ap.add_argument("--allow-partial-exec", action="store_true")
    ap.add_argument("--slippage-model", type=str, default=None)
    ap.add_argument("--slippage-params", type=str, default=None)
    ap.add_argument("--fees-bps", type=float, default=None)
    ap.add_argument("--partial-fills", choices=["on", "off"], default=None)
    ap.add_argument("--regime-aware", action="store_true", help="Enable regime-aware adjustments")
    ap.add_argument("--regime-window", type=int, default=0, help="Steps between regime checks")
    ap.add_argument("--regime-log", action=argparse.BooleanOptionalAction, default=None, help="Log adaptive regime adjustments")
    ap.add_argument("--reward-spec", type=str, default=None, help="Reward spec YAML path")
    ap.add_argument("--adaptive-spec", type=str, default=None, help="Adaptive controller YAML path")
    ap.add_argument("--strategy-failure", action=argparse.BooleanOptionalAction, default=None, help="Enable strategy failure policy")
    ap.add_argument("--safety-every", type=int, default=1, help="Check safety every N steps")
    ap.add_argument("--loss-streak", type=int)
    ap.add_argument("--max-drawdown-bp", type=float)
    ap.add_argument("--spread-jump-bp", type=float)
    ap.add_argument("--slippage-spike-bp", type=float)
    ap.add_argument(
        "--algorithm",
        type=str,
        choices=["PPO", "SAC", "TD3", "TQC"],
        default="PPO",
        help="Choose RL algorithm (TD3/TQC stubs; default PPO unless config overrides)",
    )
    ap.add_argument(
        "--continuous-env",
        action="store_true",
        help="Use Box(-1,1) action space; enables SAC/TD3/TQC (backward compatible).",
    )
    ap.add_argument("--buffer-size", type=int)
    ap.add_argument("--learning-starts", type=int)
    ap.add_argument("--train-freq", type=int)
    ap.add_argument("--gradient-steps", type=int)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--sac-gamma", type=float, help="Override gamma for SAC only")
    ap.add_argument("--warmstart-from-ppo", type=str,
                    help="(Optional) path to PPO .zip to warm-start SAC feature extractor")
    ap.add_argument(
        "--sac-warmstart-from-ppo",
        action="store_true",
        help="Warm-start SAC feature extractor from best PPO checkpoint if available",
    )
    ap.add_argument("--ai-core", action="store_true", help="Enable ai_core pipeline and bridge")
    ap.add_argument("--dry-run", action="store_true", help="Run ai_core without writing outputs")
    ap.add_argument(
        "--emit-dummy-signals",
        action="store_true",
        help="Generate synthetic demo signals",
    )
    ap.add_argument("--preset", action="append", default=[], help="Apply presets like training=name or net=name")
    defaults = vars(ap.parse_args([]))
    args = ap.parse_args()
    args._defaults = defaults
    specified = set()
    for item in sys.argv[1:]:
        if not item.startswith("--"):
            continue
        name = item[2:].split("=", 1)[0].replace("-", "_")
        specified.add(name)
    if getattr(args, "partial_fills", None) is not None:
        args.allow_partial_exec = args.partial_fills == "on"
        specified.add("allow_partial_exec")
    args._specified = specified
    if getattr(args, "adaptive_spec", None):
        args.regime_aware = True
        if args.regime_log is None:
            args.regime_log = True
    if args.regime_log is None:
        args.regime_log = bool(args.regime_aware)
    level_name = str(getattr(args, "log_level", "INFO")).upper()
    import logging
    args.log_level = getattr(logging, level_name, logging.INFO)
    args.use_sde = bool(getattr(args, "sde", False))
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.omp_threads)))
    os.environ["MKL_NUM_THREADS"] = str(max(1, int(args.mkl_threads)))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return args

def finalize_args(args, is_continuous: Optional[bool] = None):
    if is_continuous is False:
        args.sde = False
        args.use_sde = False
    try:
        import torch
        try:
            torch.set_num_threads(max(1, int(getattr(args, "torch_threads", 1))))
        except Exception:
            pass
        if getattr(args, "cuda_tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if getattr(args, "cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        total_per_rollout = int(args.n_envs) * int(args.n_steps)
        eff = min(int(getattr(args, "batch_size", 0) or 0), max(total_per_rollout, int(args.n_envs)))
        eff = (eff // int(args.n_envs)) * int(args.n_envs)
        args.batch_size_eff = eff if eff > 0 else int(args.n_envs)
    except Exception:
        pass
    return args
