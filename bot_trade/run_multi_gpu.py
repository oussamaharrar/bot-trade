#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_multi_gpu.py — مُشغّل متوازي/تسلسلي يقرأ playlist.yaml ويشغّل train_rl.py على عدّة GPUs أو بالتتابع.

المزايا:
- تشغيل كل Job في عملية مستقلة (متزامن) مع تمرير --device الصحيح.
- خيار جديد: --sequential لتشغيل الـ Jobs واحدًا تلو الآخر (مفيد للفريمات الثقيلة).
- (افتراضي) لوج لكل Job في logs/launcher/<SYMBOL>/<FRAME>/run-YYYYmmdd_HHMMSS.log
- توزيع خيوط BLAS لكل عملية (OMP/MKL/NUMEXPR) تلقائيًا أو يدويًا.
- خيار فتح كل عملية في نافذة Console جديدة على Windows (--new-windows).
- خيار جديد: --no-redirect لعرض الإخراج في النوافذ نفسها (مع إبقائها مفتوحة على Windows عبر cmd /k).
- خيار جديد: --auto-devices لاختيار جميع الكروت المتاحة تلقائيًا.
- خيار جديد: --threads-auto لتوزيع الخيوط تلقائيًا حسب عدد الأنوية وعدد الـ Jobs.
- إيقاف جميل (Ctrl+C) يوقف كل العمليات التابعة.

الاستخدام:
    python run_multi_gpu.py --playlist playlist.yaml --threads-per-job 6 --stagger-sec 5 --post-analyze
    python run_multi_gpu.py --auto-devices --threads-per-job 8
    python run_multi_gpu.py --playlist playlist.yaml --new-windows --no-redirect
    python run_multi_gpu.py --auto-devices --threads-auto
    python run_multi_gpu.py --playlist playlist.yaml --sequential
"""
import os
import sys
import time
import json
import yaml
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

CREATE_NEW_CONSOLE = 0x00000010 if os.name == 'nt' else 0


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def build_env(base_env: Dict[str, str], threads_per_job: int) -> Dict[str, str]:
    env = dict(base_env)
    # اجبر بايثون على الإخراج غير المؤقت حتى تُطبع الأسطر مباشرة في الكونسول
    env["PYTHONUNBUFFERED"] = "1"
    if threads_per_job > 0:
        env["OMP_NUM_THREADS"] = str(threads_per_job)
        env["MKL_NUM_THREADS"] = str(threads_per_job)
        env["NUMEXPR_MAX_THREADS"] = str(threads_per_job)
    return env


def _truthy(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t", "on", "enable", "enabled", "latest", "best"}


def build_cmd(python_bin: str, job: Dict[str, Any], post_analyze: bool) -> List[str]:
    cmd: List[str] = [python_bin, "-u", "-m", "bot_trade.train_rl"]

    consumed: set = set()

    if "device" in job:
        cmd += ["--device", str(job["device"])]
        consumed.add("device")
    if "symbol" in job:
        cmd += ["--symbol", str(job["symbol"])]
        consumed.add("symbol")
    if "frame" in job:
        cmd += ["--frame", str(job["frame"])]
        consumed.add("frame")

    scalar_keys = {
        "n_envs", "n_steps", "batch_size", "epochs", "total_steps",
        "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "max_grad_norm", "net_arch", "activation", "seed",
        "checkpoint_every", "eval_episodes", "eval_every_steps", "log_every_steps",
        "print_every_sec", "benchmark_every_steps", "artifact_every_steps", "tb_logdir", "torch_threads",
        "omp_threads", "mkl_threads", "clip_obs", "clip_reward", "agents_dir",
        "results_dir", "reports_dir", "memory_file", "kb_file", "playlist", "mp_start",
        "policy", "net_arch", "activation", "log_level"
    }

    for k in scalar_keys:
        if k in job and job[k] is not None:
            cmd += [f"--{k.replace('_','-')}", str(job[k])]
            consumed.add(k)

    boolean_flags = {
        "sde", "ortho_init", "progress", "safe", "use_indicators",
        "tensorboard", "quiet_device_report", "cuda_tf32", "cudnn_benchmark",
        "vecnorm", "norm_obs", "norm_reward", "post_analyze"
    }

    for k in boolean_flags:
        val = job.get(k, None)
        if k == "post_analyze" and val is None:
            val = post_analyze
        if _truthy(val):
            cmd += [f"--{k.replace('_','-')}"]
            consumed.add(k)

    if _truthy(job.get("resume_auto", None)):
        cmd += ["--resume-auto"]
        consumed.add("resume_auto")

    for k, v in job.items():
        if k in consumed:
            continue
        if v is None or v is False:
            continue
        dashed = k.replace('_', '-')
        if isinstance(v, bool):
            if v:
                cmd += [f"--{dashed}"]
        else:
            cmd += [f"--{dashed}", str(v)]

    return cmd


def main():
    p = argparse.ArgumentParser("run_multi_gpu")
    p.add_argument("--playlist", type=str, default="playlist.yaml")
    p.add_argument("--threads-per-job", type=int, default=6, help="تحديد خيوط BLAS لكل عملية (OMP/MKL/NUMEXPR)")
    p.add_argument("--threads-auto", action="store_true", help="توزيع الخيوط تلقائيًا حسب عدد الأنوية وعدد الـ Jobs")
    p.add_argument("--stagger-sec", type=int, default=5, help="تأخير بين إطلاق كل عملية لتخفيف الضغط")
    p.add_argument("--new-windows", action="store_true", help="فتح كل Job في نافذة Console جديدة (Windows)")
    p.add_argument("--no-redirect", action="store_true", help="عدم إعادة التوجيه إلى ملفات لوج — اطبع مباشرة في الكونسول")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--post-analyze", action="store_true")
    p.add_argument("--auto-devices", action="store_true", help="استخدام جميع كروت GPU المتاحة تلقائيًا")
    p.add_argument("--sequential", action="store_true", help="تشغيل الـ Jobs واحدًا تلو الآخر بدلاً من التوازي")
    args = p.parse_args()

    playlist: List[Dict[str, Any]] = []

    if args.auto_devices:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError:
            print(
                "[ERROR] PyTorch is required for --auto-devices.\n"
                "Install via 'conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia'"
            )
            raise SystemExit(1)
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            print("[ERROR] No GPUs detected!")
            sys.exit(1)
        for d in range(n_gpus):
            playlist.append({"device": d, "symbol": "BTCUSDT", "frame": "1m"})
    else:
        if not os.path.isfile(args.playlist):
            print(f"[ERROR] Playlist not found: {args.playlist}")
            sys.exit(1)
        with open(args.playlist, "r", encoding="utf-8") as f:
            playlist = yaml.safe_load(f) or []
        if not isinstance(playlist, list) or not playlist:
            print("[ERROR] playlist.yaml is empty or invalid (expected list at root)")
            sys.exit(1)

    python_bin = sys.executable
    total_cores = os.cpu_count() or 8
    jobs_count = len(playlist)
    threads_auto = max(1, total_cores // jobs_count)
    if args.threads_auto:
        threads_per_job = threads_auto
    else:
        threads_per_job = args.threads_per_job if args.threads_per_job > 0 else threads_auto

    print(f"[ℹ] CPU cores={total_cores}, jobs={jobs_count}, threads_per_job={threads_per_job}")

    if args.sequential:
        print("[ℹ] Running in sequential mode (job by job).")
        for idx, job in enumerate(playlist):
            sym = str(job.get("symbol", "SYMBOL"))
            frame = str(job.get("frame", "FRAME"))
            dev = str(job.get("device", "-1"))
            name = f"{sym}-{frame}-gpu{dev}-{now_tag()}"

            env = build_env(os.environ, threads_per_job)
            try:
                d = int(job.get("device", -1))
                if d >= 0:
                    env["CUDA_VISIBLE_DEVICES"] = str(d)
            except Exception:
                pass

            cmd = build_cmd(python_bin, job, post_analyze=args.post_analyze)
            log_dir = ensure_dir(os.path.join("logs", "launcher", sym, frame))
            log_path = os.path.join(log_dir, f"run-{name}.log")

            with open(log_path, "w", buffering=1, encoding="utf-8") as log_file:
                print(f"[▶] Launching {name} CMD: {' '.join(cmd)} LOG: {log_path}")
                if args.dry_run:
                    log_file.write("DRY RUN")
                    continue
                popen = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
                code = popen.wait()
                print(f"[✓] {name} finished with exit_code={code}")
    else:
        # Parallel mode (original)
        procs: List[Tuple[subprocess.Popen, Optional[Any], str]] = []
        tag = now_tag()

        for idx, job in enumerate(playlist):
            sym = str(job.get("symbol", "SYMBOL"))
            frame = str(job.get("frame", "FRAME"))
            dev = str(job.get("device", "-1"))
            name = f"{sym}-{frame}-gpu{dev}-{tag}"

            env = build_env(os.environ, threads_per_job)
            try:
                d = int(job.get("device", -1))
                if d >= 0:
                    env["CUDA_VISIBLE_DEVICES"] = str(d)
            except Exception:
                pass

            cmd = build_cmd(python_bin, job, post_analyze=args.post_analyze)

            creationflags = CREATE_NEW_CONSOLE if (args.new_windows and os.name == 'nt') else 0
            log_file = None

            if args.no_redirect:
                if args.new_windows and os.name == 'nt':
                    cmd = ["cmd", "/k"] + cmd
                target = "NEW CONSOLE" if (args.new_windows and os.name == 'nt') else "CURRENT CONSOLE"
                print(f"[▶] Launching {name} CMD: {' '.join(cmd)} OUTPUT: {target}")
                popen = subprocess.Popen(cmd, env=env, creationflags=creationflags)
            else:
                log_dir = ensure_dir(os.path.join("logs", "launcher", sym, frame))
                log_path = os.path.join(log_dir, f"run-{name}.log")
                log_file = open(log_path, "w", buffering=1, encoding="utf-8")
                print(f"[▶] Launching {name} CMD: {' '.join(cmd)} LOG: {log_path}")
                if args.dry_run:
                    log_file.write("DRY RUN")
                    log_file.close()
                    continue
                popen = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, creationflags=creationflags)

            procs.append((popen, log_file, name))
            time.sleep(max(0, args.stagger_sec))

        print(f"[ℹ] Running {len(procs)} jobs. Press Ctrl+C to stop all.")

        try:
            exit_codes = []
            for popen, logf, name in procs:
                code = popen.wait()
                exit_codes.append((name, code))
                try:
                    if logf is not None:
                        logf.close()
                except Exception:
                    pass
            print("========== SUMMARY ==========")
            for name, code in exit_codes:
                print(f"{name}: exit_code={code}")
            print("============================")
        except KeyboardInterrupt:
            print("[!] KeyboardInterrupt — terminating all jobs...")
            for popen, logf, _ in procs:
                try:
                    popen.terminate()
                except Exception:
                    pass
                try:
                    if logf is not None:
                        logf.close()
                except Exception:
                    pass
            time.sleep(2)
        for popen, _, _ in procs:
            if popen.poll() is None:
                try:
                    popen.kill()
                except Exception:
                    pass
            print("[✓] All jobs terminated.")

    # optional post-training knowledge sync
    if args.post_analyze:
        try:
            subprocess.run([
                sys.executable,
                "tools/knowledge_sync.py",
                "--results-dir",
                "results",
                "--agents-dir",
                "agents",
                "--out",
                os.path.join("memory", "knowledge_base_full.json"),
            ], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
