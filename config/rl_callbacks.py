# config/rl_callbacks.py
"""
Callbacks الخاصة بالتدريب (SB3) — خفيفة، قوية، وقابلة للتشغيل الإختياري.

• StepsAndRewardCallback  : يسجّل متوسطات المكافأة ويستخرج الصفقات من info ويضخّ إلى TensorBoard إن وُجد.
• BestCheckpointCallback  : يحفظ أفضل نموذج + VecNormalize بطريقة متوافقة مع جميع إصدارات SB3.
• KnowledgeAndMemoryCallback : يحدّث KB/Memory بشكل دوري.
• BenchmarkCallback (جديد) : قياسات خفيفة للأداء (FPS/CPU/GPU/RAM) تُكتب لملف benchmark.
• StrictDataSanityCallback (جديد) : فحوص صارمة للـ NaN/Inf عند تفعيل وضع —safe.

ملاحظات تكامل:
- يعتمد وجود كُتّاب writers ذوي الواجهات: .reward .trades .benchmark (اختياري) و .tb (اختياري SummaryWriter).
- يستدعي benchmark_tick من log_setup عند توافره (اختياري) لتوليد نبضات قياس موحّدة.
"""
from __future__ import annotations

import os
import json
import math
import time
import logging
import datetime as dt
from typing import Optional, Dict, Any

from stable_baselines3.common.callbacks import BaseCallback

# Optional dependencies — كلّها اختيارية
try:  # TensorBoard
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

try:  # psutil (CPU/RAM)
    import psutil  # type: ignore
except Exception:
    psutil = None

try:  # NVML (GPU)
    import pynvml  # type: ignore
except Exception:
    pynvml = None

# benchmark tick الإختياري من log_setup
try:
    from .log_setup import benchmark_tick  # type: ignore
except Exception:
    benchmark_tick = None  # type: ignore


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _safe_mean_ep_reward(model) -> float:
    try:
        buf = getattr(model, "ep_info_buffer", None)
        if buf and len(buf) > 0:
            import numpy as _np  # lazy
            return float(_np.mean([x.get("r", 0.0) for x in buf]))
    except Exception:
        pass
    return float("nan")


def _save_vecnormalize(training_env, model, out_path: str) -> bool:
    """حفظ VecNormalize بمرونة عالية عبر نسخ SB3 المختلفة."""
    try:
        if hasattr(training_env, "save_running_average"):
            training_env.save_running_average(out_path)
            return True
    except Exception as e:  # pragma: no cover
        logging.debug("[VECNORM] env.save_running_average failed: %s", e)

    try:
        getter = getattr(model, "get_vec_normalize_env", None)
        if getter is not None:
            venv = getter()
            if venv is not None and hasattr(venv, "save"):
                venv.save(out_path)  # type: ignore[attr-defined]
                return True
    except Exception as e:  # pragma: no cover
        logging.debug("[VECNORM] model.get_vec_normalize_env().save failed: %s", e)
    return False


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

class StepsAndRewardCallback(BaseCallback):
    """تجميع مكافآت/صفقات خفيف + دعم TensorBoard الاختياري.

    Parameters
    ----------
    frame, symbol : str
        لتضمينهما في السجلات.
    writers : object
        يتوقع .reward.write(list) و .trades.write(list)؛ اختيارياً writers.tb (SummaryWriter).
    log_every : int
        كل كم خطوة نسجّل سطرًا مجمّعًا حتى لو لم تصلنا infos.
    """

    def __init__(self, frame: str, symbol: str, writers, log_every: int = 2_000, verbose: int = 0):
        super().__init__(verbose)
        self.frame, self.symbol = frame, symbol
        self.writers = writers
        self.log_every = int(max(1, log_every))
        self._last_log = 0

    def _on_step(self) -> bool:  # noqa: D401
        infos   = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)
        step    = int(self.num_timesteps)
        now = dt.datetime.utcnow().isoformat()

        if infos is not None:
            for i, info in enumerate(infos):
                # مكافآت موجزة + TB الاختياري
                ep_mean = _safe_mean_ep_reward(self.model)
                avg_reward = float("nan")
                try:
                    if rewards is not None and i < len(rewards):
                        avg_reward = float(rewards[i])
                except Exception:
                    pass
                try:
                    self.writers.reward.write([now, self.frame, self.symbol, step, avg_reward, ep_mean])
                except Exception:
                    pass
                try:
                    tb = getattr(self.writers, "tb", None)
                    if tb is not None and math.isfinite(ep_mean):
                        tb.add_scalar(f"{self.symbol}/{self.frame}/ep_rew_mean", float(ep_mean), step)
                except Exception:
                    pass

                # صفقات (إن وُجدت)
                trade = info.get("trade") if isinstance(info, dict) else None
                if trade:
                    try:
                        self.writers.trades.write([
                            now, self.frame, self.symbol, step,
                            trade.get("side"), trade.get("price"), trade.get("size"),
                            trade.get("pnl"), trade.get("equity"), trade.get("reason", "n/a")
                        ])
                    except Exception:
                        pass

                # سجل قرار منسّق (اختياري)
                if hasattr(self.writers, "log_decision"):
                    try:
                        self.writers.log_decision(
                            level="info", event="trade" if trade else "step",
                            env_idx=i, timesteps=step,
                            decision_reason=info.get("decision_reason", "ai_guard") if isinstance(info, dict) else None,
                            symbol=info.get("symbol", self.symbol) if isinstance(info, dict) else self.symbol,
                            frame=info.get("frame", self.frame) if isinstance(info, dict) else self.frame,
                            active_signals=info.get("signals_active", []) if isinstance(info, dict) else [],
                        )
                    except Exception:
                        pass

        # لقطات دورية حتى بدون infos
        if step - self._last_log >= self.log_every:
            ep_mean = _safe_mean_ep_reward(self.model)
            try:
                self.writers.reward.write([now, self.frame, self.symbol, step, ep_mean, ep_mean])
            except Exception:
                pass
            self._last_log = step
        return True


class BestCheckpointCallback(BaseCallback):
    """حفظ أفضل نموذج وفق ep_rew_mean + حفظ VecNormalize بأمان."""

    def __init__(self, paths: Dict[str, str], check_every: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.paths = paths
        self.check_every = int(max(1, check_every))
        self.best = float("-inf")

    def _on_step(self) -> bool:  # noqa: D401
        step = int(self.num_timesteps)
        if step % self.check_every != 0:
            return True

        avg = _safe_mean_ep_reward(self.model)
        if not math.isfinite(avg):
            return True
        if avg <= self.best:
            return True

        # save best
        try:
            self.best = float(avg)
            self.model.save(self.paths["model_best_zip"])  # model.zip
            _save_vecnormalize(self.training_env, self.model, self.paths.get("vecnorm_best", "vecnorm_best.pkl"))
            with open(self.paths["best_meta"], "w", encoding="utf-8") as f:
                json.dump({"step": step, "ep_rew_mean": float(avg)}, f, indent=2, ensure_ascii=False)
            logging.info("[BEST] step=%d | ep_rew_mean=%.6f | saved", step, avg)
        except Exception as e:
            logging.error("[BEST] save failed: %s", e)
        return True


class KnowledgeAndMemoryCallback(BaseCallback):
    """تحديث KB/Memory كل every خطوة بدون أي تباطؤ ملحوظ."""

    def __init__(self, kb_file: str, memory_file: str, frame: str, symbol: str, every: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.kb_file = kb_file
        self.memory_file = memory_file
        self.frame, self.symbol = frame, symbol
        self.every = int(max(1, every))

    def _on_step(self) -> bool:  # noqa: D401
        step = int(self.num_timesteps)
        if step % self.every != 0:
            return True
        now = dt.datetime.utcnow().isoformat()

        # memory.json — جلسات محدثة
        try:
            mem: Dict[str, Any] = {}
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    mem = json.load(f) or {}
            sess = mem.setdefault("sessions", {})
            key = f"{self.symbol}:{self.frame}"
            sess[key] = {"last_step": step, "updated_at": now}
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(mem, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error("[MEMORY] update failed: %s", e)
        return True


class BenchmarkCallback(BaseCallback):
    """Benchmark صغير وغير مُزعِج: يكتب FPS والموارد كل every_sec إلى writers.benchmark إن وُجد.

    • GPU: يعتمد pynvml إن كان متاحًا؛ وإلا يتجاهل.
    • CPU/RAM: يعتمد psutil إن وُجد.
    • TensorBoard: يضخ scalars إلى writers.tb إن كان متاحًا.
    """

    def __init__(self, frame: str, symbol: str, writers, every_sec: float = 15.0, verbose: int = 0):
        super().__init__(verbose)
        self.frame, self.symbol = frame, symbol
        self.writers = writers
        self.every_sec = float(max(1.0, every_sec))
        self._last_t = None
        self._last_steps = 0
        self._nvml_handles = []

    def _init_callback(self) -> None:
        # NVML init
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            except Exception:
                self._nvml_handles = []

    def _on_training_end(self) -> None:
        # NVML shutdown
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _on_step(self) -> bool:  # noqa: D401
        step = int(self.num_timesteps)
        now_t = time.time()
        if self._last_t is None:
            self._last_t = now_t
            self._last_steps = step
            return True
        if now_t - self._last_t < self.every_sec:
            return True

        dt_sec = max(1e-6, now_t - self._last_t)
        dsteps = max(0, step - self._last_steps)
        fps = dsteps / dt_sec
        now = dt.datetime.utcnow().isoformat()

        # CPU/RAM
        cpu = psutil.cpu_percent(interval=None) if psutil else float("nan")
        ram_gb = float("nan")
        if psutil:
            try:
                vm = psutil.virtual_memory()
                ram_gb = round((vm.total - vm.available) / (1024**3), 3)
            except Exception:
                pass

        # GPU memory (اختياري)
        gpu_mems = []
        if self._nvml_handles:
            try:
                for h in self._nvml_handles:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    gpu_mems.append(round(mem.used / (1024**2), 1))
            except Exception:
                gpu_mems = []

        # كتابات CSV
        try:
            row = [now, self.frame, self.symbol, step, fps, cpu, ram_gb]
            # حتى 4 كروت GPU
            for idx in range(4):
                row.append(gpu_mems[idx] if idx < len(gpu_mems) else "")
            self.writers.benchmark.write(row)
        except Exception:
            pass

        # benchmark_tick (اختياري)
        try:
            if benchmark_tick is not None:
                benchmark_tick(self.writers.paths.get("results", ""))
        except Exception:
            pass

        # TensorBoard
        try:
            tb = getattr(self.writers, "tb", None)
            if tb is not None:
                tb.add_scalar(f"{self.symbol}/{self.frame}/fps", float(fps), step)
                if math.isfinite(cpu):
                    tb.add_scalar(f"{self.symbol}/{self.frame}/cpu_percent", float(cpu), step)
                if math.isfinite(ram_gb):
                    tb.add_scalar(f"{self.symbol}/{self.frame}/ram_gb", float(ram_gb), step)
                for idx, val in enumerate(gpu_mems):
                    tb.add_scalar(f"{self.symbol}/{self.frame}/gpu{idx}_mem_mb", float(val), step)
        except Exception:
            pass

        # تحديث مؤقتات
        self._last_t = now_t
        self._last_steps = step
        return True


class StrictDataSanityCallback(BaseCallback):
    """تحقق صارم للداتا أثناء التدريب، مخصص لتفعيل خيار --safe.

    • يفحص NaN/Inf في rewards و new_obs (إن وُجدت في locals).
    • يكتب تحذيرًا في writers.error (إن وُجد)، ويمكن رفع استثناء عند أول اكتشاف.
    """

    def __init__(self, writers=None, raise_on_issue: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.writers = writers
        self.raise_on_issue = bool(raise_on_issue)

    def _on_step(self) -> bool:  # noqa: D401
        import numpy as np  # lazy import
        step = int(self.num_timesteps)
        now = dt.datetime.utcnow().isoformat()

        issues = []
        # rewards sanity
        try:
            rewards = self.locals.get("rewards", None)
            if rewards is not None:
                arr = np.asarray(rewards, dtype=float)
                if not np.isfinite(arr).all():
                    issues.append("rewards contain NaN/Inf")
        except Exception:
            pass

        # new_obs sanity
        try:
            new_obs = self.locals.get("new_obs", None)
            if new_obs is not None:
                arr = np.asarray(new_obs, dtype=float)
                if not np.isfinite(arr).all():
                    issues.append("new_obs contain NaN/Inf")
        except Exception:
            pass

        if issues:
            msg = f"[SAFE] Data sanity failed @step={step}: {', '.join(issues)}"
            logging.error(msg)
            # write to writers.error if available
            try:
                if getattr(self.writers, "error", None) is not None:
                    self.writers.error.write([now, step, "; ".join(issues)])
            except Exception:
                pass
            if self.raise_on_issue:
                raise RuntimeError(msg)
        return True


class CompositeCallback(BaseCallback):
    """Lightweight bridge that forwards SB3 events to UpdateManager."""

    def __init__(self, update_manager, cfg: Optional[Dict[str, Any]] = None, verbose: int = 0):
        super().__init__(verbose)
        self.um = update_manager
        self.cfg = cfg or {}
        logging_cfg = self.cfg.get("logging", {})
        self.step_every = int(logging_cfg.get("step_every", 100))

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        metrics = {}
        infos = self.locals.get("infos")
        try:
            first = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
            comps = first.get("reward_components", {})
            if isinstance(comps, dict):
                metrics.update(comps)
        except Exception:
            pass
        if step % self.step_every == 0 or metrics:
            try:
                self.um.log_step(step, metrics)
            except Exception:
                pass
        return True

    def _on_rollout_end(self) -> None:
        try:
            self.um.on_rollout_end()
        except Exception:
            pass

    def _on_training_end(self) -> None:
        try:
            self.um.on_training_end()
        except Exception:
            pass
