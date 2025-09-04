import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import psutil


@dataclass
class RunInfo:
    """Metadata about a running command."""
    run_id: str
    cmd: List[str]
    process: subprocess.Popen
    log_file: str
    start_ts: float
    device: Optional[str] = None
    status: str = "running"
    run_dir: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


_runs: Dict[int, RunInfo] = {}
_runs_lock = threading.Lock()


def _reader(pipe, log_fh, run_id: str, queue: Optional[Any], source: str) -> None:
    with pipe:
        for line in iter(pipe.readline, ''):
            log_fh.write(line)
            log_fh.flush()
            if queue is not None:
                queue.put({"run_id": run_id, "source": source, "line": line.rstrip('\n')})


def start_command(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    run_id: Optional[str] = None,
    tee_to: Optional[str] = None,
    log_queue: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """Start a subprocess command and tee output.

    Args:
        cmd: Command list for subprocess.
        env: Environment variables to override.
        cwd: Working directory.
        run_id: Identifier for the run.
        tee_to: Path to log file.
        log_queue: Optional queue to push log lines.
        meta: Additional metadata to store.

    Returns:
        PID of started process.
    """
    env_full = os.environ.copy()
    if env:
        env_full.update(env)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env_full,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    pid = proc.pid
    run_id = run_id or str(pid)
    log_path = tee_to or os.path.join("results", run_id, "logs", "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8", buffering=1)

    threading.Thread(target=_reader, args=(proc.stdout, log_fh, run_id, log_queue, "stdout"), daemon=True).start()
    threading.Thread(target=_reader, args=(proc.stderr, log_fh, run_id, log_queue, "stderr"), daemon=True).start()

    def _waiter() -> None:
        proc.wait()
        log_fh.close()
        if log_queue is not None:
            log_queue.put({"run_id": run_id, "event": "exit", "returncode": proc.returncode})

    threading.Thread(target=_waiter, daemon=True).start()

    info = RunInfo(
        run_id=run_id,
        cmd=cmd,
        process=proc,
        log_file=log_path,
        start_ts=time.time(),
        meta=meta or {},
    )
    with _runs_lock:
        _runs[pid] = info
    return pid


def stop_process(pid: int, graceful_timeout: float = 5.0) -> Dict[str, Any]:
    """Terminate a process tree safely."""
    with _runs_lock:
        info = _runs.get(pid)
    if not info:
        return {"stopped": False, "reason": "unknown pid"}
    proc = info.process
    if proc.poll() is not None:
        return {"stopped": True, "reason": "already exited"}

    reason = "terminated"
    try:
        ps_proc = psutil.Process(pid)
        children = ps_proc.children(recursive=True)
        ps_proc.terminate()
        for child in children:
            child.terminate()
        gone, alive = psutil.wait_procs([ps_proc] + children, timeout=graceful_timeout)
        for p in alive:
            p.kill()
            reason = "killed"
    except psutil.NoSuchProcess:
        reason = "no such process"
    finally:
        with _runs_lock:
            _runs.pop(pid, None)
    return {"stopped": True, "reason": reason}


def get_run(pid: int) -> Optional[RunInfo]:
    with _runs_lock:
        return _runs.get(pid)
