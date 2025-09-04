from __future__ import annotations
"""Subprocess runner with tee logging and safe process-tree termination."""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProcessHandle:
    id: str
    process: subprocess.Popen
    pid: int
    pgroup: Optional[int]
    start_ts: float
    cmd: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)
    tee_path: Optional[Path] = None


_HANDLES: Dict[str, ProcessHandle] = {}


def _tee_output(proc: subprocess.Popen, tee_path: Path) -> None:
    with tee_path.open("a", encoding="utf-8") as fh:
        for line in proc.stdout:  # type: ignore[attr-defined]
            text = line.decode("utf-8", "replace")
            fh.write(text)
            fh.flush()


def start_command(
    cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
    tee_path: Optional[Path] = None, metadata: Dict[str, str] | None = None
) -> ProcessHandle:
    meta = metadata or {}
    if tee_path:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        preexec_fn = os.setsid  # type: ignore[assignment]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        preexec_fn=preexec_fn,
        creationflags=creationflags,
    )
    pgroup = os.getpgid(proc.pid) if os.name != "nt" else None
    job_id = f"job-{int(time.time()*1000)}-{proc.pid}"
    handle = ProcessHandle(job_id, proc, proc.pid, pgroup, time.time(), list(cmd), meta, tee_path)
    _HANDLES[job_id] = handle
    if tee_path:
        t = threading.Thread(target=_tee_output, args=(proc, tee_path), daemon=True)
        t.start()
    return handle


def stop_process_tree(handle: ProcessHandle, grace_sec: int = 10) -> int:
    proc = handle.process
    if proc.poll() is not None:
        return proc.returncode or 0
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(handle.pgroup or proc.pid, signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(handle.pgroup or proc.pid, signal.SIGKILL)
        except Exception:
            pass
        proc.wait(timeout=grace_sec)
    return proc.returncode or 0


def stop_job(job_id: str) -> int:
    handle = _HANDLES.get(job_id)
    if not handle:
        raise KeyError(job_id)
    return stop_process_tree(handle)


def list_jobs() -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for jid, h in list(_HANDLES.items()):
        status = "running" if h.process.poll() is None else "stopped"
        jobs.append({
            "id": jid,
            "pid": h.pid,
            "cmd": h.cmd,
            "started_at": h.start_ts,
            "status": status,
        })
        if status == "stopped":
            _HANDLES.pop(jid, None)
    return jobs


def tail(job_id: str, n: int = 200) -> str:
    handle = _HANDLES.get(job_id)
    if not handle or not handle.tee_path or not handle.tee_path.exists():
        return ""
    lines = handle.tee_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-n:])


def list_processes() -> List[ProcessHandle]:  # pragma: no cover - legacy
    return [h for h in _HANDLES.values() if h.process.poll() is None]


__all__ = [
    "ProcessHandle",
    "start_command",
    "stop_process_tree",
    "stop_job",
    "list_jobs",
    "tail",
    "list_processes",
]
