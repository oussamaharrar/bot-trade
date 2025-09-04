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
    process: subprocess.Popen
    pid: int
    pgroup: Optional[int]
    start_ts: float
    metadata: Dict[str, str] = field(default_factory=dict)
    tee_path: Optional[Path] = None


_HANDLES: List[ProcessHandle] = []


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
    """Launch *cmd* and return handle."""
    meta = metadata or {}
    if tee_path:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:  # posix
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
    handle = ProcessHandle(proc, proc.pid, pgroup, time.time(), meta, tee_path)
    _HANDLES.append(handle)
    if tee_path:
        t = threading.Thread(target=_tee_output, args=(proc, tee_path), daemon=True)
        t.start()
    return handle


def stop_process_tree(handle: ProcessHandle, grace_sec: int = 10) -> int:
    """Terminate process tree for *handle* and return returncode."""
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


def list_processes() -> List[ProcessHandle]:
    """Return running process handles."""
    alive: List[ProcessHandle] = []
    for h in list(_HANDLES):
        if h.process.poll() is None:
            alive.append(h)
        else:
            _HANDLES.remove(h)
    return alive
