"""Cross-platform utility to launch monitor scripts in new consoles."""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import platform
import shlex
from typing import List
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ps_quote(arg: str) -> str:
    return "'" + arg.replace("'", "''") + "'"


def _ps_command(parts: List[str]) -> str:
    return "& " + " ".join(_ps_quote(p) for p in parts)


def _cmd_command(parts: List[str]) -> str:
    out: List[str] = []
    for p in parts:
        if not p:
            out.append('""')
        elif any(ch in p for ch in [' ', '\t', '"']):
            out.append('"' + p.replace('"', r'\"') + '"')
        else:
            out.append(p)
    return " ".join(out)


def launch_new_console(title: str, module: str, args: List[str]) -> None:
    """Launch ``module`` with ``args`` in a new console window."""
    base_cmd = [sys.executable, "-m", module] + list(args)

    use_conda = os.environ.get("MONITOR_USE_CONDA_RUN") == "1"
    conda_exe = shutil.which("conda") if use_conda else None
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if use_conda and conda_exe and conda_env:
        base_cmd = [conda_exe, "run", "--no-capture-output", "-n", conda_env] + base_cmd

    debug = os.environ.get("MONITOR_DEBUG") == "1"
    system = platform.system()
    prefer_shell = os.environ.get("MONITOR_SHELL", "powershell").lower()

    if system == "Windows":
        wt = shutil.which("wt")
        if prefer_shell.startswith("power"):
            ps_cmd = _ps_command(base_cmd)
            ps_cmd = f"[console]::Title={_ps_quote(title)}; {ps_cmd}"
            if wt:
                final_cmd = [wt, "nt", "powershell", "-NoExit", "-Command", ps_cmd]
                label = "WT PowerShell"
            else:
                final_cmd = ["powershell", "-NoExit", "-Command", ps_cmd]
                label = "PowerShell"
        else:
            cmdline = _cmd_command(base_cmd)
            cmdline = f"title {title} & {cmdline}"
            if wt:
                final_cmd = [wt, "nt", "cmd", "/k", cmdline]
                label = "WT CMD"
            else:
                final_cmd = ["cmd", "/k", cmdline]
                label = "CMD"
        if debug:
            print(f"[MONITOR] {label}:", " ".join(final_cmd))
        subprocess.Popen(final_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=ROOT)
        return

    if system == "Darwin":
        cmd_str = " ".join(shlex.quote(p) for p in base_cmd)
        osa = [
            "osascript",
            "-e",
            f'tell application "Terminal" to do script {shlex.quote(cmd_str)}'
        ]
        if debug:
            print("[MONITOR]", osa)
        subprocess.Popen(osa, cwd=ROOT)
        return

    cmd_str = " ".join(shlex.quote(p) for p in base_cmd)
    terminals: List[str] = []
    term_env = os.environ.get("MONITOR_TERM")
    if term_env:
        terminals.append(term_env)
    terminals.extend(["gnome-terminal", "konsole", "xterm", "alacritty"])

    for term in terminals:
        if shutil.which(term):
            if term == "gnome-terminal":
                final_cmd = [term, "--", "bash", "-lc", cmd_str]
            elif term in ("konsole", "alacritty", "xterm"):
                final_cmd = [term, "-e", "bash", "-lc", cmd_str]
            else:
                continue
            if debug:
                print("[MONITOR]", final_cmd)
            subprocess.Popen(final_cmd, cwd=ROOT)
            return

    if debug:
        print("[MONITOR]", base_cmd)
    subprocess.Popen(base_cmd, cwd=ROOT)


def spawn_monitor_manager(root_dir: str, args, run_id: str, headless: bool = True):
    """Launch monitor_manager with minimal logic shared across callers."""
    cmd = [
        sys.executable,
        "-m",
        "bot_trade.tools.monitor_manager",
        "--symbol",
        args.symbol,
        "--frame",
        args.frame,
        "--run-id",
        run_id,
        "--base",
        root_dir,
    ]
    if headless:
        cmd.append("--no-wait")

    child_env = os.environ.copy()
    child_env.setdefault("BOT_TRADE_ROOT", root_dir)

    if not headless:
        try:
            proc = subprocess.Popen(cmd, env=child_env, shell=False)
            logging.getLogger(__name__).info("[monitor] spawned (interactive)")
            return proc
        except Exception:  # pragma: no cover
            logging.getLogger(__name__).warning("monitor launch failed")
            return None

    success = False
    if os.name == "nt":
        CREATE_NEW_CONSOLE = 0x00000010
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000

        def _try_spawn_win(flags, close_fds):
            return subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=close_fds,
                env=child_env,
                shell=False,
            )

        try:
            _try_spawn_win(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, True)
            success = True
        except Exception:
            try:
                _try_spawn_win(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, False)
                success = True
            except Exception:
                try:
                    pyw = sys.executable.replace("python.exe", "pythonw.exe")
                    if os.path.exists(pyw):
                        cmd2 = [pyw] + cmd[1:]
                        subprocess.Popen(
                            cmd2,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                            close_fds=True,
                            env=child_env,
                            shell=False,
                        )
                        success = True
                    else:
                        raise FileNotFoundError
                except Exception:
                    try:
                        _try_spawn_win(CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP, True)
                        success = True
                    except Exception:
                        success = False
    else:
        try:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_env,
                shell=False,
                start_new_session=True,
            )
            success = True
        except Exception:
            success = False
    if success:
        logging.getLogger(__name__).info("[monitor] spawned (headless)")
    return None
