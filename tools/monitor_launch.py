"""Utility to launch monitor scripts in new consoles."""
from __future__ import annotations

import os
import sys
import subprocess
from typing import List


def launch_new_console(title: str, script: str, args: List[str]) -> None:
    cmd = [sys.executable, script] + args
    try:
        if os.name == "nt":
            subprocess.Popen(["cmd", "/c", "start", title] + cmd,
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif sys.platform == "darwin":
            apple = [
                "osascript", "-e",
                'tell app "Terminal" to do script "%s"' % " ".join(cmd)
            ]
            subprocess.Popen(apple)
        else:
            term = os.environ.get("MONITOR_TERM")
            if term:
                subprocess.Popen([term, "--", "bash", "-lc", " ".join(cmd)])
            else:
                try:
                    subprocess.Popen(["gnome-terminal", "--", "bash", "-lc", " ".join(cmd)])
                except Exception:
                    subprocess.Popen(cmd)
    except Exception:
        subprocess.Popen(cmd)
