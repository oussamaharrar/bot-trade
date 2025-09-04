from __future__ import annotations
"""Simplified PySimpleGUI control panel (v3 skeleton)."""

from pathlib import Path
from typing import Dict, List

try:
    import PySimpleGUI as sg  # type: ignore
except Exception:  # pragma: no cover
    sg = None

from ._headless import ensure_headless_once
from . import whitelist, results_watcher, device_select


class PanelModel:
    """Light-weight model used for tests and the UI."""

    def __init__(self) -> None:
        self.last_run_id: str | None = None
        self.last_postrun: Dict[str, object] = {}

    def build_train_command(self, params: Dict[str, object], extra: str = "") -> List[str]:
        flags = [f for f in extra.replace(",", " ").split() if f]
        return whitelist.build_command("train", params, flags)

    def handle_log_line(self, line: str) -> None:
        ev = results_watcher.parse_log_line(line.strip())
        if not ev:
            return
        if ev.get("event") == "postrun":
            self.last_run_id = str(ev.get("run_id"))
            self.last_postrun = ev


class Panel:
    def __init__(self) -> None:
        ensure_headless_once("ui_panel")
        self.model = PanelModel()
        sg.theme("SystemDefault")
        devices = [d["label"] for d in device_select.get_devices()]
        layout = [
            [sg.Text("Symbol"), sg.Input(key="symbol", size=10)],
            [sg.Text("Frame"), sg.Input(key="frame", size=6)],
            [sg.Text("Steps"), sg.Input(key="total_steps", size=6)],
            [sg.Text("n-envs"), sg.Input(key="n_envs", size=6)],
            [sg.Text("Device"), sg.Combo(devices, default_value=devices[0] if devices else "cpu", key="device")],
            [sg.Button("Start"), sg.Button("Exit")],
        ]
        self.window = sg.Window("bot_trade Panel v3", layout, finalize=True, resizable=True)

    def run(self) -> None:  # pragma: no cover - interactive
        while True:
            event, values = self.window.read(timeout=200)
            if event in (sg.WIN_CLOSED, "Exit"):
                break
            if event == "Start":
                params = {
                    "symbol": values.get("symbol", "BTCUSDT"),
                    "frame": values.get("frame", "1m"),
                    "device": values.get("device", "cpu"),
                    "n_envs": values.get("n_envs", "1"),
                    "n_steps": "1",
                    "batch_size": "64",
                    "total_steps": values.get("total_steps", "1024"),
                    "vecnorm_flag": "",
                    "headless_flag": "--headless",
                    "allow_synth_flag": "",
                    "resume_flag": "",
                    "extra": "",
                }
                try:
                    cmd = self.model.build_train_command(params)
                except whitelist.ValidationError as exc:
                    sg.popup(str(exc))
                    continue
                sg.popup("Command", " ".join(cmd))
        self.window.close()


def main() -> None:  # pragma: no cover
    Panel().run()


if __name__ == "__main__":  # pragma: no cover
    main()
