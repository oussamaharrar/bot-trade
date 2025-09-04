
from __future__ import annotations
"""PySimpleGUI control panel (simplified).

This module provides a small GUI for orchestrating bot_trade commands. The
``PanelModel`` class is used in tests to validate command building and log line
parsing without launching the full GUI.
"""

from pathlib import Path
from typing import Dict, List

import PySimpleGUI as sg

from ._headless import ensure_headless_once
from . import whitelist, results_watcher, device_select

CONFIG_PATH = Path("config/panel.yaml")


class PanelModel:
    """Light-weight model shared by the GUI and tests."""

    def __init__(self) -> None:
        self.last_run_id: str | None = None
        self.last_postrun: Dict[str, object] = {}

    def build_train_command(self, params: Dict[str, object], extra: str = "") -> List[str]:
        flags = [f for f in extra.replace(',', ' ').split() if f]
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
        ensure_headless_once("panel_gui")
        self.model = PanelModel()
        sg.theme("SystemDefault")
        devices = [d["label"] for d in device_select.get_devices()]
        train_layout = [
            [sg.Text("Symbol"), sg.Input(key="symbol", size=10)],
            [sg.Text("Frame"), sg.Input(key="frame", size=6)],
            [sg.Text("Steps"), sg.Input(key="total_steps", size=6)],
            [sg.Text("n-envs"), sg.Input(key="n_envs", size=6)],
            [sg.Text("Device"), sg.Combo(devices, default_value=devices[0], key="device")],
            [sg.Text("Data dir"), sg.Input(key="data_dir"), sg.FolderBrowse()],
            [sg.Checkbox("allow_synth", key="allow_synth"),
             sg.Checkbox("resume_auto", key="resume_auto"),
             sg.Checkbox("vecnorm", key="vecnorm")],
            [sg.Text("Extra flags"), sg.Input(key="extra_flags")],
            [sg.Button("Start Job"), sg.Button("Exit")],
        ]
        layout = [[sg.TabGroup([[sg.Tab("Train", train_layout)]], expand_x=True, expand_y=True)]]
        self.window = sg.Window("bot_trade Panel", layout, finalize=True, resizable=True)

    def run(self) -> None:  # pragma: no cover - interactive
        while True:
            event, values = self.window.read(timeout=200)
            if event in (sg.WIN_CLOSED, "Exit"):
                break
            if event == "Start Job":
                params = {
                    "symbol": values.get("symbol", "BTCUSDT"),
                    "frame": values.get("frame", "1m"),
                    "total_steps": values.get("total_steps", "1024"),
                    "n_envs": values.get("n_envs", "1"),
                    "device": values.get("device", "cpu"),
                    "data_dir": values.get("data_dir", ""),
                    "allow_synth": values.get("allow_synth", False) and "--allow-synth" or "",
                    "resume_auto": values.get("resume_auto", False) and "--resume-auto" or "",
                    "vecnorm": values.get("vecnorm", False) and "--vecnorm" or "",
                }
                extra = values.get("extra_flags", "")
                try:
                    cmd = self.model.build_train_command(params, extra)
                except whitelist.ValidationError as exc:
                    sg.popup(str(exc))
                    continue
                sg.popup("Command", " ".join(cmd))
        self.window.close()


def main() -> None:  # pragma: no cover
    Panel().run()


if __name__ == "__main__":  # pragma: no cover
    main()
