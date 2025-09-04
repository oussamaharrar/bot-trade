from __future__ import annotations

"""Simple PySimpleGUI control panel for bot_trade."""

import datetime as dt
import json
import queue
import threading
from pathlib import Path
from typing import Dict

import PySimpleGUI as sg

from . import runner, results_watcher, whitelist

CONFIG_PATH = Path(__file__).with_name("panel_config.yaml")
DEV_NOTES_PATH = Path(__file__).resolve().parents[2] / "docs" / "dev_notes.md"
LOG_DIR = Path("panel_logs")


def load_config() -> Dict:
    import yaml

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def save_config(cfg: Dict) -> None:
    import yaml
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)


def latest_dev_note() -> str:
    if not DEV_NOTES_PATH.exists():
        return ""
    lines = DEV_NOTES_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-10:])


class Panel:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.queue: queue.Queue = queue.Queue()
        self.handles: Dict[str, runner.ProcessHandle] = {}
        self.watchers: Dict[str, results_watcher.ResultsWatcher] = {}
        sg.theme("SystemDefault")
        self.window = sg.Window(
            "bot_trade Panel",
            self._layout(),
            finalize=True,
            resizable=True,
        )
        self.window["notes"].update(latest_dev_note())

    # ------------------------------------------------------------------
    def _layout(self):
        cfg = self.cfg.get("defaults", {}).get("train_rl", {})
        config_tab = [
            [sg.Text("Symbol"), sg.Input(cfg.get("symbol", "BTCUSDT"), key="symbol")],
            [sg.Text("Frame"), sg.Input(cfg.get("frame", "1m"), key="frame")],
            [sg.Text("Steps"), sg.Input(str(cfg.get("total_steps", 1000)), key="steps")],
            [sg.Text("n-envs"), sg.Input(str(cfg.get("n_envs", 1)), key="n_envs")],
            [sg.Text("Device"), sg.Input(cfg.get("device", "cpu"), key="device")],
            [sg.Checkbox("Headless", key="headless", default=cfg.get("headless", True))],
            [sg.Checkbox("Allow synth", key="allow_synth", default=cfg.get("allow_synth", False))],
            [sg.Button("Start"), sg.Button("Stop"), sg.Button("Exit")],
            [sg.Multiline("", size=(60, 10), key="log")],
            [sg.Text("Developer Notes")],
            [sg.Multiline("", size=(60, 10), key="notes", disabled=True)],
        ]
        runs_tab = [[sg.Table(values=[], headings=["Run", "Status"], key="runs", auto_size_columns=True, expand_x=True, expand_y=True)]]
        tools_tab = [[sg.Text("Tools placeholder")]]
        live_tab = [[sg.Text("Live monitor placeholder")]]
        layout = [
            [
                sg.TabGroup(
                    [
                        [
                            sg.Tab("Config", config_tab),
                            sg.Tab("Runs", runs_tab),
                            sg.Tab("Tools", tools_tab),
                            sg.Tab("Live", live_tab),
                        ]
                    ],
                    expand_x=True,
                    expand_y=True,
                )
            ]
        ]
        return layout

    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            event, values = self.window.read(timeout=200)
            if event in (sg.WIN_CLOSED, "Exit"):
                self._shutdown()
                break
            if event == "Start":
                self._start_run(values)
            if event == "Stop":
                self._stop_selected()
            self._drain_queue()

    def _start_run(self, values):
        params = {
            "symbol": values["symbol"],
            "frame": values["frame"],
            "total_steps": int(values["steps"]),
            "n_envs": int(values["n_envs"]),
            "device": values["device"],
            "headless": values["headless"],
            "allow_synth": values["allow_synth"],
        }
        try:
            cmd = whitelist.build_command("train_rl", params)
        except whitelist.ValidationError as exc:
            sg.popup(str(exc))
            return
        LOG_DIR.mkdir(exist_ok=True)
        run_id = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        tee = LOG_DIR / f"run_{run_id}.log"
        handle = runner.start_command(cmd, tee_path=tee, metadata=params)
        self.handles[run_id] = handle
        watcher = results_watcher.ResultsWatcher(Path("results")/params["symbol"]/params["frame"]/"dummy", tee, self.queue)
        watcher.start()
        self.watchers[run_id] = watcher
        self._refresh_runs()

    def _stop_selected(self):
        table = self.window["runs"]
        sel = table.SelectedRows
        if not sel:
            return
        run_id = list(self.handles.keys())[sel[0]]
        handle = self.handles.get(run_id)
        if handle:
            runner.stop_process_tree(handle)
        watcher = self.watchers.get(run_id)
        if watcher:
            watcher.stop()
        self._refresh_runs()

    def _drain_queue(self):
        log_elem = self.window["log"]
        try:
            while True:
                item = self.queue.get_nowait()
                if isinstance(item, dict):
                    log_elem.update(str(item) + "\n", append=True)
        except queue.Empty:
            pass

    def _refresh_runs(self):
        data = []
        for rid, handle in self.handles.items():
            status = "running" if handle.process.poll() is None else "stopped"
            data.append([rid, status])
        self.window["runs"].update(values=data)

    def _shutdown(self):
        cfg = {
            "defaults": {
                "train_rl": {
                    "symbol": self.window["symbol"].get(),
                    "frame": self.window["frame"].get(),
                    "total_steps": int(self.window["steps"].get()),
                    "n_envs": int(self.window["n_envs"].get()),
                    "device": self.window["device"].get(),
                    "headless": self.window["headless"].get(),
                    "allow_synth": self.window["allow_synth"].get(),
                }
            }
        }
        save_config(cfg)
        for watcher in self.watchers.values():
            watcher.stop()
        for handle in self.handles.values():
            runner.stop_process_tree(handle)


if __name__ == "__main__":  # pragma: no cover
    Panel().run()
