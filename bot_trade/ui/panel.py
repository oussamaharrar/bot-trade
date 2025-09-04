from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List

import PySimpleGUI as sg
import yaml

from . import runner
from .results_watcher import ResultsWatcher
from .whitelist import build_command, load_whitelist

CONFIG_PATH = Path(__file__).with_name("panel_config.yaml")
WHITELIST_PATH = Path(__file__).with_name("whitelist.yaml")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> Dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def save_config(cfg: Dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    os.replace(tmp, CONFIG_PATH)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def detect_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch optional
        return False


class Panel:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.log_queue: "queue.Queue[dict]" = queue.Queue()
        self.log_buffers: Dict[str, List[str]] = {}
        self.active_log: str | None = None
        self.whitelist = load_whitelist(WHITELIST_PATH)

        gpu_available = detect_gpu()
        device_opts = ["auto", "cpu"] + (["cuda"] if gpu_available else [])

        # Run Train tab
        train_layout = [
            [
                sg.Text("Symbol"),
                sg.Input(self.cfg.get("symbol", "BTCUSDT"), key="symbol", size=(10, 1)),
                sg.Text("Frame"),
                sg.Input(self.cfg.get("frame", "1h"), key="frame", size=(5, 1)),
                sg.Text("Preset"),
                sg.Input(self.cfg.get("preset", "default"), key="preset", size=(10, 1)),
                sg.Text("Algo"),
                sg.Input(self.cfg.get("algorithm", "PPO"), key="algorithm", size=(5, 1)),
            ],
            [
                sg.Text("Max Steps"),
                sg.Input(str(self.cfg.get("max_steps", 1000)), key="max_steps", size=(8, 1)),
                sg.Text("Seed"),
                sg.Input(str(self.cfg.get("seed", 42)), key="seed", size=(5, 1)),
                sg.Text("Data Dir"),
                sg.Input(self.cfg.get("data_dir", "data"), key="data_dir", size=(10, 1)),
                sg.Text("Device"),
                sg.Combo(device_opts, default_value=self.cfg.get("device", "auto"), key="device", size=(6, 1)),
                sg.Text("n-envs"),
                sg.Input(str(self.cfg.get("n_envs", 1)), key="n_envs", size=(4, 1)),
            ],
            [
                sg.Button("Gen Synth Data", key="train_gen"),
                sg.Button("Train", key="train_start"),
                sg.Button("Stop Selected", key="train_stop"),
            ],
            [sg.Text("Command:"), sg.Input("", key="cmd_preview", size=(80, 1), readonly=True)],
            [
                sg.Table(
                    values=[],
                    headings=["RunID", "Task", "PID", "Device", "Elapsed", "Status"],
                    key="runs_table",
                    enable_events=True,
                    auto_size_columns=False,
                    col_widths=[12, 8, 6, 6, 8, 8],
                    num_rows=5,
                )
            ],
        ]

        # Eval & WFA tab
        eval_layout = [
            [sg.Text("Run Dir"), sg.Input(key="eval_run_dir", size=(40, 1)), sg.Button("Eval + HTML", key="eval_run"), sg.Button("WFA", key="wfa_run")],
            [sg.Text("Windows"), sg.Input("3", key="wfa_windows", size=(4, 1)), sg.Text("Embargo"), sg.Input("0.1", key="wfa_embargo", size=(4, 1)), sg.Text("Profile"), sg.Input("smoke", key="wfa_profile", size=(6, 1))],
        ]

        # Live tab
        live_layout = [
            [sg.Text("Exchange"), sg.Input("binance", key="live_exchange", size=(8, 1)), sg.Text("Gateway"), sg.Input("paper", key="live_gateway", size=(8, 1))],
            [sg.Text("Duration"), sg.Input("60", key="live_duration", size=(6, 1)), sg.Text("Bootstrap Price"), sg.Input("", key="live_bootstrap", size=(10, 1))],
            [sg.Checkbox("model-optional", key="live_model_optional"), sg.Checkbox("i-understand-testnet", key="live_ack")],
            [sg.Text("Model"), sg.Input("", key="live_model", size=(30, 1))],
            [sg.Button("Run Live", key="live_run"), sg.Button("Stop Live", key="live_stop")],
        ]

        # Data & Tools tab
        tool_buttons = []
        row = []
        for name in sorted(self.whitelist.keys()):
            row.append(sg.Button(name, key=f"tool_{name}"))
            if len(row) == 3:
                tool_buttons.append(row)
                row = []
        if row:
            tool_buttons.append(row)
        tools_layout = [
            [sg.Text("Params: symbol"), sg.Input(self.cfg.get("symbol", "BTCUSDT"), key="tool_symbol", size=(8, 1)), sg.Text("frame"), sg.Input(self.cfg.get("frame", "1h"), key="tool_frame", size=(5, 1))],
            *tool_buttons,
        ]

        # Logs & Errors tab
        logs_layout = [
            [sg.Combo([], key="log_selector", enable_events=True, size=(20, 1)), sg.Button("Copy Last 200", key="copy_log")],
            [sg.Multiline("", size=(100, 20), key="log_output", autoscroll=True, disabled=True)],
        ]

        # Config tab
        env_checks = [
            sg.Text(f"BINANCE_API_KEY: {'✅' if os.getenv('BINANCE_API_KEY') else '⚠️'}"),
            sg.Text(f"BYBIT_API_KEY: {'✅' if os.getenv('BYBIT_API_KEY') else '⚠️'}"),
        ]
        config_layout = [
            [sg.Text(f"GPU available: {'yes' if gpu_available else 'no'}")],
            env_checks,
            [sg.Button("Reset to defaults", key="cfg_reset")],
        ]

        layout = [
            [
                sg.TabGroup(
                    [
                        [
                            sg.Tab("Run Train", train_layout),
                            sg.Tab("Eval & WFA", eval_layout),
                            sg.Tab("Live (Paper/Sandbox)", live_layout),
                            sg.Tab("Data & Tools", tools_layout),
                            sg.Tab("Logs & Errors", logs_layout),
                            sg.Tab("Config", config_layout),
                        ]
                    ],
                    key="tabs",
                )
            ]
        ]

        self.window = sg.Window("bot-trade Control Panel", layout, finalize=True)

        threading.Thread(target=self._log_updater, daemon=True).start()

    # ------------------------------------------------------------------
    def _log_updater(self) -> None:
        while True:
            try:
                item = self.log_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            run_id = item.get("run_id")
            line = item.get("line")
            if run_id is None or line is None:
                continue
            buf = self.log_buffers.setdefault(run_id, [])
            buf.append(line)
            if len(buf) > 200:
                del buf[:-200]
            if self.active_log == run_id:
                self.window["log_output"].update("\n".join(buf))
            if run_id not in self.window["log_selector"].Values:
                self.window["log_selector"].Values.append(run_id)
                self.window["log_selector"].update(values=self.window["log_selector"].Values)

    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            event, values = self.window.read(timeout=500)
            if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, sg.WIN_CLOSED):
                save_config(self.cfg)
                break

            if event == "train_start":
                cmd = self._build_train_cmd(values)
                self.window["cmd_preview"].update(" ".join(cmd))
                run_id = str(int(time.time()))
                log_path = RESULTS_DIR / values["symbol"] / values["frame"] / run_id / "logs" / "train.log"
                pid = runner.start_command(cmd, run_id=run_id, tee_to=str(log_path), log_queue=self.log_queue)
                self.cfg.update({
                    "symbol": values["symbol"],
                    "frame": values["frame"],
                    "preset": values["preset"],
                    "algorithm": values["algorithm"],
                    "max_steps": int(values["max_steps"]),
                    "seed": int(values["seed"]),
                    "data_dir": values["data_dir"],
                    "device": values["device"],
                    "n_envs": int(values["n_envs"]),
                })
                self._refresh_runs()

            if event == "train_stop":
                selected = values.get("runs_table")
                if selected:
                    row = selected[0]
                    pid = int(self.window["runs_table"].Values[row][2])
                    runner.stop_process(pid)
                    self._refresh_runs()

            if event == "runs_table":
                pass  # selection only

            if event == "log_selector":
                run_id = values["log_selector"]
                self.active_log = run_id
                buf = self.log_buffers.get(run_id, [])
                self.window["log_output"].update("\n".join(buf))

            if event and str(event).startswith("tool_"):
                name = event.split("_", 1)[1]
                params = {
                    "symbol": values.get("tool_symbol", self.cfg.get("symbol", "BTCUSDT")),
                    "frame": values.get("tool_frame", self.cfg.get("frame", "1h")),
                    "days": "1",
                    "out": "results/synth",
                    "exchange": "binance",
                    "study": "smoke",
                }
                try:
                    cmd = build_command(name, params, self.whitelist)
                except ValueError as exc:
                    sg.popup_error(str(exc))
                    continue
                run_id = f"tool-{name}-{int(time.time())}"
                log_path = RESULTS_DIR / run_id / "logs" / f"{name}.log"
                runner.start_command(cmd, run_id=run_id, tee_to=str(log_path), log_queue=self.log_queue)
                self._refresh_runs()

            if event == "cfg_reset":
                self.cfg = load_config()  # reload defaults

            self._refresh_runs()

        self.window.close()

    # ------------------------------------------------------------------
    def _build_train_cmd(self, values: Dict) -> List[str]:
        cmd = [
            "python",
            "bot_trade/train_rl.py",
            "--config",
            values.get("preset", "default"),
            "--max-steps",
            str(values.get("max_steps", 1000)),
            "--seed",
            str(values.get("seed", 42)),
            "--data-dir",
            values.get("data_dir", "data"),
            "--algo",
            values.get("algorithm", "PPO"),
        ]
        device = values.get("device", "auto")
        if device != "auto":
            cmd += ["--device", device]
        n_envs = values.get("n_envs")
        if n_envs:
            cmd += ["--n-envs", str(n_envs)]
        cmd.append("--headless")
        return cmd

    # ------------------------------------------------------------------
    def _refresh_runs(self) -> None:
        rows = []
        for info in list(runner._runs.values()):  # type: ignore[attr-defined]
            elapsed = time.time() - info.start_ts
            rows.append(
                [info.run_id, info.meta.get("task", "cmd"), info.process.pid, info.meta.get("device", ""), f"{elapsed:.0f}s", info.status]
            )
        self.window["runs_table"].update(values=rows)


# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - manual run
    panel = Panel()
    panel.run()


if __name__ == "__main__":  # pragma: no cover
    main()
