import logging
logging.basicConfig(level=logging.INFO)
import os
import subprocess
import argparse
from datetime import datetime
import yaml
import pandas as pd
from env_config import LIVE_TRADING

CONFIG_PATH = 'config.yaml'

with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

MODEL_DIR = CONFIG['output_dirs']['models']
RESULTS_DIR = CONFIG['output_dirs']['results']
REPORTS_DIR = CONFIG['output_dirs']['reports']
LOG_DIR = CONFIG['output_dirs']['logs']
MAX_TRADES = CONFIG.get('max_trades_before_retrain', 100)
STRATEGY = CONFIG.get('strategy', 'ml')

for d in [MODEL_DIR, RESULTS_DIR, REPORTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

ERROR_LOG = os.path.join(LOG_DIR, 'errors.log')
EVAL_LOG = 'model_evaluation_log.csv'

def log_error(context: str, err: Exception):
    msg = f"[{datetime.now()}] {context}: {err}\n"
    with open(ERROR_LOG, 'a') as f:
        f.write(msg)
    logging.info(f"❌ {context}: {err}")


def run_step(name: str, cmd: list[str]):
    logging.info(f"▶ {name}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            log_error(name, res.stderr.strip())
        else:
            if CONFIG.get('debug_mode', False):
                logging.info(res.stdout)
            logging.info(f"✅ {name} done\n")
    except Exception as e:
        log_error(name, e)


def trade_count() -> int:
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('run_') and f.endswith('.csv')]
    return len(files)


def clean_eval_log():
    if not os.path.exists(EVAL_LOG):
        return
    try:
        df = pd.read_csv(EVAL_LOG)
        required = ['timestamp','accuracy','f1_score','f1_delta','num_samples','model_path']
        df = df[[c for c in required if c in df.columns]].dropna()
        df.to_csv(EVAL_LOG, index=False)
    except Exception as e:
        log_error('clean_eval_log', e)


def maybe_retrain():
    if trade_count() and trade_count() % MAX_TRADES == 0:
        run_step('Training model', ['python', 'autolearn.py'])
        clean_eval_log()
        run_step('Evaluating model', ['python', 'evaluate_model.py'])
    else:
        logging.info(f"ℹ️ Trades logged: {trade_count()}. Retrain at {MAX_TRADES} trades")


def _spawn_monitors():
    """Launch auxiliary monitors in separate consoles."""
    try:
        from tools.monitor_launch import launch_new_console
    except Exception as exc:  # pragma: no cover
        logging.warning("monitor launch helper missing: %s", exc)
        return
    tools_dir = os.path.join(os.path.dirname(__file__), "tools")
    launch_new_console(
        "RESOURCE-MON",
        os.path.join(tools_dir, "resource_monitor.py"),
        ["--base", RESULTS_DIR],
    )
    launch_new_console(
        "REPORTS",
        os.path.join(tools_dir, "generate_markdown_report.py"),
        ["--watch"],
    )


def main(args):
    logging.info('\n' + '='*60)
    logging.info('🚀 Running trading bot')
    logging.info('='*60)
    if args.spawn_monitors:
        _spawn_monitors()

    if not LIVE_TRADING:
        logging.info('⚠️  LIVE_TRADING disabled - running in simulation mode')
    try:
        if STRATEGY == 'ml':
            script = 'bot_loop_ml.py'
        elif STRATEGY == 'rl':
            script = 'run_rl_agent.py'
        else:
            script = 'bot_loop.py'
        run_step('Running trading logic', ['python', script])
        maybe_retrain()
        if CONFIG.get('knowledge', {}).get('run_after_training', False):
            try:
                subprocess.run([
                    'python',
                    'tools/knowledge_sync.py',
                    '--results-dir', 'results',
                    '--agents-dir', 'agents',
                    '--out', os.path.join('memory', 'knowledge_base_full.json'),
                ], check=False)
            except Exception:
                pass
    except Exception as e:
        log_error('main', e)
    logging.info('✅ Run complete. Check reports/ and models/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spawn-monitors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch resource and report monitors in separate windows",
    )
    cli_args = ap.parse_args()
    main(cli_args)
