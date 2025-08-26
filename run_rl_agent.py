"""
Run an RL agent on selected data, generate trades with full info, evaluation, performance charts, and PDF report.
Enhanced for compatibility with updated env_trading and RiskManager.
"""
import os
import glob
import logging
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv
from evaluate import evaluate_trades

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Load market data from CSV or Feather."""
    if path.endswith(".feather"):
        return feather.read_table(path).to_pandas()
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {path}")


def make_env(df: pd.DataFrame, vec_path: str | None = None):
    """Create a normalized vectorized environment."""
    env = DummyVecEnv([lambda: TradingEnv(df)])
    if vec_path and os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
    return env


def run(model_path: str, data_path: str, frame: str, out_dir: str = "results"):
    df = load_data(data_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    logger.info(f"🚀 Running model {model_name} on frame {frame}...")

    # Prepare environment and model
    vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    env = make_env(df, vecnorm_path)
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    done = False
    trades = []

    # Step through environment
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        act = int(action[0] if hasattr(action, '__len__') else action)

        result = env.step([act])
        # Unpack result robustly
        if len(result) == 5:
            obs, rewards, terms, truncs, infos = result
            dones = terms | truncs
        else:
            obs, rewards, dones, infos = result

        reward = float(rewards[0] if hasattr(rewards, '__len__') else rewards)
        done = bool(dones[0] if hasattr(dones, '__len__') else dones)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        # Collect trade info
        trades.append({
            'timestamp': info.get('timestamp'),
            'symbol': info.get('symbol'),
            'action': act,
            'pnl': reward,
            'total_value': info.get('total_value'),
            'risk_pct': info.get('risk_pct'),
            'danger_mode': info.get('danger_mode'),
            'freeze_mode': info.get('freeze_mode'),
            'notes': ";".join(info.get('notes', []))
        })

    # Save trades DataFrame
    df_trades = pd.DataFrame(trades)
    os.makedirs(out_dir, exist_ok=True)
    trades_file = os.path.join(out_dir, f"{model_name}_trades.csv")
    df_trades.to_csv(trades_file, index=False)
    logger.info(f"✅ Trades saved to {trades_file}")

    # Evaluate and save metrics
    eval_file = os.path.join(out_dir, f"{model_name}_evaluation.csv")
    evaluate_trades(df_trades, model_name=model_name, save_path=eval_file, additional_info={'frame': frame})
    logger.info(f"✅ Evaluation saved to {eval_file}")

    # Plot performance
    perf_dir = os.path.join(out_dir, 'performance')
    os.makedirs(perf_dir, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(df_trades['total_value'], label='Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f'Performance: {model_name} [{frame}]')
    plt.grid(True)
    perf_file = os.path.join(perf_dir, f"{model_name}_performance.png")
    plt.tight_layout()
    plt.savefig(perf_file)
    plt.close()
    logger.info(f"📈 Performance chart saved to {perf_file}")

    # Generate PDF report
    rep_dir = os.path.join(out_dir, 'reports')
    os.makedirs(rep_dir, exist_ok=True)
    pdf_file = os.path.join(rep_dir, f"{model_name}_report.pdf")
    with PdfPages(pdf_file) as pdf:
        # Performance plot
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_trades['total_value'], label='Portfolio Value')
        ax.set_xlabel('Step'); ax.set_ylabel('Value'); ax.set_title(f'Performance: {model_name}')
        ax.grid(True)
        pdf.savefig(fig); plt.close(fig)

        # Evaluation text
        from io import StringIO
        import sys
        buf = StringIO()
        old = sys.stdout; sys.stdout = buf
        evaluate_trades(df_trades, model_name=model_name, save_path=eval_file, additional_info={'frame': frame})
        sys.stdout = old
        text = buf.getvalue()
        fig_txt = plt.figure(figsize=(10,4))
        plt.axis('off')
        plt.text(0,1, text, family='monospace', va='top')
        pdf.savefig(fig_txt); plt.close(fig_txt)
    logger.info(f"📄 Report saved to {pdf_file}")


if __name__ == '__main__':
    # Select currency
    currencies = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','CUSDT']
    print("\nSelect a currency:")
    for idx, c in enumerate(currencies,1): print(f"{idx}. {c}")
    print("0. All currencies")
    choice = int(input("Enter choice: "))
    currency = currencies[choice-1] if 1<=choice<=len(currencies) else None

    # Select timeframe
    tf_dirs = sorted([d for d in os.listdir('agents') if os.path.isdir(os.path.join('agents',d))])
    print("\nSelect a timeframe:")
    for idx, tf in enumerate(tf_dirs,1): print(f"{idx}. {tf}")
    print("0. All timeframes")
    choice = int(input("Enter choice: "))
    frame = tf_dirs[choice-1] if 1<=choice<=len(tf_dirs) else None

    # Find models
    pattern = f"agents/{frame or '*'}/*.zip"
    models = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not models:
        logger.error(f"No models found for timeframe={frame}"); exit(1)
    print("\nSelect a model:")
    for idx, m in enumerate(models[:10],1): print(f"{idx}. {m}")
    choice = int(input("Enter choice: "))
    model_path = models[choice-1]

    # Find data
    data_files = sorted(glob.glob(f"data/{frame or '*'}/*.feather") + glob.glob(f"data/{frame or '*'}/*.csv"), key=os.path.getmtime, reverse=True)
    if currency: data_files = [f for f in data_files if currency in f]
    if not data_files:
        logger.error(f"No data files for currency={currency}, timeframe={frame}"); exit(1)
    print("\nSelect data file:")
    for idx, f in enumerate(data_files[:10],1): print(f"{idx}. {f}")
    choice = int(input("Enter choice: "))
    data_path = data_files[choice-1]

    run(model_path, data_path, frame)
