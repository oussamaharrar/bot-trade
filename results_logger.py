import pandas as pd
import os
import json
from datetime import datetime
import os

# Ensure required directories exist
for d in ["models", "results", "reports", "logs"]:
    os.makedirs(d, exist_ok=True)
 

def simulate_wallet(actions, initial_usdt=None, fee=0.001):
    state_path = os.path.join(os.path.dirname(__file__), "latest_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        if initial_usdt is None:
            initial_usdt = state.get("final_value", 1000.0)
    else:
        initial_usdt = 1000.0
        state = {}

    usdt = initial_usdt
    coin = 0.0
    logs = []
    last_total = usdt
    win_streak = 0
    loss_streak = 0

    for ts, price, signal in actions:
        note = "➖ Hold"
        pnl = 0.0
        status = "HOLD"

        if win_streak >= 2:
            position_size = 0.6
        elif loss_streak >= 2:
            position_size = 0.3
        else:
            position_size = 0.5

        if signal == "BUY" and usdt > 0:
            buy_amount = usdt * position_size
            new_coin = (buy_amount / price) * (1 - fee)
            coin += new_coin
            usdt -= buy_amount
            note = f"🟢 Buy @{position_size*100:.0f}%"
        elif signal == "SELL" and coin > 0:
            sell_amount = coin * position_size
            proceeds = (sell_amount * price) * (1 - fee)
            usdt += proceeds
            coin -= sell_amount
            note = f"🔴 Sell @{position_size*100:.0f}%"

        total = usdt + coin * price
        pnl = total - last_total
        last_total = total

        if pnl > 0.01:
            status = "WIN"
            win_streak += 1
            loss_streak = 0
        elif pnl < -0.01:
            status = "LOSS"
            win_streak = 0
            loss_streak += 1
        else:
            win_streak = 0
            loss_streak = 0

        logs.append((ts, price, signal, usdt, coin * price, total, note, pnl, status))

    final_value = usdt + coin * actions[-1][1]

    df = pd.DataFrame(logs, columns=[
        "timestamp", "price", "signal", "usdt", "coin_value",
        "total_value", "note", "pnl", "status"
    ])
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    run_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(os.path.join(results_dir, run_filename), index=False)

    with open(state_path, "w") as f:
        json.dump({
            "last_run_file": run_filename,
            "final_value": final_value
        }, f, indent=2)

    # Append to training dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "training_dataset.csv")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    df["price_change"] = df["price"].pct_change().fillna(0)
    df["coin_delta"] = df["coin_value"].diff().fillna(0)
    df["usdt_delta"] = df["usdt"].diff().fillna(0)
    df["value_delta"] = df["price_change"] + df["coin_delta"] + df["usdt_delta"]
    df["pnl_class"] = df["pnl"].apply(lambda x: 1 if x > 0 else (0 if x < 0 else -1))

    train_cols = [
        "timestamp", "price", "price_change", "coin_value", "usdt",
        "coin_delta", "usdt_delta", "value_delta", "signal", "pnl", "pnl_class"
    ]
    train_df = df[train_cols]
    if os.path.exists(dataset_path):
        train_df.to_csv(dataset_path, mode='a', index=False, header=False)
    else:
        train_df.to_csv(dataset_path, index=False)

    return logs, final_value
