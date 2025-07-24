from market_data import fetch_ohlcv
from ml_strategy import apply_ml_strategy
from results_logger import simulate_wallet

def main():
    print("[🟡] Fetching data...")
    df = fetch_ohlcv()
    print("[🤖] Applying ML strategy...")
    actions = apply_ml_strategy(df)

    print("[📊] Simulating wallet...")
    logs, final_value = simulate_wallet(actions)
    for ts, price, signal, usdt, coin_val, total_val, note, pnl, status in logs:
        print(f"{ts} | Price: {price:.4f} | Action: {signal:<5} | USDT: {usdt:.2f} | CoinVal: {coin_val:.2f} | Total: {total_val:.2f} | {status}")

    print("\\n[✅] Final Portfolio Value:", round(final_value, 2), "USDT")
    print("[💾] Results saved to results.csv")

if __name__ == "__main__":
    main()