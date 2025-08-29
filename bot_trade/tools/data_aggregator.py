import logging
logging.basicConfig(level=logging.INFO)
import os
import pandas as pd

def aggregate_results(directory="results"):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("run_"):
            df = pd.read_csv(os.path.join(directory, filename))
            df['source_file'] = filename
            all_data.append(df)
    if not all_data:
        return pd.DataFrame()
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['price_change'] = full_df['price'].pct_change().fillna(0)
    full_df['coin_delta'] = full_df['coin_value'].diff().fillna(0)
    full_df['usdt_delta'] = full_df['usdt'].diff().fillna(0)
    full_df['value_delta'] = full_df['total_value'].diff().fillna(0)
    full_df['pnl_class'] = full_df['status'].map({'WIN': 1, 'LOSS': 0}).fillna(-1)
    return full_df

if __name__ == "__main__":
    df = aggregate_results()
    df.to_csv("training_dataset.csv", index=False)
    logging.info("[✅] Aggregated training data saved to training_dataset.csv")