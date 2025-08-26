import pandas as pd
from pathlib import Path

file_2017 = Path("data/1s/BTCUSDT-1s-2017-08.csv")
file_2025 = Path("data/1s/BTCUSDT-1s-2025-06.csv")

df_2017 = pd.read_csv(file_2017, header=None)
df_2025 = pd.read_csv(file_2025, header=None)



def inspect_open_time(df):
    ot = pd.to_numeric(df[0], errors='coerce')
    ratio_seconds = float((ot < 1e12).mean())
    return {
        'rows': len(ot),
        'min': ot.min(),
        'max': ot.max(),
        'ratio_seconds': ratio_seconds
    }

print("2017-08:", inspect_open_time(df_2017))
print("2025-06:", inspect_open_time(df_2025))