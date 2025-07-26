import pandas as pd
import zipfile


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV, ZIP of CSVs, or Parquet file."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            dfs = [pd.read_csv(z.open(f)) for f in z.namelist() if f.endswith('.csv')]
        if not dfs:
            raise ValueError("ZIP file contains no CSVs")
        df = pd.concat(dfs, ignore_index=True)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df
