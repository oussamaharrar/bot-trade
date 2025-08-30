import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
import zipfile
from bot_trade.config.rl_paths import dataset_path


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV, ZIP of CSVs, or Parquet file.

    ``path`` is resolved via :func:`rl_paths.dataset_path` so that relative
    paths are interpreted against project ``<ROOT>``.
    """
    p = dataset_path(path)
    if str(p).endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            dfs = [pd.read_csv(z.open(f)) for f in z.namelist() if f.endswith('.csv')]
        if not dfs:
            raise ValueError("ZIP file contains no CSVs")
        df = pd.concat(dfs, ignore_index=True)
    elif str(p).endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    return df
