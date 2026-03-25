"""Materialize Feast offline features into the online store (Redis)."""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    feast_repo = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "feast_repo")
    parquet = PROJECT_ROOT / "data" / "parquet" / "transactions.parquet"

    from feast import FeatureStore
    store = FeatureStore(repo_path=feast_repo)

    if parquet.exists():
        df = pd.read_parquet(parquet, columns=["event_timestamp"])
        ts = pd.to_datetime(df["event_timestamp"])
        start = ts.min().to_pydatetime()
        end = ts.max().to_pydatetime()
    else:
        start = datetime(2020, 1, 1)
        end = datetime(2030, 12, 31)

    print(f"Materializing: {start} to {end}")
    store.materialize(start_date=start, end_date=end)
    print("Done.")


if __name__ == "__main__":
    main()
