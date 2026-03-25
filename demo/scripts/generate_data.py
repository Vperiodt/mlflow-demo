"""Generate synthetic fraud detection data."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

NUM_ROWS = 500
FRAUD_RATE = 0.15
SEED = 42


def generate():
    rng = np.random.default_rng(SEED)

    user_ids = [f"user_{i}" for i in rng.integers(1, 100, size=NUM_ROWS)]
    timestamps = pd.date_range("2025-01-01", periods=NUM_ROWS, freq="h")

    amounts = rng.exponential(scale=200, size=NUM_ROWS).round(2)
    categories = rng.choice(
        ["grocery", "electronics", "travel", "dining", "gas", "online"],
        size=NUM_ROWS,
    )
    distances = rng.exponential(scale=15, size=NUM_ROWS).round(2)
    avg_7d = amounts * rng.uniform(0.5, 1.5, size=NUM_ROWS)
    avg_7d = avg_7d.round(2)
    txn_count_24h = rng.poisson(lam=3, size=NUM_ROWS)
    is_foreign = rng.binomial(1, 0.1, size=NUM_ROWS)

    fraud_score = (
        (amounts > 400).astype(float) * 0.3
        + (is_foreign * 0.25)
        + (distances > 30).astype(float) * 0.2
        + (txn_count_24h > 5).astype(float) * 0.15
        + rng.uniform(0, 0.1, size=NUM_ROWS)
    )
    is_fraud = (fraud_score > 0.4).astype(int)
    target_fraud_count = int(NUM_ROWS * FRAUD_RATE)
    if is_fraud.sum() > target_fraud_count:
        fraud_indices = np.where(is_fraud == 1)[0]
        excess = len(fraud_indices) - target_fraud_count
        flip = rng.choice(fraud_indices, size=excess, replace=False)
        is_fraud[flip] = 0

    df = pd.DataFrame({
        "user_id": user_ids,
        "event_timestamp": timestamps,
        "transaction_amount": amounts,
        "merchant_category": categories,
        "distance_from_last_txn": distances,
        "avg_txn_amount_7d": avg_7d,
        "txn_count_24h": txn_count_24h,
        "is_foreign_txn": is_foreign,
        "is_fraud": is_fraud,
    })

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RAW_DIR / "transactions.csv"
    parquet_path = PARQUET_DIR / "transactions.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"Generated {NUM_ROWS} rows ({is_fraud.sum()} fraud)")
    print(f"  CSV:     {csv_path}")
    print(f"  Parquet: {parquet_path}")


if __name__ == "__main__":
    generate()
