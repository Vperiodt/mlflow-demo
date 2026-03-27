"""
Train FraudNet on Feast features.

ZERO special imports. ZERO glue code. The ``mlflow:`` block in
feature_store.yaml tells Feast to auto-log everything to MLflow.
This is standard Feast + standard MLflow code.

Run from demo/ directory:
    python training/train.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from feast import FeatureStore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from training.model import FraudNet

FEAST_REPO = os.environ.get("FEAST_REPO", str(PROJECT_ROOT / "feast_repo"))
PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "transactions.parquet"

HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
BATCH_SIZE = 32
VAL_FRAC = 0.2
RANDOM_STATE = 42


def prepare_xy(df: pd.DataFrame):
    exclude = {"user_id", "event_timestamp", "is_fraud"}
    feature_cols = sorted(c for c in df.columns if c not in exclude)
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            X[col] = pd.Categorical(X[col]).codes
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(df["is_fraud"].values).unsqueeze(1)
    return X_tensor, y_tensor, feature_cols


def main():
    # --- Standard Feast code (no glue) ---
    store = FeatureStore(repo_path=FEAST_REPO)
    feature_service = store.get_feature_service("fraud_feature_service")

    raw = pd.read_parquet(PARQUET_PATH)
    entity_df = raw[["user_id", "event_timestamp"]].drop_duplicates()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    ).to_df()
    # ^^^ The bridge auto-logs all Feast metadata to MLflow here ^^^

    labels = raw[["user_id", "event_timestamp", "is_fraud"]].drop_duplicates()
    training_df["event_timestamp"] = pd.to_datetime(training_df["event_timestamp"], utc=True)
    labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"], utc=True)
    training_df = training_df.merge(labels, on=["user_id", "event_timestamp"], how="left")

    X, y, feature_cols = prepare_xy(training_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRAC, random_state=RANDOM_STATE, stratify=y.squeeze().numpy(),
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # --- Standard MLflow code (no glue) ---
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        mlflow.log_params({
            "model": "FraudNet",
            "hidden_dim": HIDDEN_DIM,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        })

        model = FraudNet(input_dim=X.shape[1], hidden_dim=HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

        model.eval()
        with torch.no_grad():
            val_proba = model(X_val).squeeze().numpy()
            val_pred = (val_proba >= 0.5).astype(int)
            y_np = y_val.squeeze().numpy()
            mlflow.log_metrics({
                "accuracy": accuracy_score(y_np, val_pred),
                "auc": roc_auc_score(y_np, val_proba) if len(set(y_np)) > 1 else 0.0,
                "f1": f1_score(y_np, val_pred, zero_division=0),
                "precision": precision_score(y_np, val_pred, zero_division=0),
                "recall": recall_score(y_np, val_pred, zero_division=0),
            })

        mlflow.pytorch.log_model(model, "model")
        # ^^^ The bridge auto-attaches FeatureContract to the model here ^^^

        print(f"Training complete. Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
