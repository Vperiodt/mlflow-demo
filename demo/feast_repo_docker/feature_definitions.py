"""Feast feature definitions for fraud detection demo."""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, FileSource
from feast.data_format import ParquetFormat
from feast.field import Field
from feast.types import Float64, Int64, String
from feast.value_type import ValueType

TRANSACTIONS_PARQUET = "/data/parquet/transactions.parquet"

user = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.STRING,
)

transaction_source = FileSource(
    path=TRANSACTIONS_PARQUET,
    timestamp_field="event_timestamp",
    file_format=ParquetFormat(),
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="transaction_amount", dtype=Float64),
        Field(name="merchant_category", dtype=String),
        Field(name="distance_from_last_txn", dtype=Float64),
        Field(name="avg_txn_amount_7d", dtype=Float64),
        Field(name="txn_count_24h", dtype=Int64),
        Field(name="is_foreign_txn", dtype=Int64),
    ],
    source=transaction_source,
    online=True,
)

fraud_feature_service = FeatureService(
    name="fraud_feature_service",
    features=[transaction_features],
)
