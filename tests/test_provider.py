"""Tests for feast_mlflow.providers.base."""

from feast_mlflow.providers.base import (
    FeatureSchema,
    LineageEdge,
    LineageGraph,
    LineageNode,
)


def test_lineage_graph_to_dict():
    graph = LineageGraph(
        nodes=[
            LineageNode(id="ds:data.parquet", node_type="datasource", label="data.parquet", metadata={}),
            LineageNode(id="fv:txn", node_type="featureview", label="txn", metadata={"feature_count": 3}),
        ],
        edges=[
            LineageEdge(source="ds:data.parquet", target="fv:txn"),
        ],
    )
    d = graph.to_dict()
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 1
    assert d["nodes"][0]["type"] == "datasource"
    assert d["edges"][0]["source"] == "ds:data.parquet"
