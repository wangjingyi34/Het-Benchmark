#!/usr/bin/env python3
"""Validate paper-facing counts, files, and record metadata."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    schema = json.loads((ROOT / "data/paper_schema_counts.json").read_text())
    models = json.loads((ROOT / "data/model_dataset.json").read_text())
    operators = json.loads((ROOT / "data/operators.json").read_text())
    hardware = json.loads((ROOT / "data/hardware_platforms.json").read_text())
    assert len(models["models"]) == schema["node_counts"]["model"] == 34
    assert len(operators) == schema["node_counts"]["operator_instance"] == 6244
    assert len(hardware) == schema["node_counts"]["hardware"] == 5
    assert [row["name"] for row in hardware] == [
        "NVIDIA A100",
        "Huawei Ascend 910B",
        "Cambricon MLU370-X8",
        "Intel GPU Max 1550",
        "Intel Xeon 8380",
    ]
    assert [row["bandwidth_gbps"] for row in hardware] == [2000, 1200, 307, 3200, 204]
    assert sum(v for k, v in schema["edge_counts"].items() if k != "total") == 29199
    required = [
        "src/copa.py", "src/kg_a2o.py", "src/moh_kg.py", "src/rgat.py", "src/priority.py",
        "tests/test_priority_scoring.py",
        "results/paper_tables/benchmark_corpus.csv", "results/paper_tables/prediction_mre.csv",
        "docs/PROVENANCE.md", "ARTIFACT_MANIFEST.json",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    assert not missing, f"missing files: {missing}"
    prediction = (ROOT / "results/paper_tables/prediction_mre.csv").read_text()
    assert "Overall,32.0,25.6,14.3,7.8,10.4,14.3" in prediction
    shapley = (ROOT / "results/paper_tables/shapley_sampling.csv").read_text()
    assert "Antithetic,500,0.98,0.31,238.5" in shapley
    legacy_graph = ROOT / "data/moh_kg.json"
    if legacy_graph.exists():
        legacy = json.loads(legacy_graph.read_text())
        assert legacy["metadata"].get("paper_canonical") is False
    print("artifact validation=PASS")
    print(f"models={len(models['models'])}, operators={len(operators)}, edges={schema['edge_counts']['total']}")


if __name__ == "__main__":
    main()
