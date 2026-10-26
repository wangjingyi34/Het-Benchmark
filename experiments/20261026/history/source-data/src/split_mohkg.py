#!/usr/bin/env python3
"""Create the paper's deterministic model-group split and leakage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def grouped_split(model_ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    ids = sorted(set(model_ids))
    random.Random(seed).shuffle(ids)
    n_train = round(0.8 * len(ids))
    n_val = round(0.1 * len(ids))
    return {
        "train": sorted(ids[:n_train]),
        "validation": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="data/paper_schema_counts.json")
    parser.add_argument("--models", default="data/model_dataset.json")
    parser.add_argument("--output", default="reproduced/model_group_split.json")
    args = parser.parse_args()

    schema = json.loads(Path(args.schema).read_text())
    model_data = json.loads(Path(args.models).read_text())
    model_ids = [m["model_id"] for m in model_data["models"]]
    split = grouped_split(model_ids, schema["split"]["seed"])
    sets = [set(split[k]) for k in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(model_ids)
    expected = schema["split"]["model_counts"]
    actual = [len(split[k]) for k in ("train", "validation", "test")]
    assert actual == expected, (actual, expected)

    payload = {
        "seed": schema["split"]["seed"],
        "group": "source_model",
        "ratios": schema["split"]["ratios"],
        "split": split,
        "leakage_controls": [
            "remove held-out performance labels and edges from the training graph",
            "mask similar edges crossing train/validation/test model groups",
            "select checkpoints on validation only and open test labels once",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}; model counts={actual}; leakage audit=PASS")


if __name__ == "__main__":
    main()
