#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-stack leave-one-stack-out (LOSO) decomposition for the HET measured cohort.

Reconstructs the frozen five-stack prediction summary from the published
multipart archive (experiments/20261025/archives/prediction-parts/), extracts
the per-held-out-stack RGAT micro MRE of the device_holdout_seen_architectures
protocol, verifies the record-weighted micro-average against the published
aggregate, and writes results/loso-decomposition.json.

No new training is performed; this is a deterministic decomposition of the
already-published 2026-10-25 five-stack prediction release (code_sha256
bad26be58e40a0bf6ff939024f15e646a72ec3e1b8bc3fef54dc67745f8cce0f).
"""
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = HERE.parent / "20261025" / "archives" / "prediction-parts"
OUT = HERE / "results" / "loso-decomposition.json"

EXPECTED_CODE_SHA = "bad26be58e40a0bf6ff939024f15e646a72ec3e1b8bc3fef54dc67745f8cce0f"
STACK_LABELS = {
    "Ascend-910B3-64GB": "Ascend 910B3",
    "Cloud-x86-CPU-4T": "x86 CPU",
    "Apple-M5-MPS": "Apple M5 MPS",
    "Apple-M5-CPU-4T": "Apple M5 CPU",
    "A100-SXM4-MIG-2g20GB": "A100 MIG 2g.20GB",
}


def rebuild_summary():
    parts = sorted(PARTS.glob("prediction-formal-all-files.zip.part*"))
    if not parts:
        raise SystemExit(f"no archive parts found under {PARTS}")
    blob = b"".join(p.read_bytes() for p in parts)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.endswith("formal-five-stack/summary.json")]
        if not names:
            raise SystemExit("formal-five-stack/summary.json not found inside rebuilt archive")
        return json.loads(z.read(names[0]))


def main():
    d = rebuild_summary()
    if d.get("code_sha256") != EXPECTED_CODE_SHA:
        raise SystemExit("source summary code_sha256 mismatch")

    folds = [r for r in d["results"]
             if r.get("protocol") == "device_holdout_seen_architectures"
             and r.get("method") == "rgat" and r.get("feature_mode") == "probe"]
    if len(folds) != 5:
        raise SystemExit(f"expected 5 LOSO folds, found {len(folds)}")

    per_stack = sorted(
        ({
            "heldout_stack": r["heldout_device_id"],
            "label": STACK_LABELS.get(r["heldout_device_id"], r["heldout_device_id"]),
            "test_records": r["test_records"],
            "seed_micro_mre_mean_percent": round(r["seed_micro_mre_mean"], 2),
            "seed_micro_mre_sample_sd": round(r["seed_micro_mre_sample_sd"], 2),
            "split_id": r["split_id"],
        } for r in folds),
        key=lambda x: x["seed_micro_mre_mean_percent"])

    total = sum(f["test_records"] for f in per_stack)
    weighted = sum(f["seed_micro_mre_mean_percent"] * f["test_records"]
                   for f in per_stack) / total

    agg = next(a for a in d["aggregate_results"]
               if a.get("protocol") == "device_holdout_seen_architectures"
               and a.get("method") == "rgat" and a.get("feature_mode") == "probe")
    seed_mean = sum(s["micro_mre_percent"] for s in agg["seed_metrics"]) / len(agg["seed_metrics"])
    if abs(weighted - seed_mean) > 0.5:
        raise SystemExit(f"verification failed: weighted {weighted:.2f} vs aggregate {seed_mean:.2f}")

    out = {
        "purpose": "Per-stack decomposition of the five-fold leave-one-stack-out "
                   "(LOSO) protocol; each fold trains RGAT on four stacks and "
                   "evaluates on the held-out stack.",
        "source": {
            "archive": "experiments/20261025/archives/prediction-parts",
            "protocol": "device_holdout_seen_architectures",
            "feature_mode": "probe",
            "method": "rgat",
            "code_sha256": d["code_sha256"],
            "seeds": d["seeds"],
            "labels": d["record_counts"]["labels"],
            "stacks": d["record_counts"]["physical_runtime_stacks"],
        },
        "loso_folds": per_stack,
        "micro_average_weighted_by_records_percent": round(weighted, 2),
        "aggregate_seed_mean_percent": round(seed_mean, 2),
        "notes": [
            "Every one of the 13,905 records is scored by a model that never saw "
            "its stack during training.",
            "Runtime stacks are not independent hardware architectures: M5 CPU/MPS "
            "share a machine; cloud CPU and A100 MIG share a host.",
            "These stacks are newly measured and differ from the paper's original "
            "five platforms; the original per-edge labels are not recoverable.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({
        "per_stack": [(f["label"], f["seed_micro_mre_mean_percent"]) for f in per_stack],
        "weighted_micro_average_percent": round(weighted, 2),
        "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
