# LOSO Decomposition (2026-10-27)

Per-stack decomposition of the five-fold **leave-one-stack-out (LOSO)** protocol
backing the AAAI-27 rebuttal answer on leave-one-platform-out transfer.

## What this adds

The 2026-10-25 five-stack prediction release reports an aggregate LOSO MRE of
171.03% for the `device_holdout_seen_architectures` protocol (RGAT, probe
features). This experiment decomposes that aggregate into per-held-out-stack
MREs, taken directly from the per-fold records of the frozen summary:

| Held-out stack   | Test records | MRE (%) | Seed SD |
|------------------|--------------|---------|---------|
| Ascend 910B3     | 2,781        | 52.27   | 11.01   |
| x86 CPU          | 2,781        | 56.31   | 12.77   |
| Apple M5 MPS     | 2,781        | 104.98  | 55.48   |
| Apple M5 CPU     | 2,781        | 274.10  | 89.93   |
| A100 MIG 2g.20GB | 2,781        | 367.48  | 285.88  |
| **Record-weighted micro-average** | 13,905 | **171.03** | — |

Each fold trains RGAT on the other four stacks and evaluates on the held-out
stack; every one of the 13,905 records is scored by a model that never saw its
stack during training. The record-weighted micro-average (171.03%) matches the
published aggregate seed mean (171.03%), closing the loop between the
decomposition and the frozen release.

## Method

`loso_decomposition.py` is deterministic and performs **no new training**:

1. Rebuilds `formal-five-stack/summary.json` from the published multipart
   archive `experiments/20261025/archives/prediction-parts/`
   (8 parts, ~218 MB total).
2. Verifies the source `code_sha256` equals
   `bad26be58e40a0bf6ff939024f15e646a72ec3e1b8bc3fef54dc67745f8cce0f`.
3. Extracts the five folds with
   `protocol=device_holdout_seen_architectures`, `method=rgat`,
   `feature_mode=probe`, keyed by `heldout_device_id`.
4. Cross-checks the record-weighted average of per-fold means against the
   aggregate seed mean (must agree within 0.5 percentage points, else exit).
5. Writes `results/loso-decomposition.json`.

Run from this directory:

```bash
python3 loso_decomposition.py
```

## Scope notes

- The five stacks are **newly measured** and differ from the paper's original
  five platforms (A100/910B/MLU370/Max 1550/Xeon 8380); the original
  per-operator labels are not recoverable, and this decomposition is scoped to
  the measured cohort.
- Runtime stacks are not independent hardware architectures: M5 CPU/MPS share
  one machine; the cloud x86 CPU and A100 MIG share a host.
- Ascend 910B3 — the paper's core migration target — is the stack with the
  lowest held-out MRE (52.27%).
