HET measured validation continuation 26 October 2026

This release preserves the new scientific code, measurements, checkpoints, complete outcomes and replay inputs. It extends the immutable 20261025 release. No reviewer correspondence is included. It does not certify the original five-hardware LOPO or retrospectively authenticate the original 14.3%, 15.6% or budget-based 66% claims.

Attention execution
attention/ contains two distinct experiments on an A100 SXM4 MIG 2g.20GB partition and Apple M5 MPS. The first freezes 24 new configurations, selects three anchors using log-prediction disagreement and evaluates the other 21 in five fresh processes. MRE changes 91.36 to 313.62% on A100 and 97.63 to 71.92% on MPS. All results are retained.
The follow-up is explicitly designed after that result, on different new configurations. The support gate identifies full-MHA composite leaves absent from the functional-SDPA training target catalog and directly profiles all 24 queries. Five fresh-process repeats produce mean relative differences 0.2193%/6.6886%, with initial profiling overhead 4.1579/16.6080 seconds on A100/MPS. These are direct-measurement repeatability results, not learned prediction accuracy or three-probe transfer. All 24 new-type queries are measured, so this path has 100% profiling cost.
The two experiments contain 24 fresh processes, 504 query-process records, 5040 measured timing blocks and 20160 timed calls; warmups and feature capture are separately implied by the code. All raw timing blocks, frozen policies and input/weight hashes are retained. M5 chip identity was captured after execution on the same local host and is explicitly marked as such.
Verify without hardware: python attention/verify_attention_evidence.py
To rerun hardware, use each frozen script's CLI with a new output directory and the matching exact hardware descriptor. Never overwrite recorded outputs. Keep source bytes unchanged when replaying a frozen protocol.

Prediction
prediction/PUBLIC-REPRODUCING.txt gives the complete restoration and replay procedure. The uncertainty panel covers 2,781,000 predictions, 1100 fits and 7200 detailed cells. It separates query dispersion, conditional training-seed variability and descriptive source-model/runtime-stack cluster intervals.
The nn-Meter RF component adaptation uses source-pinned official code with its MIT license, 350 strictly compatible Conv/FC queries (2.52% of the full measured cohort), 200 RF and 200 same-information MLP fits. RF MRE 57.54% and MLP 167.83% are restricted to this narrow support; a sparse fold has one training query. All 25 undefined OOB R2 values are retained as null with reasons. Archived richer-training references are labelled separately. This is not full nn-Meter fusion discovery/adaptive profiling or an unseen-device baseline reproduction.
All 400 new checkpoints, all 10500 prediction rows including references, exclusions and scores are present. The prior 2.78-million-row archive is referenced by exact digest rather than duplicated.

Planning and strict quality gate
planning/ includes all 160 policy-fit results, every eight-context comparison and an identity-preserving A100-to-Ascend ResNet18/batch1 trace. The selected migration trace has raw mean gain -3.7524% and a wide interval; it is distinct from the target-side A100 2.097% example. All outcomes and the all-valid-action mask boundary are retained.
strict_quality_gate.py and upgrade_final_gate_example.py recompute quality from raw predictions/labels, bind sample order and model/input/run/data identities, and enforce the declared retention contract. Absolute or unknown quality modes, incompatible dataset/metric, malformed inputs and contradictory thresholds fail closed. Trusted manifests are an explicit boundary. Baseline fallback is a decision; this release does not claim a deployed rollback.
The software repair passed 30 tests containing 119 negative cases, 440 real-record decisions and 800 archived scores. The previous immutable planner is retained for historical reproducibility; use the explicit new adapter for the strengthened gate.
Restore the prior release, then prepare a separate planning workspace:
  python planning/prepare_public_replay.py --prior-release ../20261025 --restored-root ../20261025/run-data --workspace NEW_EMPTY_WORKSPACE
Run from its v7-剩余条件实证闭环/planning directory with Python 3.12, NumPy and PyTorch 2.9.1 available:
  python test_strict_quality_gate.py
  python test_quality_contract_adapter.py
  python replay_strict_quality_gate.py
  python recompute_planning.py --reload-policies
The adapter verifies 405 input digests and reconstructs the original relative layout. The public export de-identifies one non-numerical purpose field and permits omission of a private review file from test-report metadata. Exact transformations and both digests are in public-export-adjustments.json; no numerical inputs, module assertions or results change. Historical cloud-controller source snapshots are provenance only; the portable analysis does not launch cloud workloads.

Historical provenance
history/ contains public source snapshots at fixed commits, reproducible arithmetic and a strict original-platform LOPO input validator. It distinguishes source estimates from physical labels. The currently available original input candidate has no qualifying per-query physical label ledger, so its gate must fail instead of creating fabricated folds. Software fixtures only test that validator.
  python history/verify_package.py --recompute
JSON numerical recomputation tolerates 1e-12 floating-point rounding across Python versions; source and artifact hashes remain byte-exact. Non-floating fields, key/order structure and CSV bytes remain strict.

Release verification
  python verify_release.py
The manifest covers every distributed file. Workload installation dependencies and original third-party image datasets are referenced through the prior release rather than redistributed as new outputs. No paid NPU was restarted for this continuation; its preserved raw measurements were used only for offline replay.
