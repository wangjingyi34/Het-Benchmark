# HET measured validation framework and dataset

This release provides the executable framework, complete measurements, frozen prediction and planning experiments, checkpoints, run provenance, and independent numerical verification from the repeated validation campaign of 25 October 2026. Start with [REPRODUCING.md](REPRODUCING.md).

## Contents and units

| Study | Executed scope |
|---|---|
| Operator measurements | 12 executable architectures, 3 inputs each, 5 runtime stacks; 180 graphs, 13,905 measured stack/query records from 2,781 contextualized operator instances; 278,100 synchronized timing blocks |
| Runtime stacks | Apple M5 CPU, Apple M5 MPS, Xeon Platinum 8369B CPU, A100 SXM4 MIG 2g.20GB, Ascend 910B3 64GB |
| Primary prediction | Ridge, MLP, generic GCN, RGAT and six relation drops; 6 model folds and 5 device-stack folds; 5 seeds; strict/3-probe modes; 1,100 actual fits |
| Training-size curves | Fixed train-only budgets 2/4/6 models, same held-out folds, 180 fits; full-8 results reused exactly |
| Evidence-memory prediction | Training-only measured anchors, validation-only parameter selection, semantic/family/ridge controls; 66 completed evaluations |
| Calibration-budget extension | 14 additional fixed family probes on each of 5 stacks, 70 actual records; total auxiliary budget 17 workload types; 33 fixed-parameter evaluations |
| Pretrained full-model interventions | ResNet18/50, batches 1/4, 4 backends, 8 fresh processes per context; 128 blocks, 2,048 physically executed coalitions, 61,440 timings, 204,800 per-image predictions |
| Actual PPO | 8 A100/Ascend contexts, 4 configurations, 5 seeds, 4,096 steps; 160 fitted policies and checkpoints |
| Independent Ascend confirmation | 20 fresh processes, 11 budget-feasible masks each, 20 adjacent balanced AB/BA pairs; 4,400 pairs and 8,800 timed calls, including baseline/baseline controls |
| Source-to-target diagnosis | 4 A100-to-Ascend traces, 69/175 actual operator players, 500 permutations and a separate 5,000-permutation surrogate reference |
| Active measurement | Frozen source-only ensemble, 732 queries on unseen Ascend, 83 triggered queries actually remeasured; exact deployed policy and decision-before-timing receipts |
| Task-quality gate | True labelled fault rejection, followed by a new physical baseline execution and restored predictions/quality |
| Attention challenge | 12 fixed complete-MHA configurations on 5 stacks, 60 real cases, 900 frozen predictions; module-plus-shape OOD is separately reported |

Repeated timing calls, process blocks, query records, architectures, and unique quality images are different statistical units. M5 CPU/MPS share a workstation; Xeon/A100 share a host. The quality audit uses 100 calibration and 100 disjoint evaluation images from a fixed ten-class ImageNet validation subset, retaining the original 1,000-way classification head.

## Results and evidence

All results are supplied, including uncertainty intervals, regressions, out-of-support cases, capability errors, empty-action controls, and setup/printing failures. Predictor MRE, model/family averages, MAE, WAPE, ranking utility, physical coalition error, sampling fidelity, raw optimization effects, acceptance-gated effects, and actual profiling cost are separate quantities. No endpoint reconciliation or nonnegative clipping is used as physical attribution validation.

The framework also includes training-evidence retrieval and controlled hardware-calibration budgets. The fixed 3-to-17-probe comparison retains the original selected parameters and the same evaluation records. The release does not choose its method or calibration budget by the most favorable test result. Complete-MHA OOD includes projections and is not pooled with primitive SDPA labels.

The complete graph-aware PPO configuration, same-capacity zero-embedding control, no-mask control and matched-evidence search/ranking baselines are all retained. All four main fold actions are supported, so the mask comparison correctly has zero effect in that domain. Fault controls and unsupported API probes are separate evidence. Positive effects on a particular execution stack are not presented as universal hardware speedups.

## Verify and reproduce

```sh
python restore_archives.py
python restore_archives.py --extract
python run.py audit-traceability -- --raw-dir run-data/evidence/fullmodel_raw --task-dir run-data/evidence/final-tasks --output run-data/reproduced-traceability-audit.json
```

`RELEASE-SHA256.json` covers the complete published file set. The 247,705,237-byte original predictor archive contains 1,302 files, including 1,285 actual checkpoints (1,100 primary, 180 learning-curve, 5 deployment-policy), and every per-record prediction. It is split into eight normal Git-sized pieces; restoration checks each piece and the joined archive hash. The full original archive hash is `81f33e54f9d3a2c118710819bdbd3101afaba6ed2134ccb28e08e1a5ad6eca7e`.

The [technical validation matrix](TECHNICAL-VALIDATION.md) links the complete protocol and evidence coverage. [Mathematical definitions](documentation/THEORY.md) specify the additive limit, coalition-dependent runtime, independent antithetic pair-mean standard errors, quality gate and distinct error layers.

Third-party source images and official pretrained weight binaries are reconstructed from their declared original URLs and exact hashes. All newly produced measurements, labels, predictions, learned checkpoints, decision traces, source code, and relevant execution records are included. Private credentials, infrastructure access files and confidential review correspondence are kept outside the research release.
