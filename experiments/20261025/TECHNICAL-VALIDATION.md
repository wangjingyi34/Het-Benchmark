# Technical validation matrix

This matrix identifies the implemented checks and complete public evidence. It records coverage, not a promise that every method wins every test. Historical table arithmetic and newly measured validation are distinct. All policy choices use their recorded calibration or validation partitions.

| ID | Validation target | Evidence |
|---|---|---|
| V01 | Architecture and sample units | Operator measurements; training-size curves |
| V02 | Workload and shape coverage | Five domains, raw shapes/attributes, independent MHA challenge |
| V03 | Transfer regimes and applicability | Model/device-stack splits, strict/3/17-probe budgets, OOD results |
| V04 | Sampling versus physical attribution error | Independent pair means, exact same-game reference, physical 16-coalition tables |
| V05 | Executable substitutions | Pretrained full-model runs, exact action paths, task quality |
| V06 | Fusion and runtime interactions | Low-order runtime calibration, five withheld high-order masks, all eval blocks |
| V07 | Matched prediction baselines | Identical record IDs, features and partitions for ridge/MLP/GCN/RGAT |
| V08 | Held-out architecture and hardware evaluation | Six model folds, five device-stack folds and all seeds |
| V09 | Attention and measurement fallback | Primitive and complete-MHA separation; prospective active-query records |
| V10 | Cost and deployment units | Market-cost assumptions, rate/latency/batch/allocation decomposition |
| V11 | Statistical uncertainty | Model-cluster prediction intervals and physical-process paired effects |
| V12 | Measured cross-stack provenance | Five physical raw bundles, zero ratio-calibrated performance labels in refreshed cohort |
| V13 | One run identity per figure value | Input/weights/backend/shape hashes and actual timing vectors |
| V14 | Performance-record coverage ledger | 2,781 measured queries per stack, explicit status and source linkage |
| V15 | Aggregation definitions | Micro, equal-model macro, equal-family macro and same-support comparisons |
| V16 | Seed and model-cluster variability | All 1,100 primary fits and per-query predictions |
| V17 | Fallback trigger rate and cost | Frozen validation threshold; 732 decisions, 83 real queries and measured wall times |
| V18 | Graph relation ablations | All six drops, same-feature non-graph controls and comparable-capacity GCN |
| V19 | Planner component ablations | 160 actual PPO fits, real/zero embeddings, mask comparison, same-evidence search |
| V20 | Path, sum and runtime definitions | Actual DAG, additive-limit proof, constant/no-runtime/quadratic variants |
| V21 | Publicly reproducible complete-model case | Open pretrained ResNet models, full inputs and public raw intervention tables |
| V22 | Quality threshold meaning | Paired baseline top-1 retention 0.99, true sample labels, actual rejection/rollback |
| V23 | Price versus performance contribution | Hourly/monthly scenarios and allocation assumptions kept distinct |
| V24 | Contribution hierarchy and prior work | Migration-specific diagnostic interface; established sampling/GNN/PPO components attributed |
| V25 | Actual optimization utility | Frozen plans evaluated against all measured candidates and independent confirmation |
| V26 | Ranking and oracle comparisons | Same budget, same actions, latency/random/search controls and regret |
| V27 | Independent validation layers | Numerical, sampler, surrogate, physical and decision errors separated |
| V28 | Complete planner execution trace | Calibrations, checkpoints, selected actions, effects, quality and cost records |
| V29 | Hard feasibility | Measured final quality gate, corrupt-candidate rejection, physical rollback execution |
| V30 | Related mathematical methods | Covert and Lee, Shapley Flow, BRP-NAS, HELP and contemporary predictor context |
| V31 | Absolute error versus decisions | MAE/WAPE, latency-gap weighted ranking, top-operator capture and matched-action regret |
| V32 | Hardware-adaptive information budget | Strict, 3-probe and controlled 17-probe transfer reported separately |
| V33 | End-to-end migration interface | Four 69/175-player source-target traces through executable target actions |
| V34 | Evidence strength per contribution | Dataset, diagnosis, prediction and planning have distinct validation quantities |
| V35 | Abstention and measured-query accounting | Validation-only policy, pure/selective/replay/fresh metrics and actual profiling cost |
| V36 | Dependent antithetic samples | K total permutations, independent pair means, covariance-aware grouped-action SE |
| V37 | Actual dependency graph | ResNet18 71 nodes/78 edges and ResNet50 177/192; residual multi-input nodes retained |
| V38 | Device ranking interpretation | Actual per-query and full-model measurements; no assumed universal rank reversal |
| V39 | Precise operator configuration | Shapes, channels, kernels, heads, dtype, strides, layout and module attributes |
| V40 | Complete status accounting | Measured/failed/unsupported/invalid outcomes and expected-query support kept explicit |

The corresponding complete records are in `archives`, exact mathematical implementations in `code`, and independent consistency reports in `process`. Original operational paths remain provenance; restoration supplies content-hash-checked explicit path maps. Execution success is separate from prediction accuracy or statistical significance.
