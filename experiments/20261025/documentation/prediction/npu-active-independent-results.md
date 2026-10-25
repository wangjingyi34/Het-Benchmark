# NPU active fallback: independently verified results

The physical Ascend910B3 execution is complete. The exact deployed policy, frozen decisions, fresh measurements, independent reference profile and five source checkpoints pass the independent provenance and numerical checks below. The policy bytes were not edited for local checkpoint relocation.

## Scope and execution

The source-only five-seed RGAT ensemble was trained on four source runtime stacks. ResNet18/50 were absent from both training and validation; NPU target operator labels were also absent. Three independent NPU hardware microprobes supplied calibrated hardware descriptors. The threshold0.7156752616167071 was chosen only on source validation labels under a20% validation measurement budget and remained fixed during NPU deployment.

On the732 ResNet18/50 queries across six input configurations, the frozen policy triggered83 queries (11.3388%);83/83 were successfully remeasured with0 failures. Unmeasured prediction coverage was88.6612%. All83 triggered queries belonged to Activation. No claim is made that this run calibrated every operator family or provided an uncertainty/risk guarantee under hardware shift.

## Accuracy against a separate physical reference

| Evaluation | Queries | Micro MRE (%) | Equal-family macro MRE (%) | Equal-model macro MRE (%) |
|---|---:|---:|---:|---:|
| Pure frozen predictor |732|50.995351|50.819794|49.427770|
| Unmeasured selective predictor |649|51.903584|51.314788|50.084218|
| Oracle substitution replay using existing reference labels |732|46.018342|48.191243|43.954393|
| Actual fresh-measurement hybrid |732|47.470453|48.958159|45.562270|

The actual fresh hybrid reduces micro MRE by3.524898 percentage points, or6.9122% relative to pure prediction, in this fixed experiment. The lower oracle-replay error is not called a newly measured online result. On the83 selected queries, fresh measurements differ from the separate reference by12.806571% MRE; that real timing drift is retained in the fresh-hybrid calculation. Predictor error against those fresh selected-query measurements is38.628557%.

The unmeasured selective subset has51.903584% MRE, which is higher than the complete pure-predictor mean. Thus the observed hybrid improvement does not establish that relative ensemble dispersion ranks all prediction errors correctly. The selected/unselected supports are different and are not presented as a paired accuracy comparison.

## Measured cost

| Recorded stage | Seconds | Included work |
|---|---:|---|
| Decision phase |11.656317|Target shape propagation and frozen CPU predictor inference |
| Measurement phase |7.373988|Selected-graph rebuilding/setup, warmup and selected-query measurements |
| Sum of the two recorded phases |19.030306|Stage sum; excludes unrelated process startup/final I/O |
| Sum of individual selected-query wall costs |1.031161|Warmup-inclusive query timing calls |
| Raw synchronized timed blocks alone |0.852980|Excludes warmup and graph setup |

These are profiling-workflow costs, not application end-to-end inference latency or a demonstrated speedup against an unmeasured full-profiling baseline.

## Independent reproducibility checks

`npu-active-deployed-policy-audit.json` passes12 provenance checks, including all five checkpoint hashes, exact source-summary/raw/sidecar hashes, fixed seeds and disjoint model partitions. Explicit local checkpoint relocation uses `--checkpoint-dir`; deployed policy SHA256 remains `ccae163877b5726a3aa98243fdf9551de0bb9583ef7e1396986725e6f33f341a`.

`npu-active-recomputed-prediction-audit.json` recomputes all732 decisions from the local five checkpoint files, recorded unlabelled structure and hash-bound metadata. Target records were removed and y was NaN during inference. Maximum prediction difference from the NPU-side CPU predictor was3.7253e-08ms and maximum uncertainty difference was4.7413e-07. All732 triggers match exactly, including the same83 selections. This check introduced no training or new physical timing.

Primary independent analysis: `npu-active-posthoc.json`. Its source hashes bind the exact deployed policy, decisions, active results and separate physical reference. The main13905-query study and Attention OOD challenge remain separate experimental artifacts.

## Evidence-grounded response text

We additionally deployed the frozen fallback policy on a physical Ascend910B3 stack using architecture-held-out ResNet18/50 queries. The source-validation threshold selected83 of732 queries (11.34%); all83 fresh measurements succeeded. Against a separately measured reference, actual measurement substitution reduced micro-MRE from50.995% to47.470%, while retaining12.807% fresh/reference drift on selected queries. Oracle-label replay (46.018%) is reported separately. Decisions used three independent hardware probes and no target operator-latency labels. All732 trigger decisions were reproduced from the five frozen checkpoint hashes. Measured decision and measurement-stage wall costs were11.656s and7.374s, including shape propagation and graph setup.
