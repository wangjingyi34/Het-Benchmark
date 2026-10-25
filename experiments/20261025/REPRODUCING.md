# Reproducing the measured validation release

Use Python 3.12 for the portable verification and analysis commands. The actual measurement environments were PyTorch 2.9.1 on Apple M5 CPU/MPS and on the Xeon/A100 MIG host, and PyTorch 2.6.0 plus TorchNPU 2.6.0, torchvision 0.21.0 and CANN 8.2.RC1 on the rented Ascend 910B3. Exact versions, the CUDA image digest, source hashes, commands, seeds and precision receipts accompany the records.

## Recover and verify all data

From this directory:

```sh
python restore_archives.py
python restore_archives.py --extract
```

The first command verifies every listed release file. The second reconstructs the complete predictor archive from eight ordinary Git-sized pieces, verifies its full SHA256, restores all raw measurements, predictions and checkpoints, and creates an explicit SHA256-checked mapping from original raw paths to restored raw files. The unchanged original absolute paths remain provenance; they are never silently rewritten.

Run the portable raw-to-task audit without rerunning hardware:

```sh
python run.py audit-traceability -- --raw-dir run-data/evidence/fullmodel_raw --task-dir run-data/evidence/final-tasks --output run-data/reproduced-traceability-audit.json
```

## Predictors and data splits

`code/prediction/evaluate_profiles.py` is the frozen primary predictor experiment. It supplies ridge, MLP, capacity-matched generic GCN, RGAT and six relation-drop controls on identical records/features and partitions. It is an adaptation for this operator benchmark; the generic GCN is not advertised as a complete BRP-NAS or HELP reproduction.

The primary experiment has six model folds, five device-stack folds, five seeds, strict hardware descriptors and independently measured three-probe descriptors. Three-probe mode is hardware-calibrated transfer. The separate memory experiment uses only training measurements as anchors and chooses its complete fixed parameter grid using validation data. Device-holdout validation temporarily excludes the query's source stack from anchors. The 17-query extension retains the selected parameters and uses fourteen additional family references. These are separate information budgets and all outcomes are retained.

Use `python run.py predict -- --help`, `python run.py evidence-memory -- --help` and `python run.py family17 -- --help` for exact inputs. Main bundles are under `run-data/evidence/physical_profiles`; metadata sidecars are supplied. Explicitly list the five raw JSON files and their five matching sidecars rather than globbing both types together. Main predictor checkpoints and all per-record predictions are under `run-data/prediction/formal-five-stack`. Training-size curves and the original deployment-policy checkpoints are preserved in adjacent directories.

## Full-model diagnosis and planning

The executed action set is four explicit ConvBN folds. Each of four execution backends has two pretrained ResNet architectures and two batch sizes, with three calibration processes and five held-out processes per context. Every run retains all sixteen coalitions, thirty synchronized timings, actual per-image labels and predictions, input/weight hashes and the true FX dependency DAG. Quality is retained ImageNet1000 top-1 against the paired baseline on a fixed 100-image evaluation subset, with threshold 0.99; it is not absolute 99% accuracy.

The PPO study uses eight A100/Ascend contexts, four configurations, five seeds and 4,096 steps: 160 actual policy fits. Checkpoints, fixed masks and all raw and accepted effects are archived. Additional CPU/MPS planner inputs were prepared but are not represented as executed PPO fits. The separate Ascend confirmation uses twenty fresh processes, eleven budget-feasible masks, adjacent balanced AB/BA timing and an independent baseline-versus-baseline control. Its outcomes never reselect the original plans.

The exact eight PPO inputs with nonzero graph embeddings are restored under `run-data/methods/planner-matrix-inputs/tasks`. Their index, two formal embedding exports, exact device-mapping receipts and frozen model checkpoint are preserved in `planner-frozen-embedding-inputs.zip`. These are the actual policy-training inputs; the separately restored full-model `final-tasks` are the immutable physical measurement source.

Use `python run.py migration -- --help` with the restored `explicit-raw-path-map.json` when reading immutable source artifacts. Exact measured coalition Shapley, same-surrogate sampling fidelity, held-out physical surrogate error, and measured plan utility are distinct reported quantities. The source-target mixed-profile game is a diagnostic surrogate; executed target actions provide the physical decision audit.

## Input reconstruction and new hardware runs

The quality-data manifest supplies exact original sample IDs, class labels, image and tensor hashes, split membership and preprocessing. `code/interventions/prepare_assets.py` reconstructs the selected images from the declared official Imagenette archive and class mapping. The model definitions fetch the pinned official ImageNet V1 weights; their SHA256 and state hashes are supplied. Third-party source photographs and vendor pretrained binaries are obtained from their original declared URLs rather than redistributed as new experimental outputs.

`python run.py profile -- --help`, `python run.py full-model -- --help`, `python run.py paired-confirmation -- --help` and `python run.py family-probes -- --help` expose the executed protocols. On Ascend, source the installed CANN environment before running. The entry point routes NPU commands through the exact FP32 launcher, disables and reads back Conv/MatMul HF32, and records script/precision receipts. Allocate only the intended device and serialize physical timing against other training or profiling on the same host.

All failed capability probes and setup/printing errors remain identifiable in the run records. The first completed paired record used source `2f33287`; the subsequent source `e88bce1` changes only a final stdout printing call after persistent data storage. Both source identities and a byte-exact one-line verification are retained. Its physical measurements were not discarded or rerun.

## Cost scenarios

`results/market-cost-scenarios.json` and `.csv` enumerate all 216 fixed-input price/allocation scenarios, including unfavorable savings. They use the unchanged means of the five baseline evaluation processes per context, sample-request units and explicit batch sizes. Full-card public A100 prices are scenario references for an explicitly allocated MIG opportunity cost, not a quoted MIG tariff or an inference of full-card latency. The owned A100 has no observed rental invoice. Scenario exchange rates and monthly utilization assumptions are recorded as assumptions.

Recompute the complete cost matrix from restored task inputs:

```sh
python code/methods/calculate_market_cost_scenarios.py --task-dir run-data/methods/planner-matrix-inputs/tasks --output-prefix run-data/reproduced-market-cost-scenarios
```
