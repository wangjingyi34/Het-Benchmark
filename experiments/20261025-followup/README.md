# HET follow-up measured deployment and prediction studies

This additive release preserves the earlier `20261025` and `20261025-analysis` releases. It provides additional measured deployment actions, a relative-loss retrieval component, full relation-ablation cluster statistics, explicit table aggregation/provenance, and a service-cost envelope. It does not retroactively turn specification-calibrated historical records into physical measurements or authenticate missing historical case-study logs.

## Scientific contents

- `physical/`: 32 fresh A100 SXM4 MIG 2g.20GB processes, comprising 12 calibration blocks and 20 evaluation blocks. Plans were frozen before evaluation. All four fold/layout coalitions, input-conversion time, independent baseline copies, 2,560 balanced AB/BA pairs and 5,120 timed model calls are retained in `layout-physical-v4.zip`. The selected all-ConvBN-fold action has held-out mean gains of 5.6274%, 0.3344%, 2.8918% and 1.3197% for ResNet18/50 at batches 1/4. Layout changes are slower and remain in the archive. This is deployment-action validation, not a new COPA superiority result or a reconstruction of the historical FITAS case. Unique quality images remain 200; repeated predictions are not independent images.
- `prediction/`: train-evidence-only relative-L1 weighted-median retrieval, validation-only configuration selection, all 222,480 test and 137,640 selected-validation predictions, frozen folds and anchor records. On the same model-holdout support, semantic retrieval micro MRE changes 57.5481% to 43.8044%, while WAPE changes 66.3075% to 67.4602%. This is a separate retrospective component study, not the original RGAT headline. All 16 method/mode/protocol aggregates remain visible.
- `statistics/`: all six relation drops across strict/probe and model/device panels, exact matched physical labels, 12-model paired-cluster inference, model/seed sensitivity, sufficient statistics and exploratory multiplicity diagnostics. Micro and model-macro estimands are separate. Intervals are descriptive and do not establish arbitrary-model or hardware-population coverage.
- `economics/`: 576 normalized cost cases, 448 idealized periodic-service configurations and 129,024 financial cases based on 40 frozen physical baseline processes. Demand, response budget, idle utilization, billing quantum and minimum order are scenarios, not measured production traffic or invoices. Lambda eight-card reservation savings must not be described as algorithmic savings or validation of historical 500/170 budgets.
- `reconciled-tables/`: explicit common-three-family and eight-family printed-cell aggregates plus legacy source/printed-cell provenance. Table arithmetic does not recover missing per-query labels or original 5,000-edge provenance.
- `case/`: full-context independent recomputation of the earlier raw intervention evidence, retaining all methods and outcomes.

`FOLLOWUP-SHA256.json` is the public-only manifest. Internal review correspondence, author responses, infrastructure credentials and private review-request fields are excluded. A publication note records the non-scientific protocol redaction. The original execution scripts and results retain their source hashes; portable replay explicitly removes two review-only provenance paths from its executed statistical copy and records that adaptation.

## Verify and replay without accelerators

Use Python 3.12 with NumPy and PyTorch installed. The offline studies do not execute GPU/NPU kernels or contact rented machines. First restore the previous public release:

```sh
python experiments/20261025/restore_archives.py --extract
python experiments/20261025-followup/replay.py --frozen-data experiments/20261025/run-data --workspace /tmp/het-followup-replay
```

Choose a new workspace for each replay. The wrapper copies scientific scripts into an isolated workspace, resolves legacy paths to hash-restored published data, preserves train/test identities, reproduces the numeric exports and checks the physical summary. The published expected results remain unchanged. `REPLAY-VERIFICATION.json` records the tested source replay. Creation timestamps may differ; scientific outputs are compared separately from those timestamps.

`--studies layout`, `prediction`, `statistics` or `economics` permits a bounded replay. No confidential review document, SSH host, password or cloud order is required. To collect new hardware timings rather than replay archived measurements, provision the declared software/accelerator and reconstruct official pretrained weights and fixed image assets from the earlier public release's source manifest. Such new timings are new experiments and are not guaranteed to equal the archived means.
