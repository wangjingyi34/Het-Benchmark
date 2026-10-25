# HET measured experiments 20261025

This directory contains new, physically executed experiments and their reproducible evaluation. The original submitted artifact remains versioned separately.

## Recovered Ascend data

`raw/npu/npu-physical-main-complete.zip` contains 36 actual operator-profile graphs, all 32 full-model coalition runs, the exact deployed active-measurement policy, selected-query measurements, actual quality-rejection and rollback execution, hardware capability outcomes, and launch receipts. `raw/npu/npu-physical-final-confirmation.zip` contains all 20 independently launched confirmation processes, 4,400 adjacent AB/BA timing pairs, the independent 12-configuration complete-MHA challenge, and exact execution sources and precision receipts.

Both files were recovered from the rented Ascend 910B3 64GB instance and checked against the server SHA256 before archiving. Every retained performance outcome remains in the data, including regressions and failed capability probes. The small fixed image-label audit evaluates retention against its own baseline; it is not a full ImageNet accuracy result.

The first paired confirmation source has a final stdout-printing error after the result was persisted. Its raw completed result is retained. The remaining source moves `flush=True` to `print`; the timing, quality, and randomization code are unchanged. Source hashes identify both executions.

## Evaluation status

Matched five-runtime-stack operator data, frozen prediction experiments, complete planning comparisons, source manifests, reproduction commands, and independent numerical audits are being consolidated in this same directory. Performance interpretation uses held-out process blocks, separates sampler error from physical surrogate error, and reports fixed-plan confirmation separately from its calibration data.

Research archives contain timing, predictions, labels, hashes, configuration, and run provenance. Access credentials and third-party source images or pretrained weight files are not part of these archives; their deterministic preparation and download references will accompany the full reproduction package.
