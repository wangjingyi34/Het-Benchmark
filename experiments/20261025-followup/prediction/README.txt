Our V4 offline prediction extension (2026-10-25)

We froze the method before this run in protocol.json. This is a retrospective descriptive extension on an already evaluated dataset, not an independent preregistered confirmatory study. We use only our frozen public-reproduction-check five-stack real measurements, with the original six model folds and five stack folds. We add no latency labels, make no remote calls, and edit no prior release.

Our new component is a relative-loss evidence kernel. It uses the same train-only family or semantic evidence anchors and structural distances as our V2 geometric predictor. We replace the geometric mean with the weighted median under weights kernel/latency. That point minimizes weighted relative absolute loss on those training anchors. We choose k, temperature and permitted independent three-probe exponent solely on source validation data. No query test label selects a configuration, distance, anchor, or method.

The complete 16 aggregate rows are in metrics.csv/results.json, including every worsening result. Model-holdout strict/probe rows are identical by design, because same-stack evidence is available, and are not independent replications.

Our heldout-model semantic result improves micro MRE 57.54812→43.80435% and Attention MRE 50.13621→28.40070%. The 12-source-model cluster bootstrap interval for MRE reduction is 1.74004–25.07285 percentage points, descriptive and unadjusted for multiple comparisons. Family retrieval improves 52.78048→44.67314% with reduction interval 2.86695–12.26770 points. Attention comprises only 60 queries from text_encoder_ref and causal_lm_ref, not the original benchmark's Attention corpus.

We retain the loss tradeoff: semantic model-holdout WAPE 66.30749→67.46020%, MAE 81.734→83.155 microseconds, and family-macro MRE remains 102.08553%. Thus lower MRE does not establish lower absolute error or arbitrary-task accuracy.

Hardware transfer remains unresolved. Semantic strict MRE 100.01638→77.77458% and 3 probe 126.18966→73.95878%, but both five-stack cluster intervals cross zero. Family 3 probe worsens 90.24087→110.53235%. New stacks are Apple M5 CPU/MPS, A100 MIG 2g.20gb, cloud x 86 CPU and Ascend 910B3; this is not original-five-platform LOPO or a full A100 claim.

We also froze an offline selective rule: trigger if anchor log spread exceeds the 90 th percentile on source validation. On model-holdout semantic strict it triggers 2,439/13,905 (17.54%), leaving MRE 39.99252%;24/60 Attention queries trigger and retained Attention MRE is 22.53309%. The triggered records contain 93.27336 s of already recorded timed blocks, excluding warmup/setup. In stack holdout it triggers only 515/13,905 (3.70%) and no Attention queries, with retained MRE 77.11211%. This documents selective risk and the failure to preserve a source 10% budget under shift. It is not a fresh measurement trial and gives no deployed hybrid-latency benefit.

Our original checkpoint audit finds an archived 64-dimensional input layer, while our new structural whitelists are 68/71 dimensional. The archived example training entry constructs random node features and random targets; the checkpoint contains no feature encoder/scaler, node map, measured-label record IDs or provenance binding. This does not prove every historical paper result used that example. We cannot meaningfully evaluate the old checkpoint on new physical queries without its semantic input/target contract. We do not certify 14.3% or substitute a retrieval result for RGAT.

Audit: all 222,480 test predictions were independently reconstructed from saved train-anchor IDs using scalar arithmetic, maximum relative difference 5.45 e-15. All 137,640 selected validation predictions are saved. All 44 geometric comparator micro-MRE rows reproduce the frozen V2 values within 1 e-9. Each fold stores full train/validation/test IDs and hashes. The main script, helper code, raw data, sidecars, protocol, anchor streams and predictions have hashes.

Run from the rebuttal workspace:
PYTHONPATH=.experiment-deps /Users/wjy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 v4-提分条件补强/prediction/relative_loss_kernel.py --freeze-only
PYTHONPATH=.experiment-deps /Users/wjy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 v4-提分条件补强/prediction/relative_loss_kernel.py
PYTHONPATH=.experiment-deps /Users/wjy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 v4-提分条件补强/prediction/audit_relative_loss_results.py
The completed directory refuses overwrite. To reproduce, copy code into a fresh experiment folder at the same depth and keep the exact source tree. Prior releases remain read-only.
