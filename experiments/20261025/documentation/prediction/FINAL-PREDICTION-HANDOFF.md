# Completed prediction experiments and audit index

All delegated physical collections, frozen prediction protocols and offline evidence/calibration comparisons in this directory have now executed. No NPU capacity is needed for remaining prediction-side work. Root owns final paper/rebuttal integration and GitHub main publication.

## Immutable main experiment

- Core collector7050d276 and evaluatorbad26 are unchanged.
- Five measured runtime stacks,180 executable graph configurations,13905 operator queries,14 observed families and60 main primitive Attention queries. Inputs/weights, logical tensor descriptors and FX/query IDs match across stacks.
- `formal-five-stack-summary.json`:220 entries,1100 fits,40 aggregates,36 paired comparisons. `formal-five-stack-independent-audit.json` recomputes1560 metric/cluster-bootstrap objects and144 paired-seed intervals and verifies raw/metadata hashes and common test supports.
- `formal-learning-curves-summary.json`:36 entries,180 additional fits. `learning-curves-independent-audit.json` verifies252 metric/cluster-bootstrap objects, train-only workload round-robin subsets and exact full8 result reuse.
- `all1285-checkpoint-archive-audit.json` reads actual zip-entry bytes for1100 main +180 curve +5 deployment checkpoints and matches every saved SHA256. The complete archive was received, not a partial file.
- Seeds are conditional neural fit variability. They are not independent hardware samples. NPU torch/torch_npu2.6.0 differs from the other stacks' torch2.9.1 and remains disclosed.

## Main numerical facts

Probe model-held-out micro-MRE seed means: ridge77.4233%, MLP67.3343%, GCN69.4464%, RGAT68.1497%. RGAT equal-model macro-MRE59.7233% is below MLP62.4076%, but RGAT micro-MRE does not beat MLP. Device-held-out probe RGAT micro-MRE171.0263% is worse than MLP116.8920% and ridge99.5232%; no blanket cross-hardware superiority is claimed.

RGAT probe model-budget2/4/6/8 micro-MRE is88.5249/78.4155/70.3023/68.1497%. MLP values are89.0432/68.9708/68.5931/67.3343%. Every relation-drop result is retained in the full summary and paired CSV; several drops are neutral or improve a metric, so universal relation necessity is not claimed.

## Actual NPU active fallback

- `npu-active-posthoc.json`:732 queries,83 triggered,83 newly measured successes,0 failures. Pure micro-MRE50.9954%, selective51.9036%, oracle replay46.0183%, fresh-measurement hybrid47.4705%. Actual fresh hybrid improves3.5249pp; measured fresh/reference drift is12.8066% on selected queries.
- `npu-active-deployed-policy-audit.json`: exact deployed policy is unmodified; five local checkpoint files are explicitly relocated for auditing.
- `npu-active-recomputed-prediction-audit.json`: all732 decisions recomputed from unlabelled structural inputs; all83 triggers match, maximum prediction difference3.73e-08ms.
- `npu-active-independent-results.md` contains the table, complete cost definitions and response-ready text. All83 selected queries are Activation; lower selective risk or universal family calibration is not claimed.

## Additional train-only performance evidence retrieval

- `evaluate_evidence_memory.py`, sourcee9975072, is a distinct enhancement. It retrieves actual training measurements using fixed structural/semantic rules. Model queries exclude their own model labels; hardware-transfer validation excludes the query stack's anchors. k/temperature/gamma/lambda are selected only on fixed validation data.
- `evidence-memory-frozen-main/`:66 entries across11folds ×2 information arms ×3 methods. All13905 test records are retained. Model-held-out micro-MRE: family kernel52.7805%, semantic exact retrieval57.5481%, same-information semantic ridge76.5926%. Device-held-out3-probe values:90.2409%,126.1897%,99.7330%.
- `evidence-memory-independent-audit.json`: all66 supports/grid argmins verified,78 metric objects recomputed and484 fixed lexical-first query traces reproduce predictions from named training records, physical labels, weights and probe ratios.
- `evidence-memory-logic-tests.json`:8 synthetic logic tests,0 physical observations. `../methods/evidence-memory-peer-review.md` records independent static leakage review.

## Fixed seventeen-workload calibration comparison

- `../methods/family_calibration_probes.py` sourcecdd236c3 and its zero-forward/zero-timing frozen manifestea400f2b define14 independent standard family workloads, added to the retained3 probes. Five stacks each completed14/14 probes; all70 input/weight identities and raw timings are verified.
- `evaluate_family17_extension.py`, sourcea1286aa5, preserves all old k/temperature/gamma/lambda values. Same-information ridge receives all17 descriptors and refits only unchanged-fold training coefficients. If any stack fails a family, the whole family would fall back to the same original3 mapping on all stacks; no mixed-workload ratio is allowed. Actual fallback count is0.
- `family17-fixed-extension/`:33 entries,6 aggregates. Device-held-out micro-MRE: family kernel96.1013%, semantic retrieval203.9640%, semantic ridge90.8398%. Richer calibration improves the ridge control but not both retrievers. Every outcome remains visible.
- `family17-independent-audit.json`:33 fixed parameter/support checks,70 actual probe byte/timing checks,5 runtime cost checks and6 independent aggregate checks passed.
- Query budget17 means17 workload types,2720 timed calls +170 warmups +14 new output checks when successful. New14 wall costs: M5CPU0.0951s, MPS2.1851s, A100MIG0.4515s, cloudCPU0.0665s, NPU11.2339s. Original3 warmup/setup wall is not fabricated.

## Absolute error and operator planning diagnostics

- `derive_absolute_and_ranking_metrics.py` does not reselect parameters or remove small operations. It supplements MRE with MAE(ms/µs), WAPE, latency-gap-weighted within-graph ordering, top10% oracle latency capture and same-query cross-stack device-choice regret.
- `formal-and-memory-absolute-ranking/`:3504060 rows,252 groups; each group uses all13905 queries. `family17-absolute-ranking/`:83430 rows,6 groups, same support.
- Model-held-out probe RGAT seed-mean MAE87.169µs/WAPE70.717%/weighted ordering91.953%; family kernel83.794µs/67.979%/92.253%. Absolute error and ranking do not replace unfavorable MRE. Cross-stack regret is an isolated-operator diagnostic; hardware-LOPO combines crossvalidated predictors and is not a single deployed full-model migration policy.

## Complete frozen MHA module/shape OOD protocol

- `attention_ood_challenge.py`, sourcedfab9126, remains unchanged. The actual selected `attention-freeze-main-strict.json` binds15 main MLP/GCN/RGAT checkpoints before collection.
- Twelve configurations ×5 runtime stacks =60 physical cases. All12 input/weight hashes match across stacks. All five frozen-predictor CLI executions exited0, producing900 rows with0 prediction or physical-case failures.
- `attention-frozen-results/{m5cpu,m5mps,a100mig,cloudcpu,npub3}.json` retains every result. `summary.json` and `summary-independently-verified.json` retain pooled and per-stack scores. The latter is reproducible with `summarize_attention_frozen_results.py`, which independently checks all checkpoints/profile/freeze bindings, physical means and errors.
- Pooled MRE seed-mean±SD: MLP75.9951±3.8988%, GCN98.7048±0.6831%, RGAT92.8563±3.5004%. RGAT MAE13.7416ms/WAPE99.6604%. High OOD errors are retained. Full MHA includes QKV/output projections, larger shapes and a short graph; it is not pooled with primitive SDPA main-query accuracy.

This handoff confirms execution, artifact integrity and stated audit checks. It does not turn an unfavorable empirical score into a passing accuracy claim or certify untested production/task-quality behavior.
