# HET v2 measured prediction experiment

This is new rebuttal-time evidence. It does not reconstruct the historical v21 labels or authenticate the old headline error.

## Physical data

The frozen `collect_profiles.py` measures 12 executable architectures at three input configurations per physical backend. Six models are official torchvision architectures (`weights=None`); six are explicitly named torch reference networks covering NLP, LLM, audio and image/text fusion. Latency is measured, not generated from a hardware ratio. No task-accuracy claim is made for random-weight models.

Input and parameter generation uses a specified PCG64 integer stream and binary-exact FP32 conversion. All five measured runtime stacks have matching input/weights hashes at all36 graph configurations. The identities, software and exact visible accelerator partition remain in each raw bundle. A100 MIG 2g.20gb is not a full80GB A100.

Every timed record is an isolated executable FX call with captured inputs. Each of 20 raw samples is a synchronized eight-call block mean, after ten warm-up calls. Full-model timings are recorded separately. An isolated-call sum is not asserted to equal end-to-end model latency. FX dependencies are actual traced dependencies; DenseNet `_DenseLayer` is an explicitly retained composite leaf to avoid replacing data dependencies with an invented chain.

The root agent owns backend scheduling. Do not run CPU/MPS profiling concurrently on the Mac or performance collection concurrently with heavy training on the same host.

## Structural enrichment

`enrich_metadata.py` does not modify raw data. It rebuilds the frozen architecture and portable input, requires every graph's input/weights hash to match, then writes module and shape attributes into a sidecar. The feature whitelist was fixed before predictor test scores were produced. Attributes include convolution kernel/stride/padding/dilation/groups, linear dimensions, normalization epsilon, attention heads/head dimension/causality and layout. Compound DenseLayer children are recorded explicitly.

Production evaluation requires hash-matched sidecars. `--allow-unenriched-smoke` is a development escape hatch and must not be used for final experimental tables.

Example, using the bundled Python and the parent `.experiment-deps` via `PYTHONPATH`:

```
python enrich_metadata.py --input ../evidence/physical_profiles/m5cpu.json --output ../evidence/physical_profiles/m5cpu.metadata.json
```

Repeat for `m5mps`, `a100mig` and `xeon_cpu`. Four successful runs are recorded in `metadata-enrichment-execution.json`; elapsed time was 2.9–4.4 seconds each and peak RSS below 500MB. All 144 graph hashes matched.

## Predictor protocols

- Model holdout: six fixed folds, two test and two validation models per fold, eight training models; every model is tested exactly once. No held-out graph participates in training or normalizer fitting.
- Device holdout: one physical device/stack is omitted from all train and validation labels. Ten model architectures train on other devices; two validate there; all available target-device models are tested. This evaluates unseen hardware with partly seen architectures, not simultaneous architecture-and-hardware holdout.
- `strict` features use observed non-latency descriptors. `probe` additionally uses three independent physical microprobes. The latter must be called three-probe hardware-calibrated transfer, not zero-calibration zero-shot.
- Ridge, two-layer width128 MLP, two-layer width80 vanilla GCN, two-layer width32/four-head RGAT and six relation drops use the same record IDs, features, partitions and fixed training/validation policy. Parameter counts are reported. GCN uses untyped symmetric graph propagation and has approximately the same parameter capacity as RGAT; it is a graph latency baseline, not a claimed reproduction of the full BRP-NAS search algorithm. Ridge is deterministic; repeating seeds does not create independent ridge runs.
- All targets are actual positive measured means. No target latency enters any node feature or message. The `performance_context` relation is an unlabelled query-to-hardware edge. Other relations are membership, family taxonomy, actual FX dependency, label-free within-graph similarity and adapter-family context. They are not presented as recovered canonical v21 performance-label edges.
- Five seeds, maximum80 epochs and validation patience15 are the executed main settings. Test scores do not choose hyperparameters or the best seed. The small five-epoch pilot only verifies code and resource use.

The RGAT computes six per-node relation transforms then gathers edge messages. This avoids expanding a full weight matrix per edge. The optimized and reference expanded implementations have matched double-precision forward and parameter/input gradients on a fixed fixture. The graph computation is also invariant to node-index relabelling and does not read its `y` field.

## Commands

Run all experiments after performance collection finishes:

```
python evaluate_profiles.py 
  --inputs ../evidence/physical_profiles/m5cpu.json ../evidence/physical_profiles/m5mps.json ../evidence/physical_profiles/a100mig.json ../evidence/physical_profiles/xeon_cpu.json ../evidence/physical_profiles/npub3.json 
  --metadata-sidecars ../evidence/physical_profiles/m5cpu.metadata.json ../evidence/physical_profiles/m5mps.metadata.json ../evidence/physical_profiles/a100mig.metadata.json ../evidence/physical_profiles/xeon_cpu.metadata.json ../evidence/physical_profiles/npub3.metadata.json 
  --output main-evaluation --protocol both --folds 6 --seeds 11 22 33 44 55 --epochs 80 --patience 15 --feature-modes strict probe
```

The command is shown over readable lines; use a shell continuation or one line when executing. Explicitly name raw inputs rather than globbing a directory containing metadata files. `--train-device cuda` can be used on a controlled GPU research job. Do not run it concurrently with physical performance measurement on the same host.

The first real-data smoke used all four bundles, one model fold, one seed, five epochs and ridge/MLP/RGAT. Six fits completed in3.56 seconds with510MB peak RSS. The saved scores are functional pilot results, not the final multi-seed paper results and not a basis for selecting a favorable test configuration.

## Output and uncertainty

`summary.json` stores splits, raw-source hashes, sidecar hashes, seed metrics, model-level bootstrap intervals, parameter counts, validation decisions and coverage. `predictions.jsonl.gz` streams every record/seed prediction; it is compressed to avoid a large in-memory JSON copy. `ensemble-predictions.jsonl.gz` separately records seed ensembles. Checkpoint files include training/validation/test IDs, train-fitted normalizers and source hashes.

Metrics explicitly separate instance micro-MRE, model macro-MRE, equal-family macro-MRE, each family MRE/count, each runtime-stack MRE/count and Attention MRE/count. An absent Attention class has `null` MRE, not zero. Bootstrap resamples model clusters, not operators or within-block timing repetitions; small model counts still limit inference. RGAT-minus-comparator paired seed differences use identical test-record supports. A 1e-6ms denominator floor is declared consistently for all methods.

Fallback uses ensemble relative dispersion and a threshold selected only on validation under a20% validation measurement budget. Test trigger rate may differ under distribution shift and is reported, not forced to equal20%. Pure predictor, unmeasured selective risk and hybrid replay metrics are separate. Cost is the recorded synchronized timed-block duration of selected measurements; it excludes warm-up/setup and is not an online profiling-service SLA. Trigger record IDs are exported for subsequent fresh online replay if required.

## Planner embeddings

`export_action_embeddings.py` loads a frozen full RGAT checkpoint whose **test** model set contains the requested ResNet models. Validation models cannot be mislabeled as unseen. It constructs an unlabelled FX graph at actual intervention shape224 and batches1/4. In-memory target labels are NaN, and training would reject such a graph. It exports nonzero32-dimensional means for the requested Conv/BN module pairs, with checkpoint, train/validation/test IDs, exact input shape, dtype, hardware and no-target-label provenance.

```
python export_action_embeddings.py --checkpoint main-evaluation/probe-model-fold0-rgat-seed11.pt --hardware-from ../evidence/physical_profiles/a100mig.json --output action-embeddings-a100.json
```

The publisher of the intervention task must map predictor and task device IDs with explicit runtime evidence. Never attach an embedding by guessing a chip name, using a different MIG partition or silently choosing the nearest input shape.

`pilot-224-action-embeddings.json` verifies the interface with16 actual structural inference exports. It is a functionality artifact based on the pilot checkpoint, not a claim that final predictor quality has been reached.

## Executed checks

`pipeline-logic-test-results.json` records14 passed logic checks, including source/target label separation, gradient finiteness, graph permutation equivariance, model/device split separation, validation-only fallback triggering, equivalent memory-efficient attention and GCN invariants. These fixtures are explicitly synthetic and do not add physical performance samples.

The four-stack, five-seed RGAT model-fold0 policy was completed on the cloud GPU and preserved under `npu-policy-preserved/`. It is an independent deployment policy, not the final all-method comparison. Its validation-only threshold is0.7156752616167071; ResNet18/50 are absent from both training and validation. `offline-policy-audit.json` records12 read-only provenance checks passed, including every checkpoint, source/sidecar hash and model partition. It does not measure new performance.

The fifth physical runtime stack on Ascend910B3 has now completed collection with the matching torch/torch_npu2.6.0 environment. `physical-five-stack-audit.json` independently verifies all180 graphs,13905 measured operator queries and278100 raw timing blocks, with matching portable input/weights/logical shapes/FX/query IDs across all five stacks and hash-bound structural sidecars. NPU backend storage-format codes are retained as backend-specific metadata rather than required to equal CPU layout codes. The executed NPU collector adapter hash5c8cdf29 is retained in its execution receipt; optional-storage-query-tolerant c4ad1ecc is a separately frozen successor and is not mislabeled as the executed source. Both use the unchanged core collector7050d276. Software-version differences remain visible.

The completed five-stack main matrix has1100 fits: (6 model folds +5 device folds) ×10 methods ×5 seeds ×2 feature modes. `formal-five-stack-independent-audit.json` verifies all220 result entries,1560 metric/cluster-bootstrap objects,144 paired-seed intervals, physical/sidecar hashes and identical test-record supports. `all1285-checkpoint-archive-audit.json` additionally checks actual archived bytes for all1100 main checkpoints,180 learning-curve checkpoints and5 deployment-policy checkpoints. Every digest matches. All unfavorable results are retained.

## Learning curves

The frozen evaluator supports model-level training budgets2/4/6, with model-fold boundaries unchanged. Training subsets use fixed workload round-robin selection within the training models; validation or test scores never choose subsets. Full8-model results are reused only after exact manifest, seed, training configuration and evaluator-hash checks. Example after a completed compatible main run:

```
python evaluate_profiles.py --inputs RAW_FILES --metadata-sidecars SIDECARS --output learning-curves --protocol model --feature-modes probe --methods mlp rgat --seeds 11 22 33 44 55 --epochs 80 --patience 15 --learning-budgets 2 4 6 --full-budget-reference main-evaluation/summary.json
```

The180 additional fits are complete. `learning-curves-independent-audit.json` verifies all36 entries,252 metric/cluster-bootstrap objects and exact reuse of full8 results. Learning-curve protocol names include their budget to prevent accidental aggregation with the main full8 result. Probe RGAT micro-MRE at model budgets2/4/6/8 is88.5249/78.4155/70.3023/68.1497%; the respective MLP values are89.0432/68.9708/68.5931/67.3343%. These are fixed12-architecture-fold curves, not proof of unrestricted population adequacy.

## Actual active fallback

`active_fallback.py` accepts a frozen validation policy plus a hardware-only descriptor. It constructs unlabelled target graphs and performs frozen predictor inference on CPU. It writes and hashes all decisions before any selected-query timing. It then rebuilds selected graphs, verifies input/weight hashes and physically executes only frozen selected queries. Query-specific raw samples, warmup-inclusive wall cost and graph-setup wall cost are recorded. It does not read a target benchmark label file during decision making. A lack of triggers remains a valid reported outcome; the threshold must not be changed to force an effect.

```
python active_fallback.py policy --summary npu-policy-4source/summary.json --checkpoint-dir /actual/deployment/checkpoint/path --output npu-policy.json
python probe_hardware_only.py --device npu --device-id Ascend-910B3-64GB --output npu-hardware.json
python active_fallback.py run --policy npu-policy.json --hardware-only npu-hardware.json --device npu --output npu-active.json
```

The active code hash9af59a5b registers torch_npu before creating a `torch.device('npu')`. Preserve the source-freeze manifest and runtime package receipt alongside execution; runtime checkpoint hashes are verified, while imported source dependency hashes are audited through the package manifest.

Only after active execution completes, `summarize_active_fallback.py` can join an independent complete physical profile. It requires identical query support, input/weights and policy/decision hashes. It reports pure prediction, unmeasured selective risk, oracle-label replay and fresh-measurement hybrid separately. Fresh measurements are compared to the independent reference rather than to themselves, so real timing drift is retained. Failed selected calls retain their predictor outputs and remain explicit. Eleven synthetic arithmetic/provenance checks in `active-postprocess-unit-test-results.json` pass; these contain zero physical observations.

```
python summarize_active_fallback.py --policy npu-policy.json --decisions npu-active.decisions.json --active npu-active.json --reference-profile ../evidence/physical_profiles/npub3.json --output npu-active-posthoc.json
```

Actual NPU deployment is complete:732 decisions,83 triggered fresh measurements,83 successes,0 failures. `npu-active-posthoc.json` separates pure micro-MRE50.9954%, unmeasured selective micro-MRE51.9036%, oracle replay46.0183% and actual fresh hybrid47.4705%. All83 selected queries are Activation; selective risk is not claimed to decrease. Recorded phase wall costs are11.6563s for decisions and7.3740s for measurement including setup. `npu-active-recomputed-prediction-audit.json` reproduces all732 triggers from the local five source checkpoint files without changing deployed policy bytes.

## Train-only physical evidence retrieval enhancement

`evaluate_evidence_memory.py` is a separately frozen enhancement, not a rewrite of RGAT or historical figures. It retrieves actual training measurements using normalized operation identity, tensor descriptors and module attributes. It includes a family-kernel control and a semantic-information ridge control. No query latency enters signature, distance or prediction. Model holdout excludes query models; device-holdout validation also excludes each source validation query's runtime stack from anchors, imitating the unseen-device condition. k, kernel temperature, probe exponent and ridge lambda are chosen only on existing validation labels from frozen grids.

```
python evaluate_evidence_memory.py --reference-summary formal-five-stack-summary.json --profiles-dir ../evidence/physical_profiles --output NEW_EVIDENCE_DIRECTORY
```

`evidence-memory-frozen-main/` contains all66 entries and predictions from11folds ×2 information modes ×3 methods. `evidence-memory-independent-audit.json` verifies every support/grid argmin and exports484 fixed lexical-first query examples with the actual training record IDs, physical latencies, kernel weights and probe ratios. Eight synthetic leakage/identity checks are recorded separately in `evidence-memory-logic-tests.json` and add no physical observations.

The three-probe family-kernel control reaches model-held-out micro-MRE52.7805% and device-held-out micro-MRE90.2409%, compared with frozen RGAT seed means68.1497% and171.0263%. Semantic exact-type retrieval is not uniformly better:57.5481% and126.1897%. Semantic ridge is76.5926% and99.7330%. All three methods and all original regressors remain reported; deterministic retrieval is not presented as five independent seeded runs.

## Seventeen-workload auxiliary calibration control

`../methods/family_calibration_probes.py` and its frozen manifest measure14 real standard family workloads in addition to the original3. All70 new probes across five runtime stacks succeeded and have matched portable input/weight hashes. The fixed-parameter adapter `evaluate_family17_extension.py` preserves every original k/temperature/gamma/lambda, and uses no new validation-grid or test-family choice. Semantic ridge receives the same17 hardware descriptors and refits coefficients only on the unchanged training fold. If any stack fails one family, that entire family uses the old3 mapping on all five stacks; workloads are never mixed in a ratio.

`family17-fixed-extension/` contains33 entries and predictions on the unchanged13905-query support. Device-held-out micro-MRE is96.1013% for family kernel,203.9640% for exact semantic retrieval and90.8398% for same-information semantic ridge. Compared with the retained3-probe controls, richer calibration is not uniformly beneficial. `family17-independent-audit.json` checks70 probe byte/timing records, all33 fixed parameter configurations, five cost records and all six per-record aggregate errors.

Seventeen counts workload types, not function calls: successful collection uses2720 timed calls,170 warmups and14 new output checks per stack. Incremental14-workload wall costs are0.0951s(M5CPU),2.1851s(MPS),0.4515s(A100MIG),0.0665s(cloudCPU) and11.2339s(NPU). Original3 warmup/setup wall is not retroactively invented.

## Absolute error, ranking and regret

`derive_absolute_and_ranking_metrics.py` reads retained per-record prediction streams and never changes parameters, query support or MRE. It supplements MRE with MAE(ms/µs), WAPE, within-graph latency-gap-weighted pair ordering, oracle top10% latency capture and same-operator cross-stack device-choice regret. Operator sums are isolated-call diagnostics; device-holdout device-choice results stitch crossvalidated predictors and are not a single deployed migration policy.

`formal-and-memory-absolute-ranking/` checks3504060 prediction rows in252 groups, each with all13905 queries. `family17-absolute-ranking/` independently checks83430 more rows in six groups. No View, identity or small-latency query is removed. For example, model-held-out probe RGAT seed-mean MAE is87.169µs and WAPE70.717%, while family-kernel MAE is83.794µs and WAPE67.979%; their within-graph weighted ordering accuracies are91.953% and92.253%. Relative error, absolute error and ranking are different reported endpoints, and no endpoint replaces an unfavorable one.

## Completed independent MHA OOD challenge

The source-frozen `attention_ood_challenge.py` challenge has completed12 physical cases on each of five stacks. All12 input/weight hashes match across stacks. The same15 main strict MLP/GCN/RGAT checkpoints were frozen before collection. All five `predict` invocations completed, yielding900 predictions over60 physical cases with0 prediction or physical-case failures. Raw labels enter only the post-prediction error calculation.

`attention-frozen-results/` retains all per-stack predictions. `summary-independently-verified.json` and its CSV are reproducible using `summarize_attention_frozen_results.py`, which verifies every checkpoint/profile/freeze hash and independently recomputes timing means and errors. Pooled MRE seed-mean±SD is75.9951±3.8988%(MLP),98.7048±0.6831%(GCN),92.8563±3.5004%(RGAT). These high OOD errors are retained. The target is a complete MHA module including QKV/output projections, larger shapes and a short graph, whereas the main Attention targets are primitive SDPA. The two tasks are not pooled into one accuracy claim.
