# Fixed-evidence statistical and cost rechecks

These calculations use the frozen measurements from [`../20261025`](../20261025). They execute no new hardware workloads. The original measured release and its SHA256 manifest remain immutable.

Restore that release first, then run from the repository root with Python 3.12:

```sh
python experiments/20261025/restore_archives.py --extract
python experiments/20261025-analysis/recheck_frozen_evidence.py --release-dir experiments/20261025 --data-root experiments/20261025/run-data --output-dir experiments/20261025-analysis/results
python experiments/20261025-analysis/recompute_official_market_costs.py --baseline-summary experiments/20261025/results/market-cost-scenarios.json --output-dir experiments/20261025-analysis/results
```

The first recheck verifies the exact frozen task, policy-result and raw-artifact hashes and timing means. It retains all eight A100/Ascend model/batch contexts, raw effects, all fixed-policy controls and paired contrasts. The random comparator is the exact mean over eleven measured subsets with at most two actions; it is not eleven independent randomized hardware trials. Five-process t intervals describe conditional variation on a fixed task. They do not establish population-level or familywise superiority.

For A100 ResNet18/B1, runtime-calibrated action prioritization selects mask3 and reduces latency by 2.0972%; latency ranking selects mask5 and reduces it by 1.7269%. Their paired difference is 0.3703 percentage points, with a five-process descriptive 95% t interval of 0.0453 to 0.6953. The uncalibrated DAG order gives 1.1381%, and the same-evidence exhaustive search selects the calibrated mask. All other contexts, including negative differences, are retained.

The relation-control table includes all six relation drops and all matched regressors, with their full five-seed comparisons. The performance-context relation contributes in the reported model-holdout setting (68.1497% full versus 69.8262% without it); removing other relations can improve error. This is not a claim that every relation improves every task.

The second script records 288 explicit cost-allocation scenarios using official public quote snapshots retrieved on 25 October 2026: [Runpod A100 SXM80GB Pods](https://www.runpod.io/pricing), [Lambda A100 SXM80GB in an eight-GPU minimum instance](https://lambda.ai/instances), and [Tianyi PAK3 Ascend910B64GB](https://www.ctyun.cn/document/10029787/10047957). FX and MIG cost-allocation factors are scenario assumptions. Whole-card prices do not become a measured MIG tariff or full-card performance result. These scenarios use the unchanged five evaluation baseline means and retain every outcome. None supports a 66% unit-request cost reduction under its stated allocation. Historical negotiated prices or case-study budgets require their own records.

`SHA256.json` covers the files in this analysis directory. No confidential correspondence or operational credentials are included.
