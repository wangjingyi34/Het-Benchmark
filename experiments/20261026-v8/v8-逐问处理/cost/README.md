# Reproducible heterogeneous-inference costing

This scientific export candidate contains the calculator, 40 physical-process records, required inputs, 13 dated official-page snapshots and reproducible tables. Reviewer identities, response mappings, author replies and private review reports are excluded. It has not been uploaded or submitted.

## Reproduce

Run python3 recompute_costs.py from this directory. Python's standard library is sufficient; no cloud connection, credentials, paid order or network access is needed. Both manifests are verified before regenerating eight CSV tables, results-summary.json and verification.json. All tables match under relative and absolute numeric tolerances of 1e-12. Seven CSV tables are byte-identical; one standard deviation differs by 4.44e-16 in timing-summary.csv under this Python runtime. The same last-place difference propagates to results-summary.json and output hashes. Source arithmetic and raw bytes are unchanged. These physical measurements predate this costing replay; replay is not a new hardware experiment.

## Prices, orders and billing

Quotes dated 26 October 2026: Runpod A100 SXM 80GB USD 1.79/card-hour; Lambda A100 SXM 80GB USD 2.79/GPU-hour with an eight-GPU minimum instance (USD 22.32/node-hour); Tianyi pak3.4xlarge.8, one Ascend 910B 64GB, CNY 38.45/hour or CNY 18,454/month. Separately labelled user-supplied CNY 38/hour and CNY 18,500/month are not invoices. The frozen Bank of China USD selling reference is CNY 6.6855/USD at 13:48:32; later intraday rates can differ.

Runpod consumption is billed by second, but deployment requires account credits sufficient for at least one configuration-hour. The one-card rate corresponds to CNY 11.967045 equivalent credits at this FX, before additional storage/tax. This balance prerequisite differs from the short-run consumption-charge lower bound in fallback-costs.csv; initial top-up requirements are not estimated. Lambda on-demand consumption uses one-minute increments. Tianyi's minimum time quantum was not established, so one-second and one-hour assumptions are separate.

A 720-hour A100 subtotal is a calculation, not a verified monthly contract. NPU monthly purchases retain the full month fee even at low duty cycle. Lambda's eight-card minimum applies to that listed 80GB SKU, not all A100 suppliers. No capacity or region-matched order was placed.

## Interpretation

Active cost/image equals allocated hourly rate times measured batch latency in seconds divided by 3600 and batch size. Timings are from A100 MIG 2g.20GB and 910B; whole-card pricing does not turn a MIG measurement into a full-card benchmark. The 1/4-memory and 2/7-SM allocations are accounting sensitivities, not advertised MIG tariffs. One/three-slice packing is a scenario; simultaneous throughput and cloud MIG permissions were not tested.

Monthly service procurement additionally models equal request demand, minimum order size, utilization, billing duration and complete-batch delay. Its periodic-arrival response calculations, including modelled p95 columns, are not measured production SLAs. Regions, tax bases, startup, network, engineering and interference are not fully normalized. These are compute subtotals, not paid bills or TCO.

The low-demand comparison of one 910B with Lambda's eight-card minimum can show procurement savings. The corresponding one-card Runpod comparison is retained and favors A100. All 768 allocation and 13,824 service scenarios, including infeasible/unfavorable ones, are retained. A 66% active-cost reduction requires p_N*t_N<=0.34*alpha*p_A*t_A; the current 768 allocation scenarios do not satisfy it. Archived 500/170 values are budget arithmetic, not authenticated historical equal-service measurements.

Migration mask 6 and its non-regression gate are replayed offline on recorded evaluation timings. Gate decisions use those same timings; the gated 4.397 ms is not an independently evaluated deployed-policy result, and gate decision/startup overhead is excluded. Five independent processes and 100 unique labelled images define timing/quality evidence. Intervals describe timing variation, not price, tax, FX or service-tail uncertainty.

## Provenance and privacy transformation

input-snapshot-manifest.json retains measured-file hashes and package-relative paths. sources/fetch-manifest.json retains quote URLs, timestamps and content hashes while excluding response cookies/request identifiers. public-export-adjustments.json records original/public digests and nonnumerical exclusions/path changes. Original private sources are unchanged. Raw logs and official HTML bytes are preserved; no input labels, timing samples, arithmetic or prices changed. The one last-place floating-point output difference is recorded above.

Additional provenance hashes in raw records may refer to upstream artifacts outside this minimal costing package. Those artifacts are not needed to execute this calculator. This package does not claim to reconstruct the entire historical framework.

## Official sources

- [Runpod prices](https://www.runpod.io/pricing)
- [Runpod billing and minimum credit balance](https://docs.runpod.io/pods/pricing)
- [Lambda prices](https://lambda.ai/instances)
- [Lambda billing](https://docs.lambda.ai/public-cloud/billing/)
- [Tianyi prices](https://www.ctyun.cn/document/10029787/10047957)
- [Bank of China reference FX](https://www.boc.cn/sourcedb/whpj/)
- [NVIDIA MIG profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
