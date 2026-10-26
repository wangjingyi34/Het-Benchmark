HET conditional cost and finite-action decision analysis — 2026-10-26

Run with Python 3 (standard library only):
  python3 experiments/20261026-v9-theory/calculate_conditions.py

Requires the sibling 20261026-v8 release. Input hashes and the prior measurement
commit are bound in results.json. No newly acquired device measurements are
claimed here. Prices are frozen at the V8 source snapshot, not live quotations.

The measured A100 is MIG 2g.20GB. Applying a whole-card tariff (alpha=1) is an
allocation scenario, not a whole-card latency measurement or a verified cloud
MIG offer. NPU/A100 effective price ceilings for 66% active-compute savings
are derived from the four same-batch timing pairs. Current rates exceed them.

The eight-card Lambda vs one-card NPU example describes a low-load procurement
constraint, not algorithmic efficiency. Equal delivered demand, suitable
software deployment and service feasibility are premises, not a measured cloud
SLA. The single-card Runpod counterexample is retained. Taxes, storage, network,
startup and engineering are outside the quoted compute subtotal.

No confidential reviewer text, author response, credential, private host, or
reviewer identity is included in this directory.
