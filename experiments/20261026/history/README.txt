Historical source audit and original-five-platform LOPO data contract

This is a portable numerical/provenance audit of public source artifacts. It
does not create physical benchmark observations or certify historical runs.
Source files are pinned to the public commit in source-manifest.json. One
historical output-directory fallback is made portable, with both original and
derivative hashes retained. The source-data scripts are evidence inputs;
recompute_history_findings.py parses them and never launches their workloads.

Run with Python 3.10 or later, using only the standard library:
  python3 verify_package.py --recompute
  python3 recompute_history_findings.py
  python3 test_original_lopo_validator.py
  python3 validate_original_lopo.py current-original-data-candidate.json --report current-original-platform-data-gate.json

The last command must exit 2: the honest candidate contains no recovered
admissible original per-query physical labels. This is an expected evidence
gate failure, not a reason to invent records. Unit-test timings are explicitly
synthetic software fixtures and must never enter scientific result tables.

Current supported conclusions:
- Released seed-42 source split: 27 training, 3 validation, 4 test models;
  4,906 / 550 / 788 declared operator instances, respectively.
- The old 50-row prediction CSV exactly follows seeded perturbations of
  hardware-ratio values. It is not an original per-query physical label set.
- All 32 non-A100 values in the audited cross-platform summary exactly follow
  the source hardware-ratio formulas and are explicitly measured=false.
- The five-family equal-weight macro is 14.28% (rounds to 14.3%); this does not
  reconstruct a micro average without original per-query values and weights.
- MobileNetV2's 4.441 ms equals a rounded historical JSON summary. This alone
  does not authenticate the run or establish a shared origin with 4.35 ms.
- Source-level six-operator configurations are recovered, but the original
  run-to-source binding is not established. The printed Linear 4096 label
  differs from Linear(768,3072) in the source.
- A family-level graph and schema counts do not supply the missing original
  5,000 linked performance records. 6,244 x 5 = 31,220 counts opportunities,
  not measured samples.
- Original five-device LOPO excludes estimates, calibrated labels, substituted
  devices and MIG partitions. Each split uses held-out source-model groups and
  target-device label isolation; it cannot proceed before the data gate passes.

Results retain provenance boundaries and numerical sources. The historical
repository license is included as LICENSE.public-sources. This package does
not contain confidential correspondence, private archive inventories,
private machine paths, credentials or local independent-review documents.

Portability: input/artifact SHA256 hashes and CSV text remain byte-exact.
Recomputed JSON structure, list order and nonfloat scalars must match exactly;
finite float values allow relative/absolute tolerance 1e-12 because CPython
3.12 changed built-in floating-point sum. Frozen result bytes are unchanged.
