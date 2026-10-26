Hardware-adaptive operator experiment
Official neural components: https://github.com/HayeonLee/HELP at commit45f70eb92d0d735970a33b608d1544ad3c52f285. Original MIT license retained.
This release preserves frozen protocol.json, all150 HELP and150 paired pooled-MLP fits, raw predictions, weights and independent audits.
The domain is contextual operator latency, not the original NAS benchmark. Read protocol.json and post-run-method-disclosures.json together.
Three independent hardware probes;30 joint source-model/runtime-stack folds;five optimization seeds;0/3probe outputs and inference diagnostics retained.
Fixed-budget full adaptation MRE2590.98+/-3031.57%; paired pooled-MLP finetuning200.79+/-36.79%. These numbers do not establish published HELP inferiority or HET superiority. Within-stack ranking does not improve despite the pooled statistic.
The original directory layout is retained to make frozen source paths resolvable. Python3 with torch,numpy,scipy is required; exact run environment is in records.
From this directory:
python audit_help_independent.py
python summarize_help_results.py
python audit_cluster_and_source_embedding.py
For training command options: python run_help_operator.py --help
Do not rerun the protocol-freeze step over the archived frozen files. Copy the tree for new experiments.
No new NPU measurements were used in this training-only experiment.
