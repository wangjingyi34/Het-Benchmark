# Data provenance and redistribution boundary

The artifact uses three explicit provenance classes:

1. `measured`: synchronized timing captured on the named physical platform. The released A100 records identify CUDA 11.8, PyTorch 2.1.0+cu118, warm-up count, repeat count, mean, variance, and quantiles.
2. `derived`: canonical operator records, graph relations, attribution scores, aggregates, and paper tables computed from measured or run-indexed inputs.
3. `estimated`: calibrated cross-platform values or metadata-only feasibility records. These are not silently relabeled as physical measurements.

Raw vendor profiler traces and proprietary event identifiers are not redistributable. Derived records retain source-run IDs and the hardware/backend manifest needed to identify their origin. Reprofiling requires access to the stated device and vendor stack.

The paper-canonical graph counts are stored in `data/paper_schema_counts.json`, and every reported table is locked under `results/paper_tables/`. Old developer graph/table exports produced with alternate support, similarity, or experiment budgets are not included because they are not inputs to the reported paper snapshot.

The Llama 3.1, Qwen2.5, and Stable Diffusion 3 extension is metadata-only. It tests parser/schema scale and is excluded from latency, MRE, COPA, corpus, and knowledge-graph counts.
