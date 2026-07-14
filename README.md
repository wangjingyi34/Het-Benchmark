# Het-Benchmark

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**White-Box Operator Benchmarking and Counterfactual Attribution for AI Inference Migration**

Het-Benchmark is a comprehensive evaluation framework for assessing AI model migration capabilities across heterogeneous hardware platforms. It provides fine-grained operator-level analysis, cross-platform performance prediction, and knowledge graph-based optimization guidance for zero-shot model deployment.

## Key Features

- **Three-Layer Decoupled Architecture**: Model Layer → Operator Layer → Hardware Layer
- **COPA Algorithm**: Two-stage Contribution-based Operator Performance Attribution using Shapley values
- **MOH-KG**: Model-Operator-Hardware Knowledge Graph with 6,299 nodes and 29,199 edges
- **RGAT**: Relational Graph Attention Network for cross-platform performance prediction
- **KG-A2O**: Knowledge-Graph-guided Adaptive Operator Optimization using PPO
- **Hardware Abstraction Layer (HAL)**: Unified interface for 5 major hardware platforms
- **Comprehensive Dataset**: 34 models, 6,244 operator instances across 6 categories

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wangjingyi34/Het-Benchmark.git
cd Het-Benchmark

# Install dependencies
pip install -r requirements.txt

# Install Git LFS (for large model files)
git lfs install
git lfs pull
```

### Run Experiments

```bash
# Run COPA attribution analysis
python experiments/exp_copa_attribution_full.py

# Run Apple M4 operator and model validation slices
python experiments/exp_apple_m4_operator_slice.py
python experiments/exp_apple_m4_model_slice.py

# Run cross-platform evaluation
python experiments/exp_cross_platform.py

# Run GNN predictor evaluation
python experiments/exp_gnn_by_operator.py

# Run Shapley value calculation
python experiments/exp_shapley_real.py

# Run surrogate model speedup evaluation
python experiments/exp_surrogate_speedup.py

# Run FITAS migration case study
python experiments/exp_fitas_migration.py
```

### Validate and Reproduce the Paper Snapshot

The repository includes a paper-locked schema, table exports, and deterministic
model-group split. These commands validate the reported corpus and graph counts,
recreate the leakage-audited split, and copy the canonical paper tables into a
clean reproduction directory:

```bash
python src/validate_artifact.py
python src/split_mohkg.py
python src/reproduce_tables.py --output reproduced
```

The provenance policy in `docs/PROVENANCE.md` distinguishes measured, derived,
and estimated records. Canonical paper-facing tables are stored under
`results/paper_tables/`; their released copies are under `reproduced/`.

## Benchmark Dataset

### Models (34 total)

| Category | Count | Representative Models | Operators | Parameters Range |
|----------|-------|----------------------|-----------|------------------|
| LLM | 11 | GPT-2, BERT, LLaMA, T5 | 2,847 | 117M - 7B |
| CV | 11 | ResNet, ViT, YOLO, EfficientNet | 1,892 | 3.4M - 632M |
| NLP | 8 | RoBERTa, DistilBERT, ALBERT | 1,105 | 22M - 355M |
| Audio | 2 | Whisper, Wav2Vec2 | 234 | 39M - 1.5B |
| Multimodal | 2 | CLIP, BLIP | 166 | 151M - 446M |
| **Total** | **34** | - | **6,244** | - |

### Operators (16 types)

| Category | Operators |
|----------|-----------|
| **Matrix Operations** | Linear, Conv1d, Conv2d, MatMul |
| **Normalization** | LayerNorm, BatchNorm2d, RMSNorm |
| **Activation** | ReLU, ReLU6, GELU, SiLU, Softmax, Tanh |
| **Pooling** | MaxPool2d, AdaptiveAvgPool2d |
| **Others** | Embedding, Dropout |

### Hardware Platforms

| Platform | Compute (TFLOPS FP16) | Memory | Bandwidth |
|----------|----------------------|--------|-----------|
| NVIDIA A100 80GB | 312 | 80GB HBM2e | 2.0 TB/s |
| Huawei Ascend 910B | 320 | 64GB HBM2 | 1.2 TB/s |
| Cambricon MLU370-X8 | 256 | 48GB | 307 GB/s |
| Intel GPU Max 1550 | 420 | 128GB HBM2e | 3.2 TB/s |
| Intel Xeon 8380 | 3 (FP32) | 512GB DDR4 | 204 GB/s |

## Project Structure

```
het-benchmark/
├── src/                          # Source code
│   ├── hal.py                    # Hardware Abstraction Layer
│   ├── copa.py                   # COPA algorithm (Two-Stage Shapley)
│   ├── moh_kg.py                 # MOH-KG knowledge graph
│   ├── rgat.py                   # RGAT neural network
│   ├── kg_a2o.py                 # KG-A2O optimization (PPO)
│   ├── split_mohkg.py            # Deterministic model split and leakage audit
│   ├── validate_artifact.py      # Paper count/data validation
│   ├── reproduce_tables.py       # Canonical table reproduction
│   ├── priority.py               # Auditable operator-priority scoring
│   └── operators/                # Operator implementations
├── experiments/                  # Experiment scripts
│   ├── exp_copa_attribution_full.py
│   ├── exp_cross_platform.py
│   ├── exp_gnn_by_operator.py
│   ├── exp_shapley_real.py
│   ├── exp_surrogate_speedup.py
│   └── exp_fitas_migration.py
├── data/                         # Dataset files
│   ├── model_dataset.json        # 34 models with 6,244 operators
│   ├── model_scale_extensions.json # Metadata-only scale boundary checks
│   ├── moh_kg.json               # Knowledge graph
│   ├── paper_schema_counts.json  # Paper-locked node/edge/split schema
│   └── hardware_platforms.json   # Hardware specifications
├── models/                       # Trained models
│   └── rgat_final.pt             # Trained RGAT model (3.7MB, 313K params)
├── results/                      # Experiment results
│   └── paper_tables/             # Canonical paper-facing CSV tables
├── reproduced/                   # Released reproducibility snapshot
├── figures/                      # Generated figures
├── examples/                     # Example scripts
├── docs/                         # Documentation and data provenance
└── ARTIFACT_MANIFEST.json        # Artifact file/checksum manifest
```

## Experimental Results

### COPA Shapley Sampling Accuracy

Evaluated on NVIDIA A100 80GB with real operator latency measurements:

| Model Scale | Operators | Strategy | K=100 MRE (%) | K=200 MRE (%) |
|-------------|-----------|----------|---------------|---------------|
| Small | 8 | Permutation | 1.37 | 0.71 |
| Small | 8 | Stratified | 4.70 | 2.52 |
| Medium | 15 | Permutation | 1.14 | 1.68 |
| Large | 25 | Permutation | 0.89 | 0.62 |

### Surrogate Model Speedup

| Model | Operators | Full Model (ms) | Surrogate (ms) | Speedup |
|-------|-----------|-----------------|----------------|---------|
| Small MLP | 12 | 0.326 | 0.00076 | 429× |
| Medium MLP | 36 | 2.11 | 0.002 | 1,058× |
| Large MLP | 72 | 8.37 | 0.0042 | 1,993× |
| Small Transformer | 18 | 1.34 | 0.00093 | 1,440× |
| Medium Transformer | 54 | 11.64 | 0.00276 | 4,219× |
| Large Transformer | 108 | 46.12 | 0.00559 | 8,252× |

### Cross-Platform Performance (Inference Latency, ms)

Measured on NVIDIA A100 80GB, with cross-platform estimates based on hardware specifications:

| Model | A100 | Ascend 910B | MLU370 | Intel GPU Max | Intel Xeon |
|-------|------|-------------|--------|---------------|------------|
| Small Transformer | 1.78 | 2.51 | 3.70 | 1.19 | 80.0 |
| Medium Transformer | 5.48 | 7.72 | 11.40 | 3.68 | 246.5 |
| Large Transformer | 28.91 | 40.74 | 60.14 | 19.40 | 1300.7 |
| ResNet-18 | 0.90 | 1.14 | 1.62 | 0.63 | 56.4 |

### MOH-KG Guided Optimization

| Optimization Mode | Latency Reduction (%) | Accuracy Impact |
|-------------------|----------------------|-----------------|
| Top-1 MOH-KG Guided | 0.6 | -0.2% |
| Top-3 MOH-KG Guided | 1.9 | +0.0% |
| Top-5 MOH-KG Guided | 2.6 | -0.1% |
| Random Selection | 1.0 | -0.5% |
| Greedy Selection | 2.6 | -0.3% |

### Apple M4 Validation Slice

The repository now includes a compact Apple M4 MPS validation slice with directly measured model- and operator-level results. This slice is intended as bounded second-platform evidence rather than as deployment-class validation for all non-NVIDIA targets.

| Model | Device | Mean (ms) |
|-------|--------|-----------|
| ResNet50 | MPS | 8.8362 |
| ViT-Base | MPS | 80.0000 |
| BERT-Base | MPS | 49.9706 |
| GPT2-Small | MPS | 58.8377 |

| Operator | Device | Mean (ms) |
|----------|--------|-----------|
| Linear | MPS | 0.5255 |
| Conv2d | MPS | 2.2067 |
| LayerNorm | MPS | 1.4670 |
| Softmax | MPS | 2.7262 |
| MultiheadAttention | MPS | 19.6013 |

## Core Algorithms

### COPA (Contribution-based Operator Performance Attribution)

Two-stage Shapley-value based attribution:

**Stage I: Micro-benchmarking**
- Independent operator-level profiling
- CUDA Events for precise timing
- Roofline model analysis

**Stage II: Model-level Attribution**

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

### MOH-KG (Model-Operator-Hardware Knowledge Graph)

- **Node Types**: Hardware, OperatorType, Model, OperatorInstance
- **Edge Types**: r_contains, r_has_type, r_supports, r_seq, r_sim, r_perf
- **Statistics**: 6,299 nodes, 29,199 edges

### RGAT (Relational Graph Attention Network)

- Multi-head attention over heterogeneous edges
- Relation-aware message passing
- Hardware-specific embeddings
- Parameters: 313,089

### Auditable Operator Prioritization

KG-A2O can optionally rank operators before plan generation with the paper's
evidence-backed rule:

$$q_i = a_i g_i (1-u_i)(1-r_i),$$

where attribution $a_i$, optimization headroom $g_i$, uncertainty $u_i$, and
implementation risk $r_i$ must be normalized to $[0,1]$. The implementation in
`src/priority.py` rejects missing or out-of-range evidence, preserves input
order for exact ties, and exports every score component with the ranking.
`KGA2O.optimize(..., priority_evidence=...)` consumes this ranking without
mutating the caller's operator list.

The separate `data/model_scale_extensions.json` file records official scale
descriptors for Llama 3.1 405B, Qwen2.5 72B, and the Stable Diffusion 3 suite.
These records are metadata-only: they are explicitly excluded from the
34-model/6,244-operator profiled corpus and from latency, MRE, COPA, and graph
count results until hardware profiling evidence is attached.

Run the lightweight validation suite with:

```bash
python -m unittest discover -s tests -v
```

## Citation

If you use Het-Benchmark in your research, please cite:

```bibtex
@inproceedings{hetbenchmark2026,
  title={HET: White-Box Operator Benchmarking and Counterfactual Attribution for AI Inference Migration},
  author={xxx},
  booktitle={Proceedings of the AAAI 2027},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by XXX. We thank the anonymous reviewers for their valuable feedback.

---

**Het-Benchmark** - Enabling transparent, interpretable AI model migration across heterogeneous hardware platforms.
