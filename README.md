# Het-Benchmark

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**Beyond Black-Box Benchmarking: A Neuro-Symbolic Evaluation Paradigm for Zero-Shot AI Model Migration**

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

# Run cross-platform evaluation
python experiments/exp_cross_platform.py

# Run GNN predictor evaluation
python experiments/exp_gnn_by_operator.py

# Run FITAS migration case study
python experiments/exp_fitas_migration.py
```

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
│   └── operators/                # Operator implementations
├── experiments/                  # Experiment scripts
│   ├── exp_copa_attribution_full.py
│   ├── exp_cross_platform.py
│   ├── exp_gnn_by_operator.py
│   ├── exp_fitas_migration.py
│   └── exp_shapley_real.py
├── data/                         # Dataset files
│   ├── model_dataset.json        # 34 models with 6,244 operators
│   ├── moh_kg.json               # Knowledge graph
│   └── hardware_platforms.json   # Hardware specifications
├── models/                       # Trained models
│   └── rgat_final.pt             # Trained RGAT model
├── results/                      # Experiment results
├── figures/                      # Generated figures
├── examples/                     # Example scripts
└── docs/                         # Documentation
```

## Experimental Results

### COPA Attribution Accuracy

| Sampling Strategy | Samples | MRE (%) | Time (s) |
|-------------------|---------|---------|----------|
| Permutation | 100 | 0.82 | 12.3 |
| Subset | 50 | 0.95 | 8.7 |
| Stratified | 100 | 0.71 | 15.2 |

### Surrogate Model Speedup

| Model | Full Measurement (s) | Surrogate (s) | Speedup |
|-------|---------------------|---------------|---------|
| GPT-2 | 847.2 | 1.98 | 428× |
| BERT-Base | 523.1 | 0.89 | 588× |
| ResNet-50 | 312.4 | 0.12 | 2,603× |
| ViT-Base | 456.7 | 0.21 | 2,175× |

### GNN Predictor Performance

| Operator Family | MRE (%) |
|-----------------|---------|
| MatMul/Linear | 8.2 |
| Conv2D | 11.5 |
| Attention | 18.7 |
| LayerNorm | 6.3 |
| Activation | 4.1 |
| **Overall** | **14.3** |

### Cross-Platform Performance (Inference Latency, ms)

| Model | A100 | Ascend 910B | MLU370 | Intel GPU Max | Intel Xeon |
|-------|------|-------------|--------|---------------|------------|
| ResNet50 | 4.77 | 5.07 | 6.93 | 3.41 | 170.2 |
| MobileNetV2 | 4.32 | 5.32 | 8.87 | 2.94 | 78.6 |
| BERT-Base | 2.66 | 3.27 | 5.46 | 1.81 | 48.4 |
| GPT-2 | 31.98 | 39.32 | 65.57 | 21.71 | 581.4 |
| ViT-Base | 2.60 | 3.20 | 5.33 | 1.76 | 47.3 |

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

### RGAT (Relational Graph Attention Network)

- Multi-head attention over heterogeneous edges
- Relation-aware message passing
- Hardware-specific embeddings
- Parameters: 313,089

## Citation

If you use Het-Benchmark in your research, please cite:

```bibtex
@inproceedings{hetbenchmark2026,
  title={Beyond Black-Box Benchmarking: A Neuro-Symbolic Evaluation Paradigm for Zero-Shot AI Model Migration},
  author={Wang, Jingyi and others},
  booktitle={Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by [Institution Name]. We thank the anonymous reviewers for their valuable feedback.

---

**Het-Benchmark** - Enabling transparent, interpretable AI model migration across heterogeneous hardware platforms.
