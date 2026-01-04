# Het-Benchmark

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Release](https://img.shields.io/github/v/release/wangjingyi34/Het-Benchmark)](https://github.com/wangjingyi34/Het-Benchmark/releases)

**A Knowledge-Graph-Driven Evaluation Framework for Zero-Shot AI Model Migration on Heterogeneous Chips**

Het-Benchmark is a comprehensive evaluation framework designed to assess AI model migration capabilities across heterogeneous hardware platforms. It provides fine-grained operator-level analysis, cross-platform performance prediction, and knowledge graph-based insights for zero-shot model deployment.

## 🌟 Key Features

- **Three-Layer Decoupled Architecture**: Model Layer → Operator Layer → Hardware Layer
- **COPA Algorithm**: Two-stage Contribution-based Operator Performance Attribution using Shapley values
- **MOH-KG**: Model-Operator-Hardware Knowledge Graph with 6,299 nodes and 29,199 edges
- **RGAT**: Relational Graph Attention Network for cross-platform performance prediction
- **KG-A2O**: Knowledge-Graph-guided Adaptive Operator Optimization using PPO
- **Hardware Abstraction Layer (HAL)**: Unified interface for 5 major hardware platforms
- **Comprehensive Dataset**: 34 models, 6,244 operator instances across 5 categories
- **Standard Input Dataset**: 1,000 standardized inputs for reproducible benchmarking

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wangjingyi34/Het-Benchmark.git
cd Het-Benchmark

# Install dependencies
pip install -r requirements.txt

# Install Git LFS (for large files)
git lfs install
git lfs pull
```

### Download Benchmark Data

The benchmark input data (~368 MB) is available via GitHub Release:

```bash
# Option 1: Use the setup script (recommended)
python scripts/setup_data.py --data_only

# Option 2: Manual download
wget https://github.com/wangjingyi34/Het-Benchmark/releases/download/v1.0.0/benchmark_data.tar.gz
tar -xzf benchmark_data.tar.gz
```

### Download Models (Optional)

Models are hosted on Hugging Face (~66 GB total). Download only what you need:

```bash
# Download all models (requires ~66 GB)
python scripts/download_models.py --output_dir ./models_hub

# Download specific category
python scripts/download_models.py --output_dir ./models_hub --category LLM

# Download single model
python scripts/download_models.py --output_dir ./models_hub --model "Qwen2.5-7B"
```

### Run Benchmark

```bash
# Run full benchmark evaluation
python scripts/run_benchmark.py --data_dir ./benchmark_data --output_dir ./results

# Run specific category
python scripts/run_benchmark.py --data_dir ./benchmark_data --category LLM

# Profile operators
python scripts/profile_operators.py --output_dir ./operator_profiles
```

## 📊 Benchmark Dataset

### Models (34 total)

| Category | Models | Count |
|----------|--------|-------|
| **LLM** | Qwen2.5-7B, Mistral-7B, Phi-3-mini, BLOOM-560M, GPT-2, OPT-1.3B, Falcon-7B, StableLM-3B, TinyLlama-1.1B, Pythia-1.4B, Yi-6B, RedPajama-3B, InternLM-7B, GPT-Neo-1.3B | 14 |
| **CV** | ResNet-50, ViT-Base, Swin-Base, DINOv2-Base, MobileNet-V2, EfficientNet-B0, ConvNeXt-Base, RegNet-Y-4GF, BEiT-Base, DeiT-Base | 10 |
| **NLP** | BERT-Base, RoBERTa-Base, T5-Base, DistilBERT-Base, ALBERT-Base, BERT-Tiny, BERT-Mini | 7 |
| **Audio** | Whisper-Large-V3, Wav2Vec2-Base | 2 |
| **Multimodal** | CLIP-ViT-L/14 | 1 |

### Operators (16 types, 6,244 instances)

| Category | Operators |
|----------|-----------|
| **Matrix Operations** | Linear, Conv1d, Conv2d, MatMul |
| **Normalization** | LayerNorm, BatchNorm2d, RMSNorm |
| **Activation** | ReLU, ReLU6, GELU, SiLU, Softmax, Tanh |
| **Pooling** | MaxPool2d, AdaptiveAvgPool2d |
| **Others** | Embedding, Dropout |

### Standard Benchmark Inputs

| Category | Files | Size | Description |
|----------|-------|------|-------------|
| **LLM** | 20 | 1.8 MB | Tokenized sequences (128-2048 tokens) |
| **CV** | 15 | 321 MB | Image tensors (224×224 to 512×512) |
| **NLP** | 16 | 1.3 MB | BERT-style inputs (64-512 tokens) |
| **Audio** | 12 | 8.8 MB | Audio waveforms (1-5 seconds) |
| **Multimodal** | 18 | 72.9 MB | Image-text pairs |
| **Diffusion** | 9 | 7.9 MB | Latent tensors |

**Total: 90 files, 413 MB**

### Hardware Platforms

| Platform | Vendor | Type | Coverage |
|----------|--------|------|----------|
| CUDA/cuDNN | NVIDIA | GPU | 98.0% |
| ROCm/MIGraphX | AMD | GPU | 94.0% |
| oneAPI/oneDNN | Intel | GPU/CPU | 89.0% |
| CANN | Huawei | NPU | 93.0% |
| MLU/CNNL | Cambricon | MLU | 83.0% |

## 📁 Project Structure

```
het-benchmark/
├── src/                          # Source code
│   ├── hal.py                    # Hardware Abstraction Layer
│   ├── model_parser.py           # Model parsing and analysis
│   ├── operator_extractor.py     # Operator extraction
│   ├── copa.py                   # COPA algorithm (Two-Stage Shapley)
│   ├── moh_kg.py                 # MOH-KG knowledge graph
│   ├── rgat.py                   # RGAT neural network
│   ├── kg_a2o.py                 # KG-A2O optimization (PPO)
│   ├── profiler.py               # Performance profiling
│   └── operators/                # Operator implementations
│       ├── base.py               # Base operator class
│       ├── linear.py             # Linear operator
│       ├── conv.py               # Convolution operators
│       ├── normalization.py      # Normalization operators
│       ├── activation.py         # Activation functions
│       ├── pooling.py            # Pooling operators
│       ├── embedding.py          # Embedding operator
│       └── dropout.py            # Dropout operator
├── scripts/                      # Utility scripts
│   ├── setup_data.py             # Download benchmark data
│   ├── download_models.py        # Download models from HuggingFace
│   ├── generate_benchmark_data.py # Generate benchmark inputs
│   ├── run_benchmark.py          # Run benchmark evaluation
│   └── profile_operators.py      # Profile operator performance
├── data/                         # Dataset files
│   ├── model_dataset.json        # 34 models with 6,244 operators
│   ├── moh_kg.json               # Knowledge graph (6,299 nodes, 29,199 edges)
│   ├── models.json               # Model metadata
│   ├── operators.json            # All operator instances
│   ├── operator_types.json       # Operator type definitions
│   ├── benchmark_inputs.json     # Standard evaluation inputs
│   ├── hardware_platforms.json   # Hardware specifications
│   └── standard_inputs.json      # Standard inputs (alternative format)
├── models/                       # Trained models
│   ├── rgat_final.pt             # Trained RGAT model (3.7 MB)
│   └── training_report.json      # Training metrics
├── benchmark_data/               # Generated benchmark inputs (via setup_data.py)
│   ├── LLM/                      # LLM input tensors
│   ├── CV/                       # CV input tensors
│   ├── NLP/                      # NLP input tensors
│   ├── Audio/                    # Audio input tensors
│   ├── Multimodal/               # Multimodal input tensors
│   ├── Diffusion/                # Diffusion input tensors
│   └── manifest.json             # Data manifest
├── results/                      # Experiment results
│   ├── table4_model_dataset.csv
│   ├── table5_operator_coverage.csv
│   ├── table6_performance_profiling.csv
│   ├── table7_copa_attribution.csv
│   └── table8_cross_platform_prediction.csv
├── examples/                     # Example scripts
│   ├── quick_start.py            # Basic usage example
│   ├── rgat_inference.py         # RGAT inference example
│   └── kg_query.py               # Knowledge graph query example
└── docs/                         # Documentation
```

## 📈 Experimental Results

### Table 5: Operator Coverage by Platform

| Platform | Vendor | Version | Coverage |
|----------|--------|---------|----------|
| CUDA/cuDNN | NVIDIA | cuDNN 9.0 | 98.0% |
| ROCm/MIGraphX | AMD | ROCm 6.0 | 94.0% |
| oneAPI/oneDNN | Intel | oneDNN 3.0 | 89.0% |
| CANN | Huawei | CANN 8.0 | 93.0% |
| MLU/CNNL | Cambricon | CNNL 1.9 | 83.0% |

### Table 6: Performance Profiling (NVIDIA A100 80GB)

| Operator | Input Shape | Latency (ms) | Throughput (ops/s) |
|----------|-------------|--------------|-------------------|
| Linear | [32, 1024, 1024] | 0.0421 | 23,753 |
| Linear | [32, 4096, 4096] | 0.5952 | 1,680 |
| Conv2d | [32, 64, 224, 224] | 0.1031 | 9,699 |
| LayerNorm | [32, 512, 768] | 0.0231 | 43,290 |
| MultiheadAttention | [32, 512, 768] | 0.2145 | 4,662 |

### Table 7: COPA Attribution Analysis (Top Operators)

| Operator Type | Instance Count | Shapley Value | Contribution |
|---------------|----------------|---------------|--------------|
| Linear | 2,847 | 0.2341 | 23.41% |
| LayerNorm | 1,523 | 0.1876 | 18.76% |
| GELU | 1,245 | 0.1234 | 12.34% |
| Embedding | 456 | 0.0987 | 9.87% |
| Softmax | 892 | 0.0765 | 7.65% |

### Table 8: Cross-Platform Performance Prediction

| Model | Source | Target | Predicted (ms) | Actual (ms) | Error |
|-------|--------|--------|----------------|-------------|-------|
| Qwen2.5-7B | CUDA | ROCm | 18.52 | 19.23 | 3.69% |
| Mistral-7B | CUDA | oneAPI | 25.67 | 27.12 | 5.35% |
| ResNet-50 | CUDA | CANN | 3.45 | 3.62 | 4.70% |
| ViT-Base | CUDA | MLU | 8.92 | 9.45 | 5.61% |

**Average Prediction Error: 4.84%**

### RGAT Model Performance

| Metric | Value |
|--------|-------|
| Nodes | 6,299 |
| Edges | 29,199 |
| Parameters | 313,089 |
| Best Loss | 0.720311 |
| Training Epochs | 100 |

## 🔬 Core Algorithms

### COPA (Contribution-based Operator Performance Attribution)

Two-stage Shapley-value based attribution:

**Stage I: Micro-benchmarking**
- Independent operator-level profiling
- CUDA Events for precise timing
- Roofline model analysis (compute/memory bottleneck)
- Arithmetic intensity calculation

**Stage II: Model-level Attribution**
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

Where:
- $\phi_i$ is the Shapley value for operator $i$
- $N$ is the set of all operators
- $v(S)$ is the performance function for coalition $S$

### MOH-KG (Model-Operator-Hardware Knowledge Graph)

The knowledge graph captures relationships between:
- **Node Types**: Hardware, OperatorType, Model, OperatorInstance
- **Edge Types**:
  - `r_contains`: Model contains operator
  - `r_has_type`: Operator has type
  - `r_supports`: Hardware supports operator type
  - `r_seq`: Sequential execution order
  - `r_sim`: Operator similarity (cosine)
  - `r_perf`: Performance relationship

### RGAT (Relational Graph Attention Network)

RGAT predicts cross-platform performance using:
- Multi-head attention over heterogeneous edges
- Relation-aware message passing
- Hardware-specific embeddings

### KG-A2O (Knowledge-Graph-guided Adaptive Operator Optimization)

PPO-based reinforcement learning for automatic optimization:
- **12 Optimization Actions**: TensorCore, OperatorFusion, LayoutOptimization, PrecisionReduction, FlashAttention, KernelAutoTuning, MemoryPooling, StreamPipelining, GraphOptimization, QuantizationAware, SparsityExploitation, CustomKernel
- Knowledge graph-guided action selection
- Surrogate performance prediction

## 🛠️ Development

### Generate New Benchmark Data

```bash
# Generate standard benchmark inputs
python scripts/generate_benchmark_data.py --output_dir ./benchmark_data

# Generate for specific category
python scripts/generate_benchmark_data.py --output_dir ./benchmark_data --category LLM
```

### Train RGAT Model

```bash
python src/train_rgat.py
```

### Build Knowledge Graph

```bash
python src/build_knowledge_graph.py
```

## 📝 Citation

If you use Het-Benchmark in your research, please cite:

```bibtex
@inproceedings{hetbenchmark2026,
  title={Beyond Black-Box Benchmarking: A Neuro-Symbolic Evaluation Paradigm for Zero-Shot AI Model Migration},
  author={Anonymous},
  booktitle={Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and feedback, please open an issue on GitHub.

---

**Het-Benchmark** - Enabling transparent, interpretable AI model migration across heterogeneous hardware platforms.
