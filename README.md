# Het-Benchmark

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**A Knowledge-Graph-Driven Evaluation Framework for Zero-Shot AI Model Migration on Heterogeneous Chips**

Het-Benchmark is a comprehensive evaluation framework designed to assess AI model migration capabilities across heterogeneous hardware platforms. It provides fine-grained operator-level analysis, cross-platform performance prediction, and knowledge graph-based insights for zero-shot model deployment.

## 🌟 Key Features

- **Three-Layer Decoupled Architecture**: Model Layer → Operator Layer → Hardware Layer
- **COPA Algorithm**: Contribution-based Operator Performance Attribution using Shapley values
- **MOH-KG**: Model-Operator-Hardware Knowledge Graph with 8,961 nodes and 14,272 edges
- **RGAT**: Relational Graph Attention Network for cross-platform performance prediction
- **Hardware Abstraction Layer (HAL)**: Unified interface for 5 major hardware platforms
- **Comprehensive Dataset**: 29 models, 8,894 operator instances across 6 categories

## 📊 Benchmark Dataset

### Models (29 total)

| Category | Models | Count |
|----------|--------|-------|
| LLM | Qwen2.5-7B, Mistral-7B, Phi-3-mini, BLOOM-560M, GPT-2, OPT-1.3B, Falcon-7B, StableLM-3B, TinyLlama-1.1B, Pythia-1.4B | 10 |
| CV | ResNet-50, ViT-Base, Swin-Base, DINOv2-Base, MobileNet-V2, EfficientNet-B0, ConvNeXt-Base, RegNet-Y-4GF, BEiT-Base, DeiT-Base | 10 |
| NLP | BERT-Base, RoBERTa-Base, T5-Base, DistilBERT | 4 |
| Audio | Whisper-Base, Wav2Vec2-Base | 2 |
| Multimodal | CLIP-ViT-B/32, BLIP-Base, SigLIP-Base | 3 |

### Operators (8,894 instances)

- **Matrix Operations**: MatMul, Conv, Linear, Gemm
- **Activation Functions**: ReLU, GELU, SiLU, Softmax
- **Normalization**: LayerNorm, BatchNorm, RMSNorm
- **Attention**: MultiHeadAttention, ScaledDotProductAttention
- **Pooling**: MaxPool, AvgPool, AdaptiveAvgPool
- **Others**: Embedding, Dropout, Reshape, Transpose

### Hardware Platforms

| Platform | Vendor | Type | Coverage |
|----------|--------|------|----------|
| CUDA/cuDNN | NVIDIA | GPU | 98.0% |
| ROCm/MIGraphX | AMD | GPU | 94.0% |
| oneAPI/oneDNN | Intel | GPU/CPU | 89.1% |
| CANN | Huawei | NPU | 93.0% |
| MLU/CNNL | Cambricon | MLU | 83.1% |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/het-benchmark.git
cd het-benchmark

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.model_parser import ModelParser
from src.operator_extractor import OperatorExtractor
from src.copa import COPA

# Parse a model
parser = ModelParser()
model_info = parser.parse("Qwen/Qwen2.5-7B")

# Extract operators
extractor = OperatorExtractor()
operators = extractor.extract(model_info)

# Run COPA analysis
copa = COPA()
attribution = copa.analyze(operators, platform="nvidia_cuda")
print(attribution.get_bottlenecks())
```

### Run Experiments

```bash
# Run all experiments
python src/run_experiments.py --data-dir data --results-dir results --experiment all

# Run specific experiment
python src/run_experiments.py --experiment coverage
python src/run_experiments.py --experiment profiling
python src/run_experiments.py --experiment copa
```

### Build Knowledge Graph

```bash
# Build MOH-KG from dataset
python src/build_knowledge_graph.py --data-dir data --export-neo4j

# Query the knowledge graph
python src/moh_kg.py --query "operators for model Qwen2.5-7B"
```

## 📁 Project Structure

```
het-benchmark/
├── src/                    # Source code
│   ├── hal.py             # Hardware Abstraction Layer
│   ├── model_parser.py    # Model parsing and analysis
│   ├── operator_extractor.py  # Operator extraction
│   ├── copa.py            # COPA algorithm (Shapley values)
│   ├── moh_kg.py          # MOH-KG knowledge graph
│   ├── rgat.py            # RGAT neural network
│   ├── profiler.py        # Performance profiling
│   └── run_experiments.py # Experiment runner
├── data/                   # Dataset files
│   ├── model_dataset.json # Model and operator dataset
│   └── moh_kg.json        # Knowledge graph
├── results/               # Experiment results
│   ├── table4_model_dataset.csv
│   ├── table5_operator_coverage.csv
│   ├── table6_performance_profiling.csv
│   └── table8_cross_platform_prediction.csv
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── examples/              # Example scripts
└── scripts/               # Utility scripts
```

## 📈 Experimental Results

### Operator Coverage (Table 5)

| Platform | Total Operators | Supported | Coverage |
|----------|-----------------|-----------|----------|
| CUDA/cuDNN | 8,894 | 8,716 | 98.0% |
| ROCm/MIGraphX | 8,894 | 8,360 | 94.0% |
| oneAPI/oneDNN | 8,894 | 7,927 | 89.1% |
| CANN | 8,894 | 8,271 | 93.0% |
| MLU/CNNL | 8,894 | 7,394 | 83.1% |

### Cross-Platform Performance Prediction (Table 8)

| Operator | Target Platform | Predicted Ratio | Actual Ratio | Error |
|----------|-----------------|-----------------|--------------|-------|
| MatMul | AMD MI250X | 0.834 | 0.90 | 7.36% |
| MatMul | Intel PVC | 0.707 | 0.75 | 5.75% |
| Attention | Ascend 910B | 0.814 | 0.75 | 8.49% |
| Conv | AMD MI250X | 0.869 | 0.88 | 1.28% |

**Average Prediction Error: 5.7%**

## 🔬 Core Algorithms

### COPA (Contribution-based Operator Performance Attribution)

COPA uses Shapley values to attribute performance bottlenecks to individual operators:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

Where:
- $\phi_i$ is the Shapley value for operator $i$
- $N$ is the set of all operators
- $v(S)$ is the performance function for coalition $S$

### MOH-KG (Model-Operator-Hardware Knowledge Graph)

The knowledge graph captures relationships between:
- **Models**: AI model metadata and architecture
- **Operators**: Operator instances with parameters
- **Hardware**: Platform capabilities and constraints

### RGAT (Relational Graph Attention Network)

RGAT predicts cross-platform performance using:
- Multi-head attention over heterogeneous edges
- Relation-aware message passing
- Hardware-specific embeddings

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions and feedback, please open an issue on GitHub.

---

**Het-Benchmark** - Enabling transparent, interpretable AI model migration across heterogeneous hardware platforms.
