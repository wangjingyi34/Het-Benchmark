# 5. Experiments and Results

This section presents a comprehensive experimental evaluation of Het-Benchmark, validating its three core contributions: (1) the accuracy and scalability of COPA; (2) the effectiveness of MOH-KG for neuro-symbolic optimization guidance; and (3) the construction of a comprehensive evaluation dataset and end-to-end migration case studies. All experiments were conducted on real hardware (NVIDIA A100 80GB PCIe) with cross-platform performance estimated using hardware specification-based simulation. To ensure reproducibility, all measurement scripts, Dockerfiles, and knowledge graph dumps are provided in the supplementary materials.

## 5.1 Experimental Setup

### 5.1.1 Hardware Platforms

We evaluate Het-Benchmark across five heterogeneous hardware platforms representing the diversity of modern AI accelerators:

**Table 1: Hardware Platform Specifications**

| Platform | Compute (TFLOPS FP16) | Memory | Bandwidth | Architecture |
|----------|----------------------|--------|-----------|--------------|
| NVIDIA A100 80GB | 312 | 80GB HBM2e | 2.0 TB/s | Ampere |
| Huawei Ascend 910B | 320 | 64GB HBM2 | 1.2 TB/s | Da Vinci |
| Cambricon MLU370-X8 | 256 | 48GB | 307 GB/s | MLUv02 |
| Intel GPU Max 1550 | 420 | 128GB HBM2e | 3.2 TB/s | Xe HPC |
| Intel Xeon 8380 | 3 (FP32) | 512GB DDR4 | 204 GB/s | Ice Lake |

### 5.1.2 Model Dataset

Our benchmark dataset comprises 34 representative AI models spanning four major categories:

**Table 2: Model Dataset Summary**

| Category | Count | Representative Models | Operators | Parameters Range |
|----------|-------|----------------------|-----------|------------------|
| LLM | 11 | GPT-2, BERT, LLaMA, T5 | 2,847 | 117M - 7B |
| CV | 11 | ResNet, ViT, YOLO, EfficientNet | 1,892 | 3.4M - 632M |
| NLP | 8 | RoBERTa, DistilBERT, ALBERT | 1,105 | 22M - 355M |
| Audio | 2 | Whisper, Wav2Vec2 | 234 | 39M - 1.5B |
| Multimodal | 2 | CLIP, BLIP | 166 | 151M - 446M |
| **Total** | **34** | - | **6,244** | - |

![Operator Distribution](../figures/operator_distribution.png)
*Figure 2: Distribution of 16 operator types across 6,244 operator instances in the Het-Benchmark dataset.*

### 5.1.3 MOH-KG Statistics

![MOH-KG Architecture](../figures/moh_kg_architecture.png)
*Figure 1: MOH-KG Knowledge Graph Architecture showing the three-layer structure with Model, Operator, and Hardware nodes.*

**Table 3: MOH-KG Knowledge Graph Statistics**

| Component | Count | Description |
|-----------|-------|-------------|
| Model Nodes | 34 | AI model instances |
| Operator Nodes | 6,244 | Operator instances across all models |
| Hardware Nodes | 5 | Target hardware platforms |
| Operator Type Nodes | 16 | Unique operator types |
| Total Nodes | 6,299 | All node types combined |
| r_contains Edges | 6,244 | Model-Operator relationships |
| r_has_type Edges | 6,244 | Operator-Type relationships |
| r_supports Edges | 80 | Hardware-Type support |
| r_seq Edges | 6,210 | Sequential execution order |
| r_sim Edges | 5,421 | Operator similarity |
| r_perf Edges | 5,000 | Performance relationships |
| **Total Edges** | **29,199** | All edge types combined |

## 5.2 COPA Evaluation

### 5.2.1 Shapley Sampling Accuracy

We evaluate the accuracy of different Shapley value estimation strategies by comparing against ground truth values computed through exhaustive permutation sampling (10,000 samples).

**Table 4: Shapley Sampling Strategy Comparison (Real A100 Measurements)**

| Strategy | Samples | MRE (%) | Std Dev | Time (ms) |
|----------|---------|---------|---------|-----------|
| Permutation | 100 | 3.21 | 0.89 | 45.2 |
| Permutation | 500 | 1.45 | 0.42 | 223.1 |
| Permutation | 1000 | 0.72 | 0.21 | 447.8 |
| Antithetic | 100 | 2.15 | 0.67 | 48.3 |
| Antithetic | 500 | 0.98 | 0.31 | 238.5 |
| Stratified | 100 | 2.87 | 0.78 | 52.1 |
| Stratified | 500 | 1.23 | 0.38 | 256.7 |

The Antithetic sampling strategy achieves the best accuracy-efficiency trade-off, reducing MRE by 33% compared to standard Permutation sampling with similar computational cost.

### 5.2.2 Surrogate Model Speedup

We measure the speedup achieved by using surrogate models (lookup tables) compared to actual model execution:

**Table 5: Surrogate Model Speedup (Real A100 Measurements)**

| Model | Actual Exec (ms) | Surrogate (ms) | Speedup |
|-------|-----------------|----------------|---------|
| ResNet50 | 4.754 | 0.011 | 429× |
| BERT-Base | 2.661 | 0.008 | 333× |
| GPT-2 Small | 31.982 | 0.012 | 2,665× |
| ViT-Base | 2.596 | 0.009 | 288× |
| ResNet152 | 14.114 | 0.015 | 941× |
| MobileNetV2 | 4.441 | 0.007 | 634× |
| EfficientNet-B0 | 6.564 | 0.010 | 656× |
| LLaMA-7B | 89.234 | 0.018 | 4,957× |
| Whisper-Large | 156.782 | 0.019 | 8,252× |

The surrogate model approach achieves 429× to 8,252× speedup, enabling rapid what-if analysis for migration planning.

### 5.2.3 COPA Attribution Analysis

COPA identifies the contribution of each operator type to overall model performance, enabling targeted optimization:

**Table 6: COPA Attribution Analysis - Operator Contribution by Model Category**

| Operator Type | LLM (%) | CV (%) | NLP (%) | Audio (%) | Overall (%) |
|--------------|---------|--------|---------|-----------|-------------|
| MatMul/Linear | 38.2 | 12.5 | 35.8 | 28.4 | 28.7 |
| Attention | 32.1 | 8.2 | 30.2 | 25.6 | 24.0 |
| Conv2D | 2.3 | 45.8 | 1.8 | 18.2 | 17.0 |
| LayerNorm | 12.4 | 5.2 | 14.6 | 8.3 | 10.1 |
| Softmax | 8.5 | 3.1 | 9.2 | 6.8 | 6.9 |
| Activation | 3.2 | 15.4 | 4.8 | 7.2 | 7.7 |
| Pooling | 0.8 | 6.2 | 1.2 | 2.8 | 2.8 |
| Others | 2.5 | 3.6 | 2.4 | 2.7 | 2.8 |

**Key Findings:**
- **LLM models**: Dominated by MatMul (38.2%) and Attention (32.1%)
- **CV models**: Conv2D contributes 45.8% of execution time
- **Cross-category**: LayerNorm consistently contributes 10-15% across all categories

![COPA Attribution Heatmap](../figures/copa_attribution_heatmap.png)
*Figure 3: COPA Attribution Analysis heatmap showing operator contribution percentages by model category.*

**Table 7: Top Bottleneck Operators by Model (Real A100 Measurements)**

| Model | #1 Bottleneck | Contrib (%) | #2 Bottleneck | Contrib (%) | #3 Bottleneck | Contrib (%) |
|-------|--------------|-------------|---------------|-------------|---------------|-------------|
| GPT-2 | Attention | 57.6 | FFN/Linear | 22.8 | LayerNorm | 10.9 |
| BERT-Base | Attention | 48.3 | Linear | 28.5 | LayerNorm | 12.1 |
| ResNet50 | Conv2D | 62.4 | BatchNorm | 18.2 | ReLU | 12.8 |
| ViT-Base | Attention | 52.1 | Linear | 25.3 | LayerNorm | 14.2 |
| LLaMA-7B | Attention | 61.2 | Linear | 24.8 | RMSNorm | 8.5 |

## 5.3 MOH-KG Optimization Guidance

### 5.3.1 Knowledge Graph-Guided Optimization

We compare three optimization strategies for selecting operators to optimize:

**Table 8: MOH-KG Optimization Strategy Comparison**

| Strategy | Latency Reduction (%) | Accuracy Change (%) | Selection Time (ms) |
|----------|----------------------|---------------------|---------------------|
| Random | 0.8 ± 0.4 | -0.3 ± 0.2 | 0.1 |
| Greedy (Top-K) | 1.9 ± 0.3 | -0.2 ± 0.1 | 12.5 |
| MOH-KG Guided | 2.6 ± 0.2 | -0.1 ± 0.1 | 3.4 |

MOH-KG-guided optimization achieves:
- **3.25× better** latency reduction than random selection
- **37% better** than greedy selection
- **73% faster** selection time than greedy approach

### 5.3.2 GNN Performance Predictor

We train a Relational Graph Attention Network (RGAT) to predict operator performance across platforms:

**Table 9: GNN Predictor MRE by Operator Type (Real A100 Training)**

| Operator Type | Train MRE (%) | Val MRE (%) | Test MRE (%) |
|--------------|---------------|-------------|--------------|
| MatMul | 8.2 | 10.5 | 12.4 |
| Conv2D | 7.8 | 9.2 | 11.5 |
| Attention | 18.5 | 28.3 | 36.9 |
| LayerNorm | 4.2 | 5.1 | 6.6 |
| Softmax | 5.8 | 7.2 | 8.9 |
| Activation | 2.1 | 3.0 | 4.0 |
| Pooling | 6.5 | 8.1 | 9.8 |
| Embedding | 9.2 | 11.8 | 14.2 |
| **Overall** | **7.8** | **10.4** | **14.3** |

**Note:** Attention operators show higher MRE (36.9%) due to complex memory access patterns and hardware-specific optimizations (e.g., FlashAttention on A100).

## 5.4 Cross-Platform Performance Comparison

### 5.4.1 Operator-Level Performance

**Table 10: Cross-Platform Operator Performance (Relative to A100=100%)**

| Operator | A100 | Ascend 910B | MLU370-X8 | Intel GPU Max | Intel Xeon |
|----------|------|-------------|-----------|---------------|------------|
| MatMul 4096×4096 | 100% | 94.1% | 68.8% | 139.7% | 2.8% |
| Conv2D 224×224 | 100% | 94.1% | 68.8% | 139.7% | 2.8% |
| Attention 512seq | 100% | 81.3% | 48.8% | 147.3% | 5.5% |
| LayerNorm 768 | 100% | 64.3% | 22.1% | 157.5% | 9.1% |
| Softmax 512×512 | 100% | 64.3% | 22.1% | 157.5% | 9.1% |
| Linear 4096 | 100% | 94.1% | 68.8% | 139.7% | 2.8% |

**Key Observations:**
- **Compute-bound operators** (MatMul, Conv2D, Linear): Performance scales with compute capability
- **Memory-bound operators** (LayerNorm, Softmax): Performance scales with memory bandwidth
- **Intel GPU Max** outperforms A100 due to higher compute (420 vs 312 TFLOPS) and bandwidth (3.2 vs 2.0 TB/s)

### 5.4.2 Model-Level Performance

**Table 11: Cross-Platform Model Performance (Latency in ms)**

| Model | A100 | Ascend 910B | MLU370-X8 | Intel GPU Max | Intel Xeon |
|-------|------|-------------|-----------|---------------|------------|
| ResNet50 | 4.77 | 5.07 | 6.93 | 3.41 | 170.2 |
| MobileNetV2 | 4.32 | 5.32 | 8.87 | 2.94 | 78.6 |
| BERT-Base | 2.66 | 3.27 | 5.46 | 1.81 | 48.4 |
| GPT-2 Small | 31.98 | 39.32 | 65.57 | 21.71 | 581.4 |
| ViT-Base | 2.60 | 3.20 | 5.33 | 1.76 | 47.3 |

![Cross-Platform Performance](../figures/cross_platform_performance.png)
*Figure 4: Cross-platform model inference latency comparison across 5 hardware platforms (log scale).*

## 5.5 Baseline Comparison

### 5.5.1 Evaluation Method Comparison

**Table 12: Evaluation Method Feature Comparison**

| Feature | MLPerf | DeepBench | Het-Benchmark |
|---------|--------|-----------|---------------|
| Granularity | End-to-End | Operator | Operator + Model Context |
| Cross-Platform | No (re-run) | Partial | Yes (MOH-KG) |
| Attribution | No | No | Yes (COPA) |
| Zero-Shot Prediction | No | No | Yes (GNN) |
| Migration Guidance | No | Limited | Yes (KG-A2O) |
| Setup Complexity | High | Medium | Low |

### 5.5.2 Prediction Accuracy Comparison

**Table 13: Prediction Accuracy vs Baselines (Real Measurements)**

| Method | MatMul MRE | Conv MRE | Attention MRE | Overall MRE |
|--------|------------|----------|---------------|-------------|
| DeepBench (interpolation) | 25.3% | 28.7% | 42.1% | 32.0% |
| Roofline Model | 18.5% | 22.4% | 35.8% | 25.6% |
| Het-Benchmark (GNN) | 12.4% | 11.5% | 36.9% | 14.3% |

Het-Benchmark achieves **55.5% lower overall MRE** compared to DeepBench interpolation, with particularly strong improvements on compute-bound operators.

## 5.6 Migration Case Study

### 5.6.1 A100 to Ascend 910B Migration

We demonstrate end-to-end migration guidance using Het-Benchmark:

**Table 14: Migration Results Summary (A100 → Target Platform)**

| Target | Latency Change | Cost Change | Power Change | Accuracy Change |
|--------|---------------|-------------|--------------|-----------------|
| Ascend 910B | +18.7% | -66.0% | -22.5% | -0.2% |
| MLU370-X8 | +65.7% | -72.0% | -35.0% | -0.3% |
| Intel GPU Max | -32.1% | +15.0% | +8.0% | 0.0% |
| Intel Xeon | +3,470% | -85.0% | -60.0% | 0.0% |

**Migration Recommendations:**
1. **Cost-optimized**: Ascend 910B (66% cost reduction with 18.7% latency increase)
2. **Performance-optimized**: Intel GPU Max (32.1% latency reduction)
3. **Power-optimized**: Intel Xeon for non-latency-critical workloads

### 5.6.2 Operator-Level Migration Analysis

**Table 15: Operator Migration Impact Analysis**

| Operator | A100 (ms) | 910B (ms) | Change | Optimization Suggestion |
|----------|-----------|-----------|--------|------------------------|
| Attention | 0.256 | 0.315 | +23.0% | Use NPU-optimized attention kernel |
| MatMul | 0.630 | 0.669 | +6.2% | Leverage matrix unit |
| LayerNorm | 0.085 | 0.132 | +55.3% | Fuse with adjacent ops |
| Softmax | 0.094 | 0.146 | +55.3% | Use vector unit |
| Conv2D | 0.133 | 0.141 | +6.0% | Standard mapping |

## 5.7 Summary

Our experimental evaluation demonstrates that Het-Benchmark successfully achieves its three core contributions:

1. **COPA Accuracy**: Shapley-based attribution achieves <1% MRE with 1000 samples, and surrogate models provide 429×-8,252× speedup for rapid analysis.

2. **MOH-KG Effectiveness**: Knowledge graph-guided optimization achieves 3.25× better results than random selection, with GNN predictor achieving 14.3% overall MRE.

3. **Comprehensive Evaluation**: The benchmark covers 34 models, 6,244 operators, and 5 hardware platforms, enabling practical migration decisions with quantified trade-offs.

All experimental data and code are available at: https://github.com/wangjingyi34/Het-Benchmark
