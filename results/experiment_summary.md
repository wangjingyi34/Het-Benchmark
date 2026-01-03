# Het-Benchmark Experiment Results

## Experiment Information
- **Date**: 2026-01-03 20:43:20
- **Device**: cuda
- **Total Models**: 34
- **Total Operators**: 468

## Generated Tables

### Table 4: Model Dataset Statistics
- Models analyzed: 34
- Categories: LLM, CV, NLP, Audio, Multimodal, Diffusion

### Table 5: Operator Coverage
| Platform | Vendor | Coverage |
|----------|--------|----------|
| CUDA/cuDNN | NVIDIA | 98.0% |
| ROCm/MIGraphX | AMD | 94.0% |
| oneAPI/oneDNN | Intel | 89.0% |
| CANN | Huawei | 93.0% |
| MLU/CNNL | Cambricon | 83.0% |

### Table 6: Performance Profiling
- Operators profiled: 13
- Device: cuda

### Table 7: COPA Attribution Analysis
- Top operators by Shapley value: 7

### Table 8: Cross-Platform Prediction
- Predictions generated: 25
- Platforms: CUDA, ROCm, oneAPI, CANN, MLU

## Files Generated
- `table4_model_dataset.csv`
- `table5_operator_coverage.csv`
- `table6_performance_profiling.csv`
- `table7_copa_attribution.csv`
- `table8_cross_platform_prediction.csv`
- `experiment_summary.json`
