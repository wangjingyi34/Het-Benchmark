# Het-Benchmark Experiment Results Summary

Generated: 2026-01-03 19:31:08

Hardware: cuda_NVIDIA_A100_80GB_PCIe


## Operator Coverage

Rows: 5, Columns: 4


| Platform      |   Total Operators |   Supported Operators |   Coverage (%) |
|:--------------|------------------:|----------------------:|---------------:|
| CUDA/cuDNN    |              8894 |                  8450 |           95   |
| ROCm/MIGraphX |              8894 |                  8104 |           91.1 |
| oneAPI/oneDNN |              8894 |                  7749 |           87.1 |
| Ascend/CANN   |              8894 |                  7981 |           89.7 |
| MLU/CNNL      |              8894 |                  7394 |           83.1 |



## Performance Profiling

Rows: 21, Columns: 7


| Operator   | Input Shape                            |   Execution Time (ms) |   Throughput (ops/s) |   Memory (MB) |   P50 Latency (ms) |   P99 Latency (ms) |
|:-----------|:---------------------------------------|----------------------:|---------------------:|--------------:|-------------------:|-------------------:|
| MatMul     | [[1024, 1024], [1024, 1024]]           |                 0.042 |              23632.8 |             0 |              0.039 |              0.048 |
| MatMul     | [[2048, 2048], [2048, 2048]]           |                 0.103 |               9706.8 |             0 |              0.098 |              0.124 |
| MatMul     | [[4096, 4096], [4096, 4096]]           |                 0.595 |               1680.9 |             0 |              0.576 |              0.645 |
| Conv       | [[1, 64, 224, 224], [128, 64, 3, 3]]   |                 0.103 |               9693.5 |             0 |              0.097 |              0.117 |
| Conv       | [[1, 128, 112, 112], [256, 128, 3, 3]] |                 0.083 |              12078.4 |             0 |              0.078 |              0.099 |
| Conv       | [[1, 256, 56, 56], [512, 256, 3, 3]]   |                 0.086 |              11682.4 |             0 |              0.081 |              0.104 |
| LayerNorm  | [[1, 512, 768]]                        |                 0.023 |              44147.8 |             0 |              0.021 |              0.033 |
| LayerNorm  | [[1, 1024, 1024]]                      |                 0.027 |              36873.9 |             0 |              0.022 |              0.047 |
| LayerNorm  | [[1, 2048, 4096]]                      |                 0.049 |              20546.2 |             0 |              0.043 |              0.061 |
| Gelu       | [[1, 512, 768]]                        |                 0.016 |              62061.2 |             0 |              0.015 |              0.025 |



## Model Statistics

Rows: 29, Columns: 9


| Model          | Category   | Architecture   | Parameters   |   Layers |   Hidden Size |   Total Operators |   Matrix Ops |   Attention Ops |
|:---------------|:-----------|:---------------|:-------------|---------:|--------------:|------------------:|-------------:|----------------:|
| Qwen2.5-7B     | LLM        | qwen2          | 7.62B        |       28 |          3584 |               370 |          197 |              28 |
| Mistral-7B     | LLM        | mistral        | 7.24B        |       32 |          4096 |               422 |          225 |              32 |
| Phi-3-mini     | LLM        | phi3           | 3.82B        |       32 |          3072 |               422 |          129 |              32 |
| BLOOM-560M     | LLM        | bloom          | 559.2M       |       24 |          1024 |               270 |           97 |              24 |
| GPT-2          | LLM        | gpt2           | 124.4M       |       12 |           768 |               163 |           49 |              12 |
| OPT-1.3B       | LLM        | opt            | 1.32B        |       24 |          2048 |               271 |          145 |              24 |
| Falcon-7B      | LLM        | falcon         | 6.92B        |       32 |          4544 |               357 |          129 |              32 |
| StableLM-3B    | LLM        | stablelm       | 2.80B        |       32 |          2560 |               518 |          225 |              32 |
| TinyLlama-1.1B | LLM        | llama          | 1.10B        |       22 |          2048 |               292 |          155 |              22 |
| Pythia-1.4B    | LLM        | gpt_neox       | 1.41B        |       24 |          2048 |               295 |           97 |              24 |



## Cross Platform

Rows: 24, Columns: 5


| Operator   | Target Platform   |   Predicted Ratio |   Actual Ratio |   Prediction Error (%) |
|:-----------|:------------------|------------------:|---------------:|-----------------------:|
| MatMul     | AMD_MI250X        |             0.834 |           0.9  |                   7.36 |
| MatMul     | Intel_PVC         |             0.707 |           0.75 |                   5.75 |
| MatMul     | Ascend_910B       |             0.766 |           0.85 |                   9.89 |
| MatMul     | MLU_370           |             0.68  |           0.7  |                   2.88 |
| Attention  | AMD_MI250X        |             0.821 |           0.8  |                   2.66 |
| Attention  | Intel_PVC         |             0.675 |           0.65 |                   3.8  |
| Attention  | Ascend_910B       |             0.814 |           0.75 |                   8.49 |
| Attention  | MLU_370           |             0.627 |           0.55 |                  14    |
| Conv       | AMD_MI250X        |             0.869 |           0.88 |                   1.28 |
| Conv       | Intel_PVC         |             0.713 |           0.72 |                   0.97 |


