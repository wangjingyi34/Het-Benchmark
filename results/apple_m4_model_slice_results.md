# Apple M4 Model Slice Results

- Torch: 2.8.0
- MPS available: True
- Protocol: 10 warm-up runs, 20 timed runs, random-weight models, inference mode.

| Model | Device | Input | Mean (ms) | Std (ms) | Median (ms) | Status | Notes |
|---|---|---|---:|---:|---:|---|---|
| ResNet50 | mps | (1, 3, 224, 224) | 8.5946 | 0.3507 | 8.4869 | ok |  |
| ViT-Base | mps | (1, 3, 224, 224) | 79.8244 | 0.4840 | 79.7257 | ok |  |
| BERT-Base | mps | (1, 512) | 50.0143 | 0.5684 | 50.1667 | ok |  |
| GPT2-Small | mps | (1, 512) | 58.9325 | 0.6895 | 58.9114 | ok |  |
