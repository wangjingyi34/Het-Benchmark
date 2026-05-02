# Apple M4 Operator Slice Results

- Torch: 2.8.0
- MPS available: True

| Operator | Device | Shape | Mean (ms) | Std (ms) | Median (ms) | Status | Notes |
|---|---|---|---:|---:|---:|---|---|
| Linear | mps | (32, 1024) | 0.5815 | 0.1601 | 0.5430 | ok |  |
| Conv2d | mps | (1, 64, 224, 224) | 2.1709 | 0.2260 | 2.1009 | ok |  |
| LayerNorm | mps | (32, 512, 768) | 1.4826 | 0.1800 | 1.4049 | ok |  |
| Softmax | mps | (8, 12, 512, 512) | 2.7355 | 0.2651 | 2.6422 | ok |  |
| MultiheadAttention | mps | (4, 512, 768) | 20.9730 | 0.6902 | 20.8210 | ok |  |
| Linear | cpu | (32, 1024) | 0.1756 | 0.0587 | 0.1601 | ok |  |
| Conv2d | cpu | (1, 64, 224, 224) | 12.8128 | 0.6713 | 12.5604 | ok |  |
| LayerNorm | cpu | (32, 512, 768) | 1.8725 | 0.3205 | 1.8478 | ok |  |
| Softmax | cpu | (8, 12, 512, 512) | 7.8067 | 1.5547 | 7.0039 | ok |  |
| MultiheadAttention | cpu | (4, 512, 768) | 19.6222 | 1.5782 | 18.8552 | ok |  |
