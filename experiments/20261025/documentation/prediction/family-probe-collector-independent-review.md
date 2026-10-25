# Independent pre-execution source review

Reviewed collector: `../methods/family_calibration_probes.py`, SHA256 `cdd236c31f3c223889e68026f400aad5a6413ec6c1173603d92fb05a4d91dce2`.

Frozen timing/input core: `collect_profiles.py`, SHA256 `7050d276d82bead7f6b936be35f93059be5d24b0073eb7045434174bec618f26`.

Conclusion: no execution-blocking P0 found in static source/API review. This is not a claim that physical calibration has executed or improves prediction.

- The fixed list contains the14 observed executable families. There is no fabricated Softmax measurement. Composite calls actual torchvision `_DenseLayer`; Other calls actual eval-mode stochastic_depth. Dropout and Other identity semantics are explicitly recorded.
- The manifest is constructed without a forward or timing and binds the collector/core hashes, fixed specifications, module/parameter/input descriptors and portable PCG64 byte hashes. Collection reconstructs each case on CPU and requires exact metadata and input/weight identity before transfer.
- Inputs/parameters are FP32, with int64 token indices only for Embedding. CUDA TF32 and NPU HF32 flags are disabled/read back. Torch_npu registration occurs before constructing the NPU device.
- Runtime validation binds the hardware-only descriptor to the actual device/backend/host and visible accelerator partition; no model target profile is accepted as descriptor input. Current NPU descriptor contains the required source-profile hash.
- Timing uses the unchanged synchronized core: ten warmups followed by twenty eight-call block means. One post-timing output check is declared separately; it records output structure/hash/finiteness and is not counted as a timed sample. The wall clock includes construction, transfer, warmup and output checks.
- All14 attempts have explicit records. Exceptions retain status/error and continue; invalid output is retained as its own nonmeasured status. A failed attempt must never be assigned a fabricated timing or silently omitted by analysis.
- Auxiliary budget is seventeen standard workload types: original3 plus new14. With all attempts successful, this means2720 timed calls,170 warmups and14 post-timing output checks. It is not seventeen total function calls.

Downstream adapter failure policy is independently frozen: if any runtime fails one family, that entire family falls back to the original3 mapping on all five runtimes. This prevents ratios between different workloads. Both retrieval and the ridge information-control columns follow the same all-stack fallback. Original raw failures and measured outcomes remain intact.
