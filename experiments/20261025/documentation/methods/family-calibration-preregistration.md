# 14family硬件校准微探针：冻结协议

新增collector为`family_calibration_probes.py`，源码SHA256 `cdd236c31f3c223889e68026f400aad5a6413ec6c1173603d92fb05a4d91dce2`；硬件无关manifest为`family-calibration-manifest.json`，SHA256 `ea400f2b2df47abbd712dee71322a038f7282831d369b422ee8f016bff6c3c94`。

在任何新探针计时之前，已经仅在CPU构造并冻结全部14个operator、shape、参数、PCG64 seeds、逐input/weight/combined bytehash。构造未调用forward、未取得任何新latency。冻结core仍为7050，主bad26和13905条labels均不改。

| family | 固定真实运算 |
|---|---|
| Conv2D | Conv2d(8,8,3,pad1)，1×8×16×16 |
| Conv1D | Conv1d(8,8,3,pad1)，1×8×64 |
| MatMul | 128×128 FP32矩阵乘 |
| Attention | functional SDPA，1×4×32×16，dropout0、noncausal |
| Normalization | LayerNorm(32)，4×32 |
| Activation | GELU，32×32，approximate none |
| Pooling | AdaptiveAvgPool2d(1)，1×8×8×8 |
| Elementwise | torch.add，两个32×32张量 |
| View | transpose(1,2)，2×16×32，仅view |
| Embedding | Embedding(128,16)，INT64索引2×16 |
| Dropout | Dropout(.1).eval，32×32，真实identity路径 |
| Reduction | mean(dim1)，4×64 |
| Composite | 实际_DenseLayer(32,growth8,bnsize2,droprate0)，1×32×8×8，inplace关闭 |
| Other | 实际torchvision stochastic_depth(p=.2,row,training=False)，1×8×8×8 |

真实catalog有上述14类，没有实际Softmax查询，因此不另造第15类探针。函数probe的`actual_module_class=Identity`为无参数state-hash holder，真正执行的函数由`specification.operator`和collector的callable分支确定；该holder不代替函数调用。

每项10次warmup、20个同步block、每block8次真实调用（7050 timed），记录20个原始block mean、均值/标准差、实际输出结构/hash/finite、每probe wall。另在计时后做1次已声明的非计时输出检查。Dropout/View/Other的identity或view路径如实计时，不伪称有GPU kernel工作。

严格使用hardware-only descriptor，保留原3项probe、原profile SHA、实际runtime身份、线程数与精度readback。NPU使用既有FP32 launcher及双HF32 False；MPS禁用silent CPU fallback。各栈在device转移之前核对相同CPU input/weight bytes，失败保留并更新partial证据。

总预算是原3+新14=17个query types，不等于17次单调用；warmup、timed calls、输出检查和wall均分别可审计。每类只有一个固定小shape，它是受限的硬件校准信号，不保证代表该family的全部shape、kernel或调度行为。

经与预测组核对，新14只替换其已冻结evidence-memory的ratio证据，k/temperature/gamma保持不变。任一栈某类失败时，不制造ratio、不换family、不挑测试网格；该family在全部五栈统一回退原3项既定映射，保证ratio两侧与ridge列使用同一workload语义，报告原始失败、全family和全栈fallback计数。auxiliary测量不加入主训练/validation/test标签集合。

root统一安排各栈真实采集。此文件仅冻结协议，不声明尚未执行的硬件结果通过；M5额外8上下文PPO仍为prepared/not executed。
