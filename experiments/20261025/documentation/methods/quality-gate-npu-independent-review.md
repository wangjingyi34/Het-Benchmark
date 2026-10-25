# 真实NPU质量拒绝与回滚独立复核

原件：`../evidence/npu-main-backup/quality-stress-npu.json`，SHA256 `381a6c555ccae59aed7051fbe2eee71fb0ef1e0e17bc0cc8c552ae3ba4721409`。

本组从原始100个evaluation sample IDs、labels与predictions独立重算：预训练baseline top1为0.62，显式constant-logit故障候选为0.10，retention为`0.16129032258064518`，低于0.99。实际`final_gate`返回`accepted=false`、`quality_below_threshold`和baseline mask0；该比例与gate输出逐项相同。

拒绝后确实重新执行了原始预训练baseline。其top1恢复0.62；100个sample ID、标签和逐样本预测全部与原baseline一致。故障候选即使具有较低的真实延迟，也未绕过硬质量门。

这项证据验证质量门与执行后回滚路径，属于独立故障注入，排除在主优化动作、主收益和正常质量分布之外。它不用于夸大优化性能，也不代替正常候选的全部语义/质量检验。
