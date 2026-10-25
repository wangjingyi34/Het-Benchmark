# 独立paired确认：固定计划方法复核

20个fresh process blocks（4context×5），全11个≤2动作coalition、每项20对，合计4400对；主口径为每process的100*(meanA-meanB)/meanA。模型/权重/evaluation input和100张质量sample IDs均与原source证据核对。没有重选mask。

| 模型/批量 | 方法 | 冻结mask | raw收益均值% | 95%t区间% | gate后均值% |
|---|---|---:|---:|---|---:|
| resnet18 / 1 | cross_dag | 6 | 0.6522 | [-4.9218, 6.2261] | 1.9931 |
| resnet18 / 1 | cross_plain_sum | 10 | 3.4855 | [-0.2183, 7.1894] | 3.4855 |
| resnet18 / 1 | target_headroom | 6 | 0.6522 | [-4.9218, 6.2261] | 1.9931 |
| resnet18 / 1 | target_operator_latency | 6 | 0.6522 | [-4.9218, 6.2261] | 1.9931 |
| resnet18 / 4 | cross_dag | 6 | -1.3549 | [-12.0049, 9.2950] | 1.9010 |
| resnet18 / 4 | cross_plain_sum | 10 | 1.8863 | [-0.5078, 4.2803] | 2.1847 |
| resnet18 / 4 | target_headroom | 10 | 1.8863 | [-0.5078, 4.2803] | 2.1847 |
| resnet18 / 4 | target_operator_latency | 10 | 1.8863 | [-0.5078, 4.2803] | 2.1847 |
| resnet50 / 1 | cross_dag | 6 | -1.8692 | [-7.3281, 3.5898] | 0.7320 |
| resnet50 / 1 | cross_plain_sum | 6 | -1.8692 | [-7.3281, 3.5898] | 0.7320 |
| resnet50 / 1 | target_headroom | 6 | -1.8692 | [-7.3281, 3.5898] | 0.7320 |
| resnet50 / 1 | target_operator_latency | 10 | 4.3580 | [-3.5150, 12.2310] | 4.8242 |
| resnet50 / 4 | cross_dag | 5 | 3.3992 | [-1.6961, 8.4945] | 3.3992 |
| resnet50 / 4 | cross_plain_sum | 5 | 3.3992 | [-1.6961, 8.4945] | 3.3992 |
| resnet50 / 4 | target_headroom | 5 | 3.3992 | [-1.6961, 8.4945] | 3.3992 |
| resnet50 / 4 | target_operator_latency | 5 | 3.3992 | [-1.6961, 8.4945] | 3.3992 |

## 负控和科学解释

mask0是两个相同baseline对象的原始A/B差异，所有负值与order分层均保留；它不作为策略no-op的收益。策略不做优化时收益定义为0，避免将相同baseline对象之间的噪声计入random/oracle收益。
- resnet18 / B1：mask0均值-1.4920%，95%t区间[-6.852695422685248, 3.868757629436293]。
- resnet18 / B4：mask0均值4.0176%，95%t区间[-4.090232505531257, 12.125530936505296]。
- resnet50 / B1：mask0均值-0.8495%，95%t区间[-6.4879401921331725, 4.789004893425383]。
- resnet50 / B4：mask0均值1.5882%，95%t区间[-3.793416249544874, 6.969783407311072]。

四个cross-DAG主raw收益区间均跨0，mask0噪声也与小收益同量级。因此这些NPU确认数据提供了真实动作、质量与噪声审计，不能单独支撑稳定加速或优于全部baseline的宣称。质量方面，各冻结计划对相同source样本的retention全部为1；没有将小质量集扩增为500张独立样本。

gate后均值将实际延迟退化时的回退作为0收益，始终与raw并列，不替代原负效应。oracle是后验的paired-baseline-normalized收益oracle，不从新eval选择部署计划。95%t区间仅描述固定模型/数据下5个process的运行变化；没有把pair或policy seed当作硬件重复。

## 证据位置

- `paired-method-results.json`：全部方法、5run、负控、source质量与oracle/random。
- `paired-method-contrasts.json`：同一五process上cross-DAG与其他排名的paired差值，未按单独CI重叠判断方法差异。
- `paired-summary-unit-results.json`：4项纯算术fixture；不等同硬件性能证据。
- `../evidence/paired_raw/`：20份原始物理记录；collector 2f与e88仅最终print不同，独立反向hash验证在`bridge-early-results/print-only-source-independent-verification.json`。

现有PPO仍只使用原cal1–3训练；待其固定计划形成后，可用同一确认raw进行独立后验评价，禁止将本文件数据作为reward或重选依据。
