# 完整模型PPO与runtime：实际结果及独立复核

8个真实上下文（A100 MIG与Ascend910B3，ResNet18/50，B1/B4），每上下文4配置×5固定seed、4096steps，共160次真实CPU policy训练。所有数据来自原cal1–3；原eval4–8和额外paired数据未用于reward、状态或mask选择。

完整归档：`planner-matrix-results/all-complete-v1/eight-contexts-160-policies-complete-v1.tar.gz`，远/本机总SHA `9769531960c576850d73ca882ff5a3e1b953ec3e79dd01cbc803058e7ed1d518`。包含160份权重、8份完整结果、日志、逐fit partial、执行代码、输入hash与status。

独立复核通过：160个checkpoint逐份SHA与重新加载cal-only plan一致；固定策略/全部PPO的raw和gate接受算术一致；160次物理exact phi及32次代理exact phi用独立四玩家subset公式对账。该计数是检查次数，不是额外物理样本。

## 原heldout物理表上的raw收益

| 设备 | context | PPO无图% | PPO真实图% | 同容量图零化% | latency% | calibration exhaustive% |
|---|---|---:|---:|---:|---:|---:|
| 910B3 | resnet18_b1 | 0.6821 | -2.3050 | 0.6821 | -3.7524 | -14.2534 |
| 910B3 | resnet18_b4 | -0.5277 | -1.7961 | -2.2341 | -7.7421 | 1.9643 |
| 910B3 | resnet50_b1 | -7.9580 | -7.9580 | -7.9580 | -7.9580 | -7.9580 |
| 910B3 | resnet50_b4 | 8.3732 | 8.3732 | 8.3732 | 8.3732 | 8.3732 |
| A100 MIG | resnet18_b1 | 2.0972 | 2.0972 | 2.0972 | 1.7269 | 2.0972 |
| A100 MIG | resnet18_b4 | 0.2535 | 0.2478 | 0.2524 | 0.2301 | 0.2535 |
| A100 MIG | resnet50_b1 | 0.5038 | 0.5038 | 0.5038 | 0.5038 | 0.5038 |
| A100 MIG | resnet50_b4 | 0.2358 | 0.2233 | 0.2176 | 0.2421 | 0.2421 |

主KG-state配置为`ppo_with_heldout_embedding`；不得根据eval选择无图/有图中更好者作为论文主结果。hardware mask on/off在全部主动作都支持时得到相同计划；这是一项有效控制，不表示已经通过主域证明mask带来收益。独立真实失败probe与quality故障注入另验证硬门。真实图对照没有普遍提升，全部seed与同容量零化对照均应保留。

policy seed先在同一physical block内求均值，之后以5个physical process block形成运行区间；不将5seed×5block当25独立硬件样本。表格列raw，gate后收益和任何回退都另存完整JSON，不能用裁成非负后的数字替代raw。

## 二阶runtime留出验证（11个低阶训练coalition）

| 设备 | context | 二阶heldout全表MRE% | constant MRE% | no-runtime MRE% | 二阶物理phi MAE ms |
|---|---|---:|---:|---:|---:|
| 910B3 | resnet18_b1 | 35.5676 | 20.0944 | 169.5600 | 0.872327 |
| 910B3 | resnet18_b4 | 30.8759 | 16.5489 | 148.1995 | 1.168111 |
| 910B3 | resnet50_b1 | 18.8510 | 12.7834 | 150.6993 | 1.262997 |
| 910B3 | resnet50_b4 | 17.7242 | 12.9983 | 173.6698 | 0.944286 |
| A100 MIG | resnet18_b1 | 0.7355 | 0.9720 | 72.5224 | 0.005234 |
| A100 MIG | resnet18_b4 | 0.1042 | 0.5917 | 30.4426 | 0.003449 |
| A100 MIG | resnet50_b1 | 0.8689 | 0.7421 | 76.1596 | 0.041314 |
| A100 MIG | resnet50_b4 | 0.0740 | 0.2445 | 27.6210 | 0.005692 |

正面可支持的结论：低噪声A100上，固定低阶runtime校准可在完整留出物理表上达到小于1%的coalition MRE；这是实际值检验，超越只与自身代理比较采样收敛。NPU原始计时有显著过程/时段变化，未正则化二阶项没有普遍优于constant，不能宣称所有平台runtime校准增益。全部32个variant均无clamp，原始预测和误差完整保留。

DAG+二阶与sum+二阶在当前四动作固定主路径域接近相同，这是低阶拟合可抵消modular base项的结构现象。二次Boolean游戏的反序pair本身已精确分摊二次项，因此接近0的same-surrogate sampler误差不另算物理精度。69/175算子的跨硬件路径游戏与实际动作效用分别在bridge trace及paired确认中给出。

## 独立paired确认

`paired-method-results-with-ppo.json`已将80个NPU policy固定mask（包括全部seed）用于20个fresh process blocks，训练仍只看原cal。主ratio-of-means、mask0负控、source-relative质量、all同预算固定策略与posthoc oracle均分开。它没有据新paired值重新选择seed、配置或mask。

## 完成边界

这些是本轮新增完整实现和物理表驱动的实验，不追认历史0.98%、15.6%或66%的来源。每个策略的校准证据、计划、检查点、固定数据划分和真实质量门都已可审计。报告实际效应与限制，不以组件名称或代数闭合作为效果证明。
