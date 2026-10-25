# HET v2 可执行归因与规划

这是本轮新增实现，不替代、不修改原投稿代码，也不追认历史0.98%、14.3%或15.6%。它把问题转化为能实际执行的counterfactual、物理训练表和独立评价。

## 已实现

1. `copa_v2.py`：action到真实DAG节点的成本更新；critical-path与sum；常数或coalition-dependent runtime；精确Shapley和antithetic采样。K始终为总排列数，标准误按独立pair均值计算，负贡献保留。
2. `runtime_calibration.py`：预注册的二阶Boolean/Möbius residual。四动作只读取训练低阶11个mask，5个高阶mask和全部eval留出。constant、no-runtime、sum使用同一可用数据预算；负预测计数，不静默clamp。same-surrogate sampling error与对物理exact Shapley的代理误差分开。
3. `measured_planner.py`：真实actor-critic PPO，clip+GAE；state和reward都使用训练run块的实测均值，避免隐藏episode context和随机伪造latency。state含selected set、remaining budget、当前校准quality与成本、可选真实heldout图嵌入；最终eval quality和latency门失败时rollback。mask关掉的消融仍保留最终接受门。
4. `analyze_measured.py`：多seed PPO、去mask、无图/有图/同容量图零化对照；相同完整train表下greedy/exhaustive非RL对照；latency/local difference/COPA排序；random exact-k和uniform ≤k（含no-op）两种期望；heldout feasible oracle。每块raw负效应与接受后收益都保留。
5. `attach_embeddings.py`：只精确匹配model/input shape/module paths/device；跨命名device alias需要显式runtime证据文件。必须非零32维、checkpoint和target-model heldout provenance，不制造占位embedding。
6. `migration_bridge.py`：从source/target原始真实FX算子profile构建逐算子迁移DAG游戏（正值表示目标变慢，负值表示加速），保留全部算子而不是只放四个动作。按实际module_paths将算子归因线性聚合到可执行fusion动作，并结合目标端实测calibration headroom选≤2动作。加入同预算plain/sum、目标局部headroom与目标算子延迟排名对照。目标eval与源eval按原始run_index一一配对，使用完全相同sample IDs和labels核验source-relative与target-relative质量，全部raw负收益保留。未知跨硬件runtime不做端点强制闭合。

## 输入和防泄漏

输入schema为`het.measured.coalitions.v2`，由`interventions/assemble_tasks.py`从真实run生成。train/eval run IDs和quality sample IDs不得重叠；局部profile和DAG profile必须只来自train run IDs。quality使用block自身baseline，absolute与retention_ratio明确分开。`UNIT_FIXTURE`只在显式测试构造器开关下允许，正式CLI拒绝。

训练不读取eval reward/observation/quality来选择动作。动作集合选择之后，才调用heldout最终接受门。对所有策略应用同一个quality与latency非退化门，并同时报告接受前结果，避免只展示回退后的非负收益。

## 运行

使用Codex bundled Python，PYTHONPATH设置为上级工作区隔离的`.experiment-deps`（torch2.9.1）。

```
python test_methods.py
python analyze_measured.py --input task.json --output unique-result.json --budget 2 --steps 4096 --seeds 11 22 33 44 55
python attach_embeddings.py --task task.json --embeddings frozen-embeddings.json --output task-with-embeddings.json
python analyze_measured.py --input task-with-embeddings.json --output unique-result-with-embeddings.json --budget 2 --steps 4096 --seeds 11 22 33 44 55 --with-embedding
```

需要device-ID别名时，为attach增加`--embedding-device-id`与`--device-mapping-evidence`，必须由同一个实际runtime身份核验后指定，而不是模糊匹配硬件名。检查输出JSON中的run和profile provenance，不能仅凭字段名字认证测量真实性。

## 当前执行事实

`unit-test-results.json`记录已执行数学、门控、MDP、PPO、泄漏与残差校准测试，数据全部标明UNIT_FIXTURE，不是物理结果。

`pipeline-smoke-data.json`和`pipeline-smoke-results.json`仅复用上一轮v1四块微图真实计时做新代码的离线集成测试。新物理运行数为0；quality是固定tensor数值等价，不是ImageNet准确率。它们不能作为本轮完整模型主结果，只说明代码已能从真实raw table完成训练、独立选择、评价和输出。

本轮完整ResNet模型、真实ImageNet标签质量和跨设备结果等待相邻实验组实际产出后运行。任何未执行项不标通过。

## 跨硬件桥接（CPU离线运行，禁止占用付费NPU）

```
python test_migration_bridge.py
python migration_bridge.py --source-task a100-resnet18-b1.json --target-task npu-resnet18-b1.json --output a100-to-910b-resnet18-b1.json --budget 2 --permutations 500 --seed 42 --reference-permutations 5000 --reference-seed 90210
```

输入是`assemble_tasks.py`冻结的同model、同input_shape、同权重与数据集任务。两侧`source_artifacts`必须能读到原始raw文件并通过SHA256校验。远程绝对路径不在本机时，传入`--raw-path-map original-to-local.json`，该JSON只允许显式原路径到实际本地路径映射；不猜硬件别名。原始calibration1–3须有完整真实node_profiles（可仅部分train运行做profile，但任何profile不得来自eval）。eval4–8保留实际标签与逐样本预测。源/目标真实执行device_id直接取raw，无需embedding别名；此桥接不依赖图嵌入文件。

K500是总排列数，反向配对只有250个独立pair单位。K5000独立seed参考也只衡量同一数学代理的采样误差，不能替代物理归因真值。输出同时记录完整模型物理迁移gap、节点路径代理gap和未归属runtime residual，决不靠重缩放让两者相等。混合source/target节点profile是诊断代理；只有在目标端真正运行的fusion coalition及实测quality用于验证决策效果。source-relative quality不通过时，目标baseline若也不保留source quality，就回退原始source执行。

`migration-bridge-unit-results.json`当前记录11项轻量纯stdlib测试通过，未启动新物理运行或重训练。大模型MC归因请在云CPU/计时完成后的本机运行，以免污染正在执行的CPU/MPS真实计时。
