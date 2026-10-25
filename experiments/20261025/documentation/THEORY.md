# HET方法澄清与可直接采用的论文表述

本材料对应本轮新增的可执行实现及预注册验证方案。正文、补充材料和rebuttal可采用下面的定义与表述，再填入经独立复核的实测结果。代码边界为`methods/`；本文不将新增实现追认为原投稿历史结果。

## 1. 以迁移诊断与可执行决策接口凝练贡献

HET的核心贡献可集中表述为：**以具有图位置和部署上下文的算子实例为比较单位，将源端与目标端证据、拓扑相关的迁移归因、可执行优化动作、质量约束与目标端验证连接成一个可追溯的迁移决策接口。** 模型位置、形状、精度、依赖关系、硬件/后端版本、原始计时和失败状态共同确定一个比较实例，避免将名称相同但执行上下文不同的算子混为同一证据。

该定位形成三个连贯贡献层次：

1. **诊断问题与coalition定义。** 对同一模型/权重/输入/数据与配置，在实际依赖DAG中切换源端和目标端节点成本，得到带明确符号和基线的迁移诊断游戏；将正的目标端回归负担映射到具有真实实现和局部优化空间的动作。
2. **证据转移与计算接口。** MOH-KG编码无标签结构和硬件上下文，预测器通过模型留出和设备留出分别评估；归因采样、图表示与规划都保留训练/验证/评价来源及版本。这使稀疏测量能够在有审计边界的接口中被复用。
3. **受约束的动作执行和验证。** 规划器在明确的动作与预算域内学习，最终是否接受由目标端实测质量和延迟门决定；迁移验证进一步比较相同评价样本上的源端与目标端质量。每个决策可追溯到所用证据、选择规则和实际运行结果。

可直接写入英文正文：

> HET contributes a migration-specific interface between operator-instance diagnosis and executable optimization. The interface preserves graph location, tensor configuration, backend identity and measurement provenance, then connects source-to-target attribution, target-side optimization headroom, hardware-valid actions and measured acceptance. COPA provides the diagnostic decomposition; MOH-KG transfers structural and hardware evidence; KG-A2O selects a budgeted action sequence whose final feasibility is checked on the target execution.

贡献定位与已有研究的关系应正面、具体：配对采样的方差缩减与不确定性估计已有[Covert与Lee的研究](https://proceedings.mlr.press/v130/covert21a.html)；[Shapley Flow](https://proceedings.mlr.press/v130/wang21b.html)讨论在因果DAG上向边分配解释重要性。HET采用算子实例作为玩家，解释硬件/软件栈迁移的延迟诊断游戏，并连接真实目标端优化与质量验证。前者的通用方法为HET提供基础，后者的迁移域定义、证据关联和执行验证确定本文的具体贡献。

预测与规划相关工作可对应其实际任务：[BRP-NAS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/768e78024aa8fdb9b8fe87be86f64745-Abstract.html)包含基于GCN的架构性能预测和NAS搜索策略；新实现中的同特征GCN是明确的图预测控制，不等同于复现完整BRP-NAS搜索系统。[HELP](https://proceedings.neurips.cc/paper_files/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html)通过少量目标硬件样本进行适配；[Akhauri与Abdelfattah](https://proceedings.mlsys.org/paper_files/paper/2024/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html)系统研究设备划分、硬件表示和操作编码。它们支持分别报告严格descriptor模式与含少量硬件probe的模式，而不将不同测量预算合并。

## 2. 精确说明什么时候Shapley退化为普通差值

令节点集合为`N`，实际依赖路径集合为`P`。节点源端成本为`s_i`、目标端成本为`t_i`，`d_i=t_i-s_i`，coalition二进制变量为`z_i=1[i∈S]`。本轮跨硬件诊断游戏为：

```
c_i(S) = s_i + z_i d_i
p(S)   = max_(P∈Paths(G)) sum_(i∈P) c_i(S)
v(S)   = p(S) + r(S)
```

`phi_i[v]`为延迟游戏的Shapley值，**正值表示目标端回归，负值表示目标端加速**。可执行优化游戏则定义`g(S)=T(∅)-T(S)`，其正值表示优化收益；两种符号在输出中明确区分。

若游戏是modular形式`v(S)=c+Σ_i d_i z_i`，对任意前置集合都有`v(S∪{i})-v(S)=d_i`，因此`phi_i=d_i`，不需要采样。以下是具体退化条件：

- **sum聚合且runtime为常数：** `v_sum(S)=Σ_i c_i(S)+r0`，全部玩家的Shapley值精确等于普通目标减源差值。
- **一条包含所有执行算子的串行链且runtime为常数：** critical-path等于sum，也得到`phi_i=d_i`。
- **对所有coalition都由同一条固定路径P*获胜且runtime为常数：** 对路径内玩家`phi_i=d_i`，路径外玩家`phi_i=0`。只有P*包含全部玩家时，才等于全部算子的普通差值。

当不同coalition使获胜路径切换时，节点的边际贡献依赖其前置集合；Shapley对这些上下文中的边际贡献求平均。coalition-dependent runtime也会产生额外分配。若`r(S)=β0+Σ_i β_i z_i+Σ_(i<j) β_ij z_i z_j`，Shapley线性性给出：

```
phi_i[v] = phi_i[p] + β_i + (1/2) sum_(j≠i) β_ij
```

这直接区分了路径最大值贡献与runtime交互贡献。常数`β0`在每个边际差分中抵消，单纯增加固定系统开销不会改变归因。

代码对应：`copa_v2.py::ActionDAGGame`支持critical_path/sum；`migration_bridge.py`输出`plain_target_minus_source_ms`、DAG归因、二者差异和未归属物理gap；`runtime_calibration.py`分别输出二阶runtime、constant、no-runtime与sum的结果。

可直接写入英文正文：

> COPA has an explicit additive limit: with sum aggregation and coalition-independent runtime, every marginal is the operator's target-minus-source latency difference, so Shapley averaging returns that difference exactly. The topology-aware term becomes consequential when the maximizing path changes across coalitions; a coalition-dependent runtime contributes additional context-dependent marginals. We therefore report the plain-difference/sum control, the path-based allocation and the runtime ablations separately, including signed contributions.

## 3. critical path与coalition runtime分别解决什么

依赖DAG确定数据可达性与路径结构；节点profile描述在真实节点输入上的独立操作成本；runtime项吸收路径成本之外随动作组合变化的观测残差。**存在分支并不等于框架在物理执行中并行运行这些分支。** 本轮FX图保留真实残差add、flatten及module依赖，路径最大值是明确的计算代理，完整模型同步计时提供独立的端到端评价。

两个验证域承担不同作用：

1. **逐算子跨硬件域。** 在匹配模型/输入/权重/数据/配置的source-target profile之间构建诊断游戏。混合节点profile用于分解迁移上下文，不被称为可直接执行的混合硬件coalition。跨硬件runtime未知时单独报告`physical_gap - operator_surrogate_gap`，不通过锚定或重缩放使归因和物理gap强行相等。
2. **目标端可执行动作域。** 对四个实际ConvBN fusion动作运行完整16个coalition。每一项都是目标设备上的完整预训练模型执行，并有数值等价、原始计时和真实标签质量证据。该域提供物理动作游戏，以及从诊断优先级到实测优化效果的验证接口。

动作域的runtime采用预先固定的二阶Boolean残差：`r_train(S)=T_train(S)-p(S)`。只拟合阶数≤2的11个训练mask，常数、一次项与二次项通过低阶差分确定；阶数3/4的5个mask和全部eval块均用于留出检查。constant与no-runtime对照揭示固定项与组合依赖项的作用，sum对照揭示路径选择的作用。原始非正预测单独记录，未进行静默clamp。

这一二阶项是本轮实现的组合残差模型，通过留出物理coalition检验其对当前动作域的适用程度；它并不预设任意编译器fusion和任意共享资源竞争都具有二阶形式。**结果表的正面表述应报告实际留出误差和决策效用**，由实验决定二阶项是否足以描述本次任务。

## 4. antithetic pair-mean标准误的推导

随机均匀排列为`π`，其反序为`π^r`；节点i的边际为`X_i(π)=v(P_i(π)∪{i})-v(P_i(π))`。反序映射保留均匀分布，因此两者期望均为`phi_i[v]`。第j个独立pair的单位值为：

```
Z_ij = (X_i(π_j) + X_i(π_j^r))/2
K    = 2m                       # K为总排列数
hat_phi_i = (1/m) sum_j Z_ij
SE_i = sqrt( sum_j (Z_ij-hat_phi_i)^2 / [m(m-1)] )
```

同一pair的两个边际相关，独立单位是m个pair mean，而不是K个独立单排列值。若单边际方差为`σ_i²`、pair相关系数为`ρ_i`，则：

```
Var(Z_i)       = σ_i²(1+ρ_i)/2
Var(hat_phi_i) = σ_i²(1+ρ_i)/K
```

相同K下普通独立排列均值方差为`σ_i²/K`。因此，配对带来的方差变化由实际`ρ_i`决定；负相关时方差降低，不能仅凭“反序”二字保证所有游戏都改善。K500对应250个独立pair。`copa_v2.py::estimate_shapley`保留逐排列边际与pair均值并按上述公式计算；只剩一个pair时不报告无法估计的SE。

动作由多个算子组成时，其归因是这些算子值的线性聚合。对应SE应先在每个pair内对有关算子的`Z_ij`求和，再对这些group pair means计算标准误，保留算子间协方差；不能直接把各算子SE相加，也不能默认它们相互独立。`migration_bridge.py::grouped_action_priority`已实现这一处理。该聚合是算子归因的线性汇总，不改变玩家集合来另称group-as-player Shapley。

**二次游戏的精确边界。** 若整个游戏恰为`v(S)=β0+Σ_i β_i z_i+Σ_(i<j) β_ij z_i z_j`，反序中的前置集合正好是原前置集合在其他玩家中的补集。因此每个pair mean恒为`β_i+(1/2)Σ_(j≠i)β_ij`，一对就等于该二次游戏的exact Shapley。四动作低阶runtime实验在固定主路径下可能属于此情形；接近0的same-surrogate采样误差是这一代数性质，不是额外的物理准确性证据。跨69/175算子的完整source-target路径游戏与实测action-game误差仍按各自定义独立报告。

可直接写入英文正文：

> We use K to denote the total number of permutations. For K=2m antithetic permutations, the independent sampling units are the m forward/reverse pair means. The reported standard error is the sample standard deviation of those pair means divided by sqrt(m), retaining within-pair covariance. For an action spanning several operators, we sum their marginals within each pair before estimating its standard error. These intervals quantify Monte Carlo uncertainty conditional on the specified coalition game.

## 5. 将数值、采样、代理与物理测量误差分开

| 层次 | 对象与检查方式 | 对应实现/输出 |
|---|---|---|
| 数值误差 | 浮点计算的代数一致性、有限值、DAG约束、公式前向/梯度与已知小例子；不是预测误差 | 数学fixture、精确小游戏、`test_methods.py`与轻量bridge测试 |
| 采样误差 | 固定同一个代理游戏，对有限K估计与精确或独立高预算同代理参考比较 | `operator_estimate`、pair-mean SE、明确标记same-surrogate的reference |
| 代理归因误差 | 在**同一组可执行动作玩家、同一coalition语义**下，将代理归因与完整实测表的exact Shapley比较 | 四动作16coalition物理表、二阶/constant/no-runtime/sum与heldout physical exact归因 |
| 物理测量不确定性 | 独立process block的运行变化、同输入同步重复、设备/后端配置差异、有限质量样本 | 原始timings、run IDs、quality sample IDs与逐样本预测；硬件收益以真实run block汇总 |
| 决策效用 | 在固定≤k预算下，校准数据选择的实际动作在heldout上的收益、质量、拒绝、oracle regret及排名 | 固定策略、真正PPO、多seed、随机精确期望、greedy/exhaustive、posthoc oracle、paired migration trace |

固定代理下，`Σ_i phi_i = v(N)-v(∅)`以及每条完整排列的望远镜求和是代数性质。它们用于发现代码错误，**不能单独作为物理coalition准确性的证据**。新的验证将采样一致性、代理留出误差和可执行优化效果并列，使每个结论有相应的实测接口。

在共同可执行域中，可写出误差分解：

```
hat_phi_i[hat_v] - phi_i[v_phys]
 = (hat_phi_i[hat_v] - phi_i[hat_v])
 + (phi_i[hat_v] - phi_i[v_phys])
```

第一项为采样误差，第二项为代理分配误差。若所有共同coalition的value误差均≤ε，Shapley加权边际的绝对分配误差≤2ε，因为每个边际涉及两个coalition值。浮点数值误差和真实计时的估计不确定性应再各自叠加。留出子集的误差用于评价该子集，不自动构成对全部未观测coalition的uniform ε保证。

signed MAE、符号一致率、top-1匹配、top-2重叠直接衡量归因与排序。近零或负的真实贡献保留，不通过只报告正贡献或对近零分母取百分比制造表面稳定。逐算子跨硬件诊断与四动作实测游戏属于不同玩家域，二者通过**动作映射后的决策效果**连接；不把四动作exact值直接当成69/175算子游戏的同维“真值”。

## 6. 硬质量约束与真正PPO的实现闭环

`measured_planner.py`实现了actor与critic、PPO clip objective、GAE、熵项、值函数项和固定seed的实际训练。状态包含selected actions、remaining budget、当前校准归一质量与延迟、局部profile及可选冻结图嵌入；状态和reward使用同一组真实train block均值，因而没有隐藏的episode block上下文。

训练奖励由真实测量表产生：`(T(S)-T(S∪{a}))/T(∅)`，附质量短缺惩罚。**硬约束由单独接受规则执行**，而不是依赖有限reward惩罚替代：硬件不支持、执行失败、实测质量不达标或实测延迟退化时回滚baseline，同时保留未经回退的原始收益与拒绝原因。最终gate只在计划已经由calibration证据选定之后读取eval。

质量具体定义为ImageNet1000 top-1 accuracy的retention ratio，阈值0.99。calibration与evaluation各100张、样本不重叠；归一化分母为**同一个block、同一个data partition**的mask0质量。迁移验证还对相同sample IDs与labels计算目标候选相对源端baseline的retention。例如eval source=.9、target baseline=.8、target candidate=.8时，目标局部retention=1，但source-relative retention=.8/.9，迁移gate不会因只看局部retention而接受。

同一动作预算比较latency ranking、local difference、跨硬件plain/sum、COPA、greedy、exhaustive与PPO。stop/no-op允许使用少于k个动作；同时报告uniform exact-k与uniform at-most-k的精确子集期望。heldout feasible oracle只用于测量decision regret，不参与计划选择。全部seed保留，策略训练seed不算作额外硬件重复。若主fusion动作均支持，mask对照表示这一有效控制情形，独立真实capability失败试验另验证非法动作的拒绝机制。

真实冻结图嵌入来自未用目标模型训练/调参的checkpoint；有图与同容量图零化对照具有相同policy维度。主干PPO采用[原始PPO方法](https://arxiv.org/abs/1707.06347)的clip actor-critic训练结构；本文的具体机制在于真实迁移证据、有限动作域与独立实测接受接口。

