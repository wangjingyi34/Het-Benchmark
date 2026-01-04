# Het-Benchmark 论文技术要求检查清单

基于论文 "Beyond Black-Box Benchmarking: A Neuro-Symbolic Evaluation Paradigm for Zero-Shot AI Model Migration"

## 1. 核心算法要求

### 1.1 COPA (Contribution-based Operator Performance Attribution)
- [ ] 两阶段Shapley值归因
- [ ] Stage I: 微基准测试 (Micro-benchmarking)
- [ ] Stage II: 模型级归因 (Model-level Attribution)
- [ ] Shapley值公式实现: φᵢ = Σ |S|!(|N|-|S|-1)!/|N|! [v(S∪{i}) - v(S)]
- [ ] Roofline模型分析

### 1.2 MOH-KG (Model-Operator-Hardware Knowledge Graph)
- [ ] 4种节点类型: Hardware, OperatorType, Model, OperatorInstance
- [ ] 6种边类型:
  - [ ] r_contains: 模型包含算子
  - [ ] r_has_type: 算子具有类型
  - [ ] r_supports: 硬件支持算子类型
  - [ ] r_seq: 顺序执行关系
  - [ ] r_sim: 算子相似性
  - [ ] r_perf: 性能关系

### 1.3 RGAT (Relational Graph Attention Network)
- [ ] 多头注意力机制
- [ ] 关系感知消息传递
- [ ] 硬件特定嵌入
- [ ] 跨平台性能预测

### 1.4 KG-A2O (Knowledge-Graph-guided Adaptive Operator Optimization)
- [ ] PPO强化学习
- [ ] 12种优化动作
- [ ] 知识图谱引导的动作选择

### 1.5 HAL (Hardware Abstraction Layer)
- [ ] 5个硬件平台支持
- [ ] 统一接口设计

## 2. 数据集要求

### 2.1 模型数据集 (Table 4)
- [ ] 34个模型
- [ ] 5个类别: LLM, CV, NLP, Audio, Multimodal
- [ ] 每个模型包含完整算子列表
- [ ] 参数量统计

### 2.2 标准输入数据集
- [ ] 1000个样本
- [ ] LLM: 300个
- [ ] CV: 250个
- [ ] VLM: 200个
- [ ] Diffusion: 250个

## 3. 实验表格要求

### 3.1 Table 4: Model Dataset
- [ ] 模型名称
- [ ] 类别
- [ ] 参数量
- [ ] 算子数量
- [ ] 架构类型

### 3.2 Table 5: Operator Coverage by Platform
- [ ] 5个硬件平台
- [ ] 覆盖率百分比
- [ ] 版本信息

### 3.3 Table 6: Performance Profiling
- [ ] 算子类型
- [ ] 输入形状
- [ ] 延迟(ms)
- [ ] 吞吐量(ops/s)
- [ ] 内存使用(MB)

### 3.4 Table 7: COPA Attribution Analysis
- [ ] 算子类型
- [ ] 实例数量
- [ ] Shapley值
- [ ] 贡献百分比

### 3.5 Table 8: Cross-Platform Performance Prediction
- [ ] 模型名称
- [ ] 源平台
- [ ] 目标平台
- [ ] 预测延迟
- [ ] 实际延迟
- [ ] 预测误差

## 4. 硬件平台要求

- [ ] NVIDIA CUDA/cuDNN
- [ ] AMD ROCm/MIGraphX
- [ ] Intel oneAPI/oneDNN
- [ ] Huawei Ascend CANN
- [ ] Cambricon MLU/CNNL

## 5. 代码工程要求

- [ ] 完整的源代码
- [ ] requirements.txt
- [ ] README文档
- [ ] 使用示例
- [ ] 测试用例
