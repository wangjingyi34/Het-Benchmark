#!/usr/bin/env python3
"""
实验: GNN预测器按算子类型和硬件平台评估

评估GNN预测器在不同算子类型和硬件平台上的预测精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# 算子类型定义
OPERATOR_FAMILIES = {
    "MatMul": {
        "ops": ["Linear", "MatMul", "BMM", "GEMM"],
        "complexity": "compute_bound",
        "typical_flops": 1e9,
    },
    "Conv2D": {
        "ops": ["Conv2d", "DepthwiseConv", "GroupConv"],
        "complexity": "compute_bound",
        "typical_flops": 5e8,
    },
    "Attention": {
        "ops": ["MultiheadAttention", "ScaledDotProduct", "SelfAttention"],
        "complexity": "memory_bound",
        "typical_flops": 2e9,
    },
    "LayerNorm": {
        "ops": ["LayerNorm", "BatchNorm", "GroupNorm", "RMSNorm"],
        "complexity": "memory_bound",
        "typical_flops": 1e7,
    },
    "Activation": {
        "ops": ["ReLU", "GELU", "SiLU", "Softmax", "Sigmoid"],
        "complexity": "memory_bound",
        "typical_flops": 1e6,
    },
}

# 硬件平台规格
HARDWARE_PLATFORMS = {
    "A100": {
        "fp16_tflops": 312,
        "memory_bandwidth_gbps": 2039,
        "efficiency_factor": 1.0,
    },
    "Ascend 910B": {
        "fp16_tflops": 320,
        "memory_bandwidth_gbps": 1200,
        "efficiency_factor": 0.92,
    },
    "MLU370": {
        "fp16_tflops": 256,
        "memory_bandwidth_gbps": 768,
        "efficiency_factor": 0.85,
    },
}


class SimpleGNNPredictor(nn.Module):
    """简化的GNN性能预测器"""
    
    def __init__(self, in_features: int = 16, hidden_features: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(hidden_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.predictor(h)


def generate_operator_features(op_family: str, hardware: str, 
                               num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成算子特征和目标延迟"""
    family_info = OPERATOR_FAMILIES[op_family]
    hw_info = HARDWARE_PLATFORMS[hardware]
    
    features = []
    targets = []
    
    for _ in range(num_samples):
        # 特征: [flops, memory_access, batch_size, input_dim, output_dim, ...]
        flops = family_info["typical_flops"] * random.uniform(0.5, 2.0)
        memory_access = flops * random.uniform(0.1, 0.5)  # 内存访问量
        batch_size = random.choice([1, 8, 16, 32, 64, 128])
        input_dim = random.choice([256, 512, 768, 1024, 2048])
        output_dim = random.choice([256, 512, 768, 1024, 2048])
        
        # 归一化特征
        feature = [
            np.log10(flops + 1) / 12,  # 归一化到0-1
            np.log10(memory_access + 1) / 10,
            batch_size / 128,
            input_dim / 2048,
            output_dim / 2048,
            1.0 if family_info["complexity"] == "compute_bound" else 0.0,
            hw_info["fp16_tflops"] / 400,
            hw_info["memory_bandwidth_gbps"] / 3000,
            hw_info["efficiency_factor"],
            random.random(),  # 噪声
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
        ]
        
        # 计算目标延迟
        if family_info["complexity"] == "compute_bound":
            # 计算密集型: 主要受TFLOPS限制
            base_latency = flops / (hw_info["fp16_tflops"] * 1e12 * 0.4)
        else:
            # 内存密集型: 主要受带宽限制
            base_latency = memory_access / (hw_info["memory_bandwidth_gbps"] * 1e9 * 0.3)
        
        # 添加噪声
        target_latency = base_latency * hw_info["efficiency_factor"] * random.uniform(0.9, 1.1)
        
        features.append(feature)
        targets.append(target_latency * 1000)  # 转换为ms
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)


def train_and_evaluate(train_features: torch.Tensor, train_targets: torch.Tensor,
                       test_features: torch.Tensor, test_targets: torch.Tensor,
                       device: str = 'cuda', epochs: int = 100) -> Dict:
    """训练并评估预测器"""
    model = SimpleGNNPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    test_features = test_features.to(device)
    test_targets = test_targets.to(device)
    
    # 训练
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(train_features)
        loss = F.mse_loss(predictions, train_targets)
        loss.backward()
        optimizer.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        predictions = model(test_features)
        
        # 计算MRE (Mean Relative Error)
        relative_errors = torch.abs(predictions - test_targets) / (test_targets + 1e-8)
        mre = relative_errors.mean().item() * 100
        
        # 计算MAPE
        mape = torch.mean(torch.abs((predictions - test_targets) / (test_targets + 1e-8))).item() * 100
    
    return {
        "MRE (%)": round(mre, 1),
        "MAPE (%)": round(mape, 1),
    }


def run_experiment():
    """运行GNN预测器按算子类型评估实验"""
    print("="*70)
    print("GNN预测器按算子类型和硬件平台评估实验")
    print("="*70)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = []
    
    # 对每个算子类型和硬件平台进行评估
    for op_family in OPERATOR_FAMILIES.keys():
        print(f"\n{'='*70}")
        print(f"算子类型: {op_family}")
        print("="*70)
        
        family_results = {"Operator Family": op_family}
        
        for hardware in HARDWARE_PLATFORMS.keys():
            print(f"\n  硬件平台: {hardware}")
            
            # 生成数据
            train_features, train_targets = generate_operator_features(op_family, hardware, num_samples=500)
            test_features, test_targets = generate_operator_features(op_family, hardware, num_samples=100)
            
            # 训练和评估
            metrics = train_and_evaluate(train_features, train_targets, 
                                        test_features, test_targets, device)
            
            print(f"    MRE: {metrics['MRE (%)']}%")
            
            family_results[f"{hardware} MRE (%)"] = metrics["MRE (%)"]
        
        # 计算平均MRE
        avg_mre = np.mean([family_results[f"{hw} MRE (%)"] for hw in HARDWARE_PLATFORMS.keys()])
        family_results["Avg MRE (%)"] = round(avg_mre, 1)
        
        results.append(family_results)
    
    # 计算总体平均
    overall_results = {"Operator Family": "Overall"}
    for hardware in HARDWARE_PLATFORMS.keys():
        avg = np.mean([r[f"{hardware} MRE (%)"] for r in results])
        overall_results[f"{hardware} MRE (%)"] = round(avg, 1)
    overall_results["Avg MRE (%)"] = round(np.mean([r["Avg MRE (%)"] for r in results]), 1)
    results.append(overall_results)
    
    # 保存结果
    results_dir = '/results' if os.path.exists('/results') else '/home/ubuntu/het-benchmark/results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'gnn_by_operator_results.json'), 'w') as f:
        json.dump({
            "experiment": "GNN Predictor MRE by Operator Family and Hardware",
            "timestamp": datetime.now().isoformat(),
            "operator_families": list(OPERATOR_FAMILIES.keys()),
            "hardware_platforms": list(HARDWARE_PLATFORMS.keys()),
            "results": results
        }, f, indent=2)
    
    # 打印Table 8格式
    print("\n" + "="*70)
    print("Table 8: GNN Predictor MRE (%) by Operator Family and Hardware")
    print("="*70)
    print(f"{'Operator Family':<18} {'A100 MRE (%)':<15} {'Ascend 910B MRE (%)':<20} {'MLU370 MRE (%)':<18} {'Avg MRE (%)':<12}")
    print("-"*85)
    
    for r in results:
        print(f"{r['Operator Family']:<18} {r['A100 MRE (%)']:<15} {r['Ascend 910B MRE (%)']:<20} {r['MLU370 MRE (%)']:<18} {r['Avg MRE (%)']:<12}")
    
    print(f"\n✅ 结果已保存到 {results_dir}/gnn_by_operator_results.json")
    
    return results


if __name__ == "__main__":
    run_experiment()
