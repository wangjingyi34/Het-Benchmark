#!/usr/bin/env python3
"""
实验: GNN预测器按算子类型和硬件平台评估 (改进版)

使用更真实的性能模型和更好的训练策略
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


# 算子类型定义（基于真实性能特征）
OPERATOR_FAMILIES = {
    "MatMul": {
        "base_latency_us": 100,  # 基础延迟（微秒）
        "compute_intensity": 0.9,  # 计算密集度
        "memory_intensity": 0.3,
        "variance": 0.15,
    },
    "Conv2D": {
        "base_latency_us": 150,
        "compute_intensity": 0.85,
        "memory_intensity": 0.4,
        "variance": 0.12,
    },
    "Attention": {
        "base_latency_us": 200,
        "compute_intensity": 0.7,
        "memory_intensity": 0.6,
        "variance": 0.18,
    },
    "LayerNorm": {
        "base_latency_us": 20,
        "compute_intensity": 0.2,
        "memory_intensity": 0.8,
        "variance": 0.08,
    },
    "Activation": {
        "base_latency_us": 10,
        "compute_intensity": 0.1,
        "memory_intensity": 0.9,
        "variance": 0.05,
    },
}

# 硬件平台规格
HARDWARE_PLATFORMS = {
    "A100": {
        "compute_factor": 1.0,
        "memory_factor": 1.0,
        "base_overhead_us": 5,
    },
    "Ascend 910B": {
        "compute_factor": 0.95,
        "memory_factor": 0.85,
        "base_overhead_us": 8,
    },
    "MLU370": {
        "compute_factor": 0.82,
        "memory_factor": 0.75,
        "base_overhead_us": 10,
    },
}


class ImprovedGNNPredictor(nn.Module):
    """改进的GNN性能预测器"""
    
    def __init__(self, in_features: int = 8, hidden_features: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(hidden_features // 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.predictor(h)


def generate_realistic_data(op_family: str, hardware: str, 
                           num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成更真实的算子性能数据"""
    family_info = OPERATOR_FAMILIES[op_family]
    hw_info = HARDWARE_PLATFORMS[hardware]
    
    features = []
    targets = []
    
    for _ in range(num_samples):
        # 算子参数
        batch_size = random.choice([1, 2, 4, 8, 16, 32, 64, 128])
        input_size = random.choice([64, 128, 256, 512, 768, 1024, 2048])
        output_size = random.choice([64, 128, 256, 512, 768, 1024, 2048])
        
        # 计算FLOPs和内存访问量
        if op_family == "MatMul":
            flops = batch_size * input_size * output_size * 2
            memory_bytes = batch_size * (input_size + output_size) * 4
        elif op_family == "Conv2D":
            kernel_size = random.choice([1, 3, 5, 7])
            flops = batch_size * input_size * output_size * kernel_size * kernel_size
            memory_bytes = batch_size * (input_size + output_size) * 4
        elif op_family == "Attention":
            seq_len = random.choice([64, 128, 256, 512])
            flops = batch_size * seq_len * seq_len * input_size * 4
            memory_bytes = batch_size * seq_len * input_size * 4
        elif op_family == "LayerNorm":
            flops = batch_size * input_size * 5
            memory_bytes = batch_size * input_size * 8
        else:  # Activation
            flops = batch_size * input_size
            memory_bytes = batch_size * input_size * 8
        
        # 归一化特征
        feature = [
            np.log10(batch_size + 1) / 3,
            np.log10(input_size + 1) / 4,
            np.log10(output_size + 1) / 4,
            np.log10(flops + 1) / 12,
            np.log10(memory_bytes + 1) / 10,
            family_info["compute_intensity"],
            family_info["memory_intensity"],
            hw_info["compute_factor"],
        ]
        
        # 计算真实延迟（基于性能模型）
        compute_time = flops / (312e12 * hw_info["compute_factor"] * 0.4)  # 假设40%效率
        memory_time = memory_bytes / (2039e9 * hw_info["memory_factor"] * 0.5)  # 假设50%效率
        
        # 总延迟 = max(计算时间, 内存时间) + 开销
        base_latency = max(compute_time, memory_time) * 1e6  # 转换为微秒
        overhead = hw_info["base_overhead_us"]
        
        # 添加真实噪声
        noise = 1 + random.gauss(0, family_info["variance"])
        target_latency = (base_latency + overhead) * noise
        
        features.append(feature)
        targets.append(target_latency)
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)


def train_and_evaluate_v2(train_features: torch.Tensor, train_targets: torch.Tensor,
                          test_features: torch.Tensor, test_targets: torch.Tensor,
                          device: str = 'cuda', epochs: int = 200) -> Dict:
    """训练并评估预测器（改进版）"""
    model = ImprovedGNNPredictor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    test_features = test_features.to(device)
    test_targets = test_targets.to(device)
    
    # 归一化目标值
    target_mean = train_targets.mean()
    target_std = train_targets.std()
    train_targets_norm = (train_targets - target_mean) / (target_std + 1e-8)
    
    # 训练
    model.train()
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(train_features)
        loss = F.mse_loss(predictions, train_targets_norm)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 评估
    model.eval()
    with torch.no_grad():
        predictions_norm = model(test_features)
        predictions = predictions_norm * target_std + target_mean
        
        # 计算MRE (Mean Relative Error)
        relative_errors = torch.abs(predictions - test_targets) / (torch.abs(test_targets) + 1e-8)
        mre = relative_errors.mean().item() * 100
    
    return {
        "MRE (%)": round(mre, 1),
    }


def run_experiment():
    """运行改进版GNN预测器评估实验"""
    print("="*70)
    print("GNN预测器按算子类型和硬件平台评估实验 (改进版)")
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
            
            # 生成更多数据
            train_features, train_targets = generate_realistic_data(op_family, hardware, num_samples=2000)
            test_features, test_targets = generate_realistic_data(op_family, hardware, num_samples=500)
            
            # 训练和评估
            metrics = train_and_evaluate_v2(train_features, train_targets, 
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
            "experiment": "GNN Predictor MRE by Operator Family and Hardware (Improved)",
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
