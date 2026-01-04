#!/usr/bin/env python3
"""
实验: 代理模型 vs 完整模型评估时间对比

在A100 GPU上真实测量：
1. 完整模型执行时间
2. 代理模型（查表）评估时间
3. 计算加速比
"""

import torch
import torch.nn as nn
import time
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class OperatorProfiler:
    """算子性能分析器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.profile_cache = {}  # 缓存的算子延迟
    
    def profile_operator(self, op: nn.Module, input_shape: Tuple, 
                        warmup: int = 10, iterations: int = 100) -> float:
        """测量单个算子的延迟"""
        op = op.to(self.device)
        op.eval()
        
        input_tensor = torch.randn(*input_shape, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = op(input_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # 测量
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = op(input_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        return (end - start) / iterations * 1000  # 毫秒
    
    def build_profile_cache(self, operators: List[Tuple[str, nn.Module, Tuple]]):
        """构建算子性能缓存"""
        print("构建算子性能缓存...")
        for name, op, input_shape in operators:
            latency = self.profile_operator(op, input_shape)
            self.profile_cache[name] = latency
            print(f"  {name}: {latency:.4f}ms")
    
    def surrogate_estimate(self, operator_names: List[str]) -> float:
        """代理模型估计：通过查表获取延迟"""
        total = 0.0
        for name in operator_names:
            if name in self.profile_cache:
                total += self.profile_cache[name]
        return total


class SimpleModel(nn.Module):
    """简单的神经网络模型"""
    
    def __init__(self, n_layers: int, d_model: int = 512):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_names = []
        
        for i in range(n_layers):
            # 每层包含: Linear -> LayerNorm -> GELU
            self.layers.append(nn.Linear(d_model, d_model))
            self.layer_names.append(f"linear_{i}")
            
            self.layers.append(nn.LayerNorm(d_model))
            self.layer_names.append(f"layernorm_{i}")
            
            self.layers.append(nn.GELU())
            self.layer_names.append(f"gelu_{i}")
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, n_layers: int, d_model: int = 512, nhead: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model
    
    def forward(self, x):
        return self.transformer(x)


def measure_full_model_time(model: nn.Module, input_tensor: torch.Tensor,
                           warmup: int = 10, iterations: int = 100) -> float:
    """测量完整模型执行时间"""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # 毫秒


def measure_surrogate_time(profiler: OperatorProfiler, 
                          operator_names: List[str],
                          iterations: int = 10000) -> float:
    """测量代理模型（查表）时间"""
    start = time.perf_counter()
    
    for _ in range(iterations):
        _ = profiler.surrogate_estimate(operator_names)
    
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # 毫秒


def run_experiment(device: str = 'cuda'):
    """运行代理模型加速比实验"""
    print("="*70)
    print("代理模型 vs 完整模型评估时间对比实验")
    print("="*70)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 实验配置
    model_configs = [
        {"name": "Small MLP", "type": "mlp", "layers": 4, "d_model": 256},
        {"name": "Medium MLP", "type": "mlp", "layers": 12, "d_model": 512},
        {"name": "Large MLP", "type": "mlp", "layers": 24, "d_model": 768},
        {"name": "Small Transformer", "type": "transformer", "layers": 2, "d_model": 256},
        {"name": "Medium Transformer", "type": "transformer", "layers": 6, "d_model": 512},
        {"name": "Large Transformer", "type": "transformer", "layers": 12, "d_model": 768},
    ]
    
    batch_size = 32
    seq_len = 128
    
    results = []
    profiler = OperatorProfiler(device)
    
    for config in model_configs:
        print(f"\n{'='*70}")
        print(f"模型: {config['name']}")
        print("="*70)
        
        d_model = config['d_model']
        
        # 创建模型
        if config['type'] == 'mlp':
            model = SimpleModel(config['layers'], d_model).to(device)
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            n_operators = config['layers'] * 3  # Linear + LayerNorm + GELU
        else:
            model = TransformerModel(config['layers'], d_model).to(device)
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            # Transformer每层约有: 2*Linear(Q,K,V) + Attention + LayerNorm + FFN(2*Linear) + LayerNorm
            n_operators = config['layers'] * 8
        
        # 构建算子列表用于代理模型
        operator_names = []
        if config['type'] == 'mlp':
            for i in range(config['layers']):
                operator_names.extend([f"linear_{i}", f"layernorm_{i}", f"gelu_{i}"])
        else:
            for i in range(config['layers']):
                operator_names.extend([
                    f"qkv_proj_{i}", f"attention_{i}", f"out_proj_{i}",
                    f"norm1_{i}", f"ff1_{i}", f"ff2_{i}", f"norm2_{i}", f"dropout_{i}"
                ])
        
        # 构建性能缓存（如果还没有）
        if not profiler.profile_cache:
            # 为常见算子类型构建缓存
            common_ops = [
                ("linear", nn.Linear(d_model, d_model), (batch_size, seq_len, d_model)),
                ("layernorm", nn.LayerNorm(d_model), (batch_size, seq_len, d_model)),
                ("gelu", nn.GELU(), (batch_size, seq_len, d_model)),
                ("attention", nn.MultiheadAttention(d_model, 8, batch_first=True), None),
                ("dropout", nn.Dropout(0.1), (batch_size, seq_len, d_model)),
            ]
            
            for name, op, shape in common_ops:
                if shape is not None:
                    latency = profiler.profile_operator(op, shape)
                    profiler.profile_cache[name] = latency
                    print(f"  缓存 {name}: {latency:.4f}ms")
        
        # 为所有算子名称填充缓存
        for name in operator_names:
            if name not in profiler.profile_cache:
                # 根据名称推断类型
                if 'linear' in name or 'proj' in name or 'ff' in name:
                    profiler.profile_cache[name] = profiler.profile_cache.get('linear', 0.05)
                elif 'norm' in name:
                    profiler.profile_cache[name] = profiler.profile_cache.get('layernorm', 0.01)
                elif 'gelu' in name:
                    profiler.profile_cache[name] = profiler.profile_cache.get('gelu', 0.005)
                elif 'attention' in name:
                    profiler.profile_cache[name] = profiler.profile_cache.get('attention', 0.5)
                elif 'dropout' in name:
                    profiler.profile_cache[name] = profiler.profile_cache.get('dropout', 0.002)
                else:
                    profiler.profile_cache[name] = 0.01
        
        # 测量完整模型时间
        print(f"\n测量完整模型执行时间...")
        full_model_time = measure_full_model_time(model, input_tensor)
        print(f"  完整模型时间: {full_model_time:.4f}ms")
        
        # 测量代理模型时间
        print(f"测量代理模型（查表）时间...")
        surrogate_time = measure_surrogate_time(profiler, operator_names)
        print(f"  代理模型时间: {surrogate_time:.6f}ms")
        
        # 计算加速比
        speedup = full_model_time / surrogate_time
        print(f"  加速比: {speedup:.0f}×")
        
        results.append({
            "Model": config['name'],
            "Type": config['type'],
            "Layers": config['layers'],
            "Operators": n_operators,
            "Full Model Time (ms)": round(full_model_time, 4),
            "Surrogate Time (ms)": round(surrogate_time, 6),
            "Speedup": f"{speedup:.0f}×"
        })
    
    # 保存结果
    results_dir = '/results' if os.path.exists('/results') else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'surrogate_speedup_results.json'), 'w') as f:
        json.dump({
            "experiment": "Surrogate vs Full Model Evaluation Time",
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else "N/A",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("Table 6: Surrogate vs. Full-Model Evaluation Time")
    print("="*70)
    print(f"{'Model':<25} {'Operators':<12} {'Full (ms)':<15} {'Surrogate (ms)':<18} {'Speedup'}")
    print("-"*80)
    
    for r in results:
        print(f"{r['Model']:<25} {r['Operators']:<12} {r['Full Model Time (ms)']:<15} "
              f"{r['Surrogate Time (ms)']:<18} {r['Speedup']}")
    
    print(f"\n✅ 结果已保存到 {results_dir}/surrogate_speedup_results.json")
    
    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_experiment(device)
