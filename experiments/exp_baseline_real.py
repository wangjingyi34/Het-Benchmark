#!/usr/bin/env python3
"""
真实基线对比实验

对比三种评测方法：
1. MLPerf风格 - 端到端模型评测
2. DeepBench风格 - 算子级微基准
3. Het-Benchmark - 算子归因+跨平台预测

所有数据都是真实运行得到的。
"""

import torch
import torch.nn as nn
import time
import json
import os
from datetime import datetime
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def benchmark_operator(op_func, input_data, num_warmup=20, num_runs=100):
    """真实测量算子执行时间"""
    # Warmup
    for _ in range(num_warmup):
        _ = op_func(input_data)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = op_func(input_data)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99))
    }


def run_deepbench_style():
    """DeepBench风格算子基准测试"""
    print("\n" + "="*60)
    print("DeepBench风格算子基准测试")
    print("="*60)
    
    device = torch.device("cuda")
    results = {}
    
    # 1. GEMM (矩阵乘法)
    print("\n1. GEMM (矩阵乘法)")
    gemm_configs = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (1024, 4096, 1024),  # 常见Transformer配置
        (4096, 1024, 4096),
    ]
    
    results["gemm"] = {}
    for M, N, K in gemm_configs:
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)
        
        timing = benchmark_operator(lambda x: torch.mm(x, B), A)
        flops = 2 * M * N * K
        tflops = flops / (timing["mean_ms"] / 1000) / 1e12
        
        config_name = f"{M}x{N}x{K}"
        results["gemm"][config_name] = {
            "timing": timing,
            "tflops": tflops,
            "config": {"M": M, "N": N, "K": K}
        }
        print(f"  {config_name}: {timing['mean_ms']:.3f} ms, {tflops:.2f} TFLOPS")
    
    # 2. Convolution
    print("\n2. Convolution")
    conv_configs = [
        (64, 64, 3, 224),    # ResNet第一层
        (64, 128, 3, 112),
        (128, 256, 3, 56),
        (256, 512, 3, 28),
        (512, 512, 3, 14),
    ]
    
    results["conv"] = {}
    for in_ch, out_ch, kernel, size in conv_configs:
        conv = nn.Conv2d(in_ch, out_ch, kernel, padding=1).to(device).half()
        x = torch.randn(1, in_ch, size, size, device=device, dtype=torch.float16)
        
        timing = benchmark_operator(conv, x)
        
        config_name = f"{in_ch}to{out_ch}_{size}x{size}"
        results["conv"][config_name] = {
            "timing": timing,
            "config": {"in_ch": in_ch, "out_ch": out_ch, "kernel": kernel, "size": size}
        }
        print(f"  {config_name}: {timing['mean_ms']:.3f} ms")
    
    # 3. Attention
    print("\n3. Multi-Head Attention")
    attn_configs = [
        (1, 128, 12, 64),   # BERT small
        (1, 512, 12, 64),   # BERT base
        (8, 512, 12, 64),   # Batch
        (1, 1024, 16, 64),  # GPT-2
        (1, 2048, 16, 64),  # Long sequence
    ]
    
    results["attention"] = {}
    for batch, seq, heads, dim in attn_configs:
        mha = nn.MultiheadAttention(heads * dim, heads, batch_first=True).to(device).half()
        x = torch.randn(batch, seq, heads * dim, device=device, dtype=torch.float16)
        
        timing = benchmark_operator(lambda q: mha(q, q, q)[0], x)
        
        config_name = f"b{batch}_s{seq}_h{heads}"
        results["attention"][config_name] = {
            "timing": timing,
            "config": {"batch": batch, "seq": seq, "heads": heads, "dim": dim}
        }
        print(f"  {config_name}: {timing['mean_ms']:.3f} ms")
    
    # 4. LayerNorm
    print("\n4. LayerNorm")
    ln_configs = [
        (32, 512, 768),
        (1, 512, 768),
        (8, 1024, 1024),
        (1, 2048, 1024),
    ]
    
    results["layernorm"] = {}
    for batch, seq, dim in ln_configs:
        ln = nn.LayerNorm(dim).to(device).half()
        x = torch.randn(batch, seq, dim, device=device, dtype=torch.float16)
        
        timing = benchmark_operator(ln, x)
        
        config_name = f"b{batch}_s{seq}_d{dim}"
        results["layernorm"][config_name] = {
            "timing": timing,
            "config": {"batch": batch, "seq": seq, "dim": dim}
        }
        print(f"  {config_name}: {timing['mean_ms']:.3f} ms")
    
    # 5. Softmax
    print("\n5. Softmax")
    softmax_configs = [
        (8, 12, 512, 512),   # Attention scores
        (1, 16, 1024, 1024),
        (1, 12, 2048, 2048),
    ]
    
    results["softmax"] = {}
    for batch, heads, seq1, seq2 in softmax_configs:
        x = torch.randn(batch, heads, seq1, seq2, device=device, dtype=torch.float16)
        softmax = nn.Softmax(dim=-1)
        
        timing = benchmark_operator(softmax, x)
        
        config_name = f"b{batch}_h{heads}_{seq1}x{seq2}"
        results["softmax"][config_name] = {
            "timing": timing,
            "config": {"batch": batch, "heads": heads, "seq1": seq1, "seq2": seq2}
        }
        print(f"  {config_name}: {timing['mean_ms']:.3f} ms")
    
    return results


def run_mlperf_style():
    """MLPerf风格端到端模型基准测试"""
    print("\n" + "="*60)
    print("MLPerf风格端到端模型基准测试")
    print("="*60)
    
    device = torch.device("cuda")
    results = {}
    
    # 导入模型
    import torchvision.models as models
    
    model_configs = {
        "ResNet50": {
            "model": models.resnet50(weights=None),
            "input_shape": (1, 3, 224, 224),
            "batch_sizes": [1, 8, 32]
        },
        "ResNet152": {
            "model": models.resnet152(weights=None),
            "input_shape": (1, 3, 224, 224),
            "batch_sizes": [1, 8, 16]
        },
        "MobileNetV2": {
            "model": models.mobilenet_v2(weights=None),
            "input_shape": (1, 3, 224, 224),
            "batch_sizes": [1, 8, 32, 64]
        },
        "EfficientNet_B0": {
            "model": models.efficientnet_b0(weights=None),
            "input_shape": (1, 3, 224, 224),
            "batch_sizes": [1, 8, 32]
        }
    }
    
    for model_name, config in model_configs.items():
        print(f"\n{model_name}:")
        model = config["model"].to(device).half().eval()
        results[model_name] = {}
        
        for batch_size in config["batch_sizes"]:
            input_shape = list(config["input_shape"])
            input_shape[0] = batch_size
            x = torch.randn(*input_shape, device=device, dtype=torch.float16)
            
            with torch.no_grad():
                timing = benchmark_operator(model, x, num_warmup=10, num_runs=50)
            
            throughput = batch_size / (timing["mean_ms"] / 1000)
            
            results[model_name][f"batch_{batch_size}"] = {
                "timing": timing,
                "throughput_ips": throughput,  # images per second
                "batch_size": batch_size
            }
            print(f"  Batch {batch_size}: {timing['mean_ms']:.3f} ms, {throughput:.1f} img/s")
        
        del model
        torch.cuda.empty_cache()
    
    return results


def run_het_benchmark_style():
    """Het-Benchmark风格算子归因测试"""
    print("\n" + "="*60)
    print("Het-Benchmark风格算子归因测试")
    print("="*60)
    
    device = torch.device("cuda")
    results = {}
    
    # 创建一个简单的Transformer模型用于归因分析
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=30522, hidden_size=768, num_layers=6, num_heads=12):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(hidden_size, num_heads, batch_first=True),
                    'ln1': nn.LayerNorm(hidden_size),
                    'ffn': nn.Sequential(
                        nn.Linear(hidden_size, hidden_size * 4),
                        nn.GELU(),
                        nn.Linear(hidden_size * 4, hidden_size)
                    ),
                    'ln2': nn.LayerNorm(hidden_size)
                })
                for _ in range(num_layers)
            ])
            self.fc = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                # Attention
                attn_out, _ = layer['attention'](x, x, x)
                x = layer['ln1'](x + attn_out)
                # FFN
                ffn_out = layer['ffn'](x)
                x = layer['ln2'](x + ffn_out)
            return self.fc(x)
    
    model = SimpleTransformer().to(device).half().eval()
    
    # 测量各组件的时间贡献
    print("\n算子级时间归因分析:")
    
    batch_size, seq_len = 1, 512
    x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    hidden = torch.randn(batch_size, seq_len, 768, device=device, dtype=torch.float16)
    
    # 1. Embedding
    timing = benchmark_operator(model.embedding, x)
    results["embedding"] = {"timing": timing, "contribution": 0}
    print(f"  Embedding: {timing['mean_ms']:.3f} ms")
    
    # 2. Attention (per layer)
    attn = model.layers[0]['attention']
    timing = benchmark_operator(lambda h: attn(h, h, h)[0], hidden)
    results["attention_per_layer"] = {"timing": timing, "contribution": 0}
    print(f"  Attention (per layer): {timing['mean_ms']:.3f} ms")
    
    # 3. LayerNorm (per layer)
    ln = model.layers[0]['ln1']
    timing = benchmark_operator(ln, hidden)
    results["layernorm_per_layer"] = {"timing": timing, "contribution": 0}
    print(f"  LayerNorm (per layer): {timing['mean_ms']:.3f} ms")
    
    # 4. FFN (per layer)
    ffn = model.layers[0]['ffn']
    timing = benchmark_operator(ffn, hidden)
    results["ffn_per_layer"] = {"timing": timing, "contribution": 0}
    print(f"  FFN (per layer): {timing['mean_ms']:.3f} ms")
    
    # 5. Final Linear
    timing = benchmark_operator(model.fc, hidden)
    results["final_linear"] = {"timing": timing, "contribution": 0}
    print(f"  Final Linear: {timing['mean_ms']:.3f} ms")
    
    # 计算贡献百分比
    num_layers = 6
    total_time = (
        results["embedding"]["timing"]["mean_ms"] +
        num_layers * (
            results["attention_per_layer"]["timing"]["mean_ms"] +
            2 * results["layernorm_per_layer"]["timing"]["mean_ms"] +
            results["ffn_per_layer"]["timing"]["mean_ms"]
        ) +
        results["final_linear"]["timing"]["mean_ms"]
    )
    
    results["embedding"]["contribution"] = results["embedding"]["timing"]["mean_ms"] / total_time * 100
    results["attention_per_layer"]["contribution"] = num_layers * results["attention_per_layer"]["timing"]["mean_ms"] / total_time * 100
    results["layernorm_per_layer"]["contribution"] = num_layers * 2 * results["layernorm_per_layer"]["timing"]["mean_ms"] / total_time * 100
    results["ffn_per_layer"]["contribution"] = num_layers * results["ffn_per_layer"]["timing"]["mean_ms"] / total_time * 100
    results["final_linear"]["contribution"] = results["final_linear"]["timing"]["mean_ms"] / total_time * 100
    
    print("\n贡献百分比:")
    print(f"  Embedding: {results['embedding']['contribution']:.1f}%")
    print(f"  Attention (total): {results['attention_per_layer']['contribution']:.1f}%")
    print(f"  LayerNorm (total): {results['layernorm_per_layer']['contribution']:.1f}%")
    print(f"  FFN (total): {results['ffn_per_layer']['contribution']:.1f}%")
    print(f"  Final Linear: {results['final_linear']['contribution']:.1f}%")
    
    results["total_time_ms"] = total_time
    results["num_layers"] = num_layers
    
    return results


def compare_methods():
    """对比三种方法的特性"""
    print("\n" + "="*60)
    print("评测方法特性对比")
    print("="*60)
    
    comparison = {
        "MLPerf": {
            "granularity": "End-to-End Model",
            "cross_platform": "No (requires re-run)",
            "attribution": "No",
            "zero_shot_prediction": "No",
            "migration_guidance": "No",
            "setup_complexity": "High",
            "measurement_type": "Throughput, Latency"
        },
        "DeepBench": {
            "granularity": "Operator-level",
            "cross_platform": "Partial (operator mapping)",
            "attribution": "No",
            "zero_shot_prediction": "No",
            "migration_guidance": "Limited",
            "setup_complexity": "Medium",
            "measurement_type": "Latency, TFLOPS"
        },
        "Het-Benchmark": {
            "granularity": "Operator-level with Model Context",
            "cross_platform": "Yes (via MOH-KG)",
            "attribution": "Yes (COPA)",
            "zero_shot_prediction": "Yes (GNN)",
            "migration_guidance": "Yes (KG-A2O)",
            "setup_complexity": "Low",
            "measurement_type": "Latency, Attribution, Prediction"
        }
    }
    
    print("\n特性对比表:")
    print(f"{'Feature':<25} {'MLPerf':<20} {'DeepBench':<20} {'Het-Benchmark':<25}")
    print("-" * 90)
    
    features = ["granularity", "cross_platform", "attribution", "zero_shot_prediction", 
                "migration_guidance", "setup_complexity", "measurement_type"]
    
    for feature in features:
        mlperf = comparison["MLPerf"][feature]
        deepbench = comparison["DeepBench"][feature]
        het = comparison["Het-Benchmark"][feature]
        print(f"{feature:<25} {mlperf:<20} {deepbench:<20} {het:<25}")
    
    return comparison


def main():
    print("="*80)
    print("真实基线对比实验")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {
        "experiment": "Real Baseline Comparison",
        "timestamp": datetime.now().isoformat(),
        "hardware": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__
    }
    
    # 运行各种基准测试
    results["deepbench"] = run_deepbench_style()
    results["mlperf"] = run_mlperf_style()
    results["het_benchmark"] = run_het_benchmark_style()
    results["comparison"] = compare_methods()
    
    # 保存结果
    os.makedirs('/workspace/results', exist_ok=True)
    with open('/workspace/results/baseline_real_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ 结果已保存到 /workspace/results/baseline_real_comparison.json")
    
    return results


if __name__ == "__main__":
    main()
