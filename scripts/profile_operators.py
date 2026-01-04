#!/usr/bin/env python3
"""
Het-Benchmark Operator Profiler

Profile individual operator performance across different input configurations.
Uses the operator implementations in src/operators/.

Usage:
    # Profile all operators
    python profile_operators.py --output_dir ./operator_profiles
    
    # Profile specific operator
    python profile_operators.py --operator Linear --output_dir ./operator_profiles
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.operators import (
    OPERATOR_REGISTRY,
    OperatorConfig,
    LinearOperator,
    Conv2dOperator,
    LayerNormOperator,
    RMSNormOperator,
    GELUOperator,
    SiLUOperator,
    EmbeddingOperator,
)


@dataclass
class OperatorProfile:
    """Profile result for an operator."""
    op_type: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int
    flops: int
    latency_ms: float
    throughput_ops: float
    memory_mb: float
    device: str
    dtype: str
    timestamp: str


# Standard profiling configurations for each operator type
PROFILE_CONFIGS = {
    "Linear": [
        {"in_features": 768, "out_features": 768, "batch_size": 1, "seq_len": 512},
        {"in_features": 768, "out_features": 3072, "batch_size": 1, "seq_len": 512},
        {"in_features": 3072, "out_features": 768, "batch_size": 1, "seq_len": 512},
        {"in_features": 4096, "out_features": 4096, "batch_size": 1, "seq_len": 1024},
        {"in_features": 4096, "out_features": 11008, "batch_size": 1, "seq_len": 1024},
    ],
    "Conv2d": [
        {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "batch_size": 1, "h": 224, "w": 224},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 1, "batch_size": 1, "h": 56, "w": 56},
        {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 2, "batch_size": 1, "h": 28, "w": 28},
        {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 2, "batch_size": 1, "h": 14, "w": 14},
    ],
    "LayerNorm": [
        {"normalized_shape": (768,), "batch_size": 1, "seq_len": 512},
        {"normalized_shape": (1024,), "batch_size": 1, "seq_len": 1024},
        {"normalized_shape": (4096,), "batch_size": 1, "seq_len": 2048},
    ],
    "RMSNorm": [
        {"normalized_shape": (4096,), "batch_size": 1, "seq_len": 512},
        {"normalized_shape": (4096,), "batch_size": 1, "seq_len": 1024},
        {"normalized_shape": (4096,), "batch_size": 1, "seq_len": 2048},
    ],
    "GELU": [
        {"batch_size": 1, "seq_len": 512, "hidden": 3072},
        {"batch_size": 1, "seq_len": 1024, "hidden": 4096},
        {"batch_size": 1, "seq_len": 2048, "hidden": 11008},
    ],
    "SiLU": [
        {"batch_size": 1, "seq_len": 512, "hidden": 3072},
        {"batch_size": 1, "seq_len": 1024, "hidden": 4096},
        {"batch_size": 1, "seq_len": 2048, "hidden": 11008},
    ],
    "Embedding": [
        {"num_embeddings": 32000, "embedding_dim": 4096, "batch_size": 1, "seq_len": 512},
        {"num_embeddings": 50257, "embedding_dim": 768, "batch_size": 1, "seq_len": 1024},
        {"num_embeddings": 152064, "embedding_dim": 3584, "batch_size": 1, "seq_len": 2048},
    ],
}


def create_operator(op_type: str, config: Dict, device: str, dtype: torch.dtype):
    """Create an operator instance from config."""
    op_config = OperatorConfig(
        op_type=op_type,
        device=device,
        dtype=dtype,
        in_features=config.get("in_features", 0),
        out_features=config.get("out_features", 0),
        in_channels=config.get("in_channels", 0),
        out_channels=config.get("out_channels", 0),
        kernel_size=(config.get("kernel_size", 3), config.get("kernel_size", 3)),
        stride=(config.get("stride", 1), config.get("stride", 1)),
        padding=(config.get("padding", 0), config.get("padding", 0)),
        normalized_shape=config.get("normalized_shape", ()),
        num_embeddings=config.get("num_embeddings", 0),
        embedding_dim=config.get("embedding_dim", 0),
    )
    
    op_class = OPERATOR_REGISTRY.get(op_type)
    if op_class is None:
        raise ValueError(f"Unknown operator type: {op_type}")
    
    return op_class(op_config)


def create_input(op_type: str, config: Dict, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create input tensor for profiling."""
    batch_size = config.get("batch_size", 1)
    
    if op_type == "Linear":
        seq_len = config.get("seq_len", 512)
        in_features = config["in_features"]
        return torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
    
    elif op_type == "Conv2d":
        h, w = config.get("h", 224), config.get("w", 224)
        in_channels = config["in_channels"]
        return torch.randn(batch_size, in_channels, h, w, device=device, dtype=dtype)
    
    elif op_type in ["LayerNorm", "RMSNorm"]:
        seq_len = config.get("seq_len", 512)
        hidden = config["normalized_shape"][-1] if isinstance(config["normalized_shape"], tuple) else config["normalized_shape"]
        return torch.randn(batch_size, seq_len, hidden, device=device, dtype=dtype)
    
    elif op_type in ["GELU", "SiLU", "ReLU", "Tanh", "Softmax"]:
        seq_len = config.get("seq_len", 512)
        hidden = config.get("hidden", 768)
        return torch.randn(batch_size, seq_len, hidden, device=device, dtype=dtype)
    
    elif op_type == "Embedding":
        seq_len = config.get("seq_len", 512)
        vocab_size = config["num_embeddings"]
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    else:
        raise ValueError(f"Unknown operator type: {op_type}")


def profile_operator(
    op_type: str, 
    config: Dict, 
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    iterations: int = 100
) -> OperatorProfile:
    """Profile a single operator configuration."""
    
    # Create operator and input
    operator = create_operator(op_type, config, device, dtype)
    x = create_input(op_type, config, device, dtype)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = operator(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = operator(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time_ms = (end_time - start_time) * 1000
    latency_ms = total_time_ms / iterations
    throughput = iterations / (total_time_ms / 1000)
    
    # Memory
    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory_mb = 0.0
    
    # FLOPs
    flops = operator.get_flops(tuple(x.shape))
    
    return OperatorProfile(
        op_type=op_type,
        input_shape=list(x.shape),
        output_shape=list(output.shape),
        parameters=operator.get_parameters(),
        flops=flops,
        latency_ms=latency_ms,
        throughput_ops=throughput,
        memory_mb=memory_mb,
        device=device,
        dtype=str(dtype),
        timestamp=datetime.now().isoformat()
    )


def run_profiling(
    output_dir: str,
    op_types: List[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32
) -> List[OperatorProfile]:
    """Run profiling for all operators."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if op_types is None:
        op_types = list(PROFILE_CONFIGS.keys())
    
    all_results = []
    
    for op_type in op_types:
        if op_type not in PROFILE_CONFIGS:
            print(f"⚠️ No config for {op_type}, skipping")
            continue
        
        print(f"\n📊 Profiling {op_type}...")
        configs = PROFILE_CONFIGS[op_type]
        
        for i, config in enumerate(configs):
            print(f"   Config {i+1}/{len(configs)}: {config}")
            try:
                result = profile_operator(op_type, config, device, dtype)
                all_results.append(result)
                print(f"   ✅ Latency: {result.latency_ms:.3f}ms, FLOPs: {result.flops:,}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Save results
    json_path = os.path.join(output_dir, "operator_profiles.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n✅ Results saved to: {json_path}")
    
    # Generate summary table
    summary_path = os.path.join(output_dir, "operator_profiles_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Operator Profiling Summary\n\n")
        f.write(f"**Device:** {device}\n")
        f.write(f"**Dtype:** {dtype}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("| Operator | Input Shape | Parameters | FLOPs | Latency (ms) | Memory (MB) |\n")
        f.write("|----------|-------------|------------|-------|--------------|-------------|\n")
        
        for r in all_results:
            f.write(f"| {r.op_type} | {r.input_shape} | {r.parameters:,} | "
                   f"{r.flops:,} | {r.latency_ms:.3f} | {r.memory_mb:.1f} |\n")
    
    print(f"✅ Summary saved to: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Profile Het-Benchmark operators")
    parser.add_argument("--output_dir", type=str, default="./operator_profiles",
                        help="Directory to save profiling results")
    parser.add_argument("--operator", type=str, default=None,
                        help="Profile specific operator only")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to profile on (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for profiling")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    op_types = [args.operator] if args.operator else None
    
    print("🚀 Het-Benchmark Operator Profiler")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    run_profiling(args.output_dir, op_types, device, dtype)


if __name__ == "__main__":
    main()
