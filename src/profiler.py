"""
Het-Benchmark Profiler
Performance profiling for operators on different hardware platforms
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from loguru import logger
from enum import Enum
import gc


class MetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time_ms"
    MEMORY_USAGE = "memory_usage_mb"
    THROUGHPUT = "throughput_ops_per_sec"
    LATENCY_P50 = "latency_p50_ms"
    LATENCY_P99 = "latency_p99_ms"
    FLOPS = "flops"
    MEMORY_BANDWIDTH = "memory_bandwidth_gb_s"


@dataclass
class ProfileResult:
    """Result of profiling an operator"""
    operator_id: str
    operator_type: str
    hardware_id: str
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    
    # Performance metrics
    execution_time_ms: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    latency_p50_ms: float
    latency_p99_ms: float
    
    # Additional info
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OperatorBenchmark:
    """Benchmark configuration for an operator"""
    op_type: str
    op_func: Callable
    input_generator: Callable
    input_shapes: List[List[int]]
    attributes: Dict[str, Any] = field(default_factory=dict)


class Profiler:
    """
    Performance profiler for Het-Benchmark
    
    Measures execution time, memory usage, and throughput for operators
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        # Get hardware info
        self.hardware_id = self._get_hardware_id()
        
        logger.info(f"Profiler initialized on {self.hardware_id}")
    
    def _get_hardware_id(self) -> str:
        """Get hardware identifier"""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return f"cuda_{gpu_name.replace(' ', '_')}"
        else:
            import platform
            return f"cpu_{platform.processor()}"
    
    def _synchronize(self):
        """Synchronize device for accurate timing"""
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """
        Profile a function's execution
        
        Returns:
            Dictionary with timing and memory metrics
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        self._synchronize()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Measure memory before
        mem_before = self._get_memory_usage()
        
        # Benchmark
        latencies = []
        
        for _ in range(self.benchmark_iterations):
            self._synchronize()
            start = time.perf_counter()
            
            func(*args, **kwargs)
            
            self._synchronize()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Measure memory after
        mem_after = self._get_memory_usage()
        
        latencies = np.array(latencies)
        
        return {
            "execution_time_ms": float(np.mean(latencies)),
            "memory_usage_mb": float(mem_after - mem_before),
            "throughput_ops_per_sec": float(1000 / np.mean(latencies)),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_std_ms": float(np.std(latencies)),
            "latency_min_ms": float(np.min(latencies)),
            "latency_max_ms": float(np.max(latencies)),
        }
    
    def profile_operator(
        self,
        benchmark: OperatorBenchmark,
        operator_id: Optional[str] = None,
    ) -> ProfileResult:
        """
        Profile a single operator
        
        Args:
            benchmark: Operator benchmark configuration
            operator_id: Optional operator ID
            
        Returns:
            ProfileResult with performance metrics
        """
        if operator_id is None:
            operator_id = f"{benchmark.op_type}_{hash(str(benchmark.input_shapes))}"
        
        # Generate inputs
        inputs = benchmark.input_generator(benchmark.input_shapes, self.device)
        
        # Profile
        metrics = self.profile_function(benchmark.op_func, *inputs)
        
        # Get output shapes
        with torch.no_grad():
            output = benchmark.op_func(*inputs)
            if isinstance(output, torch.Tensor):
                output_shapes = [list(output.shape)]
            elif isinstance(output, (tuple, list)):
                output_shapes = [list(o.shape) for o in output if isinstance(o, torch.Tensor)]
            else:
                output_shapes = []
        
        result = ProfileResult(
            operator_id=operator_id,
            operator_type=benchmark.op_type,
            hardware_id=self.hardware_id,
            input_shapes=benchmark.input_shapes,
            output_shapes=output_shapes,
            execution_time_ms=metrics["execution_time_ms"],
            memory_usage_mb=metrics["memory_usage_mb"],
            throughput_ops_per_sec=metrics["throughput_ops_per_sec"],
            latency_p50_ms=metrics["latency_p50_ms"],
            latency_p99_ms=metrics["latency_p99_ms"],
            warmup_iterations=self.warmup_iterations,
            benchmark_iterations=self.benchmark_iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        return result


class OperatorProfiler(Profiler):
    """
    Specialized profiler for common deep learning operators
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define operator benchmarks
        self.operator_benchmarks = self._define_operators()
    
    def _define_operators(self) -> Dict[str, Callable]:
        """Define benchmark functions for common operators"""
        
        def matmul_benchmark(shapes, device):
            M, K = shapes[0]
            K2, N = shapes[1]
            assert K == K2, "Matrix dimensions must match"
            a = torch.randn(M, K, device=device, dtype=torch.float16)
            b = torch.randn(K, N, device=device, dtype=torch.float16)
            return (a, b)
        
        def matmul_func(a, b):
            return torch.matmul(a, b)
        
        def conv2d_benchmark(shapes, device):
            batch, in_channels, H, W = shapes[0]
            out_channels, _, kH, kW = shapes[1]
            x = torch.randn(batch, in_channels, H, W, device=device, dtype=torch.float16)
            weight = torch.randn(out_channels, in_channels, kH, kW, device=device, dtype=torch.float16)
            return (x, weight)
        
        def conv2d_func(x, weight):
            return torch.nn.functional.conv2d(x, weight, padding=1)
        
        def layernorm_benchmark(shapes, device):
            x = torch.randn(*shapes[0], device=device, dtype=torch.float16)
            normalized_shape = shapes[0][-1:]
            return (x, normalized_shape)
        
        def layernorm_func(x, normalized_shape):
            return torch.nn.functional.layer_norm(x, normalized_shape)
        
        def gelu_benchmark(shapes, device):
            x = torch.randn(*shapes[0], device=device, dtype=torch.float16)
            return (x,)
        
        def gelu_func(x):
            return torch.nn.functional.gelu(x)
        
        def softmax_benchmark(shapes, device):
            x = torch.randn(*shapes[0], device=device, dtype=torch.float16)
            return (x,)
        
        def softmax_func(x):
            return torch.nn.functional.softmax(x, dim=-1)
        
        def attention_benchmark(shapes, device):
            batch, seq_len, hidden = shapes[0]
            num_heads = shapes[1][0]
            head_dim = hidden // num_heads
            
            q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            return (q, k, v)
        
        def attention_func(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        def embedding_benchmark(shapes, device):
            batch, seq_len = shapes[0]
            vocab_size, embed_dim = shapes[1]
            indices = torch.randint(0, vocab_size, (batch, seq_len), device=device)
            weight = torch.randn(vocab_size, embed_dim, device=device, dtype=torch.float16)
            return (indices, weight)
        
        def embedding_func(indices, weight):
            return torch.nn.functional.embedding(indices, weight)
        
        return {
            "MatMul": (matmul_benchmark, matmul_func),
            "Conv": (conv2d_benchmark, conv2d_func),
            "LayerNorm": (layernorm_benchmark, layernorm_func),
            "Gelu": (gelu_benchmark, gelu_func),
            "Softmax": (softmax_benchmark, softmax_func),
            "Attention": (attention_benchmark, attention_func),
            "Embedding": (embedding_benchmark, embedding_func),
        }
    
    def profile_all_operators(
        self,
        shapes_config: Optional[Dict[str, List[List[int]]]] = None,
    ) -> List[ProfileResult]:
        """
        Profile all defined operators with various input shapes
        
        Args:
            shapes_config: Optional custom shapes configuration
            
        Returns:
            List of ProfileResult for all operators
        """
        if shapes_config is None:
            # Default shapes for benchmarking
            shapes_config = {
                "MatMul": [
                    [[1024, 1024], [1024, 1024]],
                    [[2048, 2048], [2048, 2048]],
                    [[4096, 4096], [4096, 4096]],
                    [[1, 4096, 4096], [4096, 4096]],  # Batch matmul
                ],
                "Conv": [
                    [[1, 64, 224, 224], [128, 64, 3, 3]],
                    [[1, 128, 112, 112], [256, 128, 3, 3]],
                    [[1, 256, 56, 56], [512, 256, 3, 3]],
                ],
                "LayerNorm": [
                    [[1, 512, 768]],
                    [[1, 1024, 1024]],
                    [[1, 2048, 4096]],
                ],
                "Gelu": [
                    [[1, 512, 768]],
                    [[1, 1024, 4096]],
                    [[1, 2048, 8192]],
                ],
                "Softmax": [
                    [[1, 12, 512, 512]],
                    [[1, 32, 1024, 1024]],
                    [[1, 64, 2048, 2048]],
                ],
                "Attention": [
                    [[1, 512, 768], [12]],
                    [[1, 1024, 1024], [16]],
                    [[1, 2048, 4096], [32]],
                ],
                "Embedding": [
                    [[1, 512], [32000, 4096]],
                    [[1, 1024], [50000, 4096]],
                    [[1, 2048], [100000, 4096]],
                ],
            }
        
        results = []
        
        for op_type, shapes_list in shapes_config.items():
            if op_type not in self.operator_benchmarks:
                logger.warning(f"Operator {op_type} not defined, skipping")
                continue
            
            input_gen, op_func = self.operator_benchmarks[op_type]
            
            for i, shapes in enumerate(shapes_list):
                logger.info(f"Profiling {op_type} with shapes {shapes}")
                
                try:
                    benchmark = OperatorBenchmark(
                        op_type=op_type,
                        op_func=op_func,
                        input_generator=input_gen,
                        input_shapes=shapes,
                    )
                    
                    result = self.profile_operator(
                        benchmark,
                        operator_id=f"{op_type}_{i}",
                    )
                    results.append(result)
                    
                    logger.info(f"  Time: {result.execution_time_ms:.3f}ms, "
                               f"Throughput: {result.throughput_ops_per_sec:.1f} ops/s")
                    
                except Exception as e:
                    logger.error(f"Error profiling {op_type}: {e}")
                
                # Clean up
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        return results
    
    def save_results(self, results: List[ProfileResult], path: str):
        """Save profiling results to JSON file"""
        data = {
            "hardware_id": self.hardware_id,
            "device": self.device,
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations,
            "results": [r.to_dict() for r in results],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(results)} profiling results to {path}")


class ModelProfiler(Profiler):
    """
    Profiler for end-to-end model inference
    """
    
    def profile_model_inference(
        self,
        model: torch.nn.Module,
        input_generator: Callable,
        model_id: str,
    ) -> Dict[str, Any]:
        """
        Profile end-to-end model inference
        
        Args:
            model: PyTorch model
            input_generator: Function to generate model inputs
            model_id: Model identifier
            
        Returns:
            Dictionary with profiling results
        """
        model = model.to(self.device)
        model.eval()
        
        # Generate input
        inputs = input_generator(self.device)
        
        # Profile
        with torch.no_grad():
            metrics = self.profile_function(model, *inputs)
        
        return {
            "model_id": model_id,
            "hardware_id": self.hardware_id,
            **metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


if __name__ == "__main__":
    # Test profiler
    profiler = OperatorProfiler(
        warmup_iterations=5,
        benchmark_iterations=50,
    )
    
    # Profile all operators
    results = profiler.profile_all_operators()
    
    # Save results
    profiler.save_results(results, "/workspace/het-benchmark/results/operator_profile.json")
    
    print(f"\n=== Profiling Complete ===")
    print(f"Hardware: {profiler.hardware_id}")
    print(f"Operators profiled: {len(results)}")
    
    for result in results:
        print(f"\n{result.operator_type}:")
        print(f"  Input shapes: {result.input_shapes}")
        print(f"  Execution time: {result.execution_time_ms:.3f} ms")
        print(f"  Throughput: {result.throughput_ops_per_sec:.1f} ops/s")
        print(f"  Memory: {result.memory_usage_mb:.2f} MB")
