"""
COPA: Cross-platform Operator Performance Attribution
Two-Stage Performance Attribution Algorithm

Stage I: Micro-benchmarking (Operator-level profiling)
Stage II: Model-level Attribution (Shapley value calculation)

Based on the paper's Algorithm 1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from itertools import combinations
from collections import defaultdict
import json
import math
import time
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operator"""
    execution_time_ms: float
    memory_usage_bytes: int
    throughput_ops: float  # Operations per second
    latency_p50_ms: float
    latency_p99_ms: float
    power_consumption_w: float = 0.0
    flops: float = 0.0  # Floating point operations
    memory_bandwidth_gbps: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_bytes": self.memory_usage_bytes,
            "throughput_ops": self.throughput_ops,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "power_consumption_w": self.power_consumption_w,
            "flops": self.flops,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
        }


@dataclass
class MicroBenchmarkResult:
    """Result from Stage I micro-benchmarking"""
    operator_id: str
    operator_type: str
    input_shape: Tuple
    output_shape: Tuple
    hardware_platform: str
    
    # Timing metrics (averaged over multiple runs)
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Throughput metrics
    throughput_ops_per_sec: float
    
    # Memory metrics
    peak_memory_bytes: int
    memory_bandwidth_utilized: float  # Percentage of theoretical max
    
    # Compute metrics
    compute_utilization: float  # Percentage of theoretical FLOPS
    
    # Roofline model metrics
    arithmetic_intensity: float  # FLOPS per byte
    is_compute_bound: bool
    is_memory_bound: bool
    
    # Raw measurements
    raw_latencies_ms: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "operator_id": self.operator_id,
            "operator_type": self.operator_type,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "hardware_platform": self.hardware_platform,
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_bandwidth_utilized": self.memory_bandwidth_utilized,
            "compute_utilization": self.compute_utilization,
            "arithmetic_intensity": self.arithmetic_intensity,
            "is_compute_bound": self.is_compute_bound,
            "is_memory_bound": self.is_memory_bound,
        }


@dataclass
class AttributionResult:
    """Result of Shapley value attribution"""
    operator_id: str
    operator_type: str
    shapley_value: float
    contribution_ratio: float  # Percentage of total
    rank: int
    metrics: PerformanceMetrics
    micro_benchmark: Optional[MicroBenchmarkResult] = None
    is_bottleneck: bool = False
    bottleneck_type: str = ""  # "compute", "memory", "communication"
    optimization_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = {
            "operator_id": self.operator_id,
            "operator_type": self.operator_type,
            "shapley_value": self.shapley_value,
            "contribution_ratio": self.contribution_ratio,
            "rank": self.rank,
            "is_bottleneck": self.is_bottleneck,
            "bottleneck_type": self.bottleneck_type,
            "optimization_suggestions": self.optimization_suggestions,
            "metrics": self.metrics.to_dict(),
        }
        if self.micro_benchmark:
            result["micro_benchmark"] = self.micro_benchmark.to_dict()
        return result


class MicroBenchmarker:
    """
    Stage I: Micro-benchmarking
    
    Performs isolated operator-level performance profiling to collect
    fine-grained performance metrics for each operator type.
    """
    
    def __init__(
        self,
        hardware_platform: str = "CUDA",
        warmup_iterations: int = 10,
        measurement_iterations: int = 100,
        use_cuda_events: bool = True
    ):
        self.hardware_platform = hardware_platform
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.use_cuda_events = use_cuda_events
        
        # Hardware specifications for roofline model
        self.hardware_specs = self._get_hardware_specs(hardware_platform)
    
    def _get_hardware_specs(self, platform: str) -> Dict:
        """Get hardware specifications for roofline model"""
        specs = {
            "CUDA": {
                "peak_flops_tflops": 312.0,  # A100 FP16 Tensor Core
                "peak_memory_bandwidth_gbps": 2039.0,  # A100 HBM2e
                "l2_cache_size_mb": 40,
                "sm_count": 108,
            },
            "ROCm": {
                "peak_flops_tflops": 181.0,  # MI250X FP16
                "peak_memory_bandwidth_gbps": 3276.0,  # MI250X HBM2e
                "l2_cache_size_mb": 8,
                "sm_count": 220,  # CUs
            },
            "oneAPI": {
                "peak_flops_tflops": 52.0,  # Intel Max 1550 FP16
                "peak_memory_bandwidth_gbps": 3276.0,  # HBM2e
                "l2_cache_size_mb": 408,
                "sm_count": 128,  # Xe cores
            },
            "CANN": {
                "peak_flops_tflops": 320.0,  # Ascend 910B FP16
                "peak_memory_bandwidth_gbps": 1200.0,  # HBM2
                "l2_cache_size_mb": 32,
                "sm_count": 32,  # AI cores
            },
            "MLU": {
                "peak_flops_tflops": 256.0,  # MLU370 FP16
                "peak_memory_bandwidth_gbps": 614.0,  # HBM2
                "l2_cache_size_mb": 48,
                "sm_count": 24,  # MLU cores
            },
        }
        return specs.get(platform, specs["CUDA"])
    
    def benchmark_operator(
        self,
        operator_id: str,
        operator_type: str,
        operator_fn: Callable,
        input_tensors: List[Any],
        output_shape: Tuple = None
    ) -> MicroBenchmarkResult:
        """
        Benchmark a single operator
        
        Args:
            operator_id: Unique identifier for the operator
            operator_type: Type of operator (e.g., "MatMul", "Conv2d")
            operator_fn: Function to execute the operator
            input_tensors: List of input tensors
            output_shape: Expected output shape (optional)
        
        Returns:
            MicroBenchmarkResult with detailed performance metrics
        """
        import torch
        
        # Get input shape
        if hasattr(input_tensors[0], 'shape'):
            input_shape = tuple(input_tensors[0].shape)
        else:
            input_shape = (len(input_tensors),)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = operator_fn(*input_tensors)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measurement
        latencies = []
        peak_memory = 0
        
        if torch.cuda.is_available() and self.use_cuda_events:
            # Use CUDA events for precise timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.reset_peak_memory_stats()
            
            for _ in range(self.measurement_iterations):
                start_event.record()
                output = operator_fn(*input_tensors)
                end_event.record()
                torch.cuda.synchronize()
                
                latency_ms = start_event.elapsed_time(end_event)
                latencies.append(latency_ms)
            
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            # CPU timing
            for _ in range(self.measurement_iterations):
                start_time = time.perf_counter()
                output = operator_fn(*input_tensors)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        # Calculate statistics
        latencies_np = np.array(latencies)
        mean_latency = float(np.mean(latencies_np))
        std_latency = float(np.std(latencies_np))
        min_latency = float(np.min(latencies_np))
        max_latency = float(np.max(latencies_np))
        
        # Calculate throughput
        throughput = 1000.0 / mean_latency if mean_latency > 0 else 0
        
        # Estimate FLOPS and arithmetic intensity
        flops = self._estimate_flops(operator_type, input_shape)
        bytes_accessed = self._estimate_memory_access(operator_type, input_shape, input_tensors)
        arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0
        
        # Calculate utilization
        achieved_flops = flops / (mean_latency / 1000) if mean_latency > 0 else 0
        compute_utilization = achieved_flops / (self.hardware_specs["peak_flops_tflops"] * 1e12) * 100
        
        achieved_bandwidth = bytes_accessed / (mean_latency / 1000) / 1e9 if mean_latency > 0 else 0
        memory_bandwidth_utilized = achieved_bandwidth / self.hardware_specs["peak_memory_bandwidth_gbps"] * 100
        
        # Determine if compute or memory bound using roofline model
        ridge_point = self.hardware_specs["peak_flops_tflops"] * 1e12 / (self.hardware_specs["peak_memory_bandwidth_gbps"] * 1e9)
        is_compute_bound = arithmetic_intensity > ridge_point
        is_memory_bound = arithmetic_intensity <= ridge_point
        
        # Get output shape
        if output_shape is None and hasattr(output, 'shape'):
            output_shape = tuple(output.shape)
        elif output_shape is None:
            output_shape = input_shape
        
        return MicroBenchmarkResult(
            operator_id=operator_id,
            operator_type=operator_type,
            input_shape=input_shape,
            output_shape=output_shape,
            hardware_platform=self.hardware_platform,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_ops_per_sec=throughput,
            peak_memory_bytes=peak_memory,
            memory_bandwidth_utilized=min(memory_bandwidth_utilized, 100.0),
            compute_utilization=min(compute_utilization, 100.0),
            arithmetic_intensity=arithmetic_intensity,
            is_compute_bound=is_compute_bound,
            is_memory_bound=is_memory_bound,
            raw_latencies_ms=latencies,
        )
    
    def _estimate_flops(self, operator_type: str, input_shape: Tuple) -> float:
        """Estimate FLOPS for an operator"""
        if len(input_shape) < 2:
            return 1e6  # Default
        
        if operator_type in ["MatMul", "Gemm", "Linear"]:
            # MatMul: 2 * M * N * K
            if len(input_shape) >= 2:
                m, k = input_shape[-2], input_shape[-1]
                n = k  # Assume square for simplicity
                return 2 * m * n * k
        
        elif operator_type in ["Conv2d", "Conv"]:
            # Conv2d: 2 * H * W * C_in * C_out * K_h * K_w
            if len(input_shape) >= 4:
                n, c, h, w = input_shape[:4]
                k_h, k_w = 3, 3  # Assume 3x3 kernel
                c_out = c  # Assume same output channels
                return 2 * n * h * w * c * c_out * k_h * k_w
        
        elif operator_type in ["Attention", "MultiHeadAttention"]:
            # Attention: 4 * seq_len^2 * hidden_dim
            if len(input_shape) >= 2:
                seq_len = input_shape[-2]
                hidden_dim = input_shape[-1]
                return 4 * seq_len * seq_len * hidden_dim
        
        elif operator_type in ["LayerNorm", "RMSNorm", "BatchNorm"]:
            # Normalization: ~5 * elements
            return 5 * np.prod(input_shape)
        
        elif operator_type in ["GELU", "ReLU", "SiLU", "Softmax"]:
            # Activation: ~elements
            return np.prod(input_shape)
        
        return np.prod(input_shape)  # Default
    
    def _estimate_memory_access(
        self,
        operator_type: str,
        input_shape: Tuple,
        input_tensors: List[Any]
    ) -> float:
        """Estimate bytes accessed for an operator"""
        # Assume FP16 (2 bytes per element)
        bytes_per_element = 2
        
        input_bytes = np.prod(input_shape) * bytes_per_element
        output_bytes = input_bytes  # Assume same size output
        
        # Some operators read input multiple times
        if operator_type in ["Attention", "MultiHeadAttention"]:
            input_bytes *= 3  # Q, K, V
        elif operator_type in ["LayerNorm", "RMSNorm"]:
            input_bytes *= 2  # Two passes
        
        return input_bytes + output_bytes


class ShapleyCalculator:
    """
    Calculates Shapley values for operator performance attribution
    
    The Shapley value φ_i for operator i is defined as:
    φ_i = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] * [v(S∪{i}) - v(S)]
    """
    
    def __init__(self, sampling_method: str = "permutation"):
        self.sampling_method = sampling_method
        self._cache: Dict[frozenset, float] = {}
    
    def calculate(
        self,
        operators: List[str],
        value_function: Callable[[Set[str]], float],
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """Calculate Shapley values for all operators"""
        n = len(operators)
        
        if n == 0:
            return {}
        
        if self.sampling_method == "exact" and n <= 12:
            return self._exact_shapley(operators, value_function)
        else:
            return self._permutation_sampling(operators, value_function, num_samples)
    
    def _exact_shapley(
        self,
        operators: List[str],
        value_function: Callable[[Set[str]], float]
    ) -> Dict[str, float]:
        """Exact Shapley value calculation"""
        n = len(operators)
        shapley_values = {op: 0.0 for op in operators}
        
        factorials = [math.factorial(i) for i in range(n + 1)]
        
        for i, op_i in enumerate(operators):
            others = [op for op in operators if op != op_i]
            
            for size in range(len(others) + 1):
                for subset in combinations(others, size):
                    S = set(subset)
                    S_with_i = S | {op_i}
                    
                    weight = (factorials[len(S)] * factorials[n - len(S) - 1]) / factorials[n]
                    
                    v_S = self._get_value(S, value_function)
                    v_S_with_i = self._get_value(S_with_i, value_function)
                    marginal = v_S_with_i - v_S
                    
                    shapley_values[op_i] += weight * marginal
        
        return shapley_values
    
    def _permutation_sampling(
        self,
        operators: List[str],
        value_function: Callable[[Set[str]], float],
        num_samples: int
    ) -> Dict[str, float]:
        """Monte Carlo permutation sampling for Shapley approximation"""
        n = len(operators)
        shapley_values = {op: 0.0 for op in operators}
        
        for _ in range(num_samples):
            perm = np.random.permutation(operators).tolist()
            
            coalition = set()
            prev_value = self._get_value(coalition, value_function)
            
            for op in perm:
                coalition.add(op)
                curr_value = self._get_value(coalition, value_function)
                marginal = curr_value - prev_value
                shapley_values[op] += marginal
                prev_value = curr_value
        
        for op in operators:
            shapley_values[op] /= num_samples
        
        return shapley_values
    
    def _get_value(
        self,
        coalition: Set[str],
        value_function: Callable[[Set[str]], float]
    ) -> float:
        """Get cached value for coalition"""
        key = frozenset(coalition)
        if key not in self._cache:
            self._cache[key] = value_function(coalition)
        return self._cache[key]
    
    def clear_cache(self):
        """Clear value function cache"""
        self._cache.clear()


class COPA:
    """
    Cross-platform Operator Performance Attribution
    
    Two-Stage Algorithm:
    - Stage I: Micro-benchmarking (isolated operator profiling)
    - Stage II: Model-level Attribution (Shapley value calculation)
    """
    
    def __init__(
        self,
        hardware_platform: str = "CUDA",
        bottleneck_threshold: float = 0.1,
        warmup_iterations: int = 10,
        measurement_iterations: int = 100
    ):
        self.hardware_platform = hardware_platform
        self.bottleneck_threshold = bottleneck_threshold
        
        # Stage I: Micro-benchmarker
        self.micro_benchmarker = MicroBenchmarker(
            hardware_platform=hardware_platform,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations
        )
        
        # Stage II: Shapley calculator
        self.shapley_calculator = ShapleyCalculator(sampling_method="permutation")
        
        # Data storage
        self._operator_metrics: Dict[str, PerformanceMetrics] = {}
        self._operator_types: Dict[str, str] = {}
        self._micro_benchmarks: Dict[str, MicroBenchmarkResult] = {}
    
    def stage_i_micro_benchmark(
        self,
        operators: List[Dict[str, Any]],
        operator_functions: Dict[str, Callable] = None
    ) -> Dict[str, MicroBenchmarkResult]:
        """
        Stage I: Micro-benchmarking
        
        Perform isolated operator-level profiling for each operator.
        
        Args:
            operators: List of operator dictionaries with id, type, input_shape
            operator_functions: Optional dict mapping operator_id to callable
        
        Returns:
            Dictionary mapping operator_id to MicroBenchmarkResult
        """
        logger.info(f"Stage I: Micro-benchmarking {len(operators)} operators...")
        
        results = {}
        
        for op in operators:
            op_id = op["id"]
            op_type = op["type"]
            input_shape = tuple(op.get("input_shape", [1, 1024, 1024]))
            
            # Create synthetic input if no function provided
            if operator_functions and op_id in operator_functions:
                op_fn = operator_functions[op_id]
                input_tensors = op.get("input_tensors", [])
            else:
                # Use synthetic benchmark
                result = self._synthetic_micro_benchmark(op_id, op_type, input_shape)
                results[op_id] = result
                self._micro_benchmarks[op_id] = result
                continue
            
            # Run actual benchmark
            result = self.micro_benchmarker.benchmark_operator(
                operator_id=op_id,
                operator_type=op_type,
                operator_fn=op_fn,
                input_tensors=input_tensors
            )
            
            results[op_id] = result
            self._micro_benchmarks[op_id] = result
        
        logger.info(f"Stage I complete: {len(results)} operators benchmarked")
        return results
    
    def _synthetic_micro_benchmark(
        self,
        operator_id: str,
        operator_type: str,
        input_shape: Tuple
    ) -> MicroBenchmarkResult:
        """Generate synthetic micro-benchmark result based on operator type"""
        # Base latency estimates by operator type (ms)
        base_latencies = {
            "MatMul": 0.5,
            "Gemm": 0.5,
            "Linear": 0.4,
            "Conv2d": 0.8,
            "Conv": 0.8,
            "Attention": 1.2,
            "MultiHeadAttention": 1.5,
            "LayerNorm": 0.1,
            "RMSNorm": 0.08,
            "BatchNorm": 0.1,
            "GELU": 0.05,
            "ReLU": 0.03,
            "SiLU": 0.05,
            "Softmax": 0.1,
            "Add": 0.02,
            "Mul": 0.02,
            "Embedding": 0.1,
            "Dropout": 0.01,
        }
        
        base_latency = base_latencies.get(operator_type, 0.1)
        
        # Scale by input size
        input_elements = np.prod(input_shape)
        scale_factor = (input_elements / 1e6) ** 0.5  # Square root scaling
        mean_latency = base_latency * max(scale_factor, 0.1)
        
        # Add some variance
        std_latency = mean_latency * 0.1
        min_latency = mean_latency * 0.9
        max_latency = mean_latency * 1.2
        
        # Estimate other metrics
        flops = self.micro_benchmarker._estimate_flops(operator_type, input_shape)
        bytes_accessed = input_elements * 2 * 2  # Input + output, FP16
        arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0
        
        # Determine if compute or memory bound
        ridge_point = 150  # Typical ridge point for modern GPUs
        is_compute_bound = arithmetic_intensity > ridge_point
        is_memory_bound = not is_compute_bound
        
        return MicroBenchmarkResult(
            operator_id=operator_id,
            operator_type=operator_type,
            input_shape=input_shape,
            output_shape=input_shape,
            hardware_platform=self.hardware_platform,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_ops_per_sec=1000.0 / mean_latency,
            peak_memory_bytes=int(bytes_accessed),
            memory_bandwidth_utilized=min(50.0 + np.random.uniform(-10, 10), 100.0),
            compute_utilization=min(60.0 + np.random.uniform(-15, 15), 100.0),
            arithmetic_intensity=arithmetic_intensity,
            is_compute_bound=is_compute_bound,
            is_memory_bound=is_memory_bound,
        )
    
    def stage_ii_attribute(
        self,
        target_metric: str = "execution_time_ms",
        num_samples: int = 1000
    ) -> List[AttributionResult]:
        """
        Stage II: Model-level Attribution
        
        Perform Shapley value-based performance attribution.
        
        Args:
            target_metric: Which metric to attribute
            num_samples: Number of samples for Shapley approximation
        
        Returns:
            List of attribution results sorted by contribution
        """
        if not self._micro_benchmarks:
            logger.warning("No micro-benchmark results. Run stage_i first.")
            return []
        
        logger.info(f"Stage II: Shapley attribution for {len(self._micro_benchmarks)} operators...")
        
        operators = list(self._micro_benchmarks.keys())
        
        # Define value function based on micro-benchmark results
        def value_function(coalition: Set[str]) -> float:
            if not coalition:
                return 0.0
            
            total = 0.0
            for op_id in coalition:
                if op_id in self._micro_benchmarks:
                    mb = self._micro_benchmarks[op_id]
                    if target_metric == "execution_time_ms":
                        total += mb.mean_latency_ms
                    elif target_metric == "memory":
                        total += mb.peak_memory_bytes
                    elif target_metric == "throughput":
                        total += mb.throughput_ops_per_sec
            
            return total
        
        # Calculate Shapley values
        shapley_values = self.shapley_calculator.calculate(
            operators, value_function, num_samples
        )
        
        # Calculate total for contribution ratios
        total_shapley = sum(abs(v) for v in shapley_values.values())
        if total_shapley == 0:
            total_shapley = 1.0
        
        # Create attribution results
        results = []
        for op_id, shapley_value in shapley_values.items():
            mb = self._micro_benchmarks[op_id]
            contribution_ratio = abs(shapley_value) / total_shapley
            
            # Determine bottleneck type
            bottleneck_type = ""
            if contribution_ratio >= self.bottleneck_threshold:
                if mb.is_compute_bound:
                    bottleneck_type = "compute"
                elif mb.is_memory_bound:
                    bottleneck_type = "memory"
                else:
                    bottleneck_type = "balanced"
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(mb, bottleneck_type)
            
            # Create PerformanceMetrics from MicroBenchmarkResult
            metrics = PerformanceMetrics(
                execution_time_ms=mb.mean_latency_ms,
                memory_usage_bytes=mb.peak_memory_bytes,
                throughput_ops=mb.throughput_ops_per_sec,
                latency_p50_ms=mb.mean_latency_ms,
                latency_p99_ms=mb.max_latency_ms,
                flops=self.micro_benchmarker._estimate_flops(mb.operator_type, mb.input_shape),
            )
            
            results.append(AttributionResult(
                operator_id=op_id,
                operator_type=mb.operator_type,
                shapley_value=shapley_value,
                contribution_ratio=contribution_ratio,
                rank=0,
                metrics=metrics,
                micro_benchmark=mb,
                is_bottleneck=contribution_ratio >= self.bottleneck_threshold,
                bottleneck_type=bottleneck_type,
                optimization_suggestions=suggestions,
            ))
        
        # Sort by Shapley value and assign ranks
        results.sort(key=lambda x: x.shapley_value, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        logger.info(f"Stage II complete: {sum(1 for r in results if r.is_bottleneck)} bottlenecks identified")
        return results
    
    def _generate_optimization_suggestions(
        self,
        mb: MicroBenchmarkResult,
        bottleneck_type: str
    ) -> List[str]:
        """Generate optimization suggestions based on bottleneck analysis"""
        suggestions = []
        
        if bottleneck_type == "compute":
            suggestions.append("Enable Tensor Core acceleration (FP16/BF16)")
            if mb.operator_type in ["MatMul", "Gemm", "Linear"]:
                suggestions.append("Consider operator fusion with adjacent operations")
            if mb.operator_type in ["Attention", "MultiHeadAttention"]:
                suggestions.append("Use Flash Attention implementation")
                suggestions.append("Enable KV-cache for inference")
        
        elif bottleneck_type == "memory":
            suggestions.append("Optimize memory layout (NHWC vs NCHW)")
            suggestions.append("Use memory-efficient attention variants")
            if mb.operator_type in ["Conv2d", "Conv"]:
                suggestions.append("Consider Winograd convolution for small kernels")
            suggestions.append("Enable gradient checkpointing for training")
        
        if mb.compute_utilization < 50:
            suggestions.append("Low compute utilization - consider batching")
        
        if mb.memory_bandwidth_utilized < 30:
            suggestions.append("Low memory bandwidth utilization - check data layout")
        
        return suggestions
    
    def run_full_analysis(
        self,
        operators: List[Dict[str, Any]],
        operator_functions: Dict[str, Callable] = None,
        target_metric: str = "execution_time_ms",
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Run complete two-stage COPA analysis
        
        Args:
            operators: List of operator dictionaries
            operator_functions: Optional dict mapping operator_id to callable
            target_metric: Metric to attribute
            num_samples: Samples for Shapley approximation
        
        Returns:
            Complete analysis results
        """
        # Stage I
        micro_results = self.stage_i_micro_benchmark(operators, operator_functions)
        
        # Stage II
        attribution_results = self.stage_ii_attribute(target_metric, num_samples)
        
        # Summary statistics
        total_latency = sum(mb.mean_latency_ms for mb in micro_results.values())
        bottlenecks = [r for r in attribution_results if r.is_bottleneck]
        bottleneck_latency = sum(r.metrics.execution_time_ms for r in bottlenecks)
        
        return {
            "stage_i_micro_benchmarks": {k: v.to_dict() for k, v in micro_results.items()},
            "stage_ii_attribution": [r.to_dict() for r in attribution_results],
            "summary": {
                "total_operators": len(operators),
                "total_latency_ms": total_latency,
                "num_bottlenecks": len(bottlenecks),
                "bottleneck_latency_ms": bottleneck_latency,
                "bottleneck_percentage": bottleneck_latency / total_latency * 100 if total_latency > 0 else 0,
                "compute_bound_ops": sum(1 for mb in micro_results.values() if mb.is_compute_bound),
                "memory_bound_ops": sum(1 for mb in micro_results.values() if mb.is_memory_bound),
            },
            "hardware_platform": self.hardware_platform,
        }
    
    def export_results(self, results: Dict[str, Any], path: str):
        """Export analysis results to JSON"""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results exported to {path}")


if __name__ == "__main__":
    # Test COPA two-stage analysis
    logger.info("Testing COPA two-stage analysis...")
    
    # Create sample operators
    operators = [
        {"id": "op_001", "type": "MatMul", "input_shape": [1, 1024, 1024]},
        {"id": "op_002", "type": "GELU", "input_shape": [1, 1024, 4096]},
        {"id": "op_003", "type": "LayerNorm", "input_shape": [1, 1024, 4096]},
        {"id": "op_004", "type": "Attention", "input_shape": [1, 1024, 4096]},
        {"id": "op_005", "type": "MatMul", "input_shape": [1, 1024, 4096]},
        {"id": "op_006", "type": "ReLU", "input_shape": [1, 1024, 4096]},
        {"id": "op_007", "type": "Conv2d", "input_shape": [1, 64, 224, 224]},
        {"id": "op_008", "type": "BatchNorm", "input_shape": [1, 64, 224, 224]},
    ]
    
    # Initialize COPA
    copa = COPA(hardware_platform="CUDA", bottleneck_threshold=0.1)
    
    # Run full analysis
    results = copa.run_full_analysis(operators, num_samples=500)
    
    print("\n=== COPA Two-Stage Analysis Results ===")
    print(f"\nSummary:")
    print(f"  Total operators: {results['summary']['total_operators']}")
    print(f"  Total latency: {results['summary']['total_latency_ms']:.2f} ms")
    print(f"  Bottlenecks: {results['summary']['num_bottlenecks']}")
    print(f"  Bottleneck contribution: {results['summary']['bottleneck_percentage']:.1f}%")
    print(f"  Compute-bound ops: {results['summary']['compute_bound_ops']}")
    print(f"  Memory-bound ops: {results['summary']['memory_bound_ops']}")
    
    print("\n=== Top 3 Bottlenecks ===")
    for attr in results['stage_ii_attribution'][:3]:
        print(f"  {attr['operator_id']} ({attr['operator_type']}): "
              f"Shapley={attr['shapley_value']:.4f}, "
              f"Contribution={attr['contribution_ratio']*100:.1f}%")
        if attr['optimization_suggestions']:
            print(f"    Suggestions: {attr['optimization_suggestions'][0]}")
    
    print("\nCOPA test complete!")
