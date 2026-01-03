"""
COPA: Cross-platform Operator Performance Attribution
Based on Shapley Value for performance bottleneck attribution
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from itertools import combinations
from collections import defaultdict
import json
import math
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
    
    def to_dict(self) -> Dict:
        return {
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_bytes": self.memory_usage_bytes,
            "throughput_ops": self.throughput_ops,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "power_consumption_w": self.power_consumption_w,
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
    is_bottleneck: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "operator_id": self.operator_id,
            "operator_type": self.operator_type,
            "shapley_value": self.shapley_value,
            "contribution_ratio": self.contribution_ratio,
            "rank": self.rank,
            "is_bottleneck": self.is_bottleneck,
            "metrics": self.metrics.to_dict(),
        }


class ShapleyCalculator:
    """
    Calculates Shapley values for operator performance attribution
    
    The Shapley value φ_i for operator i is defined as:
    φ_i = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] * [v(S∪{i}) - v(S)]
    
    where:
    - N is the set of all operators
    - S is a subset of operators not including i
    - v(S) is the characteristic function (performance with subset S)
    """
    
    def __init__(self, sampling_method: str = "permutation"):
        """
        Initialize Shapley calculator
        
        Args:
            sampling_method: Method for approximating Shapley values
                - "exact": Exact calculation (exponential complexity)
                - "permutation": Monte Carlo permutation sampling
                - "kernel": Kernel SHAP approximation
        """
        self.sampling_method = sampling_method
        self._cache: Dict[frozenset, float] = {}
    
    def calculate(
        self,
        operators: List[str],
        value_function: Callable[[Set[str]], float],
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate Shapley values for all operators
        
        Args:
            operators: List of operator IDs
            value_function: Function that returns performance value for a coalition
            num_samples: Number of samples for Monte Carlo approximation
            
        Returns:
            Dictionary mapping operator ID to Shapley value
        """
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
        
        # Precompute factorials
        factorials = [math.factorial(i) for i in range(n + 1)]
        
        for i, op_i in enumerate(operators):
            others = [op for op in operators if op != op_i]
            
            # Iterate over all subsets S of N\{i}
            for size in range(len(others) + 1):
                for subset in combinations(others, size):
                    S = set(subset)
                    S_with_i = S | {op_i}
                    
                    # Calculate weight
                    weight = (factorials[len(S)] * factorials[n - len(S) - 1]) / factorials[n]
                    
                    # Calculate marginal contribution
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
            # Random permutation
            perm = np.random.permutation(operators).tolist()
            
            # Calculate marginal contributions
            coalition = set()
            prev_value = self._get_value(coalition, value_function)
            
            for op in perm:
                coalition.add(op)
                curr_value = self._get_value(coalition, value_function)
                marginal = curr_value - prev_value
                shapley_values[op] += marginal
                prev_value = curr_value
        
        # Average over samples
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
    
    Uses Shapley values to attribute performance bottlenecks to individual operators
    in the context of cross-platform AI model migration.
    """
    
    def __init__(self, bottleneck_threshold: float = 0.1):
        """
        Initialize COPA
        
        Args:
            bottleneck_threshold: Operators with contribution ratio above this
                                  threshold are marked as bottlenecks
        """
        self.bottleneck_threshold = bottleneck_threshold
        self.shapley_calculator = ShapleyCalculator(sampling_method="permutation")
        self._operator_metrics: Dict[str, PerformanceMetrics] = {}
        self._operator_types: Dict[str, str] = {}
    
    def register_operator(
        self,
        operator_id: str,
        operator_type: str,
        metrics: PerformanceMetrics
    ):
        """Register an operator with its performance metrics"""
        self._operator_metrics[operator_id] = metrics
        self._operator_types[operator_id] = operator_type
    
    def register_operators_batch(
        self,
        operators: List[Dict[str, Any]]
    ):
        """Register multiple operators at once"""
        for op in operators:
            self.register_operator(
                op["id"],
                op["type"],
                PerformanceMetrics(**op["metrics"])
            )
    
    def attribute_performance(
        self,
        target_metric: str = "execution_time_ms",
        num_samples: int = 1000
    ) -> List[AttributionResult]:
        """
        Perform performance attribution using Shapley values
        
        Args:
            target_metric: Which metric to attribute (execution_time, memory, etc.)
            num_samples: Number of samples for Shapley approximation
            
        Returns:
            List of attribution results sorted by Shapley value (descending)
        """
        if not self._operator_metrics:
            logger.warning("No operators registered")
            return []
        
        operators = list(self._operator_metrics.keys())
        
        # Define value function based on target metric
        def value_function(coalition: Set[str]) -> float:
            if not coalition:
                return 0.0
            
            total = 0.0
            for op_id in coalition:
                metrics = self._operator_metrics[op_id]
                if target_metric == "execution_time_ms":
                    total += metrics.execution_time_ms
                elif target_metric == "memory_usage_bytes":
                    total += metrics.memory_usage_bytes
                elif target_metric == "throughput_ops":
                    total += metrics.throughput_ops
                elif target_metric == "latency_p99_ms":
                    total += metrics.latency_p99_ms
            
            return total
        
        # Calculate Shapley values
        logger.info(f"Calculating Shapley values for {len(operators)} operators...")
        shapley_values = self.shapley_calculator.calculate(
            operators, value_function, num_samples
        )
        
        # Calculate total for contribution ratios
        total_shapley = sum(abs(v) for v in shapley_values.values())
        if total_shapley == 0:
            total_shapley = 1.0  # Avoid division by zero
        
        # Create attribution results
        results = []
        for op_id, shapley_value in shapley_values.items():
            contribution_ratio = abs(shapley_value) / total_shapley
            results.append(AttributionResult(
                operator_id=op_id,
                operator_type=self._operator_types[op_id],
                shapley_value=shapley_value,
                contribution_ratio=contribution_ratio,
                rank=0,  # Will be set after sorting
                metrics=self._operator_metrics[op_id],
                is_bottleneck=contribution_ratio >= self.bottleneck_threshold,
            ))
        
        # Sort by Shapley value (descending) and assign ranks
        results.sort(key=lambda x: x.shapley_value, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        logger.info(f"Attribution complete. Found {sum(1 for r in results if r.is_bottleneck)} bottlenecks")
        
        return results
    
    def get_bottlenecks(
        self,
        target_metric: str = "execution_time_ms",
        top_k: Optional[int] = None
    ) -> List[AttributionResult]:
        """Get top bottleneck operators"""
        results = self.attribute_performance(target_metric)
        bottlenecks = [r for r in results if r.is_bottleneck]
        
        if top_k is not None:
            bottlenecks = bottlenecks[:top_k]
        
        return bottlenecks
    
    def compare_platforms(
        self,
        platform_a_metrics: Dict[str, PerformanceMetrics],
        platform_b_metrics: Dict[str, PerformanceMetrics],
        target_metric: str = "execution_time_ms"
    ) -> Dict[str, Dict]:
        """
        Compare operator performance between two platforms
        
        Returns attribution for performance difference
        """
        operators = set(platform_a_metrics.keys()) & set(platform_b_metrics.keys())
        
        # Calculate performance difference for each operator
        diff_metrics = {}
        for op_id in operators:
            a = platform_a_metrics[op_id]
            b = platform_b_metrics[op_id]
            
            diff_metrics[op_id] = PerformanceMetrics(
                execution_time_ms=b.execution_time_ms - a.execution_time_ms,
                memory_usage_bytes=b.memory_usage_bytes - a.memory_usage_bytes,
                throughput_ops=b.throughput_ops - a.throughput_ops,
                latency_p50_ms=b.latency_p50_ms - a.latency_p50_ms,
                latency_p99_ms=b.latency_p99_ms - a.latency_p99_ms,
            )
        
        # Register difference metrics
        self._operator_metrics.clear()
        for op_id, metrics in diff_metrics.items():
            self._operator_metrics[op_id] = metrics
            if op_id not in self._operator_types:
                self._operator_types[op_id] = "unknown"
        
        # Perform attribution on differences
        results = self.attribute_performance(target_metric)
        
        return {
            "attribution": [r.to_dict() for r in results],
            "summary": {
                "total_operators": len(operators),
                "bottlenecks": [r.operator_id for r in results if r.is_bottleneck],
                "top_contributor": results[0].operator_id if results else None,
            }
        }
    
    def export_results(self, results: List[AttributionResult], path: str):
        """Export attribution results to JSON"""
        data = {
            "total_operators": len(results),
            "bottleneck_threshold": self.bottleneck_threshold,
            "num_bottlenecks": sum(1 for r in results if r.is_bottleneck),
            "results": [r.to_dict() for r in results],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported attribution results to {path}")


class COPAAnalyzer:
    """
    Advanced analysis using COPA attribution results
    """
    
    def __init__(self, copa: COPA):
        self.copa = copa
    
    def analyze_operator_categories(
        self,
        results: List[AttributionResult]
    ) -> Dict[str, Dict]:
        """Analyze attribution by operator category"""
        category_stats = defaultdict(lambda: {
            "count": 0,
            "total_shapley": 0.0,
            "total_contribution": 0.0,
            "operators": [],
        })
        
        for r in results:
            cat = self._get_category(r.operator_type)
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_shapley"] += r.shapley_value
            category_stats[cat]["total_contribution"] += r.contribution_ratio
            category_stats[cat]["operators"].append(r.operator_id)
        
        return dict(category_stats)
    
    def _get_category(self, op_type: str) -> str:
        """Map operator type to category"""
        categories = {
            "matrix": ["MatMul", "Gemm", "Conv2d", "Conv3d"],
            "activation": ["ReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softmax"],
            "normalization": ["LayerNorm", "BatchNorm", "RMSNorm"],
            "attention": ["Attention", "MultiHeadAttention", "FlashAttention"],
            "elementwise": ["Add", "Mul", "Sub", "Div"],
        }
        
        for cat, ops in categories.items():
            if op_type in ops:
                return cat
        return "other"
    
    def generate_optimization_recommendations(
        self,
        results: List[AttributionResult]
    ) -> List[Dict]:
        """Generate optimization recommendations based on attribution"""
        recommendations = []
        
        for r in results:
            if not r.is_bottleneck:
                continue
            
            rec = {
                "operator_id": r.operator_id,
                "operator_type": r.operator_type,
                "contribution_ratio": r.contribution_ratio,
                "recommendations": [],
            }
            
            # Generate recommendations based on operator type
            if r.operator_type in ["MatMul", "Gemm"]:
                rec["recommendations"].extend([
                    "Consider using tensor cores for FP16/BF16 computation",
                    "Optimize matrix dimensions for hardware alignment",
                    "Explore batched matrix multiplication",
                ])
            elif r.operator_type in ["Conv2d", "Conv3d"]:
                rec["recommendations"].extend([
                    "Use cuDNN autotuning for optimal algorithm selection",
                    "Consider Winograd convolution for small kernels",
                    "Explore depthwise separable convolution",
                ])
            elif r.operator_type in ["Attention", "MultiHeadAttention"]:
                rec["recommendations"].extend([
                    "Implement Flash Attention for memory efficiency",
                    "Use KV-cache for autoregressive generation",
                    "Consider sparse attention patterns",
                ])
            elif r.operator_type in ["LayerNorm", "RMSNorm"]:
                rec["recommendations"].extend([
                    "Fuse with adjacent operations",
                    "Use fused LayerNorm kernels",
                ])
            
            recommendations.append(rec)
        
        return recommendations


if __name__ == "__main__":
    # Test COPA
    copa = COPA(bottleneck_threshold=0.15)
    
    # Register sample operators
    operators = [
        {
            "id": "op_001",
            "type": "MatMul",
            "metrics": {
                "execution_time_ms": 5.2,
                "memory_usage_bytes": 1024 * 1024 * 100,
                "throughput_ops": 1000,
                "latency_p50_ms": 5.0,
                "latency_p99_ms": 6.5,
            }
        },
        {
            "id": "op_002",
            "type": "GELU",
            "metrics": {
                "execution_time_ms": 0.8,
                "memory_usage_bytes": 1024 * 1024 * 50,
                "throughput_ops": 5000,
                "latency_p50_ms": 0.7,
                "latency_p99_ms": 1.0,
            }
        },
        {
            "id": "op_003",
            "type": "LayerNorm",
            "metrics": {
                "execution_time_ms": 1.2,
                "memory_usage_bytes": 1024 * 1024 * 30,
                "throughput_ops": 3000,
                "latency_p50_ms": 1.1,
                "latency_p99_ms": 1.5,
            }
        },
        {
            "id": "op_004",
            "type": "MultiHeadAttention",
            "metrics": {
                "execution_time_ms": 8.5,
                "memory_usage_bytes": 1024 * 1024 * 200,
                "throughput_ops": 500,
                "latency_p50_ms": 8.0,
                "latency_p99_ms": 10.0,
            }
        },
    ]
    
    copa.register_operators_batch(operators)
    
    # Perform attribution
    results = copa.attribute_performance(target_metric="execution_time_ms", num_samples=500)
    
    print("\n=== COPA Attribution Results ===")
    for r in results:
        bottleneck_marker = "🔴" if r.is_bottleneck else "🟢"
        print(f"{bottleneck_marker} Rank {r.rank}: {r.operator_type} ({r.operator_id})")
        print(f"   Shapley Value: {r.shapley_value:.4f}")
        print(f"   Contribution: {r.contribution_ratio*100:.2f}%")
    
    # Get bottlenecks
    bottlenecks = copa.get_bottlenecks()
    print(f"\nBottleneck operators: {[b.operator_id for b in bottlenecks]}")
    
    # Generate recommendations
    analyzer = COPAAnalyzer(copa)
    recommendations = analyzer.generate_optimization_recommendations(results)
    print(f"\nOptimization recommendations generated for {len(recommendations)} operators")
