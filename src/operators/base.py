"""
Base Operator Class for Het-Benchmark

Provides the foundation for all operator implementations with:
- Unified interface for forward/backward computation
- Performance profiling capabilities
- Cross-platform abstraction
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import time


@dataclass
class OperatorConfig:
    """Configuration for operator instantiation."""
    op_type: str
    name: str = ""
    # Common parameters
    in_features: int = 0
    out_features: int = 0
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: Tuple[int, ...] = (3, 3)
    stride: Tuple[int, ...] = (1, 1)
    padding: Tuple[int, ...] = (0, 0)
    # Normalization parameters
    normalized_shape: Tuple[int, ...] = ()
    eps: float = 1e-5
    # Embedding parameters
    num_embeddings: int = 0
    embedding_dim: int = 0
    # Dropout
    p: float = 0.0
    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    # Extra parameters
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Result of operator profiling."""
    op_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    latency_ms: float
    throughput_ops: float
    memory_mb: float
    flops: int
    parameters: int
    device: str
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseOperator(ABC, nn.Module):
    """
    Abstract base class for all operators in Het-Benchmark.
    
    Each operator implementation must provide:
    - forward(): Forward computation
    - get_flops(): Compute FLOPs for given input shape
    - get_memory(): Estimate memory usage
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__()
        self.config = config
        self.op_type = config.op_type
        self._profiling_enabled = False
        self._profile_results: List[ProfileResult] = []
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward computation."""
        pass
    
    @abstractmethod
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate FLOPs for given input shape."""
        pass
    
    @abstractmethod
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate memory usage in bytes."""
        pass
    
    def get_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def enable_profiling(self):
        """Enable performance profiling."""
        self._profiling_enabled = True
    
    def disable_profiling(self):
        """Disable performance profiling."""
        self._profiling_enabled = False
    
    def profile(self, x: torch.Tensor, warmup: int = 10, iterations: int = 100) -> ProfileResult:
        """
        Profile operator performance.
        
        Args:
            x: Input tensor
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
        
        Returns:
            ProfileResult with timing and memory statistics
        """
        device = x.device
        
        # Warmup
        for _ in range(warmup):
            _ = self.forward(x)
        
        # Synchronize before timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(iterations):
            output = self.forward(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        latency_ms = total_time_ms / iterations
        throughput = iterations / (total_time_ms / 1000)
        
        # Memory estimation
        memory_bytes = self.get_memory(tuple(x.shape))
        memory_mb = memory_bytes / (1024 * 1024)
        
        # FLOPs
        flops = self.get_flops(tuple(x.shape))
        
        result = ProfileResult(
            op_type=self.op_type,
            input_shape=tuple(x.shape),
            output_shape=tuple(output.shape),
            latency_ms=latency_ms,
            throughput_ops=throughput,
            memory_mb=memory_mb,
            flops=flops,
            parameters=self.get_parameters(),
            device=str(device),
        )
        
        self._profile_results.append(result)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operator to dictionary representation."""
        return {
            "op_type": self.op_type,
            "config": {
                "in_features": self.config.in_features,
                "out_features": self.config.out_features,
                "in_channels": self.config.in_channels,
                "out_channels": self.config.out_channels,
            },
            "parameters": self.get_parameters(),
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.op_type}, params={self.get_parameters():,})"
