"""
Hardware Abstraction Layer (HAL) for Het-Benchmark
Provides unified interface for heterogeneous hardware platforms
"""

import os
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
from loguru import logger


class HardwarePlatform(Enum):
    """Supported hardware platforms"""
    CUDA = "cuda"           # NVIDIA GPU
    ROCM = "rocm"           # AMD GPU
    ONEAPI = "oneapi"       # Intel GPU/CPU
    CANN = "cann"           # Huawei Ascend
    MLU = "mlu"             # Cambricon MLU


@dataclass
class HardwareSpec:
    """Hardware specification data class"""
    platform: HardwarePlatform
    device_name: str
    compute_capability: str
    memory_total: int  # in bytes
    memory_bandwidth: float  # GB/s
    peak_flops: float  # TFLOPS
    num_cores: int
    driver_version: str
    sdk_version: str
    supported_dtypes: List[str] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorProfile:
    """Operator profiling result"""
    operator_name: str
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    dtype: str
    execution_time_ms: float
    memory_usage_bytes: int
    flops: int
    memory_bandwidth_utilization: float
    compute_utilization: float
    platform: HardwarePlatform
    timestamp: float = field(default_factory=time.time)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class HardwareBackend(ABC):
    """Abstract base class for hardware backends"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._spec: Optional[HardwareSpec] = None
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the hardware backend"""
        pass
    
    @abstractmethod
    def get_spec(self) -> HardwareSpec:
        """Get hardware specification"""
        pass
    
    @abstractmethod
    def get_supported_operators(self) -> List[str]:
        """Get list of supported operators"""
        pass
    
    @abstractmethod
    def profile_operator(
        self,
        operator_name: str,
        input_tensors: List[Any],
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ) -> OperatorProfile:
        """Profile a single operator"""
        pass
    
    @abstractmethod
    def check_operator_support(self, operator_name: str) -> bool:
        """Check if operator is supported on this platform"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    @property
    def platform(self) -> HardwarePlatform:
        """Get platform type"""
        raise NotImplementedError


class CUDABackend(HardwareBackend):
    """NVIDIA CUDA backend implementation"""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._torch = None
        self._pynvml = None
    
    @property
    def platform(self) -> HardwarePlatform:
        return HardwarePlatform.CUDA
    
    def initialize(self) -> bool:
        try:
            import torch
            import pynvml
            
            self._torch = torch
            self._pynvml = pynvml
            
            if not torch.cuda.is_available():
                logger.error("CUDA is not available")
                return False
            
            torch.cuda.set_device(self.device_id)
            pynvml.nvmlInit()
            
            self._initialized = True
            logger.info(f"CUDA backend initialized on device {self.device_id}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize CUDA backend: {e}")
            return False
    
    def get_spec(self) -> HardwareSpec:
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        
        if self._spec is not None:
            return self._spec
        
        torch = self._torch
        pynvml = self._pynvml
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        
        # Get device properties
        props = torch.cuda.get_device_properties(self.device_id)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Calculate peak FLOPS (approximate)
        # For A100: 19.5 TFLOPS FP32, 312 TFLOPS FP16 Tensor Core
        cuda_cores = props.multi_processor_count * 64  # Approximate
        clock_rate = props.clock_rate / 1e6  # GHz
        peak_flops = cuda_cores * clock_rate * 2 / 1000  # TFLOPS
        
        self._spec = HardwareSpec(
            platform=HardwarePlatform.CUDA,
            device_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            memory_total=memory_info.total,
            memory_bandwidth=props.memory_clock_rate * 2 * (props.total_memory // (1024**3)) / 8 / 1e6,
            peak_flops=peak_flops,
            num_cores=cuda_cores,
            driver_version=pynvml.nvmlSystemGetDriverVersion(),
            sdk_version=torch.version.cuda,
            supported_dtypes=["float32", "float16", "bfloat16", "int8", "int4"],
            extra_info={
                "sm_count": props.multi_processor_count,
                "max_threads_per_block": props.max_threads_per_block,
                "warp_size": props.warp_size,
            }
        )
        
        return self._spec
    
    def get_supported_operators(self) -> List[str]:
        """Get list of CUDA-supported operators"""
        # Core operators supported by cuDNN and cuBLAS
        return [
            # Matrix operations
            "MatMul", "BatchMatMul", "Gemm", "Conv2d", "Conv3d",
            "ConvTranspose2d", "DepthwiseConv2d",
            # Activation functions
            "ReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softmax",
            "LeakyReLU", "ELU", "Mish", "Swish",
            # Normalization
            "BatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm", "RMSNorm",
            # Pooling
            "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "GlobalAvgPool",
            # Attention
            "ScaledDotProductAttention", "MultiHeadAttention", "FlashAttention",
            # Element-wise
            "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log",
            # Reduction
            "Sum", "Mean", "Max", "Min", "ArgMax", "ArgMin",
            # Reshape
            "Reshape", "Transpose", "Permute", "Flatten", "Squeeze", "Unsqueeze",
            # Other
            "Embedding", "Dropout", "Concat", "Split", "Gather", "Scatter",
            "RotaryPositionEmbedding", "KVCache",
        ]
    
    def check_operator_support(self, operator_name: str) -> bool:
        return operator_name in self.get_supported_operators()
    
    def profile_operator(
        self,
        operator_name: str,
        input_tensors: List[Any],
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ) -> OperatorProfile:
        """Profile operator execution on CUDA"""
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        
        torch = self._torch
        
        # Get operator function
        op_func = self._get_operator_function(operator_name)
        
        # Move tensors to GPU
        gpu_tensors = [t.cuda(self.device_id) if hasattr(t, 'cuda') else t for t in input_tensors]
        
        # Warmup
        for _ in range(warmup_runs):
            _ = op_func(*gpu_tensors)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.reset_peak_memory_stats()
        
        start_event.record()
        for _ in range(benchmark_runs):
            output = op_func(*gpu_tensors)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event) / benchmark_runs
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Calculate FLOPS (simplified)
        flops = self._estimate_flops(operator_name, input_tensors, output)
        
        # Get shapes
        input_shapes = [tuple(t.shape) if hasattr(t, 'shape') else () for t in input_tensors]
        output_shapes = [tuple(output.shape)] if hasattr(output, 'shape') else []
        
        return OperatorProfile(
            operator_name=operator_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            dtype=str(gpu_tensors[0].dtype) if gpu_tensors else "unknown",
            execution_time_ms=elapsed_ms,
            memory_usage_bytes=peak_memory,
            flops=flops,
            memory_bandwidth_utilization=0.0,  # Calculate based on data movement
            compute_utilization=0.0,  # Calculate based on peak FLOPS
            platform=HardwarePlatform.CUDA,
            extra_metrics={
                "warmup_runs": warmup_runs,
                "benchmark_runs": benchmark_runs,
            }
        )
    
    def _get_operator_function(self, operator_name: str):
        """Get PyTorch function for operator"""
        torch = self._torch
        
        op_map = {
            "MatMul": torch.matmul,
            "BatchMatMul": torch.bmm,
            "ReLU": torch.relu,
            "GELU": torch.nn.functional.gelu,
            "SiLU": torch.nn.functional.silu,
            "Sigmoid": torch.sigmoid,
            "Tanh": torch.tanh,
            "Softmax": lambda x: torch.softmax(x, dim=-1),
            "LayerNorm": lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:]),
            "Add": lambda x, y: x + y,
            "Mul": lambda x, y: x * y,
            "Exp": torch.exp,
            "Sqrt": torch.sqrt,
            "Transpose": lambda x: x.transpose(-2, -1),
            "Sum": lambda x: x.sum(),
            "Mean": lambda x: x.mean(),
        }
        
        if operator_name not in op_map:
            raise ValueError(f"Operator {operator_name} not implemented")
        
        return op_map[operator_name]
    
    def _estimate_flops(self, operator_name: str, inputs: List[Any], output: Any) -> int:
        """Estimate FLOPS for operator"""
        if operator_name == "MatMul":
            # FLOPS = 2 * M * N * K
            if len(inputs) == 2:
                m, k = inputs[0].shape[-2:]
                _, n = inputs[1].shape[-2:]
                return 2 * m * n * k
        elif operator_name in ["ReLU", "GELU", "SiLU", "Sigmoid", "Tanh"]:
            return inputs[0].numel()
        elif operator_name in ["Add", "Mul", "Sub", "Div"]:
            return inputs[0].numel()
        
        return 0
    
    def cleanup(self):
        if self._pynvml:
            self._pynvml.nvmlShutdown()
        self._initialized = False


class HAL:
    """
    Hardware Abstraction Layer - Main Interface
    Provides unified access to heterogeneous hardware platforms
    """
    
    def __init__(self):
        self._backends: Dict[HardwarePlatform, HardwareBackend] = {}
        self._active_platform: Optional[HardwarePlatform] = None
    
    def register_backend(self, backend: HardwareBackend) -> bool:
        """Register a hardware backend"""
        if backend.initialize():
            self._backends[backend.platform] = backend
            if self._active_platform is None:
                self._active_platform = backend.platform
            logger.info(f"Registered backend: {backend.platform.value}")
            return True
        return False
    
    def get_available_platforms(self) -> List[HardwarePlatform]:
        """Get list of available platforms"""
        return list(self._backends.keys())
    
    def set_active_platform(self, platform: HardwarePlatform):
        """Set the active platform for operations"""
        if platform not in self._backends:
            raise ValueError(f"Platform {platform} not registered")
        self._active_platform = platform
    
    def get_spec(self, platform: Optional[HardwarePlatform] = None) -> HardwareSpec:
        """Get hardware specification for a platform"""
        platform = platform or self._active_platform
        if platform not in self._backends:
            raise ValueError(f"Platform {platform} not available")
        return self._backends[platform].get_spec()
    
    def profile_operator(
        self,
        operator_name: str,
        input_tensors: List[Any],
        platform: Optional[HardwarePlatform] = None,
        **kwargs
    ) -> OperatorProfile:
        """Profile an operator on specified platform"""
        platform = platform or self._active_platform
        if platform not in self._backends:
            raise ValueError(f"Platform {platform} not available")
        return self._backends[platform].profile_operator(operator_name, input_tensors, **kwargs)
    
    def check_operator_support(
        self,
        operator_name: str,
        platform: Optional[HardwarePlatform] = None
    ) -> bool:
        """Check if operator is supported"""
        platform = platform or self._active_platform
        if platform not in self._backends:
            return False
        return self._backends[platform].check_operator_support(operator_name)
    
    def get_cross_platform_support(self, operator_name: str) -> Dict[HardwarePlatform, bool]:
        """Get operator support across all platforms"""
        return {
            platform: backend.check_operator_support(operator_name)
            for platform, backend in self._backends.items()
        }
    
    def cleanup(self):
        """Cleanup all backends"""
        for backend in self._backends.values():
            backend.cleanup()
        self._backends.clear()


def create_hal() -> HAL:
    """Factory function to create and initialize HAL"""
    hal = HAL()
    
    # Try to register CUDA backend
    try:
        cuda_backend = CUDABackend()
        hal.register_backend(cuda_backend)
    except Exception as e:
        logger.warning(f"Failed to register CUDA backend: {e}")
    
    # Additional backends can be added here
    # hal.register_backend(ROCmBackend())
    # hal.register_backend(OneAPIBackend())
    # hal.register_backend(CANNBackend())
    # hal.register_backend(MLUBackend())
    
    return hal


# Alias for compatibility
HardwareAbstractionLayer = HAL


if __name__ == "__main__":
    # Test HAL
    hal = create_hal()
    
    platforms = hal.get_available_platforms()
    print(f"Available platforms: {[p.value for p in platforms]}")
    
    if platforms:
        spec = hal.get_spec()
        print(f"Device: {spec.device_name}")
        print(f"Memory: {spec.memory_total / (1024**3):.2f} GB")
        print(f"Driver: {spec.driver_version}")
        
        # Test profiling
        import torch
        x = torch.randn(1024, 1024)
        y = torch.randn(1024, 1024)
        
        profile = hal.profile_operator("MatMul", [x, y])
        print(f"MatMul execution time: {profile.execution_time_ms:.4f} ms")
    
    hal.cleanup()
