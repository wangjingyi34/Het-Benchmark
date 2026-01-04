"""
Normalization Operator Implementations

Includes:
- LayerNorm: Layer normalization
- BatchNorm2d: Batch normalization for 2D inputs
- RMSNorm: Root Mean Square normalization (used in LLMs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base import BaseOperator, OperatorConfig


class LayerNormOperator(BaseOperator):
    """
    Layer Normalization Operator
    
    Applies Layer Normalization over a mini-batch of inputs.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.normalized_shape = config.normalized_shape
        self.eps = config.eps
        self.elementwise_affine = config.extra.get("elementwise_affine", True)
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=config.device, dtype=config.dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(self.normalized_shape, device=config.device, dtype=config.dtype)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        FLOPs for LayerNorm:
        - Mean: N elements
        - Variance: 2N elements (subtract mean + square)
        - Normalize: 2N elements (subtract mean + divide by std)
        - Scale and shift: 2N elements
        Total: ~7N
        """
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        norm_elements = 1
        for dim in self.normalized_shape:
            norm_elements *= dim
        
        batch_size = num_elements // norm_elements
        
        # Per normalized group: mean, var, normalize, scale, shift
        flops = batch_size * norm_elements * 7
        
        return flops
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        dtype_size = 4  # Assume float32
        
        input_mem = num_elements * dtype_size
        output_mem = num_elements * dtype_size
        
        param_elements = 1
        for dim in self.normalized_shape:
            param_elements *= dim
        param_mem = param_elements * 2 * dtype_size if self.elementwise_affine else 0
        
        return input_mem + output_mem + param_mem


class BatchNorm2dOperator(BaseOperator):
    """
    Batch Normalization Operator for 2D inputs
    
    Applies Batch Normalization over a 4D input (N, C, H, W).
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.num_features = config.in_channels
        self.eps = config.eps
        self.momentum = config.extra.get("momentum", 0.1)
        self.affine = config.extra.get("affine", True)
        self.track_running_stats = config.extra.get("track_running_stats", True)
        
        if self.affine:
            self.weight = nn.Parameter(
                torch.ones(self.num_features, device=config.device, dtype=config.dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(self.num_features, device=config.device, dtype=config.dtype)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_var", torch.ones(self.num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x, self.running_mean, self.running_var, 
            self.weight, self.bias,
            self.training or not self.track_running_stats,
            self.momentum, self.eps
        )
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        N, C, H, W = input_shape
        # Mean, var, normalize, scale, shift per element
        return N * C * H * W * 7
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        N, C, H, W = input_shape
        dtype_size = 4
        
        input_mem = N * C * H * W * dtype_size
        output_mem = N * C * H * W * dtype_size
        param_mem = C * 4 * dtype_size  # weight, bias, running_mean, running_var
        
        return input_mem + output_mem + param_mem


class RMSNormOperator(BaseOperator):
    """
    Root Mean Square Normalization Operator
    
    Used in modern LLMs like LLaMA, Qwen, etc.
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.normalized_shape = config.normalized_shape
        if isinstance(self.normalized_shape, int):
            self.normalized_shape = (self.normalized_shape,)
        
        self.eps = config.eps
        
        self.weight = nn.Parameter(
            torch.ones(self.normalized_shape, device=config.device, dtype=config.dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        # Square, mean, rsqrt, multiply (x2)
        return num_elements * 5
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        dtype_size = 4
        
        input_mem = num_elements * dtype_size
        output_mem = num_elements * dtype_size
        
        param_elements = 1
        for dim in self.normalized_shape:
            param_elements *= dim
        param_mem = param_elements * dtype_size
        
        return input_mem + output_mem + param_mem
