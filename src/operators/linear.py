"""
Linear Operator Implementation

Implements fully connected (dense) layer with:
- Standard matrix multiplication
- Optional bias
- FLOPs and memory calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base import BaseOperator, OperatorConfig


class LinearOperator(BaseOperator):
    """
    Linear (Fully Connected) Operator
    
    Computes: y = xW^T + b
    
    Parameters:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.use_bias = config.extra.get("bias", True)
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, 
                       device=config.device, dtype=config.dtype)
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, 
                           device=config.device, dtype=config.dtype)
            )
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        return F.linear(x, self.weight, self.bias)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        Calculate FLOPs for linear operation.
        
        FLOPs = 2 * batch_size * in_features * out_features
        (multiply-add counted as 2 operations)
        """
        batch_size = 1
        for dim in input_shape[:-1]:
            batch_size *= dim
        
        # Matrix multiplication: 2 * M * N * K
        flops = 2 * batch_size * self.in_features * self.out_features
        
        # Bias addition
        if self.use_bias:
            flops += batch_size * self.out_features
        
        return flops
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        """
        Estimate memory usage in bytes.
        
        Includes: input, output, weights, bias, gradients
        """
        batch_size = 1
        for dim in input_shape[:-1]:
            batch_size *= dim
        
        dtype_size = 4 if self.weight.dtype == torch.float32 else 2
        
        # Input memory
        input_mem = batch_size * self.in_features * dtype_size
        
        # Output memory
        output_mem = batch_size * self.out_features * dtype_size
        
        # Weight memory
        weight_mem = self.in_features * self.out_features * dtype_size
        
        # Bias memory
        bias_mem = self.out_features * dtype_size if self.use_bias else 0
        
        return input_mem + output_mem + weight_mem + bias_mem
