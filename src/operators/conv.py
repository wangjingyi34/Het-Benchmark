"""
Convolution Operator Implementations

Includes:
- Conv2d: 2D convolution for image processing
- Conv1d: 1D convolution for sequence processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from .base import BaseOperator, OperatorConfig


class Conv2dOperator(BaseOperator):
    """
    2D Convolution Operator
    
    Applies a 2D convolution over an input signal composed of several input planes.
    
    Parameters:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to input
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_size = config.kernel_size if isinstance(config.kernel_size, tuple) else (config.kernel_size, config.kernel_size)
        self.stride = config.stride if isinstance(config.stride, tuple) else (config.stride, config.stride)
        self.padding = config.padding if isinstance(config.padding, tuple) else (config.padding, config.padding)
        self.use_bias = config.extra.get("bias", True)
        self.groups = config.extra.get("groups", 1)
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels, 
                self.in_channels // self.groups,
                *self.kernel_size,
                device=config.device, 
                dtype=config.dtype
            )
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_channels, device=config.device, dtype=config.dtype)
            )
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, C_in, H, W)
        
        Returns:
            Output tensor of shape (N, C_out, H_out, W_out)
        """
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, groups=self.groups)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        Calculate FLOPs for Conv2d.
        
        FLOPs = 2 * N * C_out * H_out * W_out * C_in * K_h * K_w / groups
        """
        N, C_in, H_in, W_in = input_shape
        
        H_out = (H_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Convolution FLOPs
        flops = 2 * N * self.out_channels * H_out * W_out * \
                (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
        
        # Bias addition
        if self.use_bias:
            flops += N * self.out_channels * H_out * W_out
        
        return flops
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate memory usage in bytes."""
        N, C_in, H_in, W_in = input_shape
        
        H_out = (H_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        dtype_size = 4 if self.weight.dtype == torch.float32 else 2
        
        input_mem = N * C_in * H_in * W_in * dtype_size
        output_mem = N * self.out_channels * H_out * W_out * dtype_size
        weight_mem = self.weight.numel() * dtype_size
        bias_mem = self.out_channels * dtype_size if self.use_bias else 0
        
        return input_mem + output_mem + weight_mem + bias_mem


class Conv1dOperator(BaseOperator):
    """
    1D Convolution Operator
    
    Applies a 1D convolution over an input signal composed of several input planes.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_size = config.kernel_size[0] if isinstance(config.kernel_size, tuple) else config.kernel_size
        self.stride = config.stride[0] if isinstance(config.stride, tuple) else config.stride
        self.padding = config.padding[0] if isinstance(config.padding, tuple) else config.padding
        self.use_bias = config.extra.get("bias", True)
        
        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels, 
                self.in_channels,
                self.kernel_size,
                device=config.device, 
                dtype=config.dtype
            )
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_channels, device=config.device, dtype=config.dtype)
            )
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        N, C_in, L_in = input_shape
        L_out = (L_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        flops = 2 * N * self.out_channels * L_out * self.in_channels * self.kernel_size
        if self.use_bias:
            flops += N * self.out_channels * L_out
        return flops
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        N, C_in, L_in = input_shape
        L_out = (L_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        dtype_size = 4 if self.weight.dtype == torch.float32 else 2
        
        input_mem = N * C_in * L_in * dtype_size
        output_mem = N * self.out_channels * L_out * dtype_size
        weight_mem = self.weight.numel() * dtype_size
        bias_mem = self.out_channels * dtype_size if self.use_bias else 0
        
        return input_mem + output_mem + weight_mem + bias_mem
