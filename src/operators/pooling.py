"""
Pooling Operator Implementations

Includes:
- MaxPool2d: 2D max pooling
- AdaptiveAvgPool2d: Adaptive average pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from .base import BaseOperator, OperatorConfig


class MaxPool2dOperator(BaseOperator):
    """
    2D Max Pooling Operator
    
    Applies a 2D max pooling over an input signal composed of several input planes.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.kernel_size = config.kernel_size if isinstance(config.kernel_size, tuple) else (config.kernel_size, config.kernel_size)
        self.stride = config.stride if config.stride else self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = config.padding if isinstance(config.padding, tuple) else (config.padding, config.padding)
        self.dilation = config.extra.get("dilation", (1, 1))
        self.ceil_mode = config.extra.get("ceil_mode", False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            x, self.kernel_size, self.stride, 
            self.padding, self.dilation, self.ceil_mode
        )
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        N, C, H_in, W_in = input_shape
        
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Comparisons in pooling window
        comparisons_per_window = self.kernel_size[0] * self.kernel_size[1] - 1
        
        return N * C * H_out * W_out * comparisons_per_window
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        N, C, H_in, W_in = input_shape
        
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        dtype_size = 4
        
        input_mem = N * C * H_in * W_in * dtype_size
        output_mem = N * C * H_out * W_out * dtype_size
        
        return input_mem + output_mem


class AdaptiveAvgPool2dOperator(BaseOperator):
    """
    Adaptive Average Pooling 2D Operator
    
    Applies a 2D adaptive average pooling over an input signal.
    Output size is specified, and kernel size is computed automatically.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        output_size = config.extra.get("output_size", (1, 1))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        N, C, H_in, W_in = input_shape
        H_out, W_out = self.output_size
        
        # Each output element averages over (H_in/H_out) * (W_in/W_out) elements
        kernel_h = H_in // H_out
        kernel_w = W_in // W_out
        
        # Additions + division per output element
        ops_per_element = kernel_h * kernel_w
        
        return N * C * H_out * W_out * ops_per_element
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        N, C, H_in, W_in = input_shape
        H_out, W_out = self.output_size
        
        dtype_size = 4
        
        input_mem = N * C * H_in * W_in * dtype_size
        output_mem = N * C * H_out * W_out * dtype_size
        
        return input_mem + output_mem
