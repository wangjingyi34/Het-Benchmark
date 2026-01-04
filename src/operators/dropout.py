"""
Dropout Operator Implementation

Provides regularization through random zeroing of elements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base import BaseOperator, OperatorConfig


class DropoutOperator(BaseOperator):
    """
    Dropout Operator
    
    During training, randomly zeroes some elements of the input tensor
    with probability p using samples from a Bernoulli distribution.
    
    Parameters:
        p: Probability of an element to be zeroed (default: 0.5)
        inplace: If True, do operation in-place (default: False)
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.p = config.p if config.p > 0 else config.extra.get("p", 0.5)
        self.inplace = config.extra.get("inplace", False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of any shape
        
        Returns:
            Output tensor of same shape with elements randomly zeroed
        """
        return F.dropout(x, self.p, self.training, self.inplace)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        FLOPs for dropout.
        
        During training:
        - Generate random mask: ~1 op per element
        - Apply mask: 1 op per element
        - Scale by 1/(1-p): 1 op per element
        
        During inference: 0 (identity operation)
        """
        if not self.training:
            return 0
        
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        return num_elements * 3
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate memory usage in bytes."""
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        dtype_size = 4
        
        # Input
        input_mem = num_elements * dtype_size
        
        # Output (if not inplace)
        output_mem = 0 if self.inplace else num_elements * dtype_size
        
        # Mask (during training)
        mask_mem = num_elements if self.training else 0
        
        return input_mem + output_mem + mask_mem
    
    def get_parameters(self) -> int:
        """Dropout has no learnable parameters."""
        return 0
