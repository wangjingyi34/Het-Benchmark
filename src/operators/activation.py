"""
Activation Function Operator Implementations

Includes:
- ReLU: Rectified Linear Unit
- ReLU6: ReLU capped at 6
- GELU: Gaussian Error Linear Unit
- SiLU: Sigmoid Linear Unit (Swish)
- Softmax: Softmax activation
- Tanh: Hyperbolic tangent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base import BaseOperator, OperatorConfig


class ReLUOperator(BaseOperator):
    """
    ReLU Activation Operator
    
    ReLU(x) = max(0, x)
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.inplace = config.extra.get("inplace", False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=self.inplace)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        # One comparison per element
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return num_elements
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        dtype_size = 4
        
        if self.inplace:
            return num_elements * dtype_size
        return num_elements * dtype_size * 2


class ReLU6Operator(BaseOperator):
    """
    ReLU6 Activation Operator
    
    ReLU6(x) = min(max(0, x), 6)
    Used in MobileNet architectures.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.inplace = config.extra.get("inplace", False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x, inplace=self.inplace)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        # Two comparisons per element
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return num_elements * 2
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        dtype_size = 4
        
        if self.inplace:
            return num_elements * dtype_size
        return num_elements * dtype_size * 2


class GELUOperator(BaseOperator):
    """
    GELU Activation Operator
    
    GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of standard normal.
    
    Widely used in Transformers (BERT, GPT, etc.)
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.approximate = config.extra.get("approximate", "none")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        # GELU approximation: ~10 ops per element
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return num_elements * 10
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        dtype_size = 4
        return num_elements * dtype_size * 2


class SiLUOperator(BaseOperator):
    """
    SiLU (Swish) Activation Operator
    
    SiLU(x) = x * sigmoid(x)
    
    Used in modern architectures like EfficientNet, LLaMA.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.inplace = config.extra.get("inplace", False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x, inplace=self.inplace)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        # Sigmoid (~4 ops) + multiply
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return num_elements * 5
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        dtype_size = 4
        
        if self.inplace:
            return num_elements * dtype_size
        return num_elements * dtype_size * 2


class SoftmaxOperator(BaseOperator):
    """
    Softmax Activation Operator
    
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Used for attention scores and classification.
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.dim = config.extra.get("dim", -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        # exp + sum + divide per element
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return num_elements * 5
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        dtype_size = 4
        return num_elements * dtype_size * 2


class TanhOperator(BaseOperator):
    """
    Tanh Activation Operator
    
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        # ~6 ops per element
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        return num_elements * 6
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        dtype_size = 4
        return num_elements * dtype_size * 2
