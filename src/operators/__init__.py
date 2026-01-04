"""
Het-Benchmark Operator Implementations

This package contains implementations of all 16 operator types used in Het-Benchmark.
Each operator provides:
- Forward computation
- Backward computation (gradient)
- Performance profiling interface
- Cross-platform compatibility layer
"""

from .base import BaseOperator, OperatorConfig
from .linear import LinearOperator
from .conv import Conv2dOperator, Conv1dOperator
from .normalization import LayerNormOperator, BatchNorm2dOperator, RMSNormOperator
from .activation import ReLUOperator, ReLU6Operator, GELUOperator, SiLUOperator, SoftmaxOperator, TanhOperator
from .pooling import MaxPool2dOperator, AdaptiveAvgPool2dOperator
from .embedding import EmbeddingOperator
from .dropout import DropoutOperator

__all__ = [
    "BaseOperator",
    "OperatorConfig",
    "LinearOperator",
    "Conv2dOperator",
    "Conv1dOperator",
    "LayerNormOperator",
    "BatchNorm2dOperator",
    "RMSNormOperator",
    "ReLUOperator",
    "ReLU6Operator",
    "GELUOperator",
    "SiLUOperator",
    "SoftmaxOperator",
    "TanhOperator",
    "MaxPool2dOperator",
    "AdaptiveAvgPool2dOperator",
    "EmbeddingOperator",
    "DropoutOperator",
]

OPERATOR_REGISTRY = {
    "Linear": LinearOperator,
    "Conv2d": Conv2dOperator,
    "Conv1d": Conv1dOperator,
    "LayerNorm": LayerNormOperator,
    "BatchNorm2d": BatchNorm2dOperator,
    "RMSNorm": RMSNormOperator,
    "ReLU": ReLUOperator,
    "ReLU6": ReLU6Operator,
    "GELU": GELUOperator,
    "SiLU": SiLUOperator,
    "Softmax": SoftmaxOperator,
    "Tanh": TanhOperator,
    "MaxPool2d": MaxPool2dOperator,
    "AdaptiveAvgPool2d": AdaptiveAvgPool2dOperator,
    "Embedding": EmbeddingOperator,
    "Dropout": DropoutOperator,
}


def get_operator(op_type: str):
    """Get operator class by type name."""
    if op_type not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator type: {op_type}")
    return OPERATOR_REGISTRY[op_type]
