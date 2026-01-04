"""
Het-Benchmark: A Knowledge-Graph-Driven Evaluation Framework 
for Zero-Shot AI Model Migration on Heterogeneous Chips

This package provides tools for:
- Model parsing and operator extraction
- Hardware abstraction layer for multiple platforms
- COPA (Contribution-based Operator Performance Attribution)
- MOH-KG (Model-Operator-Hardware Knowledge Graph)
- RGAT (Relational Graph Attention Network) for performance prediction
- Comprehensive benchmarking and profiling
"""

__version__ = "1.0.0"
__author__ = "Het-Benchmark Team"
__license__ = "Apache-2.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "HardwareAbstractionLayer":
        from .hal import HardwareAbstractionLayer
        return HardwareAbstractionLayer
    elif name == "HAL":
        from .hal import HAL
        return HAL
    elif name == "HardwareBackend":
        from .hal import HardwareBackend
        return HardwareBackend
    elif name == "ModelParser":
        from .model_parser import ModelParser
        return ModelParser
    elif name == "OperatorExtractor":
        from .operator_extractor import OperatorExtractor
        return OperatorExtractor
    elif name == "COPA":
        from .copa import COPA
        return COPA
    elif name == "MicroBenchmarker":
        from .copa import MicroBenchmarker
        return MicroBenchmarker
    elif name == "ShapleyCalculator":
        from .copa import ShapleyCalculator
        return ShapleyCalculator
    elif name == "MOHKG":
        from .moh_kg import MOHKG
        return MOHKG
    elif name == "RGAT":
        from .rgat import RGAT
        return RGAT
    elif name == "KGA2O":
        from .kg_a2o import KGA2O
        return KGA2O
    elif name == "Profiler":
        from .profiler import Profiler
        return Profiler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HardwareAbstractionLayer",
    "HAL",
    "HardwareBackend",
    "ModelParser",
    "OperatorExtractor",
    "COPA",
    "MicroBenchmarker",
    "ShapleyCalculator",
    "MOHKG",
    "RGAT",
    "KGA2O",
    "Profiler",
]
