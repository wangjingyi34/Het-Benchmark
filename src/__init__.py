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

from .hal import HardwareAbstractionLayer, HardwareBackend
from .model_parser import ModelParser
from .operator_extractor import OperatorExtractor
from .copa import COPA, PerformanceBottleneckAnalyzer
from .moh_kg import MOHKnowledgeGraph, KGQueryEngine
from .rgat import RGAT, PerformancePredictor
from .profiler import Profiler, OperatorProfiler, ModelProfiler

__all__ = [
    "HardwareAbstractionLayer",
    "HardwareBackend",
    "ModelParser",
    "OperatorExtractor",
    "COPA",
    "PerformanceBottleneckAnalyzer",
    "MOHKnowledgeGraph",
    "KGQueryEngine",
    "RGAT",
    "PerformancePredictor",
    "Profiler",
    "OperatorProfiler",
    "ModelProfiler",
]
