"""
Operator Extractor for Het-Benchmark
Extracts and analyzes operators from model computational graphs
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict
from loguru import logger

from model_parser import ModelGraph, OperatorNode, ModelCategory


class OperatorCategory(Enum):
    """Operator categories based on computational characteristics"""
    MATRIX = "matrix"           # Matrix operations (MatMul, Gemm, Conv)
    ACTIVATION = "activation"   # Activation functions (ReLU, GELU, etc.)
    NORMALIZATION = "norm"      # Normalization (LayerNorm, BatchNorm)
    ATTENTION = "attention"     # Attention mechanisms
    POOLING = "pooling"         # Pooling operations
    ELEMENTWISE = "elementwise" # Element-wise operations
    REDUCTION = "reduction"     # Reduction operations
    RESHAPE = "reshape"         # Shape manipulation
    EMBEDDING = "embedding"     # Embedding operations
    OTHER = "other"             # Other operations


@dataclass
class OperatorSignature:
    """Unique operator signature for matching"""
    op_type: str
    input_dtypes: Tuple[str, ...]
    input_ranks: Tuple[int, ...]  # Number of dimensions
    attributes_hash: str
    
    def __hash__(self):
        return hash((self.op_type, self.input_dtypes, self.input_ranks, self.attributes_hash))
    
    def __eq__(self, other):
        if not isinstance(other, OperatorSignature):
            return False
        return (self.op_type == other.op_type and
                self.input_dtypes == other.input_dtypes and
                self.input_ranks == other.input_ranks)


@dataclass
class OperatorInstance:
    """A specific instance of an operator with concrete shapes"""
    id: str
    op_type: str
    category: OperatorCategory
    signature: OperatorSignature
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    dtype: str
    flops: int
    memory_bytes: int
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_model: str = ""
    source_layer: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "op_type": self.op_type,
            "category": self.category.value,
            "input_shapes": [list(s) for s in self.input_shapes],
            "output_shapes": [list(s) for s in self.output_shapes],
            "dtype": self.dtype,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "attributes": self.attributes,
            "source_model": self.source_model,
            "source_layer": self.source_layer,
        }


@dataclass
class OperatorStatistics:
    """Statistics for an operator type"""
    op_type: str
    category: OperatorCategory
    count: int = 0
    total_flops: int = 0
    total_memory: int = 0
    shape_variants: Set[Tuple[Tuple[int, ...], ...]] = field(default_factory=set)
    models_using: Set[str] = field(default_factory=set)
    
    def add_instance(self, instance: OperatorInstance):
        self.count += 1
        self.total_flops += instance.flops
        self.total_memory += instance.memory_bytes
        self.shape_variants.add(tuple(instance.input_shapes))
        self.models_using.add(instance.source_model)


class OperatorExtractor:
    """
    Extracts operators from model graphs and builds operator database
    """
    
    # Operator type to category mapping
    CATEGORY_MAP = {
        # Matrix operations
        "MatMul": OperatorCategory.MATRIX,
        "BatchMatMul": OperatorCategory.MATRIX,
        "Gemm": OperatorCategory.MATRIX,
        "Conv": OperatorCategory.MATRIX,
        "Conv2d": OperatorCategory.MATRIX,
        "Conv3d": OperatorCategory.MATRIX,
        "ConvTranspose": OperatorCategory.MATRIX,
        "ConvTranspose2d": OperatorCategory.MATRIX,
        "DepthwiseConv2d": OperatorCategory.MATRIX,
        
        # Activation functions
        "ReLU": OperatorCategory.ACTIVATION,
        "GELU": OperatorCategory.ACTIVATION,
        "SiLU": OperatorCategory.ACTIVATION,
        "Swish": OperatorCategory.ACTIVATION,
        "Mish": OperatorCategory.ACTIVATION,
        "Sigmoid": OperatorCategory.ACTIVATION,
        "Tanh": OperatorCategory.ACTIVATION,
        "Softmax": OperatorCategory.ACTIVATION,
        "LogSoftmax": OperatorCategory.ACTIVATION,
        "LeakyReLU": OperatorCategory.ACTIVATION,
        "ELU": OperatorCategory.ACTIVATION,
        "PReLU": OperatorCategory.ACTIVATION,
        
        # Normalization
        "LayerNorm": OperatorCategory.NORMALIZATION,
        "BatchNorm": OperatorCategory.NORMALIZATION,
        "BatchNormalization": OperatorCategory.NORMALIZATION,
        "GroupNorm": OperatorCategory.NORMALIZATION,
        "InstanceNorm": OperatorCategory.NORMALIZATION,
        "RMSNorm": OperatorCategory.NORMALIZATION,
        
        # Attention
        "Attention": OperatorCategory.ATTENTION,
        "MultiHeadAttention": OperatorCategory.ATTENTION,
        "ScaledDotProductAttention": OperatorCategory.ATTENTION,
        "FlashAttention": OperatorCategory.ATTENTION,
        
        # Pooling
        "MaxPool": OperatorCategory.POOLING,
        "MaxPool2d": OperatorCategory.POOLING,
        "AvgPool": OperatorCategory.POOLING,
        "AvgPool2d": OperatorCategory.POOLING,
        "AdaptiveAvgPool2d": OperatorCategory.POOLING,
        "GlobalAveragePool": OperatorCategory.POOLING,
        
        # Element-wise
        "Add": OperatorCategory.ELEMENTWISE,
        "Sub": OperatorCategory.ELEMENTWISE,
        "Mul": OperatorCategory.ELEMENTWISE,
        "Div": OperatorCategory.ELEMENTWISE,
        "Pow": OperatorCategory.ELEMENTWISE,
        "Sqrt": OperatorCategory.ELEMENTWISE,
        "Exp": OperatorCategory.ELEMENTWISE,
        "Log": OperatorCategory.ELEMENTWISE,
        "Neg": OperatorCategory.ELEMENTWISE,
        "Abs": OperatorCategory.ELEMENTWISE,
        "Clip": OperatorCategory.ELEMENTWISE,
        
        # Reduction
        "ReduceSum": OperatorCategory.REDUCTION,
        "ReduceMean": OperatorCategory.REDUCTION,
        "ReduceMax": OperatorCategory.REDUCTION,
        "ReduceMin": OperatorCategory.REDUCTION,
        "Sum": OperatorCategory.REDUCTION,
        "Mean": OperatorCategory.REDUCTION,
        "ArgMax": OperatorCategory.REDUCTION,
        "ArgMin": OperatorCategory.REDUCTION,
        
        # Reshape
        "Reshape": OperatorCategory.RESHAPE,
        "Transpose": OperatorCategory.RESHAPE,
        "Permute": OperatorCategory.RESHAPE,
        "Flatten": OperatorCategory.RESHAPE,
        "Squeeze": OperatorCategory.RESHAPE,
        "Unsqueeze": OperatorCategory.RESHAPE,
        "Concat": OperatorCategory.RESHAPE,
        "Split": OperatorCategory.RESHAPE,
        "Slice": OperatorCategory.RESHAPE,
        "Gather": OperatorCategory.RESHAPE,
        "Scatter": OperatorCategory.RESHAPE,
        
        # Embedding
        "Embedding": OperatorCategory.EMBEDDING,
        "RotaryPositionEmbedding": OperatorCategory.EMBEDDING,
    }
    
    def __init__(self):
        self._instances: List[OperatorInstance] = []
        self._statistics: Dict[str, OperatorStatistics] = {}
        self._instance_counter = 0
    
    def extract_from_graph(self, graph: ModelGraph) -> List[OperatorInstance]:
        """Extract operator instances from a model graph"""
        instances = []
        
        for node in graph.nodes:
            instance = self._create_instance(node, graph.model_name)
            instances.append(instance)
            self._instances.append(instance)
            
            # Update statistics
            if node.op_type not in self._statistics:
                self._statistics[node.op_type] = OperatorStatistics(
                    op_type=node.op_type,
                    category=self._get_category(node.op_type),
                )
            self._statistics[node.op_type].add_instance(instance)
        
        logger.info(f"Extracted {len(instances)} operators from {graph.model_name}")
        return instances
    
    def _create_instance(self, node: OperatorNode, model_name: str) -> OperatorInstance:
        """Create operator instance from node"""
        self._instance_counter += 1
        
        # Create signature
        input_ranks = tuple(len(s) for s in node.input_shapes) if node.input_shapes else ()
        signature = OperatorSignature(
            op_type=node.op_type,
            input_dtypes=("float32",) * len(node.inputs),  # Default dtype
            input_ranks=input_ranks,
            attributes_hash=self._hash_attributes(node.attributes),
        )
        
        # Estimate memory
        memory_bytes = self._estimate_memory(node)
        
        return OperatorInstance(
            id=f"op_{self._instance_counter:06d}",
            op_type=node.op_type,
            category=self._get_category(node.op_type),
            signature=signature,
            input_shapes=node.input_shapes,
            output_shapes=node.output_shapes,
            dtype="float32",
            flops=node.flops,
            memory_bytes=memory_bytes,
            attributes=node.attributes,
            source_model=model_name,
            source_layer=node.name,
        )
    
    def _get_category(self, op_type: str) -> OperatorCategory:
        """Get category for operator type"""
        return self.CATEGORY_MAP.get(op_type, OperatorCategory.OTHER)
    
    def _hash_attributes(self, attributes: Dict) -> str:
        """Create hash of attributes for signature matching"""
        if not attributes:
            return ""
        # Sort keys for consistent hashing
        sorted_items = sorted(attributes.items())
        return str(hash(tuple(sorted_items)))
    
    def _estimate_memory(self, node: OperatorNode) -> int:
        """Estimate memory usage for operator"""
        total = 0
        
        # Input memory
        for shape in node.input_shapes:
            if shape:
                size = 1
                for dim in shape:
                    if dim > 0:
                        size *= dim
                total += size * 4  # Assume float32
        
        # Output memory
        for shape in node.output_shapes:
            if shape:
                size = 1
                for dim in shape:
                    if dim > 0:
                        size *= dim
                total += size * 4
        
        return total
    
    def get_all_instances(self) -> List[OperatorInstance]:
        """Get all extracted operator instances"""
        return self._instances
    
    def get_statistics(self) -> Dict[str, OperatorStatistics]:
        """Get operator statistics"""
        return self._statistics
    
    def get_category_distribution(self) -> Dict[OperatorCategory, int]:
        """Get distribution of operators by category"""
        distribution = defaultdict(int)
        for instance in self._instances:
            distribution[instance.category] += 1
        return dict(distribution)
    
    def get_unique_operators(self) -> Set[str]:
        """Get set of unique operator types"""
        return {inst.op_type for inst in self._instances}
    
    def filter_by_category(self, category: OperatorCategory) -> List[OperatorInstance]:
        """Filter instances by category"""
        return [inst for inst in self._instances if inst.category == category]
    
    def filter_by_model(self, model_name: str) -> List[OperatorInstance]:
        """Filter instances by source model"""
        return [inst for inst in self._instances if inst.source_model == model_name]
    
    def export_to_json(self, path: str):
        """Export all instances to JSON"""
        data = {
            "total_instances": len(self._instances),
            "unique_operators": list(self.get_unique_operators()),
            "category_distribution": {
                cat.value: count 
                for cat, count in self.get_category_distribution().items()
            },
            "instances": [inst.to_dict() for inst in self._instances],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self._instances)} instances to {path}")
    
    def generate_benchmark_configs(self) -> List[Dict]:
        """Generate benchmark configurations for unique operator signatures"""
        configs = []
        seen_signatures = set()
        
        for instance in self._instances:
            sig_key = (instance.op_type, tuple(instance.input_shapes))
            if sig_key not in seen_signatures:
                seen_signatures.add(sig_key)
                configs.append({
                    "op_type": instance.op_type,
                    "category": instance.category.value,
                    "input_shapes": instance.input_shapes,
                    "dtype": instance.dtype,
                    "attributes": instance.attributes,
                })
        
        return configs


class OperatorAnalyzer:
    """
    Analyzes operator characteristics for performance prediction
    """
    
    def __init__(self):
        pass
    
    def analyze_compute_intensity(self, instance: OperatorInstance) -> float:
        """
        Calculate compute intensity (FLOPS / Memory Bytes)
        Higher values indicate compute-bound operations
        """
        if instance.memory_bytes == 0:
            return 0.0
        return instance.flops / instance.memory_bytes
    
    def analyze_memory_pattern(self, instance: OperatorInstance) -> str:
        """Analyze memory access pattern"""
        op_type = instance.op_type
        
        if op_type in ["MatMul", "Gemm", "Conv2d"]:
            return "strided"
        elif op_type in ["Embedding", "Gather"]:
            return "random"
        elif op_type in ["Add", "Mul", "ReLU", "GELU"]:
            return "sequential"
        elif op_type in ["Transpose", "Permute"]:
            return "non_contiguous"
        else:
            return "unknown"
    
    def estimate_parallelism(self, instance: OperatorInstance) -> int:
        """Estimate degree of parallelism available"""
        if not instance.output_shapes:
            return 1
        
        # Total output elements
        total_elements = 1
        for shape in instance.output_shapes:
            for dim in shape:
                if dim > 0:
                    total_elements *= dim
        
        return total_elements
    
    def get_operator_profile(self, instance: OperatorInstance) -> Dict:
        """Get comprehensive operator profile"""
        return {
            "id": instance.id,
            "op_type": instance.op_type,
            "category": instance.category.value,
            "compute_intensity": self.analyze_compute_intensity(instance),
            "memory_pattern": self.analyze_memory_pattern(instance),
            "parallelism": self.estimate_parallelism(instance),
            "flops": instance.flops,
            "memory_bytes": instance.memory_bytes,
            "input_shapes": instance.input_shapes,
            "output_shapes": instance.output_shapes,
        }


if __name__ == "__main__":
    # Test operator extractor
    from model_parser import ModelParser, ModelFormat, ModelCategory
    
    # Create sample graph
    from model_parser import ModelGraph, OperatorNode
    
    graph = ModelGraph(
        model_name="test_model",
        model_format=ModelFormat.PYTORCH,
        model_category=ModelCategory.LLM,
    )
    
    # Add sample nodes
    graph.nodes = [
        OperatorNode(
            id="0", op_type="MatMul", name="linear1",
            inputs=["x"], outputs=["y"],
            input_shapes=[(1, 512, 768)],
            output_shapes=[(1, 512, 3072)],
            flops=2 * 512 * 768 * 3072,
        ),
        OperatorNode(
            id="1", op_type="GELU", name="activation",
            inputs=["y"], outputs=["z"],
            input_shapes=[(1, 512, 3072)],
            output_shapes=[(1, 512, 3072)],
            flops=512 * 3072,
        ),
        OperatorNode(
            id="2", op_type="LayerNorm", name="norm",
            inputs=["z"], outputs=["out"],
            input_shapes=[(1, 512, 3072)],
            output_shapes=[(1, 512, 3072)],
            flops=4 * 512 * 3072,
        ),
    ]
    
    # Extract operators
    extractor = OperatorExtractor()
    instances = extractor.extract_from_graph(graph)
    
    print(f"Extracted {len(instances)} operators")
    print(f"Categories: {extractor.get_category_distribution()}")
    
    # Analyze
    analyzer = OperatorAnalyzer()
    for inst in instances:
        profile = analyzer.get_operator_profile(inst)
        print(f"{inst.op_type}: intensity={profile['compute_intensity']:.2f}")
