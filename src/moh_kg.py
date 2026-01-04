"""
MOH-KG: Model-Operator-Hardware Knowledge Graph
Enhanced version with complete relation types

Relation Types:
- r_contains: Model contains Operator
- r_has_type: Operator has Type
- r_supports: Hardware supports Operator Type
- r_seq: Sequential dependency between operators
- r_sim: Similarity between operators (based on embedding)
- r_optimizes: Optimization relation
- r_compatible: Hardware compatibility relation
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from loguru import logger
import hashlib


@dataclass
class Node:
    """Base class for knowledge graph nodes"""
    id: str = ""
    type: str = ""  # "model", "operator", "operator_type", "hardware"
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        result = {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
        return result


@dataclass
class Edge:
    """Edge in the knowledge graph"""
    source: str
    target: str
    relation: str  # r_contains, r_has_type, r_supports, r_seq, r_sim, r_optimizes, r_compatible
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "properties": self.properties,
            "weight": self.weight,
        }


@dataclass
class OperatorNode(Node):
    """Operator node with detailed attributes"""
    operator_type: str = ""
    input_shapes: List[Tuple] = field(default_factory=list)
    output_shapes: List[Tuple] = field(default_factory=list)
    parameters: int = 0
    flops: float = 0.0
    memory_bytes: int = 0
    execution_order: int = 0
    
    def __post_init__(self):
        self.type = "operator"
        self.properties.update({
            "operator_type": self.operator_type,
            "input_shapes": [list(s) for s in self.input_shapes],
            "output_shapes": [list(s) for s in self.output_shapes],
            "parameters": self.parameters,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "execution_order": self.execution_order,
        })


@dataclass
class ModelNode(Node):
    """Model node"""
    model_name: str = ""
    model_family: str = ""
    task_type: str = ""
    total_parameters: int = 0
    total_operators: int = 0
    
    def __post_init__(self):
        self.type = "model"
        self.properties.update({
            "model_name": self.model_name,
            "model_family": self.model_family,
            "task_type": self.task_type,
            "total_parameters": self.total_parameters,
            "total_operators": self.total_operators,
        })


@dataclass
class HardwareNode(Node):
    """Hardware platform node"""
    platform_name: str = ""
    vendor: str = ""
    compute_capability: str = ""
    peak_flops_tflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    memory_size_gb: float = 0.0
    supported_dtypes: List[str] = field(default_factory=list)
    type: str = field(default="hardware")
    
    def __post_init__(self):
        self.type = "hardware"
        self.properties.update({
            "platform_name": self.platform_name,
            "vendor": self.vendor,
            "compute_capability": self.compute_capability,
            "peak_flops_tflops": self.peak_flops_tflops,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "memory_size_gb": self.memory_size_gb,
            "supported_dtypes": self.supported_dtypes,
        })


@dataclass
class OperatorTypeNode(Node):
    """Operator type node"""
    type_name: str = ""
    category: str = ""  # "compute", "memory", "communication", "activation"
    is_compute_intensive: bool = False
    is_memory_intensive: bool = False
    typical_flops_ratio: float = 0.0
    
    def __post_init__(self):
        self.type = "operator_type"
        self.properties.update({
            "type_name": self.type_name,
            "category": self.category,
            "is_compute_intensive": self.is_compute_intensive,
            "is_memory_intensive": self.is_memory_intensive,
            "typical_flops_ratio": self.typical_flops_ratio,
        })


class OperatorEmbedding:
    """
    Generate embeddings for operators based on their properties
    Used for computing r_sim (similarity) relations
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._type_embeddings: Dict[str, np.ndarray] = {}
        self._initialize_type_embeddings()
    
    def _initialize_type_embeddings(self):
        """Initialize base embeddings for operator types"""
        operator_types = [
            "MatMul", "Gemm", "Linear", "Conv2d", "Conv", "Conv1d", "Conv3d",
            "Attention", "MultiHeadAttention", "ScaledDotProductAttention",
            "LayerNorm", "RMSNorm", "BatchNorm", "GroupNorm", "InstanceNorm",
            "GELU", "ReLU", "SiLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU",
            "Add", "Mul", "Sub", "Div", "Concat", "Split",
            "Embedding", "Dropout", "Reshape", "Transpose", "Permute",
            "MaxPool", "AvgPool", "GlobalAvgPool", "AdaptiveAvgPool",
            "Upsample", "Interpolate",
        ]
        
        # Create deterministic embeddings based on operator type characteristics
        np.random.seed(42)
        for op_type in operator_types:
            # Base random embedding
            base_emb = np.random.randn(self.embedding_dim)
            
            # Add semantic structure based on category
            if op_type in ["MatMul", "Gemm", "Linear"]:
                base_emb[:16] += 2.0  # Matrix operations cluster
            elif op_type in ["Conv2d", "Conv", "Conv1d", "Conv3d"]:
                base_emb[16:32] += 2.0  # Convolution cluster
            elif op_type in ["Attention", "MultiHeadAttention", "ScaledDotProductAttention"]:
                base_emb[32:48] += 2.0  # Attention cluster
            elif op_type in ["LayerNorm", "RMSNorm", "BatchNorm", "GroupNorm", "InstanceNorm"]:
                base_emb[48:64] += 2.0  # Normalization cluster
            elif op_type in ["GELU", "ReLU", "SiLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU"]:
                base_emb[64:80] += 2.0  # Activation cluster
            elif op_type in ["Add", "Mul", "Sub", "Div", "Concat", "Split"]:
                base_emb[80:96] += 2.0  # Element-wise cluster
            elif op_type in ["MaxPool", "AvgPool", "GlobalAvgPool", "AdaptiveAvgPool"]:
                base_emb[96:112] += 2.0  # Pooling cluster
            
            # Normalize
            base_emb = base_emb / np.linalg.norm(base_emb)
            self._type_embeddings[op_type] = base_emb
    
    def embed_operator(self, operator: OperatorNode) -> np.ndarray:
        """Generate embedding for an operator instance"""
        # Start with type embedding
        if operator.operator_type in self._type_embeddings:
            base_emb = self._type_embeddings[operator.operator_type].copy()
        else:
            base_emb = np.random.randn(self.embedding_dim)
            base_emb = base_emb / np.linalg.norm(base_emb)
        
        # Add shape-based features
        if operator.input_shapes:
            shape = operator.input_shapes[0]
            shape_features = np.zeros(16)
            for i, dim in enumerate(shape[:4]):
                shape_features[i*4:(i+1)*4] = [
                    np.log1p(dim),
                    dim % 2,
                    dim % 4 == 0,
                    dim % 8 == 0,
                ]
            shape_features = shape_features / (np.linalg.norm(shape_features) + 1e-8)
            base_emb[-16:] = shape_features
        
        # Add parameter-based features
        if operator.parameters > 0:
            param_feature = np.log1p(operator.parameters) / 25.0  # Normalize
            base_emb[-17] = param_feature
        
        # Normalize final embedding
        base_emb = base_emb / np.linalg.norm(base_emb)
        return base_emb
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return float(np.dot(emb1, emb2))


class MOHKG:
    """
    Model-Operator-Hardware Knowledge Graph
    
    Complete implementation with all relation types:
    - r_contains: Model → Operator
    - r_has_type: Operator → OperatorType
    - r_supports: Hardware → OperatorType
    - r_seq: Operator → Operator (sequential dependency)
    - r_sim: Operator → Operator (similarity)
    - r_optimizes: Hardware → Operator (optimization capability)
    - r_compatible: Hardware → Model (compatibility)
    """
    
    RELATION_TYPES = [
        "r_contains",    # Model contains Operator
        "r_has_type",    # Operator has OperatorType
        "r_supports",    # Hardware supports OperatorType
        "r_seq",         # Sequential dependency
        "r_sim",         # Similarity relation
        "r_perf",        # Performance relationship (operator -> hardware)
        "r_optimizes",   # Optimization relation
        "r_compatible",  # Compatibility relation
    ]
    
    def __init__(self, embedding_dim: int = 128, similarity_threshold: float = 0.8):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Node storage
        self.nodes: Dict[str, Node] = {}
        self.models: Dict[str, ModelNode] = {}
        self.operators: Dict[str, OperatorNode] = {}
        self.operator_types: Dict[str, OperatorTypeNode] = {}
        self.hardware: Dict[str, HardwareNode] = {}
        
        # Edge storage by relation type
        self.edges: Dict[str, List[Edge]] = {rel: [] for rel in self.RELATION_TYPES}
        
        # Adjacency lists for efficient traversal
        self.adjacency: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Embedding generator
        self.embedder = OperatorEmbedding(embedding_dim)
        
        # Initialize default hardware platforms
        self._initialize_hardware_platforms()
        
        # Initialize operator types
        self._initialize_operator_types()
    
    def _initialize_hardware_platforms(self):
        """Initialize the 5 hardware platforms from the paper"""
        platforms = [
            HardwareNode(
                id="hw_cuda",
                platform_name="CUDA/cuDNN",
                vendor="NVIDIA",
                compute_capability="8.0",
                peak_flops_tflops=312.0,
                memory_bandwidth_gbps=2039.0,
                memory_size_gb=80.0,
                supported_dtypes=["FP32", "FP16", "BF16", "INT8", "FP8"],
            ),
            HardwareNode(
                id="hw_rocm",
                platform_name="ROCm/MIGraphX",
                vendor="AMD",
                compute_capability="gfx90a",
                peak_flops_tflops=181.0,
                memory_bandwidth_gbps=3276.0,
                memory_size_gb=128.0,
                supported_dtypes=["FP32", "FP16", "BF16", "INT8"],
            ),
            HardwareNode(
                id="hw_oneapi",
                platform_name="oneAPI/oneDNN",
                vendor="Intel",
                compute_capability="Xe-HPC",
                peak_flops_tflops=52.0,
                memory_bandwidth_gbps=3276.0,
                memory_size_gb=128.0,
                supported_dtypes=["FP32", "FP16", "BF16", "INT8"],
            ),
            HardwareNode(
                id="hw_cann",
                platform_name="CANN",
                vendor="Huawei",
                compute_capability="Ascend910B",
                peak_flops_tflops=320.0,
                memory_bandwidth_gbps=1200.0,
                memory_size_gb=64.0,
                supported_dtypes=["FP32", "FP16", "BF16", "INT8"],
            ),
            HardwareNode(
                id="hw_mlu",
                platform_name="BANG/CNNL",
                vendor="Cambricon",
                compute_capability="MLU370",
                peak_flops_tflops=256.0,
                memory_bandwidth_gbps=614.0,
                memory_size_gb=48.0,
                supported_dtypes=["FP32", "FP16", "INT8"],
            ),
        ]
        
        for hw in platforms:
            self.add_hardware(hw)
    
    def _initialize_operator_types(self):
        """Initialize operator type nodes"""
        type_definitions = [
            # Matrix operations
            ("MatMul", "compute", True, False, 0.35),
            ("Gemm", "compute", True, False, 0.35),
            ("Linear", "compute", True, False, 0.30),
            
            # Convolutions
            ("Conv2d", "compute", True, False, 0.25),
            ("Conv", "compute", True, False, 0.25),
            ("Conv1d", "compute", True, False, 0.20),
            ("Conv3d", "compute", True, False, 0.30),
            
            # Attention
            ("Attention", "compute", True, True, 0.40),
            ("MultiHeadAttention", "compute", True, True, 0.45),
            ("ScaledDotProductAttention", "compute", True, True, 0.40),
            
            # Normalization
            ("LayerNorm", "memory", False, True, 0.05),
            ("RMSNorm", "memory", False, True, 0.04),
            ("BatchNorm", "memory", False, True, 0.05),
            ("GroupNorm", "memory", False, True, 0.05),
            ("InstanceNorm", "memory", False, True, 0.05),
            
            # Activations
            ("GELU", "activation", False, False, 0.02),
            ("ReLU", "activation", False, False, 0.01),
            ("SiLU", "activation", False, False, 0.02),
            ("Sigmoid", "activation", False, False, 0.02),
            ("Tanh", "activation", False, False, 0.02),
            ("Softmax", "activation", False, True, 0.03),
            ("LeakyReLU", "activation", False, False, 0.01),
            
            # Element-wise
            ("Add", "memory", False, True, 0.01),
            ("Mul", "memory", False, True, 0.01),
            ("Sub", "memory", False, True, 0.01),
            ("Div", "memory", False, True, 0.01),
            ("Concat", "memory", False, True, 0.02),
            ("Split", "memory", False, True, 0.01),
            
            # Embedding and dropout
            ("Embedding", "memory", False, True, 0.03),
            ("Dropout", "memory", False, False, 0.01),
            
            # Reshape operations
            ("Reshape", "memory", False, True, 0.01),
            ("Transpose", "memory", False, True, 0.02),
            ("Permute", "memory", False, True, 0.02),
            
            # Pooling
            ("MaxPool", "memory", False, True, 0.02),
            ("AvgPool", "memory", False, True, 0.02),
            ("GlobalAvgPool", "memory", False, True, 0.02),
            ("AdaptiveAvgPool", "memory", False, True, 0.02),
            
            # Upsampling
            ("Upsample", "memory", False, True, 0.02),
            ("Interpolate", "memory", False, True, 0.02),
        ]
        
        for type_name, category, is_compute, is_memory, flops_ratio in type_definitions:
            op_type = OperatorTypeNode(
                id=f"type_{type_name.lower()}",
                type_name=type_name,
                category=category,
                is_compute_intensive=is_compute,
                is_memory_intensive=is_memory,
                typical_flops_ratio=flops_ratio,
            )
            self.add_operator_type(op_type)
    
    def add_model(self, model: ModelNode):
        """Add a model node to the graph"""
        self.nodes[model.id] = model
        self.models[model.id] = model
        logger.debug(f"Added model: {model.id}")
    
    def add_operator(self, operator: OperatorNode, model_id: str = None):
        """Add an operator node and optionally link to a model"""
        # Generate embedding
        operator.embedding = self.embedder.embed_operator(operator)
        
        self.nodes[operator.id] = operator
        self.operators[operator.id] = operator
        
        # Add r_contains edge if model_id provided
        if model_id and model_id in self.models:
            self.add_edge(Edge(
                source=model_id,
                target=operator.id,
                relation="r_contains",
                properties={"execution_order": operator.execution_order},
            ))
        
        # Add r_has_type edge
        type_id = f"type_{operator.operator_type.lower()}"
        if type_id in self.operator_types:
            self.add_edge(Edge(
                source=operator.id,
                target=type_id,
                relation="r_has_type",
            ))
        
        logger.debug(f"Added operator: {operator.id}")
    
    def add_operator_type(self, op_type: OperatorTypeNode):
        """Add an operator type node"""
        self.nodes[op_type.id] = op_type
        self.operator_types[op_type.id] = op_type
        
        # Add r_supports edges from all hardware platforms
        for hw_id, hw in self.hardware.items():
            # Determine support level based on operator type
            support_level = self._determine_support_level(op_type, hw)
            if support_level > 0:
                self.add_edge(Edge(
                    source=hw_id,
                    target=op_type.id,
                    relation="r_supports",
                    properties={"support_level": support_level},
                    weight=support_level,
                ))
    
    def _determine_support_level(self, op_type: OperatorTypeNode, hw: HardwareNode) -> float:
        """Determine how well a hardware platform supports an operator type"""
        # Base support level
        support = 0.8
        
        # CUDA has best overall support
        if hw.vendor == "NVIDIA":
            support = 1.0
        elif hw.vendor == "AMD":
            support = 0.94
        elif hw.vendor == "Intel":
            support = 0.89
        elif hw.vendor == "Huawei":
            support = 0.93
        elif hw.vendor == "Cambricon":
            support = 0.85
        
        # Adjust based on operator type
        if op_type.is_compute_intensive:
            # Compute-intensive ops benefit from high FLOPS
            if hw.peak_flops_tflops > 200:
                support *= 1.05
        
        if op_type.is_memory_intensive:
            # Memory-intensive ops benefit from high bandwidth
            if hw.memory_bandwidth_gbps > 2000:
                support *= 1.05
        
        return min(support, 1.0)
    
    def add_hardware(self, hardware: HardwareNode):
        """Add a hardware platform node"""
        self.nodes[hardware.id] = hardware
        self.hardware[hardware.id] = hardware
        logger.debug(f"Added hardware: {hardware.id}")
    
    def add_edge(self, edge: Edge):
        """Add an edge to the graph"""
        if edge.relation not in self.RELATION_TYPES:
            logger.warning(f"Unknown relation type: {edge.relation}")
            return
        
        self.edges[edge.relation].append(edge)
        self.adjacency[edge.source][edge.relation].append(edge.target)
    
    def build_sequential_edges(self, model_id: str):
        """
        Build r_seq (sequential dependency) edges for operators in a model
        Based on execution order
        """
        if model_id not in self.models:
            logger.warning(f"Model not found: {model_id}")
            return
        
        # Get all operators for this model
        model_operators = []
        for edge in self.edges["r_contains"]:
            if edge.source == model_id:
                op_id = edge.target
                if op_id in self.operators:
                    model_operators.append(self.operators[op_id])
        
        # Sort by execution order
        model_operators.sort(key=lambda x: x.execution_order)
        
        # Create sequential edges
        for i in range(len(model_operators) - 1):
            curr_op = model_operators[i]
            next_op = model_operators[i + 1]
            
            self.add_edge(Edge(
                source=curr_op.id,
                target=next_op.id,
                relation="r_seq",
                properties={
                    "order_diff": next_op.execution_order - curr_op.execution_order,
                },
            ))
        
        logger.info(f"Built {len(model_operators) - 1} r_seq edges for model {model_id}")
    
    def build_similarity_edges(self, threshold: float = None):
        """
        Build r_sim (similarity) edges between operators
        Based on embedding cosine similarity
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        operators = list(self.operators.values())
        n = len(operators)
        sim_edges_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                op1 = operators[i]
                op2 = operators[j]
                
                if op1.embedding is None or op2.embedding is None:
                    continue
                
                similarity = self.embedder.compute_similarity(op1.embedding, op2.embedding)
                
                if similarity >= threshold:
                    self.add_edge(Edge(
                        source=op1.id,
                        target=op2.id,
                        relation="r_sim",
                        properties={"similarity": similarity},
                        weight=similarity,
                    ))
                    sim_edges_count += 1
        
        logger.info(f"Built {sim_edges_count} r_sim edges with threshold {threshold}")
    
    def build_optimization_edges(self):
        """
        Build r_optimizes edges between hardware and operators
        Based on hardware capabilities and operator characteristics
        """
        opt_edges_count = 0
        
        for hw_id, hw in self.hardware.items():
            for op_id, op in self.operators.items():
                # Check if this hardware can optimize this operator
                opt_score = self._compute_optimization_score(hw, op)
                
                if opt_score > 0.7:  # Only add edge if significant optimization potential
                    self.add_edge(Edge(
                        source=hw_id,
                        target=op_id,
                        relation="r_optimizes",
                        properties={
                            "optimization_score": opt_score,
                            "techniques": self._get_optimization_techniques(hw, op),
                        },
                        weight=opt_score,
                    ))
                    opt_edges_count += 1
        
        logger.info(f"Built {opt_edges_count} r_optimizes edges")
    
    def _compute_optimization_score(self, hw: HardwareNode, op: OperatorNode) -> float:
        """Compute how well a hardware can optimize an operator"""
        score = 0.5  # Base score
        
        # Tensor Core optimization for matrix operations
        if op.operator_type in ["MatMul", "Gemm", "Linear", "Conv2d"]:
            if hw.vendor == "NVIDIA" and "FP16" in hw.supported_dtypes:
                score += 0.3
            elif hw.vendor == "AMD" and "FP16" in hw.supported_dtypes:
                score += 0.25
            elif hw.vendor == "Huawei":
                score += 0.25
        
        # Flash Attention optimization
        if op.operator_type in ["Attention", "MultiHeadAttention"]:
            if hw.vendor == "NVIDIA":
                score += 0.35
            elif hw.vendor == "AMD":
                score += 0.25
        
        # Memory-bound optimization
        if op.operator_type in ["LayerNorm", "RMSNorm", "Softmax"]:
            if hw.memory_bandwidth_gbps > 2000:
                score += 0.2
        
        return min(score, 1.0)
    
    def _get_optimization_techniques(self, hw: HardwareNode, op: OperatorNode) -> List[str]:
        """Get applicable optimization techniques"""
        techniques = []
        
        if op.operator_type in ["MatMul", "Gemm", "Linear"]:
            if hw.vendor == "NVIDIA":
                techniques.extend(["TensorCore", "cuBLAS", "FP16"])
            elif hw.vendor == "AMD":
                techniques.extend(["MatrixCore", "rocBLAS", "FP16"])
            elif hw.vendor == "Huawei":
                techniques.extend(["CubeCore", "CANN-GEMM", "FP16"])
        
        if op.operator_type in ["Attention", "MultiHeadAttention"]:
            if hw.vendor == "NVIDIA":
                techniques.append("FlashAttention")
            techniques.append("KV-Cache")
        
        if op.operator_type in ["Conv2d", "Conv"]:
            techniques.extend(["Winograd", "Im2Col", "FFT"])
        
        return techniques
    
    def build_compatibility_edges(self):
        """
        Build r_compatible edges between hardware and models
        Based on operator coverage
        """
        compat_edges_count = 0
        
        for hw_id, hw in self.hardware.items():
            for model_id, model in self.models.items():
                # Calculate compatibility score
                compat_score = self._compute_compatibility_score(hw_id, model_id)
                
                self.add_edge(Edge(
                    source=hw_id,
                    target=model_id,
                    relation="r_compatible",
                    properties={
                        "compatibility_score": compat_score,
                        "coverage_ratio": compat_score,
                    },
                    weight=compat_score,
                ))
                compat_edges_count += 1
        
        logger.info(f"Built {compat_edges_count} r_compatible edges")
    
    def _compute_compatibility_score(self, hw_id: str, model_id: str) -> float:
        """Compute compatibility score between hardware and model"""
        # Get all operators in the model
        model_operators = []
        for edge in self.edges["r_contains"]:
            if edge.source == model_id:
                model_operators.append(edge.target)
        
        if not model_operators:
            return 0.0
        
        # Check support for each operator type
        supported = 0
        for op_id in model_operators:
            if op_id in self.operators:
                op = self.operators[op_id]
                type_id = f"type_{op.operator_type.lower()}"
                
                # Check if hardware supports this type
                for edge in self.edges["r_supports"]:
                    if edge.source == hw_id and edge.target == type_id:
                        supported += edge.weight
                        break
                else:
                    supported += 0.5  # Partial support for unknown types
        
        return supported / len(model_operators)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        stats = {
            "total_nodes": len(self.nodes),
            "models": len(self.models),
            "operators": len(self.operators),
            "operator_types": len(self.operator_types),
            "hardware_platforms": len(self.hardware),
            "edges_by_relation": {rel: len(edges) for rel, edges in self.edges.items()},
            "total_edges": sum(len(edges) for edges in self.edges.values()),
        }
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Export knowledge graph to dictionary"""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {rel: [e.to_dict() for e in edges] for rel, edges in self.edges.items()},
            "statistics": self.get_statistics(),
        }
    
    def save_json(self, path: str):
        """Save knowledge graph to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Knowledge graph saved to {path}")
    
    def export_neo4j_cypher(self, path: str):
        """Export to Neo4j Cypher format"""
        lines = []
        
        # Create nodes
        lines.append("// Create nodes")
        for node_id, node in self.nodes.items():
            props = json.dumps(node.properties)
            lines.append(f"CREATE (:{node.type} {{id: '{node_id}', properties: {props}}})")
        
        lines.append("\n// Create edges")
        for rel_type, edges in self.edges.items():
            for edge in edges:
                props = json.dumps(edge.properties)
                lines.append(
                    f"MATCH (a {{id: '{edge.source}'}}), (b {{id: '{edge.target}'}}) "
                    f"CREATE (a)-[:{rel_type} {{weight: {edge.weight}, properties: {props}}}]->(b)"
                )
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Neo4j Cypher exported to {path}")
    
    def query_neighbors(
        self,
        node_id: str,
        relation: str = None,
        direction: str = "outgoing"
    ) -> List[str]:
        """Query neighboring nodes"""
        neighbors = []
        
        if direction in ["outgoing", "both"]:
            if relation:
                neighbors.extend(self.adjacency[node_id].get(relation, []))
            else:
                for rel_neighbors in self.adjacency[node_id].values():
                    neighbors.extend(rel_neighbors)
        
        if direction in ["incoming", "both"]:
            for rel_type, edges in self.edges.items():
                if relation and rel_type != relation:
                    continue
                for edge in edges:
                    if edge.target == node_id:
                        neighbors.append(edge.source)
        
        return list(set(neighbors))
    
    def find_similar_operators(self, operator_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar operators"""
        if operator_id not in self.operators:
            return []
        
        target_op = self.operators[operator_id]
        if target_op.embedding is None:
            return []
        
        similarities = []
        for op_id, op in self.operators.items():
            if op_id == operator_id or op.embedding is None:
                continue
            
            sim = self.embedder.compute_similarity(target_op.embedding, op.embedding)
            similarities.append((op_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def build_kg_from_dataset(dataset_path: str, output_path: str):
    """Build MOH-KG from model dataset"""
    logger.info(f"Building MOH-KG from {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Initialize knowledge graph
    kg = MOHKG()
    
    # Add models and operators
    for model_data in dataset.get("models", []):
        # Create model node
        model = ModelNode(
            id=f"model_{model_data['name'].lower().replace('-', '_').replace('.', '_')}",
            model_name=model_data["name"],
            model_family=model_data.get("family", ""),
            task_type=model_data.get("task_type", ""),
            total_parameters=model_data.get("parameters", 0),
            total_operators=len(model_data.get("operators", [])),
        )
        kg.add_model(model)
        
        # Add operators
        for i, op_data in enumerate(model_data.get("operators", [])):
            operator = OperatorNode(
                id=f"{model.id}_op_{i:04d}",
                operator_type=op_data.get("type", "Unknown"),
                input_shapes=[tuple(s) for s in op_data.get("input_shapes", [[1, 1024]])],
                output_shapes=[tuple(s) for s in op_data.get("output_shapes", [[1, 1024]])],
                parameters=op_data.get("parameters", 0),
                flops=op_data.get("flops", 0),
                memory_bytes=op_data.get("memory_bytes", 0),
                execution_order=i,
            )
            kg.add_operator(operator, model.id)
        
        # Build sequential edges for this model
        kg.build_sequential_edges(model.id)
    
    # Build similarity edges (with higher threshold for large graphs)
    num_operators = len(kg.operators)
    if num_operators > 1000:
        threshold = 0.95  # Higher threshold for large graphs
    else:
        threshold = 0.85
    kg.build_similarity_edges(threshold=threshold)
    
    # Build optimization and compatibility edges
    kg.build_optimization_edges()
    kg.build_compatibility_edges()
    
    # Save
    kg.save_json(output_path)
    
    # Print statistics
    stats = kg.get_statistics()
    logger.info(f"MOH-KG Statistics:")
    logger.info(f"  Total nodes: {stats['total_nodes']}")
    logger.info(f"  Models: {stats['models']}")
    logger.info(f"  Operators: {stats['operators']}")
    logger.info(f"  Operator types: {stats['operator_types']}")
    logger.info(f"  Hardware platforms: {stats['hardware_platforms']}")
    logger.info(f"  Total edges: {stats['total_edges']}")
    for rel, count in stats['edges_by_relation'].items():
        logger.info(f"    {rel}: {count}")
    
    return kg


if __name__ == "__main__":
    # Test MOH-KG
    logger.info("Testing MOH-KG...")
    
    kg = MOHKG()
    
    # Add a test model
    model = ModelNode(
        id="model_test",
        model_name="TestModel",
        model_family="Transformer",
        task_type="LLM",
        total_parameters=7000000000,
        total_operators=10,
    )
    kg.add_model(model)
    
    # Add test operators
    for i in range(10):
        op_types = ["MatMul", "GELU", "LayerNorm", "Attention", "MatMul",
                    "Add", "Softmax", "MatMul", "ReLU", "Linear"]
        op = OperatorNode(
            id=f"model_test_op_{i:04d}",
            operator_type=op_types[i],
            input_shapes=[(1, 1024, 4096)],
            output_shapes=[(1, 1024, 4096)],
            parameters=1000000 * (i + 1),
            execution_order=i,
        )
        kg.add_operator(op, "model_test")
    
    # Build edges
    kg.build_sequential_edges("model_test")
    kg.build_similarity_edges(threshold=0.7)
    kg.build_optimization_edges()
    kg.build_compatibility_edges()
    
    # Print statistics
    stats = kg.get_statistics()
    print("\n=== MOH-KG Statistics ===")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print("\nEdges by relation:")
    for rel, count in stats['edges_by_relation'].items():
        print(f"  {rel}: {count}")
    
    # Test similarity query
    similar = kg.find_similar_operators("model_test_op_0000", top_k=3)
    print("\n=== Similar operators to model_test_op_0000 ===")
    for op_id, sim in similar:
        print(f"  {op_id}: {sim:.4f}")
    
    print("\nMOH-KG test complete!")
