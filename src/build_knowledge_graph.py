"""
Het-Benchmark Knowledge Graph Builder
Builds the MOH-KG (Model-Operator-Hardware Knowledge Graph) from dataset
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger
import time
from pathlib import Path


@dataclass
class KGNode:
    """Knowledge Graph Node"""
    node_id: str
    node_type: str  # "model", "operator", "hardware", "operator_type"
    properties: Dict[str, Any]


@dataclass
class KGEdge:
    """Knowledge Graph Edge"""
    source_id: str
    target_id: str
    edge_type: str  # "contains", "has_type", "supports", "runs_on", "depends_on"
    properties: Dict[str, Any]


class KnowledgeGraphBuilder:
    """
    Builds the MOH-KG (Model-Operator-Hardware Knowledge Graph)
    
    Node Types:
    - Model: AI models in the benchmark
    - Operator: Operator instances extracted from models
    - OperatorType: Abstract operator types (MatMul, Conv, etc.)
    - Hardware: Hardware platforms (CUDA, ROCm, etc.)
    
    Edge Types:
    - contains: Model -> Operator
    - has_type: Operator -> OperatorType
    - supports: Hardware -> OperatorType
    - runs_on: Operator -> Hardware (with performance data)
    - depends_on: Operator -> Operator (data dependencies)
    """
    
    def __init__(self, data_dir: str = "/workspace/het-benchmark/data"):
        self.data_dir = Path(data_dir)
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        
        # Define hardware platforms
        self.hardware_platforms = [
            {
                "id": "nvidia_cuda",
                "name": "NVIDIA CUDA/cuDNN",
                "vendor": "NVIDIA",
                "type": "GPU",
                "compute_capability": "8.0+",
            },
            {
                "id": "amd_rocm",
                "name": "AMD ROCm/MIGraphX",
                "vendor": "AMD",
                "type": "GPU",
                "architecture": "CDNA",
            },
            {
                "id": "intel_oneapi",
                "name": "Intel oneAPI/oneDNN",
                "vendor": "Intel",
                "type": "GPU/CPU",
                "architecture": "Xe",
            },
            {
                "id": "huawei_cann",
                "name": "Huawei Ascend CANN",
                "vendor": "Huawei",
                "type": "NPU",
                "architecture": "Da Vinci",
            },
            {
                "id": "cambricon_mlu",
                "name": "Cambricon MLU CNNL",
                "vendor": "Cambricon",
                "type": "MLU",
                "architecture": "MLUv02",
            },
        ]
        
        # Define operator types with their categories
        self.operator_types = {
            "MatMul": {"category": "matrix", "description": "Matrix multiplication"},
            "Conv": {"category": "matrix", "description": "Convolution operation"},
            "ConvTranspose": {"category": "matrix", "description": "Transposed convolution"},
            "BatchMatMul": {"category": "matrix", "description": "Batched matrix multiplication"},
            "Gemm": {"category": "matrix", "description": "General matrix multiply"},
            "Linear": {"category": "matrix", "description": "Linear transformation"},
            
            "Relu": {"category": "activation", "description": "ReLU activation"},
            "Gelu": {"category": "activation", "description": "GELU activation"},
            "Silu": {"category": "activation", "description": "SiLU/Swish activation"},
            "Sigmoid": {"category": "activation", "description": "Sigmoid activation"},
            "Tanh": {"category": "activation", "description": "Tanh activation"},
            "Softmax": {"category": "activation", "description": "Softmax activation"},
            
            "LayerNorm": {"category": "normalization", "description": "Layer normalization"},
            "BatchNorm": {"category": "normalization", "description": "Batch normalization"},
            "GroupNorm": {"category": "normalization", "description": "Group normalization"},
            "RMSNorm": {"category": "normalization", "description": "RMS normalization"},
            
            "MultiHeadAttention": {"category": "attention", "description": "Multi-head attention"},
            "ScaledDotProductAttention": {"category": "attention", "description": "Scaled dot-product attention"},
            "SelfAttention": {"category": "attention", "description": "Self attention"},
            
            "MaxPool": {"category": "pooling", "description": "Max pooling"},
            "AvgPool": {"category": "pooling", "description": "Average pooling"},
            "AdaptiveAvgPool": {"category": "pooling", "description": "Adaptive average pooling"},
            
            "Embedding": {"category": "embedding", "description": "Embedding lookup"},
            "PositionalEncoding": {"category": "embedding", "description": "Positional encoding"},
            
            "Add": {"category": "elementwise", "description": "Element-wise addition"},
            "Mul": {"category": "elementwise", "description": "Element-wise multiplication"},
            "Div": {"category": "elementwise", "description": "Element-wise division"},
            
            "Reshape": {"category": "reshape", "description": "Tensor reshape"},
            "Transpose": {"category": "reshape", "description": "Tensor transpose"},
            "Concat": {"category": "reshape", "description": "Tensor concatenation"},
            "Split": {"category": "reshape", "description": "Tensor split"},
            
            "Dropout": {"category": "other", "description": "Dropout regularization"},
            "Cast": {"category": "other", "description": "Type casting"},
        }
        
        # Hardware support matrix (coverage rates)
        self.hardware_support = {
            "nvidia_cuda": {
                "matrix": 0.98, "activation": 0.99, "normalization": 0.95,
                "attention": 0.92, "pooling": 0.98, "embedding": 0.95,
                "elementwise": 0.99, "reshape": 0.97, "other": 0.90,
            },
            "amd_rocm": {
                "matrix": 0.95, "activation": 0.96, "normalization": 0.92,
                "attention": 0.85, "pooling": 0.95, "embedding": 0.90,
                "elementwise": 0.97, "reshape": 0.94, "other": 0.85,
            },
            "intel_oneapi": {
                "matrix": 0.92, "activation": 0.93, "normalization": 0.88,
                "attention": 0.78, "pooling": 0.92, "embedding": 0.85,
                "elementwise": 0.94, "reshape": 0.90, "other": 0.80,
            },
            "huawei_cann": {
                "matrix": 0.94, "activation": 0.95, "normalization": 0.90,
                "attention": 0.88, "pooling": 0.93, "embedding": 0.88,
                "elementwise": 0.96, "reshape": 0.92, "other": 0.82,
            },
            "cambricon_mlu": {
                "matrix": 0.88, "activation": 0.90, "normalization": 0.85,
                "attention": 0.75, "pooling": 0.88, "embedding": 0.80,
                "elementwise": 0.92, "reshape": 0.86, "other": 0.75,
            },
        }
    
    def load_dataset(self) -> Dict[str, Any]:
        """Load the model dataset"""
        dataset_path = self.data_dir / "model_dataset.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        with open(dataset_path) as f:
            return json.load(f)
    
    def add_node(self, node: KGNode):
        """Add a node to the knowledge graph"""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: KGEdge):
        """Add an edge to the knowledge graph"""
        self.edges.append(edge)
    
    def build_hardware_nodes(self):
        """Create nodes for hardware platforms"""
        logger.info("Building hardware nodes...")
        
        for hw in self.hardware_platforms:
            node = KGNode(
                node_id=hw["id"],
                node_type="hardware",
                properties=hw,
            )
            self.add_node(node)
        
        logger.info(f"Created {len(self.hardware_platforms)} hardware nodes")
    
    def build_operator_type_nodes(self):
        """Create nodes for operator types"""
        logger.info("Building operator type nodes...")
        
        for op_type, props in self.operator_types.items():
            node = KGNode(
                node_id=f"optype_{op_type}",
                node_type="operator_type",
                properties={
                    "name": op_type,
                    **props,
                },
            )
            self.add_node(node)
        
        logger.info(f"Created {len(self.operator_types)} operator type nodes")
    
    def build_hardware_support_edges(self):
        """Create edges between hardware and operator types"""
        logger.info("Building hardware support edges...")
        
        edge_count = 0
        
        for hw_id, support in self.hardware_support.items():
            for op_type, props in self.operator_types.items():
                category = props["category"]
                coverage = support.get(category, 0.8)
                
                # Only create edge if coverage > 0.5
                if coverage > 0.5:
                    edge = KGEdge(
                        source_id=hw_id,
                        target_id=f"optype_{op_type}",
                        edge_type="supports",
                        properties={
                            "coverage": coverage,
                            "optimized": coverage > 0.9,
                        },
                    )
                    self.add_edge(edge)
                    edge_count += 1
        
        logger.info(f"Created {edge_count} hardware support edges")
    
    def build_model_and_operator_nodes(self, dataset: Dict[str, Any]):
        """Create nodes for models and operators from dataset"""
        logger.info("Building model and operator nodes...")
        
        model_count = 0
        operator_count = 0
        
        for model in dataset["models"]:
            # Create model node
            model_node = KGNode(
                node_id=model["model_id"],
                node_type="model",
                properties=model,
            )
            self.add_node(model_node)
            model_count += 1
        
        for op in dataset["operators"]:
            # Create operator node
            op_node = KGNode(
                node_id=op["op_id"],
                node_type="operator",
                properties=op,
            )
            self.add_node(op_node)
            operator_count += 1
            
            # Create edge from model to operator
            model_id = "_".join(op["op_id"].split("_")[:-2])
            if model_id in self.nodes:
                edge = KGEdge(
                    source_id=model_id,
                    target_id=op["op_id"],
                    edge_type="contains",
                    properties={},
                )
                self.add_edge(edge)
            
            # Create edge from operator to operator type
            op_type = op["op_type"]
            op_type_id = f"optype_{op_type}"
            if op_type_id in self.nodes:
                edge = KGEdge(
                    source_id=op["op_id"],
                    target_id=op_type_id,
                    edge_type="has_type",
                    properties={},
                )
                self.add_edge(edge)
        
        logger.info(f"Created {model_count} model nodes and {operator_count} operator nodes")
    
    def build(self) -> Tuple[Dict[str, KGNode], List[KGEdge]]:
        """Build the complete knowledge graph"""
        logger.info("Building MOH-KG knowledge graph...")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Build nodes
        self.build_hardware_nodes()
        self.build_operator_type_nodes()
        self.build_model_and_operator_nodes(dataset)
        
        # Build edges
        self.build_hardware_support_edges()
        
        logger.info(f"Knowledge graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")
        
        return self.nodes, self.edges
    
    def save(self, output_path: Optional[str] = None):
        """Save the knowledge graph to JSON"""
        if output_path is None:
            output_path = self.data_dir / "moh_kg.json"
        
        kg_data = {
            "version": "1.0.0",
            "name": "MOH-KG",
            "description": "Model-Operator-Hardware Knowledge Graph for Het-Benchmark",
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": dict(defaultdict(int, {
                    n.node_type: 1 for n in self.nodes.values()
                })),
                "edge_types": dict(defaultdict(int, {
                    e.edge_type: 1 for e in self.edges
                })),
            },
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }
        
        # Recalculate statistics properly
        node_type_counts = defaultdict(int)
        for n in self.nodes.values():
            node_type_counts[n.node_type] += 1
        kg_data["statistics"]["node_types"] = dict(node_type_counts)
        
        edge_type_counts = defaultdict(int)
        for e in self.edges:
            edge_type_counts[e.edge_type] += 1
        kg_data["statistics"]["edge_types"] = dict(edge_type_counts)
        
        with open(output_path, 'w') as f:
            json.dump(kg_data, f, indent=2)
        
        logger.info(f"Knowledge graph saved to {output_path}")
    
    def export_to_neo4j_cypher(self, output_path: Optional[str] = None):
        """Export knowledge graph as Neo4j Cypher statements"""
        if output_path is None:
            output_path = self.data_dir / "moh_kg_neo4j.cypher"
        
        cypher_statements = []
        
        # Create nodes
        for node in self.nodes.values():
            props_str = ", ".join([
                f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}'
                for k, v in node.properties.items()
                if not isinstance(v, (dict, list))
            ])
            
            stmt = f'CREATE (:{node.node_type.capitalize()} {{node_id: "{node.node_id}", {props_str}}});'
            cypher_statements.append(stmt)
        
        # Create edges
        for edge in self.edges:
            props_str = ", ".join([
                f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}'
                for k, v in edge.properties.items()
            ])
            
            stmt = f'''
MATCH (a {{node_id: "{edge.source_id}"}}), (b {{node_id: "{edge.target_id}"}})
CREATE (a)-[:{edge.edge_type.upper()} {{{props_str}}}]->(b);
'''
            cypher_statements.append(stmt)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(cypher_statements))
        
        logger.info(f"Neo4j Cypher export saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build MOH-KG Knowledge Graph")
    parser.add_argument("--data-dir", type=str, default="/workspace/het-benchmark/data")
    parser.add_argument("--export-neo4j", action="store_true", help="Export to Neo4j Cypher format")
    
    args = parser.parse_args()
    
    builder = KnowledgeGraphBuilder(data_dir=args.data_dir)
    
    # Build knowledge graph
    nodes, edges = builder.build()
    
    # Save to JSON
    builder.save()
    
    # Export to Neo4j if requested
    if args.export_neo4j:
        builder.export_to_neo4j_cypher()
    
    print("\n=== Knowledge Graph Statistics ===")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    
    # Count by type
    node_types = defaultdict(int)
    for n in nodes.values():
        node_types[n.node_type] += 1
    
    edge_types = defaultdict(int)
    for e in edges:
        edge_types[e.edge_type] += 1
    
    print("\nNode types:")
    for t, c in node_types.items():
        print(f"  {t}: {c}")
    
    print("\nEdge types:")
    for t, c in edge_types.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
