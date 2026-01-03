"""
Model Parser for Het-Benchmark
Parses AI models from various formats and extracts computational graph
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from pathlib import Path
from loguru import logger


class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    HUGGINGFACE = "huggingface"
    TENSORFLOW = "tensorflow"


class ModelCategory(Enum):
    """Model categories"""
    LLM = "llm"                    # Large Language Models
    CV = "cv"                      # Computer Vision
    NLP = "nlp"                    # Natural Language Processing
    MULTIMODAL = "multimodal"      # Vision-Language Models
    AUDIO = "audio"                # Audio/Speech Models
    DIFFUSION = "diffusion"        # Diffusion Models


@dataclass
class TensorInfo:
    """Tensor information"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    is_parameter: bool = False
    memory_bytes: int = 0


@dataclass
class OperatorNode:
    """Operator node in computational graph"""
    id: str
    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    flops: int = 0
    memory_bytes: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "op_type": self.op_type,
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
            "input_shapes": [list(s) for s in self.input_shapes],
            "output_shapes": [list(s) for s in self.output_shapes],
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
        }


@dataclass
class ModelGraph:
    """Computational graph representation"""
    model_name: str
    model_format: ModelFormat
    model_category: ModelCategory
    nodes: List[OperatorNode] = field(default_factory=list)
    inputs: List[TensorInfo] = field(default_factory=list)
    outputs: List[TensorInfo] = field(default_factory=list)
    parameters: Dict[str, TensorInfo] = field(default_factory=dict)
    total_params: int = 0
    total_flops: int = 0
    total_memory: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_operator_types(self) -> Set[str]:
        """Get unique operator types"""
        return {node.op_type for node in self.nodes}
    
    def get_operator_count(self) -> Dict[str, int]:
        """Get count of each operator type"""
        counts = {}
        for node in self.nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return counts
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "model_format": self.model_format.value,
            "model_category": self.model_category.value,
            "nodes": [n.to_dict() for n in self.nodes],
            "total_params": self.total_params,
            "total_flops": self.total_flops,
            "total_memory": self.total_memory,
            "operator_types": list(self.get_operator_types()),
            "operator_count": self.get_operator_count(),
            "metadata": self.metadata,
        }
    
    def save(self, path: str):
        """Save graph to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ModelParser:
    """
    Universal Model Parser
    Parses models from various formats into unified computational graph
    """
    
    def __init__(self):
        self._parsers = {
            ModelFormat.PYTORCH: self._parse_pytorch,
            ModelFormat.ONNX: self._parse_onnx,
            ModelFormat.HUGGINGFACE: self._parse_huggingface,
        }
    
    def parse(
        self,
        model_path: str,
        model_format: ModelFormat,
        model_category: ModelCategory,
        model_name: Optional[str] = None,
        **kwargs
    ) -> ModelGraph:
        """Parse model and return computational graph"""
        
        if model_format not in self._parsers:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        model_name = model_name or Path(model_path).stem
        
        logger.info(f"Parsing model: {model_name} ({model_format.value})")
        
        graph = self._parsers[model_format](model_path, model_name, model_category, **kwargs)
        
        # Calculate totals
        graph.total_params = sum(p.memory_bytes // self._dtype_size(p.dtype) 
                                  for p in graph.parameters.values())
        graph.total_flops = sum(n.flops for n in graph.nodes)
        graph.total_memory = sum(p.memory_bytes for p in graph.parameters.values())
        
        logger.info(f"Parsed {len(graph.nodes)} operators, {graph.total_params:,} parameters")
        
        return graph
    
    def _parse_pytorch(
        self,
        model_path: str,
        model_name: str,
        model_category: ModelCategory,
        **kwargs
    ) -> ModelGraph:
        """Parse PyTorch model"""
        import torch
        
        graph = ModelGraph(
            model_name=model_name,
            model_format=ModelFormat.PYTORCH,
            model_category=model_category,
        )
        
        # Load model
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            # Try loading as a model
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            else:
                state_dict = model
        
        # Extract parameters
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                graph.parameters[name] = TensorInfo(
                    name=name,
                    shape=tuple(param.shape),
                    dtype=str(param.dtype),
                    is_parameter=True,
                    memory_bytes=param.numel() * param.element_size(),
                )
        
        # Infer operators from parameter names
        graph.nodes = self._infer_operators_from_params(graph.parameters, model_category)
        
        return graph
    
    def _parse_onnx(
        self,
        model_path: str,
        model_name: str,
        model_category: ModelCategory,
        **kwargs
    ) -> ModelGraph:
        """Parse ONNX model"""
        import onnx
        from onnx import numpy_helper
        
        graph = ModelGraph(
            model_name=model_name,
            model_format=ModelFormat.ONNX,
            model_category=model_category,
        )
        
        # Load ONNX model
        onnx_model = onnx.load(model_path)
        onnx_graph = onnx_model.graph
        
        # Extract nodes
        for i, node in enumerate(onnx_graph.node):
            op_node = OperatorNode(
                id=f"node_{i}",
                op_type=node.op_type,
                name=node.name or f"{node.op_type}_{i}",
                inputs=list(node.input),
                outputs=list(node.output),
                attributes={attr.name: self._parse_onnx_attribute(attr) 
                           for attr in node.attribute},
            )
            graph.nodes.append(op_node)
        
        # Extract initializers (parameters)
        for init in onnx_graph.initializer:
            tensor = numpy_helper.to_array(init)
            graph.parameters[init.name] = TensorInfo(
                name=init.name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                is_parameter=True,
                memory_bytes=tensor.nbytes,
            )
        
        # Extract inputs
        for inp in onnx_graph.input:
            if inp.name not in graph.parameters:
                shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
                graph.inputs.append(TensorInfo(
                    name=inp.name,
                    shape=shape,
                    dtype=self._onnx_dtype_to_str(inp.type.tensor_type.elem_type),
                ))
        
        # Extract outputs
        for out in onnx_graph.output:
            shape = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)
            graph.outputs.append(TensorInfo(
                name=out.name,
                shape=shape,
                dtype=self._onnx_dtype_to_str(out.type.tensor_type.elem_type),
            ))
        
        return graph
    
    def _parse_huggingface(
        self,
        model_path: str,
        model_name: str,
        model_category: ModelCategory,
        **kwargs
    ) -> ModelGraph:
        """Parse HuggingFace model"""
        from transformers import AutoModel, AutoConfig
        import torch
        
        graph = ModelGraph(
            model_name=model_name,
            model_format=ModelFormat.HUGGINGFACE,
            model_category=model_category,
        )
        
        # Load config first
        try:
            config = AutoConfig.from_pretrained(model_path)
            graph.metadata['config'] = config.to_dict()
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        
        # Load model
        try:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            
            # Extract parameters
            for name, param in model.named_parameters():
                graph.parameters[name] = TensorInfo(
                    name=name,
                    shape=tuple(param.shape),
                    dtype=str(param.dtype),
                    is_parameter=True,
                    memory_bytes=param.numel() * param.element_size(),
                )
            
            # Infer operators
            graph.nodes = self._infer_operators_from_params(graph.parameters, model_category)
            
            # Add model-specific metadata
            if hasattr(config, 'num_hidden_layers'):
                graph.metadata['num_layers'] = config.num_hidden_layers
            if hasattr(config, 'hidden_size'):
                graph.metadata['hidden_size'] = config.hidden_size
            if hasattr(config, 'num_attention_heads'):
                graph.metadata['num_attention_heads'] = config.num_attention_heads
            
            del model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return graph
    
    def _infer_operators_from_params(
        self,
        parameters: Dict[str, TensorInfo],
        model_category: ModelCategory
    ) -> List[OperatorNode]:
        """Infer operators from parameter names"""
        nodes = []
        node_id = 0
        
        # Group parameters by layer
        layers = {}
        for name, param in parameters.items():
            parts = name.split('.')
            if len(parts) >= 2:
                layer_name = '.'.join(parts[:-1])
                if layer_name not in layers:
                    layers[layer_name] = {}
                layers[layer_name][parts[-1]] = param
        
        # Infer operators from layer structure
        for layer_name, layer_params in layers.items():
            param_names = set(layer_params.keys())
            
            # Detect layer type
            if 'weight' in param_names:
                weight = layer_params['weight']
                
                # Linear/Dense layer
                if len(weight.shape) == 2:
                    nodes.append(OperatorNode(
                        id=f"node_{node_id}",
                        op_type="MatMul",
                        name=f"{layer_name}.matmul",
                        inputs=[f"{layer_name}.input"],
                        outputs=[f"{layer_name}.output"],
                        input_shapes=[(-1, weight.shape[1])],
                        output_shapes=[(-1, weight.shape[0])],
                        flops=2 * weight.shape[0] * weight.shape[1],
                    ))
                    node_id += 1
                
                # Conv layer
                elif len(weight.shape) == 4:
                    nodes.append(OperatorNode(
                        id=f"node_{node_id}",
                        op_type="Conv2d",
                        name=f"{layer_name}.conv",
                        inputs=[f"{layer_name}.input"],
                        outputs=[f"{layer_name}.output"],
                        attributes={
                            "out_channels": weight.shape[0],
                            "in_channels": weight.shape[1],
                            "kernel_size": weight.shape[2:],
                        },
                    ))
                    node_id += 1
            
            # LayerNorm
            if 'layer_norm' in layer_name.lower() or 'ln' in layer_name.lower():
                if 'weight' in param_names or 'gamma' in param_names:
                    nodes.append(OperatorNode(
                        id=f"node_{node_id}",
                        op_type="LayerNorm",
                        name=f"{layer_name}",
                        inputs=[f"{layer_name}.input"],
                        outputs=[f"{layer_name}.output"],
                    ))
                    node_id += 1
            
            # Attention (Q, K, V projections)
            if any(x in layer_name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']):
                if 'attention' not in [n.op_type for n in nodes if layer_name.rsplit('.', 1)[0] in n.name]:
                    attn_name = layer_name.rsplit('.', 1)[0]
                    nodes.append(OperatorNode(
                        id=f"node_{node_id}",
                        op_type="MultiHeadAttention",
                        name=f"{attn_name}.attention",
                        inputs=[f"{attn_name}.input"],
                        outputs=[f"{attn_name}.output"],
                    ))
                    node_id += 1
        
        return nodes
    
    def _parse_onnx_attribute(self, attr) -> Any:
        """Parse ONNX attribute value"""
        if attr.type == 1:  # FLOAT
            return attr.f
        elif attr.type == 2:  # INT
            return attr.i
        elif attr.type == 3:  # STRING
            return attr.s.decode('utf-8')
        elif attr.type == 6:  # FLOATS
            return list(attr.floats)
        elif attr.type == 7:  # INTS
            return list(attr.ints)
        else:
            return None
    
    def _onnx_dtype_to_str(self, dtype: int) -> str:
        """Convert ONNX dtype to string"""
        dtype_map = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            9: "bool",
            10: "float16",
            11: "float64",
            16: "bfloat16",
        }
        return dtype_map.get(dtype, f"unknown_{dtype}")
    
    def _dtype_size(self, dtype: str) -> int:
        """Get byte size of dtype"""
        size_map = {
            "float32": 4, "torch.float32": 4,
            "float16": 2, "torch.float16": 2,
            "bfloat16": 2, "torch.bfloat16": 2,
            "int32": 4, "torch.int32": 4,
            "int64": 8, "torch.int64": 8,
            "int8": 1, "torch.int8": 1,
            "uint8": 1, "torch.uint8": 1,
        }
        return size_map.get(dtype, 4)


def parse_model(
    model_path: str,
    model_format: ModelFormat,
    model_category: ModelCategory,
    model_name: Optional[str] = None,
) -> ModelGraph:
    """Convenience function to parse a model"""
    parser = ModelParser()
    return parser.parse(model_path, model_format, model_category, model_name)


if __name__ == "__main__":
    # Test with a sample model
    parser = ModelParser()
    
    # Example: Parse a HuggingFace model
    # graph = parser.parse(
    #     "bert-base-uncased",
    #     ModelFormat.HUGGINGFACE,
    #     ModelCategory.NLP,
    # )
    # print(f"Model: {graph.model_name}")
    # print(f"Operators: {graph.get_operator_count()}")
    # print(f"Parameters: {graph.total_params:,}")
