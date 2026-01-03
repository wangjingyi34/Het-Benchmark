"""
Build Model Dataset for Het-Benchmark
Downloads models from Hugging Face and extracts operator information
"""

import os
import json
import torch
import onnx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger
import time
from pathlib import Path

# Hugging Face imports
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForImageClassification, AutoModelForSpeechSeq2Seq,
    AutoConfig, AutoTokenizer, AutoProcessor
)

# ONNX export
import torch.onnx


@dataclass
class ModelInfo:
    """Model metadata"""
    model_id: str
    name: str
    source: str  # huggingface, torchvision, etc.
    category: str  # LLM, CV, NLP, Audio, Multimodal
    architecture: str
    num_params: int
    num_layers: int
    hidden_size: int
    vocab_size: Optional[int] = None
    image_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None


@dataclass
class OperatorInfo:
    """Operator metadata extracted from model"""
    op_id: str
    op_type: str
    name: str
    category: str
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    attributes: Dict[str, Any]
    flops: Optional[int] = None
    memory_bytes: Optional[int] = None


# Define the 34 models to include in the benchmark dataset
BENCHMARK_MODELS = [
    # Large Language Models (LLM) - 12 models
    {"id": "meta-llama/Llama-3.1-8B", "category": "LLM", "name": "Llama-3.1-8B"},
    {"id": "meta-llama/Llama-3.1-70B", "category": "LLM", "name": "Llama-3.1-70B"},
    {"id": "Qwen/Qwen2.5-7B", "category": "LLM", "name": "Qwen2.5-7B"},
    {"id": "mistralai/Mistral-7B-v0.1", "category": "LLM", "name": "Mistral-7B"},
    {"id": "google/gemma-2-9b", "category": "LLM", "name": "Gemma-2-9B"},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "category": "LLM", "name": "Phi-3-mini"},
    {"id": "THUDM/chatglm3-6b", "category": "LLM", "name": "ChatGLM3-6B"},
    {"id": "internlm/internlm2_5-7b-chat", "category": "LLM", "name": "InternLM2.5-7B"},
    {"id": "baichuan-inc/Baichuan2-7B-Chat", "category": "LLM", "name": "Baichuan2-7B"},
    {"id": "meta-llama/CodeLlama-7b-hf", "category": "LLM", "name": "CodeLlama-7B"},
    {"id": "bigscience/bloom-560m", "category": "LLM", "name": "BLOOM-560M"},
    {"id": "openai-community/gpt2", "category": "LLM", "name": "GPT-2"},
    
    # Computer Vision (CV) - 10 models
    {"id": "microsoft/resnet-50", "category": "CV", "name": "ResNet-50"},
    {"id": "google/vit-large-patch16-224", "category": "CV", "name": "ViT-Large"},
    {"id": "microsoft/swin-base-patch4-window7-224", "category": "CV", "name": "Swin-Base"},
    {"id": "facebook/dinov2-base", "category": "CV", "name": "DINOv2-Base"},
    {"id": "facebook/sam-vit-huge", "category": "CV", "name": "SAM-ViT-Huge"},
    {"id": "google/mobilenet_v2_1.0_224", "category": "CV", "name": "MobileNet-V2"},
    {"id": "google/efficientnet-b7", "category": "CV", "name": "EfficientNet-B7"},
    {"id": "timm/densenet121.ra_in1k", "category": "CV", "name": "DenseNet-121"},
    {"id": "timm/vgg16.tv_in1k", "category": "CV", "name": "VGG16"},
    {"id": "ultralytics/yolov8", "category": "CV", "name": "YOLOv8"},
    
    # Natural Language Processing (NLP) - 4 models
    {"id": "google-bert/bert-base-uncased", "category": "NLP", "name": "BERT-Base"},
    {"id": "FacebookAI/roberta-base", "category": "NLP", "name": "RoBERTa-Base"},
    {"id": "google-t5/t5-base", "category": "NLP", "name": "T5-Base"},
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "category": "NLP", "name": "MiniLM-L6"},
    
    # Audio Models - 2 models
    {"id": "openai/whisper-large-v3", "category": "Audio", "name": "Whisper-Large-V3"},
    {"id": "facebook/wav2vec2-base-960h", "category": "Audio", "name": "Wav2Vec2-Base"},
    
    # Multimodal Models - 3 models
    {"id": "openai/clip-vit-large-patch14", "category": "Multimodal", "name": "CLIP-ViT-L/14"},
    {"id": "llava-hf/llava-1.5-7b-hf", "category": "Multimodal", "name": "LLaVA-1.5-7B"},
    {"id": "Salesforce/blip2-opt-2.7b", "category": "Multimodal", "name": "BLIP-2"},
    
    # Diffusion Models - 3 models
    {"id": "stabilityai/stable-diffusion-xl-base-1.0", "category": "Diffusion", "name": "SDXL-Base"},
    {"id": "runwayml/stable-diffusion-v1-5", "category": "Diffusion", "name": "SD-v1.5"},
    {"id": "black-forest-labs/FLUX.1-schnell", "category": "Diffusion", "name": "FLUX.1-Schnell"},
]


# Operator categories mapping
OPERATOR_CATEGORIES = {
    # Matrix operations
    "MatMul": "matrix", "Gemm": "matrix", "Conv": "matrix", "ConvTranspose": "matrix",
    "BatchMatMul": "matrix", "Linear": "matrix",
    
    # Activation functions
    "Relu": "activation", "Gelu": "activation", "Sigmoid": "activation", "Tanh": "activation",
    "Softmax": "activation", "Silu": "activation", "Swish": "activation", "Mish": "activation",
    "LeakyRelu": "activation", "Elu": "activation", "HardSwish": "activation",
    
    # Normalization
    "LayerNorm": "normalization", "BatchNorm": "normalization", "GroupNorm": "normalization",
    "InstanceNorm": "normalization", "RMSNorm": "normalization",
    
    # Attention mechanisms
    "Attention": "attention", "MultiHeadAttention": "attention", "ScaledDotProductAttention": "attention",
    "FlashAttention": "attention", "SelfAttention": "attention", "CrossAttention": "attention",
    
    # Pooling operations
    "MaxPool": "pooling", "AvgPool": "pooling", "GlobalAvgPool": "pooling",
    "AdaptiveAvgPool": "pooling", "AdaptiveMaxPool": "pooling",
    
    # Element-wise operations
    "Add": "elementwise", "Sub": "elementwise", "Mul": "elementwise", "Div": "elementwise",
    "Pow": "elementwise", "Sqrt": "elementwise", "Exp": "elementwise", "Log": "elementwise",
    
    # Reshape operations
    "Reshape": "reshape", "Transpose": "reshape", "Permute": "reshape", "Flatten": "reshape",
    "Squeeze": "reshape", "Unsqueeze": "reshape", "Concat": "reshape", "Split": "reshape",
    "Slice": "reshape", "Gather": "reshape", "Scatter": "reshape",
    
    # Embedding operations
    "Embedding": "embedding", "PositionalEncoding": "embedding", "RotaryEmbedding": "embedding",
    
    # Reduction operations
    "ReduceSum": "reduction", "ReduceMean": "reduction", "ReduceMax": "reduction",
    "ReduceMin": "reduction", "ReduceProd": "reduction",
    
    # Other
    "Dropout": "other", "Cast": "other", "Constant": "other", "Identity": "other",
}


def get_operator_category(op_type: str) -> str:
    """Get category for an operator type"""
    # Normalize operator type
    op_normalized = op_type.replace("aten::", "").replace("torch.", "")
    op_normalized = op_normalized.split("_")[0].capitalize()
    
    for key, category in OPERATOR_CATEGORIES.items():
        if key.lower() in op_normalized.lower():
            return category
    
    return "other"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def extract_operators_from_traced_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    model_id: str,
) -> List[OperatorInfo]:
    """Extract operators from a traced PyTorch model"""
    operators = []
    op_counter = defaultdict(int)
    
    try:
        # Trace the model
        model.eval()
        with torch.no_grad():
            traced = torch.jit.trace(model, example_input, strict=False)
        
        # Extract operations from the graph
        graph = traced.graph
        
        for node in graph.nodes():
            op_type = node.kind()
            
            # Skip certain meta operations
            if op_type in ["prim::Constant", "prim::ListConstruct", "prim::TupleConstruct"]:
                continue
            
            # Normalize operator type
            op_type_clean = op_type.replace("aten::", "").replace("prim::", "")
            
            op_counter[op_type_clean] += 1
            op_id = f"{model_id}_{op_type_clean}_{op_counter[op_type_clean]}"
            
            # Extract input/output shapes (if available)
            input_shapes = []
            output_shapes = []
            
            for inp in node.inputs():
                if inp.type().kind() == "TensorType":
                    try:
                        sizes = inp.type().sizes()
                        if sizes:
                            input_shapes.append(list(sizes))
                    except:
                        pass
            
            for out in node.outputs():
                if out.type().kind() == "TensorType":
                    try:
                        sizes = out.type().sizes()
                        if sizes:
                            output_shapes.append(list(sizes))
                    except:
                        pass
            
            # Extract attributes
            attributes = {}
            for attr_name in node.attributeNames():
                try:
                    attr_value = node[attr_name]
                    if isinstance(attr_value, (int, float, str, bool)):
                        attributes[attr_name] = attr_value
                except:
                    pass
            
            operator = OperatorInfo(
                op_id=op_id,
                op_type=op_type_clean,
                name=f"{op_type_clean}_{op_counter[op_type_clean]}",
                category=get_operator_category(op_type_clean),
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                attributes=attributes,
            )
            operators.append(operator)
    
    except Exception as e:
        logger.warning(f"Failed to trace model {model_id}: {e}")
        # Fallback: extract from module structure
        operators = extract_operators_from_modules(model, model_id)
    
    return operators


def extract_operators_from_modules(
    model: torch.nn.Module,
    model_id: str,
) -> List[OperatorInfo]:
    """Extract operators from model module structure"""
    operators = []
    op_counter = defaultdict(int)
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Skip container modules
        if module_type in ["Sequential", "ModuleList", "ModuleDict", "Module"]:
            continue
        
        op_counter[module_type] += 1
        op_id = f"{model_id}_{module_type}_{op_counter[module_type]}"
        
        # Map module type to operator type
        op_type_mapping = {
            "Linear": "MatMul",
            "Conv2d": "Conv",
            "Conv1d": "Conv",
            "ConvTranspose2d": "ConvTranspose",
            "BatchNorm2d": "BatchNorm",
            "BatchNorm1d": "BatchNorm",
            "LayerNorm": "LayerNorm",
            "RMSNorm": "RMSNorm",
            "Embedding": "Embedding",
            "MultiheadAttention": "MultiHeadAttention",
            "GELU": "Gelu",
            "ReLU": "Relu",
            "SiLU": "Silu",
            "Sigmoid": "Sigmoid",
            "Tanh": "Tanh",
            "Softmax": "Softmax",
            "Dropout": "Dropout",
            "MaxPool2d": "MaxPool",
            "AvgPool2d": "AvgPool",
            "AdaptiveAvgPool2d": "AdaptiveAvgPool",
        }
        
        op_type = op_type_mapping.get(module_type, module_type)
        
        # Extract attributes from module
        attributes = {}
        if hasattr(module, "in_features"):
            attributes["in_features"] = module.in_features
        if hasattr(module, "out_features"):
            attributes["out_features"] = module.out_features
        if hasattr(module, "in_channels"):
            attributes["in_channels"] = module.in_channels
        if hasattr(module, "out_channels"):
            attributes["out_channels"] = module.out_channels
        if hasattr(module, "kernel_size"):
            ks = module.kernel_size
            attributes["kernel_size"] = list(ks) if isinstance(ks, tuple) else ks
        if hasattr(module, "num_heads"):
            attributes["num_heads"] = module.num_heads
        if hasattr(module, "normalized_shape"):
            ns = module.normalized_shape
            attributes["normalized_shape"] = list(ns) if isinstance(ns, tuple) else ns
        
        operator = OperatorInfo(
            op_id=op_id,
            op_type=op_type,
            name=f"{name}_{module_type}",
            category=get_operator_category(op_type),
            input_shapes=[],
            output_shapes=[],
            attributes=attributes,
        )
        operators.append(operator)
    
    return operators


def load_model_and_extract_info(
    model_spec: Dict[str, str],
    cache_dir: str = "/workspace/models",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Optional[ModelInfo], List[OperatorInfo]]:
    """Load a model and extract its information and operators"""
    
    model_id = model_spec["id"]
    category = model_spec["category"]
    name = model_spec["name"]
    
    logger.info(f"Processing model: {name} ({model_id})")
    
    try:
        # Load model configuration first
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Determine model class based on category
        model = None
        example_input = None
        
        if category == "LLM":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            )
            # Create example input for LLM
            example_input = torch.randint(0, 1000, (1, 128)).to(device)
            
        elif category == "CV":
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
            except:
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
            # Create example input for CV
            image_size = getattr(config, "image_size", 224)
            if isinstance(image_size, (list, tuple)):
                image_size = image_size[0]
            example_input = torch.randn(1, 3, image_size, image_size).to(device)
            
        elif category == "NLP":
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
            except:
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
            example_input = torch.randint(0, 1000, (1, 128)).to(device)
            
        elif category == "Audio":
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
            except:
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
            example_input = torch.randn(1, 16000).to(device)  # 1 second of audio
            
        elif category in ["Multimodal", "Diffusion"]:
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            # Use text input for multimodal
            example_input = torch.randint(0, 1000, (1, 77)).to(device)
        
        if model is None:
            logger.error(f"Failed to load model: {model_id}")
            return None, []
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Extract model info
        num_params = count_parameters(model)
        
        model_info = ModelInfo(
            model_id=model_id.replace("/", "_"),
            name=name,
            source="huggingface",
            category=category,
            architecture=getattr(config, "model_type", "unknown"),
            num_params=num_params,
            num_layers=getattr(config, "num_hidden_layers", getattr(config, "num_layers", 0)),
            hidden_size=getattr(config, "hidden_size", getattr(config, "d_model", 0)),
            vocab_size=getattr(config, "vocab_size", None),
            image_size=getattr(config, "image_size", None),
            num_attention_heads=getattr(config, "num_attention_heads", None),
            intermediate_size=getattr(config, "intermediate_size", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", None),
        )
        
        # Extract operators
        operators = extract_operators_from_modules(model, model_info.model_id)
        
        logger.info(f"  Extracted {len(operators)} operators from {name}")
        logger.info(f"  Parameters: {num_params:,}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return model_info, operators
    
    except Exception as e:
        logger.error(f"Error processing {model_id}: {e}")
        return None, []


def build_dataset(
    output_dir: str = "/workspace/het-benchmark/data",
    cache_dir: str = "/workspace/models",
    models_to_process: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Build the complete model dataset"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    if models_to_process is None:
        models_to_process = BENCHMARK_MODELS
    
    all_models = []
    all_operators = []
    operator_stats = defaultdict(int)
    category_stats = defaultdict(int)
    
    for model_spec in models_to_process:
        model_info, operators = load_model_and_extract_info(
            model_spec,
            cache_dir=cache_dir,
        )
        
        if model_info:
            all_models.append(asdict(model_info))
            
            for op in operators:
                all_operators.append(asdict(op))
                operator_stats[op.op_type] += 1
                category_stats[op.category] += 1
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Create dataset summary
    dataset = {
        "version": "1.0.0",
        "name": "Het-Benchmark Model Dataset",
        "description": "34 representative AI models for heterogeneous chip evaluation",
        "models": all_models,
        "operators": all_operators,
        "statistics": {
            "total_models": len(all_models),
            "total_operators": len(all_operators),
            "operator_types": dict(operator_stats),
            "operator_categories": dict(category_stats),
            "models_by_category": {
                cat: sum(1 for m in all_models if m["category"] == cat)
                for cat in set(m["category"] for m in all_models)
            },
        },
    }
    
    # Save dataset
    dataset_path = os.path.join(output_dir, "model_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Dataset saved to {dataset_path}")
    logger.info(f"Total models: {len(all_models)}")
    logger.info(f"Total operators: {len(all_operators)}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Het-Benchmark Model Dataset")
    parser.add_argument("--output-dir", type=str, default="/workspace/het-benchmark/data")
    parser.add_argument("--cache-dir", type=str, default="/workspace/models")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of models to process")
    
    args = parser.parse_args()
    
    models = BENCHMARK_MODELS[:args.limit] if args.limit else BENCHMARK_MODELS
    
    dataset = build_dataset(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        models_to_process=models,
    )
    
    print("\n=== Dataset Statistics ===")
    print(f"Models: {dataset['statistics']['total_models']}")
    print(f"Operators: {dataset['statistics']['total_operators']}")
    print(f"\nModels by category:")
    for cat, count in dataset['statistics']['models_by_category'].items():
        print(f"  {cat}: {count}")
    print(f"\nTop operator types:")
    sorted_ops = sorted(dataset['statistics']['operator_types'].items(), key=lambda x: x[1], reverse=True)
    for op, count in sorted_ops[:10]:
        print(f"  {op}: {count}")
