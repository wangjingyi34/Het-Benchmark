"""
Build Model Dataset for Het-Benchmark (V2 - Open Source Models Only)
Downloads models from Hugging Face and extracts operator information
All models are fully open source without gated access
"""

import os
import json
import torch
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


@dataclass
class ModelInfo:
    """Model metadata"""
    model_id: str
    name: str
    source: str
    category: str
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


# Updated 34 OPEN SOURCE models (no gated access required)
BENCHMARK_MODELS = [
    # Large Language Models (LLM) - 12 models (all open source)
    {"id": "Qwen/Qwen2.5-7B", "category": "LLM", "name": "Qwen2.5-7B"},
    {"id": "mistralai/Mistral-7B-v0.1", "category": "LLM", "name": "Mistral-7B"},
    {"id": "google/gemma-2-2b", "category": "LLM", "name": "Gemma-2-2B"},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "category": "LLM", "name": "Phi-3-mini"},
    {"id": "bigscience/bloom-560m", "category": "LLM", "name": "BLOOM-560M"},
    {"id": "openai-community/gpt2", "category": "LLM", "name": "GPT-2"},
    {"id": "facebook/opt-1.3b", "category": "LLM", "name": "OPT-1.3B"},
    {"id": "tiiuae/falcon-7b", "category": "LLM", "name": "Falcon-7B"},
    {"id": "mosaicml/mpt-7b", "category": "LLM", "name": "MPT-7B"},
    {"id": "stabilityai/stablelm-3b-4e1t", "category": "LLM", "name": "StableLM-3B"},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "category": "LLM", "name": "TinyLlama-1.1B"},
    {"id": "EleutherAI/pythia-1.4b", "category": "LLM", "name": "Pythia-1.4B"},
    
    # Computer Vision (CV) - 10 models (all open source)
    {"id": "microsoft/resnet-50", "category": "CV", "name": "ResNet-50"},
    {"id": "google/vit-base-patch16-224", "category": "CV", "name": "ViT-Base"},
    {"id": "microsoft/swin-base-patch4-window7-224", "category": "CV", "name": "Swin-Base"},
    {"id": "facebook/dinov2-base", "category": "CV", "name": "DINOv2-Base"},
    {"id": "google/mobilenet_v2_1.0_224", "category": "CV", "name": "MobileNet-V2"},
    {"id": "google/efficientnet-b0", "category": "CV", "name": "EfficientNet-B0"},
    {"id": "facebook/convnext-base-224", "category": "CV", "name": "ConvNeXt-Base"},
    {"id": "facebook/regnet-y-040", "category": "CV", "name": "RegNet-Y-4GF"},
    {"id": "microsoft/beit-base-patch16-224", "category": "CV", "name": "BEiT-Base"},
    {"id": "facebook/deit-base-patch16-224", "category": "CV", "name": "DeiT-Base"},
    
    # Natural Language Processing (NLP) - 4 models (all open source)
    {"id": "google-bert/bert-base-uncased", "category": "NLP", "name": "BERT-Base"},
    {"id": "FacebookAI/roberta-base", "category": "NLP", "name": "RoBERTa-Base"},
    {"id": "google-t5/t5-base", "category": "NLP", "name": "T5-Base"},
    {"id": "distilbert/distilbert-base-uncased", "category": "NLP", "name": "DistilBERT"},
    
    # Audio Models - 2 models (all open source)
    {"id": "openai/whisper-base", "category": "Audio", "name": "Whisper-Base"},
    {"id": "facebook/wav2vec2-base-960h", "category": "Audio", "name": "Wav2Vec2-Base"},
    
    # Multimodal Models - 3 models (all open source)
    {"id": "openai/clip-vit-base-patch32", "category": "Multimodal", "name": "CLIP-ViT-B/32"},
    {"id": "Salesforce/blip-image-captioning-base", "category": "Multimodal", "name": "BLIP-Base"},
    {"id": "google/siglip-base-patch16-224", "category": "Multimodal", "name": "SigLIP-Base"},
    
    # Diffusion Models - 3 models (all open source)
    {"id": "runwayml/stable-diffusion-v1-5", "category": "Diffusion", "name": "SD-v1.5"},
    {"id": "stabilityai/stable-diffusion-2-1-base", "category": "Diffusion", "name": "SD-v2.1"},
    {"id": "segmind/small-sd", "category": "Diffusion", "name": "Small-SD"},
]


# Operator categories mapping
OPERATOR_CATEGORIES = {
    "MatMul": "matrix", "Gemm": "matrix", "Conv": "matrix", "ConvTranspose": "matrix",
    "BatchMatMul": "matrix", "Linear": "matrix",
    "Relu": "activation", "Gelu": "activation", "Sigmoid": "activation", "Tanh": "activation",
    "Softmax": "activation", "Silu": "activation", "Swish": "activation", "Mish": "activation",
    "LeakyRelu": "activation", "Elu": "activation", "HardSwish": "activation",
    "LayerNorm": "normalization", "BatchNorm": "normalization", "GroupNorm": "normalization",
    "InstanceNorm": "normalization", "RMSNorm": "normalization",
    "Attention": "attention", "MultiHeadAttention": "attention", "ScaledDotProductAttention": "attention",
    "FlashAttention": "attention", "SelfAttention": "attention", "CrossAttention": "attention",
    "MaxPool": "pooling", "AvgPool": "pooling", "GlobalAvgPool": "pooling",
    "AdaptiveAvgPool": "pooling", "AdaptiveMaxPool": "pooling",
    "Add": "elementwise", "Sub": "elementwise", "Mul": "elementwise", "Div": "elementwise",
    "Pow": "elementwise", "Sqrt": "elementwise", "Exp": "elementwise", "Log": "elementwise",
    "Reshape": "reshape", "Transpose": "reshape", "Permute": "reshape", "Flatten": "reshape",
    "Squeeze": "reshape", "Unsqueeze": "reshape", "Concat": "reshape", "Split": "reshape",
    "Slice": "reshape", "Gather": "reshape", "Scatter": "reshape",
    "Embedding": "embedding", "PositionalEncoding": "embedding", "RotaryEmbedding": "embedding",
    "ReduceSum": "reduction", "ReduceMean": "reduction", "ReduceMax": "reduction",
    "ReduceMin": "reduction", "ReduceProd": "reduction",
    "Dropout": "other", "Cast": "other", "Constant": "other", "Identity": "other",
}


def get_operator_category(op_type: str) -> str:
    """Get category for an operator type"""
    op_normalized = op_type.replace("aten::", "").replace("torch.", "")
    op_normalized = op_normalized.split("_")[0].capitalize()
    
    for key, category in OPERATOR_CATEGORIES.items():
        if key.lower() in op_normalized.lower():
            return category
    
    return "other"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def extract_operators_from_modules(
    model: torch.nn.Module,
    model_id: str,
) -> List[OperatorInfo]:
    """Extract operators from model module structure"""
    operators = []
    op_counter = defaultdict(int)
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if module_type in ["Sequential", "ModuleList", "ModuleDict", "Module"]:
            continue
        
        op_counter[module_type] += 1
        op_id = f"{model_id}_{module_type}_{op_counter[module_type]}"
        
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
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        model = None
        
        if category == "LLM":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            )
            
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
            
        elif category in ["Multimodal", "Diffusion"]:
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
        
        if model is None:
            logger.error(f"Failed to load model: {model_id}")
            return None, []
        
        model = model.to(device)
        model.eval()
        
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
        
        operators = extract_operators_from_modules(model, model_info.model_id)
        
        logger.info(f"  Extracted {len(operators)} operators from {name}")
        logger.info(f"  Parameters: {num_params:,}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
    
    for i, model_spec in enumerate(models_to_process):
        logger.info(f"Progress: {i+1}/{len(models_to_process)}")
        
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
        
        time.sleep(1)
    
    dataset = {
        "version": "1.0.0",
        "name": "Het-Benchmark Model Dataset",
        "description": "34 representative open-source AI models for heterogeneous chip evaluation",
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
