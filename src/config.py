"""
Het-Benchmark Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Main configuration for Het-Benchmark"""
    
    # Paths
    data_dir: str = "/workspace/het-benchmark/data"
    results_dir: str = "/workspace/het-benchmark/results"
    models_cache_dir: str = "/workspace/models"
    
    # Profiling settings
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    
    # COPA settings
    copa_num_samples: int = 1000
    copa_use_approximation: bool = True
    
    # Knowledge graph settings
    kg_embedding_dim: int = 128
    
    # RGAT settings
    rgat_hidden_dim: int = 256
    rgat_num_heads: int = 8
    rgat_num_layers: int = 3
    rgat_dropout: float = 0.1
    
    # Hardware platforms
    supported_platforms: List[str] = field(default_factory=lambda: [
        "nvidia_cuda",
        "amd_rocm", 
        "intel_oneapi",
        "huawei_cann",
        "cambricon_mlu",
    ])
    
    def __post_init__(self):
        # Create directories if they don't exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for a benchmark model"""
    model_id: str
    name: str
    category: str
    source: str = "huggingface"
    
    # Optional overrides
    max_sequence_length: Optional[int] = None
    batch_size: int = 1
    dtype: str = "float16"


# Default benchmark models (34 open-source models)
BENCHMARK_MODELS = [
    # LLM (12)
    ModelConfig("Qwen/Qwen2.5-7B", "Qwen2.5-7B", "LLM"),
    ModelConfig("mistralai/Mistral-7B-v0.1", "Mistral-7B", "LLM"),
    ModelConfig("google/gemma-2-2b", "Gemma-2-2B", "LLM"),
    ModelConfig("microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini", "LLM"),
    ModelConfig("bigscience/bloom-560m", "BLOOM-560M", "LLM"),
    ModelConfig("openai-community/gpt2", "GPT-2", "LLM"),
    ModelConfig("facebook/opt-1.3b", "OPT-1.3B", "LLM"),
    ModelConfig("tiiuae/falcon-7b", "Falcon-7B", "LLM"),
    ModelConfig("mosaicml/mpt-7b", "MPT-7B", "LLM"),
    ModelConfig("stabilityai/stablelm-3b-4e1t", "StableLM-3B", "LLM"),
    ModelConfig("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B", "LLM"),
    ModelConfig("EleutherAI/pythia-1.4b", "Pythia-1.4B", "LLM"),
    
    # CV (10)
    ModelConfig("microsoft/resnet-50", "ResNet-50", "CV"),
    ModelConfig("google/vit-base-patch16-224", "ViT-Base", "CV"),
    ModelConfig("microsoft/swin-base-patch4-window7-224", "Swin-Base", "CV"),
    ModelConfig("facebook/dinov2-base", "DINOv2-Base", "CV"),
    ModelConfig("google/mobilenet_v2_1.0_224", "MobileNet-V2", "CV"),
    ModelConfig("google/efficientnet-b0", "EfficientNet-B0", "CV"),
    ModelConfig("facebook/convnext-base-224", "ConvNeXt-Base", "CV"),
    ModelConfig("facebook/regnet-y-040", "RegNet-Y-4GF", "CV"),
    ModelConfig("microsoft/beit-base-patch16-224", "BEiT-Base", "CV"),
    ModelConfig("facebook/deit-base-patch16-224", "DeiT-Base", "CV"),
    
    # NLP (4)
    ModelConfig("google-bert/bert-base-uncased", "BERT-Base", "NLP"),
    ModelConfig("FacebookAI/roberta-base", "RoBERTa-Base", "NLP"),
    ModelConfig("google-t5/t5-base", "T5-Base", "NLP"),
    ModelConfig("distilbert/distilbert-base-uncased", "DistilBERT", "NLP"),
    
    # Audio (2)
    ModelConfig("openai/whisper-base", "Whisper-Base", "Audio"),
    ModelConfig("facebook/wav2vec2-base-960h", "Wav2Vec2-Base", "Audio"),
    
    # Multimodal (3)
    ModelConfig("openai/clip-vit-base-patch32", "CLIP-ViT-B/32", "Multimodal"),
    ModelConfig("Salesforce/blip-image-captioning-base", "BLIP-Base", "Multimodal"),
    ModelConfig("google/siglip-base-patch16-224", "SigLIP-Base", "Multimodal"),
    
    # Diffusion (3)
    ModelConfig("runwayml/stable-diffusion-v1-5", "SD-v1.5", "Diffusion"),
    ModelConfig("stabilityai/stable-diffusion-2-1-base", "SD-v2.1", "Diffusion"),
    ModelConfig("segmind/small-sd", "Small-SD", "Diffusion"),
]


# Operator categories
OPERATOR_CATEGORIES = {
    "matrix": ["MatMul", "Gemm", "Conv", "ConvTranspose", "BatchMatMul", "Linear"],
    "activation": ["Relu", "Gelu", "Silu", "Sigmoid", "Tanh", "Softmax", "Mish", "LeakyRelu"],
    "normalization": ["LayerNorm", "BatchNorm", "GroupNorm", "InstanceNorm", "RMSNorm"],
    "attention": ["MultiHeadAttention", "ScaledDotProductAttention", "SelfAttention", "CrossAttention"],
    "pooling": ["MaxPool", "AvgPool", "GlobalAvgPool", "AdaptiveAvgPool", "AdaptiveMaxPool"],
    "embedding": ["Embedding", "PositionalEncoding", "RotaryEmbedding"],
    "elementwise": ["Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log"],
    "reshape": ["Reshape", "Transpose", "Permute", "Flatten", "Squeeze", "Unsqueeze", "Concat", "Split"],
    "reduction": ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"],
    "other": ["Dropout", "Cast", "Constant", "Identity"],
}


# Hardware platform specifications
HARDWARE_SPECS = {
    "nvidia_cuda": {
        "name": "NVIDIA CUDA/cuDNN",
        "vendor": "NVIDIA",
        "type": "GPU",
        "library": "cuDNN",
        "version": "9.0+",
        "supported_dtypes": ["float32", "float16", "bfloat16", "int8"],
    },
    "amd_rocm": {
        "name": "AMD ROCm/MIGraphX",
        "vendor": "AMD",
        "type": "GPU",
        "library": "MIGraphX",
        "version": "6.0+",
        "supported_dtypes": ["float32", "float16", "bfloat16", "int8"],
    },
    "intel_oneapi": {
        "name": "Intel oneAPI/oneDNN",
        "vendor": "Intel",
        "type": "GPU/CPU",
        "library": "oneDNN",
        "version": "3.0+",
        "supported_dtypes": ["float32", "float16", "bfloat16", "int8"],
    },
    "huawei_cann": {
        "name": "Huawei Ascend CANN",
        "vendor": "Huawei",
        "type": "NPU",
        "library": "CANN",
        "version": "8.0+",
        "supported_dtypes": ["float32", "float16", "int8"],
    },
    "cambricon_mlu": {
        "name": "Cambricon MLU CNNL",
        "vendor": "Cambricon",
        "type": "MLU",
        "library": "CNNL",
        "version": "1.9+",
        "supported_dtypes": ["float32", "float16", "int8"],
    },
}
