#!/usr/bin/env python3
"""
Het-Benchmark Model Downloader

This script downloads all 34 models used in Het-Benchmark from Hugging Face.
Models are saved to the specified directory with consistent naming.

Usage:
    python download_models.py --output_dir ./models_hub
    python download_models.py --output_dir ./models_hub --category LLM
    python download_models.py --output_dir ./models_hub --model "Qwen2.5-7B"
"""

import os
import argparse
import json
from pathlib import Path

# Model registry with Hugging Face model IDs
MODEL_REGISTRY = {
    # LLM Models
    "Qwen2.5-7B": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "category": "LLM",
        "size_gb": 13.0
    },
    "Mistral-7B": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "category": "LLM",
        "size_gb": 13.0
    },
    "Phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "category": "LLM",
        "size_gb": 7.1
    },
    "BLOOM-560M": {
        "hf_id": "bigscience/bloom-560m",
        "category": "LLM",
        "size_gb": 1.0
    },
    "GPT-2": {
        "hf_id": "gpt2",
        "category": "LLM",
        "size_gb": 0.5
    },
    "OPT-1.3B": {
        "hf_id": "facebook/opt-1.3b",
        "category": "LLM",
        "size_gb": 2.4
    },
    "Falcon-7B": {
        "hf_id": "tiiuae/falcon-7b",
        "category": "LLM",
        "size_gb": 13.0
    },
    "StableLM-3B": {
        "hf_id": "stabilityai/stablelm-3b-4e1t",
        "category": "LLM",
        "size_gb": 5.6
    },
    "TinyLlama-1.1B": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "category": "LLM",
        "size_gb": 2.0
    },
    "Pythia-1.4B": {
        "hf_id": "EleutherAI/pythia-1.4b",
        "category": "LLM",
        "size_gb": 2.6
    },
    "GPT-Neo-1.3B": {
        "hf_id": "EleutherAI/gpt-neo-1.3B",
        "category": "LLM",
        "size_gb": 2.4
    },
    
    # CV Models
    "ResNet-50": {
        "hf_id": "microsoft/resnet-50",
        "category": "CV",
        "size_gb": 0.1
    },
    "ViT-Base": {
        "hf_id": "google/vit-base-patch16-224",
        "category": "CV",
        "size_gb": 0.3
    },
    "Swin-Base": {
        "hf_id": "microsoft/swin-base-patch4-window7-224",
        "category": "CV",
        "size_gb": 0.3
    },
    "DINOv2-Base": {
        "hf_id": "facebook/dinov2-base",
        "category": "CV",
        "size_gb": 0.3
    },
    "MobileNet-V2": {
        "hf_id": "google/mobilenet_v2_1.0_224",
        "category": "CV",
        "size_gb": 0.01
    },
    "EfficientNet-B0": {
        "hf_id": "google/efficientnet-b0",
        "category": "CV",
        "size_gb": 0.02
    },
    "ConvNeXt-Base": {
        "hf_id": "facebook/convnext-base-224",
        "category": "CV",
        "size_gb": 0.3
    },
    "RegNet-Y-4GF": {
        "hf_id": "facebook/regnet-y-040",
        "category": "CV",
        "size_gb": 0.08
    },
    "BEiT-Base": {
        "hf_id": "microsoft/beit-base-patch16-224",
        "category": "CV",
        "size_gb": 0.3
    },
    "DeiT-Base": {
        "hf_id": "facebook/deit-base-patch16-224",
        "category": "CV",
        "size_gb": 0.3
    },
    
    # NLP Models
    "BERT-Base": {
        "hf_id": "bert-base-uncased",
        "category": "NLP",
        "size_gb": 0.4
    },
    "RoBERTa-Base": {
        "hf_id": "roberta-base",
        "category": "NLP",
        "size_gb": 0.5
    },
    "T5-Base": {
        "hf_id": "t5-base",
        "category": "NLP",
        "size_gb": 0.9
    },
    "DistilBERT": {
        "hf_id": "distilbert-base-uncased",
        "category": "NLP",
        "size_gb": 0.3
    },
    "ALBERT-Base": {
        "hf_id": "albert-base-v2",
        "category": "NLP",
        "size_gb": 0.05
    },
    "BERT-Tiny": {
        "hf_id": "prajjwal1/bert-tiny",
        "category": "NLP",
        "size_gb": 0.02
    },
    "BERT-Mini": {
        "hf_id": "prajjwal1/bert-mini",
        "category": "NLP",
        "size_gb": 0.04
    },
    
    # Audio Models
    "Whisper-Base": {
        "hf_id": "openai/whisper-base",
        "category": "Audio",
        "size_gb": 0.3
    },
    "Wav2Vec2-Base": {
        "hf_id": "facebook/wav2vec2-base-960h",
        "category": "Audio",
        "size_gb": 0.4
    },
    
    # Multimodal Models
    "CLIP-ViT-B/32": {
        "hf_id": "openai/clip-vit-base-patch32",
        "category": "Multimodal",
        "size_gb": 0.6
    },
    "BLIP-Base": {
        "hf_id": "Salesforce/blip-image-captioning-base",
        "category": "Multimodal",
        "size_gb": 0.9
    },
    "SigLIP-Base": {
        "hf_id": "google/siglip-base-patch16-224",
        "category": "Multimodal",
        "size_gb": 0.4
    },
}


def download_model(model_name: str, output_dir: str, use_auth_token: str = None):
    """Download a single model from Hugging Face."""
    from huggingface_hub import snapshot_download
    
    if model_name not in MODEL_REGISTRY:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    model_info = MODEL_REGISTRY[model_name]
    hf_id = model_info["hf_id"]
    
    # Create output directory
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"📥 Downloading {model_name} ({model_info['size_gb']:.1f} GB)...")
    print(f"   From: {hf_id}")
    print(f"   To: {model_dir}")
    
    try:
        snapshot_download(
            repo_id=hf_id,
            local_dir=model_dir,
            token=use_auth_token,
            ignore_patterns=["*.md", "*.txt", "*.gitattributes"]
        )
        print(f"✅ Downloaded {model_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False


def download_all_models(output_dir: str, category: str = None, use_auth_token: str = None):
    """Download all models or models from a specific category."""
    models_to_download = []
    
    for name, info in MODEL_REGISTRY.items():
        if category is None or info["category"] == category:
            models_to_download.append(name)
    
    total_size = sum(MODEL_REGISTRY[m]["size_gb"] for m in models_to_download)
    print(f"\n📦 Downloading {len(models_to_download)} models (~{total_size:.1f} GB total)")
    print("="*60)
    
    success = 0
    failed = []
    
    for model_name in models_to_download:
        if download_model(model_name, output_dir, use_auth_token):
            success += 1
        else:
            failed.append(model_name)
    
    print("\n" + "="*60)
    print(f"✅ Successfully downloaded: {success}/{len(models_to_download)}")
    if failed:
        print(f"❌ Failed: {failed}")


def list_models():
    """List all available models."""
    print("\n📋 Available Models in Het-Benchmark")
    print("="*60)
    
    categories = {}
    for name, info in MODEL_REGISTRY.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))
    
    for cat, models in sorted(categories.items()):
        print(f"\n{cat} ({len(models)} models):")
        for name, info in models:
            print(f"  - {name}: {info['hf_id']} ({info['size_gb']:.1f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Download Het-Benchmark models")
    parser.add_argument("--output_dir", type=str, default="./models_hub",
                        help="Directory to save downloaded models")
    parser.add_argument("--category", type=str, default=None,
                        choices=["LLM", "CV", "NLP", "Audio", "Multimodal"],
                        help="Download only models from this category")
    parser.add_argument("--model", type=str, default=None,
                        help="Download a specific model by name")
    parser.add_argument("--list", action="store_true",
                        help="List all available models")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face auth token for gated models")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model:
        download_model(args.model, args.output_dir, args.token)
    else:
        download_all_models(args.output_dir, args.category, args.token)
    
    # Save model registry
    registry_path = os.path.join(args.output_dir, "model_registry.json")
    with open(registry_path, "w") as f:
        json.dump(MODEL_REGISTRY, f, indent=2)
    print(f"\n📄 Model registry saved to: {registry_path}")


if __name__ == "__main__":
    main()
