#!/usr/bin/env python3
"""
Het-Benchmark Standard Input/Output Data Generator

This script generates standardized benchmark input and output data for reproducible evaluation.
Data is saved as PyTorch tensors (.pt files) organized by model category.

Usage:
    python generate_benchmark_data.py --output_dir ./benchmark_data
    python generate_benchmark_data.py --output_dir ./benchmark_data --category LLM
"""

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# Standard input configurations for each category
INPUT_CONFIGS = {
    "LLM": {
        "batch_sizes": [1, 4, 8, 16],
        "seq_lengths": [128, 256, 512, 1024, 2048],
        "vocab_size": 152064,  # Qwen2.5 vocab size
        "dtype": torch.long,
        "description": "Tokenized text sequences"
    },
    "CV": {
        "batch_sizes": [1, 4, 8, 16, 32],
        "image_sizes": [(224, 224), (384, 384), (512, 512)],
        "channels": 3,
        "dtype": torch.float32,
        "description": "RGB image tensors normalized to [0, 1]"
    },
    "NLP": {
        "batch_sizes": [1, 8, 16, 32],
        "seq_lengths": [64, 128, 256, 512],
        "vocab_size": 30522,  # BERT vocab size
        "dtype": torch.long,
        "description": "Tokenized text sequences for encoder models"
    },
    "Audio": {
        "batch_sizes": [1, 4, 8],
        "audio_lengths": [16000, 32000, 48000, 80000],  # 1s, 2s, 3s, 5s at 16kHz
        "sample_rate": 16000,
        "dtype": torch.float32,
        "description": "Raw audio waveforms"
    },
    "Multimodal": {
        "batch_sizes": [1, 4, 8],
        "image_sizes": [(224, 224), (336, 336)],
        "seq_lengths": [32, 64, 77],  # CLIP max length is 77
        "vocab_size": 49408,  # CLIP vocab size
        "dtype_image": torch.float32,
        "dtype_text": torch.long,
        "description": "Image-text pairs"
    },
    "Diffusion": {
        "batch_sizes": [1, 2, 4],
        "latent_sizes": [(64, 64), (96, 96), (128, 128)],
        "latent_channels": 4,
        "seq_lengths": [77],  # CLIP text encoder
        "dtype": torch.float32,
        "description": "Latent noise tensors and text embeddings"
    }
}


def generate_llm_inputs(output_dir: str, config: dict) -> List[Dict]:
    """Generate LLM input tensors."""
    inputs = []
    category_dir = os.path.join(output_dir, "LLM")
    os.makedirs(category_dir, exist_ok=True)
    
    for batch_size in config["batch_sizes"]:
        for seq_len in config["seq_lengths"]:
            # Generate random token IDs
            input_ids = torch.randint(
                0, config["vocab_size"], 
                (batch_size, seq_len), 
                dtype=config["dtype"]
            )
            
            # Attention mask (all ones for simplicity)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            
            # Save tensors
            filename = f"input_b{batch_size}_s{seq_len}.pt"
            filepath = os.path.join(category_dir, filename)
            
            torch.save({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "batch_size": batch_size,
                "seq_length": seq_len,
            }, filepath)
            
            inputs.append({
                "file": filename,
                "batch_size": batch_size,
                "seq_length": seq_len,
                "shape": list(input_ids.shape),
                "size_mb": os.path.getsize(filepath) / (1024 * 1024)
            })
            
            print(f"  Generated: {filename} ({input_ids.shape})")
    
    return inputs


def generate_cv_inputs(output_dir: str, config: dict) -> List[Dict]:
    """Generate CV input tensors."""
    inputs = []
    category_dir = os.path.join(output_dir, "CV")
    os.makedirs(category_dir, exist_ok=True)
    
    for batch_size in config["batch_sizes"]:
        for h, w in config["image_sizes"]:
            # Generate random normalized images
            images = torch.rand(
                batch_size, config["channels"], h, w,
                dtype=config["dtype"]
            )
            
            filename = f"input_b{batch_size}_{h}x{w}.pt"
            filepath = os.path.join(category_dir, filename)
            
            torch.save({
                "pixel_values": images,
                "batch_size": batch_size,
                "height": h,
                "width": w,
            }, filepath)
            
            inputs.append({
                "file": filename,
                "batch_size": batch_size,
                "image_size": [h, w],
                "shape": list(images.shape),
                "size_mb": os.path.getsize(filepath) / (1024 * 1024)
            })
            
            print(f"  Generated: {filename} ({images.shape})")
    
    return inputs


def generate_nlp_inputs(output_dir: str, config: dict) -> List[Dict]:
    """Generate NLP input tensors."""
    inputs = []
    category_dir = os.path.join(output_dir, "NLP")
    os.makedirs(category_dir, exist_ok=True)
    
    for batch_size in config["batch_sizes"]:
        for seq_len in config["seq_lengths"]:
            input_ids = torch.randint(
                0, config["vocab_size"],
                (batch_size, seq_len),
                dtype=config["dtype"]
            )
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
            
            filename = f"input_b{batch_size}_s{seq_len}.pt"
            filepath = os.path.join(category_dir, filename)
            
            torch.save({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "batch_size": batch_size,
                "seq_length": seq_len,
            }, filepath)
            
            inputs.append({
                "file": filename,
                "batch_size": batch_size,
                "seq_length": seq_len,
                "shape": list(input_ids.shape),
                "size_mb": os.path.getsize(filepath) / (1024 * 1024)
            })
            
            print(f"  Generated: {filename} ({input_ids.shape})")
    
    return inputs


def generate_audio_inputs(output_dir: str, config: dict) -> List[Dict]:
    """Generate Audio input tensors."""
    inputs = []
    category_dir = os.path.join(output_dir, "Audio")
    os.makedirs(category_dir, exist_ok=True)
    
    for batch_size in config["batch_sizes"]:
        for audio_len in config["audio_lengths"]:
            # Generate random audio waveform
            waveform = torch.randn(
                batch_size, audio_len,
                dtype=config["dtype"]
            )
            
            duration_s = audio_len / config["sample_rate"]
            filename = f"input_b{batch_size}_{duration_s:.1f}s.pt"
            filepath = os.path.join(category_dir, filename)
            
            torch.save({
                "input_values": waveform,
                "batch_size": batch_size,
                "audio_length": audio_len,
                "sample_rate": config["sample_rate"],
            }, filepath)
            
            inputs.append({
                "file": filename,
                "batch_size": batch_size,
                "audio_length": audio_len,
                "duration_s": duration_s,
                "shape": list(waveform.shape),
                "size_mb": os.path.getsize(filepath) / (1024 * 1024)
            })
            
            print(f"  Generated: {filename} ({waveform.shape})")
    
    return inputs


def generate_multimodal_inputs(output_dir: str, config: dict) -> List[Dict]:
    """Generate Multimodal input tensors."""
    inputs = []
    category_dir = os.path.join(output_dir, "Multimodal")
    os.makedirs(category_dir, exist_ok=True)
    
    for batch_size in config["batch_sizes"]:
        for h, w in config["image_sizes"]:
            for seq_len in config["seq_lengths"]:
                # Image input
                images = torch.rand(
                    batch_size, 3, h, w,
                    dtype=config["dtype_image"]
                )
                
                # Text input
                input_ids = torch.randint(
                    0, config["vocab_size"],
                    (batch_size, seq_len),
                    dtype=config["dtype_text"]
                )
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
                
                filename = f"input_b{batch_size}_{h}x{w}_s{seq_len}.pt"
                filepath = os.path.join(category_dir, filename)
                
                torch.save({
                    "pixel_values": images,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "batch_size": batch_size,
                    "image_size": [h, w],
                    "seq_length": seq_len,
                }, filepath)
                
                inputs.append({
                    "file": filename,
                    "batch_size": batch_size,
                    "image_size": [h, w],
                    "seq_length": seq_len,
                    "image_shape": list(images.shape),
                    "text_shape": list(input_ids.shape),
                    "size_mb": os.path.getsize(filepath) / (1024 * 1024)
                })
                
                print(f"  Generated: {filename}")
    
    return inputs


def generate_diffusion_inputs(output_dir: str, config: dict) -> List[Dict]:
    """Generate Diffusion model input tensors."""
    inputs = []
    category_dir = os.path.join(output_dir, "Diffusion")
    os.makedirs(category_dir, exist_ok=True)
    
    for batch_size in config["batch_sizes"]:
        for h, w in config["latent_sizes"]:
            # Latent noise
            latents = torch.randn(
                batch_size, config["latent_channels"], h, w,
                dtype=config["dtype"]
            )
            
            # Timesteps
            timesteps = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
            
            # Text embeddings (from CLIP)
            text_embeds = torch.randn(
                batch_size, 77, 768,  # CLIP-ViT-L hidden size
                dtype=config["dtype"]
            )
            
            filename = f"input_b{batch_size}_{h}x{w}.pt"
            filepath = os.path.join(category_dir, filename)
            
            torch.save({
                "latents": latents,
                "timesteps": timesteps,
                "encoder_hidden_states": text_embeds,
                "batch_size": batch_size,
                "latent_size": [h, w],
            }, filepath)
            
            inputs.append({
                "file": filename,
                "batch_size": batch_size,
                "latent_size": [h, w],
                "latent_shape": list(latents.shape),
                "size_mb": os.path.getsize(filepath) / (1024 * 1024)
            })
            
            print(f"  Generated: {filename} ({latents.shape})")
    
    return inputs


def generate_all_inputs(output_dir: str, category: str = None):
    """Generate all benchmark inputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    generators = {
        "LLM": generate_llm_inputs,
        "CV": generate_cv_inputs,
        "NLP": generate_nlp_inputs,
        "Audio": generate_audio_inputs,
        "Multimodal": generate_multimodal_inputs,
        "Diffusion": generate_diffusion_inputs,
    }
    
    manifest = {
        "description": "Het-Benchmark Standard Input Data",
        "version": "1.0",
        "categories": {}
    }
    
    categories_to_generate = [category] if category else list(INPUT_CONFIGS.keys())
    
    for cat in categories_to_generate:
        if cat not in INPUT_CONFIGS:
            print(f"Unknown category: {cat}")
            continue
        
        print(f"\n📦 Generating {cat} inputs...")
        config = INPUT_CONFIGS[cat]
        inputs = generators[cat](output_dir, config)
        
        manifest["categories"][cat] = {
            "description": config["description"],
            "inputs": inputs,
            "total_files": len(inputs),
            "total_size_mb": sum(i["size_mb"] for i in inputs)
        }
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Manifest saved to: {manifest_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Generation Summary")
    print("="*60)
    total_files = 0
    total_size = 0
    for cat, info in manifest["categories"].items():
        print(f"{cat}: {info['total_files']} files, {info['total_size_mb']:.2f} MB")
        total_files += info["total_files"]
        total_size += info["total_size_mb"]
    print(f"\nTotal: {total_files} files, {total_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate Het-Benchmark input data")
    parser.add_argument("--output_dir", type=str, default="./benchmark_data",
                        help="Directory to save generated data")
    parser.add_argument("--category", type=str, default=None,
                        choices=["LLM", "CV", "NLP", "Audio", "Multimodal", "Diffusion"],
                        help="Generate data for specific category only")
    
    args = parser.parse_args()
    
    print("🚀 Het-Benchmark Input Data Generator")
    print("="*60)
    
    generate_all_inputs(args.output_dir, args.category)


if __name__ == "__main__":
    main()
