#!/usr/bin/env python3
"""
Het-Benchmark Evaluation Runner

This script runs the complete benchmark evaluation pipeline:
1. Load models from Hugging Face or local cache
2. Load standard benchmark inputs
3. Profile operator performance across hardware platforms
4. Generate evaluation reports

Usage:
    # Run full benchmark
    python run_benchmark.py --data_dir ./benchmark_data --output_dir ./results
    
    # Run specific category
    python run_benchmark.py --data_dir ./benchmark_data --category LLM
    
    # Run specific model
    python run_benchmark.py --data_dir ./benchmark_data --model "Qwen2.5-7B"
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    data_dir: str
    output_dir: str
    models_dir: Optional[str] = None
    category: Optional[str] = None
    model_name: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16]


@dataclass
class ProfileResult:
    """Result of a single profiling run."""
    model_name: str
    category: str
    input_config: str
    batch_size: int
    latency_ms: float
    throughput_samples_per_sec: float
    memory_mb: float
    device: str
    timestamp: str


class BenchmarkDataLoader:
    """Load benchmark input data from generated files."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """Load data manifest."""
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run scripts/generate_benchmark_data.py first."
            )
        with open(manifest_path) as f:
            return json.load(f)
    
    def get_categories(self) -> List[str]:
        """Get available categories."""
        return list(self.manifest.get("categories", {}).keys())
    
    def get_inputs(self, category: str) -> List[Dict]:
        """Get input files for a category."""
        if category not in self.manifest.get("categories", {}):
            raise ValueError(f"Unknown category: {category}")
        return self.manifest["categories"][category]["inputs"]
    
    def load_input(self, category: str, filename: str) -> Dict[str, torch.Tensor]:
        """Load a specific input file."""
        filepath = self.data_dir / category / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        return torch.load(filepath, map_location="cpu")


class ModelRegistry:
    """Registry of models and their configurations."""
    
    # Model category mapping
    CATEGORY_MODELS = {
        "LLM": [
            "Qwen2.5-7B", "Mistral-7B", "Phi-3-mini", "BLOOM-560M", "GPT-2",
            "OPT-1.3B", "Falcon-7B", "StableLM-3B", "TinyLlama-1.1B", 
            "Pythia-1.4B", "GPT-Neo-1.3B"
        ],
        "CV": [
            "ResNet-50", "ViT-Base", "Swin-Base", "DINOv2-Base", "MobileNet-V2",
            "EfficientNet-B0", "ConvNeXt-Base", "RegNet-Y-4GF", "BEiT-Base", "DeiT-Base"
        ],
        "NLP": [
            "BERT-Base", "RoBERTa-Base", "T5-Base", "DistilBERT", 
            "ALBERT-Base", "BERT-Tiny", "BERT-Mini"
        ],
        "Audio": ["Whisper-Base", "Wav2Vec2-Base"],
        "Multimodal": ["CLIP-ViT-B/32", "BLIP-Base", "SigLIP-Base"],
    }
    
    # Hugging Face model IDs
    HF_MODEL_IDS = {
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
        "Mistral-7B": "mistralai/Mistral-7B-v0.1",
        "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "BLOOM-560M": "bigscience/bloom-560m",
        "GPT-2": "gpt2",
        "OPT-1.3B": "facebook/opt-1.3b",
        "ResNet-50": "microsoft/resnet-50",
        "ViT-Base": "google/vit-base-patch16-224",
        "BERT-Base": "bert-base-uncased",
        "CLIP-ViT-B/32": "openai/clip-vit-base-patch32",
        # Add more as needed
    }
    
    @classmethod
    def get_models_by_category(cls, category: str) -> List[str]:
        """Get model names for a category."""
        return cls.CATEGORY_MODELS.get(category, [])
    
    @classmethod
    def get_hf_id(cls, model_name: str) -> Optional[str]:
        """Get Hugging Face model ID."""
        return cls.HF_MODEL_IDS.get(model_name)
    
    @classmethod
    def get_category(cls, model_name: str) -> Optional[str]:
        """Get category for a model."""
        for category, models in cls.CATEGORY_MODELS.items():
            if model_name in models:
                return category
        return None


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_loader = BenchmarkDataLoader(config.data_dir)
        self.results: List[ProfileResult] = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _profile_inference(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor],
        model_name: str,
        category: str,
        input_config: str
    ) -> ProfileResult:
        """Profile model inference."""
        device = self.config.device
        model = model.to(device)
        model.eval()
        
        # Move inputs to device
        inputs_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }
        
        batch_size = inputs_device.get("batch_size", 1)
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                try:
                    _ = model(**{k: v for k, v in inputs_device.items() 
                               if isinstance(v, torch.Tensor)})
                except Exception:
                    # Try with just input_ids or pixel_values
                    if "input_ids" in inputs_device:
                        _ = model(inputs_device["input_ids"])
                    elif "pixel_values" in inputs_device:
                        _ = model(inputs_device["pixel_values"])
        
        # Synchronize
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Memory before
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.config.benchmark_iterations):
                try:
                    _ = model(**{k: v for k, v in inputs_device.items() 
                               if isinstance(v, torch.Tensor)})
                except Exception:
                    if "input_ids" in inputs_device:
                        _ = model(inputs_device["input_ids"])
                    elif "pixel_values" in inputs_device:
                        _ = model(inputs_device["pixel_values"])
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency_ms = (total_time / self.config.benchmark_iterations) * 1000
        throughput = (self.config.benchmark_iterations * batch_size) / total_time
        
        # Memory
        if device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            memory_mb = 0.0
        
        return ProfileResult(
            model_name=model_name,
            category=category,
            input_config=input_config,
            batch_size=batch_size,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput,
            memory_mb=memory_mb,
            device=device,
            timestamp=datetime.now().isoformat()
        )
    
    def run_category(self, category: str) -> List[ProfileResult]:
        """Run benchmark for a category."""
        print(f"\n{'='*60}")
        print(f"Running benchmark for category: {category}")
        print(f"{'='*60}")
        
        results = []
        inputs_list = self.data_loader.get_inputs(category)
        models = ModelRegistry.get_models_by_category(category)
        
        for model_name in models:
            if self.config.model_name and model_name != self.config.model_name:
                continue
            
            print(f"\n📦 Model: {model_name}")
            
            # Try to load model
            hf_id = ModelRegistry.get_hf_id(model_name)
            if not hf_id:
                print(f"  ⚠️ No HF ID found, skipping")
                continue
            
            try:
                model = self._load_model(hf_id, category)
            except Exception as e:
                print(f"  ❌ Failed to load model: {e}")
                continue
            
            # Run on each input configuration
            for input_info in inputs_list[:3]:  # Limit to first 3 for speed
                filename = input_info["file"]
                print(f"  📊 Input: {filename}")
                
                try:
                    inputs = self.data_loader.load_input(category, filename)
                    result = self._profile_inference(
                        model, inputs, model_name, category, filename
                    )
                    results.append(result)
                    print(f"    Latency: {result.latency_ms:.2f}ms, "
                          f"Throughput: {result.throughput_samples_per_sec:.1f} samples/s")
                except Exception as e:
                    print(f"    ❌ Error: {e}")
            
            # Clear memory
            del model
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
        
        return results
    
    def _load_model(self, hf_id: str, category: str) -> nn.Module:
        """Load model from Hugging Face."""
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageClassification
        
        if category == "LLM":
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, 
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
        elif category == "CV":
            model = AutoModelForImageClassification.from_pretrained(hf_id)
        else:
            model = AutoModel.from_pretrained(hf_id)
        
        return model
    
    def run(self) -> Dict[str, Any]:
        """Run the complete benchmark."""
        print("🚀 Het-Benchmark Evaluation Runner")
        print(f"Device: {self.config.device}")
        print(f"Data directory: {self.config.data_dir}")
        print(f"Output directory: {self.config.output_dir}")
        
        all_results = []
        
        categories = [self.config.category] if self.config.category else \
                     self.data_loader.get_categories()
        
        for category in categories:
            try:
                results = self.run_category(category)
                all_results.extend(results)
            except Exception as e:
                print(f"❌ Error running {category}: {e}")
        
        # Save results
        self._save_results(all_results)
        
        return {
            "total_runs": len(all_results),
            "categories": categories,
            "output_dir": self.config.output_dir
        }
    
    def _save_results(self, results: List[ProfileResult]):
        """Save benchmark results."""
        # Save as JSON
        json_path = Path(self.config.output_dir) / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n✅ Results saved to: {json_path}")
        
        # Save as CSV
        csv_path = Path(self.config.output_dir) / "benchmark_results.csv"
        with open(csv_path, "w") as f:
            if results:
                headers = list(asdict(results[0]).keys())
                f.write(",".join(headers) + "\n")
                for r in results:
                    values = [str(v) for v in asdict(r).values()]
                    f.write(",".join(values) + "\n")
        print(f"✅ CSV saved to: {csv_path}")
        
        # Generate summary
        self._generate_summary(results)
    
    def _generate_summary(self, results: List[ProfileResult]):
        """Generate benchmark summary."""
        summary_path = Path(self.config.output_dir) / "benchmark_summary.md"
        
        with open(summary_path, "w") as f:
            f.write("# Het-Benchmark Evaluation Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Device:** {self.config.device}\n")
            f.write(f"**Total Runs:** {len(results)}\n\n")
            
            # Group by category
            categories = {}
            for r in results:
                if r.category not in categories:
                    categories[r.category] = []
                categories[r.category].append(r)
            
            for category, cat_results in categories.items():
                f.write(f"## {category}\n\n")
                f.write("| Model | Input | Batch | Latency (ms) | Throughput (samples/s) | Memory (MB) |\n")
                f.write("|-------|-------|-------|--------------|------------------------|-------------|\n")
                
                for r in cat_results:
                    f.write(f"| {r.model_name} | {r.input_config} | {r.batch_size} | "
                           f"{r.latency_ms:.2f} | {r.throughput_samples_per_sec:.1f} | "
                           f"{r.memory_mb:.1f} |\n")
                f.write("\n")
        
        print(f"✅ Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Het-Benchmark evaluation")
    parser.add_argument("--data_dir", type=str, default="./benchmark_data",
                        help="Directory containing benchmark input data")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory to save results")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="Directory containing downloaded models")
    parser.add_argument("--category", type=str, default=None,
                        choices=["LLM", "CV", "NLP", "Audio", "Multimodal"],
                        help="Run benchmark for specific category only")
    parser.add_argument("--model", type=str, default=None,
                        help="Run benchmark for specific model only")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        category=args.category,
        model_name=args.model,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations
    )
    
    runner = BenchmarkRunner(config)
    summary = runner.run()
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Results saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()
