#!/usr/bin/env python3
"""
Standard Input Dataset Generator for Het-Benchmark
Generates 1000+ standardized input samples for different model types:
- LLM: Text prompts with various lengths and complexities
- VLM: Image-text pairs
- Diffusion: Text prompts for image generation
- CV: Standard image inputs (ImageNet-style)
"""

import json
import os
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class TextInput:
    """Standard text input for LLM models"""
    input_id: str
    text: str
    token_count: int
    category: str  # qa, summarization, translation, code, reasoning
    complexity: str  # simple, medium, complex
    language: str


@dataclass
class ImageInput:
    """Standard image input specification for CV models"""
    input_id: str
    resolution: Tuple[int, int]
    channels: int
    format: str  # RGB, BGR, Grayscale
    normalization: str  # imagenet, [-1,1], [0,1]
    category: str  # classification, detection, segmentation


@dataclass
class ImageTextInput:
    """Standard image-text pair for VLM models"""
    input_id: str
    image_spec: Dict
    text: str
    task: str  # captioning, vqa, reasoning


@dataclass
class DiffusionInput:
    """Standard input for diffusion models"""
    input_id: str
    prompt: str
    negative_prompt: str
    resolution: Tuple[int, int]
    steps: int
    guidance_scale: float
    category: str  # portrait, landscape, abstract, realistic


class StandardInputGenerator:
    """Generate standardized input datasets for Het-Benchmark"""
    
    def __init__(self, output_dir: str = "data/standard_inputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # LLM prompt templates
        self.qa_templates = [
            "What is {topic}?",
            "Explain the concept of {topic} in detail.",
            "How does {topic} work?",
            "What are the main advantages of {topic}?",
            "Compare and contrast {topic1} and {topic2}.",
            "Describe the history of {topic}.",
            "What are the applications of {topic}?",
            "Analyze the impact of {topic} on {domain}.",
        ]
        
        self.topics = [
            "machine learning", "deep learning", "neural networks",
            "transformer architecture", "attention mechanism", "backpropagation",
            "gradient descent", "convolutional neural networks", "recurrent neural networks",
            "natural language processing", "computer vision", "reinforcement learning",
            "generative AI", "large language models", "diffusion models",
            "quantum computing", "blockchain", "cloud computing",
            "edge computing", "federated learning", "transfer learning",
            "meta-learning", "few-shot learning", "zero-shot learning",
            "knowledge graphs", "graph neural networks", "self-supervised learning",
            "contrastive learning", "multimodal learning", "vision transformers",
        ]
        
        self.domains = [
            "healthcare", "finance", "education", "manufacturing",
            "transportation", "agriculture", "energy", "retail",
            "entertainment", "security", "research", "government",
        ]
        
        self.code_tasks = [
            "Write a Python function to {task}.",
            "Implement a {algorithm} algorithm in Python.",
            "Debug the following code: {code_snippet}",
            "Optimize this function for better performance: {code_snippet}",
            "Convert this Python code to {language}: {code_snippet}",
        ]
        
        self.code_algorithms = [
            "binary search", "quicksort", "merge sort", "depth-first search",
            "breadth-first search", "dynamic programming", "greedy algorithm",
            "Dijkstra's shortest path", "A* search", "hash table",
        ]
        
        self.reasoning_templates = [
            "If {premise1} and {premise2}, what can we conclude?",
            "Solve this problem step by step: {problem}",
            "What is the logical flaw in this argument: {argument}?",
            "Given the following constraints, find the optimal solution: {constraints}",
        ]
        
        # Diffusion prompt templates
        self.diffusion_styles = [
            "photorealistic", "oil painting", "watercolor", "digital art",
            "anime style", "sketch", "3D render", "cinematic",
            "minimalist", "surrealist", "impressionist", "pop art",
        ]
        
        self.diffusion_subjects = [
            "a serene mountain landscape at sunset",
            "a futuristic city skyline",
            "a portrait of a wise elderly person",
            "an abstract representation of music",
            "a cozy coffee shop interior",
            "a mystical forest with glowing plants",
            "an underwater coral reef scene",
            "a steampunk mechanical device",
            "a peaceful zen garden",
            "a dramatic stormy ocean",
        ]
        
        # VLM task templates
        self.vqa_questions = [
            "What is the main object in this image?",
            "How many {object} are there in the image?",
            "What color is the {object}?",
            "Where is the {object} located?",
            "What is happening in this image?",
            "Describe the mood of this image.",
            "What time of day does this image appear to be taken?",
            "Is there any text visible in the image?",
        ]
        
        self.vqa_objects = [
            "person", "car", "dog", "cat", "tree", "building",
            "chair", "table", "book", "phone", "computer", "flower",
        ]
    
    def generate_llm_inputs(self, count: int = 300) -> List[Dict]:
        """Generate standardized LLM text inputs"""
        inputs = []
        categories = ["qa", "summarization", "translation", "code", "reasoning"]
        complexities = ["simple", "medium", "complex"]
        
        for i in range(count):
            category = random.choice(categories)
            complexity = random.choice(complexities)
            
            if category == "qa":
                template = random.choice(self.qa_templates)
                topic = random.choice(self.topics)
                topic2 = random.choice([t for t in self.topics if t != topic])
                domain = random.choice(self.domains)
                text = template.format(topic=topic, topic1=topic, topic2=topic2, domain=domain)
                
            elif category == "summarization":
                # Generate longer text for summarization
                paragraphs = random.randint(2, 5) if complexity != "simple" else 1
                topic = random.choice(self.topics)
                text = f"Please summarize the following text about {topic}:\n\n"
                for _ in range(paragraphs):
                    text += f"{topic.capitalize()} is an important concept in modern technology. "
                    text += f"It has applications in {random.choice(self.domains)} and {random.choice(self.domains)}. "
                    text += f"Researchers continue to explore new ways to improve {topic}. "
                
            elif category == "translation":
                source_lang = random.choice(["English", "Chinese", "French", "German", "Spanish"])
                target_lang = random.choice([l for l in ["English", "Chinese", "French", "German", "Spanish"] if l != source_lang])
                topic = random.choice(self.topics)
                text = f"Translate the following text from {source_lang} to {target_lang}:\n\n"
                text += f"{topic.capitalize()} is a fundamental concept that has revolutionized many industries."
                
            elif category == "code":
                task_template = random.choice(self.code_tasks)
                algorithm = random.choice(self.code_algorithms)
                code_snippet = "def example(): pass  # placeholder"
                language = random.choice(["Java", "C++", "JavaScript", "Go", "Rust"])
                text = task_template.format(
                    task=f"implement {algorithm}",
                    algorithm=algorithm,
                    code_snippet=code_snippet,
                    language=language
                )
                
            else:  # reasoning
                template = random.choice(self.reasoning_templates)
                text = template.format(
                    premise1="all A are B",
                    premise2="all B are C",
                    problem=f"calculate the optimal path in a graph with {random.randint(5, 20)} nodes",
                    argument="if it rains, the ground is wet; the ground is wet; therefore it rained",
                    constraints=f"maximize profit with budget constraint of ${random.randint(1000, 10000)}"
                )
            
            # Estimate token count (rough approximation: ~4 chars per token)
            token_count = len(text) // 4
            
            # Adjust complexity based on token count
            if complexity == "complex" and token_count < 100:
                text = text + " " + text  # Double the text for complexity
                token_count = len(text) // 4
            
            input_id = hashlib.md5(f"llm_{i}_{text[:50]}".encode()).hexdigest()[:12]
            
            inputs.append(asdict(TextInput(
                input_id=f"llm_{input_id}",
                text=text,
                token_count=token_count,
                category=category,
                complexity=complexity,
                language="en"
            )))
        
        return inputs
    
    def generate_cv_inputs(self, count: int = 250) -> List[Dict]:
        """Generate standardized CV image input specifications"""
        inputs = []
        
        resolutions = [
            (224, 224),   # Standard ImageNet
            (256, 256),   # Common alternative
            (384, 384),   # ViT-Large
            (512, 512),   # High resolution
            (640, 640),   # YOLO
            (1024, 1024), # High-res models
        ]
        
        categories = ["classification", "detection", "segmentation", "feature_extraction"]
        normalizations = ["imagenet", "[-1,1]", "[0,1]"]
        
        for i in range(count):
            resolution = random.choice(resolutions)
            category = random.choice(categories)
            normalization = random.choice(normalizations)
            
            input_id = hashlib.md5(f"cv_{i}_{resolution}_{category}".encode()).hexdigest()[:12]
            
            inputs.append(asdict(ImageInput(
                input_id=f"cv_{input_id}",
                resolution=resolution,
                channels=3,
                format="RGB",
                normalization=normalization,
                category=category
            )))
        
        return inputs
    
    def generate_vlm_inputs(self, count: int = 200) -> List[Dict]:
        """Generate standardized VLM image-text pair inputs"""
        inputs = []
        
        tasks = ["captioning", "vqa", "reasoning", "grounding"]
        resolutions = [(224, 224), (336, 336), (384, 384), (448, 448)]
        
        for i in range(count):
            task = random.choice(tasks)
            resolution = random.choice(resolutions)
            
            if task == "captioning":
                text = "Describe this image in detail."
            elif task == "vqa":
                question_template = random.choice(self.vqa_questions)
                obj = random.choice(self.vqa_objects)
                text = question_template.format(object=obj)
            elif task == "reasoning":
                text = random.choice([
                    "What will happen next in this scene?",
                    "Why might the person in the image be doing this?",
                    "What is the relationship between the objects in this image?",
                    "Explain the context of this image.",
                ])
            else:  # grounding
                obj = random.choice(self.vqa_objects)
                text = f"Locate and describe the {obj} in this image."
            
            input_id = hashlib.md5(f"vlm_{i}_{task}_{text[:30]}".encode()).hexdigest()[:12]
            
            inputs.append(asdict(ImageTextInput(
                input_id=f"vlm_{input_id}",
                image_spec={
                    "resolution": resolution,
                    "channels": 3,
                    "format": "RGB",
                    "normalization": "[-1,1]"
                },
                text=text,
                task=task
            )))
        
        return inputs
    
    def generate_diffusion_inputs(self, count: int = 250) -> List[Dict]:
        """Generate standardized diffusion model inputs"""
        inputs = []
        
        resolutions = [
            (512, 512),   # SD 1.x
            (768, 768),   # SD 2.x
            (1024, 1024), # SDXL
            (512, 768),   # Portrait
            (768, 512),   # Landscape
        ]
        
        step_options = [20, 25, 30, 50]
        guidance_options = [7.0, 7.5, 8.0, 10.0, 12.0]
        categories = ["portrait", "landscape", "abstract", "realistic", "artistic", "fantasy"]
        
        for i in range(count):
            style = random.choice(self.diffusion_styles)
            subject = random.choice(self.diffusion_subjects)
            resolution = random.choice(resolutions)
            steps = random.choice(step_options)
            guidance = random.choice(guidance_options)
            category = random.choice(categories)
            
            prompt = f"{subject}, {style}, highly detailed, professional quality"
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"
            
            input_id = hashlib.md5(f"diff_{i}_{prompt[:30]}".encode()).hexdigest()[:12]
            
            inputs.append(asdict(DiffusionInput(
                input_id=f"diff_{input_id}",
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                steps=steps,
                guidance_scale=guidance,
                category=category
            )))
        
        return inputs
    
    def generate_full_dataset(self) -> Dict[str, Any]:
        """Generate complete standardized input dataset"""
        print("Generating LLM inputs...")
        llm_inputs = self.generate_llm_inputs(300)
        
        print("Generating CV inputs...")
        cv_inputs = self.generate_cv_inputs(250)
        
        print("Generating VLM inputs...")
        vlm_inputs = self.generate_vlm_inputs(200)
        
        print("Generating Diffusion inputs...")
        diffusion_inputs = self.generate_diffusion_inputs(250)
        
        dataset = {
            "metadata": {
                "name": "Het-Benchmark Standard Input Dataset",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(llm_inputs) + len(cv_inputs) + len(vlm_inputs) + len(diffusion_inputs),
                "categories": {
                    "llm": len(llm_inputs),
                    "cv": len(cv_inputs),
                    "vlm": len(vlm_inputs),
                    "diffusion": len(diffusion_inputs)
                }
            },
            "llm_inputs": llm_inputs,
            "cv_inputs": cv_inputs,
            "vlm_inputs": vlm_inputs,
            "diffusion_inputs": diffusion_inputs
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any]):
        """Save dataset to files"""
        # Save complete dataset
        output_path = os.path.join(self.output_dir, "standard_inputs.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved complete dataset to {output_path}")
        
        # Save individual category files
        for category in ["llm_inputs", "cv_inputs", "vlm_inputs", "diffusion_inputs"]:
            category_path = os.path.join(self.output_dir, f"{category}.json")
            with open(category_path, 'w', encoding='utf-8') as f:
                json.dump(dataset[category], f, indent=2, ensure_ascii=False)
            print(f"Saved {category} to {category_path}")
        
        # Generate statistics
        stats = self.generate_statistics(dataset)
        stats_path = os.path.join(self.output_dir, "dataset_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")
        
        return output_path
    
    def generate_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        stats = {
            "total_samples": dataset["metadata"]["total_samples"],
            "categories": dataset["metadata"]["categories"],
            "llm_stats": {
                "total": len(dataset["llm_inputs"]),
                "by_category": {},
                "by_complexity": {},
                "avg_token_count": 0,
            },
            "cv_stats": {
                "total": len(dataset["cv_inputs"]),
                "by_resolution": {},
                "by_category": {},
            },
            "vlm_stats": {
                "total": len(dataset["vlm_inputs"]),
                "by_task": {},
            },
            "diffusion_stats": {
                "total": len(dataset["diffusion_inputs"]),
                "by_resolution": {},
                "by_category": {},
                "avg_steps": 0,
            }
        }
        
        # LLM statistics
        token_counts = []
        for inp in dataset["llm_inputs"]:
            cat = inp["category"]
            comp = inp["complexity"]
            stats["llm_stats"]["by_category"][cat] = stats["llm_stats"]["by_category"].get(cat, 0) + 1
            stats["llm_stats"]["by_complexity"][comp] = stats["llm_stats"]["by_complexity"].get(comp, 0) + 1
            token_counts.append(inp["token_count"])
        stats["llm_stats"]["avg_token_count"] = sum(token_counts) / len(token_counts) if token_counts else 0
        
        # CV statistics
        for inp in dataset["cv_inputs"]:
            res = str(inp["resolution"])
            cat = inp["category"]
            stats["cv_stats"]["by_resolution"][res] = stats["cv_stats"]["by_resolution"].get(res, 0) + 1
            stats["cv_stats"]["by_category"][cat] = stats["cv_stats"]["by_category"].get(cat, 0) + 1
        
        # VLM statistics
        for inp in dataset["vlm_inputs"]:
            task = inp["task"]
            stats["vlm_stats"]["by_task"][task] = stats["vlm_stats"]["by_task"].get(task, 0) + 1
        
        # Diffusion statistics
        steps_list = []
        for inp in dataset["diffusion_inputs"]:
            res = str(inp["resolution"])
            cat = inp["category"]
            stats["diffusion_stats"]["by_resolution"][res] = stats["diffusion_stats"]["by_resolution"].get(res, 0) + 1
            stats["diffusion_stats"]["by_category"][cat] = stats["diffusion_stats"]["by_category"].get(cat, 0) + 1
            steps_list.append(inp["steps"])
        stats["diffusion_stats"]["avg_steps"] = sum(steps_list) / len(steps_list) if steps_list else 0
        
        return stats


def main():
    """Main function to generate standard input dataset"""
    generator = StandardInputGenerator(output_dir="data/standard_inputs")
    
    print("=" * 60)
    print("Het-Benchmark Standard Input Dataset Generator")
    print("=" * 60)
    
    # Generate dataset
    dataset = generator.generate_full_dataset()
    
    # Save dataset
    output_path = generator.save_dataset(dataset)
    
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"Total samples: {dataset['metadata']['total_samples']}")
    print(f"  - LLM inputs: {len(dataset['llm_inputs'])}")
    print(f"  - CV inputs: {len(dataset['cv_inputs'])}")
    print(f"  - VLM inputs: {len(dataset['vlm_inputs'])}")
    print(f"  - Diffusion inputs: {len(dataset['diffusion_inputs'])}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
