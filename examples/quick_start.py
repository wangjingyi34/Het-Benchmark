#!/usr/bin/env python3
"""
Het-Benchmark Quick Start Example

This script demonstrates the basic usage of Het-Benchmark framework:
1. Loading model dataset
2. Building knowledge graph
3. Running COPA analysis
4. Using RGAT for performance prediction
"""

import sys
import json
sys.path.insert(0, '..')

from src.hal import HAL, HardwarePlatform
from src.copa import COPA, MicroBenchmarker
from src.moh_kg import MOHKG


def main():
    print("="*60)
    print("Het-Benchmark Quick Start Example")
    print("="*60)
    
    # 1. Load model dataset
    print("\n[1] Loading Model Dataset...")
    with open('../data/model_dataset.json', 'r') as f:
        model_data = json.load(f)
    
    print(f"    Loaded {len(model_data['models'])} models")
    print(f"    Categories: {set(m['category'] for m in model_data['models'])}")
    
    # 2. Initialize Hardware Abstraction Layer
    print("\n[2] Initializing HAL...")
    hal = HAL()
    print(f"    Supported platforms: {[p.value for p in HardwarePlatform]}")
    
    # 3. Build Knowledge Graph
    print("\n[3] Building Knowledge Graph...")
    kg = MOHKG()
    print(f"    Hardware platforms: {len(kg.hardware)}")
    print(f"    Operator types: {len(kg.operator_types)}")
    
    # 4. Initialize COPA
    print("\n[4] Initializing COPA...")
    copa = COPA()
    print("    COPA ready for two-stage Shapley analysis")
    
    # 5. Load pre-computed knowledge graph
    print("\n[5] Loading Pre-computed Knowledge Graph...")
    with open('../data/moh_kg.json', 'r') as f:
        kg_data = json.load(f)
    
    stats = kg_data['statistics']
    print(f"    Nodes: {stats['total_nodes']}")
    print(f"    Edges: {stats['total_edges']}")
    print(f"    Node types: {stats['node_types']}")
    print(f"    Edge types: {list(stats['edge_types'].keys())}")
    
    # 6. Show sample model analysis
    print("\n[6] Sample Model Analysis...")
    sample_model = model_data['models'][0]
    print(f"    Model: {sample_model['name']}")
    print(f"    Category: {sample_model['category']}")
    print(f"    Parameters: {sample_model['parameters']:,}")
    print(f"    Operators: {len(sample_model['operators'])}")
    
    # Show operator distribution
    op_types = {}
    for op in sample_model['operators']:
        op_type = op['type']
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    print(f"    Operator distribution:")
    for op_type, count in sorted(op_types.items(), key=lambda x: -x[1])[:5]:
        print(f"      - {op_type}: {count}")
    
    print("\n" + "="*60)
    print("Quick Start Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
