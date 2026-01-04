#!/usr/bin/env python3
"""
Het-Benchmark RGAT Inference Example

This script demonstrates how to use the trained RGAT model
for cross-platform performance prediction.
"""

import sys
import json
import torch
sys.path.insert(0, '..')

from src.rgat import RGAT


def main():
    print("="*60)
    print("RGAT Cross-Platform Performance Prediction")
    print("="*60)
    
    # 1. Load model configuration
    print("\n[1] Loading RGAT Model...")
    checkpoint = torch.load('../models/rgat_final.pt', map_location='cpu')
    
    config = checkpoint['config']
    print(f"    Model config: {config}")
    print(f"    Training epochs: {checkpoint['epoch']}")
    print(f"    Best loss: {checkpoint['loss']:.4f}")
    
    # 2. Initialize RGAT model
    print("\n[2] Initializing RGAT...")
    model = RGAT(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        num_relations=config['num_relations'],
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")
    
    # 3. Create sample input
    print("\n[3] Running Inference...")
    
    # Simulate knowledge graph data
    num_nodes = 100
    num_edges = 500
    
    x = torch.randn(num_nodes, config['in_channels'])
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, config['num_relations'], (num_edges,))
    
    # Run inference
    with torch.no_grad():
        output = model(x, edge_index, edge_type)
    
    print(f"    Input shape: {x.shape}")
    print(f"    Edge index shape: {edge_index.shape}")
    print(f"    Output shape: {output.shape}")
    
    # 4. Load experiment results
    print("\n[4] Loading Experiment Results...")
    import csv
    
    with open('../results/table8_cross_platform_prediction.csv', 'r') as f:
        reader = csv.DictReader(f)
        predictions = list(reader)
    
    print(f"    Total predictions: {len(predictions)}")
    
    # Calculate average error
    total_error = 0
    for pred in predictions:
        total_error += float(pred['prediction_error_percent'])
    avg_error = total_error / len(predictions)
    
    print(f"    Average prediction error: {avg_error:.2f}%")
    
    # Show sample predictions
    print("\n    Sample predictions:")
    for pred in predictions[:5]:
        print(f"      {pred['model']}: {pred['source_platform']} -> {pred['target_platform']}")
        print(f"        Predicted: {pred['predicted_latency_ms']}ms, Actual: {pred['actual_latency_ms']}ms")
        print(f"        Error: {pred['prediction_error_percent']}%")
    
    print("\n" + "="*60)
    print("RGAT Inference Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
