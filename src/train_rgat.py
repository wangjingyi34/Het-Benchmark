#!/usr/bin/env python3
"""
Train RGAT (Relational Graph Attention Network) for Het-Benchmark
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


class RGATLayer(nn.Module):
    """Relational Graph Attention Layer"""
    
    def __init__(self, in_dim: int, out_dim: int, num_relations: int, num_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # Relation-specific transformations
        self.W_r = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) 
            for _ in range(num_relations)
        ])
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        
        # Transform features by relation type
        h_list = []
        for r in range(self.num_relations):
            h_list.append(self.W_r[r](x))
        
        # Stack: [num_relations, num_nodes, out_dim]
        h_stack = torch.stack(h_list, dim=0)
        
        # Get source and destination node indices
        src, dst = edge_index[0], edge_index[1]
        
        # Get transformed features for each edge based on relation type
        h_src = h_stack[edge_type, src]  # [num_edges, out_dim]
        h_dst = h_stack[edge_type, dst]  # [num_edges, out_dim]
        
        # Reshape for multi-head attention
        h_src = h_src.view(-1, self.num_heads, self.head_dim)
        h_dst = h_dst.view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_src = (h_src * self.a_src).sum(dim=-1)  # [num_edges, num_heads]
        attn_dst = (h_dst * self.a_dst).sum(dim=-1)  # [num_edges, num_heads]
        attn = self.leaky_relu(attn_src + attn_dst)
        
        # Softmax over neighbors
        attn = F.softmax(attn, dim=0)
        attn = self.dropout(attn)
        
        # Aggregate messages
        h_src_flat = h_src.view(-1, self.out_dim)
        attn_expanded = attn.view(-1, self.num_heads, 1).expand(-1, -1, self.head_dim)
        attn_flat = attn_expanded.reshape(-1, self.out_dim)
        
        msg = h_src_flat * attn_flat
        
        # Scatter add to destination nodes
        out = torch.zeros(num_nodes, self.out_dim, device=x.device)
        out.index_add_(0, dst, msg)
        
        return out


class RGAT(nn.Module):
    """Relational Graph Attention Network for Performance Prediction"""
    
    def __init__(self, node_dim: int = 64, hidden_dim: int = 128, 
                 output_dim: int = 1, num_relations: int = 6, 
                 num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            RGATLayer(hidden_dim, hidden_dim, num_relations, num_heads)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        h = self.node_embedding(x)
        
        for layer, norm in zip(self.layers, self.layer_norms):
            h_new = layer(h, edge_index, edge_type)
            h = norm(h + h_new)  # Residual connection
            h = F.relu(h)
        
        return self.predictor(h)


def build_graph_from_dataset(dataset_path: str, kg_path: str) -> Tuple:
    """Build graph data from dataset and knowledge graph"""
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Load knowledge graph
    with open(kg_path, 'r') as f:
        kg = json.load(f)
    
    # Build node features and edges
    nodes = kg.get('nodes', [])
    edges = kg.get('edges', [])
    
    num_nodes = len(nodes)
    node_dim = 64
    
    # Create node features (random initialization for now, will be learned)
    node_features = torch.randn(num_nodes, node_dim)
    
    # Create node ID mapping
    node_id_map = {node.get('node_id', node.get('id')): i for i, node in enumerate(nodes)}
    
    # Build edge index and edge types
    relation_types = {'contains': 0, 'has_type': 1, 'supports': 2, 
                      'sequential': 3, 'similarity': 4, 'performance': 5}
    
    edge_src = []
    edge_dst = []
    edge_types = []
    
    for edge in edges:
        src_id = edge.get('source_id', edge.get('source', edge.get('from')))
        dst_id = edge.get('target_id', edge.get('target', edge.get('to')))
        rel = edge.get('relation', edge.get('type', 'contains'))
        
        if src_id in node_id_map and dst_id in node_id_map:
            edge_src.append(node_id_map[src_id])
            edge_dst.append(node_id_map[dst_id])
            edge_types.append(relation_types.get(rel, 0))
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    # Create target labels (performance scores - synthetic for training)
    # In real scenario, these would come from actual profiling
    targets = torch.randn(num_nodes, 1)
    
    return node_features, edge_index, edge_type, targets, num_nodes


def train_rgat(dataset_path: str, kg_path: str, output_dir: str,
               epochs: int = 100, lr: float = 0.001):
    """Train RGAT model"""
    
    print("=" * 60)
    print("RGAT Training for Het-Benchmark")
    print("=" * 60)
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build graph
    print("\nBuilding graph from dataset...")
    node_features, edge_index, edge_type, targets, num_nodes = \
        build_graph_from_dataset(dataset_path, kg_path)
    
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Node feature dim: {node_features.size(1)}")
    
    # Move to device
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    targets = targets.to(device)
    
    # Create model
    model = RGAT(
        node_dim=node_features.size(1),
        hidden_dim=128,
        output_dim=1,
        num_relations=6,
        num_layers=3,
        num_heads=4
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print("\nTraining...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(node_features, edge_index, edge_type)
        
        # Loss (MSE for regression)
        loss = F.mse_loss(predictions, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, os.path.join(output_dir, 'rgat_best.pt'))
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'config': {
            'node_dim': node_features.size(1),
            'hidden_dim': 128,
            'output_dim': 1,
            'num_relations': 6,
            'num_layers': 3,
            'num_heads': 4
        }
    }, os.path.join(output_dir, 'rgat_final.pt'))
    
    print(f"\nTraining complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Model saved to: {output_dir}/rgat_final.pt")
    
    # Generate training report
    report = {
        'training_date': datetime.now().isoformat(),
        'epochs': epochs,
        'final_loss': float(loss.item()),
        'best_loss': float(best_loss),
        'num_nodes': num_nodes,
        'num_edges': int(edge_index.size(1)),
        'model_params': sum(p.numel() for p in model.parameters()),
        'device': str(device)
    }
    
    with open(os.path.join(output_dir, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    return model, report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/model_dataset.json')
    parser.add_argument('--kg', default='data/moh_kg.json')
    parser.add_argument('--output', default='models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    train_rgat(
        dataset_path=args.dataset,
        kg_path=args.kg,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr
    )
