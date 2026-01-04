#!/usr/bin/env python3
"""
实验: GNN性能预测器训练与评估 (优化版)

在A100 GPU上训练RGAT模型，用于跨平台性能预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import random

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class SimpleRGAT(nn.Module):
    """简化的RGAT模型"""
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, num_relations: int = 6):
        super().__init__()
        
        self.input_proj = nn.Linear(in_features, hidden_features)
        
        # 关系特定的变换
        self.relation_weights = nn.ModuleList([
            nn.Linear(hidden_features, hidden_features) 
            for _ in range(num_relations)
        ])
        
        self.attention = nn.Linear(hidden_features * 2, 1)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, out_features)
        )
        
        self.predictor = nn.Linear(out_features, 1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = F.relu(h)
        
        # 简化的消息传递
        if edge_index.size(1) > 0:
            src, dst = edge_index
            
            # 聚合消息
            messages = torch.zeros_like(h)
            counts = torch.zeros(h.size(0), 1, device=h.device)
            
            for r in range(len(self.relation_weights)):
                mask = edge_type == r
                if mask.sum() > 0:
                    src_r = src[mask]
                    dst_r = dst[mask]
                    
                    h_src = self.relation_weights[r](h[src_r])
                    messages.index_add_(0, dst_r, h_src)
                    counts.index_add_(0, dst_r, torch.ones(len(dst_r), 1, device=h.device))
            
            # 归一化
            counts = counts.clamp(min=1)
            messages = messages / counts
            
            # 残差连接
            h = h + messages
        
        embeddings = self.output_proj(h)
        predictions = self.predictor(embeddings)
        
        return embeddings, predictions


class OperatorGraphDataset(Dataset):
    """算子图数据集"""
    
    def __init__(self, num_samples: int = 500, num_nodes_range: Tuple[int, int] = (10, 30),
                 in_features: int = 32, num_relations: int = 6):
        self.samples = []
        
        for _ in range(num_samples):
            num_nodes = random.randint(*num_nodes_range)
            
            # 生成节点特征
            x = torch.randn(num_nodes, in_features)
            
            # 生成边（稀疏图）
            num_edges = random.randint(num_nodes, num_nodes * 2)
            src = torch.randint(0, num_nodes, (num_edges,))
            dst = torch.randint(0, num_nodes, (num_edges,))
            edge_index = torch.stack([src, dst])
            
            # 生成边类型
            edge_type = torch.randint(0, num_relations, (num_edges,))
            
            # 生成目标值（模拟性能指标）
            target = torch.sum(x, dim=0).mean() + num_nodes * 0.1 + num_edges * 0.01
            target = target + torch.randn(1) * 0.1
            
            self.samples.append({
                'x': x,
                'edge_index': edge_index,
                'edge_type': edge_type,
                'target': target.unsqueeze(0),
                'num_nodes': num_nodes
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return batch


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        for sample in batch:
            x = sample['x'].to(device)
            edge_index = sample['edge_index'].to(device)
            edge_type = sample['edge_type'].to(device)
            target = sample['target'].to(device)
            
            optimizer.zero_grad()
            
            _, predictions = model(x, edge_index, edge_type)
            graph_pred = predictions.mean()
            
            loss = F.mse_loss(graph_pred, target.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict:
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            for sample in batch:
                x = sample['x'].to(device)
                edge_index = sample['edge_index'].to(device)
                edge_type = sample['edge_type'].to(device)
                target = sample['target'].to(device)
                
                _, preds = model(x, edge_index, edge_type)
                graph_pred = preds.mean()
                
                predictions.append(graph_pred.cpu().item())
                targets.append(target.squeeze().cpu().item())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((predictions - targets) / (np.abs(targets) + 1e-8))) * 100
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2
    }


def run_experiment(device: str = 'cuda'):
    print("="*70)
    print("GNN性能预测器训练与评估实验 (优化版)")
    print("="*70)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    configs = [
        {"name": "Small", "hidden": 64, "out": 32},
        {"name": "Medium", "hidden": 128, "out": 64},
        {"name": "Large", "hidden": 256, "out": 128},
    ]
    
    in_features = 32
    num_relations = 6
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.001
    
    print("\n创建数据集...")
    train_dataset = OperatorGraphDataset(num_samples=400, in_features=in_features, num_relations=num_relations)
    val_dataset = OperatorGraphDataset(num_samples=50, in_features=in_features, num_relations=num_relations)
    test_dataset = OperatorGraphDataset(num_samples=50, in_features=in_features, num_relations=num_relations)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"模型配置: {config['name']}")
        print("="*70)
        
        model = SimpleRGAT(
            in_features=in_features,
            hidden_features=config['hidden'],
            out_features=config['out'],
            num_relations=num_relations
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {num_params:,}")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print("\n开始训练...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics['MSE']
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val MSE={val_loss:.4f}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成，耗时: {training_time:.1f}秒")
        
        # 加载最佳模型
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        test_metrics = evaluate(model, test_loader, device)
        
        print(f"\n测试结果:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        results.append({
            "Model": config['name'],
            "Hidden": config['hidden'],
            "Output": config['out'],
            "Parameters": num_params,
            "Training Time (s)": round(training_time, 1),
            "Test MSE": round(test_metrics['MSE'], 4),
            "Test RMSE": round(test_metrics['RMSE'], 4),
            "Test MAE": round(test_metrics['MAE'], 4),
            "Test MAPE (%)": round(test_metrics['MAPE (%)'], 2),
            "Test R²": round(test_metrics['R²'], 4)
        })
    
    # 保存结果
    results_dir = '/results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'gnn_predictor_results.json'), 'w') as f:
        json.dump({
            "experiment": "GNN Performance Predictor Training",
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else "N/A",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset)
            },
            "results": results
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("Table 8: GNN Predictor Performance")
    print("="*70)
    print(f"{'Model':<10} {'Params':<12} {'Time (s)':<12} {'MAPE (%)':<12} {'R²':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['Model']:<10} {r['Parameters']:<12,} {r['Training Time (s)']:<12} "
              f"{r['Test MAPE (%)']:<12} {r['Test R²']:<10}")
    
    print(f"\n✅ 结果已保存到 {results_dir}/gnn_predictor_results.json")
    
    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_experiment(device)
