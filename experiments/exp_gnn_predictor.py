#!/usr/bin/env python3
"""
实验: GNN性能预测器训练与评估

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


class RGATLayer(nn.Module):
    """关系图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, 
                 num_relations: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_heads = num_heads
        
        # 每个关系类型的权重矩阵
        self.W = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_features, out_features))
            for _ in range(num_relations)
        ])
        
        # 注意力参数
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features // num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, in_features]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
        """
        num_nodes = x.size(0)
        
        # 对每种关系类型应用变换
        h_list = []
        for r in range(self.num_relations):
            h_r = torch.matmul(x, self.W[r])
            h_list.append(h_r)
        
        # 聚合邻居信息
        out = torch.zeros(num_nodes, self.out_features, device=x.device)
        
        if edge_index.size(1) > 0:
            src, dst = edge_index
            
            for r in range(self.num_relations):
                mask = edge_type == r
                if mask.sum() > 0:
                    src_r = src[mask]
                    dst_r = dst[mask]
                    
                    # 获取源节点和目标节点的变换特征
                    h_src = h_list[r][src_r]
                    h_dst = h_list[r][dst_r]
                    
                    # 计算注意力分数 - 简化版本
                    # 使用点积注意力
                    attn = torch.sum(h_src * h_dst, dim=-1)
                    attn = self.leaky_relu(attn)
                    attn = F.softmax(attn, dim=0)
                    attn = self.dropout(attn)
                    
                    # 聚合
                    out.index_add_(0, dst_r, h_src * attn.unsqueeze(-1))
        
        return out


class RGATPredictor(nn.Module):
    """RGAT性能预测器"""
    
    def __init__(self, in_features: int = 64, hidden_features: int = 128,
                 out_features: int = 64, num_relations: int = 6,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(in_features, hidden_features)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RGATLayer(
                hidden_features, hidden_features, 
                num_relations, num_heads, dropout
            ))
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features)
        )
        
        # 性能预测头
        self.predictor = nn.Sequential(
            nn.Linear(out_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_embeddings: 节点嵌入
            predictions: 性能预测值
        """
        h = self.input_proj(x)
        
        for layer in self.layers:
            h_new = layer(h, edge_index, edge_type)
            h = h + h_new  # 残差连接
            h = F.relu(h)
        
        embeddings = self.output_proj(h)
        predictions = self.predictor(embeddings)
        
        return embeddings, predictions


class OperatorGraphDataset(Dataset):
    """算子图数据集"""
    
    def __init__(self, num_samples: int = 1000, num_nodes_range: Tuple[int, int] = (10, 50),
                 in_features: int = 64, num_relations: int = 6):
        self.samples = []
        
        for _ in range(num_samples):
            num_nodes = random.randint(*num_nodes_range)
            
            # 生成节点特征
            x = torch.randn(num_nodes, in_features)
            
            # 生成边（稀疏图）
            num_edges = random.randint(num_nodes, num_nodes * 3)
            src = torch.randint(0, num_nodes, (num_edges,))
            dst = torch.randint(0, num_nodes, (num_edges,))
            edge_index = torch.stack([src, dst])
            
            # 生成边类型
            edge_type = torch.randint(0, num_relations, (num_edges,))
            
            # 生成目标值（模拟性能指标）
            # 基于节点特征和图结构生成合理的目标值
            target = torch.sum(x, dim=0).mean() + num_nodes * 0.1 + num_edges * 0.01
            target = target + torch.randn(1) * 0.1  # 添加噪声
            
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
    """批处理函数"""
    # 由于图大小不同，我们逐个处理
    return batch


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: optim.Optimizer, device: str) -> float:
    """训练一个epoch"""
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
            
            # 使用全局池化得到图级预测
            graph_pred = predictions.mean()
            
            loss = F.mse_loss(graph_pred, target.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict:
    """评估模型"""
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
    
    # 计算指标
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # MAPE
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
    
    # R²
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
    """运行GNN预测器实验"""
    print("="*70)
    print("GNN性能预测器训练与评估实验")
    print("="*70)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 实验配置
    configs = [
        {"name": "Small", "hidden": 64, "layers": 2, "heads": 2},
        {"name": "Medium", "hidden": 128, "layers": 3, "heads": 4},
        {"name": "Large", "hidden": 256, "layers": 4, "heads": 8},
    ]
    
    in_features = 64
    num_relations = 6
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = OperatorGraphDataset(num_samples=800, in_features=in_features, num_relations=num_relations)
    val_dataset = OperatorGraphDataset(num_samples=100, in_features=in_features, num_relations=num_relations)
    test_dataset = OperatorGraphDataset(num_samples=100, in_features=in_features, num_relations=num_relations)
    
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
        print(f"  Hidden: {config['hidden']}, Layers: {config['layers']}, Heads: {config['heads']}")
        print("="*70)
        
        # 创建模型
        model = RGATPredictor(
            in_features=in_features,
            hidden_features=config['hidden'],
            out_features=config['hidden'] // 2,
            num_relations=num_relations,
            num_layers=config['layers'],
            num_heads=config['heads']
        ).to(device)
        
        # 计算参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {num_params:,}")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 训练
        best_val_loss = float('inf')
        best_model_state = None
        train_losses = []
        val_losses = []
        
        print("\n开始训练...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics['MSE']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, Val MSE={val_loss:.4f}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成，耗时: {training_time:.1f}秒")
        
        # 加载最佳模型并测试
        model.load_state_dict(best_model_state)
        test_metrics = evaluate(model, test_loader, device)
        
        print(f"\n测试结果:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        results.append({
            "Model": config['name'],
            "Hidden": config['hidden'],
            "Layers": config['layers'],
            "Heads": config['heads'],
            "Parameters": num_params,
            "Training Time (s)": round(training_time, 1),
            "Test MSE": round(test_metrics['MSE'], 4),
            "Test RMSE": round(test_metrics['RMSE'], 4),
            "Test MAE": round(test_metrics['MAE'], 4),
            "Test MAPE (%)": round(test_metrics['MAPE (%)'], 2),
            "Test R²": round(test_metrics['R²'], 4)
        })
    
    # 保存结果
    results_dir = '/results' if os.path.exists('/results') else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
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
    
    # 打印汇总表格
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
