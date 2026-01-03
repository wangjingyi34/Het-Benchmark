"""
RGAT: Relational Graph Attention Network for Het-Benchmark
Predicts cross-platform operator performance using graph neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import softmax
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from loguru import logger


class RelationalGraphAttention(MessagePassing):
    """
    Relational Graph Attention Layer
    
    Extends GAT with relation-specific attention for heterogeneous graphs
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat = concat
        
        # Relation-specific transformations
        self.W = nn.Parameter(torch.Tensor(num_relations, in_channels, heads * out_channels))
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_rel = nn.Parameter(torch.Tensor(num_relations, heads, 1))
        
        # Bias
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_rel)
        nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge/relation types [num_edges]
            
        Returns:
            Updated node features [num_nodes, heads * out_channels] or [num_nodes, out_channels]
        """
        # Apply dropout to input
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Transform nodes based on relation types
        # We'll use a simplified approach: average transformation across relations
        # For efficiency, we compute all transformations and select based on edge type
        
        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            edge_type=edge_type,
            size=None,
        )
        
        # Add bias
        out = out + self.bias
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_type: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """Compute messages with relation-aware attention"""
        
        # Transform source and target nodes
        # Shape: [num_edges, heads * out_channels]
        x_j_transformed = torch.zeros(x_j.size(0), self.heads * self.out_channels, device=x_j.device)
        x_i_transformed = torch.zeros(x_i.size(0), self.heads * self.out_channels, device=x_i.device)
        
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if mask.any():
                x_j_transformed[mask] = torch.matmul(x_j[mask], self.W[rel])
                x_i_transformed[mask] = torch.matmul(x_i[mask], self.W[rel])
        
        # Reshape for multi-head attention
        x_j_transformed = x_j_transformed.view(-1, self.heads, self.out_channels)
        x_i_transformed = x_i_transformed.view(-1, self.heads, self.out_channels)
        
        # Compute attention scores
        alpha_src = (x_j_transformed * self.att_src).sum(dim=-1)
        alpha_dst = (x_i_transformed * self.att_dst).sum(dim=-1)
        
        # Add relation-specific attention
        alpha_rel = self.att_rel[edge_type].squeeze(-1)
        
        alpha = alpha_src + alpha_dst + alpha_rel
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to messages
        out = x_j_transformed * alpha.unsqueeze(-1)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        return out


class RGAT(nn.Module):
    """
    Relational Graph Attention Network
    
    Multi-layer RGAT for operator performance prediction
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            RelationalGraphAttention(
                in_channels=in_channels,
                out_channels=hidden_channels,
                num_relations=num_relations,
                heads=heads,
                dropout=dropout,
                concat=True,
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                RelationalGraphAttention(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    num_relations=num_relations,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
        
        # Output layer
        self.layers.append(
            RelationalGraphAttention(
                in_channels=hidden_channels * heads,
                out_channels=out_channels,
                num_relations=num_relations,
                heads=1,
                dropout=dropout,
                concat=False,
            )
        )
        
        # Layer normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels * heads)
            for _ in range(num_layers - 1)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through RGAT layers
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_type)
            x = self.norms[i](x)
            x = F.elu(x)
        
        x = self.layers[-1](x, edge_index, edge_type)
        
        return x


class PerformancePredictor(nn.Module):
    """
    Performance prediction head using RGAT embeddings
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_metrics: int = 5,  # execution_time, memory, throughput, latency_p50, latency_p99
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # Concatenate operator and hardware embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_metrics),
        )
    
    def forward(
        self,
        operator_embedding: torch.Tensor,
        hardware_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict performance metrics
        
        Args:
            operator_embedding: Operator node embedding [batch_size, embedding_dim]
            hardware_embedding: Hardware node embedding [batch_size, embedding_dim]
            
        Returns:
            Predicted metrics [batch_size, num_metrics]
        """
        combined = torch.cat([operator_embedding, hardware_embedding], dim=-1)
        return self.mlp(combined)


class RGATPerformanceModel(nn.Module):
    """
    Complete RGAT-based performance prediction model
    
    Combines RGAT for graph encoding with performance prediction head
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_relations: int = 10,
        num_layers: int = 3,
        num_heads: int = 4,
        num_metrics: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.rgat = RGAT(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_relations=num_relations,
            num_layers=num_layers,
            heads=num_heads,
            dropout=dropout,
        )
        
        self.predictor = PerformancePredictor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_metrics=num_metrics,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        operator_indices: torch.Tensor,
        hardware_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for performance prediction
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
            operator_indices: Indices of operator nodes to predict [batch_size]
            hardware_indices: Indices of hardware nodes [batch_size]
            
        Returns:
            Predicted performance metrics [batch_size, num_metrics]
        """
        # Get node embeddings
        embeddings = self.rgat(x, edge_index, edge_type)
        
        # Extract operator and hardware embeddings
        operator_emb = embeddings[operator_indices]
        hardware_emb = embeddings[hardware_indices]
        
        # Predict performance
        predictions = self.predictor(operator_emb, hardware_emb)
        
        return predictions
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings without prediction"""
        return self.rgat(x, edge_index, edge_type)


@dataclass
class RGATConfig:
    """Configuration for RGAT model"""
    node_feature_dim: int = 64
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_relations: int = 10
    num_layers: int = 3
    num_heads: int = 4
    num_metrics: int = 5
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 32


class RGATTrainer:
    """
    Trainer for RGAT performance prediction model
    """
    
    def __init__(
        self,
        model: RGATPerformanceModel,
        config: RGATConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.criterion = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": []}
    
    def train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        operator_indices: torch.Tensor,
        hardware_indices: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        operator_indices = operator_indices.to(self.device)
        hardware_indices = hardware_indices.to(self.device)
        targets = targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        predictions = self.model(
            x, edge_index, edge_type,
            operator_indices, hardware_indices,
        )
        
        loss = self.criterion(predictions, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        operator_indices: torch.Tensor,
        hardware_indices: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """Evaluate model"""
        self.model.eval()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        operator_indices = operator_indices.to(self.device)
        hardware_indices = hardware_indices.to(self.device)
        targets = targets.to(self.device)
        
        predictions = self.model(
            x, edge_index, edge_type,
            operator_indices, hardware_indices,
        )
        
        loss = self.criterion(predictions, targets)
        
        return loss.item(), predictions.cpu()
    
    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(
                train_data["x"],
                train_data["edge_index"],
                train_data["edge_type"],
                train_data["operator_indices"],
                train_data["hardware_indices"],
                train_data["targets"],
            )
            self.history["train_loss"].append(train_loss)
            
            if val_data is not None:
                val_loss, _ = self.evaluate(
                    val_data["x"],
                    val_data["edge_index"],
                    val_data["edge_type"],
                    val_data["operator_indices"],
                    val_data["hardware_indices"],
                    val_data["targets"],
                )
                self.history["val_loss"].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.config.epochs}, Train Loss: {train_loss:.4f}"
                if val_data is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                logger.info(msg)
        
        return self.history
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
        }, path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        logger.info(f"Loaded model from {path}")


def create_graph_from_kg(kg_data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Convert MOH-KG data to PyTorch Geometric format
    
    Returns:
        x: Node features
        edge_index: Edge indices
        edge_type: Edge types
        node_mapping: Mapping from entity ID to node index
    """
    entities = kg_data["entities"]
    relations = kg_data["relations"]
    
    # Create node mapping
    node_mapping = {e["id"]: i for i, e in enumerate(entities)}
    
    # Create node features (placeholder - should be replaced with actual features)
    num_nodes = len(entities)
    feature_dim = 64
    x = torch.randn(num_nodes, feature_dim)
    
    # Create edge index and types
    edge_list = []
    edge_types = []
    
    relation_type_mapping = {
        "contains": 0,
        "depends_on": 1,
        "runs_on": 2,
        "supported_by": 3,
        "optimized_for": 4,
        "has_performance": 5,
        "compared_to": 6,
        "similar_to": 7,
        "equivalent_to": 8,
    }
    
    for rel in relations:
        src_idx = node_mapping.get(rel["source_id"])
        tgt_idx = node_mapping.get(rel["target_id"])
        
        if src_idx is not None and tgt_idx is not None:
            edge_list.append([src_idx, tgt_idx])
            rel_type = relation_type_mapping.get(rel["type"], 9)
            edge_types.append(rel_type)
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    
    return x, edge_index, edge_type, node_mapping


if __name__ == "__main__":
    # Test RGAT model
    config = RGATConfig(
        node_feature_dim=64,
        hidden_dim=128,
        embedding_dim=64,
        num_relations=10,
        num_layers=3,
        num_heads=4,
    )
    
    model = RGATPerformanceModel(
        node_feature_dim=config.node_feature_dim,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_relations=config.num_relations,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    
    # Create dummy data
    num_nodes = 100
    num_edges = 500
    batch_size = 16
    
    x = torch.randn(num_nodes, config.node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, config.num_relations, (num_edges,))
    operator_indices = torch.randint(0, num_nodes, (batch_size,))
    hardware_indices = torch.randint(0, num_nodes, (batch_size,))
    
    # Forward pass
    predictions = model(x, edge_index, edge_type, operator_indices, hardware_indices)
    print(f"Predictions shape: {predictions.shape}")
    
    # Test training
    targets = torch.randn(batch_size, config.num_metrics)
    
    train_data = {
        "x": x,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "operator_indices": operator_indices,
        "hardware_indices": hardware_indices,
        "targets": targets,
    }
    
    trainer = RGATTrainer(model, config, device="cpu")
    
    # Train for a few epochs
    config.epochs = 5
    history = trainer.train(train_data)
    print(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")
