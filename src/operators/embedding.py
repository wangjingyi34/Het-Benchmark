"""
Embedding Operator Implementation

Provides word/token embedding lookup functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base import BaseOperator, OperatorConfig


class EmbeddingOperator(BaseOperator):
    """
    Embedding Operator
    
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    Used as the first layer in NLP models.
    
    Parameters:
        num_embeddings: Size of the dictionary (vocabulary size)
        embedding_dim: Size of each embedding vector
    """
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.padding_idx = config.extra.get("padding_idx", None)
        self.max_norm = config.extra.get("max_norm", None)
        self.norm_type = config.extra.get("norm_type", 2.0)
        self.scale_grad_by_freq = config.extra.get("scale_grad_by_freq", False)
        self.sparse = config.extra.get("sparse", False)
        
        # Initialize embedding weights
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim,
                       device=config.device, dtype=config.dtype)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (*, ) containing indices
        
        Returns:
            Output tensor of shape (*, embedding_dim)
        """
        return F.embedding(
            x, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        FLOPs for embedding lookup.
        
        Embedding is essentially a memory lookup, so FLOPs are minimal.
        We count the memory accesses as the primary cost.
        """
        num_lookups = 1
        for dim in input_shape:
            num_lookups *= dim
        
        # Each lookup reads embedding_dim values
        return num_lookups * self.embedding_dim
    
    def get_memory(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate memory usage in bytes."""
        num_lookups = 1
        for dim in input_shape:
            num_lookups *= dim
        
        dtype_size = 4 if self.weight.dtype == torch.float32 else 2
        
        # Input indices (int64)
        input_mem = num_lookups * 8
        
        # Output embeddings
        output_mem = num_lookups * self.embedding_dim * dtype_size
        
        # Embedding table
        table_mem = self.num_embeddings * self.embedding_dim * dtype_size
        
        return input_mem + output_mem + table_mem
    
    def get_parameters(self) -> int:
        """Get total number of parameters."""
        return self.num_embeddings * self.embedding_dim
