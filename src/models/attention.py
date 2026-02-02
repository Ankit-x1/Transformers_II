import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes: softmax(QK^T / sqrt(d_k))V
    
    This is the core attention mechanism used in transformers.
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        Args:
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, num_heads, seq_len_q, d_k]
            key:   [batch_size, num_heads, seq_len_k, d_k]
            value: [batch_size, num_heads, seq_len_v, d_v] (seq_len_k == seq_len_v)
            mask:  [batch_size, 1, seq_len_q, seq_len_k] or None
            
        Returns:
            output:            [batch_size, num_heads, seq_len_q, d_v]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T
        # [batch, heads, seq_q, d_k] @ [batch, heads, d_k, seq_k] → [batch, heads, seq_q, seq_k]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by sqrt(d_k) to prevent gradient explosion
        scores = scores / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, d_v] → [batch, heads, seq_q, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism
    - We reshape instead of actually splitting (more efficient)
    - Transpose for efficient batched matrix multiplication
    - Single output projection after concatenation
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Total dimension of model (must be divisible by num_heads)
            num_heads: Number of parallel attention heads
            dropout: Dropout probability for attention weights
            
        Raises:
            AssertionError: If d_model not divisible by num_heads
        """
        super().__init__()
        
        # Validate that we can evenly split dimensions across heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        Reshape for parallel attention computation.
        """
        batch_size, seq_len, d_model = x.size()
        
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads: merge heads back together.
        
        Args:
            x: [batch_size, num_heads, seq_len, d_k]
            
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        
        x = x.transpose(1, 2)
        
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: [batch_size, seq_len_q, d_model]
            key:   [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]  (seq_len_k == seq_len_v)
            mask:  [batch_size, seq_len_q, seq_len_k] or [batch_size, 1, seq_len_q, seq_len_k]
            
        Returns:
            output:            [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        
        batch_size = query.size(0)
        
 
        Q = self.W_q(query) 
        K = self.W_k(key)    
        V = self.W_v(value)  
        
        Q = self.split_heads(Q)  
        K = self.split_heads(K)  
        V = self.split_heads(V)  
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        

        attention_output, attention_weights = self.attention(Q, K, V, mask)
        

        attention_output = self.combine_heads(attention_output)
        

        output = self.W_o(attention_output)

        output = self.dropout(output)
        
        return output, attention_weights