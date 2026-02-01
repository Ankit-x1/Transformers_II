import torch
import torch.nn as nn
from typing import Optional, Tuple


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
        
        from src.models.attention import ScaledDotProductAttention
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
        
        # Step 1: Linear projections
        # [batch, seq_len, d_model] @ [d_model, d_model] → [batch, seq_len, d_model]
        Q = self.W_q(query)  # [batch, seq_len_q, d_model]
        K = self.W_k(key)    # [batch, seq_len_k, d_model]
        V = self.W_v(value)  # [batch, seq_len_v, d_model]
        
        # Step 2: Split into multiple heads
        # [batch, seq_len, d_model] → [batch, num_heads, seq_len, d_k]
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len_q, d_k]
        K = self.split_heads(K)  # [batch, num_heads, seq_len_k, d_k]
        V = self.split_heads(V)  # [batch, num_heads, seq_len_v, d_k]
        
        # Step 3: Adjust mask for multiple heads
        if mask is not None:
            # Add head dimension: [batch, 1, seq_len_q, seq_len_k]
            # This broadcasts across all heads
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        
        # Step 4: Apply attention to each head in parallel
        # All heads processed simultaneously via batched operations
        # [batch, num_heads, seq_len_q, d_k]
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 5: Concatenate heads
        # [batch, num_heads, seq_len_q, d_k] → [batch, seq_len_q, d_model]
        attention_output = self.combine_heads(attention_output)
        
        # Step 6: Final linear projection
        # [batch, seq_len_q, d_model] @ [d_model, d_model] → [batch, seq_len_q, d_model]
        output = self.W_o(attention_output)
        
        # Apply dropout to output
        output = self.dropout(output)
        
        return output, attention_weights