import torch
import torch.nn as nn
import math
from typing import Optional

from .attention import MultiHeadAttention
from .embeddings import PositionalEncoding


"""
Position-wise Feed-Forward Network

Architecture:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
Or with GELU:
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

This is applied identically to each position separately and identically.
"""

class PositionWiseFeedForward(nn.Module):
    """
    Two-layer MLP applied to each position independently.
    
    Architecture:
    ------------
    Input: [batch, seq_len, d_model]
    → Linear: [batch, seq_len, d_ff]
    → Activation (GELU/ReLU)
    → Dropout
    → Linear: [batch, seq_len, d_model]
    → Dropout
    
    Key Points:
    ----------
    - Same network applied to each position (no cross-position interaction)
    - Typically d_ff = 4 * d_model (expansion then compression)
    - GELU activation is standard for transformers
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden layer dimension (typically 4x d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # GELU is smoother than ReLU, works better for transformers
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Expand: [batch, seq_len, d_model] → [batch, seq_len, d_ff]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Compress: [batch, seq_len, d_ff] → [batch, seq_len, d_model]
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x




"""
Single Transformer Encoder Block

Architecture:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network
    4. Add & Norm

Mathematical Flow:
    x = LayerNorm(x + MultiHeadAttention(x))
    x = LayerNorm(x + FFN(x))
"""

class TransformerEncoderBlock(nn.Module):
    """
    Single encoder block of the Transformer.
    
    Components:
    ----------
    1. Multi-Head Self-Attention: Learn relationships between positions
    2. Feed-Forward Network: Process each position independently
    3. Residual Connections: Gradient flow and training stability
    4. Layer Normalization: Stabilize activations
    
    Pre-LN vs Post-LN:
    -----------------
    We use Pre-LN (LayerNorm before sub-layer) because:
    - More stable training
    - Better gradient flow
    - Less sensitive to learning rate
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
            
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Pre-LN: Normalize before attention
        # Sub-layer 1: Multi-head attention with residual
        normed_x = self.norm1(x)
        attention_output, _ = self.self_attention(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attention_output)  # Residual connection
        
        # Sub-layer 2: Feed-forward with residual
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)  # Residual connection
        
        return x



"""
Complete Transformer Encoder for Time-Series Classification

Architecture:
    Input Embedding → Positional Encoding → 
    [Transformer Blocks] × N → 
    Global Pooling → Classifier
"""

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for sensor time-series.
    
    Full Architecture:
    -----------------
    1. Input projection: [batch, seq_len, n_features] → [batch, seq_len, d_model]
    2. Positional encoding: Add position information
    3. Transformer encoder: N stacked encoder blocks
    4. Global pooling: Aggregate sequence info
    5. Classification head: MLP → binary prediction
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 5000,
        n_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            n_features: Number of input features (sensor readings)
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            n_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection: Map input features to model dimension
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling: Aggregate sequence information
        # We use both mean and max pooling for richer representation
        self.pool_type = 'both'  # 'mean', 'max', or 'both'
        
        # Classification head
        pooled_dim = d_model * 2 if self.pool_type == 'both' else d_model
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        Critical for stable transformer training.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, n_features]
            mask: [batch_size, seq_len, seq_len]
            
        Returns:
            logits: [batch_size, n_classes]
        """
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Scale by sqrt(d_model) as in original paper
        # This keeps variance stable after positional encoding addition
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        
        # Global pooling over sequence dimension
        if self.pool_type == 'mean':
            pooled = x.mean(dim=1)  # [batch, d_model]
        elif self.pool_type == 'max':
            pooled = x.max(dim=1)[0]  # [batch, d_model]
        else:  # 'both'
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1)[0]
            pooled = torch.cat([mean_pool, max_pool], dim=1)  # [batch, d_model*2]
        
        # Classification
        logits = self.classifier(pooled)  # [batch, n_classes]
        
        return logits