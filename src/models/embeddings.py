import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of model embeddings
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        

        pe = torch.zeros(max_len, d_model)
        

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)
    
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Token embeddings [batch_size, seq_len, d_model]
            
        Returns:
            x + positional encoding, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
