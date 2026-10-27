import torch
import torch.nn as nn
import torch.nn.functional as F

# this script is for non-RF simple classifier architecture, priority is zero-shot and OOD performance

class ResBlock(nn.Module):
    """residual block with normalization and dropout"""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),  #non linear activation func, should be better for embeddings
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))  #residual connection, adds input to output


class ContextResMLP(nn.Module):
    """
    MLP classifier to replace context-random forest
    takes stacked embeddings [t-k, ..., t, ..., t+k] and outputs class logits
    """
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 1024, num_classes: int = 3, num_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   #compresses embeddings immediatelly
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        #makes sequence of resblocks 
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        #classifier head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), #final compression
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes) #3 logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:   #pass input through each block
            x = block(x)
        return self.head(x) #pass through classifier head


class TemporalConvClassifier(nn.Module):
    """
    1D CNN over sequential embeddings [Batch, Seq_Len, Embedding_Dim].
    """
    def __init__(self, emb_dim: int = 512, num_classes: int = 3, hidden_channels: int = 256, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(emb_dim, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Batch, Seq_Len, Emb_Dim] -> [Batch, Emb_Dim, Seq_Len]
        x = x.transpose(1, 2)
        out = self.net(x) # Output: [Batch, Num_Classes, Seq_Len]
        return out.transpose(1, 2) # Return [Batch, Seq_Len, Num_Classes]