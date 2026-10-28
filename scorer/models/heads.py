"""
Everything that sits on top of a window encoder.

  ContrastiveModel          encoder + projection head (SimCLR / SupCon). Same attribute names
                            as the old SupConSleepCNN (encoder, projection_head) -> old weights load.
  ContextResMLP / ResBlock  classifier over stacked embeddings [t-k .. t+k] (unchanged, key-compatible)
  TemporalConvClassifier    1D conv over embedding sequences (unchanged)
  GRUSequenceHead           bi-GRU over embedding sequences. Port of ContextAwareSleepScorer
                            without the frozen encoder inside: the encoder was frozen and in eval
                            mode anyway, so running it once in the `embed` stage is equivalent and
                            far cheaper. load_legacy_gru_state() splits old combined checkpoints.
  LinearProbe               single linear layer on frozen embeddings (port of linear_train_eval).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int = 512, embedding_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim), nn.ReLU(), nn.Linear(input_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection_head(self.encoder(x)), p=2, dim=1)


SupConSleepCNN = ContrastiveModel  # old name


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ContextResMLP(nn.Module):
    """input: flattened [t-k, ..., t, ..., t+k] embeddings -> class logits for window t"""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 1024, num_classes: int = 3, num_blocks: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                        nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TemporalConvClassifier(nn.Module):
    """[B, L, D] embedding sequence -> [B, L, num_classes]"""

    def __init__(self, emb_dim: int = 512, num_classes: int = 3, hidden_channels: int = 256, kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(emb_dim, hidden_channels, kernel_size, padding=pad), nn.BatchNorm1d(hidden_channels), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad), nn.BatchNorm1d(hidden_channels),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class GRUSequenceHead(nn.Module):
    """[B, L, D] embeddings -> [B, L, num_classes] logits (bi-directional GRU)"""

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 64, num_classes: int = 3, num_layers: int = 2,
                 l2_normalize_inputs: bool = True):
        super().__init__()
        self.l2_normalize_inputs = l2_normalize_inputs
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=0.5 if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, num_classes))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if self.l2_normalize_inputs:  # as in ContextAwareSleepScorer
            emb = F.normalize(emb, p=2, dim=-1)
        out, _ = self.rnn(emb)
        return self.classifier(out)


def load_legacy_gru_state(state_dict: dict) -> tuple[dict, dict]:
    """split an old ContextAwareSleepScorer state dict into (encoder_state, head_state)"""
    enc = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    head = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
    return enc, head


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
