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
    """Encoder plus projection head, trained with SimCLR or SupCon.

    Attribute names match the old ``SupConSleepCNN`` (``encoder``, ``projection_head``),
    so old checkpoints load. After training only ``encoder`` is kept; the projection
    head exists for the contrastive loss.

    Args:
        encoder: Window encoder returning [B, input_dim].
        input_dim: Encoder output size.
        embedding_dim: Size of the projected, L2-normalised output.
    """
    def __init__(self, encoder: nn.Module, input_dim: int = 512, embedding_dim: int = 128):
        """Build the projection head; see the class docstring."""
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim), nn.ReLU(), nn.Linear(input_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projections [B, embedding_dim] of windows ``x``."""
        return F.normalize(self.projection_head(self.encoder(x)), p=2, dim=1)


SupConSleepCNN = ContrastiveModel  # old name


class ResBlock(nn.Module):
    """Residual MLP block: two Linear-LayerNorm layers with GELU and dropout, added to the input."""
    def __init__(self, dim: int, dropout: float = 0.3):
        """Build a block of width ``dim`` with the given dropout."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x):
        """Return ``GELU(x + block(x))``."""
        return self.act(x + self.block(x))


class ContextResMLP(nn.Module):
    """Residual MLP that classifies window t from the embeddings of windows t-k .. t+k.

    Args:
        input_dim: Length of the flattened context, ``(2k + 1) * embedding_dim``.
        hidden_dim: Width of the residual blocks.
        num_classes: Number of output classes.
        num_blocks: Number of ``ResBlock`` layers.
        dropout: Dropout inside the blocks (halved in the output head).
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 1024, num_classes: int = 3, num_blocks: int = 2,
                 dropout: float = 0.2):
        """Build the input projection, residual blocks and output head."""
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                        nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        """Return class logits [B, num_classes] for flattened context vectors ``x``."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TemporalConvClassifier(nn.Module):
    """Two 1-D convolutions over an embedding sequence: [B, L, D] -> [B, L, num_classes].

    Args:
        emb_dim: Embedding size D.
        num_classes: Number of output classes.
        hidden_channels: Channels of the hidden convolutions.
        kernel_size: Temporal kernel size (odd keeps the length).
        dropout: Dropout after each hidden convolution.
    """

    def __init__(self, emb_dim: int = 512, num_classes: int = 3, hidden_channels: int = 256, kernel_size: int = 3,
                 dropout: float = 0.3):
        """Build the convolution stack; see the class docstring."""
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
        """Return per-window logits [B, L, num_classes] for sequences ``x`` [B, L, D]."""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class GRUSequenceHead(nn.Module):
    """Bidirectional GRU over an embedding sequence: [B, L, D] -> [B, L, num_classes].

    Port of the old ``ContextAwareSleepScorer`` without the encoder inside it. That
    encoder was frozen and in eval mode, so computing embeddings once in the ``embed``
    stage gives the same inputs at a fraction of the cost.

    Args:
        embedding_dim: Embedding size D.
        hidden_dim: GRU hidden size per direction.
        num_classes: Number of output classes.
        num_layers: Stacked GRU layers (dropout 0.5 between them when > 1).
        l2_normalize_inputs: L2-normalise embeddings first, as the old model did.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 64, num_classes: int = 3, num_layers: int = 2,
                 l2_normalize_inputs: bool = True):
        """Build the GRU and classifier; see the class docstring."""
        super().__init__()
        self.l2_normalize_inputs = l2_normalize_inputs
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=0.5 if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, num_classes))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Return per-window logits [B, L, num_classes] for embeddings ``emb`` [B, L, D]."""
        if self.l2_normalize_inputs:  # as in ContextAwareSleepScorer
            emb = F.normalize(emb, p=2, dim=-1)
        out, _ = self.rnn(emb)
        return self.classifier(out)


def load_legacy_gru_state(state_dict: dict) -> tuple[dict, dict]:
    """Split an old ``ContextAwareSleepScorer`` state dict into its two parts.

    Returns:
        ``(encoder_state, head_state)``: the ``encoder.*`` keys with the prefix removed,
        and every other key, ready for ``SCDSSleepCNN`` and ``GRUSequenceHead``.
    """
    enc = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    head = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
    return enc, head


class LinearProbe(nn.Module):
    """One linear layer on frozen embeddings: a quick check of how separable the classes are."""
    def __init__(self, input_dim: int, num_classes: int):
        """Build a ``Linear(input_dim, num_classes)`` layer."""
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """Return class logits for embeddings ``x``."""
        return self.fc(x)
