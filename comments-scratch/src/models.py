"""
Scratch text models (PyTorch).

We use a small TextCNN-style architecture:
Embedding -> Conv1D (multiple kernels) -> GlobalMaxPool -> Dense -> Softmax

This matches the user's requested knobs: vectorizing, layers, filters, (N)Adam, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TextCNNConfig:
    vocab_size: int
    num_classes: int
    max_len: int
    embedding_dim: int = 64
    num_filters: int = 128
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    hidden_dim: int = 0  # 0 => no hidden layer
    dropout: float = 0.2


class TextCNNClassifier(nn.Module):
    def __init__(self, cfg: TextCNNConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(int(cfg.vocab_size), int(cfg.embedding_dim), padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=int(cfg.embedding_dim),
                    out_channels=int(cfg.num_filters),
                    kernel_size=int(k),
                )
                for k in cfg.kernel_sizes
            ]
        )

        conv_out = int(cfg.num_filters) * int(len(cfg.kernel_sizes))
        self.dropout = nn.Dropout(float(cfg.dropout))

        if int(cfg.hidden_dim) > 0:
            self.fc1 = nn.Linear(conv_out, int(cfg.hidden_dim))
            self.fc2 = nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes))
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(conv_out, int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) token ids
        """

        # (batch, seq_len, emb) -> (batch, emb, seq_len)
        e = self.emb(x).transpose(1, 2)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e))
            # Global max over time
            p = torch.max(h, dim=2).values
            pooled.append(p)
        z = torch.cat(pooled, dim=1)
        z = self.dropout(z)

        if self.fc1 is not None:
            z = F.relu(self.fc1(z))
            z = self.dropout(z)
        logits = self.fc2(z)
        return logits


