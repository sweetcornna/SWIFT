from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HCAConfig:
    target_attention_heads: int = 4
    threat_attention_heads: int = 4
    embedding_dim: int = 128
    dropout: float = 0.1
