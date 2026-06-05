from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.scorer(x).squeeze(-1)            # (B,T)
        w = torch.softmax(a, dim=1).unsqueeze(-1) # (B,T,1)
        return (x * w).sum(dim=1)                 # (B,D)


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class CosineClassifier(nn.Module):
    """
    Normalized (cosine) classifier:
      logits = s * cos(emb, W)
    """
    def __init__(self, in_dim: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        cos = x @ W.t()
        return cos * self.scale


class LSTMClassifier(nn.Module):
    """
    Output tetap logits (B, num_classes),
    tapi head sekarang cosine classifier (lebih bagus untuk class imbalance).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.40,
        bidirectional: bool = True,
        input_proj_dim: int | None = None,
        embed_dim: int | None = None,
        attention_heads: int = 0,
        attention_layers: int = 0,
        predict_coords: bool = False,
    ):
        super().__init__()

        if input_proj_dim is None:
            # Legacy auto size. Keep this path for old checkpoints that do not
            # store explicit architecture dimensions.
            proj_dim = max(64, min(128, input_size * 8))
        else:
            proj_dim = int(max(input_size, input_proj_dim))

        self.in_proj = nn.Sequential(
            nn.Linear(input_size, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        heads = int(max(0, attention_heads))
        if heads > 0 and out_dim % heads != 0:
            divisors = [h for h in range(min(heads, out_dim), 0, -1) if out_dim % h == 0]
            heads = divisors[0] if divisors else 1

        self.attention_heads = heads
        self.attention_layers = int(max(0, attention_layers if heads > 0 else 0))
        self.self_attn = nn.Sequential(
            *[
                TemporalSelfAttention(out_dim, num_heads=heads, dropout=dropout)
                for _ in range(self.attention_layers)
            ]
        )
        self.attn = AttentionPool(out_dim, dropout=dropout)

        pooled_dim = out_dim * 4
        self.norm = nn.LayerNorm(pooled_dim)
        self.pooled_dropout = nn.Dropout(dropout)

        if embed_dim is None:
            emb_dim = max(192, min(320, pooled_dim // 2))
        else:
            emb_dim = int(max(num_classes, embed_dim))

        self.embed = nn.Sequential(
            nn.Linear(pooled_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = CosineClassifier(emb_dim, num_classes, scale=30.0)
        self.predict_coords = bool(predict_coords)
        if self.predict_coords:
            geo_hidden = max(64, min(256, emb_dim // 2))
            self.geo_head = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, geo_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(geo_hidden, 2),
            )
        else:
            self.geo_head = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        out, _ = self.lstm(x)
        out = self.self_attn(out)
        last = out[:, -1, :]
        mean = out.mean(dim=1)
        mx = out.max(dim=1).values
        att = self.attn(out)
        feat = torch.cat([att, mean, mx, last], dim=1)
        feat = self.norm(feat)
        feat = self.pooled_dropout(feat)
        emb = self.embed(feat)
        return emb

    def predict_latlon(self, emb: torch.Tensor) -> torch.Tensor | None:
        if self.geo_head is None:
            return None
        raw = self.geo_head(emb)
        lat = 90.0 * torch.tanh(raw[:, 0])
        lon = 180.0 * torch.tanh(raw[:, 1])
        return torch.stack([lat, lon], dim=1)

    def forward_with_aux(self, x: torch.Tensor):
        emb = self.forward_features(x)
        logits = self.head(emb)
        pred_latlon = self.predict_latlon(emb)
        return logits, pred_latlon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        return self.head(emb)
