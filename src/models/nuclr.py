from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class NuCLROutput:
    """Raw temporal latents, trained pooled embeddings, and contrastive projections."""

    latents: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor
    unit_mask: torch.Tensor


class NuCLR(nn.Module):
    def __init__(self, time_bins=500, bins_per_latent=10, dim=48, heads=4, dropout=0.25):
        super().__init__()

        if min(time_bins, bins_per_latent, dim, heads) < 1:
            raise ValueError("model dimensions must be positive")
        if time_bins % bins_per_latent:
            raise ValueError("time_bins must be divisible by bins_per_latent")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        
        self.time_bins = int(time_bins)
        self.bins_per_latent = int(bins_per_latent)
        self.n_latents = self.time_bins // self.bins_per_latent
        self.embedding_dim = self.emb_dim = self.dim = int(dim)
        self.activation_checkpointing = False
        self.attention_chunk_size = None
        self.compact_temporal = False

        self.read_in = nn.Linear(self.bins_per_latent, self.dim)
        self.position = nn.Parameter(torch.randn(self.n_latents, self.dim) * 0.02)

        self.temporal_before = self._block(heads, dropout)
        self.population = self._block(heads, dropout)
        self.temporal_after = self._block(heads, dropout)

        self.norm = nn.LayerNorm(self.dim)

        self.projector = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim),
            nn.ReLU(), nn.Linear(self.dim, self.dim),
        )

    def _block(self, heads, dropout):
        return nn.TransformerEncoderLayer(
            self.dim, heads, 4 * self.dim, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )

    def _apply_block(self, block, value, **kwargs):
        chunk = self.attention_chunk_size
        if chunk is not None and len(value) > chunk:
            return torch.cat([
                self._run_block(block, value[start:start + chunk],
                                **{key: mask[start:start + chunk] for key, mask in kwargs.items()})
                for start in range(0, len(value), chunk)
            ], dim=0)
        return self._run_block(block, value, **kwargs)

    def _run_block(self, block, value, **kwargs):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, value, use_reentrant=False, **kwargs)
        return block(value, **kwargs)

    def _apply_temporal(self, block, value, unit_mask):
        """Skip padded neuron sequences; restore their slots for population attention."""
        if not self.compact_temporal:
            return self._apply_block(block, value.flatten(0, 1)).reshape(value.shape)
        # Select by identity mask, not activity: silent real neurons must be retained.
        real = self._apply_block(block, value[unit_mask])
        restored = real.new_zeros(value.shape)
        restored[unit_mask] = real
        return restored

    def forward(self, window, unit_mask=None) -> NuCLROutput:
        """Encode [N,T] or padded [B,N,T] spike counts; True means a real unit."""

        if window.ndim not in (2, 3) or window.shape[-1] != self.time_bins:
            raise ValueError(f"expected [N,{self.time_bins}] or [B,N,{self.time_bins}]")
        if unit_mask is None:
            unit_mask = torch.ones(window.shape[:-1], dtype=torch.bool, device=window.device)
        if unit_mask.shape != window.shape[:-1] or unit_mask.dtype != torch.bool:
            raise ValueError("unit_mask must be boolean and match the window's unit dimensions")
        
        batched = window.ndim == 3

        if not batched:
            window, unit_mask = window.unsqueeze(0), unit_mask.unsqueeze(0)

        batch_size, n_units, _ = window.shape
        if batch_size == 0 or not unit_mask.any(dim=1).all():
            raise ValueError("each sample must contain at least one real unit")

        # Sanitize padding before attention, including arbitrary/NaN padded values.
        window = window.masked_fill(~unit_mask[..., None], 0)
        latents = window.reshape(batch_size, n_units, self.n_latents, self.bins_per_latent)

        latents = self.read_in(torch.log1p(latents.float())) + self.position
        latents = self._apply_temporal(self.temporal_before, latents, unit_mask)
        population = latents.permute(0, 2, 1, 3).flatten(0, 1)

        padding = (~unit_mask)[:, None, :].expand(-1, self.n_latents, -1).flatten(0, 1)

        population = self._apply_block(self.population, population, src_key_padding_mask=padding)

        latents = population.reshape(batch_size, self.n_latents, n_units, self.dim).permute(0, 2, 1, 3)
        latents = self._apply_temporal(self.temporal_after, latents, unit_mask)
        # Preserve the trained pooling path while exposing raw tokens for export.
        normalized = self.norm(latents).masked_fill(~unit_mask[..., None, None], 0)
        embedding = self.norm(normalized.mean(dim=-2)).masked_fill(~unit_mask[..., None], 0)
        latents = latents.masked_fill(~unit_mask[..., None, None], 0)
        projection = F.normalize(self.projector(embedding), dim=-1)
        projection = projection.masked_fill(~unit_mask[..., None], 0)

        if not batched:
            latents, embedding, projection, unit_mask = (
                value[0] for value in (latents, embedding, projection, unit_mask)
            )
        return NuCLROutput(latents, embedding, projection, unit_mask)

    def encode(self, window, unit_mask=None):
        """Return pooled neuron embeddings (before the contrastive projector)."""
        return self(window, unit_mask).embedding

    def forward_packed(self, bins, spax_seqlen):
        """[sum(N),T] input, isolated by session/probe lengths.

        Uses portable padded PyTorch attention internally. Returns [sum(N),D].
        """
        lengths = torch.as_tensor(spax_seqlen)
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("spax_seqlen must be a nonempty vector")
        if (lengths <= 0).any() or (lengths != lengths.long()).any():
            raise ValueError("spax_seqlen must contain positive integers")
        lengths = lengths.long().tolist()
        if bins.ndim != 2 or sum(lengths) != len(bins):
            raise ValueError("spax_seqlen must sum to the number of units in bins")
        window = nn.utils.rnn.pad_sequence(bins.split(lengths), batch_first=True)
        mask = torch.arange(max(lengths), device=bins.device)[None] < torch.tensor(lengths, device=bins.device)[:, None]
        return self.encode(window, mask)[mask]

    @classmethod
    def from_pretrained(cls, checkpoint, device="cpu"):
        from src.common.checkpoints import load_model
        return load_model(checkpoint, device)[0]
