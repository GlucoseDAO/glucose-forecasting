#!/usr/bin/env python3
"""
SugarJepa architecture module — SugarOne + a frozen/fine-tunable CGM-JEPA
embedding as a 4th cross-attention auxiliary stream.

Covariates: Basal Rate (U/h), Bolus Insulin (U), Carbohydrates (g), plus a
288-step (24h) glucose-only lookback encoded by a pretrained CGM-JEPA encoder
(see src/sugar_jepa/vendor/cgm_jepa/, vendored from
https://github.com/cruiseresearchgroup/CGM-JEPA, MIT license).

Architecture:
  Identical to SugarOne (src/sugar_one/sugar_one_model.py) — parallel
  cross-attention + multi-scale self-attention — except the cross-attention
  block now mixes FOUR auxiliaries (basal, bolus, carbs, jepa) instead of
  three, still via a learnable softmax mixing weight. The JEPA auxiliary is
  a sequence of 24 patch embeddings (dim 96) from a longer, glucose-only 24h
  lookback window, projected into d_model — independent of and longer than
  the model's own `input_steps` window, which is unchanged from SugarOne.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.network import apply_windows_tls_workarounds
from sugar_jepa.vendor.cgm_jepa.encoder import Encoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (identical to base GluMind/SugarOne)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class JepaEncoderWrapper(nn.Module):
    """
    Wraps the vendored, pretrained CGM-JEPA `Encoder` to turn a raw
    (batch, jepa_window) glucose-only lookback into a (seq, batch, d_model)
    cross-attention K/V stream.

    Frozen by default (encoder runs under torch.no_grad()); pass
    freeze=False to fine-tune the encoder's own weights alongside the rest
    of the model (train_sugar_jepa.py then puts its params in a separate,
    smaller-LR optimizer group).
    """

    def __init__(
        self,
        d_model: int,
        weights_dir: str,
        patch_size: int = 12,
        freeze: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.freeze = freeze
        # Harmless when weights_dir is a local path (the default); makes
        # weights_dir="CRUISEResearchGroup/CGM-JEPA"-style Hub ids work too
        # on Windows machines hitting the TLS issues in src/common/network.py.
        apply_windows_tls_workarounds()
        self.encoder = Encoder.from_pretrained(weights_dir)
        self.embed_dim = self.encoder.embed_dim
        self.proj = nn.Linear(self.embed_dim, d_model)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def forward(self, glucose_window: torch.Tensor) -> torch.Tensor:
        """glucose_window: (batch, jepa_window), z-score normalized.

        Returns (seq=num_patches, batch, d_model) for nn.MultiheadAttention.
        """
        batch, total_steps = glucose_window.shape
        num_patches = total_steps // self.patch_size
        patches = glucose_window[:, : num_patches * self.patch_size].view(
            batch, num_patches, self.patch_size
        )
        if self.freeze:
            self.encoder.eval()
            with torch.no_grad():
                hidden, _ = self.encoder(patches, x_mark=None, mask=None)
        else:
            hidden, _ = self.encoder(patches, x_mark=None, mask=None)
        # hidden: (batch, num_patches, embed_dim) -> (num_patches, batch, d_model)
        return self.proj(hidden).permute(1, 0, 2)


class CrossAttentionSugarJepaBlock(nn.Module):
    """
    Four-stream cross-attention: glucose (Q) attends to basal, bolus, carbs,
    and the CGM-JEPA embedding (each as K/V) via separate attention heads.

    Outputs are merged with a learnable softmax weight (extended from
    SugarOne's 3-way mix to 4-way) so the model can learn how much weight
    to give the foundation-model glucose embedding relative to the
    pharmacological covariates.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_basal = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_bolus = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_carbs = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_jepa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # Learnable 4-way mixing: after softmax these become non-negative and sum to 1.
        self.mix_logits = nn.Parameter(torch.zeros(4))

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )

    def forward(
        self,
        glucose: torch.Tensor,   # (seq, batch, d_model)
        basal: torch.Tensor,
        bolus: torch.Tensor,
        carbs: torch.Tensor,
        jepa: torch.Tensor,      # (jepa_seq, batch, d_model)
    ) -> torch.Tensor:
        out_basal, _ = self.attn_basal(glucose, basal, basal)
        out_bolus, _ = self.attn_bolus(glucose, bolus, bolus)
        out_carbs, _ = self.attn_carbs(glucose, carbs, carbs)
        out_jepa, _ = self.attn_jepa(glucose, jepa, jepa)

        res_basal = self.ln1(glucose + self.dropout(out_basal))
        res_bolus = self.ln1(glucose + self.dropout(out_bolus))
        res_carbs = self.ln1(glucose + self.dropout(out_carbs))
        res_jepa = self.ln1(glucose + self.dropout(out_jepa))

        w = F.softmax(self.mix_logits, dim=0)
        merged = w[0] * res_basal + w[1] * res_bolus + w[2] * res_carbs + w[3] * res_jepa

        ff = self.ffn(merged)
        return self.ln2(merged + self.dropout(ff))


class MultiScaleAttentionBlock(nn.Module):
    """
    Multi-scale self-attention at 3 resolutions: DS=1, DS=2, DS=4.
    Low-resolution outputs are upsampled back and summed (identical to base GluMind/SugarOne).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_high = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_low2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_low4 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq_len, batch, d_model)."""
        seq_len = x.size(0)

        high_out, _ = self.attn_high(x, x, x)
        high = self.ln1(x + self.dropout(high_out))

        xt = high.permute(1, 2, 0)  # (batch, d_model, seq)

        low2 = self.pool2(xt).permute(2, 0, 1)
        low2_out, _ = self.attn_low2(low2, low2, low2)
        up2 = F.interpolate(
            low2_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        low4 = self.pool4(xt).permute(2, 0, 1)
        low4_out, _ = self.attn_low4(low4, low4, low4)
        up4 = F.interpolate(
            low4_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        fused = high + self.dropout(up2) + self.dropout(up4)
        ff = self.ffn(fused)
        return self.ln2(fused + self.dropout(ff))


class SugarJepaParallelBlock(nn.Module):
    """
    One SugarJepa block: cross-attention (4-aux, incl. JEPA) and multi-scale
    run IN PARALLEL, outputs summed — same philosophy as SugarOne/GluMind.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = CrossAttentionSugarJepaBlock(d_model, n_heads, ff_units, dropout)
        self.multiscale = MultiScaleAttentionBlock(d_model, n_heads, ff_units, dropout)
        self.ln_fuse = nn.LayerNorm(d_model)

    def forward(
        self,
        glucose: torch.Tensor,
        basal: torch.Tensor,
        bolus: torch.Tensor,
        carbs: torch.Tensor,
        jepa: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs: (seq_len, batch, d_model), jepa may have a different seq_len. Returns glucose's shape."""
        cross_out = self.cross_attn(glucose, basal, bolus, carbs, jepa)
        ms_out = self.multiscale(glucose)
        return self.ln_fuse(cross_out + ms_out)


class SugarJepaModel(nn.Module):
    """
    SugarJepa: SugarOne's Multimodal Parallel-Attention Transformer, plus a
    pretrained CGM-JEPA glucose embedding as a 4th cross-attention auxiliary.

    Inputs:
      x:             (batch, n_time_steps, 4)  — [glucose, basal_rate, bolus_insulin, carbohydrates]
      glucose_jepa:  (batch, jepa_window)      — raw glucose, z-score normalized, jepa_window steps
                                                  (default 288 = 24h @ 5min), independent of n_time_steps.
    Output: (batch, horizon) — predicted glucose values

    Differences vs SugarOne:
      - CrossAttentionSugarJepaBlock uses 4 auxiliaries (basal/bolus/carbs/jepa) with
        a learnable 4-way mixing weight instead of SugarOne's 3-way mix.
      - JepaEncoderWrapper runs a pretrained, frozen-by-default CGM-JEPA encoder over
        a separate, longer glucose-only lookback window each forward pass.
    """

    def __init__(
        self,
        n_time_steps: int,
        n_features: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        ff_units: int = 128,
        n_blocks: int = 3,
        prediction_horizon: int = 12,
        dropout: float = 0.1,
        jepa_weights_dir: str = "src/sugar_jepa/pretrained/cgm_jepa",
        jepa_patch_size: int = 12,
        jepa_freeze: bool = True,
    ):
        super().__init__()
        self.n_time_steps = n_time_steps
        self.d_model = d_model
        self.n_features = n_features

        self.embed_glucose = nn.Linear(1, d_model)
        self.embed_basal = nn.Linear(1, d_model)
        self.embed_bolus = nn.Linear(1, d_model)
        self.embed_carbs = nn.Linear(1, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=n_time_steps)

        self.jepa_encoder = JepaEncoderWrapper(
            d_model=d_model,
            weights_dir=jepa_weights_dir,
            patch_size=jepa_patch_size,
            freeze=jepa_freeze,
        )

        self.blocks = nn.ModuleList(
            [
                SugarJepaParallelBlock(d_model, n_heads, ff_units, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.flatten_fc = nn.Linear(d_model * n_time_steps, d_model)
        self.out_fc = nn.Linear(d_model, prediction_horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, glucose_jepa: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, 4) — glucose, basal, bolus, carbs. glucose_jepa: (batch, jepa_window)."""
        g = x[..., 0:1]  # (batch, seq, 1)
        b = x[..., 1:2]  # basal rate
        bo = x[..., 2:3]  # bolus insulin
        c = x[..., 3:4]  # carbohydrates

        g_e = self.pos_enc(self.embed_glucose(g))    # (batch, seq, d_model)
        b_e = self.pos_enc(self.embed_basal(b))
        bo_e = self.pos_enc(self.embed_bolus(bo))
        c_e = self.pos_enc(self.embed_carbs(c))

        # (seq, batch, d_model) for nn.MultiheadAttention
        g_e = g_e.permute(1, 0, 2)
        b_e = b_e.permute(1, 0, 2)
        bo_e = bo_e.permute(1, 0, 2)
        c_e = c_e.permute(1, 0, 2)

        jepa_e = self.jepa_encoder(glucose_jepa)  # (jepa_seq, batch, d_model)

        out = g_e
        for block in self.blocks:
            out = block(out, b_e, bo_e, c_e, jepa_e)

        out = out.permute(1, 2, 0)          # (batch, d_model, seq)
        out = out.reshape(out.size(0), -1)  # (batch, d_model * seq)
        out = self.dropout(F.gelu(self.flatten_fc(out)))
        return self.out_fc(out)             # (batch, horizon)
