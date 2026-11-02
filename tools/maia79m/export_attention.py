"""Fixed-shape export lowering for Maia3 self-attention."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


PARITY_ATOL = 5e-5
PARITY_RTOL = 1e-5


class FixedShapeSelfAttention(nn.Module):
    def __init__(self, source: nn.MultiheadAttention, heads: int):
        super().__init__()
        if source.dropout != 0:
            raise ValueError("Export lowering requires upstream eval-time dropout=0")
        if (
            not source.batch_first
            or not source._qkv_same_embed_dim
            or source.bias_k is not None
            or source.bias_v is not None
            or source.add_zero_attn
            or source.embed_dim != 1024
            or heads != 32
        ):
            raise ValueError("Unexpected Maia3-79M self-attention configuration")
        if source.in_proj_weight is None or source.in_proj_weight.shape[0] != 3 * source.embed_dim:
            raise ValueError("Expected upstream packed self-attention Q/K/V projection")

        self.width = source.embed_dim
        self.heads = heads
        self.head_width = self.width // heads
        self.in_proj_weight = source.in_proj_weight
        self.in_proj_bias = source.in_proj_bias
        self.out_proj = source.out_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if (
            need_weights
            or attn_mask is None
            or (key is not None and key is not query)
            or (value is not None and value is not query)
            or kwargs
        ):
            raise ValueError("Maia export expects masked self-attention without returned weights")

        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(1, 64, self.heads, self.head_width).transpose(1, 2)
        k = k.reshape(1, 64, self.heads, self.head_width).transpose(1, 2)
        v = v.reshape(1, 64, self.heads, self.head_width).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_width ** -0.5)
        mask = attn_mask.reshape(1, self.heads, 64, 64)
        weights = torch.softmax(scores + mask, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(1, 64, self.width)
        return self.out_proj(attended), None


def make_export_model(upstream_model: nn.Module) -> nn.Module:
    exported = copy.deepcopy(upstream_model).eval()
    heads = exported.transformer.heads
    for block in exported.transformer.layers:
        original_attention = block.self_attn.mha
        block.self_attn.mha = FixedShapeSelfAttention(original_attention, heads)
    return exported


def compare_outputs(torch_module, upstream_model, export_model, cases) -> list[dict]:
    results = []
    for name, _fen, tokens, self_elo, oppo_elo in cases:
        with torch_module.no_grad():
            upstream = upstream_model(tokens, self_elo, oppo_elo)
            lowered = export_model(tokens, self_elo, oppo_elo)

        for output_name, expected, actual in (
            ("move_logits", upstream[0], lowered[0]),
            ("value_logits", upstream[1], lowered[1]),
        ):
            difference = (expected - actual).abs()
            max_abs = float(difference.max())
            max_rel = float(
                (difference / expected.abs().clamp_min(1e-6)).max()
            )
            passed = torch_module.allclose(
                expected,
                actual,
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
            )
            result = {
                "position": name,
                "output": output_name,
                "shape": list(actual.shape),
                "max_abs_error": max_abs,
                "max_rel_error_floor_1e-6": max_rel,
                "atol": PARITY_ATOL,
                "rtol": PARITY_RTOL,
                "passed": bool(passed),
            }
            results.append(result)
            if not passed:
                raise AssertionError(f"Export attention parity failed: {result}")
    return results