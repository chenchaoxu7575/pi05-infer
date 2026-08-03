# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One QKV GEMM per prefix-LM layer instead of three.

Concatenating ``q``/``k``/``v`` along dim 0 into ``[2560, 2048]`` is a
mathematical identity (``attention_bias=False``, so no bias to concatenate). It
pays because ``N = 256`` is too narrow to amortise the A-traffic: k+v are 0.9 %
of the LM's FLOPs but 3.3 % of its kernel time.

WARNING: layer 17 is excluded. A wider N makes inductor pick a different kernel:
``cat[q,k,v] -> 2560`` is bit-identical, but ``cat[k,v] -> 512`` (what the last
layer would use, since ``prefix_last_layer.py`` reduced it to k/v) moves 39 % of
elements by 1 ULP -- and the denoise loop consumes that layer's KV cache.

WARNING: ``_pi05_qkv_w`` is weight-derived and goes stale on an in-place weight sync.
``refresh_fused_prefix_qkv`` re-derives it via ``copy_`` (never reallocating, so
a captured graph stays valid) and is wired into
``invalidate_weight_derived_caches``.

Only the prefill call is patched; everything else delegates to the original
forward. Kill switch: ``RLINF_FUSE_PREFIX_QKV=0``.
"""

from __future__ import annotations

import os
import types

import torch

# Resolved from the *installed* transformers, so the ops below are literally the
# ones the unpatched layer would have run.
from transformers.models.gemma.modeling_gemma import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

ENV_VAR = "RLINF_FUSE_PREFIX_QKV"

_FALSEY = {"0", "false", "no", "off", ""}

__all__ = [
    "fuse_enabled",
    "install_fused_prefix_qkv",
    "refresh_fused_prefix_qkv",
]


def fuse_enabled() -> bool:
    """Kill switch: ``RLINF_FUSE_PREFIX_QKV=0`` restores the three separate GEMMs."""
    return os.environ.get(ENV_VAR, "1").strip().lower() not in _FALSEY


def _fusable(attn) -> bool:
    """Whether this attention module's q/k/v can be concatenated along dim 0.

    Requires three bias-free ``nn.Linear`` projections that read the same input
    width. Gemma sets ``attention_bias=False``, so the bias case is a guard
    against a future config, not a live path.
    """
    projs = [getattr(attn, n, None) for n in ("q_proj", "k_proj", "v_proj")]
    if any(p is None or getattr(p, "weight", None) is None for p in projs):
        return False
    if any(getattr(p, "bias", None) is not None for p in projs):
        return False
    in_features = {p.weight.shape[1] for p in projs}
    dtypes = {p.weight.dtype for p in projs}
    devices = {p.weight.device for p in projs}
    return len(in_features) == 1 and len(dtypes) == 1 and len(devices) == 1


def _build_stack(attn, names: tuple[str, ...], attr: str, split_attr: str) -> None:
    """Attach ``attr`` = ``cat([<names>.weight], dim=0)`` and its split sizes.

    Stored as a plain attribute, not a parameter or a persistent buffer: the
    projections stay the checkpoint's source of truth and ``state_dict`` is
    unchanged.
    """
    if getattr(attn, attr, None) is not None:
        return
    weights = [getattr(attn, n).weight for n in names]
    setattr(attn, attr, torch.cat([w.detach() for w in weights], dim=0).contiguous())
    setattr(attn, split_attr, tuple(w.shape[0] for w in weights))


def _fused_qkv_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    use_cache: bool = False,
    **kwargs,
):
    """``GemmaAttention.forward`` with the three projections merged into one GEMM.

    Byte-for-byte the same ops as the original in the same order; only the three
    ``F.linear`` calls become one ``F.linear`` plus a free ``split`` view. Anything
    that is not the plain prefill-with-cache call is delegated to the original.
    """
    if (
        getattr(self, "_pi05_qkv_w", None) is None
        or not use_cache
        or past_key_value is None
        or self.training
    ):
        return self._pi05_full_attn_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    qkv = torch.nn.functional.linear(hidden_states, self._pi05_qkv_w)
    q_lin, k_lin, v_lin = torch.split(qkv, self._pi05_qkv_split, dim=-1)
    query_states = q_lin.view(hidden_shape).transpose(1, 2)
    key_states = k_lin.view(hidden_shape).transpose(1, 2)
    # `.contiguous()` is load-bearing: v is a strided slice of the fused buffer and,
    # unlike q and k, nothing downstream rematerialises it -- the stride would reach
    # `prime_kv_static`'s copy_ and cost 17 eager kernels (53.9 us/predict).
    value_states = v_lin.view(hidden_shape).transpose(1, 2).contiguous()

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(
        key_states, value_states, self.layer_idx, cache_kwargs
    )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":  # noqa: SLF001
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # noqa: SLF001

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _prefix_layers(model):
    """The prefix LM's decoder layers, or ``[]`` if the model has no prefix LM."""
    try:
        return model.paligemma_with_expert.paligemma.model.language_model.layers
    except AttributeError:
        return []


def install_fused_prefix_qkv(model) -> int:
    """Merge each prefix-LM layer's q/k/v projections into one GEMM.

    Must run **after** the checkpoint is loaded (the fused weight is a copy of the
    projection weights) and **before** ``torch.compile`` (the traced graph has to
    contain the fused ``linear``).

    Layers whose *decoder layer* forward has been replaced by
    ``prefix_last_layer.py`` are skipped: that forward calls ``k_proj``/``v_proj``
    directly and never reaches ``self_attn.forward``, and fusing its k+v is not
    bit-exact anyway (see the module docstring).

    Args:
        model: the ``OpenPi0Inference`` engine.

    Returns:
        How many layers were patched.
    """
    if not fuse_enabled():
        return 0
    layers = _prefix_layers(model)
    if len(layers) == 0:
        return 0

    patched = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not _fusable(attn):
            continue
        if getattr(layer, "_pi05_skip_installed", False):
            # KV-only layer: prefix_last_layer._kv_only_forward owns it and bypasses
            # self_attn.forward. Left unfused -- cat[k, v] is not bit-exact.
            continue
        if getattr(attn, "_pi05_qkv_installed", False):
            continue
        _build_stack(
            attn, ("q_proj", "k_proj", "v_proj"), "_pi05_qkv_w", "_pi05_qkv_split"
        )
        attn._pi05_full_attn_forward = attn.forward
        attn.forward = types.MethodType(_fused_qkv_attention_forward, attn)
        attn._pi05_qkv_installed = True
        patched += 1
    return patched


def refresh_fused_prefix_qkv(model) -> int:
    """Re-derive every fused prefix weight IN PLACE from the current projections.

    **Call this after an RL rollout weight sync.**  The fused tensors are copies of
    ``q_proj``/``k_proj``/``v_proj``; an in-place weight update leaves them stale
    with no symptom other than wrong actions.  ``copy_`` (never reallocation) keeps
    their addresses, so anything that captured them stays valid.

    Args:
        model: the ``OpenPi0Inference`` engine.

    Returns:
        How many fused weights were refreshed.
    """
    n = 0
    for layer in _prefix_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_pi05_qkv_w", None) is not None:
            attn._pi05_qkv_w.copy_(
                torch.cat(
                    [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
                )
            )
            n += 1
    return n
