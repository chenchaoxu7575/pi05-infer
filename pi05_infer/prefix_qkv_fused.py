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

Gemma-2B's attention projections are ``q: 2048x2048`` and, with a single KV head of
width 256, ``k: 256x2048`` and ``v: 256x2048``.  Concatenating the three weights
along dim 0 into one ``[2560, 2048]`` matrix is a **mathematical identity** -- every
output column is an independent dot product over the same input row, so stacking
them along N changes no value and no accumulation order.  Gemma has
``attention_bias=False``, so there is no bias to concatenate.

Why it pays on the prefix (M = 968)
-----------------------------------
This is *not* the reason the same trick pays on the action expert.  There
(``pi05_infer/gemma/modeling_gemma.py::build_qkv_fused``) M is 50, k/v emit only
50x256, the Triton template lands on grid=8 and 8 of 110 SMs do all the work: an
occupancy collapse.  The prefix has M = 968 and is not occupancy-starved.

Its problem is that ``N = 256`` is too narrow to amortise anything.  Inductor's
champion for the ``968x256x2048`` shape is ``BLOCK_M=32, BLOCK_N=32`` (248 CTAs),
and a 32x32 output tile needs 2048 K-steps to produce 1024 results: the kernel
spends its time streaming B and re-reading A with almost no reuse, and lands at
~41 TFLOP/s where the same card does ~190 TFLOP/s on the MLP GEMMs.  k+v together
are 0.9 % of the LM's FLOPs but 3.3 % of its kernel time.  Widening N from 256 to
2560 puts k and v inside tiles that are already paying for their A-traffic.

MEASURED in isolation on the real shapes (bf16, ``torch.randn`` weights, SM clock
locked at 2100 MHz, inductor ``max-autotune``, RTX PRO 5000 Blackwell)::

    q(2048) + k(256) + v(256), three GEMMs   103.7 us
    fused qkv (2560), one GEMM                60.0 us   -43.7 us / layer

i.e. an upper bound of 17 x 43.7 = 743 us/predict.  The end-to-end number is in
``claude_mem/pi05_rollout_forward/results/RESULTS_prefix_qkv_geglu.md``.

Bit-exactness, and why the last layer is left alone
---------------------------------------------------
"Mathematically identical" is not the same as "bit-identical": concatenating along
N changes which *kernel* cuBLAS/inductor picks, and a different kernel can split
the K accumulation differently.  It has to be measured per shape.  Measured here,
bf16, M = 968, K = 2048::

    cat[q(2048), k(256), v(256)] -> 2560   bit-identical to the three GEMMs
    cat[k(256), v(256)]          -> 512    39 % of elements move, by 1 bf16 ULP

So layer 17 -- which ``prefix_last_layer.py`` has already reduced to k/v only, and
whose ``forward`` therefore never reaches this module's patched attention forward
-- keeps its two separate GEMMs.  Fusing it was worth 22 us of the 765 us total and
would have cost the bit-exactness claim on the KV cache the denoise loop consumes.

Why a monkeypatch and not an edit
---------------------------------
Same reason as ``prefix_last_layer.py``: the prefix runs on the *installed*
transformers, not on our vendored ``pi05_infer.gemma`` fork, and that boundary is
deliberate -- it makes "a denoise-kernel change silently reached the prefix"
structurally impossible.  So this module edits nothing.  It swaps the *instance*
``forward`` of each ``GemmaAttention`` object and hangs one extra tensor off it.
The module tree, the parameter names and ``state_dict`` are untouched, so RL
weight sync keeps working.

Two structural properties make the swap safe:

* The joint prefix+suffix **training** forward
  (``PaliGemmaWithExpertModel.forward``'s third branch) never calls
  ``GemmaAttention.forward``.  It reaches into ``layer.self_attn.q_proj`` /
  ``k_proj`` / ``v_proj`` directly and runs its own attention.  Replacing
  ``self_attn.forward`` therefore cannot touch the training path at all.
* The patched forward handles **only** the prefill call (``use_cache=True`` with a
  cache object, not training).  Everything else -- decode, no-cache, the vendored
  fork's static-KV denoise branch -- is delegated verbatim to the original
  ``forward``.

Weight sync
-----------
``_pi05_qkv_w`` is a *weight-derived* tensor: an in-place update of
``q_proj.weight`` does not touch it, so it would silently keep serving the old
weights.  ``refresh_fused_prefix_qkv`` re-derives it and is wired into
``OpenPi0Inference.invalidate_weight_derived_caches``.  The refresh uses ``copy_``
and never reallocates, so the tensor's address is stable (a captured CUDA graph
that referenced it stays valid).

Kill switch: ``RLINF_FUSE_PREFIX_QKV=0``.
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
    # `.contiguous()` on v only, and it is not cosmetic. Unfused, v_proj's output is
    # its own [1, M, 256] buffer and the [1, 1, M, 256] transpose of it is contiguous
    # (the kv-head dim is 1). Fused, v is a stride-2560 slice of the qkv buffer, and
    # nothing downstream rewrites it -- q and k are both rematerialised by RoPE, v is
    # not -- so the strided layout survives all the way into `prime_kv_static`'s
    # `copy_`. Measured: 17 extra at::native::elementwise_kernel launches per predict,
    # 3.17 us each (53.9 us/predict, ~315 GB/s), eagerly, outside the compiled graph.
    # Making it contiguous here puts the copy inside the graph at full bandwidth.
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
        _build_stack(attn, ("q_proj", "k_proj", "v_proj"), "_pi05_qkv_w", "_pi05_qkv_split")
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
