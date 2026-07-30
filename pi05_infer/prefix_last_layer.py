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
"""Drop the dead tail of the prefix LM's *last* decoder layer.

``sample_actions`` runs the PaliGemma prefix once, and throws its output embedding
away -- only the KV cache is consumed, by the denoise loop (``engine.py``,
``_build_prefix_cache``; the caller binds the hidden state to ``_prefix_output``
and never reads it).  Nothing downstream of layer 17's ``k_proj`` / ``v_proj``
therefore has a consumer:

    input_layernorm -> k_proj, v_proj, RoPE(k), cache.update    <- LIVE
    q_proj, attention, o_proj, residual,
    post_attention_layernorm, mlp, residual                     <- DEAD

At the checkpoint's shapes (968 prefix tokens, hidden 2048, ffn 16384, 8 q heads
and 1 kv head of width 256) that is 226.0 of 228.1 MFLOP/token, i.e. **99.1 % of
the last layer** and 5.5 % of the 18-layer LM.

Why a monkeypatch and not an edit
---------------------------------
The prefix deliberately runs on the *installed* transformers, not on our vendored
``pi05_infer.gemma`` fork: the vendoring boundary is drawn at
``PaliGemmaWithExpertModel.__init__`` precisely so that a denoise-kernel change
cannot reach the prefix (that mistake once cost +4 ms, visible only in a
per-stream profile).  So this module changes nothing in ``transformers`` and
nothing in the fork; it swaps the *instance* ``forward`` of one layer object.
The module tree, parameter names and ``state_dict`` are untouched, so RL weight
sync is unaffected.

When it is NOT applied
----------------------
* ``RLINF_SKIP_LAST_LM_LAYER=0`` -- kill switch.
* The model has a VLM value head.  In RLinf, ``openpi_action_model.py``'s
  ``get_value_from_vlm(prefix_output)`` reads exactly this hidden state, gated by
  ``use_vlm_value = value_after_vlm and add_value_head``.  That is **True in
  every shipped pi0.5 PPO config** (``examples/embodiment/config/*_ppo_openpi_pi05.yaml``),
  so the skip must stay off there.  It is False for the DSRL/SAC configs and for
  this inference-only package, which has no value head at all.
* Per call: whenever ``use_cache`` is off, no cache object was handed in,
  ``output_attentions`` is requested, or RoPE tables are missing -- then the
  original layer forward runs, unchanged.

The joint prefix+suffix *training* forward
(``PaliGemmaWithExpertModel.forward``'s third branch) never calls
``GemmaDecoderLayer.forward``: it reaches into ``layer.input_layernorm`` /
``layer.self_attn.q_proj`` / ``layer.mlp`` directly.  It is therefore structurally
untouched by this patch, which only replaces the layer's ``forward``.
"""

from __future__ import annotations

import os
import types

import torch
from transformers.models.gemma.modeling_gemma import rotate_half

ENV_VAR = "RLINF_SKIP_LAST_LM_LAYER"

_FALSEY = {"0", "false", "no", "off", ""}


def skip_enabled() -> bool:
    """Kill switch: ``RLINF_SKIP_LAST_LM_LAYER=0`` restores the full last layer."""
    return os.environ.get(ENV_VAR, "1").strip().lower() not in _FALSEY


def _kv_only_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
    adarms_cond=None,
    **kwargs,
):
    """``GemmaDecoderLayer.forward`` reduced to the part the KV cache needs.

    Returns the layer *input* as the layer output.  That value is not an
    approximation of the real hidden state and must never be consumed -- the
    engine converts it to ``None`` at the ``_build_prefix_cache`` boundary.
    """
    if (
        not use_cache
        or past_key_value is None
        or output_attentions
        or position_embeddings is None
    ):
        # Anything other than the plain prefill-with-cache call goes down the
        # original path; this patch has no opinion about those.
        return self._pi05_full_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            adarms_cond=adarms_cond,
            **kwargs,
        )

    attn = self.self_attn
    # Byte-for-byte the same ops as GemmaDecoderLayer/GemmaAttention, minus q_proj,
    # the attention itself, o_proj, both residuals, post_attention_layernorm and the
    # MLP.  The surviving ops are in the same order and see the same inputs, so the
    # cache they write is bit-identical.
    normed, _gate = self.input_layernorm(hidden_states, adarms_cond)

    hidden_shape = (*normed.shape[:-1], -1, attn.head_dim)
    kv_w = getattr(attn, "_pi05_kv_w", None)
    if kv_w is not None:
        # One [k+v, hidden] GEMM instead of two 256-wide ones -- concatenation along N,
        # so bit-identical (pi05_infer/prefix_qkv_fused.py).
        kv = torch.nn.functional.linear(normed, kv_w)
        k_lin, v_lin = torch.split(kv, attn._pi05_kv_split, dim=-1)
        key_states = k_lin.view(hidden_shape).transpose(1, 2)
        value_states = v_lin.view(hidden_shape).transpose(1, 2)
    else:
        key_states = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    # == the k_embed line of transformers' apply_rotary_pos_emb (unsqueeze_dim=1),
    # with the q_embed line deleted.
    key_states = (key_states * cos.unsqueeze(1)) + (
        rotate_half(key_states) * sin.unsqueeze(1)
    )

    past_key_value.update(
        key_states,
        value_states,
        attn.layer_idx,
        {"sin": sin, "cos": cos, "cache_position": cache_position},
    )
    return (hidden_states,)


def _has_vlm_value_head(model) -> bool:
    """True when something reads the prefix LM's output embedding.

    Mirrors ``use_vlm_value`` in RLinf's ``openpi_action_model.py`` so that this
    file stays correct if the patch is lifted into RLinf.
    """
    if getattr(model, "use_vlm_value", False):
        return True
    cfg = getattr(model, "config", None)
    return bool(
        getattr(cfg, "value_after_vlm", False) and getattr(cfg, "add_value_head", False)
    )


def install_skip_last_lm_layer(model) -> bool:
    """Patch the prefix LM's last decoder layer down to its KV-cache contribution.

    Must run before ``torch.compile``: the compiled ``language_model.forward``
    traces through this layer, and only a pre-compile install lets inductor drop
    the dead ops from the graph instead of merely skipping them at runtime.

    Returns:
        Whether the patch was installed.
    """
    if not skip_enabled():
        return False
    if _has_vlm_value_head(model):
        return False

    lm = model.paligemma_with_expert.paligemma.model.language_model
    if len(lm.layers) == 0:
        return False
    layer = lm.layers[-1]
    if getattr(layer, "_pi05_skip_installed", False):
        return True

    layer._pi05_full_forward = layer.forward
    layer.forward = types.MethodType(_kv_only_forward, layer)
    layer._pi05_skip_installed = True
    return True
