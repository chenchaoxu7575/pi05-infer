"""Vendored, **modified** Gemma decoder used by the pi0.5 action expert.

This is *not* a copy of upstream ``transformers.models.gemma`` for its own sake.
It is the +245-line modified fork that carries the bs=1 denoise optimizations:

  * adaRMS modulation threading (``mod=`` on ``GemmaRMSNorm.forward``,
    ``adarms_mod`` on the decoder layer / model forward, ``build_adarms_stack``),
  * fused QKV projection (``build_qkv_fused``),
  * a static KV buffer (``prime_kv_static`` / ``clear_kv_static``),
  * ``refresh_derived_weights`` for in-place weight sync,
  * the two Triton epilogue fusions in :mod:`rlinf_fused_denoise`.

Everything the file does **not** modify -- ``GemmaConfig``, ``PreTrainedModel``,
``GradientCheckpointingLayer``, ``ALL_ATTENTION_FUNCTIONS``, the rope utils, the
output dataclasses -- is imported from the installed ``transformers``; only the
five modified classes and the two kernels live here.

**Only the action expert uses this module.** The PaliGemma prefix keeps using
``transformers.models.gemma`` (built by ``AutoModel.from_config`` inside
``PaliGemmaForConditionalGeneration``). That split is deliberate: a kernel tuned
for the 50-token denoise suffix silently capturing the 968-token prefix is what
caused the +4 ms regression during the fusion work. See ``README.md``.
"""
