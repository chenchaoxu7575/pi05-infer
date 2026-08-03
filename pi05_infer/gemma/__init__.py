"""Vendored, **modified** Gemma decoder used by the pi0.5 action expert.

The +245-line fork that carries the bs=1 denoise optimizations: adaRMS
modulation threading, fused QKV, a static KV buffer, ``refresh_derived_weights``
for in-place weight sync, and the two Triton epilogue fusions in
:mod:`rlinf_fused_denoise`. Everything it does not modify is imported from the
installed ``transformers``.

WARNING: only the action expert uses this. The PaliGemma prefix keeps stock
``transformers.models.gemma``. That split is what stops a kernel tuned for the
50-token denoise suffix from silently capturing the 968-token prefix.
"""
