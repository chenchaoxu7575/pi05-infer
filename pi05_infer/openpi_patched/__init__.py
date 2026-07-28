"""Vendored, **modified** copies of the two openpi PyTorch model files.

``openpi`` itself stays installed and is imported normally for everything we did
not change (``Observation``, ``transforms``, ``Pi0Config``, checkpoint loading,
``preprocessing_pytorch``, ``models.gemma``). Only these two files carry local
modifications, so only these two are vendored:

* ``pi0_pytorch.py``  -- batched SigLIP over all camera views, and the
  device-side ``att_masks`` construction (the original built a host tensor from a
  Python list, which is a sync on the hot path and illegal during CUDA-graph
  capture).
* ``gemma_pytorch.py`` -- ``adarms_mod`` plumbing through
  ``PaliGemmaWithExpertModel.forward`` into the expert, and the expert is built
  from :mod:`pi05_infer.gemma.modeling_gemma` instead of ``transformers``.

The PaliGemma **prefix** is still ``transformers.PaliGemmaForConditionalGeneration``
and therefore still uses ``transformers.models.gemma``; only the **expert** is
ours. See ``pi05_infer/gemma/__init__.py``.
"""
