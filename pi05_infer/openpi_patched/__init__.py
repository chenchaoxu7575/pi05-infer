"""Vendored, **modified** copies of the two openpi PyTorch model files.

``openpi`` stays installed and is imported normally for everything unchanged.
Only these two carry local modifications:

* ``pi0_pytorch.py``  -- batched SigLIP over all camera views, and device-side
  ``att_masks`` construction (the original built a host tensor from a Python
  list: a sync on the hot path, and illegal during CUDA-graph capture).
* ``gemma_pytorch.py`` -- ``adarms_mod`` plumbing, and the expert resolving to
  :mod:`pi05_infer.gemma` instead of stock ``transformers``.
"""
