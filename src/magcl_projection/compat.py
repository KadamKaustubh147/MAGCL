"""Two tiny patches RecBole 1.2.0 (2022) needs on current SciPy/PyTorch.

Import this before anything else touches recbole.

1. RecBole's LightGCN-family adjacency builders call `dok_matrix._update(...)`
   (e.g. recbole/model/general_recommender/lightgcn.py). That method is gone
   on every SciPy version tested here. Modern SciPy also moved dok_matrix's
   actual storage into a private `self._dict` attribute -- dok_matrix still
   inherits from `dict` for compatibility, but its own `.update()` now raises
   NotImplementedError and writes to `self` would never be read back. So the
   one-line fix has to target `self._dict`, not `self`.

2. PyTorch 2.6 flipped `torch.load`'s default from `weights_only=False` to
   `True`. RecBole's checkpoints (trainer.py's `_save_checkpoint`) pickle a
   dict containing the Config and other plain Python objects, not just
   tensors, so the new default's stricter unpickler rejects them outright --
   confirmed by running an actual RecBole checkpoint through `evaluate()`
   after `weights_only`'s default flipped (Unsupported operand error).
   The checkpoint is one we just wrote ourselves, so trusting it is fine.
"""

import scipy.sparse as sp

if not hasattr(sp.dok_matrix, "_update"):
    sp.dok_matrix._update = lambda self, data: self._dict.update(data)

import functools

import torch

if not getattr(torch.load, "_magcl_patched", False):
    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def _torch_load_trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    _torch_load_trusted._magcl_patched = True
    torch.load = _torch_load_trusted
