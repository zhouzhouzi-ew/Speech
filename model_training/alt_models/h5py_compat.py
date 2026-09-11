"""Make size-1 HDF5 attributes read back as Python scalars.

Why this exists
---------------
The MATLAB pipeline writes per-trial attributes as 1-element arrays, so
`g.attrs['paired_diagnostic_block_num']` comes back as `array([8])`, not `8`.
`model_training/dataset.py` copes with that via its `_hdf5_scalar` helper --
which is why *training* works.  `model_training/evaluate_model_helpers.py`
does not: line 186 calls `int(g.attrs['paired_diagnostic_block_num'])`, and

    int(np.array([8]))

was a DeprecationWarning in numpy 1.x and is a hard `TypeError` in numpy 2.x
("only 0-dimensional arrays can be converted to Python scalars").  Observed on
numpy 2.4.3 / Python 3.14.

So the evaluation entry point is broken for *every* model produced by the
MATLAB pipeline, diphone or not.  Rewriting the upstream helper is out of scope
here (the `model_training/` tree is meant to stay byte-identical), so instead
this patch teaches `h5py` itself to hand back a scalar whenever an attribute
holds exactly one element.

Semantics: an attribute of size > 1 is returned untouched, so genuine array
attributes behave exactly as before.  Size-1 attributes become Python scalars,
which is what every caller in this repo assumes.

Usage
-----
    from h5py_compat import install
    install()          # idempotent; call before opening any HDF5 file
"""

from __future__ import annotations

import h5py
import numpy as np

_INSTALLED = False


def install() -> bool:
    """Patch `h5py.AttributeManager.__getitem__`. Returns True on first install."""
    global _INSTALLED
    if _INSTALLED:
        return False

    original = h5py.AttributeManager.__getitem__

    def getitem(self, name):
        value = original(self, name)
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.reshape(-1)[0].item()
        return value

    # Preserve the original so a process that imports both patched and
    # unpatched code paths can still reach it if it ever needs to.
    getitem.__wrapped__ = original  # type: ignore[attr-defined]
    h5py.AttributeManager.__getitem__ = getitem

    _INSTALLED = True
    return True


def uninstall() -> None:
    """Restore the original `h5py.AttributeManager.__getitem__`."""
    global _INSTALLED
    if not _INSTALLED:
        return
    wrapped = h5py.AttributeManager.__getitem__
    original = getattr(wrapped, "__wrapped__", None)
    if original is not None:
        h5py.AttributeManager.__getitem__ = original
    _INSTALLED = False
