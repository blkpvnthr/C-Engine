"""Pytest bootstrap: make this git worktree the authoritative import source.

This repo is installed editable via scikit-build-core, which registers a
``sys.meta_path`` finder (``_editable_skbc_cengine``) that hard-maps every
``cengine.*`` and ``market_data.*`` module to absolute paths in the ORIGINAL
checkout. Because meta_path finders run before any ``sys.path`` entry, that
finder shadows this worktree's copies of those packages — so edits made here to
``cengine/*`` or ``market_data/*`` would be silently tested against the other
checkout.

We remove that editable finder (the compiled ``_cengine_native`` extension and
``cengine.metallib`` live directly in site-packages, so native imports still
resolve) and make sure this worktree's root is first on ``sys.path``. Top-level
modules already resolve correctly via ``pythonpath = ["."]`` in pyproject; this
extends the same guarantee to the ``cengine`` and ``market_data`` packages.

Scoped to test runs in this worktree only — it never touches the shared venv.
"""

from __future__ import annotations

import os
import sys

_WORKTREE_ROOT = os.path.abspath(os.path.dirname(__file__))


def _drop_editable_finder() -> None:
    kept = []
    for finder in sys.meta_path:
        module = type(finder).__module__ or ""
        if module.startswith("_editable_skbc") or "skbuild" in module:
            continue  # drop the editable-install finder
        kept.append(finder)
    sys.meta_path[:] = kept


def _purge_foreign_packages() -> None:
    # If any cengine.*/market_data.* module was imported from the OTHER checkout
    # before this conftest ran, evict it so a fresh import resolves to this tree.
    for name, mod in list(sys.modules.items()):
        if name == "cengine" or name.startswith(("cengine.", "market_data")):
            origin = getattr(getattr(mod, "__spec__", None), "origin", None) or ""
            if origin and not origin.startswith(_WORKTREE_ROOT):
                del sys.modules[name]


# Ensure this worktree wins on sys.path.
if _WORKTREE_ROOT in sys.path:
    sys.path.remove(_WORKTREE_ROOT)
sys.path.insert(0, _WORKTREE_ROOT)

_drop_editable_finder()
_purge_foreign_packages()
