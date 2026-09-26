"""Startup source-integrity verification — a fail-closed gate for live trading.

This protects the engine against silently executing a stale or divergent copy of its
own source code (the failure mode that scikit-build-core editable installs create when
top-level modules are frozen into ``site-packages`` and shadow the repo):

* **Always (any mode):** every critical module must resolve to a file that lives in the
  repository tree and *not* inside an installed ``site-packages``/``dist-packages``
  shadow. In non-strict mode a violation is only logged.
* **Strict mode** (``CENGINE_STRICT_SOURCE_INTEGRITY=1``, intended for LIVE/production):
  additionally, every resolved source file must be byte-identical to its committed git
  blob. Any deviation raises :class:`SourceIntegrityError` and the caller refuses to
  start. If git is unavailable, a committed ``integrity_manifest.json`` of sha256 digests
  is required instead; if neither is available the check fails closed.

The resolver and git runner are injectable so the logic is unit-testable without a real
git repository or installed package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("market_engine")

# Top-level application-entry modules that are NOT part of a live-redirected wheel
# package (see pyproject ``wheel.packages``) and have historically been frozen into
# site-packages by the editable install, plus the ``cengine`` package itself.
CRITICAL_MODULES: tuple[str, ...] = (
    "run",
    "order_manager",
    "strategies",
    "liquidity",
    "native_adapters",
    "cengine",
)

_SITE_MARKERS: tuple[str, ...] = ("site-packages", "dist-packages")

# Injection seams: a resolver mapping a module name to its file path, and a git runner
# returning stdout (or None on any failure / git-absent).
Resolver = Callable[[str], Optional[str]]
GitRunner = Callable[[list[str], Path], Optional[str]]


class SourceIntegrityError(RuntimeError):
    """Raised when a critical source file is missing, shadowed, or modified."""


def _default_repo_root() -> Path:
    # cengine/integrity.py -> repo root is two parents up.
    return Path(__file__).resolve().parent.parent


def _default_resolver(name: str) -> Optional[str]:
    """Resolve a module name to the file that would back an ``import`` of it."""
    import sys

    module = sys.modules.get(name)
    if module is not None:
        origin = getattr(module, "__file__", None)
        if origin:
            return origin
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and spec.origin not in ("built-in", "frozen", "namespace"):
        return spec.origin
    locations = list(spec.submodule_search_locations or ())
    return locations[0] if locations else None


def _default_git_runner(args: list[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def verify_source_integrity(
    strict: bool,
    *,
    repo_root: Optional[Path] = None,
    module_names: Iterable[str] = CRITICAL_MODULES,
    resolver: Resolver = _default_resolver,
    git_runner: Optional[GitRunner] = _default_git_runner,
    manifest_path: Optional[Path] = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """Verify the engine's critical source is live from the repo and (strict) unmodified.

    Args:
        strict: When True, any violation raises :class:`SourceIntegrityError` (fail
            closed). When False, violations are only logged as warnings.
        repo_root: Repository root; defaults to this file's grandparent.
        module_names: Critical module names to verify.
        resolver: Maps a module name to its backing file path.
        git_runner: Runs ``git`` with args in ``cwd`` and returns stdout (None on
            failure/absence). Pass None to skip git and force the manifest path.
        manifest_path: Committed sha256 manifest used when git is unavailable.
        logger: Logger for non-strict warnings.

    Raises:
        SourceIntegrityError: In strict mode, on the first integrity violation.
    """
    root = (repo_root or _default_repo_root()).resolve()

    def fail(message: str) -> None:
        if strict:
            raise SourceIntegrityError(message)
        logger.warning("source-integrity: %s", message)

    # Phase 1 — liveness: every critical module must resolve inside the repo tree and
    # never from an installed shadow. NOTE the site-marker test must come before the
    # repo-root test because the virtualenv (and its site-packages) lives *under* the
    # repo root, so a frozen shadow is nominally "inside" the repo too.
    resolved: dict[str, Path] = {}
    for name in module_names:
        origin = resolver(name)
        if not origin:
            fail(f"critical module {name!r} could not be resolved to a file")
            continue
        path = Path(origin).resolve()
        if any(marker in path.parts for marker in _SITE_MARKERS):
            fail(f"critical module {name!r} resolves to an installed shadow copy: {path}")
            continue
        try:
            path.relative_to(root)
        except ValueError:
            fail(f"critical module {name!r} resolves outside the repo tree: {path}")
            continue
        resolved[name] = path

    if not strict:
        return

    # Phase 2 (strict only) — provenance: each resolved file must be byte-identical to
    # its committed git blob. Falls back to a committed sha256 manifest without git.
    head = git_runner(["rev-parse", "HEAD"], root) if git_runner is not None else None
    if head is not None:
        for path in resolved.values():
            rel = path.relative_to(root).as_posix()
            committed = git_runner(["rev-parse", f"HEAD:{rel}"], root)
            if committed is None:
                raise SourceIntegrityError(
                    f"{rel} is not tracked in git HEAD; refusing to run unvetted source"
                )
            working = git_runner(["hash-object", rel], root)
            if working is None:
                raise SourceIntegrityError(f"could not hash working copy of {rel}")
            if working != committed:
                raise SourceIntegrityError(
                    f"{rel} differs from its committed HEAD blob "
                    f"(working={working[:12]} head={committed[:12]}); refusing to start"
                )
        return

    manifest_file = manifest_path or (root / "integrity_manifest.json")
    if not manifest_file.exists():
        raise SourceIntegrityError(
            "strict source integrity requested but neither git nor a committed "
            f"{manifest_file.name} is available to verify sources"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for path in resolved.values():
        rel = path.relative_to(root).as_posix()
        expected = manifest.get(rel)
        if expected is None:
            raise SourceIntegrityError(f"{rel} is missing from {manifest_file.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SourceIntegrityError(
                f"{rel} sha256 differs from {manifest_file.name}; refusing to start"
            )
