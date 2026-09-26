"""Unit tests for cengine.integrity.verify_source_integrity.

The resolver and git runner are injected so both the passing and fail-closed paths are
exercised deterministically, without depending on the working tree's real git state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cengine.integrity import SourceIntegrityError, verify_source_integrity

NAMES = ("run", "order_manager")


def _repo_with_modules(root: Path, names=NAMES) -> dict[str, str]:
    """Create <name>.py files under root; return a resolver mapping name -> path."""
    mapping: dict[str, str] = {}
    for i, name in enumerate(names):
        f = root / f"{name}.py"
        f.write_text(f"# module {name}\nX = {i}\n", encoding="utf-8")
        mapping[name] = str(f)
    return mapping


def _resolver(mapping: dict[str, str]):
    return lambda name: mapping.get(name)


def _git(head="HEAD_SHA", committed=None, working=None):
    """Fake git runner responding to rev-parse HEAD / HEAD:<rel> and hash-object."""
    committed = committed or {}
    working = working or {}

    def runner(args, cwd):
        if args == ["rev-parse", "HEAD"]:
            return head
        if len(args) == 2 and args[0] == "rev-parse" and args[1].startswith("HEAD:"):
            return committed.get(args[1][len("HEAD:") :])
        if len(args) == 2 and args[0] == "hash-object":
            return working.get(args[1])
        return None

    return runner


# --- liveness (phase 1) ---------------------------------------------------------------


def test_nonstrict_shadow_only_warns(tmp_path, caplog) -> None:
    mapping = _repo_with_modules(tmp_path)
    mapping["run"] = str(tmp_path / "site-packages" / "run.py")  # a frozen shadow
    # Non-strict must never raise, even on a violation.
    verify_source_integrity(
        strict=False,
        repo_root=tmp_path,
        module_names=NAMES,
        resolver=_resolver(mapping),
        git_runner=None,
    )
    assert any("installed shadow" in r.message for r in caplog.records)


def test_strict_shadow_raises(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    mapping["run"] = str(tmp_path / "lib" / "site-packages" / "run.py")
    with pytest.raises(SourceIntegrityError, match="installed shadow"):
        verify_source_integrity(
            strict=True,
            repo_root=tmp_path,
            module_names=NAMES,
            resolver=_resolver(mapping),
            git_runner=_git(),
        )


def test_strict_outside_repo_raises(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    mapping["run"] = "/somewhere/else/run.py"
    with pytest.raises(SourceIntegrityError, match="outside the repo"):
        verify_source_integrity(
            strict=True,
            repo_root=tmp_path,
            module_names=NAMES,
            resolver=_resolver(mapping),
            git_runner=_git(),
        )


# --- provenance via git (phase 2) -----------------------------------------------------


def test_strict_git_match_passes(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    rels = [f"{n}.py" for n in NAMES]
    committed = {rel: f"blob-{rel}" for rel in rels}
    working = dict(committed)  # working tree == HEAD
    verify_source_integrity(
        strict=True,
        repo_root=tmp_path,
        module_names=NAMES,
        resolver=_resolver(mapping),
        git_runner=_git(committed=committed, working=working),
    )


def test_strict_git_mismatch_raises(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    rels = [f"{n}.py" for n in NAMES]
    committed = {rel: f"blob-{rel}" for rel in rels}
    working = dict(committed)
    working["run.py"] = "tampered"  # working differs from committed
    with pytest.raises(SourceIntegrityError, match="differs from its committed HEAD blob"):
        verify_source_integrity(
            strict=True,
            repo_root=tmp_path,
            module_names=NAMES,
            resolver=_resolver(mapping),
            git_runner=_git(committed=committed, working=working),
        )


def test_strict_untracked_raises(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    committed = {"order_manager.py": "blob"}  # run.py absent from HEAD
    working = {"order_manager.py": "blob", "run.py": "anything"}
    with pytest.raises(SourceIntegrityError, match="not tracked in git HEAD"):
        verify_source_integrity(
            strict=True,
            repo_root=tmp_path,
            module_names=NAMES,
            resolver=_resolver(mapping),
            git_runner=_git(committed=committed, working=working),
        )


# --- provenance via manifest fallback (no git) ----------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_no_git_no_manifest_fails_closed(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    with pytest.raises(SourceIntegrityError, match="neither git nor a committed"):
        verify_source_integrity(
            strict=True,
            repo_root=tmp_path,
            module_names=NAMES,
            resolver=_resolver(mapping),
            git_runner=None,  # git unavailable
            manifest_path=tmp_path / "missing.json",
        )


def test_strict_manifest_match_passes(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    manifest = {f"{n}.py": _sha256(tmp_path / f"{n}.py") for n in NAMES}
    manifest_file = tmp_path / "integrity_manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    verify_source_integrity(
        strict=True,
        repo_root=tmp_path,
        module_names=NAMES,
        resolver=_resolver(mapping),
        git_runner=None,
        manifest_path=manifest_file,
    )


def test_strict_manifest_mismatch_raises(tmp_path) -> None:
    mapping = _repo_with_modules(tmp_path)
    manifest = {f"{n}.py": "deadbeef" for n in NAMES}  # wrong digests
    manifest_file = tmp_path / "integrity_manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SourceIntegrityError, match="sha256 differs"):
        verify_source_integrity(
            strict=True,
            repo_root=tmp_path,
            module_names=NAMES,
            resolver=_resolver(mapping),
            git_runner=None,
            manifest_path=manifest_file,
        )
