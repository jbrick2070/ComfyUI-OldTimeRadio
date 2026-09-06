"""Credits code identity for Git checkouts AND shipped registry packages.

No Git executable, parent-repository discovery, network or new dependency.
SOURCE is a deterministic SHA-256 fingerprint of the package's root Python
entry points and nodes/**/*.py, not a release version or a Git commit. Runtime
logs, model caches and generated outputs cannot change that fingerprint.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat


class CodeProvenanceError(RuntimeError):
    """Declared code identity is missing or corrupt."""


def _oid(text: str) -> str:
    value = text.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value):
        raise CodeProvenanceError("Git object ID must be 40 or 64 hex characters")
    return value[:8]


def git_short_sha(repo: Path) -> str:
    """Read only this package's Git identity, including linked worktrees."""
    repo = Path(repo)
    git_dir = repo / ".git"
    try:
        if git_dir.is_file():
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: ") or not pointer[8:].strip():
                raise CodeProvenanceError("Invalid .git worktree pointer")
            git_dir = (repo / pointer[8:].strip()).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return _oid(head)
        ref = head[4:].strip()
        parts = ref.split("/")
        if (len(parts) < 3 or parts[0] != "refs"
                or any(p in ("", ".", "..") for p in parts)
                or any(c.isspace() or c in "\\~^:?*[" for c in ref)):
            raise CodeProvenanceError("Invalid Git symbolic HEAD")
        common_dir = git_dir
        if (git_dir / "commondir").exists():
            common = (git_dir / "commondir").read_text(encoding="utf-8").strip()
            if not common:
                raise CodeProvenanceError("Empty Git commondir")
            common_dir = (git_dir / common).resolve()
        for directory in dict.fromkeys((git_dir, common_dir)):
            loose = directory.joinpath(*parts)
            if loose.exists():
                return _oid(loose.read_text(encoding="utf-8"))
            packed = directory / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) == 2 and fields[1] == ref:
                        return _oid(fields[0])
        raise CodeProvenanceError("Git HEAD reference has no object ID")
    except (OSError, UnicodeError) as exc:
        raise CodeProvenanceError(f"Package Git identity is unreadable: {exc}") from exc


def _reject_source_link(path: Path) -> None:
    metadata = path.lstat()
    if (stat.S_ISLNK(metadata.st_mode)
            or getattr(metadata, "st_file_attributes", 0) & 0x400):
        raise CodeProvenanceError("Linked/reparse-point Python source is not a package receipt")


def source_fingerprint(repo: Path) -> str:
    """Hash named Python source bytes, not caches or enclosing repositories."""
    repo = Path(repo)
    try:
        required = (repo / "__init__.py", repo / "nodes" / "otr_credits_roll.py")
        if not all(p.is_file() for p in required):
            raise CodeProvenanceError("Package entry point or credits source is missing")
        _reject_source_link(repo / "nodes")
        # iterdir propagates directory-read failures; glob may suppress them.
        paths = [path for path in repo.iterdir() if path.name.endswith(".py")]
        def unreadable(error):
            raise error

        for directory, _dirs, files in os.walk(
                repo / "nodes", followlinks=False, onerror=unreadable):
            for name in _dirs:
                _reject_source_link(Path(directory) / name)
            paths.extend(Path(directory) / name for name in files if name.endswith(".py"))
        digest = hashlib.sha256(b"otr-python-source-v1\0")
        for path in sorted(paths, key=lambda p: p.relative_to(repo).as_posix()):
            _reject_source_link(path)
            name = path.relative_to(repo).as_posix().encode("utf-8")
            payload = path.read_bytes()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()
    except (OSError, UnicodeError) as exc:
        raise CodeProvenanceError(f"Package Python source is unreadable: {exc}") from exc


def code_receipt(repo: Path) -> tuple[str, str]:
    """An explicit SOURCE receipt is not a fallback for broken Git metadata."""
    repo = Path(repo)
    # lexists preserves a broken .git symlink as a corrupt checkout, not ZIP.
    if os.path.lexists(repo / ".git"):
        return "COMMIT:", git_short_sha(repo)
    return "SOURCE:", "sha256 " + source_fingerprint(repo)[:16]
