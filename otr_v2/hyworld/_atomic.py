"""
_atomic.py  --  Atomic JSON / text writes for HyWorld contract files.
=====================================================================
Phase A robustness: STATUS.json and the job-dir contract files are
read by ComfyUI nodes (poll, renderer) while being written by the
sidecar worker.  A naive ``Path.write_text`` truncates+writes in two
syscalls, so a poll cycle can land on an empty or partial JSON file
and raise ``json.JSONDecodeError``.

Fix: write to a sibling temp file in the same directory, then
``os.replace`` it onto the target path.  ``os.replace`` is atomic on
both POSIX and Windows for same-volume renames, which is the only
case we use it in (the temp file is always created next to the
destination).

This module has no third-party deps and is safe to import from
either the main ComfyUI process or the spawned sidecar (which may
be in a different conda env with a different torch).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically replace ``path`` with ``text``.

    The temp file is written in the same parent directory so the final
    rename is a same-volume operation (atomic on Windows + POSIX).

    Parent directory is created if it doesn't exist.

    Raises whatever ``open``/``write`` raise on disk full / permission
    issues; callers are responsible for catching those.
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # delete=False so we control the rename ourselves; suffix .tmp so a
    # debugging human can identify orphans from a crash mid-write.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync isn't supported on every Windows filesystem
                # (e.g. some network mounts).  The os.replace below is
                # still atomic; we just lose the durability guarantee.
                pass
        os.replace(str(tmp_path), str(path))
    except Exception:
        # Best-effort cleanup of the temp file if the rename failed.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Atomically replace ``path`` with ``json.dumps(data, indent=indent)``.

    ``ensure_ascii=False`` so SIGNAL LOST style anchors with em-dashes
    or curly quotes round-trip without mojibake.
    """
    payload = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write_text(path, payload)
