"""Ledger-driven, content-addressed portrait resolution (A-Seam AS-5).

A character's portrait still is identified by the CONTENT of its decoded pixels,
not by a filename or a generation seed: ``portrait_content_hash =
sha256(decoded_pixels.tobytes())``. The file lives at a deterministic,
content-addressed location
``<output>/otr/episodes/_shared/cache/{portrait_content_hash}.png``
(the OH-1 output-tree contract, 2026-06-11: the pool moved from the old
top-level ``otr/stills/`` into the ``episodes/_shared`` system tier --
cache entries are cross-episode copies, NEVER the only copy; the
episode's own ``stills/`` dir holds the asset of record).

Two consequences the platform relies on:

* **Never-overwrite / idempotent.** Identical pixels hash to the same name, so
  re-stamping the same portrait is a no-op write. DIFFERENT pixels (a
  regenerated portrait) hash to a NEW name -- the old file is never clobbered
  and the ledger is re-stamped to point at the new hash. This is what lets the
  3D mesh cache (keyed on ``portrait_content_hash``) stay correct across
  regenerations: a regen MUST invalidate the cached mesh, and a new hash does.
* **One resolver, three consumers.** The image director (C1), the 3D
  ``character_3d`` adapter (B1) and A all resolve a character's portrait through
  THIS module, so they agree on the path + the hash rule instead of drifting.

Character lookup is by ``char_id`` (the alias-stable key), NEVER the display
name (BUG-098). Dependency-free at module scope -- ``folder_paths`` (ComfyUI),
PIL and numpy are imported lazily inside the functions that need them, so
importing this module pulls in nothing heavy (invariant V-12, cold-import).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("OTR.portrait_ledger")

#: ``<output>/otr/episodes/_shared/cache`` -- the content-addressed
#: cross-episode pool (OH-1: moved from the old top-level ``otr/stills``;
#: ``_shared`` is the reserved system tier, never an episode).
_STILLS_SUBDIR = ("otr", "episodes", "_shared", "cache")


class PortraitUnresolved(LookupError):
    """Raised when no ``portrait_content_hash`` can be found for a ``char_id``
    (fail-closed). A consumer must KNOW a portrait is missing rather than
    receive a silently-guessed path."""


def _output_base(output_dir: Optional[str] = None) -> str:
    """The ComfyUI output root (``output_dir`` overrides; tests). Falls back
    to ComfyUI ``folder_paths``, then the CWD -- resolved LAZILY (V-12)."""
    if output_dir is None:
        try:
            import folder_paths  # type: ignore
            output_dir = folder_paths.get_output_directory()
        except Exception:  # noqa: BLE001
            output_dir = "."
    return str(output_dir)


def stills_root(output_dir: Optional[str] = None) -> Path:
    """The ``<output>/otr/episodes/_shared/cache`` directory (the
    content-addressed cross-episode pool; OH-1). ``output_dir``
    overrides the ComfyUI output root (used by tests)."""
    return Path(_output_base(output_dir)).joinpath(*_STILLS_SUBDIR)


def episode_stills_dir(episode_id: str,
                       output_dir: Optional[str] = None) -> Path:
    """``<output>/otr/episodes/<episode_id>/stills`` -- the EPISODE-LOCAL
    still directory (still-spine ST-3 / W3): every still the dispatcher mints
    or cache-reuses is materialized here so the episode folder is the one
    self-contained unit of output (and the 3D plan's future consumers read
    character stills from the same rows). The global pool stays the
    content-addressed cache; this is the per-episode view of it."""
    ep = str(episode_id or "").strip()
    if not ep:
        raise PortraitUnresolved("empty episode_id for episode_stills_dir")
    return Path(_output_base(output_dir)) / "otr" / "episodes" / ep / "stills"


def compute_portrait_hash(decoded_pixels) -> str:
    """``sha256`` over the DECODED pixel bytes (re-encoding-stable identity).

    ``decoded_pixels`` is anything exposing ``.tobytes()`` (a numpy uint8 image
    array) or raw ``bytes`` / ``bytearray`` / ``memoryview``. We hash the
    decoded pixels -- NOT the encoded PNG file -- so two byte-different PNG
    encodings of the same image collapse to one identity.
    """
    if isinstance(decoded_pixels, (bytes, bytearray, memoryview)):
        raw = bytes(decoded_pixels)
    elif hasattr(decoded_pixels, "tobytes"):
        raw = decoded_pixels.tobytes()
    else:
        raise TypeError(
            "decoded_pixels must be bytes-like or expose .tobytes() "
            "(a decoded uint8 image array)"
        )
    return hashlib.sha256(raw).hexdigest()


def portrait_path_for_hash(portrait_content_hash: str,
                           output_dir: Optional[str] = None) -> Path:
    """Deterministic path ``<output>/otr/stills/{hash}.png`` for a hash."""
    h = str(portrait_content_hash or "").strip()
    if not h:
        raise PortraitUnresolved("empty portrait_content_hash")
    return stills_root(output_dir) / f"{h}.png"


def _cast_entry_for_char(ledger: dict, char_id: str) -> Optional[dict]:
    """The cast dict whose ``char_id`` matches (alias-stable key), or None.
    Matches ``char_id`` ONLY -- never the display name (BUG-098)."""
    cid = str(char_id or "")
    if not cid:
        return None
    for entry in (ledger or {}).get("cast") or []:
        if isinstance(entry, dict) and str(entry.get("char_id") or "") == cid:
            return entry
    return None


def portrait_hash_for_char(ledger: dict, char_id: str) -> Optional[str]:
    """The stamped ``portrait_content_hash`` for ``char_id``, or None."""
    entry = _cast_entry_for_char(ledger, char_id)
    if entry is None:
        return None
    h = entry.get("portrait_content_hash")
    return str(h) if h else None


def resolve_portrait_path(ledger: dict, char_id: str,
                          output_dir: Optional[str] = None) -> Path:
    """Resolve ``char_id`` -> its content-addressed portrait PNG path.

    Reads the ``portrait_content_hash`` the ledger stamped for the character
    (looked up by ``char_id``) and returns ``<output>/otr/stills/{hash}.png``.
    Raises :class:`PortraitUnresolved` (fail-closed) if the character is unknown
    or carries no portrait hash -- never returns a guessed path.
    """
    h = portrait_hash_for_char(ledger, char_id)
    if not h:
        raise PortraitUnresolved(
            f"no portrait_content_hash for char_id {char_id!r} "
            f"(lookup is by char_id, never display name)"
        )
    return portrait_path_for_hash(h, output_dir)


def stamp_portrait(ledger: dict, char_id: str, decoded_pixels,
                   output_dir: Optional[str] = None,
                   require_cast_entry: bool = True) -> Path:
    """Hash ``decoded_pixels``, write the PNG content-addressed (never
    overwriting an existing identical file), stamp the cast entry's
    ``portrait_content_hash`` and return the path.

    A regenerated (pixel-different) portrait yields a NEW hash + path and
    re-stamps the ledger -- the prior file is left intact. PIL is imported
    lazily (V-12).

    ``require_cast_entry`` (default True, the BUG-098 fail-closed contract):
    when False, a ``char_id`` with no cast row skips the cast-entry stamp
    instead of raising -- for SYNTHETIC non-cast portrait subjects (the
    station ANNOUNCER radio-style portrait). ``ledger['cast']`` stays
    CastLock's frozen authority: this function never ADDS a cast row; the
    portrait is still recorded by the caller in ``ledger['images']`` (the
    index the video render path resolves ``init_image`` from).
    """
    h = compute_portrait_hash(decoded_pixels)
    path = portrait_path_for_hash(h, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():               # content-addressed -> never overwrite
        _write_png(decoded_pixels, path)
    entry = _cast_entry_for_char(ledger, char_id)
    if entry is None:
        if require_cast_entry:
            raise PortraitUnresolved(
                f"cannot stamp portrait: char_id {char_id!r} not in ledger cast"
            )
        log.info("[portrait_ledger] non-cast portrait char_id=%s -> %s "
                 "(cast stamp skipped; recorded via ledger['images'])",
                 char_id, path.name)
        return path
    entry["portrait_content_hash"] = h
    log.info("[portrait_ledger] stamped char_id=%s -> %s", char_id, path.name)
    return path


def _write_png(decoded_pixels, path: Path) -> None:
    """Encode decoded pixels to a real PNG at ``path`` (PIL, lazy import).
    Accepts a numpy uint8 array (H,W,3|4) or an already-built PIL image."""
    from PIL import Image  # lazy (V-12)
    if hasattr(decoded_pixels, "save"):          # already a PIL image
        decoded_pixels.save(path, format="PNG")
        return
    Image.fromarray(decoded_pixels).save(path, format="PNG")


__all__ = [
    "PortraitUnresolved", "stills_root", "compute_portrait_hash",
    "portrait_path_for_hash", "portrait_hash_for_char",
    "resolve_portrait_path", "stamp_portrait",
]
