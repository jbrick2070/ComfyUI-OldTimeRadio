"""In-tree temp allocator for video-engine intermediates (R1 fix).

Every engine intermediate .mp4 must land under the sanctioned in-tree tmp tier
(otr/episodes/_shared/tmp) -- NOT the ambient system temp dir -- so the OH-3
janitor sweeps it and the soak hygiene gate stays green regardless of whether the
launcher repointed TEMP. Cold-import clean (V-12): stdlib only at module scope;
the paths import is lazy. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

import os
import tempfile


def _in_tree_tmp_dir():
    """otr/episodes/_shared/tmp (created), or None if the output tree cannot be
    resolved (headless CPU unit tests with no ComfyUI output dir)."""
    try:
        try:
            from .._otr_paths import otr_shared_tmp_dir
        except ImportError:
            from _otr_paths import otr_shared_tmp_dir  # type: ignore
        d = str(otr_shared_tmp_dir())
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:  # noqa: BLE001
        return None


def otr_engine_tmp_mp4(prefix: str) -> str:
    """Reserve a unique in-tree ``.mp4`` path. See :func:`otr_engine_tmp_path`,
    which this has always been -- the suffix is now the parameter it implied."""
    return otr_engine_tmp_path(prefix, ".mp4")


def otr_engine_tmp_path(prefix: str, suffix: str = ".mp4") -> str:
    """Reserve a unique in-tree path and return it. The path does NOT exist
    on return (matches the legacy tempfile.mktemp semantics the call sites relied
    on); the caller's ffmpeg/encoder creates it. Fail-closed in production: if the
    in-tree tmp dir cannot be resolved, only OTR_TEST_MODE permits the tempfile
    default -- production raises rather than silently leak to the system temp dir
    (roundtable MUST-FIX #2)."""
    d = _in_tree_tmp_dir()
    if d is None:
        if os.environ.get("OTR_TEST_MODE"):
            d = None  # tempfile default dir, tests only
        else:
            raise RuntimeError(
                "otr_engine_tmp_path: cannot resolve the in-tree tmp dir and "
                "OTR_TEST_MODE is unset -- refusing to leak to the system temp "
                "dir (R1). Check comfy_output_dir()/otr_shared_tmp_dir().")
    # mkstemp reserves a unique name atomically; unlink so we hand back a
    # non-existent path (the legacy mktemp contract). Every OTR ffmpeg cmd
    # passes -y and encode_frames_to_silent_mp4 overwrites, so an existing file
    # would also be fine -- this just removes any future dependency on -y and
    # avoids a 0-byte .mp4 lingering if a writer fails before its first frame
    # (roundtable MUST-FIX #1; the claimed ffmpeg hang does NOT occur today).
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=d)
    os.close(fd)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return path
