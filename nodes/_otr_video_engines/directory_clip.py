"""Directory-clip contract helpers (3D plan 7.2 -- the ONE alpha handoff).

The character_3d lane delivers ``CanonicalClip(type="directory",
pixel_format="rgba", alpha="straight", has_audio=False, frame_count=N)``: a
DIRECTORY of straight-alpha PNG/EXR frames, sorted by name, exactly N of them.
This module is the canonicalize-time validator + the shared frame-listing rule
("dir exists + exactly N sorted nonzero frames") that ``_clip_summary`` /
``build_clip_manifest`` / the soak assertions and ``OTR_SilentComposite``'s
directory read path all consume -- ONE rule, no per-caller drift.

Never trust the schema defaults for a directory clip: ``container="mp4"`` /
``codec="h264"`` are meaningless here and are deliberately ignored. The
``webm/vp9`` and ``mov/prores4444`` alpha-VIDEO branches are CUT from v1
[H-RT, GPT] -- a directory of frames is the only alpha handoff.

``alpha`` must be ``"straight"``: ffmpeg's ``overlay`` filter composites a
non-premultiplied (straight) RGBA input; a premultiplied handoff would
double-multiply at the flatten. Fail closed on anything else.

Dependency-free (stdlib only; V-12 cold-import clean). UTF-8, no BOM, ASCII.
"""
from __future__ import annotations

import os

#: Frame extensions the directory contract accepts (7.2: PNG/EXR frames).
FRAME_EXTS = (".png", ".exr")


def list_directory_frames(path):
    """The frames of a directory clip: every ``FRAME_EXTS`` file in ``path``,
    SORTED BY NAME (the one ordering rule), each nonzero on disk.

    Raises ``ValueError`` (fail closed, named) when the directory is missing,
    contains no frames, or contains a zero-byte frame. Mixed extensions are
    tolerated (sorted together by name); non-frame files are ignored."""
    if not path or not os.path.isdir(path):
        raise ValueError(
            "directory clip: frame directory missing or not a directory: %r"
            % (path,))
    names = sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(FRAME_EXTS)
    )
    if not names:
        raise ValueError(
            "directory clip: %r contains no %s frames" % (path, FRAME_EXTS))
    frames = []
    for name in names:
        fp = os.path.join(path, name)
        if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
            raise ValueError(
                "directory clip: frame %r is missing or zero-byte (a partial "
                "render must never composite)" % fp)
        frames.append(fp)
    return frames


def frame_dir_summary(path, expect_frames=None):
    """Tolerant probe for summaries/manifests: ``(exists, n_frames,
    total_bytes)`` under the SAME rule as :func:`list_directory_frames`,
    never raising. ``exists`` is True only when the directory holds >=1
    sorted nonzero frame AND (when given) exactly ``expect_frames`` of them."""
    try:
        frames = list_directory_frames(path)
    except ValueError:
        return False, 0, 0
    n = len(frames)
    total = sum(os.path.getsize(f) for f in frames)
    if expect_frames is not None and n != int(expect_frames):
        return False, n, total
    return True, n, total


def validate_directory_clip(clip, expect_frames=None):
    """Canonicalize-time validator for a directory CanonicalClip dict
    (3D plan 7.2). Enforces type / pixel_format / alpha / has_audio and
    ``frame_count == frames on disk`` (== ``expect_frames`` when the caller
    supplies the ledger's ``timing.target_frame_count``). Returns the sorted
    frame list on success; raises ``ValueError`` naming the violation."""
    get = clip.get if isinstance(clip, dict) else (
        lambda k, d=None: getattr(clip, k, d))
    if get("type") != "directory":
        raise ValueError(
            "directory clip: type must be 'directory', got %r"
            % (get("type"),))
    if get("pixel_format") != "rgba":
        raise ValueError(
            "directory clip: pixel_format must be 'rgba' (straight-alpha "
            "frames), got %r" % (get("pixel_format"),))
    if get("alpha") != "straight":
        raise ValueError(
            "directory clip: alpha must be 'straight' (ffmpeg overlay "
            "composites non-premultiplied RGBA; premultiplied would "
            "double-multiply at the flatten), got %r" % (get("alpha"),))
    if get("has_audio") is not False:
        raise ValueError(
            "directory clip: has_audio must be False (V-1: only "
            "OTR_MasterAudioMux emits audio), got %r" % (get("has_audio"),))
    frames = list_directory_frames(get("path"))
    declared = int(get("frame_count") or 0)
    if declared != len(frames):
        raise ValueError(
            "directory clip: declared frame_count %d != %d frames on disk "
            "in %r (never trust the declaration)"
            % (declared, len(frames), get("path")))
    if expect_frames is not None and declared != int(expect_frames):
        raise ValueError(
            "directory clip: frame_count %d != the ledger target %d (V-1: "
            "the delivered clip keeps EXACTLY the audio-derived frame count)"
            % (declared, int(expect_frames)))
    return frames


__all__ = ["FRAME_EXTS", "list_directory_frames", "frame_dir_summary",
           "validate_directory_clip"]
