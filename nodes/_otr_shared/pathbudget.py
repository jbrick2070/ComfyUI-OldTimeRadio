"""``pathbudget`` -- the ONE answer to "will Windows let us write this path".

WHY THIS FILE EXISTS (2026-09-07, PBUG-20260907-01). A 40-minute render on an
8 GB 4060 wrote the script, the voices, the music, every still and all eight LTX
video beats, then died at the caption burn with ``FileNotFoundError: [Errno 2]``
naming a directory that plainly existed. Errno 2 is what Windows returns for a
path past MAX_PATH: the overflow surfaces as a missing parent, so the message
points at the wrong thing and the real cause is four characters of filename.

THE ARITHMETIC, measured on that box rather than argued. ComfyUI Desktop puts
the output tree 96 units deep before ``episodes\\`` ends. The episode id then
appears TWICE -- once as the folder, once as the filename stem -- and every
stage APPENDS rather than replaces, so the stem grows
``<id>`` -> ``<id>_silent`` -> ``<id>_silent_procgen_blended`` ->
``<id>_silent_procgen_blended_captioned``. Measured lengths for a 65-character
episode id, against Windows' 260-unit limit:

    SilentComposite                238  ok
    ProcgenBlend                   254  tight
    ProcgenBlend __nobars_tmp      266  OVER
    ProcgenBlend __bars_tmp        264  OVER
    CaptionBurn (as it was)        264  OVER   <- the one that actually failed
    CreditsRoll .concat.txt        265  OVER
    Mux _final.mp4                 260  OVER

Five stages over the line and only one of them had ever fired, because the
procgen node happened to be bypassed and the render never reached the mux. That
is luck, not correctness, which is why the rule lives in one module now instead
of being rediscovered per node.

THE THREE WAYS TO SOLVE THIS, and why this module picks the ones it picks:

1. DO NOT GENERATE PATHOLOGICAL NAMES. The folder already carries the episode
   identity, so repeating it in the filename and then stacking stage suffixes on
   top buys nothing. :func:`compact_artifact` drops the STAGE suffixes -- never
   the identity -- which is the rule ``otr_credits_roll._credits_artifact_paths``
   already stated for its own scratch and which the other nodes never adopted.
2. THE ``\\?\`` EXTENDED-LENGTH PREFIX (:func:`long_path`) is the canonical
   Win32 answer and needs no machine configuration. It is applied at the SYSCALL
   BOUNDARY only and never stored, because a prefixed string would leak into
   ledgers, logs and equality checks. Note the two places it must NOT go: an
   ffmpeg FILTERGRAPH argument (``ass=`` is parsed, and a backslash is escape
   syntax there -- the caption node already sidesteps this by passing a bare
   basename with cwd set to the folder), and anywhere a path is compared as text
   against an unprefixed twin.
3. ``LongPathsEnabled`` in the registry is deliberately NOT used. It is
   machine-wide, needs elevation, and a stranger installing this pack from
   ComfyUI Manager will not have set it. A fix the user has to perform is not a
   fix for a zero-friction install.

The budget is 250, not 260, and the ten units of headroom are load-bearing: a
name that merely fits will be handed to a LATER stage that appends its own
suffix, so ``_final.mp4`` lands exactly on 260 and fails while looking legal.
"""
from __future__ import annotations

import hashlib
import os

#: Windows MAX_PATH is 260 UTF-16 code units INCLUDING the terminating NUL, so
#: 260 already fails. 250 leaves room for the suffix the next stage appends.
#: ``otr_credits_roll`` uses the same number; ``tests/`` asserts they agree,
#: because one rule spelled twice is how the caption node drifted from it.
WINDOWS_PATH_BUDGET = 250

#: Stage suffixes the composite chain accumulates, longest-first so a compound
#: tail is removed in one pass. Kept in step with the strip lists in
#: ``otr_credits_roll`` and ``otr_master_audio_mux``: those nodes identify their
#: input by reducing the stem back to the episode id, so anything dropped here
#: must be something they also drop, or the compacted name stops matching.
STAGE_SUFFIXES = (
    "_silent_procgen_blended_captioned_with_credits",
    "_procgen_blended_captioned_with_credits",
    "_captioned_with_credits",
    "_silent_procgen_blended_captioned",
    "_silent_procgen_blended",
    "_procgen_blended",
    "_with_credits",
    "_captioned",
    "_blend",
    "_silent",
)


def path_length(path: str) -> int:
    """The length Windows actually counts: UTF-16 code units, not characters.

    An emoji or a CJK title is one character and two units, so measuring
    ``len(str)`` understates a non-ASCII path and passes a check the OS fails.
    """
    return len(os.path.abspath(path).encode("utf-16-le", "surrogatepass")) // 2


def path_fits(path: str, budget: int = WINDOWS_PATH_BUDGET) -> bool:
    """True when ``path`` is safe to write. Always True off Windows."""
    if os.name != "nt":
        return True
    return path_length(path) <= budget


def long_path(path: str) -> str:
    """``path`` prepared for a Python file syscall, prefixed only if needed.

    Returns the path unchanged off Windows, for short paths, for UNC and
    already-prefixed paths, and for anything not absolute -- so it is safe to
    wrap every ``open``/``stat``/``remove`` with it. USE IT AT THE CALL, never
    to build a value that gets stored or compared: the prefix would leak into
    the ledger and two spellings of one file would stop comparing equal.
    """
    if os.name != "nt" or not path:
        return path
    if path.startswith("\\\\?\\") or path.startswith("\\\\.\\"):
        return path
    absolute = os.path.abspath(path)
    if path_length(absolute) <= WINDOWS_PATH_BUDGET:
        return path
    if absolute.startswith("\\\\"):          # UNC: \\server\share -> \\?\UNC\server\share
        return "\\\\?\\UNC" + absolute[1:]
    return "\\\\?\\" + absolute


def strip_stage_suffixes(stem: str) -> str:
    """``stem`` with the pipeline's stage suffixes removed, longest match first."""
    for suffix in STAGE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def compact_artifact(in_dir: str, stem: str, suffix: str, ext: str) -> str:
    """A writable path for a NAMED deliverable, ordinary name when it fits.

    Deliverables keep their ordinary ``<stem><suffix><ext>`` unless that will
    not fit; then the STAGE suffixes are dropped from the stem so the name
    reduces to the episode id the folder already carries. That specific
    reduction is what keeps the downstream nodes working: both
    ``otr_credits_roll`` and ``otr_master_audio_mux`` recognise their input by
    stripping exactly these suffixes and requiring what remains to equal the
    episode id, so ``<id>_captioned.mp4`` still matches while an invented short
    name would silently drop them into a legacy branch.

    THE EPISODE IDENTITY IS NEVER SHORTENED. If even the reduced form will not
    fit, the ordinary name is returned so the failure is loud, in the same place
    it is today, rather than quietly renamed into something still broken.
    """
    ordinary = os.path.join(in_dir, f"{stem}{suffix}{ext}")
    if path_fits(ordinary):
        return ordinary
    reduced = strip_stage_suffixes(stem)
    episode_id = os.path.basename(os.path.abspath(in_dir))
    if reduced and reduced.casefold() == episode_id.casefold():
        candidate = os.path.join(in_dir, f"{reduced}{suffix}{ext}")
        if path_fits(candidate):
            return candidate
    return ordinary


def compact_scratch(in_dir: str, hint: str, ext: str) -> str:
    """A writable path for a THROWAWAY file, ordinary name when it fits.

    For intermediates nothing else identifies by name -- a subtitle sidecar, a
    concat list, a bars/nobars temporary. These may be renamed freely, so when
    the ordinary name will not fit they collapse to a short deterministic digest
    of it: stable across re-runs (so a retry overwrites its own file instead of
    littering), and distinct per hint (so two scratch files in one folder cannot
    collide). The name is plain ASCII with no ffmpeg filtergraph syntax in it.
    """
    ordinary = os.path.join(in_dir, f"{hint}{ext}")
    if path_fits(ordinary):
        return ordinary
    digest = hashlib.sha1(hint.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return os.path.join(in_dir, f"otr_tmp_{digest}{ext}")
