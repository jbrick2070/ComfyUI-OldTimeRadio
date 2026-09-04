"""``ffprobe`` -- the ONE tape measure this pack measures media with.

Order 8 of the lean/mean campaign, approved by the operator on 2026-08-23 and
deliberately scoped against the order-7 ruling that came two hours earlier.
That ruling says every video LANE is independent and must not be consolidated;
this module is the carve-out the operator drew himself: the lanes share the
TAPE MEASURE, never the judgment calls. Resolving which ``ffprobe`` binary to
run, launching it, and reading a rational frame-rate string are facts about
this BOX -- they are the same fact for every caller, and getting them wrong is
the same bug every time. What a failed probe MEANS is not a fact about the box,
it is a policy, and every policy stays exactly where it was:

* ``wan_shared`` raises a NAMED ``GraphExecutionError`` -- a missing ffprobe is
  a broken install and the clip contract is unproven.
* ``otr_credits_roll`` raises ``CreditsDataError`` -- credits cannot be laid out
  without the source's dimensions.
* ``otr_master_audio_mux`` / ``otr_silent_composite`` return ``-1`` / ``0.0``
  and report the gate as UNPROVEN -- a finished episode is not thrown away
  because the box lacks a diagnostic tool.
* ``otr_post_upscale_procgen_blend`` / ``otr_scene_aware_scopes`` fall back to a
  documented default and log -- a blemish, never a lost render.
* ``cloud_media_canonical`` raises ``CORRUPT_OUTPUT`` -- partial provider media
  never proceeds.

THREE THINGS THIS FIXES, all of them found by reading the callers rather than
guessed at:

1. **Only ``otr_credits_roll`` honoured ``OTR_FFPROBE``.** Every other caller
   trusted ``PATH`` or a literal ``"ffprobe"``, so on a box where ffmpeg is
   configured but not on ``PATH`` the credits rendered and the clip-contract
   proof did not. One resolver, one answer, everywhere.
2. **The rational frame-rate parse had been re-fixed independently at least
   three times.** ffprobe answers ``r_frame_rate`` as ``"25/1"`` or
   ``"30000/1001"``, and ``float("25/1")`` raises -- so every caller that
   forgot grew its own crash or its own silent zero.
3. **A bare ``"ffprobe"`` in an argv is not a configuration**, it is a hope.
   It is now the ONE thing this module refuses to treat as a caller's choice.

Stdlib only, no ComfyUI, no torch: importing this must never pull a framework
into memory (invariant V-12, the cold-import test). UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import json as _json
import os
import shutil
import subprocess

__all__ = [
    "FFprobeError",
    "FFprobeMissing",
    "resolve_ffprobe",
    "probe_raw",
    "probe_json",
    "parse_rate",
    "parse_fps_int",
]

#: Names that carry NO information. A caller that hands us one of these handed
#: us the default literal from its own signature, not a decision -- see
#: :func:`resolve_ffprobe`.
_BARE_PROBE_NAMES = frozenset({"ffprobe", "ffprobe.exe"})
_BARE_FFMPEG_NAMES = frozenset({"ffmpeg", "ffmpeg.exe"})


class FFprobeError(RuntimeError):
    """A probe that did not answer.

    Raised ONLY for failures of the tool itself -- it could not be found, it
    could not be launched, it timed out, or it returned bytes that are not the
    document it was asked for. It is never raised for what the MEDIA says: a
    clip with no video stream, a duration of zero, or a colour tag the caller
    dislikes are the caller's business, and the caller owns that verdict.
    """


class FFprobeMissing(FFprobeError):
    """No ffprobe binary could be resolved at all.

    Its own class because several callers distinguish "this box has no ffprobe"
    (a degraded but honest run) from "ffprobe looked and did not like the file"
    (a real defect in the media).
    """


def _usable(candidate):
    """``candidate`` as something that can actually be run, or ``None``.

    A path is accepted when the file EXISTS; a bare name is accepted when
    ``PATH`` resolves it. Nothing here ever executes the candidate -- a file
    that exists and is not an ffprobe is a broken install, and the probe call
    that follows says so far more clearly than a guess here could.
    """
    if not candidate:
        return None
    text = str(candidate).strip()
    if not text:
        return None
    if os.path.isfile(text):
        return text
    return shutil.which(text)


def _explicit(value, bare_names):
    """``value`` when it expresses a CHOICE, else ``None``.

    The distinction matters because most callers spell their default as the
    bare tool name in the signature itself (``ffprobe="ffprobe"``). Honouring
    that as a preference would make it beat ``OTR_FFPROBE`` at every call site
    in the pack, which is precisely the bug this module exists to remove. So a
    bare name with no directory reads as "no preference"; anything carrying a
    directory -- including a full path whose basename is also bare -- is a real
    choice and wins outright.
    """
    text = str(value or "").strip()
    if not text:
        return None
    head, base = os.path.split(text)
    if not head and base.lower() in bare_names:
        return None
    return text


def _sibling_of_ffmpeg(ffmpeg):
    """The ``ffprobe`` shipped beside a resolved ``ffmpeg``, or ``None``.

    ffmpeg builds ship the two binaries in one directory, so a box that
    configured one has configured the other. The basename swap preserves the
    surrounding characters and the extension (``ffmpeg.exe`` -> ``ffprobe.exe``,
    ``ffmpeg-7.1`` -> ``ffprobe-7.1``), and the result must EXIST before it is
    offered -- a constructed path is a guess until the filesystem agrees.
    """
    resolved = _usable(ffmpeg)
    if not resolved:
        return None
    head, base = os.path.split(resolved)
    at = base.lower().find("ffmpeg")
    if at < 0:
        return None
    sibling = os.path.join(head, base[:at] + "ffprobe" + base[at + len("ffmpeg"):])
    return sibling if os.path.isfile(sibling) else None


def resolve_ffprobe(preferred=None, *, ffmpeg=None):
    """Which ffprobe THIS box should run, or ``None`` when it has none.

    The order, most explicit first, and every step is skipped unless it
    resolves to something that exists:

    1. ``preferred`` -- a real path, or a non-default name the caller chose.
    2. the sibling of ``ffmpeg`` -- a caller handed a configured ffmpeg (a node
       widget, a profile) was configured for this box too.
    3. ``$OTR_FFPROBE`` -- the operator's explicit pin.
    4. ``ffprobe`` on ``PATH`` -- what almost every box actually uses, and
       deliberately ahead of the ffmpeg-sibling guesses below so a normal
       install keeps resolving exactly as it always did.
    5. the sibling of the ffmpeg this box runs (``.ffmpeg.resolve_ffmpeg``:
       the pin, else PATH, else the Windows install dirs).
    6. the sibling of ``ffmpeg`` on ``PATH``.

    NEVER RAISES. "This box has no ffprobe" is a fact, and each caller has
    already decided what that fact costs it.
    """
    chosen = _usable(_explicit(preferred, _BARE_PROBE_NAMES))
    if chosen:
        return chosen
    from_argument = _explicit(ffmpeg, _BARE_FFMPEG_NAMES)
    if from_argument:
        sibling = _sibling_of_ffmpeg(from_argument)
        if sibling:
            return sibling
    chosen = _usable(os.environ.get("OTR_FFPROBE"))
    if chosen:
        return chosen
    chosen = shutil.which("ffprobe")
    if chosen:
        return chosen
    try:
        from .ffmpeg import resolve_ffmpeg
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        try:
            from _otr_shared.ffmpeg import resolve_ffmpeg  # type: ignore  # nodes/ on sys.path
        except ImportError:
            # _otr_shared/ itself on sys.path -- INSERTED, so the local file
            # shadows the third-party `ffmpeg` package (Fable gate, 2026-09-04).
            from ffmpeg import resolve_ffmpeg  # type: ignore
    for candidate in (resolve_ffmpeg(), shutil.which("ffmpeg")):
        sibling = _sibling_of_ffmpeg(candidate)
        if sibling:
            return sibling
    return None


def probe_raw(args, *, ffprobe=None, ffmpeg=None, timeout=None):
    """Run ffprobe with ``args`` and hand back the finished process.

    ``args`` is everything AFTER the binary -- each caller keeps its own query,
    because what a caller asks for is part of what it is measuring. Returns the
    :class:`subprocess.CompletedProcess` untouched, INCLUDING a non-zero return
    code: whether a refusal is fatal is the caller's policy, and this function
    holds no opinion about it.

    Raises :class:`FFprobeMissing` when no binary resolves, and
    :class:`FFprobeError` when one resolves but cannot be run to completion.
    """
    binary = resolve_ffprobe(ffprobe, ffmpeg=ffmpeg)
    if not binary:
        raise FFprobeMissing(
            "ffprobe not found (OTR_FFPROBE / PATH / ffmpeg sibling)")
    argv = [binary] + [str(a) for a in args]
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout)
    except FileNotFoundError as exc:
        raise FFprobeMissing("ffprobe not found: %s" % exc)
    except subprocess.TimeoutExpired as exc:
        raise FFprobeError("ffprobe timed out after %rs: %s" % (timeout, exc))
    except OSError as exc:
        raise FFprobeError("ffprobe could not be launched (%s): %s"
                           % (binary, exc))


def probe_json(path, entries=None, *, select_streams=None, extra_args=(),
               ffprobe=None, ffmpeg=None, timeout=None):
    """The parsed ``-of json`` document for ``path``.

    ``entries`` is one ``-show_entries`` expression or a sequence of them
    (ffprobe accepts the flag repeatedly, and ``otr_credits_roll`` needs a
    stream query and a format query in a single read). ``extra_args`` carries
    whatever else a caller's query needs -- ``-count_frames``, ``-show_streams``.

    Raises :class:`FFprobeError` on a non-zero exit or unparseable output, so a
    caller that asks for the document either gets a document or gets told why
    not. A caller that wants to inspect the return code itself uses
    :func:`probe_raw`.
    """
    argv = ["-v", "error"]
    if select_streams:
        argv += ["-select_streams", str(select_streams)]
    argv += [str(a) for a in extra_args]
    if entries:
        for entry in ([entries] if isinstance(entries, str) else list(entries)):
            argv += ["-show_entries", str(entry)]
    argv += ["-of", "json", str(path)]
    proc = probe_raw(argv, ffprobe=ffprobe, ffmpeg=ffmpeg, timeout=timeout)
    if proc.returncode != 0:
        raise FFprobeError("ffprobe failed for %r: %s"
                           % (str(path), (proc.stderr or "").strip()[:300]))
    try:
        return _json.loads(proc.stdout or "{}")
    except ValueError as exc:
        raise FFprobeError("ffprobe returned unparseable JSON for %r: %s"
                           % (str(path), exc))


def parse_rate(rate):
    """An ffprobe frame-rate field as float fps, or ``None`` when unreadable.

    ``r_frame_rate`` and ``avg_frame_rate`` come back as RATIONAL STRINGS --
    ``"25/1"``, ``"30000/1001"`` -- and ``float("25/1")`` raises ``ValueError``.
    Callers that forgot that grew either a crash or a silent zero, three times
    over in this repo. A plain number is accepted too, because a container that
    reports one is not an error.

    ``None`` -- never a zero, never a guess -- for empty, ``"N/A"``, a zero or
    unparseable denominator, and any non-positive result. Zero is a legitimate
    answer to "how many frames per second" only for a file with no video, and
    every caller here would rather be told "unknown" than "still".
    """
    if rate is None:
        return None
    text = str(rate).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        if "/" in text:
            numerator_text, _, denominator_text = text.partition("/")
            denominator = float(denominator_text)
            value = (float(numerator_text) / denominator) if denominator else 0.0
        else:
            value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def parse_fps_int(rate):
    """:func:`parse_rate` rounded to a whole fps; ``0`` when unreadable.

    The shape the silent-clip contract proof has always used: it compares the
    emitted clip against an integer engine fps, and ``30000/1001`` is a 30 fps
    clip for that purpose.
    """
    value = parse_rate(rate)
    return int(round(value)) if value else 0
