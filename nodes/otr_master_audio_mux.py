"""OTR_MasterAudioMux -- the TERMINAL audio mux (A-S3 / CW-4).

THE ONLY node that may add audio to a video (invariant V-1). It muxes the FROZEN
master episode audio onto the always-silent composite with ``-c:a copy`` -- the
audio stream passes through with ZERO re-encode, so the output's audio is
byte-identical to the master. ``ffmpeg -shortest`` is FORBIDDEN here (it would
truncate the episode to the shorter stream and silently drop an audio tail); the
silent composite is already built to the audio-derived frame budget, so a
duration assertion (within 1/fps) runs BEFORE the mux to catch any drift.

Replaces the legacy ``OTR_VideoComposite`` audio path (which re-encoded to AAC in
its humo_concat mode and used ``-shortest``). This node does NO model work and
holds NO CUDA residency -- it is pure ffmpeg, so the BUG-291 patcher-ref / NVML
guards do not apply here; it only polls the interrupt flag so Cancel is honoured.

Audio-identity is asserted by decoding both the muxed output's audio and the
master to canonical PCM and comparing SHA-256 -- container-agnostic proof that
the audio was copied, not re-encoded or trimmed. Cold-import clean (stdlib only).

IT IS ALSO THE PUBLICATION BOUNDARY (2026-08-15, build contract D5a). Producing
the episode and publishing it are two decisions, and only the second one is
governed by the source's rights. The archival final is written for every
episode that gets this far; the OBS copy -- the file the operator watches -- is
made only when the ledger's publication-eligibility receipt says so. That rule
used to be enforced upstream by refusing to freeze, which destroyed a finished
render to prevent a copy; here it withholds the copy and keeps the render.
"""
from __future__ import annotations

import collections
import hashlib
import math
import os
import re
import shutil
from pathlib import Path

import logging

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore
try:
    from ._otr_shared import proc as otr_proc
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import proc as otr_proc  # type: ignore

try:  # ComfyUI loads these node modules flat as well as packaged
    from ._otr_shared import ffprobe as _ffp
except ImportError:  # pragma: no cover -- flat (sys.path) test import
    from _otr_shared import ffprobe as _ffp  # type: ignore

# The show prefix the published (watching) filename drops. Spelled once, in the
# ledger, whose `_published_obs_path` must accept the name written here; bound
# at import so a test that stubs the ledger module cannot silently change the
# name (PBUG-20260904-06).
try:
    from ._otr_ledger import SHOW_PREFIX as _SHOW_PREFIX
except ImportError:  # pragma: no cover -- flat (sys.path) test import
    from _otr_ledger import SHOW_PREFIX as _SHOW_PREFIX  # type: ignore

log = logging.getLogger("OTR")


def _ffmpeg_bin(ffmpeg: str) -> str:
    """The ffmpeg this box should run, or ``""`` when it has none.

    HONOURS ``OTR_FFMPEG`` BEFORE PATH (2026-08-28). It did not, and that was
    the mirror image of a bug the pack had already fixed once: the shared
    ``_otr_shared/ffprobe.py`` resolver exists because only `otr_credits_roll`
    honoured ``OTR_FFPROBE`` while every other caller trusted PATH. That
    consolidation was scoped to the PROBE; the ENCODER kept the same hole here,
    in `otr_caption_burn`, `otr_master_audio_mux` and `otr_silent_composite` --
    which are the caption burn, the terminal audio mux and the silent-video
    normalize, i.e. the LAST three stages of an episode.

    So on a box where ffmpeg is reachable only through ``OTR_FFMPEG`` -- the
    AMD/Mac/alternate-box case the variant workflows exist for -- every earlier
    stage would succeed (the video engines all honour the variable) and the
    episode would die at the end, having spent the whole render.

    A NODE WIDGET NO LONGER WINS -- it no longer even arrives (2026-09-04).
    Each execute method discards its `ffmpeg` widget before anything calls this,
    so what reaches here is either nothing or a value a TRUSTED caller already
    resolved. `OTR_FFMPEG` is the operator's channel and PATH the last resort.
    Left as it was, the next reader would re-wire the widget to match this
    paragraph and quietly reopen the hole.

    ONE OWNER ANSWERS NOW (``_otr_shared.ffmpeg.resolve_ffmpeg``, 2026-09-04),
    and the widget's own default literal ``"ffmpeg"`` is not a choice: with
    ffmpeg on PATH that literal used to win here and the pin was never read.
    """
    try:
        from ._otr_shared.ffmpeg import resolve_ffmpeg
    except ImportError:  # pragma: no cover -- flat (sys.path) test import
        from _otr_shared.ffmpeg import resolve_ffmpeg  # type: ignore
    return resolve_ffmpeg(ffmpeg) or ""


def _ffprobe_bin() -> str:
    """The ffprobe this box should run, or ``""`` when it has none.

    THE POLICY IS THIS MODULE'S AND IT DOES NOT MOVE: an absent probe yields
    ``-1`` and a duration receipt that says UNPROVEN out loud, never a lost
    episode. Only the SEARCH is shared -- and sharing it is why ``OTR_FFPROBE``
    now reaches the duration gate, which it never did before.
    """
    return _ffp.resolve_ffprobe() or ""


def _run(cmd):
    return otr_proc.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _probe_float(path: str, stream: str) -> float:
    """Duration (s) of the first ``stream`` (``v:0`` / ``a:0``) via ffprobe."""
    fp = _ffprobe_bin()
    if not fp:
        return -1.0
    p = _run([fp, "-v", "error", "-select_streams", stream, "-show_entries",
              "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1", path])
    try:
        return float((p.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        # container duration fallback
        p2 = _run([fp, "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=nokey=1:noprint_wrappers=1", path])
        try:
            return float((p2.stdout or "").strip())
        except ValueError:
            return -1.0


# `_count_audio_streams` was removed 2026-08-28: no caller anywhere. The
# identically named helper in tests/test_credits_roll_spec.py is that test's
# own local, not a consumer of this one.


def audio_pcm_sha(path: str, ffmpeg: str = "ffmpeg") -> str:
    """SHA-256 of decoded s16le mono @24k -- codec/container-agnostic audio
    identity (same method the A-S2 mux probe used). '' on failure.

    Resolves its binary through :func:`_ffmpeg_bin` -- explicit argument, then
    ``OTR_FFMPEG``, then PATH -- so the fail-closed identity proof honours the
    same resolution order the encode did. It used to call ``shutil.which``
    directly, which meant that on an env-only install (ffmpeg reachable ONLY
    via ``OTR_FFMPEG``) the mux encoded fine and then this returned '' --
    failing a FINISHED episode at the last boundary.
    """
    fp = _ffmpeg_bin(ffmpeg)
    if not fp:
        return ""
    raw = otr_proc.run(
        [fp, "-v", "error", "-i", path, "-map", "0:a", "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", "-"],
        capture_output=True)
    if raw.returncode != 0 or not raw.stdout:
        return ""
    return hashlib.sha256(raw.stdout).hexdigest()


#: The legacy credits-tail ceiling when no roll declares its own duration.
_MAX_CREDITS_TAIL_S_DEFAULT = 45.0


def duration_receipt_line(v_dur: float, a_dur: float, max_tail_s: float,
                          tail_src: str, tol: float) -> str:
    """The ``duration_check`` receipt line. Numbers in, one line out.

    Pure ON PURPOSE, and extracted after the 4060 made the right criticism of
    how this was first covered. The original test for PBUG-20260830-01 asserted
    on the AST -- branch shape and string presence -- because reaching the real
    branch needs ffprobe, two rendered media files and a full mux. That test was
    mutation-verified and did detect the defect, but as the 4060 put it: a
    source-shape assertion "proves the code is written correctly, not that it
    runs correctly", cannot catch a behavioural regression that preserves the
    shape, and goes red on a refactor that preserves the behaviour. Both are the
    wrong failure mode for a receipt.

    So the decision moved somewhere it can be CALLED. Three verdicts, and they
    are three different claims that must never collapse into each other:

    ``UNPROVEN``
        ``_probe_float`` returns ``-1.0`` when ffprobe is absent or the duration
        is unparsable. The budget comparison is then meaningless, so the gate is
        SKIPPED -- and a skipped gate must not report as a passed one. Reported
        rather than made fatal because this is the final sanity ceiling, not the
        primary correctness guard (the CreditsRoll and composite frame budgets
        are), and refusing here would lose a finished episode on a box that
        merely lacks ffprobe.
    ``OVER_BUDGET``
        The video runs past the audio by more than the credits tail allows. This
        no longer raises -- the operator ruled on 2026-08-30 that a length
        disagreement must never discard a finished episode -- which is exactly
        why the receipt has to say so. With the raise gone this line is the ONLY
        compact signal a reader gets, and an always-OK receipt made every
        overshoot invisible in the summary this node exists to emit. The overage
        rides the line at full precision so a reader never has to go find the
        warning to get the number.
    ``OK``
        The fall-through, reachable only when the probe worked AND the budget
        held.

    The budget arithmetic mirrors the warning above it deliberately: both ask
    ``v_dur > a_dur + max_tail_s + tol``, so the receipt and the log can never
    disagree about whether this episode was over.
    """
    head = (f"duration_check v={v_dur:.3f}s a={a_dur:.3f}s "
            f"tail_budget={max_tail_s:.1f}s ({tail_src})")

    if v_dur < 0 or a_dur < 0:
        return head + " UNPROVEN (duration probe failed -- gate SKIPPED, not passed)"

    if v_dur > a_dur + max_tail_s + tol:
        over_by = (v_dur - a_dur) - max_tail_s - tol
        return head + f" OVER_BUDGET by {over_by:.4f}s (published anyway)"

    return head + " OK"


def _credits_tail_ceiling() -> float:
    """The ``OTR_MAX_CREDITS_TAIL_S`` ceiling -- NAMED and ignored when
    malformed, never fatal.

    This knob is read at the LAST node of the graph, after the whole episode has
    rendered. It used to be a bare ``float(otr_env.get(...))``, so a single
    typo in a server's launch environment (``45s``, ``forty-five``) killed a
    finished episode at the finish line with an uncaught ValueError -- hours of
    render lost to a value that only widens a sanity ceiling.

    That is the ``PBUG-20260723-02`` shape this build has now closed three times
    over: a knob exported at launch cannot bind work submitted to an
    already-booted server, so a malformed one must be IGNORED, never FATAL. The
    house posture is `otr_silent_composite._unsharp_amount`; this adds the
    WARNING that one omits, because a ceiling silently reverting to the default
    is a ceiling the operator thinks they moved. (It is not the mux's only env
    read -- `_ffmpeg_bin` consults `OTR_FFMPEG` in the same call.)"""
    raw = otr_env.get("OTR_MAX_CREDITS_TAIL_S")
    if raw in (None, ""):
        return _MAX_CREDITS_TAIL_S_DEFAULT
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning(
            "[OTR_MasterAudioMux] OTR_MAX_CREDITS_TAIL_S=%r is not a number; "
            "IGNORING it and using the %.1fs default. A malformed knob must "
            "not lose a finished episode at the mux (PBUG-20260723-02).",
            raw, _MAX_CREDITS_TAIL_S_DEFAULT)
        return _MAX_CREDITS_TAIL_S_DEFAULT


class _Interrupted(RuntimeError):
    pass


def _poll_interrupt():
    """Honour ComfyUI Cancel (BUG-073). No-op outside ComfyUI."""
    try:
        import comfy.model_management as mm  # type: ignore
        mm.throw_exception_if_processing_interrupted()
    except _Interrupted:
        raise
    except Exception:  # noqa: BLE001  (not running under comfy / no such API)
        return


#: Slack on the duration gate, in FRAMES. Not a fudge factor -- a quantization
#: bound. The published video is a CONCAT of the body and the silent credits roll,
#: and ffmpeg lands each segment on a frame boundary, so the assembled duration can
#: exceed (body + declared_credits_tail) by a fraction of a frame per segment plus
#: container-timebase rounding. ONE frame does not cover that.
#:
#: Live 2026-07-14, 420w scifi_codex re-leg: video 294.1600s, audio 218.9307s, an
#: excess of 75.2293s against a declared credits tail of ~75.18s -- over budget by
#: ~0.05s, i.e. about ONE AND A QUARTER frames. The gate refused to publish a
#: finished episode over three hundredths of a second.
#:
#: THREE frames (0.12s at 25fps) covers a per-segment rounding plus timebase slop.
#: This is NOT the "blind widening" the guard's comment warns against: the gate
#: exists to catch GROSS drift -- a composite/credits frame-budget bug that runs
#: the video tens of seconds (or 2x) past the audio -- and 0.12s of quantization
#: slack leaves that protection completely intact.
DEFAULT_DURATION_TOL_FRAMES = 3.0


def mux_master_audio(silent_video_path: str, master_audio_path: str, out_path: str,
                     ffmpeg: str = "ffmpeg", fps: int = 25,
                     duration_tol_frames: float = DEFAULT_DURATION_TOL_FRAMES,
                     declared_credits_tail_s: float = 0.0):
    """Mux the frozen master audio onto the silent video; FAIL CLOSED.

    Pure function (used by the node + tests). Steps: validate inputs -> duration
    assert (video must NOT exceed audio by > the credits-tail budget) BEFORE the
    mux -> ffmpeg ``-map 0:v -map 1:a -c:v copy -c:a copy`` (NO ``-shortest``) ->
    assert the output audio decodes identically to the master. Returns
    ``(out_path, report_lines)``; raises ``ValueError`` on any gate failure
    (never produces a silently-wrong episode).

    The gate permits ``a_dur > v_dur``: the master audio includes
    opening/closing themes that play over black frames before/after the drama
    clips; those seconds are not represented in the silent composite.

    Credits-aware tail (credits enrichment 2026-07-03, silent-tail model):
    when ``OTR_CreditsRoll`` appends a silent credits roll to the video tail it
    DECLARES that roll's duration here via ``declared_credits_tail_s``. The
    guard then permits ``v_dur <= a_dur + declared_credits_tail_s + tol`` -- the
    intentional silent credits segment is expected, while anything BEYOND the
    declared tail is still caught (a real frame-budget bug). When no roll is
    declared (0), the legacy ``OTR_MAX_CREDITS_TAIL_S`` env ceiling applies.
    The guard is never blind-widened past what the roll declares.
    """
    report: list = []
    fb = _ffmpeg_bin(ffmpeg)
    if not fb:
        raise ValueError(f"OTR_MasterAudioMux: ffmpeg not found ({ffmpeg!r})")
    if not os.path.isfile(silent_video_path):
        raise ValueError(f"OTR_MasterAudioMux: silent video missing: {silent_video_path!r}")
    if not os.path.isfile(master_audio_path):
        raise ValueError(f"OTR_MasterAudioMux: master audio missing: {master_audio_path!r}")

    # Duration gate: the silent composite covers drama beats only; the master
    # audio also includes opening/closing themes (typically 10s + 8s).  It is
    # therefore EXPECTED that a_dur > v_dur -- the theme audio plays while a
    # black frame holds at start/end.  We only refuse to mux when the VIDEO is
    # LONGER than the audio, which would cause the tail of the video to play
    # silently (a genuine error).  Audio-longer-than-video is intentional and
    # safe: ffmpeg copies both streams with -c copy; the container duration
    # equals max(v_dur, a_dur) and the audio plays out in full.
    v_dur = _probe_float(silent_video_path, "v:0")
    a_dur = _probe_float(master_audio_path, "a:0")
    tol = float(duration_tol_frames) / float(fps or 25)
    # BUG-LOCAL-410 / credits enrichment 2026-07-03: the credits roll
    # legitimately runs the VIDEO past the master audio -- OTR_CreditsRoll
    # appends a SILENT scrolling-credits tail AFTER the body, and it plays in
    # silence. The guard is CREDITS-AWARE: when the roll declares its duration
    # (declared_credits_tail_s > 0) that IS the budget; otherwise the legacy
    # OTR_MAX_CREDITS_TAIL_S env ceiling applies. Either way we still FAIL LOUD
    # on gross drift BEYOND the declared/allowed tail (a real frame-budget bug
    # that doubles the length) -- never blind-widened. The audio stays
    # byte-identical (-c:a copy of the master: the output audio STREAM is
    # unchanged, only the container is longer; the SHA check below still proves
    # it). The CreditsRoll/composite frame budgets are the primary correctness
    # guards; this bound is the final sanity ceiling.
    declared = float(declared_credits_tail_s or 0.0)
    env_ceiling = _credits_tail_ceiling()      # NAMED-and-ignored if malformed
    max_tail_s = declared if declared > 0 else env_ceiling
    tail_src = "declared" if declared > 0 else "env_ceiling"
    if v_dur >= 0 and a_dur >= 0 and v_dur > a_dur + max_tail_s + tol:
        # Print the EXCESS and the OVERAGE at full precision. The old message
        # rendered the budget as "%.1f" -- so a declared tail of 75.1800s printed
        # as "75.2s" next to an excess of 75.2293s, and the failure looked like it
        # violated a budget it was under. Never round the number the reader is
        # being asked to compare against.
        _excess = v_dur - a_dur
        # OPERATOR DIRECTIVE 2026-08-30: "don't kill a duration mismatch, just
        # let it fly." THIS USED TO RAISE, and raising here is the most
        # expensive refusal in the pipeline: by the time this runs the writer,
        # the voices, the music, every video beat and the full audio master are
        # already rendered. Refusing to mux discards a finished episode over a
        # length disagreement -- a consistency judgement, not a resource limit,
        # which is exactly the class the operator has ruled must never kill a
        # render. An OOM is the only killer.
        #
        # Observed cost: a complete scifi_news_pro leg died here at the last
        # step because the silent video ran 41.99s past the master audio, 18.87s
        # beyond the credits-tail budget. Every frame of it was already on disk.
        #
        # The number is still WORTH KNOWING -- a gross overshoot usually IS a
        # frame-budget bug upstream (PBUG-20260829-16's music-beat duration is a
        # live example) -- so it is logged loudly, at full precision, with the
        # same wording the exception carried. What changes is that it no longer
        # throws the episode away to tell you.
        log.warning(
            "[OTR_MasterAudioMux] DURATION MISMATCH (publishing anyway): silent "
            "video %.4fs exceeds master audio %.4fs by %.4fs, over the "
            "credits-tail budget (%.4fs [%s] + %.4fs tol = %.4fs) by %.4fs -- "
            "usually a composite/credits frame-budget bug upstream, not the "
            "intended silent credits tail. The muxed episode will carry %.4fs "
            "of video past the end of its audio.",
            v_dur, a_dur, _excess, max_tail_s, tail_src, tol,
            max_tail_s + tol, _excess - max_tail_s - tol, _excess,
        )
    if v_dur < 0 or a_dur < 0:
        log.warning(
            "[OTR_MasterAudioMux] duration gate SKIPPED: probe returned "
            "v=%.3f a=%.3f. The video-longer-than-audio ceiling was NOT "
            "checked for this episode.", v_dur, a_dur)
    report.append(
        duration_receipt_line(v_dur, a_dur, max_tail_s, tail_src, tol))

    _poll_interrupt()
    # UNCONDITIONAL master copy (rip-sfx bed, 2026-08-06): the SFX mix branch
    # is retired; every episode takes the passthrough that already ran on every
    # shipped episode.
    report.append("audio_mode=master_copy")
    # mux-LAST: copy both streams, NO -shortest.
    cmd = [
        fb, "-y", "-loglevel", "error",
        "-i", silent_video_path,
        "-i", master_audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "copy",
        out_path,
    ]
    assert "-shortest" not in cmd, "V-2: -shortest must never appear in the mux"
    # V-1: the audio stream is COPIED, never re-encoded. The rip kept the
    # passthrough branch; this assertion is what notices if a later edit swaps
    # it for a re-encode (no behavioural test can tell the two apart).
    assert cmd[cmd.index("-c:a") + 1] == "copy", (
        "V-1: the terminal mux must pass the master audio through with "
        "-c:a copy, never re-encode")
    p = _run(cmd)
    if p.returncode != 0:
        raise ValueError(f"OTR_MasterAudioMux: ffmpeg mux failed :: {p.stderr.strip()[:300]}")

    # byte-identity: the output audio must decode identically to the master.
    # Pass the ALREADY-RESOLVED binary so the proof cannot resolve differently
    # from the encode that just ran.
    h_master = audio_pcm_sha(master_audio_path, fb)
    h_out = audio_pcm_sha(out_path, fb)
    if not h_out or h_out != h_master:
        raise ValueError(
            f"OTR_MasterAudioMux: output audio NOT byte-identical to master "
            f"(master={h_master[:12]} out={h_out[:12]}); the audio was re-encoded "
            f"or trimmed -- C7/V-1 violated"
        )
    report.append(f"audio_byte_identical OK ({h_out[:12]})")
    return out_path, report


def _quiet_file_sha256(path: str) -> str:
    """SHA-256 of a file, or ``""`` if it cannot be read.

    For cache keys only, and the empty string is a SAFE answer there: an
    unreadable file makes the key differ from the next run's, which re-executes
    the node. Never used to prove an identity -- that is
    :func:`mux_master_audio`'s job and it raises."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, ValueError):
        return ""


def _foley_receipt_digest(receipts_json: str) -> str:
    """A cache digest over the foley stems a manifest names.

    Hashes each row's stem SHA **and the bytes on disk**, because those are two
    different facts: the manifest can carry a stale SHA for a file that has been
    rewritten, and it is the FILE the mix reads. Best-effort -- see
    :func:`_quiet_file_sha256` for why an unreadable input is safe here.
    """
    import json

    if not receipts_json:
        return ""
    try:
        rows = (json.loads(receipts_json) or {}).get("clips") or []
    except (TypeError, ValueError):
        return ""
    parts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = str(row.get("foley_path") or "")
        if not path:
            continue
        parts.append("%s:%s:%s" % (path, row.get("foley_sha256") or "",
                                   _quiet_file_sha256(path)))
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest() \
        if parts else ""


def _master_wav_owes_a_delivery_gain(master_audio_path: str) -> bool:
    """Has the master WAV still to receive its delivery loudness pass?

    Read off the ledger stamp ``OTR_EpisodeAssembler`` writes on every episode
    -- NOT inferred from the policy. The stamp is a fact about the bytes on
    disk; the policy is an inference about that fact, and this node runs long
    after the fact is settled. When the two disagree, the fact wins.

    Absent stamp -> False, which is the historical behaviour: every master
    written by a build older than the foley bed had its loudness applied
    upstream. Best-effort throughout -- a receipt read must never block a
    finished episode.
    """
    try:
        from . import _otr_ledger as _OTRL
        from .scene_sequencer import MASTER_WAV_PRE_LOUDNESS
    except ImportError:  # pragma: no cover -- flat (sys.path) test import
        try:
            import _otr_ledger as _OTRL  # type: ignore
            from scene_sequencer import MASTER_WAV_PRE_LOUDNESS  # type: ignore
        except ImportError:
            return False
    try:
        ledger_path = _OTRL.in_flight_ledger_path()
        if ledger_path is None:
            return False
        ledger = _OTRL.load_ledger_safe(ledger_path) or {}
        flavour = str((ledger.get("audio") or {}).get("master_wav_flavour")
                      or "")
    except Exception as exc:  # noqa: BLE001 -- a receipt read never blocks
        log.warning("[OTR_MasterAudioMux] could not read the master WAV "
                    "flavour stamp (%s); assuming it is already levelled", exc)
        return False
    return flavour == MASTER_WAV_PRE_LOUDNESS


def _foley_route(video_policy_json: str) -> bool:
    """Is any role on this episode rendering with the LTX 2.5 foley lane?

    Delegated to ``foley_stems.is_foley_route`` -- the SAME function
    ``OTR_EpisodeAssembler`` used to decide whether to write a provisional
    master -- so on a correctly wired graph the two nodes read one source and
    reach one answer.

    THIS ALONE IS NOT THE GUARANTEE, and it would be an over-claim to say it
    is: sharing a function makes the two nodes agree only while both are
    actually HANDED the policy. What closes the gap is the ledger stamp --
    see :func:`_master_wav_owes_a_delivery_gain`, whose answer is OR-ed with
    this one at the call site.

    Fail-soft to False: an unreadable policy means today's copy path, which is
    what every non-foley episode already does.
    """
    if not video_policy_json:
        return False
    try:
        from ._otr_video_engines import foley_stems as _fs
    except ImportError:  # pragma: no cover -- flat (sys.path) test import
        try:
            from _otr_video_engines import foley_stems as _fs  # type: ignore
        except ImportError:
            log.warning("[OTR_MasterAudioMux] foley_stems is unavailable; "
                        "this episode is muxed on the copy path")
            return False
    return bool(_fs.is_foley_route(video_policy_json))


def _compile_foley_master(master_audio_path: str, receipts_json: str,
                          fps: int, video_policy_json: str = ""):
    """Mix the foley bed under the provisional master and LEVEL the result.

    Returns ``(path_to_mixed_wav, report_lines)``.

    THIS IS THE ONLY DELIVERY LOUDNESS PASS ON A FOLEY EPISODE, and it runs
    even when the bed turns out to be entirely silent. ``OTR_EpisodeAssembler``
    deliberately wrote its WAV PRE-loudness on this route so the ratio could be
    set once here and one gain follow it; a route that declined to level would
    ship the un-levelled provisional master as the deliverable. So: mix, then
    ``_master_loudness`` exactly once, then freeze.

    ``_master_loudness`` is REUSED rather than reimplemented -- it is the only
    loudness algorithm in this repo, it is the one that produced the measured
    -14 LUFS target, and a second implementation here would be a second
    delivery level that drifts. Imported lazily: this module is stdlib-only at
    the top and must stay cold-import clean.

    A NEW FILE, NEVER AN OVERWRITE. ``<ep>_master.wav`` remains exactly what the
    assembler wrote, so a failed mix leaves a re-runnable episode behind rather
    than a half-mixed master with the original gone.
    """
    import json

    try:
        from ._otr_video_engines import foley_stems as _fs
        from .scene_sequencer import _master_loudness
    except ImportError:  # pragma: no cover -- flat (sys.path) test import
        from _otr_video_engines import foley_stems as _fs  # type: ignore
        from scene_sequencer import _master_loudness  # type: ignore
    import torch

    rows = []
    if receipts_json:
        try:
            rows = [r for r in ((json.loads(receipts_json) or {}).get("clips")
                                or []) if isinstance(r, dict)]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "this episode is on a foley route but its foley receipts are "
                "not readable JSON (%s). NO SILENT COPY -- shipping the "
                "un-levelled provisional master would be a quiet 20%% "
                "loudness error nobody sees until playback" % exc)
    bearing = [r for r in rows if r.get("foley_path")]

    # RE-RAISED IN THIS NODE'S VOCABULARY. FoleyStemError is a RuntimeError,
    # and mux()'s terminal handler catches (ValueError, OSError) -- so an
    # unwrapped stem fault would skip the named "the render FAILED" path and
    # surface as a bare traceback with no episode context attached.
    try:
        master, master_rate = _fs.read_pcm16_wav(master_audio_path)
        mixed, stats = _fs.mix_foley_under_master(
            master, master_rate, bearing, fps=int(fps),
            lane_ids=_fs.route_lane_ids(video_policy_json))
    except _fs.FoleyStemError as exc:
        raise ValueError(
            "the foley bed could not be mixed under %s: %s"
            % (os.path.basename(master_audio_path), exc)) from exc

    # (1, channels, n) is the shape _master_loudness is written against, and
    # its legacy fallback path uses torch.tanh -- so hand it a real tensor
    # rather than an array that only works down the LUFS branch.
    levelled, loud = _master_loudness(
        torch.from_numpy(mixed).unsqueeze(0), ceiling_dbfs=-1.0,
        sample_rate=int(master_rate))
    levelled = levelled.squeeze(0).detach().cpu().numpy()

    root, _ext = os.path.splitext(master_audio_path)
    mixed_path = root + "_foley.wav"
    _fs.write_pcm16_wav(mixed_path, levelled, master_rate)

    peak_dbfs = 20.0 * math.log10(max(float(abs(levelled).max()), 1e-9))
    lanes = stats["lanes"]
    report = [
        "foley_bed=mixed beats=%d/%d lanes=%s master_gain=%.2f out=%s"
        % (stats["placed"], len(bearing),
           ",".join("%s:%d" % kv for kv in lanes.items()) or "none",
           stats["global_master_gain"], os.path.basename(mixed_path)),
        "foley_loudness=%s measured=%s -> target=%s gain_db=%s peak_dbfs=%.2f"
        % (loud.get("mode"), loud.get("measured_lufs"),
           loud.get("target_lufs"), loud.get("gain_db"), peak_dbfs),
    ]
    if stats["muted_samples"]:
        # MIME MUTES REAL AUDIO ON PURPOSE, and a receipt that did not say how
        # much would make an accidental mute indistinguishable from the
        # intended one. Seconds, because samples mean nothing to a reader.
        report.append("foley_muted_s=%.2f (mime windows; the TTS and cues "
                      "there were generated and discarded)"
                      % (stats["muted_samples"] / float(master_rate)))
    if stats["skipped"]:
        report.append("foley_skipped=%d (outside the master)"
                      % stats["skipped"])
    # BEATS THAT MADE A STEM BUT OWN NO WINDOW IN THE MASTER. A music_inter
    # bridge is the normal case -- it renders a picture and occupies no
    # master-mix time at all. Reported because the same count is a FAULT for
    # any other role, and a bed quietly missing from half an episode must not
    # be invisible.
    #
    # UNCONDITIONAL, INCLUDING AT ZERO (2026-08-30). This used to be gated on a
    # non-zero count, which makes the number unreadable: the 4060 went looking
    # for it to confirm the sentinel fix on 8 GB hardware and could only report
    # ABSENT, which is not the same claim as ZERO and does not close the
    # question it was asked to close. A counter that disappears at its most
    # common value cannot be trusted at any value -- "I did not see it" and "it
    # was fine" must not share a representation. The neighbouring counters stay
    # conditional on purpose: they are exceptions worth noticing, while this one
    # is a standing invariant somebody will come to verify.
    report.append("foley_unpositioned=%d (no master-mix slot; normal for "
                  "music_inter bridges)" % stats["unpositioned"])
    if stats["conform_notes"]:
        report.append("foley_conformed=" + "; ".join(stats["conform_notes"]))
    log.info("[OTR_MasterAudioMux] %s", " | ".join(report))
    return mixed_path, report


def _reresolve_master_audio(master_audio_path: str) -> str:
    """Rename-proof the master-audio path WITHOUT changing the audio source.

    Upstream nodes capture ``master_audio_path`` while the episode dir is still
    ``pending_<ts>``; the ledger then renames that dir to its final slug. The
    captured absolute path becomes stale (its ``pending_<ts>`` directory no
    longer exists) even though the FILE moved into the renamed dir keeping the
    SAME basename. Re-resolve to that same file via the active in-flight ledger
    (the same durable-ledger contract OTR_ShotLock uses for audio timing), never
    a newest-mtime sibling guess.

    Returns the original path unchanged when it already exists, when disk state
    is disabled (``OTR_TEST_MODE``), or when no exact-basename match is found --
    in which case the caller fails closed. It NEVER points at a different audio
    source: only the byte-for-byte same basename under the renamed episode
    ``audio`` dir is accepted, and ``mux_master_audio`` still asserts the output
    is PCM-byte-identical to it.
    """
    if not master_audio_path or os.path.isfile(master_audio_path):
        return master_audio_path
    if otr_env.get("OTR_TEST_MODE") == "1":
        return master_audio_path
    want = os.path.basename(master_audio_path)
    try:
        from pathlib import Path
        from . import _otr_ledger as _OL
        p = _OL.in_flight_ledger_path()
        if not p:
            return master_audio_path
        p = Path(p)
        disk = _OL.load_ledger_safe(p)
        episode_dir = p.parent.parent
        if (
            not isinstance(disk, dict)
            or str(disk.get("episode_id") or "").strip()
                != episode_dir.name
        ):
            log.warning(
                "[OTR_MasterAudioMux] master audio re-resolve REJECTED: "
                "active ledger identity does not match its episode directory"
            )
            return master_audio_path
        cand = Path(p).parent / want          # <episode>/audio/<same-basename>
        if cand.is_file():
            log.warning(
                "[OTR_MasterAudioMux] LOUD re-resolve: master audio path stale "
                "(episode dir renamed after capture); %r -> %r "
                "(same file, post-rename dir)",
                master_audio_path, str(cand),
            )
            return str(cand)
    except Exception as exc:  # noqa: BLE001 - never mask the fail-closed path
        log.warning("[OTR_MasterAudioMux] master audio re-resolve skipped: %s", exc)
    return master_audio_path


def _episodes_root() -> Path:
    """The episode workspace root -- the pack's ONE owner's answer
    (``_otr_paths.otr_episodes_root``). This read ``folder_paths`` itself, so
    on a server launched with ``--output-directory`` the mux looked under a
    tree the stills, audio and ledger had already left (kibitz
    runpod-found-fixes, 2026-09-04). Kept under this name because the
    publication tests monkeypatch it."""
    try:
        from ._otr_paths import otr_episodes_root
    except ImportError:  # pragma: no cover -- flat (sys.path) test import
        from _otr_paths import otr_episodes_root  # type: ignore
    return otr_episodes_root()


def _episode_stem(silent_video_path: str) -> str:
    return os.path.splitext(os.path.basename(silent_video_path or "episode"))[0]


def _obs_dir() -> Path:
    """The operator-facing OBS folder -- the pack's ONE owner's answer
    (``_otr_paths.otr_obs_dir``): ``OTR_OBS_DIR`` when pinned (the headless
    launcher renders into one tree while the operator watches another --
    the two-tree split, 2026-06-09), else ``<output>/otr/obs``. Pure:
    resolves a path and creates nothing. Kept under this name because the
    publication tests monkeypatch it; until 2026-09-04 it was the second
    owner of this answer."""
    try:
        from ._otr_paths import otr_obs_dir
    except ImportError:  # pragma: no cover -- flat (sys.path) test import
        from _otr_paths import otr_obs_dir  # type: ignore
    return otr_obs_dir()


def _is_inside_obs_dir(path: str) -> bool:
    """Would writing ``path`` put a file in the operator's watch folder?

    Used to refuse an unpublishable episode a back door into obs via the
    operator's own ``output_path``. Compares RESOLVED paths so ``..`` segments,
    case differences and short/long Windows forms cannot walk around it. Never
    raises -- an unresolvable path is not inside obs by any reading, and a
    crash here would take a finished episode down over a string comparison.
    """
    try:
        target = Path(path).resolve()
        obs = _obs_dir().resolve()
        return obs == target or obs in target.parents
    except Exception:  # noqa: BLE001
        return False


def _stem_belongs_to_episode(stem: str, episode_id: str) -> bool:
    """Does this video stem belong to THIS episode, on a name boundary?

    A BARE PREFIX TEST IS NOT AN IDENTITY TEST. `stem.startswith(episode_id)`
    says yes for episode `ep1` against episode `ep10`'s video, because `ep10`
    starts with `ep1`. This check came from `_default_out`, where the cost was a
    file in the wrong folder; it now decides which ledger answers for an
    episode, so the same slip would let one episode PUBLISH under another's
    rights receipt -- the exact stale-singleton confusion the check exists to
    prevent.

    The chain appends suffixes with a leading underscore (`_silent`,
    `_procgen_blended`, `_captioned`, `_with_credits`), so the episode id is
    either the whole stem or is followed by one. Anything else is a different
    episode that happens to share an opening.
    """
    if not stem or not episode_id:
        return False
    return stem == episode_id or stem.startswith(episode_id + "_")


def _inflight_episode_for_stem(stem: str) -> "tuple[Path | None, Path | None]":
    """The in-flight ledger and episode dir, but ONLY if they are THIS episode.

    ONE PIECE OF VALIDATION, THREE CONSUMERS. The output path, the publication
    decision and the cache key all need the same answer to the same question --
    "which episode is this node actually finishing?" -- and three copies of that
    check is three chances for one of them to drift and answer for a different
    episode.

    The in-flight ledger is the path authority: filename suffix peeling is
    necessarily incomplete whenever a new terminal enrichment lands (captions
    exposed this -- ``_captioned`` created a fake sibling episode directory).
    But the singleton is accepted only when it is a direct child of the
    configured episodes root AND the incoming video stem starts with that
    episode id. The prefix check is what rejects a STALE singleton left over
    from a prior episode -- the difference between reading this episode's
    receipt and gating this episode on somebody else's.

    Returns ``(ledger_path, episode_dir)``, or ``(None, None)`` when no
    singleton answers for this stem. Never raises.
    """
    try:
        try:
            from . import _otr_ledger as _OTRL
        except ImportError:  # pragma: no cover -- direct-script fallback
            import _otr_ledger as _OTRL  # type: ignore
        ledger_path = _OTRL.in_flight_ledger_path()
        if ledger_path is None:
            return None, None
        candidate = Path(ledger_path).resolve().parent.parent
        expected_root = _episodes_root().resolve()
        if (candidate.parent == expected_root
                and candidate.name
                and not candidate.name.startswith("_")
                and _stem_belongs_to_episode(stem, candidate.name)):
            return Path(ledger_path), candidate
    except Exception as exc:  # noqa: BLE001 -- callers all have a safe path
        log.info(
            "[OTR_MasterAudioMux] in-flight episode path unavailable: %s", exc,
        )
    return None, None


def _publication_decision(silent_video_path: str):
    """May this episode be copied to the operator's OBS folder?

    Consumes the ONE durable receipt stamped at the freeze
    (``_otr_publication_eligibility``). This node decides nothing about rights
    itself -- it reads a verdict, which is the whole point of the receipt.

    FAILS CLOSED, ON PURPOSE. No ledger, no receipt, a malformed receipt, an
    unreadable version, or a receipt stamped for a DIFFERENT episode all come
    back not-publishable. Publishing is the irreversible half of this node (the
    file lands in the folder the operator watches, and on a research-only
    source that is the exact thing the rule forbids), while withholding is
    recoverable -- the archival final is still on disk and a re-freeze
    republishes it. Between a guess and a receipt, take the receipt.
    """
    try:
        try:
            from . import _otr_ledger as _OTRL
            from . import _otr_publication_eligibility as _PE
        except ImportError:  # pragma: no cover -- direct-script fallback
            import _otr_ledger as _OTRL  # type: ignore
            import _otr_publication_eligibility as _PE  # type: ignore
        ledger_path, _ = _inflight_episode_for_stem(
            _episode_stem(silent_video_path))
        if ledger_path is None:
            return _PE.PublicationDecision(
                publishable=False,
                reason=_PE.DECISION_NO_RECEIPT,
                detail="no in-flight ledger answers for this episode",
            )
        led = _OTRL.load_ledger_safe(ledger_path)
        if led is None:
            return _PE.PublicationDecision(
                publishable=False,
                reason=_PE.DECISION_NO_RECEIPT,
                detail="could not load %s" % ledger_path.name,
            )
        return _PE.decide_from_meta(
            led.get("meta"),
            expected_episode_id=str(led.get("episode_id") or ""),
        )
    except Exception as exc:  # noqa: BLE001 -- fail closed, never crash the mux
        try:
            from . import _otr_publication_eligibility as _PE  # type: ignore
        except ImportError:  # pragma: no cover
            import _otr_publication_eligibility as _PE  # type: ignore
        return _PE.PublicationDecision(
            publishable=False,
            reason=_PE.DECISION_MALFORMED,
            detail="%s: %s" % (type(exc).__name__, exc),
        )


#: Pipeline-stage suffixes the archival stem accumulates on its way through the
#: graph. They are meaningful in `otr/episodes/` and pure noise in `otr/obs/`.
_PIPELINE_SUFFIXES = ("_silent_procgen_blended_captioned_with_credits",
                      "_procgen_blended_captioned_with_credits",
                      "_captioned_with_credits", "_procgen_blended",
                      "_captioned", "_blend", "_silent")

#: Cap so a long title plus six fields cannot approach the Windows path limit.
_OBS_NAME_MAX = 150


def _obs_field(text, fallback="none"):
    """One filename-safe field: lowercase, no separators of our own."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip()).strip("-.")
    return (s or fallback).lower()


def _obs_basename(final: str) -> str:
    """The OPERATOR-FACING filename for the published episode.

    THE ARCHIVAL NAME IS NOT THE WATCHING NAME (operator, 2026-09-03). The obs
    copy used to inherit the archival stem verbatim, so every published episode
    read:

        signal_lost_<title>_<ts>_silent_procgen_blended_captioned_with_credits_final.mp4

    -- and in a file browser every row truncated at the identical point,
    `..._silent_procgen_blended_captioned_wit...`, telling the operator nothing
    about what actually made the episode. Worse, that tail is actively
    misleading: `procgen` is a COMPOSITING stage, and this session read it as
    the render engine and built a whole wrong diagnosis on it before the
    ledgers corrected it.

    So the obs copy is renamed to carry what a viewer is comparing -- the
    episode, then the five choices that produced it:

        <title>_<ts>__<style>__<video>__<image>__<tts>__<bank>_final.mp4
        arms_at_the_ready_20260903_092133__cartoon__wan_ti2v__z_image_turbo__
            indextts2__public_domain_final.mp4

    Episode first (operator's pick) so the folder still sorts by episode and
    keeps its identity; style and video next because those are the axes he
    actually compares, and they survive the truncation on ordinary titles.

    `_final` IS PRESERVED: `scripts/otr_pod_obs_bridge.py` keys on that marker
    to recognise a published episode, and the archival copy in `otr/episodes/`
    is deliberately UNTOUCHED -- its suffixes carry pipeline provenance and
    `otr_caption_burn` strips those exact spellings.

    Fails soft to the archival basename: a publish must never die over a name.
    """
    base = os.path.basename(final)
    try:
        stem, ext = os.path.splitext(base)
        for suf in ("_final",):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        for suf in _PIPELINE_SUFFIXES:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break

        try:
            from . import _otr_ledger as _OL
        except ImportError:  # pragma: no cover -- flat test imports
            import _otr_ledger as _OL  # type: ignore
        path = _OL.in_flight_ledger_path()
        led = (_OL.load_ledger_safe(path) or {}) if path else {}
        meta = led.get("meta") or {}
        video = led.get("video") or {}

        vid = collections.Counter(
            s.get("engine_id") for s in (video.get("shots") or [])
            if s.get("engine_id"))
        img = collections.Counter()
        for _role, per in ((meta.get("image_engines") or {}).get("by_role")
                           or {}).items():
            for eng, n in (per or {}).items():
                img[eng] += int(n or 0)

        def _trim_engine(name):
            # `animatediff15_v3_haunted_video` -> `animatediff15_v3_haunted`;
            # the role is already implied by the field position.
            return re.sub(r"_(video|image)$", "", str(name or ""))

        # The show prefix is constant across every episode, so it buys nothing
        # in a folder of them; the title and timestamp are the identity. The
        # prefix is spelled ONCE, in the ledger, because the ledger's
        # `_published_obs_path` must accept the name written here
        # (PBUG-20260904-06: it demanded the prefix this line strips).
        title = re.sub("^" + re.escape(_SHOW_PREFIX), "", stem)
        fields = [
            _obs_field(meta.get("visual_style"), "nostyle"),
            _obs_field(_trim_engine(vid.most_common(1)[0][0]) if vid else None),
            _obs_field(_trim_engine(img.most_common(1)[0][0]) if img else None),
            _obs_field(meta.get("char_voice_engine"), "novoice"),
            _obs_field(meta.get("source_bank"), "nobank"),
        ]
        name = "%s__%s_final%s" % (_obs_field(title, "episode"),
                                   "__".join(fields), ext or ".mp4")
        if len(name) > _OBS_NAME_MAX:
            keep = _OBS_NAME_MAX - (len(name) - len(title))
            name = "%s__%s_final%s" % (_obs_field(title[:max(16, keep)]),
                                       "__".join(fields), ext or ".mp4")
        return name
    except Exception as exc:  # noqa: BLE001 -- a publish never dies over a name.
        # BUT IT SAYS SO. The first cut of this helper referenced `re` without
        # importing it, and this except swallowed the NameError -- every publish
        # "worked" while silently reverting to the old confusing name, with no
        # trace anywhere. A test now covers that specific bug; this log covers
        # the NEXT one, because every other fallback in this file already logs
        # when it degrades and this one was the exception.
        log.warning("[OTR_MasterAudioMux] descriptive obs name failed (%s: %s); "
                    "published under the archival name %s",
                    type(exc).__name__, exc, base)
        return base


class OTRMasterAudioMux:
    """Registered as ``OTR_MasterAudioMux``. Terminal audio mux (V-1: the ONLY
    node that adds audio). ``-c:a copy``, NO ``-shortest``, byte-identical assert."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "mux"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_video_path", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "silent_video_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Always-silent composite (OTR_SilentComposite). Audio is added HERE only.",
                }),
                "master_audio_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Frozen master mix from OTR_EpisodeAssembler.output_path (the master WAV). Copied with -c:a copy.",
                }),
            },
            "optional": {
                "audio_done": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Audio-done gate (EpisodeAssembler out3). Orders the mux AFTER the audio freezes. Opaque.",
                }),
                "declared_credits_tail_s": ("FLOAT", {
                    "default": 0.0, "forceInput": True,
                    "tooltip": "OTR_CreditsRoll's declared silent-credits tail "
                               "duration. Makes the tail guard credits-aware "
                               "(v <= a + declared + tol); 0 -> the "
                               "OTR_MAX_CREDITS_TAIL_S env ceiling.",
                }),
                "clip_manifest_json": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Retired connector kept for topology compatibility (rip-sfx 2026-08-06). The manifest is hashed by IS_CHANGED but has no effect on the output.",
                }),
                "fps": ("INT", {
                    "default": 25, "min": 1, "max": 120,
                    "tooltip": "Frame rate declared to the mux for timing "
                               "math. Must match the silent video's real rate "
                               "(the composite's manifest fps); the mux never "
                               "resamples frames.",
                }),
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "tooltip": "DEPRECATED and IGNORED (2026-09-04). A workflow value cannot name the binary this pack runs -- it arrives over an unauthenticated /prompt request. Set the OTR_FFMPEG environment variable to pin a build.",
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Final mp4 path. Empty -> <output>/otr/episodes/<stem>_final.mp4.",
                }),
                # THE FOLEY BED'S TWO CONNECTORS (2026-08-26), APPENDED at the
                # end and never inserted (BUG-LOCAL-097).
                #
                # WHY THESE ARE NEW INPUTS RATHER THAN A USE FOR
                # ``clip_manifest_json`` ABOVE -- the operator decided this, and
                # it is not a style preference. That connector is a deliberate
                # tripwire: `tests/test_rip_sfx_bed_guard.py` asserts it is
                # "accepted, hashed, unused" and its docstring says plainly
                # "never invent a use". Driving the mix off it would satisfy the
                # test's strings while making the test's own name and reasoning
                # false. A dedicated input leaves that tripwire meaning exactly
                # what it says, at the cost of one link on a graph that is being
                # edited anyway.
                #
                # BOTH ARE READ, AND BOTH ARE NEEDED. The policy answers "is
                # this a foley route?" -- the SAME question the assembler asked
                # of the SAME JSON, so the WAV's flavour and this node's
                # mix-or-copy decision cannot diverge. The receipts answer
                # "which beats have a bed, and where does it go?". Deciding the
                # route from the receipts alone would copy an un-levelled
                # provisional master whenever a foley role rendered no beats.
                "video_policy_json": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Video policy JSON from OTR_VideoDirector. Read "
                               "for ONE question: is any role on the LTX 2.5 "
                               "foley lane? If so this node mixes the foley "
                               "bed under the provisional master and performs "
                               "the single delivery loudness pass.",
                }),
                "foley_receipts_json": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Clip manifest carrying the per-beat foley "
                               "receipts (foley_path / foley_sha256 / start_s). "
                               "The bed the mix is built from. Ignored unless "
                               "video_policy_json names a foley route.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _default_out(self, silent_video_path: str) -> str:
        episodes_root = _episodes_root()
        stem = _episode_stem(silent_video_path)
        _, out_dir = _inflight_episode_for_stem(stem)

        # OUTPUT HYGIENE (operator directive 2026-06-09): the final lands in
        # the episode's OWN folder under otr/episodes/<ep>/ (the obs copy is
        # the only file outside it). <ep> = the input stem minus the known
        # post-chain suffixes only when no live ledger authority exists. The
        # chain appends OUTERMOST-LAST, so peel in that order. `_captioned` is
        # between credits and the procgen blend on the canonical graph.
        if out_dir is None:
            ep = stem
            for suffix in (
                "_with_credits", "_captioned", "_procgen_blended", "_silent",
            ):
                if ep.endswith(suffix):
                    ep = ep[: -len(suffix)]
            out_dir = episodes_root / ep
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(str(out_dir), f"{stem}_final.mp4")

    def _publish_to_obs(self, final: str) -> str:
        """OUTPUT HYGIENE (operator directive 2026-06-09): the FINAL playable
        episode mp4 is the deliverable and must land in ``<output>/otr/obs``
        (the folder the OPERATOR watches), not only ``otr/episodes``. Publish a
        copy there LOUDLY; failure to publish is a real error (the deliverable
        gate), not a warning.

        ``OTR_OBS_DIR`` pins the operator-facing obs dir explicitly -- on this
        box the headless server renders into the ComfyUI-Installs tree while
        the operator watches ``Documents\\ComfyUI\\output\\otr\\obs``, so the
        launch recipe sets it (two-tree split, 2026-06-09 operator report)."""
        obs_dir = str(_obs_dir())
        os.makedirs(obs_dir, exist_ok=True)
        dst = os.path.join(obs_dir, _obs_basename(final))
        # PLAYABILITY (operator screenshot 2026-06-09): -c:a copy from the WAV
        # master leaves raw PCM ("ipcm") in the MP4 -- byte-identical but
        # unplayable in standard players (Windows Media Player refuses the
        # audio). The obs deliverable is the WATCHABLE copy: video stream
        # copied untouched, audio encoded AAC-320k. The ARCHIVAL byte-identical
        # PCM final stays in otr/episodes/<ep>/ (mux gate already asserted it
        # against the frozen master; the master itself is never touched).
        fb = _ffmpeg_bin("ffmpeg")
        if not fb:
            # The mux itself refused earlier on the same answer; keep the
            # empty-string contract here too instead of running a literal
            # that fails inside subprocess with a less useful name.
            raise OSError("obs publish: ffmpeg not found (OTR_FFMPEG / PATH)")
        p = _run([fb, "-y", "-loglevel", "error", "-i", final,
                  "-map", "0:v", "-map", "0:a",
                  "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                  "-ar", "48000", dst])
        if p.returncode != 0:
            raise OSError("obs publish (aac viewing copy) failed: %s"
                          % p.stderr.strip()[:300])
        log.warning("[OTR_MasterAudioMux] LOUD publish: final episode -> %s "
                    "(%d bytes; video copy + AAC-320k viewing audio; archival "
                    "PCM byte-identical final: %s)",
                    dst, os.path.getsize(dst), final)
        return dst

    def _stamp_terminal_paths(
        self,
        final_path: str,
        obs_path: "str | None",
        master_audio_path: str,
    ) -> str:
        """Truthfully stamp all terminal asset pointers in the live ledger.

        The tail chain grew a credits roll (node 95) + this mux (node 85) AFTER
        the procgen blend (node 93), whose ``_stamp_ledger_final_video_path``
        left the ledger pointing at the pre-credits / pre-mux intermediate blend.
        This node is the terminal stage, so it owns the archival video, frozen
        master audio, and published OBS pointers together. ``save_ledger_safe``
        validates the published path and synchronizes ``meta.paths.obs_final``.
        Best-effort -- a stamp failure must NEVER block the deliverable, so it is
        caught and reported, never raised. Returns a single report line.

        ``obs_path`` is ``None`` when publication was WITHHELD. It stamps no OBS
        pointer then and actively clears any stale one, because a path key on a
        ledger reads as "the deliverable is there" to everything downstream --
        and on a blocked episode nothing ever arrives. The archival
        ``final_video_path`` / ``final_audio_path`` are stamped either way: the
        work exists, it simply may not be published.
        """
        try:
            try:
                from . import _otr_ledger as _OTRL  # type: ignore
            except ImportError:  # pragma: no cover -- direct-script fallback
                import sys as _sys
                _here = os.path.dirname(os.path.abspath(__file__))
                if _here not in _sys.path:
                    _sys.path.insert(0, _here)
                import _otr_ledger as _OTRL  # type: ignore
            ledger_p = _OTRL.in_flight_ledger_path()
            if ledger_p is None:
                return "terminal path stamp skipped: no in-flight ledger singleton"
            led = _OTRL.load_ledger_safe(ledger_p)
            if led is None:
                return f"terminal path stamp skipped: could not load {ledger_p.name}"
            led["final_audio_path"] = str(master_audio_path)
            led["final_video_path"] = str(final_path)
            meta = led.setdefault("meta", {})
            if obs_path is None:
                meta.pop("obs_final_path", None)
                paths = meta.get("paths")
                if isinstance(paths, dict):
                    paths.pop("obs_final", None)
                    paths.pop("obs_dir", None)
            else:
                meta["obs_final_path"] = str(obs_path)
            if not _OTRL.save_ledger_safe(ledger_p, led):
                return f"terminal path stamp failed: save returned False for {ledger_p.name}"
            obs_note = (
                "no obs_final_path (publication withheld)" if obs_path is None
                else f"obs_final_path -> {os.path.basename(str(obs_path))}"
            )
            return (
                f"stamped ledger {ledger_p.name}: final_audio_path -> "
                f"{os.path.basename(str(master_audio_path))}; final_video_path -> "
                f"{os.path.basename(str(final_path))}; {obs_note}"
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks the mux
            return f"terminal path stamp failed: {type(exc).__name__}: {exc}"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache key: the manifest, the EPISODE, and its publication verdict.

        THE OLD KEY COULD REUSE A PUBLISHED RESULT FOR A BLOCKED EPISODE. It
        hashed ``clip_manifest_json`` alone -- a RETIRED connector that feeds
        nothing -- so two runs with the same manifest looked identical to
        ComfyUI's cache even when they were different episodes with different
        rights. A cached node does not execute, and a mux that does not execute
        cannot withhold anything: the receipt would say blocked and the earlier
        run's published output would stand in for it.

        Episode identity closes the cross-episode half; the eligibility digest
        closes the same-episode half, so re-freezing an episode after its rights
        record changes re-runs the mux rather than serving the old verdict.

        THE DIGEST IS READ FROM DISK, AND THAT IS SAFE IN THE ONE DIRECTION
        THAT MATTERS. If the ledger is not yet written when ComfyUI computes
        the key, the digest is empty and the key simply differs from the next
        run's -- which re-executes the node. The failure direction is "run
        again", never "serve a cached publish", so a missing receipt can cost a
        remux and can never leak an unpublishable episode.

        Returns a SHA-256 hex string, not ``hash()``. Python salts string
        hashing per interpreter, so the previous key silently changed at every
        server boot -- harmless while nothing depended on it, wrong the moment
        it gates a deliverable.
        """
        manifest = str(kwargs.get("clip_manifest_json") or "")
        stem = _episode_stem(str(kwargs.get("silent_video_path") or ""))
        # THE FOLEY BED IS PART OF THE OUTPUT, SO IT IS PART OF THE KEY. Two
        # runs whose manifests match STRING-for-string can still owe different
        # masters: a re-render writes new bytes to the same stem paths, and the
        # provisional master WAV is rewritten at a fixed name every episode. A
        # key that hashed only the JSON would serve the first run's mix for the
        # second run's audio -- a cached node does not execute, and a mux that
        # does not execute cannot re-mix.
        #
        # Both digests are best-effort by design: a file not yet on disk hashes
        # to "", which makes the key DIFFER from the next run's and re-executes.
        # The failure direction is "mux again", never "serve a stale mix".
        foley_key = "\x1f".join([
            str(kwargs.get("video_policy_json") or ""),
            _foley_receipt_digest(str(kwargs.get("foley_receipts_json") or "")),
            _quiet_file_sha256(str(kwargs.get("master_audio_path") or "")),
        ])
        _, episode_dir = _inflight_episode_for_stem(stem)
        episode_id = episode_dir.name if episode_dir is not None else ""
        decision = _publication_decision(str(kwargs.get("silent_video_path") or ""))
        # The REASON rides in the key as well as the digest: a blocked decision
        # has no digest (there is no receipt to hash), and two different
        # blocking reasons must not collide into one cache entry.
        parts = (manifest, stem, episode_id, decision.reason,
                 "1" if decision.publishable else "0", decision.digest,
                 foley_key)
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def mux(self, silent_video_path, master_audio_path, audio_done="",
            declared_credits_tail_s=0.0, clip_manifest_json="", fps=25,
            ffmpeg="ffmpeg", output_path="", video_policy_json="",
            foley_receipts_json=""):
        # ``clip_manifest_json`` is a RETIRED connector (rip-sfx 2026-08-06):
        # still wired on the canonical graph and hashed by IS_CHANGED, but it
        # feeds nothing -- the SFX bed compiler it once armed is deleted.
        # B1 (2026-09-04): the widget is UNTRUSTED /prompt input, not
        # operator intent. Discarded HERE, at the node boundary, so no
        # helper underneath can be handed it.
        try:
            from ._otr_shared.ffmpeg import widget_ffmpeg_is_ignored
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_shared.ffmpeg import widget_ffmpeg_is_ignored  # type: ignore
        ffmpeg = widget_ffmpeg_is_ignored(ffmpeg, "OTR_MasterAudioMux")
        master_audio_path = _reresolve_master_audio(master_audio_path)
        # DECIDE BEFORE WRITING. The verdict is a pure read, and knowing it
        # first is what lets the archival write choose a lawful destination --
        # deciding after the file exists would leave the one case below already
        # on disk in the folder the rule forbids.
        decision = _publication_decision(silent_video_path)
        out = output_path.strip() or self._default_out(silent_video_path)
        if not decision.publishable and _is_inside_obs_dir(out):
            # A BLOCKED EPISODE MAY NOT REACH obs BY ANY ROUTE, INCLUDING THE
            # OPERATOR'S OWN output_path. Withholding the published COPY while
            # the archival write lands in the watch folder anyway would satisfy
            # the code and defeat the rule. `output_path` keeps its meaning
            # everywhere else; only this destination is refused, and loudly.
            out = self._default_out(silent_video_path)
            log.warning(
                "[OTR_MasterAudioMux] output_path %r points into the OBS "
                "folder and this episode may not be published (%s); the "
                "archival final goes to %s instead",
                output_path.strip(), decision.summary(), out,
            )
        # THE FOLEY BED, MIXED BEFORE THE COPY AND NEVER INSIDE IT
        # (2026-08-26). ``mux_master_audio`` below is UNCHANGED: it still
        # copies with ``-c:a copy`` and still asserts the muxed audio's PCM
        # SHA against the file it was handed. What changes on a foley route is
        # only WHICH file that is -- a new, durable, fully levelled
        # ``<ep>_master_foley.wav`` rather than the provisional master the
        # assembler wrote. The copy discipline is intact; the identity target
        # moved. ffmpeg never mixes anything.
        foley_report = []
        try:
            # INSIDE the try, so a foley failure takes the SAME terminal path
            # every other failure here takes. This node is the publication
            # boundary: an episode whose bed cannot be mixed is a failed
            # render, not an episode that quietly ships the un-levelled
            # provisional master with 20% of its loudness missing.
            # TWO SOURCES FOR ONE DECISION, AND EITHER ONE IS ENOUGH. The
            # POLICY says this episode's roles rendered foley; the STAMP says
            # the WAV on disk has not been levelled yet. They agree on every
            # graph that is wired correctly -- and when they do not, taking the
            # OR is what stops an un-levelled episode shipping silently.
            #
            # No separate "level only" branch is needed for the stamp-only
            # case: a foley master compiled from ZERO stems is
            # `master * <some constant> + silence`, and a pure scale followed
            # by normalise-to-target is exactly normalise-to-target -- so the
            # constant cannot reach the deliverable whatever it happens to be.
            #
            # THE CONSTANT IS DELIBERATELY NOT NAMED HERE. An earlier draft of
            # this comment asserted 0.80, which was true when the master gain
            # was one fixed number and became false the moment mime made it
            # depend on which lanes the episode carries: on the stamp-only path
            # there is no foley lane in the policy and no bearing row, so
            # `global_master_gain` is 1.0, not 0.80. The invariance is the
            # load-bearing part of this argument; the value never was.
            if (_foley_route(video_policy_json)
                    or _master_wav_owes_a_delivery_gain(master_audio_path)):
                master_audio_path, foley_report = _compile_foley_master(
                    master_audio_path, foley_receipts_json, int(fps),
                    video_policy_json)
            final, report = mux_master_audio(
                silent_video_path, master_audio_path, out, ffmpeg=ffmpeg, fps=int(fps),
                declared_credits_tail_s=float(declared_credits_tail_s or 0.0),
            )
            report.extend(foley_report)
            # PUBLICATION IS A SEPARATE DECISION FROM PRODUCTION. The archival
            # final above is written unconditionally -- it is finished work and
            # the operator keeps it. The OBS copy is the PUBLISHED deliverable,
            # and a research-only source is cleared for exactly the first and
            # not the second. Withholding is not a failure: this node returns
            # success either way and says which happened, out loud.
            if decision.publishable:
                obs_copy = self._publish_to_obs(final)
                report.append("obs_publish OK -> " + obs_copy)
            else:
                obs_copy = None
                report.append("obs_publish BLOCKED -- " + decision.summary())
                log.warning(
                    "[OTR_MasterAudioMux] obs_publish BLOCKED (%s) -- the "
                    "archival final is on disk at %s and no OBS copy was made",
                    decision.summary(), final,
                )
            # N2 (truthful ledger): this terminal node restamps
            # final_video_path over node 93's pre-credits/pre-mux blend.
            report.append(self._stamp_terminal_paths(
                final, obs_copy, master_audio_path))
            # OH-3 (output-tree contract 2026-06-11): post-publish janitor
            # pass over episodes/_shared/tmp -- the ONE sanctioned
            # auto-delete; fully fail-soft (PD1, never blocks the mux).
            try:
                from ._otr_janitor import sweep_shared_tmp
                _jrep = sweep_shared_tmp()
                if _jrep.deleted:
                    report.append("janitor: swept %d stale tmp entr%s"
                                  % (len(_jrep.deleted),
                                     "y" if len(_jrep.deleted) == 1
                                     else "ies"))
            except Exception as _jexc:  # noqa: BLE001 -- PD1
                log.info("[OTR_MasterAudioMux] janitor sweep skipped: %s",
                         _jexc)
        except _Interrupted:
            raise
        except (ValueError, OSError) as exc:
            # FAIL THE RUN. This node is TERMINAL -- it muxes the master audio and
            # PUBLISHES the episode to obs. If it cannot, there IS no episode, and a
            # render with no episode is a FAILED render.
            #
            # This used to swallow the exception and RETURN an empty path plus an
            # "error: ..." report string. That single line silently
            # neutralized EVERY fail-closed gate in mux_master_audio() -- missing
            # ffmpeg, missing silent video, missing master audio, the duration drift
            # guard, the audio-SHA identity check. All of them raise ValueError BY
            # DESIGN (the docstring: "raises ValueError on any gate failure (never
            # produces a silently-wrong episode)"; one arm is even commented "never
            # mask the fail-closed path"). Catching them here masked all of them:
            # the graph completed, ComfyUI logged "Prompt executed", the harness
            # recorded RESULT SUCCESS -- and no file existed.
            #
            # Live 2026-07-14, 420w scifi_codex re-leg (prompt 5ab3884b): the
            # duration guard tripped on a ~1-frame concat rounding, this handler ate
            # it, and the leg was recorded GREEN with nothing in otr\obs\. It would
            # have entered the bake-off as a phantom episode. THE ONLY reason it was
            # caught is the operator's standing law: confirm the asset on disk --
            # API success is not proof. A node that cannot fail cannot be trusted,
            # and a guard that cannot abort is not a guard.
            log.error("[OTR_MasterAudioMux] FAILED -- no episode published: %s", exc)
            raise
        for line in report:
            log.info("[OTR_MasterAudioMux] %s", line)
        return (final, "OTR_MasterAudioMux OK -> " + final + "\n" + "\n".join(report))


__all__ = ["OTRMasterAudioMux", "mux_master_audio", "audio_pcm_sha",
           "_reresolve_master_audio"]
