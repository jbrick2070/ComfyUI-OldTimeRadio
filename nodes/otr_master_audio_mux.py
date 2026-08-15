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

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import logging

log = logging.getLogger("OTR")


def _ffmpeg_bin(ffmpeg: str) -> str:
    return ffmpeg if (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)) else ""


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or ""


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
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


def _count_audio_streams(path: str) -> int:
    fp = _ffprobe_bin()
    if not fp:
        return -1
    p = _run([fp, "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=index", "-of", "csv=p=0", path])
    return len([ln for ln in (p.stdout or "").splitlines() if ln.strip()])


def audio_pcm_sha(path: str) -> str:
    """SHA-256 of decoded s16le mono @24k -- codec/container-agnostic audio
    identity (same method the A-S2 mux probe used). '' on failure."""
    fp = shutil.which("ffmpeg")
    if not fp:
        return ""
    raw = subprocess.run(
        [fp, "-v", "error", "-i", path, "-map", "0:a", "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", "-"],
        capture_output=True)
    if raw.returncode != 0 or not raw.stdout:
        return ""
    return hashlib.sha256(raw.stdout).hexdigest()


#: The legacy credits-tail ceiling when no roll declares its own duration.
_MAX_CREDITS_TAIL_S_DEFAULT = 45.0


def _credits_tail_ceiling() -> float:
    """The ``OTR_MAX_CREDITS_TAIL_S`` ceiling -- NAMED and ignored when
    malformed, never fatal.

    This knob is read at the LAST node of the graph, after the whole episode has
    rendered. It used to be a bare ``float(os.environ.get(...))``, so a single
    typo in a server's launch environment (``45s``, ``forty-five``) killed a
    finished episode at the finish line with an uncaught ValueError -- hours of
    render lost to a value that only widens a sanity ceiling.

    That is the ``PBUG-20260723-02`` shape this build has now closed three times
    over: a knob exported at launch cannot bind work submitted to an
    already-booted server, so a malformed one must be IGNORED, never FATAL. The
    house posture is `otr_silent_composite._unsharp_amount`; this adds the
    WARNING that one omits, because a ceiling silently reverting to the default
    is a ceiling the operator thinks they moved. This is the one env read in
    the mux."""
    raw = os.environ.get("OTR_MAX_CREDITS_TAIL_S")
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
        raise ValueError(
            f"OTR_MasterAudioMux: silent video {v_dur:.4f}s exceeds master audio "
            f"{a_dur:.4f}s by {_excess:.4f}s, over the credits-tail budget "
            f"({max_tail_s:.4f}s [{tail_src}] + {tol:.4f}s tol = "
            f"{max_tail_s + tol:.4f}s) by {_excess - max_tail_s - tol:.4f}s "
            f"-- likely a composite/credits frame-budget bug, not the intended "
            f"silent credits tail"
        )
    if v_dur < 0 or a_dur < 0:
        # THE RECEIPT MAY NOT SAY OK OVER A FAILED PROBE. `_probe_float`
        # returns -1.0 when ffprobe is absent or the duration is unparsable,
        # and the gate above is skipped in that case -- so the old line
        # appended "duration_check v=-1.000s a=-1.000s ... OK", which reads as
        # a passed check to every downstream reader of this report.
        #
        # Reported as UNPROVEN rather than made fatal: the gate is the FINAL
        # SANITY CEILING, not the primary correctness guard (the CreditsRoll
        # and composite frame budgets are), and refusing here would lose a
        # finished episode on a box that merely lacks ffprobe. What must not
        # happen is a receipt claiming a proof it does not have.
        report.append(
            f"duration_check v={v_dur:.3f}s a={a_dur:.3f}s "
            f"tail_budget={max_tail_s:.1f}s ({tail_src}) UNPROVEN "
            f"(duration probe failed -- gate SKIPPED, not passed)")
        log.warning(
            "[OTR_MasterAudioMux] duration gate SKIPPED: probe returned "
            "v=%.3f a=%.3f. The video-longer-than-audio ceiling was NOT "
            "checked for this episode.", v_dur, a_dur)
    else:
        report.append(
            f"duration_check v={v_dur:.3f}s a={a_dur:.3f}s "
            f"tail_budget={max_tail_s:.1f}s ({tail_src}) OK")

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
    h_master = audio_pcm_sha(master_audio_path)
    h_out = audio_pcm_sha(out_path)
    if not h_out or h_out != h_master:
        raise ValueError(
            f"OTR_MasterAudioMux: output audio NOT byte-identical to master "
            f"(master={h_master[:12]} out={h_out[:12]}); the audio was re-encoded "
            f"or trimmed -- C7/V-1 violated"
        )
    report.append(f"audio_byte_identical OK ({h_out[:12]})")
    return out_path, report


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
    if os.environ.get("OTR_TEST_MODE") == "1":
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
    """``<output>/otr/episodes``, or ``./otr/episodes`` outside ComfyUI."""
    try:
        import folder_paths  # type: ignore
        root = folder_paths.get_output_directory()
    except Exception:  # noqa: BLE001
        root = "."
    return Path(root) / "otr" / "episodes"


def _episode_stem(silent_video_path: str) -> str:
    return os.path.splitext(os.path.basename(silent_video_path or "episode"))[0]


def _obs_dir() -> Path:
    """The operator-facing OBS folder. ONE owner for where "published" means.

    ``OTR_OBS_DIR`` pins it explicitly -- on this box the headless server
    renders into the ComfyUI-Installs tree while the operator watches
    ``Documents\\ComfyUI\\output\\otr\\obs`` (two-tree split, 2026-06-09), so
    the launch recipe sets it. Pure: resolves a path and creates nothing, so a
    caller may ASK where obs is without bringing it into existence.
    """
    pinned = os.environ.get("OTR_OBS_DIR", "").strip()
    if pinned:
        return Path(pinned)
    try:
        import folder_paths  # type: ignore
        root = folder_paths.get_output_directory()
    except Exception:  # noqa: BLE001
        root = "."
    return Path(root) / "otr" / "obs"


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
                and stem.startswith(candidate.name)):
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
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "ffmpeg": ("STRING", {"default": "ffmpeg"}),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Final mp4 path. Empty -> <output>/otr/episodes/<stem>_final.mp4.",
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
        dst = os.path.join(obs_dir, os.path.basename(final))
        # PLAYABILITY (operator screenshot 2026-06-09): -c:a copy from the WAV
        # master leaves raw PCM ("ipcm") in the MP4 -- byte-identical but
        # unplayable in standard players (Windows Media Player refuses the
        # audio). The obs deliverable is the WATCHABLE copy: video stream
        # copied untouched, audio encoded AAC-320k. The ARCHIVAL byte-identical
        # PCM final stays in otr/episodes/<ep>/ (mux gate already asserted it
        # against the frozen master; the master itself is never touched).
        fb = _ffmpeg_bin("ffmpeg") or "ffmpeg"
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
        _, episode_dir = _inflight_episode_for_stem(stem)
        episode_id = episode_dir.name if episode_dir is not None else ""
        decision = _publication_decision(str(kwargs.get("silent_video_path") or ""))
        # The REASON rides in the key as well as the digest: a blocked decision
        # has no digest (there is no receipt to hash), and two different
        # blocking reasons must not collide into one cache entry.
        parts = (manifest, stem, episode_id, decision.reason,
                 "1" if decision.publishable else "0", decision.digest)
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def mux(self, silent_video_path, master_audio_path, audio_done="",
            declared_credits_tail_s=0.0, clip_manifest_json="", fps=25,
            ffmpeg="ffmpeg", output_path=""):
        # ``clip_manifest_json`` is a RETIRED connector (rip-sfx 2026-08-06):
        # still wired on the canonical graph and hashed by IS_CHANGED, but it
        # feeds nothing -- the SFX bed compiler it once armed is deleted.
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
        try:
            final, report = mux_master_audio(
                silent_video_path, master_audio_path, out, ffmpeg=ffmpeg, fps=int(fps),
                declared_credits_tail_s=float(declared_credits_tail_s or 0.0),
            )
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
