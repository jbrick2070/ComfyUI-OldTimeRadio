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
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile

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


def mux_master_audio(silent_video_path: str, master_audio_path: str, out_path: str,
                     ffmpeg: str = "ffmpeg", fps: int = 25,
                     duration_tol_frames: float = 1.0,
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
    env_ceiling = float(os.environ.get("OTR_MAX_CREDITS_TAIL_S", "45"))
    max_tail_s = declared if declared > 0 else env_ceiling
    tail_src = "declared" if declared > 0 else "env_ceiling"
    if v_dur >= 0 and a_dur >= 0 and v_dur > a_dur + max_tail_s + tol:
        raise ValueError(
            f"OTR_MasterAudioMux: silent video {v_dur:.4f}s exceeds master audio "
            f"{a_dur:.4f}s by > the credits-tail budget ({max_tail_s:.1f}s "
            f"[{tail_src}] + {tol:.4f}s) -- likely a composite/credits "
            f"frame-budget bug, not the intended silent credits tail"
        )
    report.append(
        f"duration_check v={v_dur:.3f}s a={a_dur:.3f}s "
        f"tail_budget={max_tail_s:.1f}s ({tail_src}) OK")

    _poll_interrupt()
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
    SAME basename. Re-resolve to that same file via the newest on-disk ledger
    (the same durable-ledger contract OTR_ShotLock uses for audio timing).

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
        roots = []
        try:
            from . import _otr_paths as _OP
            roots.append(Path(_OP.otr_episodes_root()))
        except Exception:  # noqa: BLE001
            base = os.environ.get("OTR_OUTPUT_DIR") or "."
            roots.append(Path(base) / "otr" / "episodes")
        p = _OL.find_most_recent_ledger(roots)
        if not p:
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
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "ffmpeg": ("STRING", {"default": "ffmpeg"}),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Final mp4 path. Empty -> <output>/otr/episodes/<stem>_final.mp4.",
                }),
                "declared_credits_tail_s": ("FLOAT", {
                    "default": 0.0, "forceInput": True,
                    "tooltip": "OTR_CreditsRoll's declared silent-credits tail "
                               "duration. Makes the tail guard credits-aware "
                               "(v <= a + declared + tol); 0 -> the "
                               "OTR_MAX_CREDITS_TAIL_S env ceiling.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _default_out(self, silent_video_path: str) -> str:
        try:
            import folder_paths  # type: ignore
            root = folder_paths.get_output_directory()
        except Exception:  # noqa: BLE001
            root = "."
        # OUTPUT HYGIENE (operator directive 2026-06-09): the final lands in
        # the episode's OWN folder under otr/episodes/<ep>/ (the obs copy is
        # the only file outside it). <ep> = the input stem minus the known
        # post-chain suffixes. The chain appends OUTERMOST-LAST, so peel in that
        # order: OTR_CreditsRoll appends "_with_credits" (credits enrichment
        # 2026-07-03), the blend stage "_procgen_blended", the composite
        # "_silent". Order matters -- the loop strips each suffix once.
        stem = os.path.splitext(os.path.basename(silent_video_path or "episode"))[0]
        ep = stem
        for suffix in ("_with_credits", "_procgen_blended", "_silent"):
            if ep.endswith(suffix):
                ep = ep[: -len(suffix)]
        out_dir = os.path.join(root, "otr", "episodes", ep)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{stem}_final.mp4")

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
        obs_dir = os.environ.get("OTR_OBS_DIR", "").strip()
        if not obs_dir:
            try:
                import folder_paths  # type: ignore
                root = folder_paths.get_output_directory()
            except Exception:  # noqa: BLE001
                root = "."
            obs_dir = os.path.join(root, "otr", "obs")
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

    def mux(self, silent_video_path, master_audio_path, audio_done="", fps=25,
            ffmpeg="ffmpeg", output_path="", declared_credits_tail_s=0.0):
        master_audio_path = _reresolve_master_audio(master_audio_path)
        out = output_path.strip() or self._default_out(silent_video_path)
        try:
            final, report = mux_master_audio(
                silent_video_path, master_audio_path, out, ffmpeg=ffmpeg, fps=int(fps),
                declared_credits_tail_s=float(declared_credits_tail_s or 0.0),
            )
            obs_copy = self._publish_to_obs(final)
            report.append("obs_publish OK -> " + obs_copy)
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
            log.error("[OTR_MasterAudioMux] %s", exc)
            return ("", f"error: {exc}")
        for line in report:
            log.info("[OTR_MasterAudioMux] %s", line)
        return (final, "OTR_MasterAudioMux OK -> " + final + "\n" + "\n".join(report))


__all__ = ["OTRMasterAudioMux", "mux_master_audio", "audio_pcm_sha",
           "_reresolve_master_audio"]
