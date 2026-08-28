r"""
Scene Sequencer + Episode Assembler - Orchestrate the Full Radio Show
======================================================================

Two nodes:
  1. SceneSequencer - Takes the L3 ledger and renders each line through
     the appropriate TTS engine (Bark or Parler), passes music
     cues through, and outputs a scene. Features:
     intelligent pacing (breath buffers, BEAT/PAUSE tags), continuous
     room tone bed, voice_assignments dispatch from the ledger's cast
     block.

  2. EpisodeAssembler - Takes multiple rendered scenes, adds act
     breaks, opening/closing themes, and assembles the complete
     episode WAV.

These nodes consume the L3 ledger produced by LedgerScriptWriter and
emitted via the FreezeCascade.script_json fanout. The legacy parser-
list / Director-shape inputs were removed in voice-path-cleanbreak
S2-S8; forensic comments downstream mark the migration points.

v1.0  2026-04-04  Jeffrey Brick
v2.0  2026-05-13  voice-path-cleanbreak S23.4 (docstring scrub)
"""

import json
import hashlib
import logging
import math
import os
import re

import numpy as np
import torch

from .story_orchestrator import _runtime_log

log = logging.getLogger("OTR")


def _sha256_file(path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 for an on-disk asset.

    Master WAVs can be hundreds of megabytes; byte identity must not require a
    second full-file allocation. The file is read only after ``wave.close`` has
    finalized its header, so this digest names the exact bytes video slicing
    and the terminal mux consume.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(max(1, int(chunk_bytes)))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stamp_master_audio_identity(ledger: dict, master_sha256: str) -> None:
    """Stamp the frozen master-byte identity into its owned ledger section."""
    sha = str(master_sha256 or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("master audio sha256 must be 64 lowercase hex chars")
    audio = ledger.setdefault("audio", {})
    if not isinstance(audio, dict):
        raise TypeError("ledger.audio must be a dict")
    audio["master_audio_sha256"] = sha
    audio["ledger_frozen"] = True


#: What the master WAV on disk actually IS. Stamped by the assembler on EVERY
#: episode, read by ``OTR_MasterAudioMux``. See :func:`_stamp_master_wav_flavour`.
MASTER_WAV_DELIVERY_LEVELLED = "delivery_levelled"
MASTER_WAV_PRE_LOUDNESS = "pre_loudness"


def _stamp_master_wav_flavour(ledger: dict, flavour: str) -> None:
    """Record whether the master WAV still owes its delivery loudness pass.

    THE STAMP IS A FACT ABOUT THE BYTES ON DISK; THE POLICY IS AN INFERENCE
    ABOUT THAT FACT. Both this node and ``OTR_MasterAudioMux`` read the same
    ``video_policy_json`` off the same OTR_VideoDirector output, so they
    normally agree -- but a hand-edited graph that wires the policy to one and
    not the other has two possible outcomes and only one is benign:

    * assembler LEVELLED, mux mixes and levels -> the 0.20/0.80 balance is
      struck against an already-levelled master. The delivery level is still
      correct, because ``_master_loudness`` normalises TO a target rather than
      applying a relative boost.
    * assembler PRE-LOUDNESS, mux copies -> the episode ships roughly 4 dB
      off, which is the exact error the -14 LUFS target was measured to fix,
      and nothing anywhere says so.

    This stamp exists for the second one: the mux takes the delivery gain
    whenever it reads ``pre_loudness`` here, whatever it concluded from the
    policy.

    STAMPED ON EVERY ROUTE, not only the foley one. A stamp that appeared only
    sometimes could not be trusted when absent, because absence would mean both
    "already levelled" and "written by a build that predates the stamp".
    """
    audio = ledger.setdefault("audio", {})
    if not isinstance(audio, dict):
        raise TypeError("ledger.audio must be a dict")
    audio["master_wav_flavour"] = str(flavour)


def _reconcile_active_music_manifest(manifest: dict | None, *, source: str):
    """Persist manifest-backed ``music[]`` rows before timeline mutation.

    Legacy banks author music sentinels but no ``music[]`` array.  The rendered
    cue manifest is therefore the first complete cue authority.  Reconcile it
    into the active ledger before either audio consumer writes timing, while an
    authored-row identity mismatch remains a loud terminal error.
    """
    if manifest is None:
        return None
    from . import _otr_cue_manifest as _CM
    from . import _otr_ledger as _OTRL

    ledger_path = _OTRL.in_flight_ledger_path()
    if ledger_path is None:
        return None
    ledger = _OTRL.load_ledger_safe(ledger_path)
    if not isinstance(ledger, dict):
        return None
    receipt = _CM.reconcile_ledger_music(ledger, manifest, allow_create=True)
    _OTRL.save_ledger_safe(ledger_path, ledger)
    log.info(
        "[%s] cue-manifest ledger reconciliation: matched=%d created=%d "
        "paths=%d total=%d",
        source,
        receipt["matched"], receipt["created"],
        receipt["paths_updated"], receipt["total"],
    )
    return receipt


# THE INLINE BARK FALLBACK AND ITS PRIVATE HELPER CHAIN WERE REMOVED
# 2026-08-28. `_generate_bark_for_line` had no production caller -- the
# only mention of it in `sequence()` is a comment saying the fallback is
# retired, and the shortfall branch RAISES instead of calling it. Its
# own docstring already said 'has no live caller in the sequencer'.
#
# The three helpers went with it because each had exactly ONE external
# caller and it was this function (the other references were recursion
# and a comment). They were NOT shared with the live path: the comment
# claiming shared-helper status was about regex PARITY with independent
# copies in `_otr_bark_lib.py`, and `eng_bark.py` imports from there,
# never from here.
#
# THE NO-FALLBACK CONTRACT IS UNAFFECTED. It is proved by
# test_sequencer_ledger.py::test_sequencer_clip_shortfall_fails_loud,
# which asserts the ValueError against sequence() directly and never
# touches the deleted function.


# -----------------------------------------------------------------------------
# LOG CLEANUP - suppress urllib3/httpx cache-check spam from HuggingFace
# -----------------------------------------------------------------------------
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


def _trim_trailing_silence(clip_np, threshold=1e-4, min_samples=100):
    """Strip trailing zero-padding from a 1-D float32 numpy array.

    ComfyUI batched AUDIO tensors are zero-padded to uniform length.
    This removes the silent tail so dialogue clips don't insert
    unnatural dead air in the assembled episode.

    Keeps at least `min_samples` to avoid returning an empty array
    for genuinely quiet clips.
    """
    abs_amp = np.abs(clip_np)
    # Find last sample above noise floor
    above = np.where(abs_amp > threshold)[0]
    if len(above) == 0:
        # Entire clip is below threshold - return a tiny slice
        return clip_np[:min_samples] if len(clip_np) > min_samples else clip_np
    last_idx = above[-1]
    # Keep a small tail (50ms at 48kHz - 2400 samples) for natural decay
    tail_pad = min(2400, len(clip_np) - last_idx - 1)
    end = min(last_idx + tail_pad + 1, len(clip_np))
    return clip_np[:end]


def _normalize_clip(clip_np, target_peak=0.85):
    """Normalize a 1-D float32 clip to a target peak amplitude.

    Bark outputs vary wildly in volume between characters and takes.
    This brings every dialogue clip to a consistent level so the Commander
    doesn't whisper while the Pilot screams.

    Uses sample-peak normalization (not RMS) to preserve dynamics within each clip
    while matching overall loudness across clips.
    """
    peak = np.abs(clip_np).max()
    if peak < 1e-6:
        return clip_np  # silence - don't amplify noise floor
    return (clip_np * (target_peak / peak)).astype(np.float32)


#: Delivery loudness target. YouTube normalises playback to about -14 LUFS and
#: the normalisation is DOWN-ONLY: content louder than target is attenuated,
#: content quieter is played as-is and never boosted. Mastering hot therefore
#: buys nothing (the loudness is removed at playback while the limiting used to
#: obtain it stays in the audio), and mastering quiet is a permanent handicap.
_MASTER_TARGET_LUFS = -14.0

#: pyloudnorm needs at least one 400 ms block for a valid integrated reading.
_LUFS_MIN_SECONDS = 0.4


def _master_loudness(waveform, ceiling_dbfs: float = -1.0, makeup_db=None,
                     target_lufs=None, sample_rate: int = 48000):
    """Final episode loudness master: measure LUFS, gain to target, peak-safe.

    THIS STAGE SETS THE DELIVERY LEVEL. It does NOT balance the mix -- that
    already happened per clip, upstream, where every spoken line is levelled to
    ``OTR_SEGMENT_TARGET_RMS_DBFS`` (-16 dBFS active RMS) by
    ``_level_dialogue_clip`` on both the announcer and character buses. **The
    two do different jobs and must not be confused:** per-clip levelling makes
    one character sit correctly against another; this stage moves the finished
    mix as a whole. Because it applies a SINGLE LINEAR GAIN it cannot disturb
    the clip-to-clip balance established upstream -- every relative level
    survives untouched.

    WHY THIS REPLACED PEAK NORMALISATION (2026-08-19, PBUG-20260819-01,
    panelled by Fable + Sonnet on the operator's instruction to confirm best
    practice for YouTube). The old stage applied a fixed +4 dB makeup into a
    tanh soft knee, then trimmed the PEAK to ``ceiling_dbfs``. Peak is the
    wrong control for a platform that normalises by loudness: two files can
    share a -1.0 dBFS peak and differ by 10 dB in LUFS. Measured across 8 real
    masters spanning two months, that stage delivered a mean of **-9.87 LUFS
    (std 0.41)**, roughly 4 dB hotter than target -- so every episode was
    attenuated at playback while the limiting that bought the loudness stayed
    in the audio.

    A TRAP RECORDED SO NOBODY REPEATS IT. Peak ceiling and delivered loudness
    are NOT linearly related in the old algorithm, because it renormalised to
    the ceiling BEFORE the tanh saturated: at ceiling -1.0 the limiter sits deep
    in its knee; at -9.0 it barely engages. Measured on the real function, an
    8 dB ceiling move produced a **10.3 dB** loudness move (-13.26 -> -23.58
    LUFS). A blind fader A/B on this stage cannot predict its own delivered
    level, so ``ceiling_dbfs`` must never be "tuned by ear".

    Fully deterministic (no RNG). Returns ``(waveform, info)``; ``info`` carries
    the measurement for the caller's log and any receipt.

    ``makeup_db`` is retained ONLY for the fallback path below. It is no longer
    the loudness engine.
    """
    import os
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak = waveform.abs().max()
    if float(peak) < 1e-8:
        return waveform, {"mode": "silent", "makeup_db": 0.0}

    if target_lufs is None:
        try:
            target_lufs = float(
                os.environ.get("OTR_MASTER_TARGET_LUFS", _MASTER_TARGET_LUFS))
        except (TypeError, ValueError):
            target_lufs = _MASTER_TARGET_LUFS

    # --- LUFS path: measure, then ONE linear gain, then a peak safety rail --
    measured = None
    if target_lufs is not None:
        try:
            import numpy as _np
            import pyloudnorm as _pln

            arr = (waveform.detach().cpu().numpy()
                   if hasattr(waveform, "detach") else _np.asarray(waveform))
            while arr.ndim > 2:          # (1, ch, n) -> (ch, n)
                arr = arr[0]
            if arr.ndim == 2:            # (ch, n) -> (n, ch): what the meter wants
                arr = arr.T
            if arr.shape[0] >= int(_LUFS_MIN_SECONDS * sample_rate):
                measured = float(
                    _pln.Meter(sample_rate).integrated_loudness(
                        _np.ascontiguousarray(arr, dtype=_np.float64)))
        except Exception as exc:  # noqa: BLE001 -- never fail an episode on this
            log.warning("[EpisodeAssembler] LUFS measurement unavailable (%s); "
                        "falling back to the legacy peak master", exc)
            measured = None

    # -70 rejects the -inf pyloudnorm returns for effectively silent input.
    if measured is not None and measured > -70.0:
        gain_db = float(target_lufs) - measured
        waveform = waveform * (10.0 ** (gain_db / 20.0))
        new_peak = float(waveform.abs().max())
        limited = new_peak > ceiling
        if limited:
            # Safety rail only. On real episodes the gain is NEGATIVE -- we now
            # master quieter than before -- so this does not fire; but a very
            # peaky mix must never leave here above the ceiling.
            waveform = waveform * (ceiling / new_peak)
        return waveform, {
            "mode": "lufs",
            "measured_lufs": round(measured, 2),
            "target_lufs": float(target_lufs),
            "gain_db": round(gain_db, 2),
            "ceiling_dbfs": float(ceiling_dbfs),
            "peak_limited": bool(limited),
            "makeup_db": 0.0,
        }

    # --- Fallback: the legacy tanh master, unchanged ------------------------
    # Reached only when measurement is impossible (pyloudnorm missing, or audio
    # shorter than one 400 ms block). Kept so a render never fails for want of
    # a loudness reading.
    # Normalize to the ceiling first so the limiter sees a known full scale.
    waveform = waveform * (ceiling / peak)
    if makeup_db is None:
        try:
            makeup_db = float(os.environ.get("OTR_MASTER_MAKEUP_DB", "4.0"))
        except (TypeError, ValueError):
            makeup_db = 4.0
    makeup_db = max(0.0, min(12.0, float(makeup_db)))
    if makeup_db > 0.0:
        g = 10.0 ** (makeup_db / 20.0)
        denom = float(torch.tanh(torch.tensor(g)))
        if denom > 1e-8:
            # tanh soft-knee: slope > 1 near zero lifts quiet/mid speech;
            # smooth compression near full scale -> no hard clip. Unity at FS.
            waveform = torch.tanh(waveform * g) / denom
        peak2 = waveform.abs().max()
        if float(peak2) > 1e-8:
            waveform = waveform * (ceiling / peak2)
    return waveform, {
        "mode": "legacy_peak",
        "ceiling_dbfs": float(ceiling_dbfs),
        "makeup_db": makeup_db,
    }


# ---------------------------------------------------------------------------
# Per-segment loudness normalization. RMS leveling is the BAKED-IN DEFAULT
# (evens PERCEIVED dialogue loudness shot-to-shot); OTR_SEGMENT_LOUDNORM=peak
# is an escape hatch to the legacy sample-peak path. The episode master is
# left UNCHANGED (makeup 4.0), and the target sits at ~ the level peak-norm
# already produced, so overall loudness is preserved -- only the per-line
# balance evens out. CPU/numpy-only, deterministic. Roundtable-converged
# 2026-06-22 + operator "bake it in" (docs/2026-06-22-loudness-normalization).
# ---------------------------------------------------------------------------

_LOUDNORM_PREFLIGHT_LOGGED = False


def _segment_loudnorm_mode():
    """Per-segment loudness mode. Default 'rms' (baked in); 'peak' = legacy escape hatch."""
    return os.environ.get("OTR_SEGMENT_LOUDNORM", "rms").strip().lower()


def _env_float(name, default, lo=None, hi=None):
    """Parse an env float; empty/junk/non-finite -> default; optional clamp.

    Read per call (no cache) so tests can monkeypatch and operators can set
    the value at server boot without a stale module-level snapshot.
    """
    raw = os.environ.get(name, "").strip()
    try:
        v = float(raw) if raw else float(default)
    except (TypeError, ValueError):
        v = float(default)
    if not math.isfinite(v):
        v = float(default)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v


def _rms(x):
    """Root-mean-square of a signal in float64; empty/non-finite -> 0.0."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _loudness_normalize_clip(clip_np, target_rms_dbfs, max_boost_db,
                             max_cut_db, gate_dbfs, peak_ceiling):
    """Level a dialogue clip toward a target RMS loudness, peak-safe.

    Applies a single gain (no compression -> intra-clip dynamics preserved):
    measure active-sample RMS, compute the dB gain toward ``target_rms_dbfs``
    clamped to [``max_cut_db``, ``max_boost_db``], convert to linear, then cap
    by ``peak_ceiling``. The peak-safety cap may attenuate BELOW max_cut to
    avoid clipping -- that is intentional. Deterministic, CPU/numpy-only.
    All non-finite / empty / silent / room-tone paths return float32 unchanged.
    """
    x = np.asarray(clip_np)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return np.asarray(clip_np, dtype=np.float32)
    peak = float(np.abs(x).max())
    if peak < 1e-6:
        return np.asarray(clip_np, dtype=np.float32)  # digital silence
    gate_amp = 10.0 ** (gate_dbfs / 20.0)
    active = x[np.abs(x) >= gate_amp]
    rms = _rms(active if active.size else x)
    if rms <= 0.0:
        return np.asarray(clip_np, dtype=np.float32)
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-10))
    if rms_dbfs <= gate_dbfs:
        return np.asarray(clip_np, dtype=np.float32)  # room tone -> no gain
    gain_db = min(max(target_rms_dbfs - rms_dbfs, max_cut_db), max_boost_db)
    gain_lin = 10.0 ** (gain_db / 20.0)               # dB -> linear
    gain_lin = min(gain_lin, peak_ceiling / peak)     # peak-safety cap
    return (x * gain_lin).astype(np.float32)


def _loudnorm_params():
    """Resolve the rms-mode parameters from the environment (read per call)."""
    return dict(
        target_rms_dbfs=_env_float("OTR_SEGMENT_TARGET_RMS_DBFS", -16.0, -60.0, 0.0),
        max_boost_db=max(_env_float("OTR_SEGMENT_MAX_BOOST_DB", 9.0), 0.0),
        max_cut_db=min(_env_float("OTR_SEGMENT_MAX_CUT_DB", -12.0), 0.0),
        gate_dbfs=_env_float("OTR_SEGMENT_GATE_DBFS", -50.0),
        peak_ceiling=_env_float("OTR_SEGMENT_PEAK_CEILING", 0.95, 0.0, 1.0),
    )


def _log_loudnorm_preflight(params):
    """Log the effective loudness mode + params once per process."""
    global _LOUDNORM_PREFLIGHT_LOGGED
    if _LOUDNORM_PREFLIGHT_LOGGED:
        return
    _LOUDNORM_PREFLIGHT_LOGGED = True
    if params is not None:
        log.info(
            "[loudnorm] per-segment RMS leveling ON (default) -- target=%.1f dBFS "
            "boost<=%.1f cut>=%.1f gate=%.1f ceiling=%.2f",
            params["target_rms_dbfs"], params["max_boost_db"],
            params["max_cut_db"], params["gate_dbfs"], params["peak_ceiling"],
        )
    else:
        log.info("[loudnorm] OTR_SEGMENT_LOUDNORM=peak override -- legacy sample-peak leveling")


def _level_dialogue_clip(clip_np):
    """Route a DIALOGUE clip through the active loudness mode.

    Default 'rms' -> perceived-loudness leveling (baked in). 'peak' escape
    hatch -> legacy sample-peak. Only dialogue clips route through here.
    """
    if _segment_loudnorm_mode() != "peak":
        params = _loudnorm_params()
        _log_loudnorm_preflight(params)
        return _loudness_normalize_clip(clip_np, **params)
    _log_loudnorm_preflight(None)
    return _normalize_clip(clip_np)


def _resample_audio(clip_np, src_rate, dst_rate):
    """Resample a 1-D float32 numpy array (CPU-only).

    Post-engine audio DSP runs on CPU (invariant I-11): resampling must not
    touch CUDA, or the deterministic audio baseline varies under the
    non-strict determinism default. Path selection:
      - scipy.signal.resample_poly - high-quality, anti-aliased CPU path
      - No scipy - np.interp linear fallback
    """
    if src_rate == dst_rate:
        return clip_np.astype(np.float32)

    # I-11: the prior GPU torchaudio fast path was removed so post-engine
    # resampling stays on CPU and the audio baseline is determinism-stable.

    # CPU path: scipy polyphase (high quality, anti-aliased)
    g = math.gcd(int(dst_rate), int(src_rate))
    up = int(dst_rate) // g
    down = int(src_rate) // g

    try:
        from scipy.signal import resample_poly
        return resample_poly(clip_np, up, down).astype(np.float32)
    except ImportError:
        log.warning("[SceneSequencer] scipy not available - falling back to linear "
                    "interpolation for resampling. Install scipy for proper anti-aliasing.")
        new_len = int(len(clip_np) * dst_rate / src_rate)
        return np.interp(
            np.linspace(0, len(clip_np) - 1, new_len),
            np.arange(len(clip_np)),
            clip_np
        ).astype(np.float32)

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "output", "otr", "audio")


# -----------------------------------------------------------------------------
# ROOM TONE BED - continuous background that fills silence between dialogue
# -----------------------------------------------------------------------------

def _generate_room_tone(duration_sec, sample_rate=48000, intensity=0.03, descriptors=""):
    """Generate a dynamic background bed based on Canonical 1.0 ENV descriptors.

    Uses the descriptors (e.g. 'night city street, distant traffic') to skew
    the noise profile and add textures like wind, sirens, or electronic hums.

    Path selection (RTX 5080 optimized):
      - CUDA + duration > 60s - GPU torch path (noise + sin on Tensor Cores)
      - Otherwise - CPU numpy path (low overhead for short beds)
    """
    import torch
    n_samples = int(duration_sec * sample_rate)
    desc = descriptors.lower()
    _use_gpu = (torch.cuda.is_available() and duration_sec >= 60)

    if _use_gpu:
        # -- GPU path: all noise + trig on CUDA ------------------------------
        dev = torch.device("cuda")
        t = torch.arange(n_samples, dtype=torch.float32, device=dev) / sample_rate

        # Base: tape hiss
        hiss = torch.randn(n_samples, dtype=torch.float32, device=dev)
        hiss_cutoff = 800 if ("wind" in desc or "storm" in desc) else 4000
        hiss_intensity = intensity * 1.5 if ("wind" in desc or "storm" in desc) else intensity
        # FFT bandpass on GPU (replaces scipy.sosfilt)
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / sample_rate, device=dev)
        mask = ((freqs >= 100) & (freqs <= hiss_cutoff)).float()
        hiss = torch.fft.irfft(torch.fft.rfft(hiss) * mask, n=n_samples)
        hiss *= hiss_intensity * 0.6

        # Mains hum
        hum_freq = 50 if "euro" in desc else 60
        hum_amp = intensity * 0.15 if ("electronic" in desc or "fluorescent" in desc or "ship" in desc) else intensity * 0.1
        hum = torch.sin(2 * math.pi * hum_freq * t) * hum_amp

        # Textures
        texture = torch.zeros(n_samples, dtype=torch.float32, device=dev)
        if "traffic" in desc or "street" in desc:
            texture += torch.sin(2 * math.pi * 30 * t) * (intensity * 0.2)
        if "siren" in desc:
            siren_mod = torch.sin(2 * math.pi * 0.2 * t) * 100 + 400
            texture += torch.sin(2 * math.pi * siren_mod * t) * (intensity * 0.05)

        # Sporadic crackle (stays on CPU - tiny loop, negligible cost)
        crackle = np.zeros(n_samples, dtype=np.float32)
        n_pops = int(duration_sec * (8 if "vinyl" in desc else 3))
        pop_positions = np.random.randint(0, n_samples, size=n_pops)
        for pos in pop_positions:
            p_len = np.random.randint(int(sample_rate * 0.001), int(sample_rate * 0.004))
            end = min(pos + p_len, n_samples)
            crackle[pos:end] += np.linspace(1.0, 0, end - pos) * intensity * 0.4
        crackle_t = torch.from_numpy(crackle).to(dev, non_blocking=True)

        result = hiss + hum + texture + crackle_t
        log.info("[SceneSequencer] Room tone: GPU path (%.1fs, %d samples)", duration_sec, n_samples)
        return result.cpu().numpy()

    # -- CPU path: numpy (low overhead for short beds) -----------------------
    hiss = np.random.randn(n_samples).astype(np.float32)
    if "wind" in desc or "storm" in desc:
        cutoff = 800
        intensity *= 1.5
    else:
        cutoff = 4000

    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, [100, cutoff], btype='bandpass', fs=sample_rate, output='sos')
        hiss = sosfilt(sos, hiss).astype(np.float32)
    except Exception:
        pass
    hiss *= intensity * 0.6

    # Mains Hum
    hum_freq = 50 if "euro" in desc else 60
    hum_amp = intensity * 0.3 if ("electronic" in desc or "fluorescent" in desc or "ship" in desc) else intensity * 0.1
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    hum = np.sin(2 * np.pi * hum_freq * t) * hum_amp

    # Textures
    texture = np.zeros(n_samples, dtype=np.float32)
    if "traffic" in desc or "street" in desc:
        texture += np.sin(2 * np.pi * 30 * t) * (intensity * 0.2)
    if "siren" in desc:
        siren_mod = np.sin(2 * np.pi * 0.2 * t) * 100 + 400
        texture += np.sin(2 * np.pi * siren_mod * t) * (intensity * 0.05)

    # Sporadic crackle
    crackle = np.zeros(n_samples, dtype=np.float32)
    n_pops = int(duration_sec * (8 if "vinyl" in desc else 3))
    pop_positions = np.random.randint(0, n_samples, size=n_pops)
    for pos in pop_positions:
        p_len = np.random.randint(int(sample_rate * 0.001), int(sample_rate * 0.004))
        end = min(pos + p_len, n_samples)
        crackle[pos:end] += np.linspace(1.0, 0, end - pos) * intensity * 0.4

    return hiss + hum + texture + crackle


# -----------------------------------------------------------------------------
# (BANNER CORRECTED 2026-08-28: this section header used to read "INLINE BARK
# TTS - called by SceneSequencer for dynamic dialogue generation", advertising
# as live a path whose removal is recorded in the tombstone near the top of
# this file. Nothing here generates audio any more.)
#
# Voice preset resolution: cast.voice_preset is the only source.
# The legacy Director voice_map fallback + _voice_preset_for_character +
# gender-aware grab-bag pools were deleted in voice-path-cleanbreak
# 2026-05-12. Empty / non-v2 cast.voice_preset is a writer contract
# violation -- the sequencer's clip-shortfall gate raises ValueError
# (Gate 3 mirror); there is no inline-Bark fallback left to reach.








def _verify_bus_clip_counts(
    *,
    announcer_consumed: int,
    announcer_provided: int,
    character_consumed: int,
    character_provided: int,
    announcer_line_ids: "list | None" = None,
    character_line_ids: "list | None" = None,
) -> None:
    """C2 (S2 P1.3) two-bus terminal check. `consumed` MUST equal
    `provided` on BOTH the announcer bus (node 82) and the character bus
    (node 81). A SURPLUS (provided > consumed) means the voice nodes
    emitted more clips than there are dialogue lines for that bus, leaving
    orphan clips unused and the timeline mis-aligned; a SHORTFALL is
    already caught mid-loop, and is caught here too if reached. Either
    direction raises ValueError naming both buses + the routed line_ids.
    No silent tolerance."""
    if (announcer_consumed != announcer_provided
            or character_consumed != character_provided):
        raise ValueError(
            f"SceneSequencer: clip-count MISMATCH after assembly -- "
            f"announcer bus {announcer_consumed}/{announcer_provided} "
            f"consumed (lines {list(announcer_line_ids or [])}), character "
            f"bus {character_consumed}/{character_provided} consumed (lines "
            f"{list(character_line_ids or [])}). Consumed must equal "
            f"provided on BOTH buses -- a surplus leaves orphan clips and a "
            f"mis-aligned timeline; fix the upstream clip count."
        )


class SceneSequencer:
    """Render a script scene: TTS for each line, music passthrough, pauses."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "sequence"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("scene_audio", "render_log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "Parsed script JSON from LLMScriptWriter"
                }),
            },
            "optional": {
                "tts_audio_clips": ("AUDIO", {
                    "tooltip": "Pre-rendered TTS audio clips (from Bark/Parler batch). "
                               "If provided, dialogue lines use these clips instead of "
                               "placeholder silence. Clips are matched to dialogue lines "
                               "in order. ANNOUNCER lines are NOT expected here - they "
                               "flow through announcer_audio_clips on a separate bus."
                }),
                "announcer_audio_clips": ("AUDIO", {
                    "tooltip": "Pre-rendered ANNOUNCER audio clips from KokoroAnnouncer. "
                               "Consumed in script order for dialogue lines whose "
                               "character_name is ANNOUNCER. Keeps the Voice of God "
                               "bookends separated from the Bark character pool."
                }),
                "start_line": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "First line to render (for chunked processing)"
                }),
                "end_line": ("INT", {
                    "default": 999, "min": 1, "max": 9999,
                    "tooltip": "Last line to render"
                }),
                "output_dir": ("STRING", {"default": ""}),
                # v1.5 Phase 3: Time-Alignment Offset Pin
                "dialogue_offset_ms": ("FLOAT", {
                    "default": 0.0, "min": -500.0, "max": 500.0, "step": 10.0,
                    "tooltip": "Shift all dialogue clips on the timeline (ms). "
                               "Positive = delay, negative = advance."
                }),
                # 720-bakeoff C3: the music bus. The theme node (node 83) emits
                # ALL cues in ONE padded AUDIO batch + a manifest that maps each
                # batch row to a cue. The sequencer inserts INTERSTITIAL cues
                # inline at their music_inter boundary line (resolved by
                # anchor_line_id); opening/closing are prepended/appended by
                # EpisodeAssembler. Optional -- legacy lanes leave both unwired
                # and every music line stays a passthrough (byte-parity).
                "music_cue_audio": ("AUDIO", {
                    "tooltip": "Padded AUDIO batch of every music cue from "
                               "OTR_StableAudioTheme. Sliced per manifest "
                               "sample_count; never the padded tail."
                }),
                "music_cue_manifest_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Cue manifest JSON from OTR_StableAudioTheme "
                               "(maps cue_id/anchor_line_id -> batch row)."
                }),
            },
        }

    def _extract_clips_from_audio(self, audio_input):
        """Extract individual clips from a batched AUDIO input.

        If the AUDIO has batch dim > 1, each batch element is a separate clip.
        If batch dim == 1, it's a single long clip that we return as-is.
        """
        if audio_input is None:
            return []

        if isinstance(audio_input, dict):
            waveform = audio_input.get("waveform")
            sr = audio_input.get("sample_rate", 48000)
        else:
            waveform = audio_input
            sr = 48000

        if waveform is None:
            return []

        # Ensure 3D: (B, C, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Return list of (waveform_np, sample_rate) per batch element.
        # Strip trailing zero-padding from batched tensors (ComfyUI pads
        # shorter clips to match the longest in the batch).
        clips = []
        for b in range(waveform.shape[0]):
            clip_np = waveform[b].cpu().numpy().squeeze()
            clip_np = _trim_trailing_silence(clip_np)
            if len(clip_np) > 0:
                clips.append((clip_np, sr))

        return clips

    def sequence(self, script_json,
                 tts_audio_clips=None,
                 announcer_audio_clips=None,
                 start_line=0, end_line=999, output_dir=DEFAULT_OUT,
                 dialogue_offset_ms=0.0,
                 ):


        _runtime_log("SceneSequencer: Starting 1.0 audio assembly...")
        # Schema l3 (2026-04-28): track wall-clock for meta.phase_ms.scene_sequencer.
        import time as _time
        _phase_t0 = _time.time()

        # Read-side: parse the wire input as a v2 ledger dict.
        # load_ledger raises ValueError on the legacy parser-list shape;
        # Sequencer is in the loud-fail group (Pattern 1) -- bad wiring
        # halts the run early. The legacy Director production_plan_json
        # secondary input was deleted in voice-path-cleanbreak 2026-05-12 --
        # pacing defaults are inlined here, voice_map was unused.
        from . import _otr_ledger_consumers as _OTRLC
        led = _OTRLC.load_ledger(script_json)

        # PACING: Breath buffer + dramatic pauses (v1.4 - 50% duration reduction).
        # Defaults inlined; the legacy Director pacing override is gone.
        breath_ms = 200          # between every dialogue line
        beat_pause_ms = 750      # [BEAT] tag - dramatic beat
        pause_ms = 1000          # [PAUSE] tag - longer pause
        scene_transition_ms = 1250
        act_break_ms = 2500

        # Guard: fall back to DEFAULT_OUT if output_dir is empty/None
        if not output_dir or not output_dir.strip():
            output_dir = DEFAULT_OUT
        os.makedirs(output_dir, exist_ok=True)

        # Free LLM VRAM before TTS generation - Bark needs GPU headroom.
        # LLM is done by this point (script + plan already generated).
        # S30 B4b: route through the modern loader. Cloud-only LLM runs leave
        # no local cache entry, so skip the full torch/CUDA teardown there.
        try:
            from ._otr_model_loader import unload_llm_if_local_resident
            if unload_llm_if_local_resident():
                log.info("[SceneSequencer] Freed LLM VRAM for inline TTS")
            else:
                log.info("[SceneSequencer] No local LLM VRAM to free")
        except Exception:
            pass  # LLM may already be unloaded or not imported

        # Extract pre-rendered clips from batched AUDIO inputs
        tts_clips = self._extract_clips_from_audio(tts_audio_clips)
        announcer_clips = self._extract_clips_from_audio(announcer_audio_clips)
        tts_clip_idx = 0
        announcer_clip_idx = 0
        # C2 (S2 P1.3): per-bus expected line_id arrays. Node 82 = announcer
        # bus, node 81 = character bus. Recorded as each dialogue line is
        # routed so the post-loop terminal check can report exactly which
        # lines consumed which bus when a count mismatch is found.
        announcer_bus_line_ids: list = []
        character_bus_line_ids: list = []
        log.info(
            "[SceneSequencer] Pre-rendered clips: %d TTS, %d ANNOUNCER",
            len(tts_clips), len(announcer_clips),
        )


        # We accumulate silence/audio segments as numpy arrays
        sample_rate = 48000  # standardize output
        all_segments = []
        render_log = []

        # Canonical 1.0+ state tracking
        current_character_name = None
        current_env = "silent room"
        env_timeline = []  # List of (start_sample, end_sample, desc)

        # v1.5 Phase 3: Convert TA offset from ms to samples
        dialogue_offset_samples = int(dialogue_offset_ms * sample_rate / 1000.0)
        if dialogue_offset_ms != 0.0:
            _runtime_log(f"SceneSequencer: TA_Offset active: dialogue={dialogue_offset_ms:+.0f}ms")

        # Materialize the ledger lines list (no role filter -- Sequencer
        # processes character / announcer via the dispatch below;
        # music_* lines pass through with a forensic log line, no
        # segment / no stamp -- preserves the existing behavior where
        # Sequencer never handled music timing).
        _all_ledger_lines = list(_OTRLC.iter_lines(led))
        lines_to_render = _all_ledger_lines[start_line:end_line]
        log.info(f"[SceneSequencer] Rendering ledger lines {start_line}-{min(end_line, len(_all_ledger_lines))}")

        current_sample_pos = 0

        # BUG-LOCAL-106 (2026-04-29): track per-dialogue-line positions
        # in scene_audio space so we can write authoritative start_s /
        # dur_s back to ledger.lines[] at the end of this method.
        # We record positions in SCENE-AUDIO space (no opening theme).
        # EpisodeAssembler shifts these to MASTER-MIX space when it
        # prepends the opening_theme.
        #
        # v2 ledger rewrite (Pattern 4): each entry now carries
        # line_id from the wire ledger so the write-back uses
        # patch_line_fields(led, line_id, ...) instead of fragile
        # text-match.
        dialogue_positions: list[dict] = []

        for i, item in enumerate(lines_to_render):
            # Ledger discriminator: speaker_role replaces the legacy
            # parser-list "type" tag. Mapping:
            #   character / announcer  -> dialogue branch
            #   music_open / music_close / music_inter -> passthrough
            #     (no segment / no stamp -- Sequencer never handled
            #     music timing; EpisodeAssembler prepends opening_theme
            #     / closing_theme audio separately).
            #   anything else -> RAISE (NO FALLBACKS, rip-sfx-broll
            #     2026-07-01: the old silent default-to-dialogue path
            #     let unknown roles ride the wrong bus; an old ledger
            #     carrying speaker_role="sfx" fails LOUD here).
            speaker_role = item.get("speaker_role") or ""
            if speaker_role in ("character", "announcer"):
                item_type = "dialogue"
            elif speaker_role in ("music_open", "music_close", "music_inter"):
                item_type = "music"
            else:
                raise ValueError(
                    f"SceneSequencer: line "
                    f"{str(item.get('line_id') or '?')!r} carries "
                    f"unknown speaker_role {speaker_role!r} (valid: "
                    f"character/announcer/music_open/music_close/"
                    f"music_inter). The 'sfx' role was removed "
                    f"2026-07-01 (rip-sfx-broll) -- regenerate the "
                    f"episode; NO FALLBACKS."
                )
            global_idx = start_line + i

            # S29: Interrupt check
            if i % 10 == 0:
                try:
                    import comfy.model_management
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except ImportError:
                    pass

            segment_np = None

            line_id_for_log = item.get("line_id") or "?"
            # dedicated music_inter sentinel (legacy/scifi_news_pro) OR an ordinary
            # dialogue row.  In the latter case the cue is
            # inserted first and the dialogue still consumes its own voice
            # clip and receives its normal timing stamp.


            # -- LEDGER ROLE DISPATCH --------------------------------------

            if item_type == "music":
                # Passthrough -- music_open / music_close (EpisodeAssembler
                # prepends/appends the theme audio) and any music_inter with
                # no manifest cue (legacy lanes: byte-parity preserved).
                log.info(
                    "[SceneSequencer] passthrough music line %s (role=%s)",
                    line_id_for_log, speaker_role,
                )
                render_log.append(
                    f"[{global_idx}] MUSIC passthrough: {speaker_role}"
                )
                continue

            elif item_type == "dialogue":
                character_name = _OTRLC.speaker_name(led, item)
                line = item.get("text") or ""
                is_announcer = (speaker_role == "announcer")

                if is_announcer and announcer_clip_idx < len(announcer_clips):
                    # Dedicated announcer bus - clean, no character-bus filler
                    # sounds. Kokoro by default; Bark since 2026-08-24 when
                    # announcer_voice_engine="bark" -- the log line below does
                    # not name an engine so it never has to know which.
                    clip_np, clip_sr = announcer_clips[announcer_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _level_dialogue_clip(segment_np)
                    announcer_clip_idx += 1
                    announcer_bus_line_ids.append(item.get("line_id"))
                    render_log.append(f"[{global_idx}] ANNOUNCER: {line[:40]}...")
                elif tts_clip_idx < len(tts_clips):
                    clip_np, clip_sr = tts_clips[tts_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _level_dialogue_clip(segment_np)
                    tts_clip_idx += 1
                    character_bus_line_ids.append(item.get("line_id"))
                    render_log.append(f"[{global_idx}] {character_name}: {line[:40]}...")
                else:
                    # NO-FALLBACK (operator 2026-07-03): a dialogue line with NO
                    # pre-rendered clip is a clip-count SHORTFALL (the voice nodes
                    # emitted fewer clips than there are dialogue lines) -- a
                    # pipeline defect. FAIL LOUD naming the line + the counts; never
                    # silently inline-generate Bark filler. The old inline-Bark
                    # fallback is RETIRED (_generate_bark_for_line is no longer
                    # called from the sequencer).
                    raise ValueError(
                        f"SceneSequencer: no pre-rendered voice clip for dialogue "
                        f"line {item.get('line_id')!r} (character {character_name!r}, "
                        f"char_id={item.get('char_id')!r}, announcer={is_announcer}). "
                        f"Clip-count shortfall: announcer {announcer_clip_idx}/"
                        f"{len(announcer_clips)}, character {tts_clip_idx}/"
                        f"{len(tts_clips)} consumed. NO inline-Bark fallback "
                        f"(no-fallback rip) -- the voice nodes must render one clip "
                        f"per dialogue line; fix the upstream clip count."
                    )

                current_character_name = character_name

            # -- Accumulate Audio and Track Environment Span --------------
            if segment_np is not None:
                # v1.5: Apply dialogue TA_Offset for dialogue/announcer items
                if item_type == "dialogue" and dialogue_offset_samples != 0:
                    offset_silence = max(0, dialogue_offset_samples)
                    if offset_silence > 0:
                        segment_np = np.concatenate([
                            np.zeros(offset_silence, dtype=np.float32),
                            segment_np
                        ])
                seg_len = len(segment_np)
                env_timeline.append((current_sample_pos, current_sample_pos + seg_len, current_env))
                all_segments.append(segment_np)
                # BUG-LOCAL-106: capture authoritative scene-audio
                # position for every dialogue line so the ledger
                # write-back below can give BatchHumoRender real
                # placements instead of the BUG-094 word-share
                # estimator's wrong-when-music-or-pauses-exist guess.
                if item_type == "dialogue":
                    # v2 ledger: speaker_role is already authoritative
                    # on the input row. line_id is carried for the
                    # line_id-keyed write-back below (Pattern 4) so
                    # we no longer text-match.
                    dialogue_positions.append({
                        "line_id": item.get("line_id"),
                        "speaker": _OTRLC.speaker_name(led, item),
                        "speaker_role": speaker_role,
                        "start_s": float(current_sample_pos) / float(sample_rate),
                        "dur_s": float(seg_len) / float(sample_rate),
                    })
                current_sample_pos += seg_len

        # C2 (S2 P1.3) TWO-BUS TERMINAL CHECK: consumed == provided on BOTH
        # buses. The mid-loop else-branch already fails loud on a SHORTFALL
        # (more dialogue lines than clips). This catches the mirror defect a
        # shortfall check misses -- a SURPLUS (the voice node emitted MORE
        # clips than there are dialogue lines for that bus), which otherwise
        # left orphan clips silently unused and a mis-aligned timeline.
        _verify_bus_clip_counts(
            announcer_consumed=announcer_clip_idx,
            announcer_provided=len(announcer_clips),
            character_consumed=tts_clip_idx,
            character_provided=len(tts_clips),
            announcer_line_ids=announcer_bus_line_ids,
            character_line_ids=character_bus_line_ids,
        )

        # Log clip usage stats
        render_log.append(f"--- Audio units assembled: {len(all_segments)}")

        # Concatenate all dialogue segments
        if all_segments:
            combined = np.concatenate(all_segments)
        else:
            combined = np.zeros(int(sample_rate * 1), dtype=np.float32)

        # -- CANONICAL 1.0 ENVIRONMENT MIXING --------------------------
        total_len = len(combined)
        final_bed = np.zeros(total_len, dtype=np.float32)
        # vintage_settings was legacy Director-derived; default room tone
        # intensity inlined here after the production_plan_json socket
        # deletion (P2).
        room_intensity = 0.01
        
        for start, end, desc in env_timeline:
            span_len_sec = (end - start) / sample_rate
            # Generate a specialized texture for this description
            bed_segment = _generate_room_tone(span_len_sec, sample_rate, intensity=room_intensity, descriptors=desc)
            fit_len = min(len(bed_segment), end - start)
            final_bed[start : start + fit_len] += bed_segment[:fit_len]
            
        combined = combined + final_bed
        render_log.append(f"--- Layered {len(env_timeline)} environment segments")

        total_len = len(combined)
        total_sec = total_len / sample_rate
        _runtime_log(f"SceneSequencer: 1.0 Mix complete ({total_sec:.1f}s)")

        waveform = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)
        audio_out = {"waveform": waveform, "sample_rate": sample_rate}
        log_text = "\n".join(render_log)

        # Schema l3 ledger write-back: phase_ms.scene_sequencer +
        # audio_gates "post_scene_sequencer" sha256-of-leading-1KB.
        # Best-effort: any failure logs WARNING but never aborts the
        # render. Audio integrity gate per CLAUDE.md C7 -- if two
        # consecutive runs disagree on this hash after a no-op change
        # we know audio drift slipped in.
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
            # S28 cleanbreak: dropped dead inline `from ._otr_paths
            # import otr_episodes_root, otr_legacy_audio_dir`.
            _phase_ms = int((_time.time() - _phase_t0) * 1000)
            # BUG-LOCAL-021 (Phase G): use in-flight singleton, not mtime
            # walker. See _otr_ledger.in_flight_ledger_path docstring.
            _ledger_p = _OTRL.in_flight_ledger_path()
            if _ledger_p is not None:
                _led = _OTRL.load_ledger_safe(_ledger_p)
                if _led is not None:
                    _OTRL.record_phase_ms(_led, "scene_sequencer", _phase_ms)
                    _wb = combined.tobytes()[: _OTRL.GATE_HASH_BYTES]
                    _gate = _OTRL.audio_gate_record(
                        gate_name="post_scene_sequencer",
                        waveform_bytes=_wb,
                        dur_s=float(total_sec),
                        sample_count=int(total_len),
                        sample_rate=int(sample_rate),
                    )
                    _OTRL.append_audio_gate(_led, _gate)

                    # BUG-LOCAL-106: write authoritative scene-audio
                    # positions back to ledger.lines[]. v2 ledger
                    # rewrite (Pattern 4): patch by line_id from the
                    # wire ledger -- robust against duplicate text
                    # ("Okay." x 2) and skipping the rebuild of a
                    # text->idx map. Positions are in SCENE-AUDIO
                    # space; EpisodeAssembler shifts them to
                    # MASTER-MIX space when it prepends opening_theme.
                    _matched = 0
                    for _pos in dialogue_positions:
                        _line_id = _pos.get("line_id")
                        if not _line_id:
                            continue
                        _fields = {
                            "start_s":       float(_pos["start_s"]),
                            "dur_s":         float(_pos["dur_s"]),
                            "start_s_space": "scene_audio",
                            "speaker_role":  _pos.get("speaker_role") or "character",
                        }
                        if _OTRL.patch_line_fields(_led, _line_id, _fields):
                            _matched += 1

                    # (The sfx line write-back died with the sfx role,
                    # rip-sfx-broll 2026-07-01; the S27 note about the
                    # deleted ledger.sfx[] mirror walk lives in git
                    # history.)


                    _gc = _OTRL.lookup_git_commit(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    if _gc:
                        _OTRL.set_meta(_led, "git_commit", _gc)
                    _OTRL.save_ledger_safe(_ledger_p, _led)
                    log.info(
                        "[SceneSequencer] wrote ledger audio metadata. "
                        "phase_ms=%d total_sec=%.2f "
                        "lines_positioned=%d/%d",
                        _phase_ms, total_sec,
                        _matched, len(dialogue_positions),
                    )
        except Exception as _meta_exc:
            log.warning(
                "[SceneSequencer] schema-l3 ledger update failed: %s", _meta_exc
            )

        return (audio_out, log_text)


class EpisodeAssembler:
    """Assemble multiple scenes into a complete episode with intro/outro.

    Sprint H 3.7 Path G Option D follow-up (Jeffrey 2026-05-18,
    after retest #15 hung at Bark + FLUX co-residence): this is the
    canonical audio-branch sink. It emits a STRING output
    `audio_done` carrying the final episode metadata (length,
    sample rate, segment count). The IMAGE/AUDIO outputs are
    unchanged; the new STRING is a topological gate signal for
    downstream consumers that MUST wait for the audio branch to
    complete.

    Concrete use: OTR_DeferredCheckpointLoader (FLUX) sources its
    `gate_signal` from this output. ComfyUI's executor then
    topologically defers the FLUX 22-GiB GPU load until the audio
    branch (Bark dialogue + Kokoro announcer + MusicGen cues) has
    finished. Audio-is-king prime directive (CLAUDE.md) honored
    structurally: audio runs FIRST and FULLY at low VRAM (Bark
    ~4 GiB, MusicGen ~3 GiB, Kokoro ~1 GiB) before video phase
    starts. No more Bark-vs-FLUX VRAM contention.
    """

    CATEGORY = "OldTimeRadio"
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("episode_audio", "output_path", "episode_info", "audio_done")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_audio": ("AUDIO",),
                "episode_title": ("STRING", {"default": "The Last Frequency"}),
            },
            "optional": {
                "opening_theme_audio": ("AUDIO",),
                "closing_theme_audio": ("AUDIO",),
                "opening_duration_sec": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Max duration of opening theme"
                }),
                "closing_duration_sec": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Max duration of closing theme"
                }),
                "crossfade_ms": ("INT", {
                    "default": 500, "min": 0, "max": 3000, "step": 100,
                    "tooltip": "Crossfade between theme and content"
                }),
                # 720-bakeoff C3: the music bus. When a manifest is wired, the
                # opening/closing themes are sliced out of this padded batch
                # (by sample_count) and fed through the SAME segment/crossfade
                # machinery as the legacy opening_theme_audio/closing_theme_audio
                # inputs -- which stay DECLARED but unlinked (BUG-LOCAL-097
                # widget-slot-drift guard). Legacy lanes (no manifest) fall back
                # to the theme inputs untouched.
                "music_cue_audio": ("AUDIO", {
                    "tooltip": "Padded AUDIO batch of every music cue from "
                               "OTR_StableAudioTheme."
                }),
                "music_cue_manifest_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Cue manifest JSON from OTR_StableAudioTheme "
                               "(maps placement -> batch row)."
                }),
                # THE FOLEY ROUTE SIGNAL (the LTX 2.5 foley bed, 2026-08-26).
                # APPENDED AT THE END, never inserted -- widgets_values is
                # positional and inserting mid-list shifts every saved value
                # (BUG-LOCAL-097).
                #
                # WHY THIS NODE NEEDS IT AT ALL, which is not obvious: this
                # assembler runs at order 12 and the first writer of a per-shot
                # engine_id is ShotLock at order 14, so there is no other way
                # for it to know the episode is on a foley route. Without the
                # signal the "skip the delivery gain" branch below is a silent
                # no-op and the mux applies 0.80 to an already--14 LUFS master
                # -- a double gain, on every foley episode, with nothing in any
                # log to say so.
                #
                # ABSENT / EMPTY -> the historical path, byte for byte. Every
                # non-foley episode behaves exactly as it did before.
                "video_policy_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Video policy JSON from OTR_VideoDirector. Read "
                               "for ONE question: is any role on the LTX 2.5 "
                               "foley lane? If so this node writes an "
                               "UN-LEVELLED provisional master WAV and "
                               "OTR_MasterAudioMux does the single delivery "
                               "loudness pass after mixing the bed in. Empty "
                               "-> today's behaviour, unchanged."
                }),
            },
        }

    def assemble(self, scene_audio, episode_title,
                 music_cue_audio=None, music_cue_manifest_json="",
                 opening_theme_audio=None, closing_theme_audio=None,
                 opening_duration_sec=10.0, closing_duration_sec=8.0,
                 crossfade_ms=500, video_policy_json="",
                 ):


        # Schema l3: phase wall-clock for meta.phase_ms.episode_assembler.
        import time as _time
        _phase_t0 = _time.time()

        # 720-bakeoff C3: prefer the manifest path. Slice the opening/closing
        # cues out of the padded batch and bind them to the SAME local vars the
        # legacy theme inputs use, so every downstream segment / crossfade /
        # BUG-106 shift / placement / mirror step below is unchanged. "" ->
        # None means no music bus is wired (legacy) -> theme inputs are used.
        from . import _otr_cue_manifest as _CM
        _cue_manifest = _CM.parse_manifest(music_cue_manifest_json)
        if (_cue_manifest is None) != (music_cue_audio is None):
            raise _CM.CueManifestError(
                "music cue AUDIO and manifest must be supplied together"
            )
        _reconcile_active_music_manifest(
            _cue_manifest, source="EpisodeAssembler",
        )
        if _cue_manifest is not None and music_cue_audio is not None:
            _cue_batch_wf = (
                music_cue_audio.get("waveform")
                if isinstance(music_cue_audio, dict)
                else music_cue_audio
            )
            if _cue_batch_wf is None:
                raise _CM.CueManifestError(
                    "music cue AUDIO has no waveform"
                )
            _CM.validate_manifest(
                _cue_manifest, batch_size=int(_cue_batch_wf.shape[0]),
            )
            _by_placement: dict = {}
            for _r in _cue_manifest.get("cues", []):
                _by_placement.setdefault(_r["placement"], _r)
            _op = self._cue_from_batch(
                music_cue_audio, _cue_manifest, _by_placement.get("opening"))
            _cl = self._cue_from_batch(
                music_cue_audio, _cue_manifest, _by_placement.get("closing"))
            if _op is not None:
                opening_theme_audio = _op
            if _cl is not None:
                closing_theme_audio = _cl

        # Extract main scene waveform
        if isinstance(scene_audio, dict):
            main_waveform = scene_audio["waveform"]
            sample_rate = scene_audio["sample_rate"]
        else:
            main_waveform = scene_audio
            sample_rate = 48000

        # Ensure 3D
        if main_waveform.dim() == 1:
            main_waveform = main_waveform.unsqueeze(0).unsqueeze(0)
        elif main_waveform.dim() == 2:
            main_waveform = main_waveform.unsqueeze(0)

        xfade_samples = int(sample_rate * crossfade_ms / 1000.0)

        segments = []

        # Opening theme
        if opening_theme_audio is not None:
            opening = self._extract_waveform(opening_theme_audio, target_sr=sample_rate)
            max_samples = int(opening_duration_sec * sample_rate)
            if opening.shape[-1] > max_samples:
                opening = opening[:, :, :max_samples]
            segments.append(opening)

        # Main content
        segments.append(main_waveform)

        # Closing theme
        if closing_theme_audio is not None:
            closing = self._extract_waveform(closing_theme_audio, target_sr=sample_rate)
            max_samples = int(closing_duration_sec * sample_rate)
            if closing.shape[-1] > max_samples:
                closing = closing[:, :, :max_samples]
            segments.append(closing)

        # Match channel counts across all segments
        max_channels = max(s.shape[1] for s in segments)
        matched = []
        for s in segments:
            while s.shape[1] < max_channels:
                s = torch.cat([s, s[:, :1, :]], dim=1)
            matched.append(s)

        # Assemble with real crossfade overlaps between adjacent segments
        # Instead of hard-cutting, we overlap the tail of segment A with the
        # head of segment B using equal-power (sqrt) fades for smooth transitions.
        episode_waveform = matched[0]
        for i in range(1, len(matched)):
            nxt = matched[i]
            xf = min(xfade_samples, episode_waveform.shape[-1], nxt.shape[-1])
            if xf > 0:
                # Equal-power crossfade curves (sqrt for constant energy)
                t = torch.linspace(0.0, 1.0, xf, device=episode_waveform.device)
                fade_out = torch.sqrt(1.0 - t)  # tail of current segment
                fade_in = torch.sqrt(t)          # head of next segment

                # Overlap region: blend tail of current with head of next
                tail = episode_waveform[:, :, -xf:] * fade_out
                head = nxt[:, :, :xf] * fade_in
                blended = tail + head

                # Stitch: everything before overlap + blended + remainder of next
                episode_waveform = torch.cat([
                    episode_waveform[:, :, :-xf],
                    blended,
                    nxt[:, :, xf:]
                ], dim=-1)
            else:
                # Fallback: straight concat if segments too short for crossfade
                episode_waveform = torch.cat([episode_waveform, nxt], dim=-1)

        log.info("[EpisodeAssembler] Assembled %d segments with %dms crossfades",
                 len(matched), crossfade_ms)

        # Final loudness master (post-crossfade): measure integrated LUFS, apply
        # ONE linear gain to the delivery target, then a true-peak safety rail.
        # `sample_rate` is passed because the meter needs it, and `ceiling_dbfs`
        # is passed EXPLICITLY rather than left to the silent default -- the old
        # call site passed neither, which is how the ceiling became invisible.
        # Deterministic. Override the target with OTR_MASTER_TARGET_LUFS.
        # ON A FOLEY ROUTE THE DELIVERY GAIN MOVES, IT DOES NOT DISAPPEAR --
        # and getting that distinction wrong in either direction is a defect.
        #
        # The foley bed is mixed under this master at OTR_MasterAudioMux, four
        # topological stages later. Levelling here AND mixing there would apply
        # 0.80 to an already--14 LUFS master and 0.20 to un-levelled VAE audio,
        # then a third gain over the result -- contradicting the operator's
        # ruling that the ratio is set once and ONE delivery gain follows.
        #
        # BUT SIMPLY SKIPPING THE PASS WOULD BREAK TWO OTHER CONSUMERS. This
        # node's episode_audio output fans out to OTR_SignalLostVideo (node 12)
        # and OTR_SceneAwareScopes (node 94), and both receive the POST-LUFS
        # tensor today; skipping would make procgen and the scopes hotter on
        # foley episodes only, for no reason connected to the bed.
        #
        # So the pass still RUNS and the returned AUDIO stays byte-identical
        # for those two. What changes is only which waveform is written to the
        # WAV the mux later reads: the PRE-loudness one, snapshotted here. The
        # mux mixes into that and performs the single delivery gain itself --
        # unconditionally, even when every splat turns out to be silence, so a
        # foley episode can never ship un-levelled.
        _foley_route = False
        try:
            from ._otr_video_engines import foley_stems as _foley
            _foley_route = _foley.is_foley_route(video_policy_json)
        except Exception as _froute_exc:  # noqa: BLE001 -- never blocks audio
            log.warning(
                "[EpisodeAssembler] could not read the video policy (%s); "
                "treating this as a non-foley episode", _froute_exc)
        _provisional_waveform = (
            episode_waveform.clone() if _foley_route
            and hasattr(episode_waveform, "clone") else None)
        episode_waveform, _loud = _master_loudness(
            episode_waveform, ceiling_dbfs=-1.0, sample_rate=sample_rate)
        if _provisional_waveform is not None:
            log.info(
                "[EpisodeAssembler] FOLEY ROUTE: the master WAV is written "
                "PRE-loudness as a provisional stem; OTR_MasterAudioMux mixes "
                "the foley bed under it and performs the single delivery "
                "loudness pass. episode_audio (SignalLostVideo, scopes) still "
                "carries the levelled tensor.")
        if _loud.get("mode") == "lufs":
            log.info(
                "[EpisodeAssembler] Final loudness master: measured %.2f LUFS "
                "-> target %.1f LUFS (gain %+.2f dB), true-peak ceiling "
                "%.1f dBFS%s (post-crossfade)",
                _loud["measured_lufs"], _loud["target_lufs"], _loud["gain_db"],
                _loud["ceiling_dbfs"],
                " [peak-limited]" if _loud.get("peak_limited") else "")
        else:
            log.info(
                "[EpisodeAssembler] Final loudness master: %s path, makeup %s, "
                "ceiling %s dBFS (post-crossfade)",
                _loud.get("mode"), _loud.get("makeup_db"),
                _loud.get("ceiling_dbfs"))

        # Chunk-E GATE 1: write the frozen master mix to a standalone WAV
        # so OTR_MasterAudioMux can source audio from the audio pipeline
        # directly (decoupled from the procgen MP4 / OTR_SignalLostVideo).
        # Uses Python stdlib `wave` (always available).  Best-effort:
        # any failure keeps the placeholder path and the episode still
        # renders (SignalLostVideo + procgen path act as fallback until
        # the operator re-runs with a clean ledger).
        output_path = "(video-only - master WAV save failed)"
        _master_wav_sha256 = ""
        try:
            import wave as _wave_mod
            import pathlib as _pathlib

            # Derive audio dir from the in-flight ledger path.  The
            # ledger lives at <ep_root>/audio/<id>_ledger.json so its
            # parent IS the episode audio dir.
            _wav_dir = None
            _ep_id = ""
            try:
                from . import _otr_ledger as _OTRL_wav  # type: ignore
                _lp = _OTRL_wav.in_flight_ledger_path()
                if _lp is not None:
                    _wav_dir = _pathlib.Path(_lp).parent
                    # Ledger filename is <episode_id>_ledger.json
                    _ep_id = _pathlib.Path(_lp).stem.replace("_ledger", "")
            except Exception:  # noqa: BLE001
                pass

            if _wav_dir is None:
                # Headless / no ledger: fall back to a stable temp location
                import tempfile as _tempfile
                _wav_dir = _pathlib.Path(_tempfile.gettempdir())
                _ep_id = "otr_episode"

            _wav_dir.mkdir(parents=True, exist_ok=True)
            _master_wav = str(_wav_dir / f"{_ep_id}_master.wav")

            # Save 16-bit PCM WAV (lossless; same format video_engine.py
            # uses for the tmp audio muxed into the procgen MP4).
            # THE ONE PLACE THE FOLEY ROUTE CHANGES WHAT LANDS ON DISK.
            # Non-foley episodes write the levelled master exactly as before.
            _wav_source = (_provisional_waveform
                           if _provisional_waveform is not None
                           else episode_waveform)
            _ew_np = _wav_source.detach().cpu() if hasattr(_wav_source, "detach") else _wav_source
            import numpy as _np_wav
            _pcm16 = (_ew_np.numpy() * 32767).clip(-32768, 32767).astype(_np_wav.int16)
            # episode_waveform is (1, channels, samples) or (channels, samples)
            if _pcm16.ndim == 3:
                _pcm16 = _pcm16[0]          # drop batch dim -> (channels, samples)
            _n_ch = int(_pcm16.shape[0])
            _n_samp = int(_pcm16.shape[1])
            # WAV wants interleaved (samples, channels) in C order
            _pcm16_interleaved = _pcm16.T.copy(order="C")

            with _wave_mod.open(_master_wav, "wb") as _wf:
                _wf.setnchannels(_n_ch)
                _wf.setsampwidth(2)          # 16-bit
                _wf.setframerate(int(sample_rate))
                _wf.writeframes(_pcm16_interleaved.tobytes())

            output_path = _master_wav
            try:
                _master_wav_sha256 = _sha256_file(_master_wav)
            except Exception as _hash_exc:  # noqa: BLE001 -- nonterminal receipt
                # The already-closed WAV remains a valid deliverable and the
                # existing render-driver path fallback stays available. A
                # receipt failure is LOUD, but must not relabel good audio as
                # "video-only" or erase the asset path.
                log.warning(
                    "[EpisodeAssembler] master WAV identity receipt failed "
                    "(audio remains usable): %s",
                    _hash_exc,
                )
            log.info(
                "[EpisodeAssembler] master WAV saved -> %s (sha256=%s)",
                _master_wav,
                _master_wav_sha256[:12],
            )
        except Exception as _wav_exc:  # noqa: BLE001
            log.warning(
                "[EpisodeAssembler] master WAV save failed (non-fatal): %s", _wav_exc
            )
            output_path = "(video-only - master WAV save failed)"

        from datetime import datetime as _dt
        audio_out = {"waveform": episode_waveform, "sample_rate": sample_rate}

        total_sec = episode_waveform.shape[-1] / sample_rate
        info_dict = {
            "title": episode_title,
            "duration_sec": round(total_sec, 1),
            "duration_min": round(total_sec / 60, 1),
            "sample_rate": sample_rate,
            "channels": episode_waveform.shape[1],
            "output_path": output_path,
            "timestamp": _dt.now().isoformat(),
        }
        info = json.dumps(info_dict, indent=2)

        log.info(f"[EpisodeAssembler] '{episode_title}' - {total_sec/60:.1f} min")

        # Schema l3 ledger write-back: phase_ms.episode_assembler +
        # audio_gates "post_episode_assembler" sha256 + BUG-LOCAL-106
        # opening-theme offset shift on lines[].start_s and
        # clips[].start_s. Best-effort; failures are warned not raised.
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
            # S28 cleanbreak: dropped dead inline `from ._otr_paths
            # import otr_episodes_root, otr_legacy_audio_dir`.
            _phase_ms = int((_time.time() - _phase_t0) * 1000)
            # BUG-LOCAL-021 (Phase G): use in-flight singleton, not mtime
            # walker. See _otr_ledger.in_flight_ledger_path docstring.
            _ledger_p = _OTRL.in_flight_ledger_path()
            if _ledger_p is not None:
                _led = _OTRL.load_ledger_safe(_ledger_p)
                if _led is not None:
                    if _cue_manifest is not None:
                        _CM.reconcile_ledger_music(
                            _led, _cue_manifest, allow_create=True,
                        )
                    if _master_wav_sha256:
                        _stamp_master_audio_identity(
                            _led, _master_wav_sha256)
                    _stamp_master_wav_flavour(
                        _led,
                        MASTER_WAV_PRE_LOUDNESS if _provisional_waveform
                        is not None else MASTER_WAV_DELIVERY_LEVELLED)
                    _OTRL.record_phase_ms(_led, "episode_assembler", _phase_ms)
                    # post_episode_assembler audio gate.
                    _ew_cpu = (
                        episode_waveform.detach().cpu().numpy()
                        if hasattr(episode_waveform, "detach")
                        else episode_waveform
                    )
                    _wb = bytes(_ew_cpu.tobytes()[: _OTRL.GATE_HASH_BYTES])
                    _sample_count = int(episode_waveform.shape[-1])
                    _gate = _OTRL.audio_gate_record(
                        gate_name="post_episode_assembler",
                        waveform_bytes=_wb,
                        dur_s=float(total_sec),
                        sample_count=_sample_count,
                        sample_rate=int(sample_rate),
                    )
                    _OTRL.append_audio_gate(_led, _gate)

                    # BUG-LOCAL-106: shift lines[].start_s and
                    # clips[].start_s by the actual master-timeline
                    # offset of the scene audio. Compute the offset
                    # from the segments that were prepended BEFORE
                    # the scene segment. With the equal-power
                    # crossfade in this method, scene audio starts
                    # at (sum of pre-scene segment lengths) -
                    # (number of crossfades crossed * xfade_samples).
                    # Most builds: opening_theme prepended -> scene
                    # starts at opening_theme_dur_samples - xfade.
                    # Compute generically: segments[] is built in
                    # order, scene_audio is at index `_scene_idx`.
                    # _shift_samples = sum of all segments before
                    # scene minus xfade_samples * num_crossfades.
                    try:
                        # `segments` and `xfade_samples` are local to
                        # this method (defined above). We track the
                        # scene's index in the segments list -- with
                        # the current code, segments is opening +
                        # scene + closing so scene is at index 1 if
                        # opening_theme present, else 0.
                        _scene_idx = (
                            1 if opening_theme_audio is not None else 0
                        )
                        _shift_samples = 0
                        for _i, _seg in enumerate(segments[:_scene_idx]):
                            _shift_samples += int(_seg.shape[-1])
                        # Each pre-scene segment that crossfades into
                        # the next subtracts xfade_samples from the
                        # shift (the overlap region eats time off
                        # both sides). With opening->scene only, one
                        # crossfade boundary applies.
                        if _scene_idx > 0:
                            _shift_samples -= int(xfade_samples) * _scene_idx
                            _shift_samples = max(0, _shift_samples)
                        _shift_s = float(_shift_samples) / float(sample_rate)
                    except Exception:
                        _shift_s = 0.0

                    _shifted_lines = 0
                    _shifted_clips = 0
                    if _shift_s > 0.0:
                        for _ln in (_led.get("lines") or []):
                            # Only shift entries that are in
                            # scene-audio space (set by SceneSequencer
                            # write-back). Avoids double-shifting on
                            # re-runs. One loop covers every positioned
                            # lines[] row.
                            if (
                                _ln.get("start_s_space") == "scene_audio"
                                and isinstance(_ln.get("start_s"), (int, float))
                            ):
                                _ln["start_s"] = float(_ln["start_s"]) + _shift_s
                                _ln["start_s_space"] = "master_mix"
                                _shifted_lines += 1
                        for _cl in (_led.get("clips") or []):
                            # clips[] start_s is currently written by
                            # BatchHumoRender BEFORE EpisodeAssembler
                            # in some workflow orderings. If clips
                            # already exist when this runs, shift
                            # them too. Use a flag to be idempotent.
                            if _cl.get("start_s_space") in (None, "scene_audio") and isinstance(_cl.get("start_s"), (int, float)):
                                _cl["start_s"] = float(_cl["start_s"]) + _shift_s
                                _cl["start_s_space"] = "master_mix"
                                _shifted_clips += 1

                    # 720-bakeoff C3 (MF-H): promote every positioned
                    # interstitial from scene-audio to master-mix space even
                    # when the numeric offset is exactly zero (no opening
                    # theme).  Gating this on ``_shift_s > 0`` left valid
                    # zero-offset cues unmirrorable.
                    _shifted_music = 0
                    for _mrow in (_led.get("music") or []):
                        if (
                            _mrow.get("start_s_space") == "scene_audio"
                            and isinstance(_mrow.get("start_s"), (int, float))
                        ):
                            _mrow["start_s"] = (
                                float(_mrow["start_s"]) + _shift_s
                            )
                            _mrow["start_s_space"] = "master_mix"
                            _shifted_music += 1
                    if _shifted_music:
                        log.info(
                            "[EpisodeAssembler] BUG-106/C3 music shift: "
                            "+%.3fs applied to/promoted %d interstitial cue(s)",
                            _shift_s, _shifted_music,
                        )

                    # BUG-LOCAL-107: stamp the music cue placements.
                    # Opening theme starts at master t=0 and ends at
                    # opening_theme_dur. Closing theme starts at
                    # (scene_end - xfade) + scene_dur - xfade
                    # = total_master - closing_dur. Write them by
                    # canonical placement match against ledger.music[].
                    _music_rows = _led.get("music") or []
                    log.info(
                        "[EpisodeAssembler] music_rows=%d segments=%d "
                        "(BUG-LOCAL-130 diagnostic)",
                        len(_music_rows),
                        len(segments) if segments else 0,
                    )
                    if _music_rows and segments:
                        _opening_dur_s = (
                            float(segments[0].shape[-1]) / float(sample_rate)
                            if opening_theme_audio is not None else 0.0
                        )
                        _closing_dur_s = (
                            float(segments[-1].shape[-1]) / float(sample_rate)
                            if closing_theme_audio is not None and len(segments) > 1
                            else 0.0
                        )
                        _master_total_s = (
                            float(episode_waveform.shape[-1]) / float(sample_rate)
                        )
                        for _mc in _music_rows:
                            _placement = _CM.canonical_placement(
                                _mc.get("placement"), _mc.get("cue_id"),
                            )
                            if (
                                _placement == "opening"
                                and opening_theme_audio is not None
                            ):
                                _mc["start_s"] = 0.0
                                _mc["dur_s"] = _opening_dur_s
                                _mc["start_s_space"] = "master_mix"
                            elif (
                                _placement == "closing"
                                and closing_theme_audio is not None
                            ):
                                _mc["start_s"] = max(
                                    0.0, _master_total_s - _closing_dur_s
                                )
                                _mc["dur_s"] = _closing_dur_s
                                _mc["start_s_space"] = "master_mix"

                    # BUG-LOCAL-130 fix (2026-05-01): the music-line
                    # mirror was previously nested inside the
                    # `if _music_rows and segments:` block, so any
                    # workflow ordering that left `segments` empty (or
                    # any silent exception in the placement step
                    # above) skipped the mirror entirely -- ledger.music
                    # had valid opening/closing entries but
                    # ledger.lines never got music_* mirror lines, so
                    # VideoComposite saw no music timeline and the
                    # audio bookend played without visual coverage.
                    # Mirror now runs unconditionally on _music_rows
                    # (filters per-row for valid start_s_space + dur_s).
                    if _music_rows:
                        # ROADMAP P0 step 4c (2026-04-30): mirror music
                        # cues into ledger.lines[] with speaker_role so
                        # downstream visual coverage (post-BUG-129b:
                        # VideoComposite static-radio fill, since
                        # music_* no longer dispatches to HuMo)
                        # has the per-cue ledger entries to walk.
                        #
                        # 2026-04-30 round-robin hardening:
                        #   (a) Chunk music tracks > _MUSIC_MAX_CHUNK_DUR_S
                        #       into multiple sequential entries so
                        #       BatchLTXRender's per-chunk cap doesn't
                        #       leave silent_combined.mp4 shorter than
                        #       the master mix audio (which would let
                        #       VideoComposite's -shortest truncate
                        #       audio at the end of the episode).
                        #   (b) Stamp ``mirrored_from = "music"`` so a
                        #       refresh on re-run can drop stale
                        #       entries before re-mirroring (Gemini's
                        #       append-only-leak catch).
                        #   (c) Sort ledger.lines[] by start_s after
                        #       mirroring so downstream consumers walk
                        #       the timeline in chronological order
                        #       regardless of mirror sequence.

                        # 2026-05-07 PM (BUG-LOCAL-117e): bumped from 7.0s
                        # to 22.0s. Empirical mega-duration test on
                        # 2026-05-07 PM rendered a 25s @ 832x480 LTX clip
                        # cleanly on RTX 5080 16 GB + 64 GB RAM with no
                        # temporal collapse and no identity drift (i2v
                        # anchor + radio mechanical-scene content). 22s
                        # leaves a 3s safety headroom under the 25s
                        # observed ceiling, while letting opening +
                        # closing themes flow as a single continuous
                        # radio scene instead of being chopped into
                        # 5s/7s chunklets that needed concat-stitching.
                        # Combined with BUG-LOCAL-117d (ffmpeg boomerang)
                        # the rendered clip is half-duration -> 11s, a
                        # safe sample window with comfortable headroom.
                        # Post-BUG-129b: music cues route to LTX (not
                        # HuMo), so this cap is gated by LTX coherence,
                        # not HuMo's 177-frame ceiling. See
                        # batch_ltx_render.py LTX_MAX_FRAMES=705 (28.16s
                        # absolute hardware ceiling) and the clip_length
                        # widget default (also 22.0s for parity).
                        _MUSIC_MAX_CHUNK_DUR_S = 22.0

                        _lines_for_music = _led.get("lines") or []

                        # (b) Refresh: drop any prior mirrored music
                        # rows so a re-run with different music config
                        # doesn't leave stale entries behind.  Detect
                        # by mirrored_from marker (set below).
                        _kept = []
                        _removed = 0
                        for _ln in _lines_for_music:
                            if (
                                isinstance(_ln, dict)
                                and _ln.get("mirrored_from") == "music"
                            ):
                                _removed += 1
                                continue
                            _kept.append(_ln)
                        if _removed:
                            log.info(
                                "[EpisodeAssembler] dropped %d stale "
                                "mirrored music line(s) before re-mirror",
                                _removed,
                            )
                        _lines_for_music = _kept

                        _existing_music_ids = {
                            ln.get("line_id") for ln in _lines_for_music
                            if isinstance(ln, dict) and ln.get("line_id")
                        }
                        _PLACEMENT_TO_ROLE = {
                            "opening":   "music_open",
                            "closing":   "music_close",
                            "interstitial": "music_inter",
                        }
                        _appended_music = 0
                        _chunked_cues = 0
                        for _mc in _music_rows:
                            if (
                                _mc.get("start_s_space") != "master_mix"
                                or not isinstance(_mc.get("dur_s"), (int, float))
                                or float(_mc.get("dur_s", 0.0)) <= 0.0
                            ):
                                continue
                            _cue = (_mc.get("cue_id") or "").strip().lower()
                            _placement = _CM.canonical_placement(
                                _mc.get("placement"), _cue,
                            )
                            _role = _PLACEMENT_TO_ROLE[_placement]
                            _full_start = float(_mc.get("start_s") or 0.0)
                            _full_dur = float(_mc.get("dur_s"))
                            _description = (
                                _mc.get("description")
                                or _mc.get("title")
                                or f"{_placement} music"
                            )

                            # (a) Chunk math: how many ≤_MUSIC_MAX_CHUNK_DUR_S
                            # chunks does this cue need?  ceil(full_dur / max).
                            # Chunk durations are equal so the boundaries
                            # land cleanly without remainder fragments.
                            _chunk_count = max(
                                1,
                                int(
                                    (_full_dur + _MUSIC_MAX_CHUNK_DUR_S - 1e-6)
                                    // _MUSIC_MAX_CHUNK_DUR_S
                                ),
                            )
                            _chunk_dur = _full_dur / float(_chunk_count)
                            if _chunk_count > 1:
                                _chunked_cues += 1

                            for _chunk_idx in range(_chunk_count):
                                _chunk_start = _full_start + _chunk_idx * _chunk_dur
                                # Namespaced ID per round-robin advice:
                                # "music_<cue>_NNN" never collides with
                                # dialogue line IDs ("l001", "l002").
                                _line_id = (
                                    f"music_{_cue or 'inter'}_"
                                    f"{_chunk_idx + 1:03d}"
                                )
                                if _line_id in _existing_music_ids:
                                    continue
                                _lines_for_music.append({
                                    "line_id":      _line_id,
                                    "speaker":      "RADIO",
                                    "speaker_role": _role,
                                    "text":         (
                                        _description
                                        if _chunk_idx == 0
                                        else f"{_description} (cont. "
                                             f"{_chunk_idx + 1}/"
                                             f"{_chunk_count})"
                                    ),
                                    "start_s":      _chunk_start,
                                    "dur_s":        _chunk_dur,
                                    "start_s_space": "master_mix",
                                    "shot_id":      _mc.get("shot_id"),
                                    # Markers for the refresh logic
                                    # above + future post-mortem.
                                    "mirrored_from":      "music",
                                    "music_cue_id":       _cue or "inter",
                                    "music_chunk_index":  _chunk_idx + 1,
                                    "music_chunk_count":  _chunk_count,
                                })
                                _appended_music += 1

                        # BUG-LOCAL-130 diagnostic: log mirror outcome
                        # ALWAYS, even when 0 lines appended, so silent
                        # zero-mirror runs are visible in the log.
                        log.info(
                            "[EpisodeAssembler] music mirror outcome: "
                            "music_rows=%d appended=%d chunked_cues=%d "
                            "stale_removed=%d existing_music_ids=%d",
                            len(_music_rows), _appended_music,
                            _chunked_cues, _removed,
                            len(_existing_music_ids),
                        )
                        if _appended_music or _removed:
                            # (c) Sort the entire lines[] by start_s
                            # so dialogue + announcer + music
                            # all walk in chronological order.  Lines
                            # without start_s sort to the end via a
                            # large fallback key (they shouldn't exist
                            # post Step 4b/4c but we don't crash on
                            # legacy rows).
                            _lines_for_music.sort(
                                key=lambda _l: (
                                    float(_l.get("start_s"))
                                    if isinstance(_l, dict)
                                    and isinstance(_l.get("start_s"), (int, float))
                                    else 1e18
                                )
                            )
                            _led["lines"] = _lines_for_music
                            log.info(
                                "[EpisodeAssembler] music mirror: "
                                "appended=%d, chunked_cues=%d, "
                                "stale_removed=%d, total_lines=%d "
                                # THE BOOMERANG CLAUSE IS GONE (no-mirror step 6,
                                # 2026-08-06). This line printed into the
                                # operator's LIVE RUN that "BUG-LOCAL-117d
                                # boomerang doubles the rendered half-clip back
                                # to full audio duration" -- describing machinery
                                # that was disarmed on 2026-08-02 and deleted on
                                # 2026-08-06. A stale comment misleads a reader;
                                # a stale LOG misleads the operator mid-render,
                                # while they are deciding whether a leg is
                                # behaving. Music beats past the chunk ceiling
                                # are covered by chained forward segments now.
                                "(post-BUG-117e: music chunks <= %.1fs "
                                "to fit the LTX 22B 25s safe envelope)",
                                _appended_music, _chunked_cues,
                                _removed, len(_lines_for_music),
                                _MUSIC_MAX_CHUNK_DUR_S,
                            )

                    # BUG-LOCAL-107: append crossfade boundaries to
                    # ledger.transitions[] so post-mortem can audit
                    # the seam between opening->scene and scene->closing.
                    _xfade_ms = int(crossfade_ms)
                    if opening_theme_audio is not None and len(segments) >= 2:
                        _OTRL.append_transition(
                            _led,
                            from_line_id="opening_theme",
                            to_line_id="scene_audio",
                            crossfade_ms=_xfade_ms,
                            boundary_s=(
                                float(segments[0].shape[-1]) / float(sample_rate)
                            ),
                        )
                    if closing_theme_audio is not None and len(segments) >= 2:
                        _OTRL.append_transition(
                            _led,
                            from_line_id="scene_audio",
                            to_line_id="closing_theme",
                            crossfade_ms=_xfade_ms,
                            boundary_s=(
                                float(episode_waveform.shape[-1]) / float(sample_rate)
                                - float(segments[-1].shape[-1]) / float(sample_rate)
                            ),
                        )

                    if _shift_s > 0.0:
                        log.info(
                            "[EpisodeAssembler] BUG-106 master-mix shift: "
                            "+%.3fs applied to %d line(s) + %d clip(s)",
                            _shift_s, _shifted_lines, _shifted_clips,
                        )
                    _gc = _OTRL.lookup_git_commit(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    if _gc:
                        _OTRL.set_meta(_led, "git_commit", _gc)
                    _OTRL.save_ledger_safe(_ledger_p, _led)
                    log.info(
                        "[EpisodeAssembler] schema-l3 ledger update: "
                        "phase_ms=%d gate=post_episode_assembler dur=%.1fs",
                        _phase_ms, total_sec,
                    )
        except Exception as _meta_exc:
            log.warning(
                "[EpisodeAssembler] schema-l3 ledger update failed: %s", _meta_exc
            )

        # Path G Option D (Jeffrey 2026-05-18 follow-up to retest
        # #15): emit audio_done signal. Content is informational --
        # the final episode's length + sample rate + segment count
        # so downstream consumers (FLUX deferred loader's
        # gate_signal, primarily) can log the audio-branch outcome
        # at gate fire. Presence is what gates dependency: ComfyUI's
        # executor cannot fire any node taking this STRING as
        # forceInput until EpisodeAssembler.assemble() has returned.
        try:
            _final_waveform = audio_out.get("waveform") if isinstance(audio_out, dict) else None
            if _final_waveform is not None:
                _final_length_samples = int(_final_waveform.shape[-1])
            else:
                _final_length_samples = 0
            _final_sr = (
                audio_out.get("sample_rate")
                if isinstance(audio_out, dict)
                else sample_rate
            )
            _final_length_sec = _final_length_samples / max(1, int(_final_sr or 1))
        except Exception:  # noqa: BLE001
            _final_length_samples = 0
            _final_sr = sample_rate
            _final_length_sec = 0.0

        audio_done = (
            f"audio_done:"
            f"length_sec={_final_length_sec:.2f};"
            f"sample_rate={int(_final_sr or 0)};"
            f"length_samples={_final_length_samples};"
            f"segments={len(segments)}"
        )
        log.info(
            "[EpisodeAssembler] emit audio_done signal: %s", audio_done,
        )
        return (audio_out, output_path, info, audio_done)

    @staticmethod
    def _cue_from_batch(music_cue_audio, manifest, row):
        """Slice one cue out of the padded music batch as a standalone AUDIO
        dict (by manifest sample_count -- never the padded tail, MF-G). Returns
        None when the row is absent or the batch index is out of range."""
        if row is None:
            return None
        wf = (
            music_cue_audio.get("waveform")
            if isinstance(music_cue_audio, dict) else music_cue_audio
        )
        if wf is None:
            return None
        try:
            batch_index = int(row["batch_index"])
            sample_count = int(row["sample_count"])
        except (KeyError, TypeError, ValueError):
            return None
        if batch_index < 0 or batch_index >= int(wf.shape[0]):
            return None
        sliced = wf[batch_index:batch_index + 1, :, :sample_count]  # [1, C, sc]
        sliced = sliced.clone() if hasattr(sliced, "clone") else sliced
        return {
            "waveform": sliced,
            "sample_rate": int(manifest.get("sample_rate")),
        }

    def _extract_waveform(self, audio, target_sr=None):
        """Extract waveform tensor from AUDIO input, resampling to target_sr if needed.

        MusicGen emits at 32 kHz, Bark at 24 kHz, Kokoro at 24 kHz, and the
        main scene bus runs at 48 kHz. Themes coming in on the opening /
        closing inputs must be rate-matched before they can be concatenated
        with the main content, otherwise they play at the wrong speed and
        pitch. Resampling happens here at the concatenation boundary - the
        same pattern SceneSequencer uses for TTS clips via _resample_audio.
        """
        if isinstance(audio, dict):
            wf = audio.get("waveform")
            src_sr = int(audio.get("sample_rate") or 0) or None
        else:
            wf = audio
            src_sr = None
        if wf.dim() == 1:
            wf = wf.unsqueeze(0).unsqueeze(0)
        elif wf.dim() == 2:
            wf = wf.unsqueeze(0)

        if target_sr and src_sr and src_sr != target_sr:
            # Use torchaudio if available for high-quality resampling, fall
            # back to numpy-based _resample_audio (already imported in this
            # module for the TTS bus) for correctness without an extra dep.
            try:
                import torchaudio.functional as AF
                wf = AF.resample(wf, src_sr, target_sr)
            except Exception:
                import numpy as _np
                chans = []
                for c in range(wf.shape[1]):
                    clip_np = wf[0, c].cpu().numpy().astype(_np.float32)
                    resampled = _resample_audio(clip_np, src_sr, target_sr)
                    chans.append(torch.from_numpy(resampled).float())
                wf = torch.stack(chans, dim=0).unsqueeze(0)
            log.info("[EpisodeAssembler] Resampled theme %d Hz -> %d Hz",
                     src_sr, target_sr)
        return wf
