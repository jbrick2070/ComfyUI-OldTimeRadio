"""S-C C1 -- per-beat ``audio_motion_profile`` EXTRACTION (analysis only).

Computes a small, deterministic, READ-ONLY feature profile from a conditioning
WAV (a per-beat slice of the FROZEN master mix, or a per-line TTS clip). There
is NO consumer yet -- C2 (per-engine consumers + HuMo phrase-chunking) is
deferred per the coverage-soak sprint plan. The whole point of C1 is the
extraction + ledger stamp, and to PROVE the extraction path never touches
audio: this module opens WAVs READ-ONLY (soundfile) and never writes an audio
file, so the frozen master (invariant V-1) stays byte-identical.

Import-clean: stdlib + numpy + soundfile ONLY (no torch, no ComfyUI, no cv2), so
it costs nothing at custom-node import and runs in the headless test venv.

The 8-field profile contract (``PROFILE_FIELDS``) is stable so the deferred C2
consumers + tests can rely on it:

  duration_s        clip length in seconds
  rms_dbfs          full-clip RMS level in dBFS (loudness)
  peak_dbfs         full-clip sample peak in dBFS
  dynamic_range_db  robust loud-vs-quiet spread (p95 - p5 of non-silent frames)
  silence_ratio     fraction of 20 ms frames below the silence floor (0..1)
  onset_s           time of the first supra-threshold frame; -1.0 if never
  brightness        normalized spectral centroid (0..1; high = bright/treble)
  speech_vs_music   deterministic v1 heuristic label: speech | music | mixed
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("OTR")

#: Bumps when the feature recipe changes (thresholds / added fields) so a
#: downstream consumer or a stamped ledger can tell which extractor ran.
PROFILE_VERSION = "amp-1"

#: The stable field contract. Every profile dict carries EXACTLY these keys
#: (plus ``ok``/``reason`` bookkeeping on the row wrapper, never inside here).
PROFILE_FIELDS = (
    "duration_s", "rms_dbfs", "peak_dbfs", "dynamic_range_db",
    "silence_ratio", "onset_s", "brightness", "speech_vs_music",
)

_SILENCE_DBFS = -45.0     # 20 ms frames quieter than this count as silence
_ONSET_DBFS = -35.0       # first frame louder than this = onset
_FRAME_S = 0.020          # 20 ms analysis frame
_DBFS_FLOOR = -120.0      # clamp for log10 of (near-)zero amplitude


def _dbfs(amp: float) -> float:
    """Amplitude (0..1 full-scale) -> dBFS, floored so silence is finite."""
    a = float(amp)
    if not math.isfinite(a) or a <= 0.0:
        return _DBFS_FLOOR
    return max(_DBFS_FLOOR, 20.0 * math.log10(a))


def _read_wav(path: str) -> Optional["tuple[np.ndarray, int]"]:
    """Read a WAV READ-ONLY into a mono float32 array + sample rate.

    Returns ``None`` (LOUD warning) on any failure -- extraction is best-effort
    analysis and must never crash the pipeline. NEVER writes to ``path``."""
    try:
        import soundfile as sf  # lazy: keep module import torch/comfy-free
    except Exception as exc:  # noqa: BLE001
        log.warning("[audio_motion] soundfile unavailable: %s", exc)
        return None
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("[audio_motion] read failed (%s): %s", path, exc)
        return None
    if data.size == 0 or int(sr) <= 0:
        return None
    mono = data.mean(axis=1).astype("float32", copy=False)  # downmix
    return mono, int(sr)


def _frame_dbfs(samples: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame RMS level (dBFS) over non-overlapping 20 ms frames."""
    n = int(max(1, round(_FRAME_S * sr)))
    total = samples.shape[0]
    if total < n:
        rms = float(np.sqrt(np.mean(np.square(samples)))) if total else 0.0
        return np.array([_dbfs(rms)], dtype="float64")
    nfr = total // n
    trimmed = samples[: nfr * n].reshape(nfr, n)
    rms = np.sqrt(np.mean(np.square(trimmed.astype("float64")), axis=1))
    return np.array([_dbfs(float(r)) for r in rms], dtype="float64")


def _brightness(samples: np.ndarray, sr: int) -> float:
    """Normalized spectral centroid (0..1). 0 = all energy at DC, 1 = Nyquist."""
    n = samples.shape[0]
    if n < 2:
        return 0.0
    mag = np.abs(np.fft.rfft(samples.astype("float64")))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    denom = float(mag.sum())
    if denom <= 0.0:
        return 0.0
    centroid = float((freqs * mag).sum() / denom)
    nyq = float(sr) / 2.0
    if nyq <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, centroid / nyq)))


def _zero_crossing_rate(samples: np.ndarray) -> float:
    """Fraction of adjacent samples that change sign (0..1)."""
    if samples.shape[0] < 2:
        return 0.0
    signs = np.sign(samples)
    signs[signs == 0] = 1.0
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def _speech_vs_music(silence_ratio: float, zcr: float, brightness: float) -> str:
    """Deterministic v1 heuristic (no ML): speech has more pauses + a burstier
    envelope; sustained low-silence energy reads as music. Documented as v1 --
    C2 will refine it against real clips; nothing consumes it yet."""
    if silence_ratio >= 0.35:
        return "speech"
    if silence_ratio <= 0.10 and zcr <= 0.12:
        return "music"
    return "mixed"


def analyze_samples(samples: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute the 8-field profile from a mono float32 array. Deterministic."""
    samples = np.asarray(samples, dtype="float32").ravel()
    n = samples.shape[0]
    sr = int(sr)
    duration_s = (n / sr) if sr > 0 else 0.0

    peak = float(np.max(np.abs(samples))) if n else 0.0
    rms_full = float(np.sqrt(np.mean(np.square(samples.astype("float64"))))) if n else 0.0

    fdb = _frame_dbfs(samples, sr) if n else np.array([_DBFS_FLOOR])
    silent = fdb < _SILENCE_DBFS
    silence_ratio = float(np.mean(silent)) if fdb.size else 1.0

    # onset: first frame above the onset floor.
    above = np.where(fdb > _ONSET_DBFS)[0]
    onset_s = float(above[0] * _FRAME_S) if above.size else -1.0

    # dynamic range: robust spread of the non-silent frames.
    voiced = fdb[~silent]
    if voiced.size >= 2:
        dynamic_range_db = float(np.percentile(voiced, 95) - np.percentile(voiced, 5))
    else:
        dynamic_range_db = 0.0

    brightness = _brightness(samples, sr) if n else 0.0
    zcr = _zero_crossing_rate(samples) if n else 0.0
    label = _speech_vs_music(silence_ratio, zcr, brightness)

    return {
        "duration_s": round(duration_s, 6),
        "rms_dbfs": round(_dbfs(rms_full), 3),
        "peak_dbfs": round(_dbfs(peak), 3),
        "dynamic_range_db": round(dynamic_range_db, 3),
        "silence_ratio": round(silence_ratio, 4),
        "onset_s": round(onset_s, 4) if onset_s >= 0 else -1.0,
        "brightness": round(brightness, 4),
        "speech_vs_music": label,
    }


def analyze_wav(path: str) -> Optional[Dict[str, float]]:
    """Read a WAV READ-ONLY and return its 8-field profile, or ``None`` on
    failure (LOUD warning). Never mutates or writes the file."""
    got = _read_wav(path)
    if got is None:
        return None
    samples, sr = got
    return analyze_samples(samples, sr)


def build_audio_motion_profiles(
    rows: Sequence[Dict],
    resolver: Callable[[Dict], Optional[str]],
) -> List[Dict]:
    """Build one profile row per input ``row`` (a beat/line dict carrying an
    ``id``/``beat_id``/``line_id`` + ``start_s``/``dur_s``).

    ``resolver(row)`` returns the on-disk conditioning WAV path for that row (a
    read-only slice of the frozen master, or a per-line clip) or ``None``. The
    returned rows carry: the source id + timing, ``profile_version``, ``ok``,
    an optional ``reason`` when unresolved/unreadable, and -- when ``ok`` -- the
    8 ``PROFILE_FIELDS``. Deterministic and side-effect-free (no disk writes)."""
    out: List[Dict] = []
    for row in rows or []:
        rid = (row.get("beat_id") or row.get("line_id") or row.get("id") or "")
        rec: Dict = {
            "id": str(rid),
            "start_s": float(row.get("start_s") or 0.0),
            "dur_s": float(row.get("dur_s") or 0.0),
            "profile_version": PROFILE_VERSION,
        }
        wav = None
        try:
            wav = resolver(row)
        except Exception as exc:  # noqa: BLE001 -- best-effort analysis
            rec["ok"] = False
            rec["reason"] = f"resolver error: {type(exc).__name__}: {exc}"
            out.append(rec)
            continue
        if not wav:
            rec["ok"] = False
            rec["reason"] = "no conditioning wav resolved"
            out.append(rec)
            continue
        prof = analyze_wav(wav)
        if prof is None:
            rec["ok"] = False
            rec["reason"] = "wav unreadable"
            out.append(rec)
            continue
        rec["ok"] = True
        rec.update(prof)
        out.append(rec)
    return out


def stamp_ledger_audio_motion(ledger: Dict, profiles: List[Dict]) -> Dict:
    """Stamp ``ledger['audio_motion_profiles']`` (a NEW top-level section) with
    the computed rows + a small meta block. Mutates + returns the same dict;
    touches ONLY this section (never audio, never meta.paths). The caller owns
    the durable save (``save_ledger_safe``)."""
    if not isinstance(ledger, dict):
        raise TypeError("stamp_ledger_audio_motion: ledger must be a dict")
    ledger["audio_motion_profiles"] = list(profiles or [])
    meta = ledger.setdefault("meta", {})
    meta["audio_motion_profile"] = {
        "version": PROFILE_VERSION,
        "count": len(profiles or []),
        "ok_count": sum(1 for p in (profiles or []) if p.get("ok")),
        "fields": list(PROFILE_FIELDS),
    }
    return ledger


__all__ = [
    "PROFILE_VERSION", "PROFILE_FIELDS",
    "analyze_samples", "analyze_wav",
    "build_audio_motion_profiles", "stamp_ledger_audio_motion",
]
