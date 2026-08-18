"""otr_dl_indextts2_refs.py -- expand the IndexTTS2 voice-reference bank with CC0
voices.

Pulls voices from kyutai/tts-voices `voice-donations/` (licensed CC0 -- volunteers
donated them to the Unmute Voice Donation Project for public-domain use), tags
each by acoustic gender (median F0 -- generic timbre, NOT speaker identity), trims
a clean mono segment, hashes it, copies it next to the existing refs, and wires a
`voice_reference_bank.json` entry. The goal is more FEMALE references so same-gender
characters stop sharing one voice.

Only CC0 is used. The other kyutai sets are skipped on purpose: vctk / cml-tts /
alba-mackenna are CC-BY (attribution), expresso / ears are CC-BY-NC (non-commercial).

NOTE on commercial use: these reference clips are CC0, so the bank stamps
commercial_clean=true for the REFERENCE. The IndexTTS2 *model* itself is
non-commercial (bilibili license), so effective commercial cleanliness is a
separate gate -- see docs/2026-06-05-voice-casting-architecture/pass01_plan.md.

Usage (ComfyUI venv python):
    python scripts\\otr_dl_indextts2_refs.py --dry-run        # plan only
    python scripts\\otr_dl_indextts2_refs.py --want-female 5 --want-male 2
    python scripts\\otr_dl_indextts2_refs.py --scan 40        # classify more to choose from
Idempotent: a voice_ref_id already in the bank is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(REPO, "config", "voice_reference_bank.json")
DEFAULT_REFS_DIR = os.path.join("C:\\", "ComfyUI-Models", "TTS", "refs", "indextts2")
HF_REPO = "kyutai/tts-voices"
FEMALE_F0_HZ = 165.0     # median voiced F0 at/above this -> tag female (generic, pitch-based)
OUT_RATE = 44100         # match the existing refs (mono PCM_16, ~8 s)
PICK_SEED = 20260605     # deterministic shuffle so re-runs scan the same order


def _to_mono(wav):
    import numpy as np
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype("float32")


# --------------------------------------------------------------------------- #
# GENDER: THE NAME BEATS THE PITCH (operator ruling 2026-08-18 -- hard).
#
# His words: *"so if we [don't] have real gender info use the name not some
# pitch"*, and *"common sense too, glenn as a name would probably be male"*.
#
# THIS IS NOT A PREFERENCE, IT IS A MEASURED CORRECTION. `vz_donor_glenn` was
# imported FEMALE, cast on 115 female slots for two and a half months, and
# corrected to MALE only when the operator heard it (`e1c84cf6`). The audit
# measures that clip at **median F0 = 261.5 Hz** -- not marginal, not near the
# 165 Hz line, but confidently inside the female range. So the pitch signal was
# not merely badly thresholded, it was WRONG WITH CONFIDENCE, and no threshold
# tuning would have saved it. A donor's own chosen handle is weak evidence, but
# it is evidence about a PERSON rather than about a waveform, and on the one
# case where the two disagreed and a human adjudicated, the handle was right.
#
# SCOPE, deliberately narrow: this reads a self-chosen donation HANDLE to bucket
# a synthetic voice reference for casting. It is not identification, it is not
# applied to any human being, and an unrecognised handle yields no opinion at
# all rather than a guess.
# --------------------------------------------------------------------------- #
#: Handles whose given name is unambiguous enough to overrule the pitch bucket.
#: Add only names that are decisive; an ambiguous one belongs in neither list
#: (`rup` is the standing example -- left unresolved on purpose 2026-08-18).
_HANDLE_MALE = frozenset({
    "james", "jim", "hillbilly_jim", "glenn", "bill", "mark", "phil", "peter",
    "stuart", "hugo", "daniel", "john", "tom", "sam", "bob", "frank", "carl",
})
_HANDLE_FEMALE = frozenset({
    "hannah", "kathleen", "kerstin", "linda", "caro", "andrea", "emma", "lily",
    "clara", "sarah", "mary", "anna",
})


def gender_from_handle(handle):
    """The donor handle's gender, or '' when the handle says nothing.

    Returns '' for hex handles (`0a67`, `1410`) and for genuinely ambiguous
    names -- the caller then falls back to the pitch bucket AND should flag the
    row for a listen. Never guesses.
    """
    stem = str(handle or "").strip().lower().replace("-", "_")
    if stem in _HANDLE_MALE:
        return "male"
    if stem in _HANDLE_FEMALE:
        return "female"
    for token in stem.split("_"):
        if token in _HANDLE_MALE:
            return "male"
        if token in _HANDLE_FEMALE:
            return "female"
    return ""


def classify_and_trim(path, seg_seconds=12.0):
    """Return (gender, median_f0, voiced_frac, trimmed_mono_at_OUT_RATE) or None.

    Gender from median F0 over voiced frames (librosa.pyin) -- a generic acoustic
    bucket for casting, not identification. Trims to the longest voiced span up to
    seg_seconds to avoid leading/trailing silence.
    """
    import numpy as np
    import soundfile as sf
    import librosa
    from scipy.signal import resample_poly

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = _to_mono(np.asarray(wav))
    if wav.size < sr:  # under 1 s -- unusable
        return None
    f0, voiced, _ = librosa.pyin(wav, sr=sr, fmin=65, fmax=400,
                                 frame_length=2048, hop_length=256)
    vmask = np.isfinite(f0) & (voiced if voiced is not None else np.isfinite(f0))
    if vmask.sum() < 10:
        return None
    med_f0 = float(np.nanmedian(f0[vmask]))
    voiced_frac = float(vmask.mean())
    gender = "female" if med_f0 >= FEMALE_F0_HZ else "male"
    # NOTE: this pitch bucket is now only a FALLBACK. `gender_from_handle`
    # overrides it whenever the donor's own handle names a person -- see the
    # operator ruling there. Kept because a hex-handled donation has no name to
    # read, and some bucket is better than none for an unnamed clip.

    # longest contiguous voiced run -> a clean span, capped at seg_seconds
    hop = 256
    idx = np.where(vmask)[0]
    best_s, best_e, run_s = 0, 0, idx[0] if idx.size else 0
    prev = idx[0] if idx.size else 0
    for i in idx[1:]:
        if i != prev + 1:
            if prev - run_s > best_e - best_s:
                best_s, best_e = run_s, prev
            run_s = i
        prev = i
    if prev - run_s > best_e - best_s:
        best_s, best_e = run_s, prev
    a = max(0, best_s * hop)
    b = min(wav.size, (best_e + 1) * hop)
    span = wav[a:b]
    if span.size > int(seg_seconds * sr):
        span = span[: int(seg_seconds * sr)]
    if span.size < int(2.0 * sr):  # too short after trimming
        span = wav[: int(seg_seconds * sr)]
    # resample to OUT_RATE mono
    if sr != OUT_RATE:
        from math import gcd
        g = gcd(int(sr), OUT_RATE)
        span = resample_poly(span, OUT_RATE // g, int(sr) // g).astype("float32")
    peak = float(np.max(np.abs(span))) or 1.0
    span = (span / peak * 0.97).astype("float32")
    return gender, med_f0, voiced_frac, span


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank():
    with open(BANK, "r", encoding="utf-8") as f:
        return json.load(f)


def save_bank(data):
    with open(BANK, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def list_donation_wavs():
    """CC0 voice-donation wavs (prefer the _enhanced/cleaned version of each)."""
    from huggingface_hub import list_repo_files
    files = list_repo_files(HF_REPO, repo_type="model")
    base = {}
    for f in files:
        if not (f.startswith("voice-donations/") and f.lower().endswith(".wav")):
            continue
        stem = f[len("voice-donations/"):]
        handle = stem[:-len("_enhanced.wav")] if stem.endswith("_enhanced.wav") else stem[:-4]
        enhanced = stem.endswith("_enhanced.wav")
        # prefer the enhanced (cleaned) version when both exist
        if handle not in base or enhanced:
            base[handle] = f
    return base  # {handle: repo_path}


# --------------------------------------------------------------------------- #
# Whiny-fix P2a: --audit -- acoustic report over the EXISTING bank (CPU, no
# downloads, no bank writes). Composite "must-listen" ranking only -- NEVER
# auto-rejects (tiers are stamped by the operator's blind reel, P2b).
# --------------------------------------------------------------------------- #
_QUIET_ORIGINALS = ("stuart_bell", "peter_yearsley", "bill_boerst", "caro_davy")
_SUSPECT_FEMALE_TAGS = ("vz_donor_james", "vz_donor_hillbilly_jim")


def _resolve_ref_audit_path(ref_path):
    """Resolve a bank ref_path to disk (mirrors the adapter's resolution)."""
    if os.path.isabs(ref_path) and os.path.exists(ref_path):
        return ref_path
    rp = ref_path.replace("\\", "/")
    stripped = rp[len("models/"):] if rp.startswith("models/") else rp
    for cand in (os.path.join("C:\\ComfyUI-Models", stripped),
                 os.path.join(os.path.dirname(REPO), rp),
                 os.path.abspath(ref_path)):
        if os.path.exists(cand):
            return cand
    return None


def _audit_one(path, gender_tag):
    """All P2a metrics for ONE reference WAV. Returns a dict (pure CPU)."""
    import numpy as np
    import soundfile as sf
    import librosa

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = _to_mono(np.asarray(wav))
    n = wav.size
    dur = n / float(sr)
    dc = float(np.mean(wav))
    rms = float(np.sqrt(np.mean(np.square(wav)))) or 1e-9
    peak = float(np.max(np.abs(wav))) or 1e-9
    crest = peak / rms
    clipping = int(np.sum(np.abs(wav) >= 0.985))

    f0, voiced, _ = librosa.pyin(wav, sr=sr, fmin=65, fmax=400,
                                 frame_length=2048, hop_length=256)
    vmask = np.isfinite(f0) & (voiced if voiced is not None else np.isfinite(f0))
    voiced_frac = float(vmask.mean()) if vmask.size else 0.0
    med_f0 = float(np.nanmedian(f0[vmask])) if vmask.sum() else 0.0
    f0_std = float(np.nanstd(f0[vmask])) if vmask.sum() else 0.0

    # end-of-line rise: median voiced F0 of the last 500 ms vs the body
    hop = 256
    tail_frames = int(0.5 * sr / hop)
    rise_semitones = 0.0
    if vmask.sum() > tail_frames:
        tail_mask = vmask.copy()
        tail_mask[:-tail_frames] = False
        body_mask = vmask & ~tail_mask
        if tail_mask.sum() >= 3 and body_mask.sum() >= 3:
            tail_f0 = float(np.nanmedian(f0[tail_mask]))
            body_f0 = float(np.nanmedian(f0[body_mask]))
            if tail_f0 > 0 and body_f0 > 0:
                rise_semitones = 12.0 * float(np.log2(tail_f0 / body_f0))

    # average magnitude spectrum -> centroid, slope, band ratios (voiced-ish
    # proxy: whole file; the refs are speech-dense by construction)
    spec = np.abs(np.fft.rfft(wav * np.hanning(n))) if n else np.zeros(1)
    freqs = np.fft.rfftfreq(n, 1.0 / sr) if n else np.zeros(1)
    total = float(np.sum(spec)) or 1e-9
    centroid = float(np.sum(freqs * spec) / total)
    band = (freqs >= 200) & (freqs <= 4000)
    slope = 0.0
    if band.sum() > 10:
        x = np.log10(freqs[band])
        y = 20.0 * np.log10(spec[band] + 1e-12)
        slope = float(np.polyfit(x, y, 1)[0])          # dB per decade
    lo, hi = (120, 350) if gender_tag != "female" else (180, 450)
    chest = (freqs >= lo) & (freqs <= hi)
    chest_ratio = float(np.sum(spec[chest]) / total)
    nasal = (freqs >= 1000) & (freqs <= 2500)
    nasal_ratio = float(np.sum(spec[nasal]) / total)

    # frame RMS stats: breath tail + pause-SNR proxy
    frame = 1024
    nf = max(1, n // frame)
    frames = wav[: nf * frame].reshape(nf, frame)
    frms = np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9
    body_rms = float(np.median(frms))
    floor_rms = float(np.percentile(frms, 10))
    pause_snr_db = 20.0 * float(np.log10(body_rms / floor_rms))
    tail_n = int(0.2 * sr)
    tail_rms = float(np.sqrt(np.mean(np.square(wav[-tail_n:])))) if n > tail_n else rms
    breath_tail_ratio = tail_rms / (rms or 1e-9)

    return {
        "duration_s": round(dur, 2), "sample_rate": int(sr),
        "dc_offset": round(dc, 5), "rms": round(rms, 4),
        "rms_db": round(20.0 * float(np.log10(rms)), 1),
        "crest_factor": round(crest, 2), "clipping_count": clipping,
        "median_f0_hz": round(med_f0, 1), "f0_std_hz": round(f0_std, 1),
        "voiced_fraction": round(voiced_frac, 3),
        "final_rise_semitones": round(rise_semitones, 2),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_slope_db_per_decade": round(slope, 1),
        "chest_weight_ratio": round(chest_ratio, 4),
        "nasal_band_ratio": round(nasal_ratio, 4),
        "pause_snr_db": round(pause_snr_db, 1),
        "breath_tail_ratio": round(breath_tail_ratio, 3),
    }


def audit(args):
    """--audit: metric report + composite must-listen ranking over the bank."""
    bank = load_bank()
    rows = []
    for v in bank["voices"]:
        if v.get("engine") != args.audit_engine:
            continue
        vid = v["voice_ref_id"]
        path = _resolve_ref_audit_path(v.get("ref_path") or "")
        if not path:
            rows.append({"voice_ref_id": vid, "error": "ref WAV not found on disk",
                         "ref_path": v.get("ref_path")})
            continue
        try:
            m = _audit_one(path, str(v.get("gender") or ""))
        except Exception as exc:  # noqa: BLE001 -- report, never abort the audit
            rows.append({"voice_ref_id": vid, "error": repr(exc)})
            continue
        flags = []
        if 145.0 <= m["median_f0_hz"] <= 185.0:
            flags.append("gender_ambiguous_f0")
        if m["rms"] < 0.07:
            flags.append("low_rms_unnormalized")
        if any(q in vid for q in _QUIET_ORIGINALS):
            flags.append("quiet_original_g18")
        if vid in _SUSPECT_FEMALE_TAGS and v.get("gender") == "female":
            flags.append("suspect_female_tag")
        if m["final_rise_semitones"] > 2.0:
            flags.append("terminal_rise_in_ref")
        if m["clipping_count"] > 50:
            flags.append("clipping")
        # Composite must-listen score (BIGGER = listen sooner). Heuristic
        # whine-risk weighting; ranking only -- tiers come from the reel.
        score = (
            max(0.0, 0.10 - m["rms"]) * 40.0
            + max(0.0, m["final_rise_semitones"]) * 0.5
            + max(0.0, 0.05 - m["chest_weight_ratio"]) * 20.0
            + m["nasal_band_ratio"] * 5.0
            + (1.0 if m["clipping_count"] > 50 else 0.0)
            + max(0.0, 20.0 - m["pause_snr_db"]) * 0.05
        )
        rows.append({"voice_ref_id": vid, "gender": v.get("gender"),
                     "quality_tier": v.get("quality_tier") or "",
                     "must_listen_score": round(score, 3),
                     "flags": flags, **m})
    ranked = sorted(
        (r for r in rows if "error" not in r),
        key=lambda r: -r["must_listen_score"])
    print(f"AUDIT: {len(rows)} {args.audit_engine} refs "
          f"({sum(1 for r in rows if 'error' in r)} unreadable)")
    print(f"{'rank':<5}{'voice_ref_id':<28}{'score':<8}{'rms':<7}"
          f"{'f0':<7}{'rise':<7}flags")
    for i, r in enumerate(ranked, 1):
        print(f"{i:<5}{r['voice_ref_id']:<28}{r['must_listen_score']:<8}"
              f"{r['rms']:<7}{r['median_f0_hz']:<7}"
              f"{r['final_rise_semitones']:<7}{','.join(r['flags']) or '-'}")
    for r in rows:
        if "error" in r:
            print(f"ERROR {r['voice_ref_id']}: {r['error']}")
    out = args.audit_json
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"engine": args.audit_engine, "rows": rows}, f, indent=1)
    print(f"full report -> {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=30, help="how many donations to download + classify")
    ap.add_argument("--want-female", type=int, default=4)
    ap.add_argument("--want-male", type=int, default=2)
    ap.add_argument("--seg-seconds", type=float, default=12.0)
    ap.add_argument("--refs-dir", default=DEFAULT_REFS_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="P2a: acoustic audit of the EXISTING bank (no downloads, "
                         "no bank writes); composite must-listen ranking")
    ap.add_argument("--audit-engine", default="indextts2")
    ap.add_argument("--audit-json",
                    default=os.path.join(REPO, "scripts", "_otr_ref_audit_report.json"))
    ap.add_argument("--source", choices=["donations", "lj-speech", "rhasspy"], default="donations",
                    help="donations = kyutai CC0 voice-donations; lj-speech = LJ Speech (PD US female); "
                         "rhasspy = Rhasspy Kathleen + Kerstin (CC0, GitHub sparse-checkout)")
    args = ap.parse_args()

    if args.audit:
        return audit(args)

    if args.source == "lj-speech":
        return add_lj_speech(args)
    if args.source == "rhasspy":
        return add_rhasspy(args)

    from huggingface_hub import hf_hub_download

    donations = list_donation_wavs()
    handles = sorted(donations)
    random.Random(PICK_SEED).shuffle(handles)
    bank = load_bank()
    existing_ids = {v["voice_ref_id"] for v in bank["voices"]}
    print(f"voice-donations available (CC0): {len(handles)}; bank has "
          f"{sum(1 for v in bank['voices'] if v['engine'] == 'indextts2')} indextts2 refs")
    if args.dry_run:
        print(f"[dry-run] would scan up to {args.scan} donations, classify by F0, and add "
              f"up to {args.want_female} female + {args.want_male} male CC0 refs.")
        print("first 8 candidate handles:", handles[:8])
        return 0

    os.makedirs(args.refs_dir, exist_ok=True)
    picked = {"female": [], "male": []}
    scanned = 0
    for handle in handles:
        if scanned >= args.scan:
            break
        if len(picked["female"]) >= args.want_female and len(picked["male"]) >= args.want_male:
            break
        vid = "vz_donor_" + "".join(c for c in handle if c.isalnum() or c in "_-").lower()[:40]
        if vid in existing_ids:
            continue
        scanned += 1
        try:
            src = hf_hub_download(HF_REPO, donations[handle], repo_type="model")
            res = classify_and_trim(src, args.seg_seconds)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {handle}: {e!r}")
            continue
        if res is None:
            print(f"  skip {handle}: unclassifiable")
            continue
        gender, med_f0, vfrac, span = res
        # THE NAME BEATS THE PITCH (operator ruling 2026-08-18). An unnamed or
        # ambiguous handle returns '' and the pitch bucket stands, but it is
        # announced as UNCONFIRMED so the row can be listened to rather than
        # trusted -- pitch was confidently wrong on glenn at 261.5 Hz.
        named_gender = gender_from_handle(handle)
        if named_gender and named_gender != gender:
            print(f"  NAME OVERRIDES PITCH for {handle}: "
                  f"f0={med_f0:.0f}Hz said {gender}, handle says {named_gender}")
            gender = named_gender
        elif not named_gender:
            print(f"  unconfirmed gender for {handle}: pitch bucket {gender} "
                  f"(f0={med_f0:.0f}Hz), handle gives no signal -- needs a listen")
        if len(picked[gender]) >= (args.want_female if gender == "female" else args.want_male):
            print(f"  pass {handle}: {gender} f0={med_f0:.0f} (quota full)")
            continue
        import soundfile as sf
        out_wav = os.path.join(args.refs_dir, vid + ".wav")
        sf.write(out_wav, span, OUT_RATE, subtype="PCM_16")
        sha = _sha256_file(out_wav)
        entry = {
            "voice_ref_id": vid, "engine": "indextts2", "gender": gender,
            "timbre": ["donated", "bright" if gender == "female" else "warm"],
            "roles": ["char_voice"], "age_band": "adult",
            "ref_path": "models/TTS/refs/indextts2/" + vid + ".wav",
            "ref_sha256": sha, "commercial_clean": True,
        }
        bank["voices"].append(entry)
        existing_ids.add(vid)
        picked[gender].append((vid, med_f0, span.size / OUT_RATE))
        print(f"  ADD {vid}: {gender} f0={med_f0:.0f}Hz voiced={vfrac:.2f} "
              f"dur={span.size/OUT_RATE:.1f}s sha={sha[:8]}")

    if picked["female"] or picked["male"]:
        save_bank(bank)
        print(f"\nwrote {len(picked['female'])} female + {len(picked['male'])} male CC0 refs "
              f"to {BANK}")
        print("RESTART ComfyUI to load the expanded bank.")
    else:
        print("\nnothing added (quota already met or no new classifiable voices).")
    return 0


def add_lj_speech(args):
    """Add one clean reference from LJ Speech (keithito/lj_speech, public domain,
    single US female narrator). Streams a few consecutive clips into one ~seg-second
    continuous span so the cloner has natural prosody. Idempotent."""
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    from scipy.signal import resample_poly

    bank = load_bank()
    if any(v["voice_ref_id"] == "vz_ljspeech" for v in bank["voices"]):
        print("vz_ljspeech already in bank; nothing to do.")
        return 0
    import io
    from datasets import Audio
    print("streaming MikhailT/lj-speech [full] (PD)...")
    ds = load_dataset("MikhailT/lj-speech", split="full", streaming=True)
    # decode=False -> raw WAV bytes, decoded with soundfile (avoids the torchcodec
    # audio backend that this stack deliberately does not use).
    ds = ds.cast_column("audio", Audio(decode=False))
    chunks, sr0, total = [], None, 0.0
    for s in ds:
        raw = s["audio"]
        data, sr0 = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
        if getattr(data, "ndim", 1) == 2:
            data = data.mean(axis=1)
        chunks.append(np.asarray(data, dtype="float32"))
        total += len(chunks[-1]) / int(sr0)
        if total >= args.seg_seconds:
            break
    wav = np.concatenate(chunks)[: int(args.seg_seconds * sr0)]
    if sr0 != OUT_RATE:
        from math import gcd
        g = gcd(int(sr0), OUT_RATE)
        wav = resample_poly(wav, OUT_RATE // g, int(sr0) // g).astype("float32")
    peak = float(np.max(np.abs(wav))) or 1.0
    wav = (wav / peak * 0.97).astype("float32")
    os.makedirs(args.refs_dir, exist_ok=True)
    out_wav = os.path.join(args.refs_dir, "vz_ljspeech.wav")
    sf.write(out_wav, wav, OUT_RATE, subtype="PCM_16")
    sha = _sha256_file(out_wav)
    bank["voices"].append({
        "voice_ref_id": "vz_ljspeech", "engine": "indextts2", "gender": "female",
        "timbre": ["clear", "narration"], "roles": ["char_voice"], "age_band": "adult",
        "ref_path": "models/TTS/refs/indextts2/vz_ljspeech.wav",
        "ref_sha256": sha, "commercial_clean": True,
    })
    save_bank(bank)
    print(f"ADD vz_ljspeech: female (LJ Speech, PD US narrator) "
          f"dur={len(wav)/OUT_RATE:.1f}s sha={sha[:8]}")
    print("RESTART ComfyUI to load the expanded bank.")
    return 0


# Rhasspy CC0 voice datasets (GitHub). Known gender, so no F0 tagging needed.
RHASSPY_VOICES = [
    ("kathleen", "female", "https://github.com/rhasspy/dataset-voice-kathleen.git"),
    ("kerstin", "female", "https://github.com/rhasspy/dataset-voice-kerstin.git"),
]


def add_rhasspy(args):
    """Add Rhasspy Kathleen (CC0 US female) + Kerstin (CC0 German female) by
    sparse-checking out a handful of clips from each GitHub repo (no full clone),
    concatenating to a ~seg-second reference. Idempotent."""
    import subprocess
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    bank = load_bank()
    existing = {v["voice_ref_id"] for v in bank["voices"]}
    tmp_base = os.path.join(os.path.dirname(REPO), "_otr_voice_tmp")
    os.makedirs(tmp_base, exist_ok=True)
    os.makedirs(args.refs_dir, exist_ok=True)
    added = []
    for name, gender, url in RHASSPY_VOICES:
        vid = "vz_" + name
        if vid in existing:
            print(f"{vid} already in bank; skip")
            continue
        d = os.path.join(tmp_base, name)
        if not os.path.isdir(os.path.join(d, ".git")):
            print(f"cloning {name} (blob-less)...")
            subprocess.run(["git", "clone", "--no-checkout", "--depth", "1",
                            "--filter=blob:none", url, d], check=True)
        tree = subprocess.run(["git", "-C", d, "ls-tree", "-r", "--name-only", "HEAD"],
                              capture_output=True, text=True).stdout.splitlines()
        auds = [f for f in tree if f.lower().endswith((".wav", ".flac"))]
        auds = [f for f in auds if f.lower().startswith("data/")] or auds
        auds = sorted(auds)[:6]
        if not auds:
            print(f"{vid}: no audio files in repo; skip")
            continue
        subprocess.run(["git", "-C", d, "sparse-checkout", "init", "--no-cone"], check=False)
        subprocess.run(["git", "-C", d, "sparse-checkout", "set", *auds], check=False)
        subprocess.run(["git", "-C", d, "checkout"], check=False,
                       capture_output=True, text=True)
        chunks, sr0, total = [], None, 0.0
        for rel in auds:
            fp = os.path.join(d, rel.replace("/", os.sep))
            if not os.path.exists(fp):
                continue
            data, sr0 = sf.read(fp, dtype="float32")
            if getattr(data, "ndim", 1) == 2:
                data = data.mean(axis=1)
            chunks.append(np.asarray(data, dtype="float32"))
            total += len(chunks[-1]) / int(sr0)
            if total >= args.seg_seconds:
                break
        if not chunks:
            print(f"{vid}: fetched no audio; skip")
            continue
        wav = np.concatenate(chunks)[: int(args.seg_seconds * sr0)]
        if int(sr0) != OUT_RATE:
            from math import gcd
            g = gcd(int(sr0), OUT_RATE)
            wav = resample_poly(wav, OUT_RATE // g, int(sr0) // g).astype("float32")
        peak = float(np.max(np.abs(wav))) or 1.0
        wav = (wav / peak * 0.97).astype("float32")
        out_wav = os.path.join(args.refs_dir, vid + ".wav")
        sf.write(out_wav, wav, OUT_RATE, subtype="PCM_16")
        sha = _sha256_file(out_wav)
        bank["voices"].append({
            "voice_ref_id": vid, "engine": "indextts2", "gender": gender,
            "timbre": ["clear", "narration"], "roles": ["char_voice"],
            "age_band": "adult",
            "ref_path": "models/TTS/refs/indextts2/" + vid + ".wav",
            "ref_sha256": sha, "commercial_clean": True,
        })
        existing.add(vid)
        added.append(vid)
        print(f"ADD {vid}: {gender} (Rhasspy, CC0) dur={len(wav)/OUT_RATE:.1f}s sha={sha[:8]}")
    if added:
        save_bank(bank)
        print(f"\nwrote {len(added)} Rhasspy CC0 ref(s): {added}")
        print("RESTART ComfyUI to load the expanded bank.")
    else:
        print("\nnothing added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
