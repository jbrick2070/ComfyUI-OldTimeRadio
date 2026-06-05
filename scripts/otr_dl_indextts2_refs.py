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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=30, help="how many donations to download + classify")
    ap.add_argument("--want-female", type=int, default=4)
    ap.add_argument("--want-male", type=int, default=2)
    ap.add_argument("--seg-seconds", type=float, default=12.0)
    ap.add_argument("--refs-dir", default=DEFAULT_REFS_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())
