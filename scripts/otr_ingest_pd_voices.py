r"""Ingest PUBLIC-DOMAIN voice reference clips into the OTR voice bank.

ONE distinct VOICE per SPEAKER (not per clip): LJSpeech is a single narrator and a
LibriVox book is one reader, so a speaker's clips are CONCATENATED into one richer
~25 s reference -- registering each clip as its own "voice" would hand the caster
duplicate voices. Each voice is MIRRORED across the cloner engines -- indextts2
(vz_), chatterbox (cb_), dia (dia_) -- one physical WAV, three bank entries
(matching the existing convention). kokoro is NOT a cloner (built-in .pt voices),
so reference clips do not apply to it.

Pipeline per voice:
  1. download each clip (WAV direct, or LibriVox/archive MP3),
  2. NORMALIZE each with ffmpeg -> mono 24 kHz s16 (+ gentle band-pass + loudnorm,
     the operator's recipe); an MP3 chapter is trimmed (skip the ~25 s LibriVox
     spoken intro, take 25 s),
  3. CONCATENATE the speaker's normalized clips and CAP at 25 s -> the ref WAV at
     <models-base>/TTS/refs/indextts2/vz_<voice_id>.wav,
  4. sha256 + three bank entries (vz_/cb_/dia_), commercial_clean=True,
  5. merge into config/voice_reference_bank.json (dedup), re-validate the bank.

USAGE (the OPERATOR runs this -- it does the network downloads + ffmpeg):
  <venv>\python.exe scripts\otr_ingest_pd_voices.py
  <venv>\python.exe scripts\otr_ingest_pd_voices.py --dry-run
  <venv>\python.exe scripts\otr_ingest_pd_voices.py --models-base C:\ComfyUI-Models

For more male / elder-female / androgynous variety, add more PD_VOICE_SPEAKERS --
one entry per DISTINCT narrator (different LibriVox readers / books). Idempotent:
a voice whose ref WAV + bank entries already exist is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

BANK_JSON = os.path.join(_REPO, "config", "voice_reference_bank.json")
_CLONER_ENGINES = (("indextts2", "vz"), ("chatterbox", "cb"), ("dia", "dia"))
_REF_REL_DIR = "models/TTS/refs/indextts2"
_REF_CAP_SECONDS = 25


# --- operator-supplied public-domain speakers (ONE distinct voice each) ---------
PD_VOICE_SPEAKERS = [
    {
        "voice_id": "pd_ljspeech_linda_johnson", "gender": "female",
        "age_band": "adult", "timbre": ["neutral"],
        "source": "LJ Speech / LibriVox public domain (single female narrator)",
        "clips": [
            {"kind": "wav", "url": f"https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/wavs/LJ001-{n:04d}.wav?download=true"}
            for n in (1, 3, 5, 7, 9, 10, 12, 14, 15, 17)
        ],
    },
    {
        "voice_id": "pd_librivox_mark_f_smith", "gender": "male",
        "age_band": "adult", "timbre": ["warm"],
        "source": "LibriVox public domain (Mark F. Smith, Around the World in 80 Days)",
        "clips": [
            {"kind": "mp3", "url": "https://archive.org/download/around_world_80_days_mfs_librivox/around_world_in_80_days_01_verne.mp3"},
        ],
    },
    {
        # Same narrator, his RESONANT/grandfatherly delivery -> an elder-leaning
        # clone ref (operator pick). NOTE: still male; the FEMALE-elder gap stays
        # open until a female elder PD source is added.
        "voice_id": "pd_librivox_mark_f_smith_elder", "gender": "male",
        "age_band": "elder", "timbre": ["warm", "resonant"],
        "source": "LibriVox public domain (Mark F. Smith, The Mysterious Island)",
        "clips": [
            {"kind": "mp3", "url": "https://archive.org/download/mysterious_island_ms_librivox/mysterious_island_1_01_verne.mp3"},
        ],
    },
    {
        # A genuinely DISTINCT male narrator (crisp, mid-century space-age delivery
        # -- a great fit for the OTR sci-fi engines).
        "voice_id": "pd_librivox_phil_chenevert", "gender": "male",
        "age_band": "adult", "timbre": ["crisp", "bright"],
        "source": "LibriVox public domain (Phil Chenevert, Search the Sky -- Pohl/Kornbluth)",
        # Real archive.org filenames verified via the item metadata API. The
        # operator's search_the_sky_01_pohl_kornbluth.mp3 / howtoliveon... paths
        # 404; the actual files are searchthesky_NN_pohl.mp3 (read by Phil
        # Chenevert, public-domain mark 1.0).
        "clips": [
            {"kind": "mp3", "url": "https://archive.org/download/search_the_sky_1606_librivox/searchthesky_01_pohl.mp3"},
            {"kind": "mp3", "url": "https://archive.org/download/search_the_sky_1606_librivox/searchthesky_02_pohl.mp3"},
        ],
    },
]


def _ffmpeg() -> str:
    return os.environ.get("OTR_FFMPEG", "ffmpeg")


def _normalize_clip(src: str, dest: str, *, is_long: bool) -> None:
    """One clip -> mono 24 kHz s16 WAV (+ band-pass + loudnorm). A long MP3 chapter
    is trimmed: skip 25 s (LibriVox intro) and take 25 s."""
    cmd = [_ffmpeg(), "-y"]
    if is_long:
        cmd += ["-ss", "25", "-t", "25"]
    cmd += ["-i", src, "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
            "-af", "highpass=f=80,lowpass=f=12000,loudnorm=I=-23:LRA=7:TP=-2", dest]
    subprocess.run(cmd, check=True, capture_output=True)


def _concat_cap(wavs: list, dest: str) -> None:
    """Concatenate same-format WAVs and cap at _REF_CAP_SECONDS -> dest."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if len(wavs) == 1:
        cmd = [_ffmpeg(), "-y", "-i", wavs[0], "-t", str(_REF_CAP_SECONDS),
               "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", dest]
        subprocess.run(cmd, check=True, capture_output=True)
        return
    with tempfile.TemporaryDirectory() as td:
        listfile = os.path.join(td, "list.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for w in wavs:
                f.write(f"file '{w.replace(chr(92), '/')}'\n")
        cmd = [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", listfile,
               "-t", str(_REF_CAP_SECONDS), "-ac", "1", "-ar", "24000",
               "-sample_fmt", "s16", dest]
        subprocess.run(cmd, check=True, capture_output=True)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "otr-pd-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_entries(spec: dict, ref_rel_path: str, sha: str) -> list:
    """Three bank entries (one per cloner engine) for one PD voice. Pure."""
    base = str(spec["voice_id"])
    out = []
    for engine, prefix in _CLONER_ENGINES:
        out.append({
            "voice_ref_id": f"{prefix}_{base}",
            "engine": engine,
            "gender": str(spec["gender"]),
            "timbre": list(spec.get("timbre") or []),
            "roles": ["char_voice"],
            "age_band": str(spec.get("age_band") or "adult"),
            "ref_path": ref_rel_path,
            "ref_sha256": sha,
            "commercial_clean": True,
        })
    return out


def merge_into_bank(bank_json_path: str, new_entries: list) -> tuple:
    with open(bank_json_path, encoding="utf-8") as f:
        data = json.load(f)
    voices = data.setdefault("voices", [])
    have = {v.get("voice_ref_id") for v in voices}
    added, skipped = 0, 0
    for e in new_entries:
        if e["voice_ref_id"] in have:
            skipped += 1
            continue
        voices.append(e)
        have.add(e["voice_ref_id"])
        added += 1
    if added:
        with open(bank_json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
            f.write("\n")
    return added, skipped


def _ingest_speaker(spec: dict, models_base: str) -> list:
    vz_name = f"vz_{spec['voice_id']}"
    rel_path = f"{_REF_REL_DIR}/{vz_name}.wav"
    abs_wav = os.path.join(models_base, rel_path.replace("models/", "", 1))
    if not os.path.isfile(abs_wav):
        with tempfile.TemporaryDirectory() as td:
            norm: list = []
            for i, clip in enumerate(spec["clips"]):
                raw = os.path.join(
                    td, f"raw{i}" + (".mp3" if clip["kind"] == "mp3" else ".wav"))
                _download(clip["url"], raw)
                n = os.path.join(td, f"n{i}.wav")
                _normalize_clip(raw, n, is_long=(clip["kind"] == "mp3"))
                norm.append(n)
            _concat_cap(norm, abs_wav)
    sha = _sha256(abs_wav)
    return build_entries(spec, rel_path, sha)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest public-domain voice refs.")
    ap.add_argument("--models-base", default=os.environ.get(
        "OTR_MODELS_BASE", r"C:\ComfyUI-Models"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[pd-ingest] speakers={len(PD_VOICE_SPEAKERS)}  "
          f"models_base={args.models_base}  dry_run={args.dry_run}", flush=True)
    all_entries: list = []
    for spec in PD_VOICE_SPEAKERS:
        if args.dry_run:
            rel = f"{_REF_REL_DIR}/vz_{spec['voice_id']}.wav"
            all_entries.extend(build_entries(spec, rel, "<sha-after-download>"))
            print(f"  [plan] {spec['voice_id']} ({spec['gender']}/"
                  f"{spec['age_band']}, {len(spec['clips'])} clip(s)) "
                  f"-> 1 voice x vz_/cb_/dia_", flush=True)
            continue
        try:
            all_entries.extend(_ingest_speaker(spec, args.models_base))
            print(f"  [ok]   {spec['voice_id']} -> 3 entries", flush=True)
        except Exception as exc:  # noqa: BLE001 -- one bad voice never stops the run
            print(f"  [FAIL] {spec['voice_id']}: {exc!r}", flush=True)

    if args.dry_run:
        print(f"[pd-ingest] would add {len(all_entries)} entries "
              f"({len(all_entries)//3} distinct voices x 3 engines)", flush=True)
        return 0

    added, skipped = merge_into_bank(BANK_JSON, all_entries)
    print(f"[pd-ingest] merged: added={added} skipped={skipped}", flush=True)
    from nodes._otr_voice_bank import compute_bank_coverage, load_voice_bank
    bank, sha = load_voice_bank()
    print(f"[pd-ingest] bank OK: {len(bank)} entries, sha={sha[:12]}", flush=True)
    for eng, c in compute_bank_coverage(bank).items():
        print(f"  {eng}: adult M={c['adult_male']} F={c['adult_female']} "
              f"total={c['total']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
