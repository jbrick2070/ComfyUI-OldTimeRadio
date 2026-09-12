"""WHICH MUSIC MODEL SHIPS -- the bench arm for the GO_FORWARD_PLAN section 1 row.

Three candidates, one recipe held constant per family, the show's OWN composed
prompts (read through nodes/_otr_music_prompt at run time, never retyped):

  small_base   stable_audio_3_small_music_base.safetensors   (0.6B, on disk)
  medium_base  stable_audio_3_medium_base.safetensors        (2B,   on disk)
  ace15        ACE-Step 1.5 turbo + qwen 0.6b/4b + vae        (blueprint recipe)

The two SA3 arms run the engine's base-checkpoint recipe exactly as it would
ship (cfg 4.0, 100 steps, dpmpp_3m_sde_gpu / exponential, a 3x conditioning
window), so the ONLY variable between them is the checkpoint. ACE-Step is a
different architecture and runs the recipe its own ComfyUI blueprint ships.

Six prompt families: two organ pieces (the burst control from
scripts/otr_organ_bench.py), the Shakespeare opening and closing cue (the
sustained lane, the only one still carrying the anti-loop negative), and the
Chicago house and Detroit techno opening cues (the two genre lanes that came
back as pads on 2026-09-12). Three seeds per family per arm.

Per render: bursts and loopiness (the organ bench's own measure, so the numbers
compare with the 65-render campaign), onset rate and tempo commit (librosa),
peak / RMS / clipped, wall seconds, and the VRAM peak read from nvidia-smi.

A lossless FLAC goes under output/organ_bench/ for the measurement; an MP3 of
the SAME decode goes to output/otr/obs/music_model_bench/ for the operator's
ear, labelled by arm and family. THIS QUALIFIES NOTHING: it is a stock-node
graph, not the canonical workflow, and it settles a design fork only.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
sys.path.insert(0, r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts")
os.environ.setdefault("OTR_TEST_MODE", "1")

import otr_organ_bench as organ  # noqa: E402  (post / get / measure / NEGATIVE / PIECES)
from nodes import _otr_music_prompt as MP  # noqa: E402
from nodes import _otr_music_palette as PAL  # noqa: E402

OUTPUT_ROOT = organ.OUTPUT_ROOT
OBS_SUBDIR = "otr/obs/music_model_bench"
FLAC_SUBDIR = "organ_bench/mmb"
TEXT_ENCODER = organ.TEXT_ENCODER

# The engine's shipped base-checkpoint recipe (eng_stable_audio_3.py).
SA3_RECIPE = dict(cfg=4.0, steps=100, sampler="dpmpp_3m_sde_gpu", scheduler="exponential")
SA3_ARMS = {
    "small_base": "stable_audio_3_small_music_base.safetensors",
    "medium_base": "stable_audio_3_medium_base.safetensors",
}
# The ComfyUI blueprint "Text to Audio (ACE-Step 1.5)" recipe, verbatim.
ACE = dict(unet="acestep_v1.5_turbo.safetensors", clip_a="qwen_0.6b_ace15.safetensors",
           clip_b="qwen_4b_ace15.safetensors", vae="ace_1.5_vae.safetensors",
           shift=3.0, steps=8, cfg=1.0, sampler="euler", scheduler="simple",
           lm_cfg=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0)
ACE_FILES = (
    (r"C:\ComfyUI-Models\diffusion_models\acestep_v1.5_turbo.safetensors", 4787825604),
    (r"C:\ComfyUI-Models\text_encoders\qwen_0.6b_ace15.safetensors", 1191588248),
    (r"C:\ComfyUI-Models\text_encoders\qwen_4b_ace15.safetensors", 8379154232),
    (r"C:\ComfyUI-Models\vae\ace_1.5_vae.safetensors", 337431732),
)

# The same story metas the compose probe used; the composer does the rest.
METAS = {
    "shakespeare": {
        "source_bank": "shakespeare",
        "source_meta": {"bank": "shakespeare", "year": 1606, "title": "Macbeth"},
        "story_brief_terms": {"setting": ["a windswept heath", "a castle at night"],
                              "atmosphere": ["dread", "ambition", "fog"]},
        "produced_story": {"logline": "A general is told he will be king and murders his way there."},
    },
    "public_domain": {
        "source_bank": "public_domain",
        "source_meta": {"bank": "public_domain", "year": 1890, "title": "The Signal-Man"},
        "story_brief_terms": {"setting": ["a railway cutting", "a signal box"],
                              "atmosphere": ["unease", "isolation", "warning"]},
        "produced_story": {"logline": "A signalman is haunted by a spectre that warns of accidents."},
    },
    "scifi_news_pro": {
        "source_bank": "scifi_news_pro",
        "source_meta": {"bank": "scifi_news_pro", "year": 2091},
        "story_brief_terms": {"setting": ["an orbital newsroom", "a lunar dock"],
                              "atmosphere": ["urgent", "tense", "danger"]},
        "produced_story": {"logline": "A lunar dock loses pressure during a broadcast."},
    },
}


def composed(bank, cue):
    row, seconds = MP.compose_music_prompt(METAS[bank], cue)
    ep = MP.compose_engine_prompt(METAS[bank], row)
    return ep.text, ep.negative, float(seconds), PAL.story_palette(METAS[bank]).key


def families():
    organ_pieces = dict(organ.PIECES)
    out = [
        {"key": "organ_cathedral", "text": organ_pieces["cathedral"], "negative": organ.NEGATIVE,
         "seconds": 20.0, "bpm": 60, "keyscale": "D minor", "lane": "burst control"},
        {"key": "organ_toccata", "text": organ_pieces["toccata"], "negative": organ.NEGATIVE,
         "seconds": 20.0, "bpm": 72, "keyscale": "D minor", "lane": "burst control"},
    ]
    for key, bank, cue, bpm in (("shakespeare_opening", "shakespeare", "opening", 66),
                                ("shakespeare_closing", "shakespeare", "closing", 66),
                                ("house_opening", "public_domain", "opening", 122),
                                ("techno_opening", "scifi_news_pro", "opening", 128)):
        text, negative, seconds, palette = composed(bank, cue)
        out.append({"key": key, "text": text, "negative": negative, "seconds": seconds,
                    "bpm": bpm, "keyscale": "E minor", "lane": palette})
    return out


def sa3_graph(ckpt, fam, seed, label):
    window = fam["seconds"] * 3.0
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "9": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "stable_audio"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": fam["text"]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": fam["negative"]}},
        "4": {"class_type": "ConditioningStableAudio",
              "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                         "seconds_start": 0.0, "seconds_total": float(window)}},
        "5": {"class_type": "EmptyLatentAudio",
              "inputs": {"seconds": float(fam["seconds"]), "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": int(seed), "steps": SA3_RECIPE["steps"],
                         "cfg": SA3_RECIPE["cfg"], "sampler_name": SA3_RECIPE["sampler"],
                         "scheduler": SA3_RECIPE["scheduler"],
                         "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveAudio",
              "inputs": {"audio": ["7", 0], "filename_prefix": "%s/%s" % (FLAC_SUBDIR, label)}},
        "10": {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["7", 0], "quality": "V0",
                          "filename_prefix": "%s/%s" % (OBS_SUBDIR, label)}},
    }


def ace_graph(fam, seed, label):
    tags = fam["text"]
    return {
        "104": {"class_type": "UNETLoader",
                "inputs": {"unet_name": ACE["unet"], "weight_dtype": "default"}},
        "105": {"class_type": "DualCLIPLoader",
                "inputs": {"clip_name1": ACE["clip_a"], "clip_name2": ACE["clip_b"],
                           "type": "ace", "device": "default"}},
        "106": {"class_type": "VAELoader", "inputs": {"vae_name": ACE["vae"]}},
        "78": {"class_type": "ModelSamplingAuraFlow",
               "inputs": {"model": ["104", 0], "shift": ACE["shift"]}},
        "94": {"class_type": "TextEncodeAceStepAudio1.5",
               "inputs": {"clip": ["105", 0], "tags": tags, "lyrics": "[Instrumental]",
                          "seed": int(seed), "bpm": int(fam["bpm"]),
                          "duration": float(fam["seconds"]), "timesignature": "4",
                          "language": "en", "keyscale": fam["keyscale"],
                          "generate_audio_codes": True, "cfg_scale": ACE["lm_cfg"],
                          "temperature": ACE["temperature"], "top_p": ACE["top_p"],
                          "top_k": ACE["top_k"], "min_p": ACE["min_p"]}},
        "47": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["94", 0]}},
        "98": {"class_type": "EmptyAceStep1.5LatentAudio",
               "inputs": {"seconds": float(fam["seconds"]), "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"model": ["78", 0], "seed": int(seed), "steps": ACE["steps"],
                         "cfg": ACE["cfg"], "sampler_name": ACE["sampler"],
                         "scheduler": ACE["scheduler"], "positive": ["94", 0],
                         "negative": ["47", 0], "latent_image": ["98", 0], "denoise": 1.0}},
        "18": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["3", 0], "vae": ["106", 0]}},
        "8": {"class_type": "SaveAudio",
              "inputs": {"audio": ["18", 0], "filename_prefix": "%s/%s" % (FLAC_SUBDIR, label)}},
        "10": {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["18", 0], "quality": "V0",
                          "filename_prefix": "%s/%s" % (OBS_SUBDIR, label)}},
    }


def vram_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10).decode().strip().splitlines()[0]
        return int(out)
    except Exception:  # noqa: BLE001 -- a missing reading never fails a render
        return -1


class VramWatch:
    def __init__(self):
        self.peak = vram_used_mib()
        self.baseline = self.peak
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            v = vram_used_mib()
            if v > self.peak:
                self.peak = v
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


def render(graph):
    """Queue one graph; return (flac_path, mp3_path, wall_seconds, vram_peak_mib)."""
    with VramWatch() as vw:
        prompt_id = organ.post("/prompt", {"prompt": graph})["prompt_id"]
        started = time.time()
        while time.time() - started < 1800:
            entry = (organ.get("/history/%s" % prompt_id) or {}).get(prompt_id) or {}
            status = entry.get("status") or {}
            if status.get("completed"):
                flac = mp3 = None
                for node in (entry.get("outputs") or {}).values():
                    for item in node.get("audio") or []:
                        path = os.path.join(OUTPUT_ROOT, item.get("subfolder", ""), item["filename"])
                        if path.lower().endswith(".flac"):
                            flac = path
                        elif path.lower().endswith(".mp3"):
                            mp3 = path
                return flac, mp3, time.time() - started, vw.peak
            if status.get("status_str") == "error":
                raise RuntimeError("render failed: %s" % json.dumps(status.get("messages") or [])[:600])
            time.sleep(1.5)
    raise TimeoutError("no result in 1800 s")


def rhythm(path):
    """Onsets per minute, the tempo librosa commits to, and pulse clarity (the
    normalised onset-envelope autocorrelation peak in the 60-180 BPM range)."""
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size < sr:
        return {"onsets_per_min": 0.0, "tempo_bpm": 0.0, "pulse": 0.0}
    env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="frames")
    per_min = len(onsets) / (len(y) / sr) * 60.0
    tempo = librosa.feature.tempo(onset_envelope=env, sr=sr, aggregate=None)
    tempo_bpm = float(np.median(np.atleast_1d(tempo))) if np.size(tempo) else 0.0
    hop = 512
    env = env - env.mean()
    pulse = 0.0
    if np.dot(env, env) > 0:
        ac = np.correlate(env, env, mode="full")[env.size - 1:]
        ac = ac / ac[0]
        lo = int(round(60.0 / 180.0 * sr / hop))
        hi = min(int(round(60.0 / 60.0 * sr / hop)), ac.size - 1)
        if hi > lo:
            pulse = float(np.max(ac[lo:hi]))
    return {"onsets_per_min": round(per_min, 1), "tempo_bpm": round(tempo_bpm, 1),
            "pulse": round(pulse, 3)}


def ace_ready():
    return all(os.path.isfile(p) and os.path.getsize(p) == n for p, n in ACE_FILES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["small_base", "medium_base", "ace15"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--base-seed", type=int, default=90120001)
    args = ap.parse_args()

    fams = [f for f in families() if not args.families or f["key"] in args.families]
    rows = []
    if os.path.isfile(args.json_out):
        with open(args.json_out, encoding="utf-8") as fh:
            rows = json.load(fh)
    done = {(r["arm"], r["family"], r["seed"]) for r in rows}

    for arm in args.arms:
        if arm == "ace15" and not ace_ready():
            print("ARM %s SKIPPED: weights not all on disk yet" % arm, flush=True)
            continue
        print("== ARM %s" % arm, flush=True)
        for i, fam in enumerate(fams):
            for k in range(args.seeds):
                seed = args.base_seed + i * 17 + k * 1000
                if (arm, fam["key"], seed) in done:
                    continue
                label = "%s__%s__seed%d" % (arm, fam["key"], k)
                graph = (ace_graph(fam, seed, label) if arm == "ace15"
                         else sa3_graph(SA3_ARMS[arm], fam, seed, label))
                try:
                    flac, mp3, took, vram = render(graph)
                except Exception as exc:  # noqa: BLE001 -- a bench never dies mid-arm
                    print("  %-28s FAILED: %s" % (label, str(exc)[:300]), flush=True)
                    rows.append({"arm": arm, "family": fam["key"], "seed": seed,
                                 "error": str(exc)[:300]})
                    continue
                if not flac:
                    print("  %-28s no flac came back" % label, flush=True)
                    continue
                m = organ.measure(flac)
                m.update(rhythm(flac))
                m.update({"arm": arm, "family": fam["key"], "lane": fam["lane"], "seed": seed,
                          "seed_index": k, "flac": flac, "mp3": mp3,
                          "wall_s": round(took, 1), "vram_peak_mib": vram,
                          "cue_seconds": fam["seconds"]})
                rows.append(m)
                print("  %-28s %5.1fs vram=%5d bursts=%-2d loop=%.3f@%.2fs onsets/min=%5.1f "
                      "tempo=%5.1f pulse=%.2f peak=%6.2f rms=%6.2f"
                      % (label, took, vram, m["bursts"], m["loopiness"], m["loop_lag_s"],
                         m["onsets_per_min"], m["tempo_bpm"], m["pulse"], m["peak_dbfs"],
                         m["rms_dbfs"]), flush=True)
                with open(args.json_out, "w", encoding="utf-8") as fh:
                    json.dump(rows, fh, indent=1)
    print("BENCH DONE rows=%d" % len(rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
