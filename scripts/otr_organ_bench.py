"""THE ORGAN BENCH -- a side project, and a clean-room test of the music model.

Operator, 2026-09-12: "make a fun musical organ side project and see if you can
get it clean."

WHY AN ORGAN IS THE RIGHT TEST. A pipe organ is sustained, harmonically dense
and has no transients of its own, so anything percussive, noisy or broadband in
the output came from the model and not from the instrument. It is the least
forgiving material for the exact defect the operator hears as "an odd tape loop
scratch", and the most pleasant to listen to while checking.

THIS QUALIFIES NOTHING ABOUT THE EPISODE PIPELINE. It renders a small Stable
Audio graph directly against the running server, not the canonical workflow, so
its numbers are for this side project and for choosing a checkpoint to A/B
properly later. The canonical harness (scripts/otr_music_ab.py) remains the only
thing that may qualify a change to the show.

WHAT IT MEASURES, per render:
  * bursts -- 50 ms blocks that are loud (within 12 dB of the piece's own peak),
    noise-like (spectral flatness over 20 Hz-8 kHz above 0.20) and bright (more
    than 45% of energy above 4 kHz). This is the signature measured on the cue
    the operator flagged: flatness 0.35, 57% above 4 kHz, sitting at the peak.
  * loopiness -- the strongest autocorrelation of the 10 ms loudness envelope
    between 0.25 s and 6 s. The other complaint, in one number.
  * peak, RMS and clipped samples.

LOOPINESS HAS A BLIND SPOT, AND IT COST A NIGHT'S CONCLUSION. It is the
strongest autocorrelation of the loudness ENVELOPE, so it cannot tell a
repeating figure from a SUSTAINED one: a held organ chord has a nearly constant
envelope and scores as high as a two-bar loop. A clean 45 s cathedral piece
measured 0.730 while its spectrogram shows the harmony changing four times and
no repeat at all.

WORSE, IT IS PUSHED DOWN BY THE OTHER DEFECT. A broadband burst decorrelates
the envelope, so an artifact LOWERS the score: measured across 29 renders,
those carrying a burst averaged 0.157 against 0.510 for clean ones. A drop in
loopiness is therefore not by itself good news -- read it beside the burst count
or it will reward the very thing this campaign is removing.
"""
import argparse
import glob
import json
import math
import os
import time
import urllib.request

import numpy as np
import soundfile as sf

SERVER = "http://127.0.0.1:8000"
OUTPUT_ROOT = r"C:\Users\jeffr\Documents\ComfyUI\output"
TEXT_ENCODER = "t5gemma_b_b_ul2.safetensors"


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path, timeout=60).read())


def graph(ckpt, prompt, negative, seconds, seed, steps, cfg, sampler,
          scheduler, seconds_start, seconds_total, prefix):
    """The Stable Audio graph the pack's own engine drives, as an API prompt."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        # SA3 ships its text encoder SEPARATELY -- the checkpoint carries no
        # CLIP, and CLIPTextEncode raises "clip input is invalid: None" if you
        # take slot 1 from the loader. The pack's own engine has the same
        # fallback (eng_stable_audio_3.py:196-197).
        "9": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "stable_audio"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["9", 0], "text": prompt}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["9", 0], "text": negative}},
        "4": {"class_type": "ConditioningStableAudio",
              "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                         "seconds_start": float(seconds_start),
                         "seconds_total": float(seconds_total)}},
        "5": {"class_type": "EmptyLatentAudio",
              "inputs": {"seconds": float(seconds), "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": int(seed),
                         "steps": int(steps), "cfg": float(cfg),
                         "sampler_name": sampler, "scheduler": scheduler,
                         "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveAudio",
              "inputs": {"audio": ["7", 0], "filename_prefix": prefix}},
    }


def render(**kw):
    prefix = "organ_bench/%s" % kw.pop("label")
    prompt_id = post("/prompt", {"prompt": graph(prefix=prefix, **kw)})["prompt_id"]
    started = time.time()
    while time.time() - started < 900:
        hist = get("/history/%s" % prompt_id)
        entry = hist.get(prompt_id) or {}
        status = entry.get("status") or {}
        if status.get("completed"):
            for node in (entry.get("outputs") or {}).values():
                for item in node.get("audio") or []:
                    path = os.path.join(OUTPUT_ROOT, item.get("subfolder", ""),
                                        item["filename"])
                    if os.path.isfile(path):
                        return path, time.time() - started
            return None, time.time() - started
        if status.get("status_str") == "error":
            msgs = status.get("messages") or []
            raise RuntimeError("render failed: %s" % json.dumps(msgs)[:400])
        time.sleep(2)
    raise TimeoutError("no result in 900 s")


def dbfs(x):
    return 20.0 * math.log10(max(float(abs(x)), 1e-12))


def measure(path):
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(mono))) or 1e-12
    floor = peak * (10.0 ** (-12.0 / 20.0))
    n = int(sr * 0.05)
    hits = []
    for i in range(0, len(mono) - n, n):
        b = mono[i:i + n]
        if float(np.max(np.abs(b))) < floor:
            continue
        spec = np.abs(np.fft.rfft(b * np.hanning(b.size))) ** 2
        fr = np.fft.rfftfreq(b.size, 1.0 / sr)
        band = (fr >= 20.0) & (fr <= 8000.0)
        p = spec[band]
        p = p[p > 0]
        if p.size < 8:
            continue
        flat = float(np.exp(np.mean(np.log(p))) / np.mean(p))
        total = float(spec.sum()) or 1.0
        if flat > 0.20 and float(spec[fr >= 4000.0].sum() / total) > 0.45:
            hits.append(round(i / sr, 2))
    # loopiness: the strongest repeat in the 10 ms loudness envelope
    step = max(1, int(sr * 0.01))
    env = np.array([float(np.sqrt(np.mean(mono[i:i + step] ** 2)))
                    for i in range(0, len(mono) - step, step)])
    env = env - env.mean()
    loop, lag_s = 0.0, 0.0
    if env.size > 64 and float(np.dot(env, env)) > 0:
        ac = np.correlate(env, env, mode="full")[env.size - 1:]
        ac = ac / ac[0]
        lo, hi = int(0.25 / 0.01), min(int(6.0 / 0.01), ac.size - 1)
        if hi > lo:
            k = int(np.argmax(ac[lo:hi]) + lo)
            loop, lag_s = float(ac[k]), k * 0.01
    return {"seconds": round(len(mono) / sr, 2), "sr": sr,
            "peak_dbfs": round(dbfs(peak), 2),
            "rms_dbfs": round(dbfs(np.sqrt(np.mean(mono ** 2))), 2),
            "clipped": int(np.sum(np.abs(mono) >= 0.999)),
            "bursts": len(hits), "burst_at": hits[:8],
            "loopiness": round(loop, 3), "loop_lag_s": round(lag_s, 2)}


#: THE PRODUCTION NEGATIVE, verbatim from nodes/_otr_music_prompt.py, plus the
#: sound-design words an organ piece additionally does not want. The first cut
#: of this bench quietly DROPPED the anti-loop half, so the arm that was meant
#: to prove the negative prompt works on a base checkpoint was tested without
#: the very words in question. Caught by reading the two negatives side by side.
NEGATIVE = ("noise, static, hiss, white noise, radio static, crackle, "
            "distortion, clipping, silence, speech, vocals, singing, lyrics, "
            "spoken word, loop, looping, repetitive, ostinato, sequencer, "
            "arpeggiator, drum machine, metronome, click track, drum loop, "
            "beat, percussion, cymbal, sound effect, foley")

PIECES = [
    ("cathedral", "solo pipe organ in a stone cathedral, slow sustained chords, "
                  "deep pedal notes, warm diapason stops, a long reverberant tail, "
                  "a solemn hymn-like melody unfolding, clearly recorded, "
                  "instrumental only"),
    ("toccata", "solo pipe organ, a flowing toccata figure over a held pedal, "
                "bright principal stops, a rising melodic line, grand and "
                "unhurried, clearly recorded, instrumental only"),
    ("reed", "small chamber organ, soft reed stops, a tender slow melody over "
             "quiet held chords, intimate wooden room, clearly recorded, "
             "instrumental only"),
    ("nocturne", "pipe organ nocturne, soft flute stops, a gentle floating "
                 "melody, minor key, unmetered and drifting, clearly recorded, "
                 "instrumental only"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Defaults are what the engine SHIPS for a base checkpoint (2026-09-12):
    # the base file and its guidance. The old pair -- the post-trained file
    # at cfg 7 -- is PBUG-20260912-03 itself, and a bench whose default arm
    # is the defect measures the wrong thing unless every caller remembers.
    ap.add_argument("--ckpt", default="stable_audio_3_small_music_base.safetensors")
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--sampler", default="dpmpp_3m_sde_gpu")
    ap.add_argument("--scheduler", default="exponential")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--window", type=float, default=None,
                    help="seconds_total; default 3x the piece")
    ap.add_argument("--seed", type=int, default=90120001)
    ap.add_argument("--pieces", nargs="*", default=None)
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    window = args.window or args.seconds * 3.0
    chosen = [p for p in PIECES if not args.pieces or p[0] in args.pieces]
    rows = []
    print("organ bench | ckpt=%s cfg=%.1f steps=%d %s/%s %.0fs in a %.0fs window"
          % (args.ckpt, args.cfg, args.steps, args.sampler, args.scheduler,
             args.seconds, window))
    for index, (name, prompt) in enumerate(chosen):
        label = "%s_%s" % (args.tag, name)
        try:
            path, took = render(
                ckpt=args.ckpt, prompt=prompt, negative=NEGATIVE,
                seconds=args.seconds, seed=args.seed + index * 17,
                steps=args.steps, cfg=args.cfg, sampler=args.sampler,
                scheduler=args.scheduler, seconds_start=0.0,
                seconds_total=window, label=label)
        except Exception as exc:  # noqa: BLE001 -- a bench never fails a night
            print("  %-12s FAILED: %s" % (name, str(exc)[:160]))
            continue
        if not path:
            print("  %-12s no audio came back" % name)
            continue
        m = measure(path)
        m.update({"piece": name, "file": path, "seconds_rendered": round(took, 1),
                  "ckpt": args.ckpt, "cfg": args.cfg, "steps": args.steps,
                  "sampler": args.sampler, "scheduler": args.scheduler})
        rows.append(m)
        print("  %-12s %5.1fs  bursts=%-3d loopiness=%-6.3f peak=%-7.2f rms=%-7.2f %s"
              % (name, took, m["bursts"], m["loopiness"], m["peak_dbfs"],
                 m["rms_dbfs"], os.path.basename(path)))
    if rows:
        print("  %-12s bursts total %d | mean loopiness %.3f | mean peak %.2f dBFS"
              % ("SUMMARY", sum(r["bursts"] for r in rows),
                 float(np.mean([r["loopiness"] for r in rows])),
                 float(np.mean([r["peak_dbfs"] for r in rows]))))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
