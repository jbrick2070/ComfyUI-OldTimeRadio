"""otr_h3_mime_runner.py -- the MIME lane: MiniMax H3, self-scored, KEEPS its audio.

LANE 21, and it is the odd one out of the whole 21-lane campaign.

WHAT IT IS. A STANDALONE RUNNER -- a script, not a registered engine. It takes a
portrait/still, a prompt and a target duration, snaps that duration to H3's own
frame grid, renders ONE self-scored mp4 into a durable deliverables directory,
and writes its own receipt beside it.

**IT IS NOT REGISTERED, AND THAT IS THE POINT** (operator ruling 2026-08-10,
spec P3). r3 proved a registered engine cannot be kept OUT of the episode
dropdowns -- `role_compat` grants every role the full input vocabulary and the
dropdown IS the registry -- so a mime engine in the registry would immediately be
selectable for ordinary beats, and a mime beat's audio does not exist until its
video renders. The dropdown entry follows in a LATER build behind a real
standalone-only boundary, together with the "mime overrules TTS/music" wiring
and the phase inversion that needs (audio-first ordering freezes the master WAV
before video renders; a mime beat inverts that). This runner is the immediate
capability and the test bed those pieces will be built on.

**IT KEEPS THE AUDIO. Every other H3 path in this repo proves SILENCE.**
Lanes 19 and 20 (`minimax_h3_video`, `minimax_h3_audio_in`) never even decode
H3's audio latent, and `canonicalize` ffprobes their emitted file to PROVE
`has_audio: False` -- V-1 says only `OTR_MasterAudioMux` adds audio to an
EPISODE. This runner is the G5.2 exemption: the clip's invented score IS its
audio, it is what the lane exists for, and V-1 is not violated because **these
outputs never enter episode assembly.** They are runner deliverables. If a mix
ever graduates toward episodes it re-enters through a fresh ear gate and its own
spec, not through this script.

So the assert here is INVERTED from every other lane: the delivered mp4 MUST
carry exactly one audio stream, and the runner fails loudly if it does not.

DELIVERY IS AT THE MODEL'S OWN 24 fps, and this is the one place in the build
that does NOT convert to the 25 fps canvas. Lanes 19/20 remap their frames to 25
because they hand clips to a 25 fps timeline. This clip is not going to a
timeline -- and resampling the video while its generated score stays at the
model's timebase is exactly how you desync a soundtrack from the picture it was
scored for. Native rate, both streams, one timebase.

THE GRID FLOOR IS DELIBERATELY LOWER HERE. The registered lanes publish only the
TRAINED band (model 124..362 -> canvas 129..377) because a beat that renders
below it is a beat that renders badly. The lab's two passing mime legs are model
**f90** -- 3.750 s, below that floor and on the 17k+5 grid -- and both scored
well. A mime clip is a short authored moment, not a beat, so this runner accepts
the whole legal grid from f5 up and simply SAYS when it is below the trained
band rather than refusing.

`--stem-wav` exports the invented score as a lossless FLAC sidecar (one extra
`SaveAudioAdvanced`, no second sampler pass). `--mux-tts` writes an ADDITIONAL
review copy with a supplied real voice over the music, ducked; both originals
are preserved untouched.

RELATIONSHIP TO CLAUDE.md SECTION 0, stated rather than hidden. Section 0 says
every API/headless run must load `workflows/otr_canonical.json`, and section 0A
carves out exactly two bench runners "and no other runner". This runner submits
its own H3 graph, so it does not fit either. It is authorized by the LATER and
more specific operator ruling of 2026-08-10 (spec P3, "ships THIS build as a
STANDALONE RUNNER ... rendering ONE self-scored mp4 into a durable deliverables
directory with its own receipt"), which is a decision about this exact script.
It borrows the carve-out's DISCIPLINE anyway -- every node class confirmed live
in `/object_info` before submit, the asset resolved from `/history` and never a
glob, and the result ffprobe'd -- so the audit trail is the same shape.
**The carve-out's "no other runner" sentence needs an operator amendment to name
this script; flagged in the lane 21 receipt rather than silently widened.**

Usage:
  python scripts/otr_h3_mime_runner.py --portrait P.png --seconds 3.75 \
      [--prompt "..."] [--seed 42] [--stem-wav] [--mux-tts voice.wav] \
      [--out-dir DIR] [--width 864] [--height 480]

Exit 0 only when the clip rendered AND carried its audio. UTF-8, no BOM,
ASCII-only source.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import requests  # noqa: E402

from otr_api import COMFYUI_URL, submit_prompt, poll_history  # noqa: E402

# The grid math, the loader tokens and the frozen recipe all come from the
# ADAPTER module -- the same numbers lanes 19 and 20 render with. Importing them
# is what stops this runner from growing a second, drifting copy of the H3
# contract. It registers nothing by importing (the module's @register calls are
# on the adapter classes, which are already registered by the package import).
from nodes._otr_video_engines.eng_minimax_h3 import (  # noqa: E402
    H3_CANVAS_FPS,
    H3_MODEL_FPS,
    H3_RECIPE,
    H3_TRAINED_MAX_MODEL_FRAMES,
    H3_TRAINED_MIN_MODEL_FRAMES,
    align_frame_count,
)

#: The mime lane renders on the FL2VA DiT (a still + a prompt), like lane 19.
MIME_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
MIME_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
MIME_VAE = "minimax_h3_video_vae_fp16.safetensors"
MIME_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

#: The lab's two passing mime legs both ran here (`h3_mime_i2v` / `h3_mime_r2v`,
#: 864x480 model f90, 7.28 / 7.23 GiB cold), and it is the canvas both registered
#: H3 lanes declare.
MIME_W, MIME_H = 864, 480

#: The operator's prompt rule: a mime score request is derived from the beat's
#: DRAMATIC INTENT, in the "tense orchestral drama" register family, never
#: boilerplate. This default is a usable starting point for a solo run and is
#: meant to be REPLACED per clip with --prompt.
DEFAULT_PROMPT = (
    "A restrained cinematic moment: the subject from the first frame acts with "
    "the body alone, no speech and no lip movement, while a tense orchestral "
    "score rises under the scene. Warm analog light, slow stable camera, "
    "physically coherent movement.")

#: Where deliverables land. NOT tmp/scratch: these are outputs the operator
#: keeps (CLAUDE.md section 6 -- point a render at its canonical path the FIRST
#: time; never stage in tmp "to move later").
DEFAULT_OUT_DIR = os.path.join(
    os.environ.get("OTR_OUTPUT_DIR",
                   r"C:\Users\jeffr\Documents\ComfyUI\output"),
    "otr", "episodes", "_mime")

#: Every class the mime graph needs. Confirmed live before submit.
REQUIRED_NODE_CLASSES = (
    "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage",
    "MiniMaxH3ImageToVideo", "BasicGuider", "RandomNoise", "KSamplerSelect",
    "BasicScheduler", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio",
    "CreateVideo", "SaveVideo",
)
STEM_NODE_CLASS = "SaveAudioAdvanced"


# --------------------------------------------------------------------------- #
# Pure helpers (CPU-testable; no server, no GPU)
# --------------------------------------------------------------------------- #
def model_frames_for_seconds(seconds, fps=H3_MODEL_FPS):
    """Target SECONDS -> the H3 model frame count that covers them.

    Snaps UP through :func:`align_frame_count`, which is the node's own rule, so
    the number handed to the graph is one the node will not silently move. Never
    below the grid's own floor of 5.
    """
    if float(seconds) <= 0:
        raise ValueError("--seconds must be positive, got %r" % (seconds,))
    raw = max(5, int(round(float(seconds) * float(fps))))
    return align_frame_count(raw)


def delivered_seconds(model_frames, fps=H3_MODEL_FPS):
    """What the clip will ACTUALLY be, at the model's own rate."""
    return int(model_frames) / float(fps)


def below_trained_band(model_frames):
    """True when this length is legal on the grid but under the trained band.

    Not an error on this lane -- the lab's two passing mime legs are f90 -- but
    it is SAID, because the registered lanes refuse below it and a reader
    comparing receipts deserves to know which side of that line a clip is on.
    """
    return int(model_frames) < H3_TRAINED_MIN_MODEL_FRAMES


def build_mime_graph(*, image_name, prompt, model_frames, seed, width, height,
                     filename_prefix, stem_prefix=None):
    """The API-format mime graph: the lab's pinned topology, KEEPING the audio.

    It is lane 19's graph plus the audio half the registered lanes deliberately
    drop -- ``VAEDecodeAudio`` off the same sampled latent, into ``CreateVideo``,
    into ``SaveVideo``. ComfyUI writes the container here rather than OTR's
    silent encoder, because the whole deliverable is a clip WITH its score.

    ``stem_prefix`` adds a lossless FLAC export of that same decoded audio. It
    reuses node 15's output, so the stem costs one file write and no extra
    decode.
    """
    graph = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MIME_UNET,
                         "weight_dtype": H3_RECIPE["weight_dtype"]}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MIME_CLIP,
                         "type": H3_RECIPE["clip_type"],
                         "device": H3_RECIPE["clip_device"]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MIME_VAE}},
        "4": {"class_type": "VAELoader",
              "inputs": {"vae_name": MIME_AUDIO_VAE}},
        "5": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "6": {"class_type": "BasicGuider",
              "inputs": {"model": ["1", 0], "conditioning": ["7", 0]}},
        "7": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                         "width": int(width), "height": int(height),
                         "length": int(model_frames),
                         "first_frame": ["11", 0]}},
        "8": {"class_type": "SamplerCustomAdvanced",
              "inputs": {"noise": ["5", 0], "guider": ["6", 0],
                         "sampler": ["13", 0], "sigmas": ["14", 0],
                         "latent_image": ["7", 1]}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "CreateVideo",
               "inputs": {"images": ["9", 0], "fps": float(H3_MODEL_FPS),
                          "bit_depth": 8, "audio": ["15", 0]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["10", 0],
                          "filename_prefix": filename_prefix,
                          "format": "auto", "codec": "auto"}},
        "13": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": H3_RECIPE["sampler"]}},
        "14": {"class_type": "BasicScheduler",
               "inputs": {"model": ["1", 0],
                          "scheduler": H3_RECIPE["scheduler"],
                          "steps": H3_RECIPE["steps"],
                          "denoise": H3_RECIPE["denoise"]}},
        "15": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
    }
    if stem_prefix:
        # `format` is a DynamicCombo, and this is an API graph, so it takes the
        # flat SELECTOR STRING -- "flac" -- not the dict. The executor expands a
        # DynamicCombo the way it expands an Autogrow: the selector arrives as
        # `format`, any option sub-inputs arrive dotted (`format.quality`), and
        # it reassembles them into the dict `execute` receives. Passing that
        # dict directly is what an IN-PROCESS call would do, and it fails here
        # with "missing 1 required positional argument: 'format'".
        #
        # Same lesson as lane 20's reference sockets, pointing the other way:
        # the API serialization is not the in-process call. flac has no
        # sub-inputs, so the selector is the whole of it -- and flac because a
        # stem meant to be reusable material must be lossless.
        graph["16"] = {"class_type": STEM_NODE_CLASS,
                       "inputs": {"audio": ["15", 0],
                                  "filename_prefix": stem_prefix,
                                  "format": "flac"}}
    return graph


def ducked_mux_cmd(video_path, voice_path, out_path, *, ffmpeg="ffmpeg",
                   duck_db=-9.0):
    """ffmpeg args for the --mux-tts REVIEW copy: voice over ducked music.

    Deliberately simple and deliberately NON-destructive: it reads the finished
    mp4 and the supplied voice and writes a THIRD file. Both originals are
    untouched, because the point of the flag is comparison, not replacement.

    The music is attenuated by a fixed amount rather than side-chain compressed:
    a fixed duck is predictable and reviewable, and a compressor's attack/release
    would be one more thing to tune before anyone can judge the SCORE, which is
    what this runner exists to evaluate.
    """
    return [
        ffmpeg, "-y", "-v", "error",
        "-i", video_path,
        "-i", voice_path,
        "-filter_complex",
        "[0:a]volume=%.2fdB[m];[1:a]aresample=async=1[v];"
        "[m][v]amix=inputs=2:duration=first:dropout_transition=0[a]" % duck_db,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", out_path,
    ]


def existing_deliverables(out_dir, name):
    """Which of this run's deliverables are already on disk, in path order.

    A DELIVERABLE IS NEVER SILENTLY OVERWRITTEN. The default name carries the
    seed and a one-SECOND timestamp, so two runs with the same seed inside one
    second -- a scripted retry loop -- collide, and an explicit ``--name``
    collides on every repeat. These are outputs the operator keeps, so the
    collision is refused by name unless ``--overwrite`` says otherwise.

    A pure list-returning helper rather than an inline comprehension so the
    rule can be tested by CALLING it (lesson L26).
    """
    return [p for p in (os.path.join(out_dir, name + ".mp4"),
                        os.path.join(out_dir, name + ".flac"),
                        os.path.join(out_dir, name + ".receipt.json"))
            if os.path.exists(p)]


def write_review_mix(clip_path, voice_path, out_path):
    """``(path, error)`` -- writes the ducked review copy and NEVER raises.

    THE REVIEW MIX MAY NOT COST THE RECEIPT. It runs after the clip has been
    validated and written to its durable path, so an ffmpeg that is missing or
    that chokes on the supplied voice must not take down a deliverable that
    already succeeded. An uncaught raise here would leave a valid
    audio-carrying clip on disk with NO receipt beside it -- breaking the one
    thing this runner's docstring promises.

    Returns the error as a STRING rather than swallowing it: the caller records
    it in the receipt and exits non-zero, so a missing review copy is loud
    without being fatal to the clip.
    """
    try:
        subprocess.check_call(ducked_mux_cmd(clip_path, voice_path, out_path))
        return out_path, ""
    except Exception as exc:  # noqa: BLE001 -- never lose the deliverable
        return "", "%s: %s" % (type(exc).__name__, exc)


def probe_streams(path, *, ffprobe="ffprobe"):
    """``{"video": n, "audio": n, "duration": float, "w": int, "h": int,
    "fps": str, "frames": int}`` for a rendered clip. Raises on a bad probe."""
    out = subprocess.check_output(
        [ffprobe, "-v", "error", "-count_packets",
         "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,nb_read_packets",
         "-show_entries", "format=duration,nb_streams",
         "-of", "json", path], text=True, encoding="utf-8")
    data = json.loads(out)
    info = {"video": 0, "audio": 0, "w": 0, "h": 0, "fps": "", "frames": 0,
            "duration": float(data.get("format", {}).get("duration") or 0.0)}
    for st in data.get("streams", []):
        kind = st.get("codec_type")
        if kind == "video":
            info["video"] += 1
            info["w"] = int(st.get("width") or 0)
            info["h"] = int(st.get("height") or 0)
            info["fps"] = str(st.get("r_frame_rate") or "")
            info["frames"] = int(st.get("nb_read_packets") or 0)
        elif kind == "audio":
            info["audio"] += 1
    return info


# --------------------------------------------------------------------------- #
# Live helpers
# --------------------------------------------------------------------------- #
def assert_node_classes_installed(url=COMFYUI_URL, *, want_stem=False):
    """Confirm every class LIVE before submitting (lane 7's rule).

    Collects every miss into ONE error: naming them one at a time turns a fresh
    install into a sequence of failed renders.
    """
    info = requests.get("%s/object_info" % url, timeout=120).json()
    need = list(REQUIRED_NODE_CLASSES) + ([STEM_NODE_CLASS] if want_stem else [])
    missing = [c for c in need if c not in info]
    if missing:
        raise RuntimeError(
            "the running ComfyUI is missing node class(es) %s -- MiniMax H3 "
            "needs a build carrying comfy_extras/nodes_minimax_h3.py; update "
            "ComfyUI and restart" % ", ".join(missing))
    return info


#: Classified by EXTENSION, not by the /history key. `SaveVideo` reports its mp4
#: under `images` (with `animated: [true]`) rather than `videos` -- the key names
#: are a UI convention and this one does not mean what it says. Keying on it made
#: the first successful render report "no video output for this prompt".
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")
AUDIO_EXTS = (".flac", ".wav", ".mp3", ".opus", ".ogg")


def artifact_from_history(prompt_id, url=COMFYUI_URL):
    """``(videos, audios)`` this prompt produced, READ OFF /history.

    Never a directory glob: a glob picks up whatever else the box wrote while
    this ran, which is how a runner reports someone else's file as its own.
    """
    hist = requests.get("%s/history/%s" % (url, prompt_id),
                        timeout=30).json().get(prompt_id, {})
    videos, audios = [], []
    for node_out in (hist.get("outputs") or {}).values():
        for rows in node_out.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not (isinstance(row, dict) and row.get("filename")):
                    continue
                ext = os.path.splitext(row["filename"])[1].lower()
                if ext in VIDEO_EXTS:
                    videos.append(row)
                elif ext in AUDIO_EXTS:
                    audios.append(row)
    return videos, audios


def comfy_output_root():
    """Where the SERVER writes. The launcher pins --output-directory to the real
    output tree, which is also what OTR_OUTPUT_DIR names."""
    return os.environ.get(
        "OTR_OUTPUT_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output")


def resolve_saved(row):
    root = comfy_output_root()
    sub = row.get("subfolder") or ""
    return os.path.join(root, sub, row["filename"])


def stage_portrait(path, url=COMFYUI_URL):
    """UPLOAD the portrait through the server's own endpoint, and return the
    name ``LoadImage`` will resolve.

    NOT `wrapper_bridge.stage_into_comfy_input`, and the difference is exactly
    the difference between an in-process engine and a standalone runner. That
    helper asks `folder_paths` for the input directory -- correct INSIDE the
    ComfyUI process, where every registered adapter runs. This script is a
    separate process: `folder_paths` does not import, the helper falls back to a
    path derived from THIS repo's location, and the file lands in
    `Documents\\ComfyUI\\input` while the server reads
    `ComfyUI-Installs\\ComfyUI\\ComfyUI\\input`. The graph then validates
    against a filename that is not there ("Invalid image file"), which is how
    this runner failed its first live attempt.

    `POST /upload/image` is the server's own answer to "put a file where I can
    load it", so it is right by construction on any install layout -- no path
    guessing, no assumption about where ComfyUI lives.
    """
    with open(path, "rb") as fh:
        resp = requests.post(
            "%s/upload/image" % url,
            files={"image": (os.path.basename(path), fh, "image/png")},
            data={"overwrite": "true"}, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    name = body.get("name") or os.path.basename(path)
    sub = body.get("subfolder") or ""
    return "%s/%s" % (sub, name) if sub else name


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--portrait", required=True,
                    help="the still the mime acts from (LoadImage first_frame)")
    ap.add_argument("--seconds", type=float, default=3.75,
                    help="target duration; snapped UP to H3's 17k+5 grid")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=MIME_W)
    ap.add_argument("--height", type=int, default=MIME_H)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--name", default="", help="deliverable basename stem")
    ap.add_argument("--stem-wav", action="store_true",
                    help="also export the invented score as a lossless FLAC")
    ap.add_argument("--mux-tts", default="",
                    help="ALSO write a review copy with this voice over the "
                         "ducked music (both originals preserved)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing deliverables that already exist "
                         "under this name (refused by default)")
    ap.add_argument("--timeout", type=int, default=3000)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.portrait):
        print("[mime] portrait not found: %s" % args.portrait, flush=True)
        return 1
    if args.mux_tts and not os.path.isfile(args.mux_tts):
        print("[mime] --mux-tts voice not found: %s" % args.mux_tts, flush=True)
        return 1
    if int(args.width) % 32 or int(args.height) % 32:
        print("[mime] canvas %dx%d is not /32-legal; every H3 node declares "
              "width/height at step=32" % (args.width, args.height), flush=True)
        return 1

    model_frames = model_frames_for_seconds(args.seconds)
    secs = delivered_seconds(model_frames)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    name = args.name or "h3_mime_%dx%d_f%d_seed%d_%s" % (
        args.width, args.height, model_frames, args.seed, stamp)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[mime] target %.3f s -> model f%d at %d fps = %.3f s delivered"
          % (args.seconds, model_frames, H3_MODEL_FPS, secs), flush=True)
    if below_trained_band(model_frames):
        print("[mime] NOTE f%d is BELOW the trained band (%d..%d) the "
              "registered H3 lanes enforce. Legal on the grid and what the "
              "lab's two passing mime legs used (f90); said out loud so a "
              "receipt is never read as a trained-band render."
              % (model_frames, H3_TRAINED_MIN_MODEL_FRAMES,
                 H3_TRAINED_MAX_MODEL_FRAMES), flush=True)

    assert_node_classes_installed(want_stem=args.stem_wav)
    image_name = stage_portrait(args.portrait)

    prefix = "otr_mime/%s" % name
    graph = build_mime_graph(
        image_name=image_name, prompt=args.prompt, model_frames=model_frames,
        seed=args.seed, width=args.width, height=args.height,
        filename_prefix=prefix,
        stem_prefix=("otr_mime/%s_score" % name) if args.stem_wav else None)

    t0 = time.time()
    prompt_id = submit_prompt(graph)
    print("[mime] QUEUED %s" % prompt_id, flush=True)
    status, err = poll_history(prompt_id, timeout_s=args.timeout)
    elapsed = round(time.time() - t0, 1)
    print("[mime] history status=%s %s" % (status, err[:300]), flush=True)
    if status != "SUCCESS":
        print("[mime] FAIL: render did not succeed", flush=True)
        return 1

    videos, stems = artifact_from_history(prompt_id)
    if not videos:
        print("[mime] FAIL: /history lists no video output for this prompt",
              flush=True)
        return 1
    if len(videos) != 1:
        # Symmetric with the stream asserts below: this graph has exactly ONE
        # SaveVideo, so more than one row means the prompt did something this
        # runner does not model. Taking [0] and dropping the rest would deliver
        # one file and silently discard others.
        print("[mime] FAIL: /history lists %d video outputs for this prompt "
              "and this graph has exactly one SaveVideo -- refusing rather "
              "than picking one: %s"
              % (len(videos), [v.get("filename") for v in videos]), flush=True)
        return 1
    src = resolve_saved(videos[0])
    if not os.path.isfile(src):
        print("[mime] FAIL: /history named %s but it is not on disk" % src,
              flush=True)
        return 1

    clip_path = os.path.join(args.out_dir, name + ".mp4")
    # A DELIVERABLE IS NEVER SILENTLY OVERWRITTEN. The default name carries the
    # seed and a one-SECOND timestamp, so two runs with the same seed inside the
    # same second -- a scripted retry loop -- collide, and an explicit --name
    # collides on every repeat. These are outputs the operator keeps, so a
    # collision is refused by name unless --overwrite says otherwise.
    existing = existing_deliverables(args.out_dir, name)
    if existing and not args.overwrite:
        print("[mime] FAIL: these deliverables already exist and would be "
              "overwritten: %s. Pass --overwrite, or give a different --name."
              % ", ".join(os.path.basename(p) for p in existing), flush=True)
        return 1
    with open(src, "rb") as fh_in, open(clip_path, "wb") as fh_out:
        fh_out.write(fh_in.read())

    info = probe_streams(clip_path)
    # THE INVERTED ASSERT. Every other H3 path in this repo proves SILENCE; this
    # one exists to keep the score, so a silent mime clip is a FAILED mime clip.
    if info["audio"] != 1:
        print("[mime] FAIL: the clip carries %d audio stream(s). This lane's "
              "whole purpose is its invented score -- a silent mime render is "
              "a failure, not a quiet success." % info["audio"], flush=True)
        return 1
    if info["video"] != 1:
        print("[mime] FAIL: expected exactly one video stream, got %d"
              % info["video"], flush=True)
        return 1

    stem_path = ""
    if args.stem_wav:
        if not stems:
            print("[mime] FAIL: --stem-wav was asked for and /history lists no "
                  "audio output", flush=True)
            return 1
        stem_src = resolve_saved(stems[0])
        stem_path = os.path.join(args.out_dir,
                                 name + os.path.splitext(stem_src)[1])
        with open(stem_src, "rb") as fh_in, open(stem_path, "wb") as fh_out:
            fh_out.write(fh_in.read())

    # THE REVIEW MIX MAY NOT COST THE RECEIPT. It runs AFTER the clip has been
    # validated and written to its durable path, so an ffmpeg that is missing or
    # that chokes on the supplied voice must not take down a deliverable that
    # already succeeded -- an uncaught raise here would leave a valid
    # audio-carrying clip on disk with NO receipt beside it, which is exactly
    # the contract this runner promises. The failure is recorded IN the receipt
    # and reported at the end; the primary deliverable stands.
    mux_path, mux_error = "", ""
    if args.mux_tts:
        mux_path, mux_error = write_review_mix(
            clip_path, args.mux_tts,
            os.path.join(args.out_dir, name + "_with_voice_REVIEW.mp4"))
        if mux_error:
            print("[mime] review mix FAILED (%s) -- the clip and its receipt "
                  "are still written; only the optional review copy is "
                  "missing" % mux_error, flush=True)

    receipt = {
        "runner": "otr_h3_mime_runner",
        "lane": "h3_low_mime",
        "registered_engine": False,
        "prompt_id": prompt_id,
        "elapsed_s": elapsed,
        "portrait": os.path.abspath(args.portrait),
        "prompt": args.prompt,
        "seed": args.seed,
        "canvas": "%dx%d" % (args.width, args.height),
        "model_frames": model_frames,
        "model_fps": H3_MODEL_FPS,
        "delivered_s": round(secs, 6),
        "below_trained_band": below_trained_band(model_frames),
        "canvas_fps_NOT_applied": H3_CANVAS_FPS,
        "delivery_rate_note": (
            "delivered at the MODEL's 24 fps, deliberately un-converted: this "
            "clip carries its own generated score and never enters the 25 fps "
            "episode timeline, so resampling the picture would desync it"),
        "recipe": dict(H3_RECIPE),
        "probed": info,
        "clip": clip_path,
        "stem": stem_path,
        "review_mix": mux_path,
        "review_mix_error": mux_error,
        "audio_policy": (
            "KEEPS its audio (G5.2 exemption). Never enters episode assembly; "
            "V-1 is unaffected because only OTR_MasterAudioMux adds audio to an "
            "EPISODE."),
    }
    receipt_path = os.path.join(args.out_dir, name + ".receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)

    print("[mime] PASS  %s" % clip_path, flush=True)
    print("[mime]   %dx%d  %s  %d frames  %.3f s  audio streams=%d"
          % (info["w"], info["h"], info["fps"], info["frames"],
             info["duration"], info["audio"]), flush=True)
    if stem_path:
        print("[mime]   score stem: %s" % stem_path, flush=True)
    if mux_path:
        print("[mime]   review mix: %s" % mux_path, flush=True)
    print("[mime]   receipt: %s" % receipt_path, flush=True)
    if mux_error:
        # The clip is good and its receipt is written, but the operator ASKED
        # for a review copy and did not get one -- so the exit code says so.
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
