# The upscale step -- and why it ships off

`upscale_engine` lives on the **OTR_SilentComposite** node. That node is the
one that takes every rendered clip, every still plate and the base video and
cuts them into one silent timeline at the episode's canvas -- 1920x1080 in the
canonical graph and in every saved variant. The upscale step runs
*inside* that assembly, clip by clip, after the clips are rendered and before
the captions are burned, the credits rolled and the audio muxed back on.

It is optional, and every shipped graph sets it to **off**. That is a real
choice, not an unfinished one -- see the end of this page.

---

## The two choices

| `upscale_engine` | What it does | What it needs from you |
|---|---|---|
| **`off`** -- ships this way | Leaves the composite's own resize in charge | Nothing |
| **`spandrel_esrgan`** | Runs Real-ESRGAN x2plus over each eligible clip, one frame at a time, then fits the result to the canvas | One 64 MB file you download by hand |

**Those are the only two.** The dropdown is built from the pack's live upscale
registry, so a third would appear by itself the day one ships. Today the list
is two long, and this page is short for that reason rather than because it is
unfinished.

There is a second box beside it, `upscale_device`, covered below.

---

## `off` does not mean "no resizing"

Worth knowing before you go looking for a sharper picture: with `off`, clips
and real still plates are still scaled up to the canvas by ffmpeg, using a
lanczos resample plus a light sharpen pass. That pass exists because the
project measured the problem -- a native render looks soft once it reaches the
1920x1080 canvas, and lanczos plus a gentle unsharp raised the measured
sharpness by about nine percent at no GPU cost at all. The procedural floor,
the black gap fill and the credits roll are deliberately left alone.

That Lanczos path is how **standard OTR** lands a true 1920x1080 file
(`composite_res` on every shipping graph). Basic is the cheap generate
(Veo lite at 720p, native still size). Standard is this composite. No
extra engine, no Veo 1080p generate, no Flow 1080 download step.

If you only want a little more bite, the sharpen amount is an environment
variable, `OTR_COMPOSITE_UNSHARP_AMOUNT`. It defaults to `0.4`; `0.8` is the
heavier setting the same comparison exposed, and it can put halos around faces.
That is a free knob. The model below is not.

---

## Turning it on: the one download

`spandrel_esrgan` needs one checkpoint, and **nothing fetches it for you.**

| | |
|---|---|
| File | `RealESRGAN_x2plus.pth` |
| Where it goes | `models/upscale_models/` under your ComfyUI folder |
| Size | 67,061,725 bytes -- about 64 MB |
| From | `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth` |
| Licence | BSD-3-Clause |

The library half is already handled: `spandrel` is a declared dependency of the
pack, so a Node Manager install has it. It is only the weights that are manual.

The pack checks the file's SHA-256 against a pinned value before loading it, so
a half-finished download or a different file with the same name is refused by
name rather than quietly used.

**Download it before you flip the switch.** The engine is loaded at composite
time, which is near the *end* of a run -- the script is written, the cast is
voiced, the music is scored and every clip is rendered by then. A missing
checkpoint stops the render at that point and the error names the file, the
folder and the URL. Nothing is silently substituted, and nothing is silently
skipped, but you will have paid for the whole episode first.

**Only that one filename counts.** The engine reads exactly
`RealESRGAN_x2plus.pth`. Other Real-ESRGAN checkpoints sitting in the same
folder -- x4plus and the rest -- are not alternatives and are not picked up;
they simply do nothing.

If you installed from a git clone rather than the Node Manager, there is a
helper at `scripts/ensure_upscale_models.py` that downloads and verifies it for
you. `scripts/` is not part of the Manager bundle, so for most readers the
download above is the path.

---

## The device box

`upscale_device` is a typed box, not a menu. It accepts exactly four shapes:

| Value | Meaning |
|---|---|
| `cpu` | Ships this way. Works everywhere. |
| `cuda` | First NVIDIA GPU. An AMD card running a ROCm build of torch also reports as `cuda`. |
| `cuda:0`, `cuda:1`, ... | A specific GPU, when you have more than one. |
| `mps` | Apple Silicon. |

Anything else is refused by name -- and so is `cuda` on a machine with no CUDA,
or `mps` on a machine with no Metal. It never quietly falls back to the CPU,
because a run that silently ignores what you asked for is harder to explain
than one that stops.

While `upscale_engine` is `off`, this box is not read at all, so a stale value
there cannot break anything.

`cpu` is the default because it works on every machine, not because it is fast.
If you have a GPU with room, that is where this belongs. Check your machine's
row in [MACHINES.md](MACHINES.md) section 2 first.

---

## You turned it on and nothing happened

Three things make the pack skip the model on purpose. All three say so in the
ComfyUI console, so read it before assuming a break.

1. **Every clip already stated how it must be enlarged.** The AnimateDiff
   lanes -- `animatediff15_lightning_video`,
   `animatediff15_v3_haunted_video`, `animatediff15_v3_stillin_lab_video` --
   declare their own clean full-frame enlargement, and the upscaler is not
   allowed to overrule it. On an episode made entirely of those, the engine is
   not even loaded. The log line begins `upscale SKIPPED: no model-eligible
   clip`.
2. **The clip is already as big as the canvas.** Upscaling it 2x and shrinking
   it back would change pixels without getting anywhere, so the composite takes
   the ordinary path instead and logs `model skip (source ... >= canvas ...)`.
3. **The segment is not a real clip.** The procedural floor, the black gap fill
   and the credits roll never go through the model.

When it *does* run, it says so: `upscale MODEL PATH for ...` names the engine,
the device and the geometry, and `upscale engine LOADED:` names the exact
checkpoint file it found. If neither line is in your log, the model did not run.

---

## What it costs

The model works one frame at a time -- that is the library's own contract, not
a setting -- so the cost scales with how many frames your episode has, not with
how many clips.

The one place it gets cheap by itself is a held image. On a still-based lane a
beat is one picture held for its whole length, so consecutive frames are
identical; the pack notices, reuses the previous result, and a whole beat costs
one model call instead of hundreds. The console reports that as
`upscale HELD-FRAME REUSE`. On a genuinely moving lane every frame is different
and every frame is paid for.

The 64 MB checkpoint is small next to the writer model or a video model, so
loading it is not the expensive part. The frames are.

---

## Off is a legitimate place to leave it

Every graph this pack ships -- the canonical and every variant -- sets
`upscale_engine` to `off`, and that is the recommendation, not an oversight:

- The composite already resamples and sharpens on the way to the canvas, and
  that path was tuned against a measured comparison.
- The one available model needs a manual download, so the shipped default has
  to be the one that works on a fresh install.
- It is a radio drama. The picture is a visualizer or a procedural lane, and
  super-resolution has less to give that than it would give a photograph.

Turn it on when you are feeding the composite real rendered clips that land
well under 1920x1080 and you want more detail in them than a resample can
invent. Otherwise `off` is a finished answer.
