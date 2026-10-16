# The upscale step -- on for NVIDIA, off everywhere else

`upscale_engine` lives on the **OTR_SilentComposite** node. That node is the
one that takes every rendered clip, every still plate and the base video and
cuts them into one silent timeline at the episode's canvas -- 1920x1080 in the
canonical graph and in every saved variant. The upscale step runs
*inside* that assembly, clip by clip, after the clips are rendered and before
the captions are burned, the credits rolled and the audio muxed back on.

It is optional. The NVIDIA workflows (8, 16 and 24 GB) set it to `rtx_vsr`
on `cuda`; the canonical graph and the Mac, AMD, cloud and Google workflows set
it to **off**. See the end of this page for why each.

---

## The choices

| `upscale_engine` | What it does | What it needs from you |
|---|---|---|
| **`off`** -- ships this way | Leaves the composite's own resize in charge | Nothing |
| **`spandrel_esrgan`** | Runs Real-ESRGAN x2plus over each eligible clip, one frame at a time, then fits the result to the canvas | Nothing: its one 64 MB file downloads itself the first time the engine loads |
| **`rtx_vsr`** | NVIDIA RTX Video Super Resolution: the same upscaler an RTX card applies to browser video, x2, then fitted to the canvas | An NVIDIA RTX card. Its ~540 MB NVIDIA package installs itself the first time it runs (below) |

**Those are the ones that ship.** The dropdown is built from the pack's live upscale
registry, so a new engine would appear by itself the day one ships. Today the list
is short for that reason rather than because it is unfinished.

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
(a 720p provider clip, native still size). Standard is this composite. No
extra engine, no 1080p provider generate, no Flow 1080 download step.

If you only want a little more bite, the sharpen amount is an environment
variable, `OTR_COMPOSITE_UNSHARP_AMOUNT`. It defaults to `0.4`; `0.8` is the
heavier setting the same comparison exposed, and it can put halos around faces.
That is a free knob. The model below is not.

---

## Turning it on: the one download

`spandrel_esrgan` needs one checkpoint, and **the engine fetches it itself** the
first time it loads.

| | |
|---|---|
| File | `RealESRGAN_x2plus.pth` |
| Where it goes | `models/upscale_models/` under your ComfyUI folder |
| Size | 67,061,725 bytes -- about 64 MB |
| Fetched from | the pinned copy in `nateraw/real-esrgan` on Hugging Face |
| By hand | `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth` |
| Licence | BSD-3-Clause |

The library half is handled too: `spandrel` is a declared dependency of the
pack, so a Node Manager install has it.

The pack checks the file's SHA-256 against a pinned value before loading it, so
a half-finished download or a different file with the same name is refused by
name rather than quietly used.

**The fetch happens late.** The engine is loaded at composite time, which is
near the *end* of a run -- the script is written, the cast is voiced, the music
is scored and every clip is rendered by then. If the download cannot complete
there, the render stops at that point and the error names the file, the folder
and the URL above. Nothing is silently substituted, and nothing is silently
skipped, but you will have paid for the whole episode first -- so on a machine
that will be offline, put the file in place beforehand.

**Only that one filename counts.** The engine reads exactly
`RealESRGAN_x2plus.pth`. Other Real-ESRGAN checkpoints sitting in the same
folder -- x4plus and the rest -- are not alternatives and are not picked up;
they simply do nothing.

A git clone also has `scripts/ensure_upscale_models.py`, which downloads the
file from the GitHub release ahead of time and checks the same SHA-256.
`scripts/` is not part of the Manager bundle, and nobody needs it: the engine's
own fetch lands the identical file.

---

## `rtx_vsr`: the fast one, for NVIDIA RTX cards

Measured on an RTX 4060 Laptop (8 GB), 48 frames of a real OTR clip at its
native 1472x832:

| | per frame, straight to 1920x1080 | fidelity (downscale x2, upscale x2, compare) |
|---|---|---|
| `rtx_vsr` | 11-14 ms | PSNR 35.8, SSIM 0.936, visibly crisper edges |
| `spandrel_esrgan` | about 1.8 s (fp16); at fp32 it spilled the 8 GB card and stalled | PSNR 34.5, SSIM 0.899 |
| `off` (lanczos + unsharp) | 37 ms (ffmpeg, same benchmark; 16 ms at the half-size fidelity shape) | PSNR 36.0, SSIM 0.944 |

So it is the upscaler to pick on an RTX card, at roughly a hundredth of
Real-ESRGAN's cost. `off` scores as well on fidelity -- VSR's gain is crisper
edges -- so it is the canonical graph's default, while every NVIDIA workflow
selects `rtx_vsr` (since 2026-10-15). First published with it on the RTX 4060
on 2026-10-16: AnimateDiff episodes upscaled clip by clip.

**The install happens by itself.** Its models ship inside NVIDIA's own
package, `nvidia-vfx`, so there is no checkpoint to fetch. The first time the
engine loads on an NVIDIA card -- which means a workflow picked it; only the
NVIDIA workflows do -- it runs `pip install nvidia-vfx` into ComfyUI's own
Python, once, and carries on. It is not a requirements.txt line, so it never
lands on an AMD, Intel or Mac machine.

It needs an RTX card (Turing, the RTX 20 series, or newer) and NVIDIA driver
570.65 or later on Windows, per NVIDIA's own package notes. It is about
540 MB, Windows and Linux x86-64 only, under NVIDIA's licence; running an
NVIDIA workflow on an NVIDIA card is taken as accepting it, as you did for the
driver. It skips the download where it could not help: a card older than
Turing (a GTX 10-series, say), ARM Linux, macOS, or Python older than 3.12.
If ComfyUI's Python has no pip (some ComfyUI Desktop installs use uv), it
installs with `uv pip` instead. If an install fails, the whole pip output is in
the ComfyUI log, and the server does not retry until it restarts. To install it
ahead of time, or on a machine whose first run is offline, run this with
ComfyUI's own Python:

```
pip install "nvidia-vfx>=0.2,<0.3"
```

Budget about 1.1 GB of disk the first time: the 540 MB package plus pip's
cached copy of it.

**It never costs you an episode.** If `rtx_vsr` cannot run where you picked
it -- the install failed, the card is AMD or an older GTX, the machine is a
Mac --
the composite says why in the log and in its report line, and finishes the
episode with `off`. The upscaler is the last stage of a run; everything before
it is already done.

**The device box does not matter for it.** It only runs on the NVIDIA GPU, so
the shipped `cpu` value means `cuda:0` for this engine. `OTR_RTX_VSR_QUALITY`
picks another of NVIDIA's quality levels (`ULTRA` is the default).

---

## The device box

`upscale_device` is a typed box, not a menu. It accepts these shapes:

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
row in [MACHINES.md](MACHINES.md#will-this-engine-run-on-my-machine) first.

---

## You turned it on and nothing happened

These make the pack skip the model on purpose. Each one says so in the
ComfyUI console, so read it before assuming a break.

- **Every clip already stated how it must be enlarged, and the engine is
  `spandrel_esrgan`.** The AnimateDiff lanes -- `animatediff15_lightning_video`,
  `animatediff15_v3_haunted_video`, `animatediff15_v3_stillin_lab_video` --
  declare a clean full-frame enlargement (no sharpening, no crop, no pad).
  Real-ESRGAN invents texture, so it is not allowed to overrule that; on an
  episode made entirely of those clips it is not even loaded, and the log line
  begins `upscale SKIPPED: no model-eligible clip`. `rtx_vsr` *does* upscale
  them (since 2026-10-15): it is an exact 2x that crisps edges without adding
  an unsharp pass or inventing texture, and its result is resized to fill the
  whole frame, as the clean chain would. The per-clip receipts still name the
  clips' declared mode (`lanczos_clean_full_frame`); the composite's log line
  `upscale MODEL PATH` is the record that VSR enlarged them.
- **The clip is already as big as the canvas.** Upscaling it 2x and shrinking
  it back would change pixels without getting anywhere, so the composite takes
  the ordinary path instead and logs `model skip (source ... >= canvas ...)`.
- **The segment is not a real clip.** The procedural floor, the black gap fill
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

For `spandrel_esrgan` the 64 MB checkpoint is small next to the writer model
or a video model, so loading it is not the expensive part. The frames are.
`rtx_vsr` has no checkpoint of its own (its models live in NVIDIA's package)
and its frames are cheap -- see its section above.

---

## Which graphs turn it on, and why off is still a fine answer

Every NVIDIA workflow sets `upscale_engine` to `rtx_vsr` on `cuda` (operator,
2026-10-15: "every nvidia lane to 1080p"). Several NVIDIA lanes render well
under 1920x1080 -- 832x480 for HuMo and the 8 GB LTX 2.5 lanes, 512x288 for
LTX 0.9.8 -- and that is where super-resolution earns its few milliseconds a
frame. It installs its package itself on the first run, and it cannot cost
the episode: if that install fails, the card cannot run it, or it fails
partway through a clip, the composite logs a warning and uses `off` instead.

The canonical graph and every non-NVIDIA workflow keep `off`, and that is the
recommendation there, not an oversight:

- The composite already resamples and sharpens on the way to the canvas, and
  that path was tuned against a measured comparison.
- `spandrel_esrgan`'s checkpoint downloads at the end of the first run that
  uses it, and a failed fetch there costs the whole episode; `rtx_vsr` needs an
  RTX card and a 540 MB NVIDIA package. The shipped default has to be the one
  that works on a fresh, offline install of any machine.
- It is a radio drama. The picture is a visualizer or a procedural lane, and
  super-resolution has less to give that than it would give a photograph.

Turn it on when you are feeding the composite real rendered clips that land
well under 1920x1080 and you want more detail in them than a resample can
invent. Otherwise `off` is a finished answer.
