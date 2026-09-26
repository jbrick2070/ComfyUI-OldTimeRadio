# Install

From "I have ComfyUI" to "the OTR nodes are in the menu." Most of it
waiting on pip.

If you only read one line: install the pack, make sure `ffmpeg` and `ffprobe` are
on your PATH, restart ComfyUI, and look for `[OldTimeRadio]` in the console.

---

## Get the pack

**Through the Node Manager** (easiest). In ComfyUI, open **Extensions -> Node
Manager** and search for **old time radio**, then install it. The registry id is
`comfyui-old-time-radio`, published by `fluxus`.

**Or clone it yourself**, into your ComfyUI `custom_nodes/` folder:

```bash
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
```

`main` is the branch: it is the default and where the work lands. If you
cloned before 2026-09-13, clone again rather than pull.

**What you get is not the same either way.** A Manager install is a trimmed
bundle: it has the nodes, the workflows and this folder, but not `docs/`, not
`tests/`, and not most of `scripts/`. Anything in this folder that asks you to
run a script says so explicitly and tells you which install you need.

## Python libraries

Into **ComfyUI's own interpreter**, not a system Python and not a different
virtual environment. If you use the portable build that means the Python inside
the portable folder; if you made a venv for ComfyUI, activate it first.

```bash
python -m pip install -r requirements.txt
```

Manager usually does this for you. If it did, skip **this step only** -- do not
skip ffmpeg. A missing ffprobe does not stop the nodes loading, and since the
2026-09-11 PyAV fallback it no longer stops a normal episode either -- the pack
reads durations through PyAV when no binary resolves. Some optional engines
still refuse without a real ffprobe, which is why the instruction is still to
install both.

## ffmpeg and ffprobe

**Both.** Not just ffmpeg. Episodes are muxed, captioned and published through
ffmpeg, and several steps probe a file's duration with ffprobe first. The
`imageio-ffmpeg` wheel that comes down with the requirements ships an ffmpeg
binary and **no ffprobe**, so a system install is what you want.

**Take a current build.** The final mux copies the master audio into the MP4
losslessly (PCM in an ISOBMFF container), which older builds cannot write:
they render the whole episode and then fail at that last step with "Could not
find tag for codec pcm_s16le", leaving a zero-byte file. Ubuntu 22.04's
`apt install ffmpeg` is 4.4 and does exactly that.

The floor is 6.1, where FFmpeg's MP4 muxer gained PCM. Measured on
2026-09-13 by running this pack's own probe on four builds across three
machines: 4.4.2 fails; 7.0.2-static, 8.0.1 and 9.0 pass. No 6.x build has
been run, so the floor is documented rather than measured.
**You do not have to work this out.** Nothing reads the version number --
`OTR_WorkflowValidator` muxes a fifth of a second of silence when a workflow
containing the mux is queued, and refuses in about a second if the build
cannot do it.

- **Windows:** `winget install Gyan.FFmpeg`
- **macOS:** `brew install ffmpeg`
- **Debian/Ubuntu:** `sudo apt install ffmpeg`, then `ffmpeg -version`; if it
  is older than 6.1, install a static build instead (for example the release
  build from johnvansickle.com into `/usr/local/bin`) or set `OTR_FFMPEG` to one.

If you cannot install system-wide, `ffdl install` fetches both binaries into
your user profile -- `ffmpeg-downloader` comes down with the requirements above
and provides the fetcher.

Check both answer. On macOS or Linux:

```bash
ffmpeg -version; ffprobe -version
```

On Windows PowerShell use `;` as above — `&&` is a parser error there.

If you used `ffdl install` rather than a system package, these may report
"not found" and still be fine: that installs into its own profile directory and
this pack locates the pair through the `ffmpeg-downloader` package, not through
PATH.

If you keep ffmpeg somewhere unusual, set `OTR_FFMPEG` to the full path of the
binary. That environment variable wins over PATH. (There is an `ffmpeg` widget on
some nodes; it is ignored, deliberately — a saved workflow is not allowed to
choose which binary runs on your machine.)

**On Linux, also install one monospace font.** Burned captions and the credit
roll need a real TTF; a headless server image often has none at all.
`fonts-dejavu-core` is enough for Latin scripts. **Japanese and Mandarin need a
CJK font as well -- `apt install fonts-noto-cjk`. Without one the text still
renders, as a row of empty boxes, and nothing warns you: measured on a fresh
container that carried eight DejaVu fonts and zero CJK coverage.**

## Restart ComfyUI and read the console

Restart fully -- the Node Manager's "reload" is not enough for a new pack.

In the console you want to see `[OldTimeRadio]` lines and no traceback. In the
node menu you want an **OldTimeRadio** category.

**If one node is missing**, look for a line like:

```
[OldTimeRadio] Skipped 'OTR_SomeNode': No module named 'something'
```

That is the pack working as designed. Each node is loaded in its own
try/except so that one missing library costs you one node instead of the whole
pack. The fix is to install the library it names, into ComfyUI's interpreter.

**If the whole category is missing**, it is not a missing library — a missing
library never zeroes out the pack. Look for the pack being in the wrong folder,
or for a crash during ComfyUI's startup, before any node loaded.

## Python versions

| Your Python | Voices | Notes |
|---|---|---|
| 3.10 – 3.12 | Kokoro on torch | The multilingual path; every admitted language. |
| 3.13 | Kokoro through `kokoro-onnx`, on CPU | English works and the ONNX runtime is picked automatically. Non-English rows require the torch path above. |
| 3.14 and up | Refused | No Kokoro build exists yet. The provisioner refuses rather than installing something that cannot speak. |

Japanese and Mandarin also need their `misaki[ja]` / `misaki[zh]` readiness
extras. They are checked only when that language is selected. Install the one
you need with ComfyUI's own Python:
`<ComfyUI Python> -m pip install "misaki[ja]"` or
`<ComfyUI Python> -m pip install "misaki[zh]"`. See
[MULTILINGUAL.md](MULTILINGUAL.md).

## Weights: what you do not have to do

**Two things the pack deletes, so you are not surprised by either.**

- **Always:** on every ComfyUI start, and again after each publish, it sweeps
  `<output>/otr/episodes/_shared/tmp` and removes anything older than 24 hours.
  That folder is scratch space the render writes through; nothing you are meant
  to keep lives there, and the sweep touches no other folder, skips anything
  locked, logs every deletion and never fails a render. Set
  `OTR_TMP_SWEEP_MAX_AGE_S` to change the age, and do not park files under that
  path.
- **Only if you ask:** `asset_cleanup` on **OTR_LedgerScriptWriter** ships
  `off`. Set it to `partial` or `full` and, once an episode is published, that
  episode's own folder in `otr/episodes/` loses its sound and pictures
  (`partial`) or goes altogether (`full`). `otr/obs/` is never touched. See
  [RUN.md](RUN.md#saving-disk-space).

**Nothing by hand.** There is no setup script to run and no model to place
yourself. Almost everything arrives the first time you queue: the one exception
is the Kokoro voice set, which the pack fetches at STARTUP so the default voice
is ready before you ever press Run -- a few hundred MB on Python 3.13, where
it also brings the ONNX model. If the console pauses on `[OldTimeRadio]` lines
during a restart, that is what it is doing.

What comes down on that first run is about **12 GB**:

| What | Size | From |
|---|---|---|
| The writer — `Qwen/Qwen3.5-4B` | ~8.7 GB | Hugging Face, ungated |
| Music — Stable Audio 3 checkpoint | 2.11 GiB | `Comfy-Org/stable-audio-3` |
| Music — its text encoder | 1.11 GiB | `Comfy-Org/stable-audio-3` |
| Voices — Kokoro | a few hundred MB | with the Python package |

The canonical's procedural video lanes draw their own frames
and consume no still image -- so the image model its dropdowns name stays
**dormant** and is not downloaded. Switch a video lane to one that consumes a
still and the image weights are fetched then, not before.

Everything else -- the lanes you have to fetch by hand, which repository each file
comes from, and which folder to put it in -- is in
[MACHINES.md](MACHINES.md#where-do-the-manual-weights-come-from).

## Hugging Face login

Most of what this pack uses is ungated. A few of the heavier video lanes are not,
and for those you accept the licence on the model's Hugging Face page while
signed in, then log in locally:

```bash
python -m pip install huggingface_hub
hf auth login
```

That stores a token in your Hugging Face config. **Do not paste a token into a
workflow widget** — no node here asks for one, and a token saved in a
workflow travels with it.

**Everything lands inside your ComfyUI models tree**, not in your home
directory, in two places:

- **The writer** (`Qwen/Qwen3.5-4B` on most workflows, ~8.7 GB) goes to
  `models/LLM/Qwen--Qwen3.5-4B/` as ordinary files -- a normal ComfyUI model
  folder, relocatable with an `LLM:` entry in `extra_model_paths.yaml`. No
  symlinks, so Windows never asks for Developer Mode. If you already have the
  writer in the Hugging Face cache from an earlier version, it stays there and
  keeps working; nothing is moved or downloaded twice.
- **The rest** (the music model and its text encoder) goes to the Hugging Face
  cache: if `HF_HOME` is unset, this pack points it at
  `ComfyUI/models/huggingface` during startup.

If that volume is short of room, set `HF_HOME` yourself **before launching
ComfyUI**. Setting it later does not move the cache, it adds a second one, and
you end up with two copies of everything.

## Node packs, only for some lanes

Most lanes need nothing beyond this pack. A few build their picture out of classes
that belong to someone else's pack, and those are listed per engine in
[MACHINES.md](MACHINES.md#which-workflow-do-i-open) -- with which of the shipped
workflows needs which. If you pick one of those lanes without its pack, the render
stops with an error naming the pack and its URL.

---

Next: [RUN.md](RUN.md) — making your first episode.
