# Install

From "I have ComfyUI" to "the OTR nodes are in the menu." Ten minutes, most of it
waiting on pip.

If you only read one line: install the pack, make sure `ffmpeg` and `ffprobe` are
on your PATH, restart ComfyUI, and look for `[OldTimeRadio]` in the console.

---

## 1. Get the pack

**Through ComfyUI Manager** (easiest). Search for **Old Time Radio** and install
it. The registry id is `comfyui-old-time-radio`, published by `fluxus`.

**Or clone it yourself**, into your ComfyUI `custom_nodes/` folder:

```bash
git clone -b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio
```

The branch matters. `v2.0-alpha` is the default branch and the only one that is
current; `main` is a stale v1.7 merge thousands of commits behind.

**What you get is not the same either way.** A Manager install is a trimmed
bundle: it has the nodes, the workflows and this folder, but not `docs/`, not
`tests/`, and not most of `scripts/`. Anything in this folder that asks you to
run a script says so explicitly and tells you which install you need.

## 2. Install the Python libraries

Into **ComfyUI's own interpreter**, not a system Python and not a different
virtual environment. If you use the portable build that means the Python inside
the portable folder; if you made a venv for ComfyUI, activate it first.

```bash
python -m pip install -r requirements.txt
```

Manager usually does this for you. If it did, skip ahead — step 4 will tell you
whether it worked.

## 3. ffmpeg and ffprobe

**Both.** Not just ffmpeg. Episodes are muxed, captioned and published through
ffmpeg, and several steps probe a file's duration with ffprobe first. The
`imageio-ffmpeg` wheel that comes down with the requirements ships an ffmpeg
binary and **no ffprobe**, so a system install is what you want.

- **Windows:** `winget install Gyan.FFmpeg`
- **macOS:** `brew install ffmpeg`
- **Debian/Ubuntu:** `sudo apt install ffmpeg`

If you cannot install system-wide, `ffdl install` fetches both binaries into
your user profile — `ffmpeg-downloader` comes down with the requirements in step
2 and provides the fetcher.

Check both answer:

```bash
ffmpeg -version && ffprobe -version
```

If you keep ffmpeg somewhere unusual, set `OTR_FFMPEG` to the full path of the
binary. That environment variable wins over PATH. (There is an `ffmpeg` widget on
some nodes; it is ignored, deliberately — a saved workflow is not allowed to
choose which binary runs on your machine.)

**On Linux, also install one monospace font.** Burned captions and the credit
roll need a real TTF; a headless server image often has none at all.
`fonts-dejavu-core` is enough.

## 4. Restart ComfyUI and read the console

Restart fully — Manager's "reload" is not enough for a new pack.

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

## 5. Python versions

| Your Python | Voices | Notes |
|---|---|---|
| 3.10 – 3.12 | Kokoro on torch | The most tested path. |
| 3.13 | Kokoro through `kokoro-onnx`, on CPU | Works; the ONNX runtime is picked automatically. |
| 3.14 and up | Refused | No Kokoro build exists yet. The provisioner refuses rather than installing something that cannot speak. |

## 6. Weights: what you do not have to do

**Nothing, to start.** The canonical workflow downloads what it needs the first
time you queue it, and nothing before that. There is no setup script to run and
no model to place by hand.

What comes down on that first run is about **12 GB**:

| What | Size | From |
|---|---|---|
| The writer — `Qwen/Qwen3.5-4B` | ~8.7 GB | Hugging Face, ungated |
| Music — Stable Audio 3 checkpoint | 2.11 GiB | `Comfy-Org/stable-audio-3` |
| Music — its text encoder | 1.11 GiB | `Comfy-Org/stable-audio-3` |
| Voices — Kokoro | a few hundred MB | with the Python package |

The canonical's three video lanes are procedural — they draw their own frames
and consume no still image — so the image model its dropdowns name stays
**dormant** and is not downloaded. Switch a video lane to one that consumes a
still and the image weights are fetched then, not before.

Everything else — the lanes you have to fetch by hand, which repository each file
comes from, and which folder to put it in — is in
[MACHINES.md](MACHINES.md), section 3.

## 7. A Hugging Face account, only if you want the gated lanes

Most of what this pack uses is ungated. A few of the heavier video lanes are not,
and for those you accept the licence on the model's Hugging Face page while
signed in, then log in locally:

```bash
python -m pip install huggingface_hub
hf auth login
```

That stores a token in your Hugging Face config. **Do not paste a token into a
workflow widget** — no node here asks for one, and a token saved in a graph
travels with the graph.

Set `HF_HOME` **before** anything downloads if you want the cache somewhere with
room. Setting it later does not move the cache, it adds a second one, and you
will have two copies of everything.

## 8. Node packs, only for some lanes

Most lanes need nothing beyond this pack. A few build their graph out of classes
that belong to someone else's pack, and those are listed per engine in
[MACHINES.md](MACHINES.md), section 1 — with which of the five shipped machine
graphs needs which. If you pick one of those lanes without its pack, the render
stops with an error naming the pack and its URL.

---

Next: [RUN.md](RUN.md) — making your first episode.
