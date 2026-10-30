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
If it installs an older version than the registry's latest, the Manager's cached
list is lagging the release: open the pack's card, pick **Latest** in its version
picker, then **Apply Changes**. Measured 2026-09-27: a fresh Manager gave 2.3.9
while 2.3.11 was live.

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
binary and **no ffprobe**; when no ffprobe binary is found the pack reads media
through PyAV (`av`, also in the requirements), which carries a normal episode,
but a system install of both is still what you want for the optional engines
that probe with the binary.

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

If you cannot install system-wide, the pack still runs on what
`pip install -r requirements.txt` brought: the `imageio-ffmpeg` wheel's ffmpeg
for encoding and PyAV for probing. Set `OTR_FFMPEG` or `OTR_FFPROBE` to a
static build if you want a particular one.

Check both answer. On macOS or Linux:

```bash
ffmpeg -version; ffprobe -version
```

On Windows PowerShell use `;` as above — `&&` is a parser error there.

If you rely on the wheel rather than a system package, `ffprobe -version`
reports "not found" and that is fine: the pack probes through PyAV then, and
finds the wheel's ffmpeg through the `imageio-ffmpeg` package, not through PATH.

If you keep ffmpeg somewhere unusual, set `OTR_FFMPEG` to the full path of the
binary. That environment variable wins over PATH. (There is an `ffmpeg` widget on
some nodes; it is ignored, deliberately — a saved workflow is not allowed to
choose which binary runs on your machine.)

**On Linux, also install one monospace font.** Burned captions and the credit
roll need a real TTF; a headless server image often has none at all.
`fonts-dejavu-core` is enough for Latin scripts. **Japanese and Mandarin need a
CJK font as well -- `apt install fonts-noto-cjk`. Without one the text still
renders, as a row of empty boxes, and nothing warns you: measured on a fresh
container that carried eight DejaVu fonts and zero CJK coverage.** Hindi needs a
Devanagari font the same way -- `apt install fonts-noto-core`. Seen again on a
RunPod pod 2026-10-21 (a Mandarin episode, every caption and credit boxed);
`scripts/otr_pod_provision.sh` now installs both, and since the same day the pack
ships Noto Sans SC and a Noto Devanagari face merged with Noto Sans's Latin
(`nodes/fonts`; plain Noto Sans Devanagari has no Latin letters) and falls back to them
when a machine has no face of its own, so a bare box draws the script instead of
boxes. The OS packages are still better for Japanese, whose kanji the SC face
draws in Chinese forms.

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
| 3.13 | Kokoro through `kokoro-onnx`, on CPU (on an NVIDIA card it moves to CUDA by itself once `onnxruntime-gpu` is installed) | Picked automatically. Every language works (Mandarin and Japanese after their libraries below). |
| 3.14 and up | Refused | No Kokoro build exists yet. The provisioner refuses rather than installing something that cannot speak. |

Japanese and Mandarin also need their readiness extras, checked only when
that language is selected: the libraries under the pack's own copy of
misaki's code, which reads those two languages on every Python. On 3.13
they install with the pack's requirements; on 3.10 - 3.12 the requirements
still declare them for 3.13 only, so install the one you need by hand with
ComfyUI's own Python (the same lines repair a 3.13 install that lost them):
`<ComfyUI Python> -m pip install fugashi jaconv unidic-lite` for Japanese and
`<ComfyUI Python> -m pip install jieba pypinyin cn2an ordered-set` for
Mandarin. Do not install `misaki[ja]`: it pulls `unidic`, an empty
dictionary downloader that fugashi prefers over unidic-lite (PBUG-20260918-06).
See [MULTILINGUAL.md](MULTILINGUAL.md).

## Weights: what you do not have to do

**Two things the pack deletes, so you are not surprised by either.**

- **Always:** on every ComfyUI start, and again after each publish, it sweeps
  `<output>/otr/episodes/_shared/tmp` and removes anything older than 24 hours.
  That folder is scratch space the render writes through; nothing you are meant
  to keep lives there, and the sweep touches no other folder, skips anything
  locked, logs every deletion and never fails a render. Set
  `OTR_TMP_SWEEP_MAX_AGE_S` to change the age, and do not park files under that
  path.
- **Space saver, on by default:** `asset_cleanup` on
  **OTR_LedgerScriptWriter** (Space saver on the form) ships `full`: once an
  episode is published, its own folder in `otr/episodes/` goes altogether and
  only the published video in `otr/obs/` remains. Pick `partial` to keep the
  text files, or `off` to keep everything. `otr/obs/` is never touched. See
  [RUN.md](RUN.md#saving-disk-space).

**Nothing by hand.** There is no setup script to run and no model to place
yourself. Almost everything arrives the first time you queue: the one exception
is the Kokoro voice set, which the pack fetches at STARTUP so the default voice
is ready before you ever press Run -- small files (~523 KB a voice). Where
Kokoro runs through `kokoro-onnx` (what the requirements install on Python
3.13), its ~326 MB ONNX model comes at startup too; with the torch `kokoro`
package (Python 3.10 - 3.12) Kokoro's own model (`hexgrad/Kokoro-82M`, about
the same size) arrives with the first voice line instead. `HF_HUB_OFFLINE=1`
or `OTR_SKIP_KOKORO_PREFETCH=1` skips the startup fetch; under `kokoro-onnx`
nothing fetches the model later, so the first voice line then stops and names
the command that fetches it. If the console pauses
on `[OldTimeRadio]` lines during a restart, that is what it is doing.

What `otr_canonical` selects, and so what its first run fetches -- the file
sizes the pack records, not a measured transfer (a cache you already have, a
partial download or a second copy all change what actually moves):

| What | File size | From |
|---|---|---|
| The writer — native Gemma 4 E2B (`comfy_native:gemma4-e2b-it-int8-convrot`) | ~4.8 GiB | `Comfy-Org/gemma-4`, ungated |
| Music — Stable Audio 3 checkpoint | 2.11 GiB | `Comfy-Org/stable-audio-3` |
| Music — its text encoder | 1.11 GiB | `Comfy-Org/stable-audio-3` |
| Voices — Kokoro | ~523 KB a voice, plus a ~326 MB model | voices at startup; the model at startup under `kokoro-onnx` (Python 3.13), else at the first voice line (`hexgrad/Kokoro-82M`) |

Other workflows select other writers, and the writer is the biggest single
file: the 16 GB and 24 GB graphs use native Gemma 4 12B (~11.2 GiB), the Mac
and AMD graphs use `Qwen/Qwen3.5-4B` (~8.7 GiB), and the cloud graphs download
no writer at all. [WRITERS.md](WRITERS.md) has every writer and what it costs.

The canonical's procedural video lanes draw their own frames
and consume no still image -- so the image model its dropdowns name stays
**dormant** and is not downloaded. Switch a video lane to one that consumes a
still and the image weights are fetched then, not before.

Almost every pick downloads itself; the few that still need a hand download
are listed in [MACHINES.md](MACHINES.md#where-do-the-manual-weights-come-from).

## Hugging Face login

Most of what this pack uses is ungated. The few picks that are not are marked
GATED in [MACHINES.md](MACHINES.md#will-this-engine-run-on-my-machine); for those
you accept the licence on the model's Hugging Face page while signed in, then log
in locally:

```bash
python -m pip install huggingface_hub
hf auth login
```

That stores a token in your Hugging Face config. **Do not paste a token into a
workflow widget** — no node here asks for one, and a token saved in a
workflow travels with it.

**Where it lands depends on what kind of file it is.** Four mechanisms, and
only the first two are ComfyUI registered folders that an
`extra_model_paths.yaml` entry can move:

- **ComfyUI-native weights** go in ComfyUI's own model categories, like any
  other ComfyUI model. The native Gemma writer is one file in `text_encoders`;
  Stable Audio 3 is a file in `checkpoints` plus its text encoder in
  `text_encoders`.
- **Other writers** (Qwen and the rest of the transformers rows) download as
  ordinary files to `LLM/<org>--<name>/` under your LLM folder -- relocatable
  with an `LLM:` entry. No symlinks, so Windows never asks for Developer Mode.
  A complete copy already in the Hugging Face cache from an earlier version
  stays there and keeps working; nothing is moved or fetched twice.
- **Kokoro** keeps its files in `TTS/KokoroTTS/` under ComfyUI's models folder
  (a fixed folder there, not a registered category). Its voices are fetched
  through the Hugging Face cache and copied in; on Python 3.13 its ONNX model
  downloads straight there, while on 3.10-3.12 the torch package loads its
  model from the Hugging Face cache.
- **Everything a library fetches for itself** goes to the Hugging Face cache,
  which follows `HF_HOME`, not `extra_model_paths.yaml`.
  A location you set yourself (`HF_HOME`, or `HF_HUB_CACHE`) always wins.
  Otherwise the pack points it during startup at a `huggingface` folder beside
  your ComfyUI models; on Windows, when that path would be too long for a
  machine without long paths enabled, it falls back to a shared models folder
  and then to your user cache. The startup log names the folder it chose.

If the Hugging Face cache's volume is short of room, set `HF_HOME` yourself
**before launching ComfyUI**. Setting it later does not move the cache, it adds
a second one, and you end up with two copies of everything.

## Node packs, only for some lanes

Most lanes need nothing beyond this pack. A few build their picture out of classes
that belong to someone else's pack, and those are listed per engine in
[MACHINES.md](MACHINES.md#which-workflow-do-i-open) -- with which of the shipped
workflows needs which. If you pick one of those lanes without its pack, the render
stops with an error naming the pack and its URL.

---

Next: [RUN.md](RUN.md) — making your first episode.
