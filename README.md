# ComfyUI-OldTimeRadio (SIGNAL LOST)

Turn **real news, public-domain stories, Shakespeare, or fully original LLM fiction** into a
finished **radio-drama video** — script, voices, music, and CRT-style video — fully automated
inside ComfyUI. Drop it in, queue one workflow, walk away, and a complete episode lands in
your output folder.

**Pipeline:** story source → LLM script → character voices + announcer + music themes (a
swappable 7-voice / 5-music engine roster; IndexTTS2 + Kokoro + Stable Audio 3 ship as the
defaults) → 48 kHz master mix → model-agnostic video (procedural CRT floor by default, or
HuMo / LTX / Wan / AnimateDiff / MiniMax H3 once you dial a heavier lane in) → final MP4.

100% local by default. No API keys required on NVIDIA and AMD. Optional hosted LLM
and all-cloud routes exist; they stay off unless you turn them on. One exception as
of this writing: the Apple Silicon **draft** profile has no local image engine wired
up yet, so its pictures come from Google's paid image API and need a key -- see the
Mac row in "Pick the graph" below.

> **Already installed it? Load the show:** **Workflow → Browse Templates →
> EXTENSIONS → comfyui-old-time-radio**. You will see two entries:
> **`otr_canonical`** (the one shipped graph: Kokoro voices for announcer and
> characters, Gemma-4-12B writer, Z-Image stills -- pick it, then **Queue Prompt**; or
> drag `workflows/otr_canonical.json` onto the canvas; on an 8 GB card load the matching
> saved-dropdown variant from "Pick the graph" instead),
> and `otr_story_only` (skip it for a first episode: it only writes the script, no
> voices, music or video; it exists for comparing writer models). The 25 `OTR_`
> nodes are the parts; the workflow is the thing you run.

> **Branch note:** active development lives on the **`v2.0-alpha`** branch (the Open Video
> Model Platform below). Check out `v2.0-alpha` to get the current pipeline — or skip the
> branch juggling entirely and install the packaged alpha from the
> [ComfyUI Registry](https://registry.comfy.org/publishers/fluxus/nodes/comfyui-old-time-radio).

---

## New to ComfyUI? Start here

You need four things: ComfyUI, a GPU, the models, and this node pack. ~20 minutes start to finish.

### 1. Install ComfyUI

Use the official [ComfyUI Desktop installer](https://www.comfy.org/download) (easiest), or a
manual/portable ComfyUI install. Launch it once to confirm it opens in your browser.

**Renting a GPU instead?** Follow [docs/RUNPOD_INSTALL.md](docs/RUNPOD_INSTALL.md) -- it covers RunPod and similar NVIDIA CUDA Linux hosts.

### 2. Install this node pack

**Use git.** Clone into your `ComfyUI/custom_nodes/` folder and check out the
active branch:

```
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
cd ComfyUI-OldTimeRadio
git checkout v2.0-alpha
pip install -r requirements.txt
```

Then restart ComfyUI so it loads the nodes.

> **Don't skip the `pip install`.** Manager and `comfy node registry-install` run it for you;
> a bare `git clone` does not. Each node imports in its own `try`/`except`, so a missing
> library doesn't break the pack — it silently *skips* the affected node and prints
> `[OldTimeRadio] Skipped '<name>': <reason>` in the console. If a node you expect is
> missing from the menu, that line is where to look. Use the same Python that runs
> ComfyUI (for portable installs: `python_embeded\python.exe -m pip install -r requirements.txt`).

> **Two system tools pip cannot install for you.** `ffmpeg` must already be on your
> PATH with the libx264 and aac encoders built in (Windows: the ComfyUI portable and
> Desktop builds bundle one; Mac: `brew install ffmpeg`; Linux: your package manager's
> ffmpeg). OTR mixes and muxes every episode through it, and a missing ffmpeg fails at
> render time with no earlier warning. **`pycairo` is Windows-only in
> `requirements.txt`** (`pycairo>=1.24; sys_platform == 'win32'`, because pycairo
> publishes zero Linux wheels), so pip on Linux or Mac skips it entirely and there
> are no headers to pre-install. Exactly ONE engine imports cairo --
> `viz_mxc_mandala` -- and it refuses loudly, naming the pip command, if you select
> it without cairo; every other visualizer lane is cairo-free. It is not in the
> canonical workflow, **but the two AMD profiles do select it** (`otr_amd8_rocm` and
> `otr_amd16_rocm` use `viz_mxc_mandala` for `music_visual`). So on Linux, and only
> if you run an AMD profile or pick that engine: install `libcairo2-dev`
> (Debian/Ubuntu) or your distro's equivalent plus `pkg-config`, then
> `pip install pycairo` yourself.

**The ComfyUI Registry route does not currently work, and Manager cannot install
this pack by any route.** No published version is `Active`
([registry page](https://registry.comfy.org/publishers/fluxus/nodes/comfyui-old-time-radio)),
so `latest_version` resolves to null: `@latest` has no target, and Manager
refuses the `nightly` git path on any network-exposed instance. Checked live
2026-09-04 -- 5 versions, 0 active: alpha.15/16/17 `Flagged`, alpha.13/14
`Banned`. If Manager reports "not a CNR node" or
"cannot resolve install target", that is this, not a fault on your machine. Use
the clone above.

### 2b. ComfyUI node packs — required by some video lanes

A few OTR video engines drive **other people's ComfyUI nodes**. Those are node
packs, not Python packages, so `pip install` cannot supply them and they are
deliberately absent from `requirements.txt` and `pyproject.toml`. Install them
into `ComfyUI/custom_nodes/` (ComfyUI-Manager, or `git clone`) and restart
ComfyUI.

| lane / profile | needs | why |
|---|---|---|
| `animatediff15_*` — including **`otr_nvidia_8gb_haunted`**, the 8 GB default | [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) at commit `92576512` (release 1.6.0, the checkout behind the published receipts) | provides the `ADE_*` classes the haunted lane samples through |
| `ltx25_*` (LTX 2.5 video, foley, mime) | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) at commit `6ea2651e`, **plus** the one-file patch in `patches/` (see `patches/README.md` for the exact `git apply` line), then its `requirements.txt` | the two GGUF loaders (`UnetLoaderGGUF`, `CLIPLoaderGGUF`); every other LTX 2.5 class is already in ComfyUI 0.34+. Measured on a clean Windows install 2026-09-01: without the pack the render refuses at the video stage and names both classes |
| `flux2_klein` (image; **the 8 GB / 12 GB / AMD default**) | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) at the same commit `6ea2651e` (the patch is harmless here) | its DiT is a 2.6 GB GGUF file loaded through `UnetLoaderGGUF`. Measured on a physical RTX 4060 8 GB under plain stock launch flags, 2026-09-02: about 21 seconds a still, no `--lowvram` needed |
| `wan22_*` / `wan_ti2v` (Wan 2.2 video) | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | its shipped DiT and umt5 text encoder are GGUF files (`UnetLoaderGGUF`, `CLIPLoaderGGUF`) |
| `ltx_8gb`, `ltx_video`, `ltx_audio_in` (the LTX 0.9.x lanes) | [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) at commit `3b9c5cde`, **plus** the one-file patch `patches/ComfyUI-LTXVideo-kornia-pad.patch` (Kornia 0.8.3 removed a symbol it imports) | the `LTXV*` node classes those lanes sample through; the engine's own preflight names this pack if it is missing |
| `humo_1.7B*`, `humo*`, `minimax_h3_*`, every `still_*` / `viz_*` lane | **nothing extra** | all their classes ship in stock ComfyUI 0.34+ (verified against a clean portable install 2026-09-01) |

`scripts/otr_provision.py` installs all three packs at their pinned commits for you
on Linux pods (GGUF `6ea2651e`, LTXVideo `3b9c5cde`, AnimateDiff-Evolved `92576512`,
release 1.6.0); on Windows, install the row you need by hand as above.

> **Python 3.13 and the Kokoro voice (ComfyUI Desktop and the portable build both
> ship Python 3.13).** The torch `kokoro` package cannot be pip-installed on 3.13
> (its newest releases declare `Requires-Python <3.13`), so since 2026-09-02 the
> same Kokoro voices run there through **kokoro-onnx** on the CPU: `requirements.txt`
> carries `kokoro>=0.7.16` for Python 3.12 and `kokoro-onnx>=0.6.1` for 3.13, one line
> installs per interpreter, and the engine picks whichever is present (same engine
> name, same 28 voices, about six times faster than realtime on a laptop CPU, no GPU
> contention with the video). The 326 MB ONNX model is fetched once at boot into
> `models/TTS/KokoroTTS/onnx/` when that backend will be used, never during a render.
> **Registry installs older than 2.0.0-alpha.17 do not carry the kokoro-onnx line
> yet:** on those, run `python -m pip install kokoro-onnx` with ComfyUI's own
> interpreter once, or open **OTR_CastLock** and set `voice_bank` -> `bark_legacy`,
> `char_voice_engine` -> `bark`, `announcer_voice_engine` -> `bark` (bark installs
> everywhere and downloads its own weights). A missing backend fails at the first
> voice line with the exact pip line to run. Python 3.14 has no Kokoro backend
> packaged yet; use bark there. On Python 3.12 nothing changes. Every voice engine and what it needs, generated from the code:
> `docs/MACHINE_MATRIX.md`, section "Voice engines".

**You do not have to memorise this.** If a pack is missing, the render stops
with a named error that now tells you which pack to install and where to get
it — it does not fail silently or half-render. But it stops at render time,
*after* the model weights have downloaded, which is why it is written here too.

**What the proven 8 GB episode path downloads on a clean machine: about 16
GB.** The invoked video lane itself is
only ~3.9 GB (SD1.5 1.99, `v3_sd15_mm` 1.56, the
domain adapter 0.10, kokoro voices 0.30). The rest arrives through the
Hugging Face cache the first time the pipeline runs — the writer
(`gemma-4-E2B-it`, 6.0 GB measured on a clean install 2026-09-01), `musicgen-small`
(~2.2 GB) and `Kokoro-82M` (~0.3 GB). The row also offers Klein as its image
selection, but AnimateDiff accepts no init still, so provisioning does not
download or gate the proven episode path on those extra 10,985,506,708 bytes.
The exact optional Klein recipe is in `docs/RUNPOD_INSTALL.md` for people who
want to qualify a still-consuming profile.

### 2b-ii. The GGUF writer lane — install 0.3.33, not the latest

Only needed if you select a `*-GGUF` writer row. It is the established lane
for running a large writer on a small card off NVIDIA: bitsandbytes NF4 is
CUDA-only, so the committed Mac, AMD and CPU experimental profiles
(`otr_mac_mps`, `otr_amd8_rocm`, `otr_amd16_rocm`, `cpu_floor`) use GGUF
through in-process llama.cpp. The new `--machine amd` front door instead uses
the smaller E2B Transformers writer with `quant_policy=none`; that route is a
draft candidate until physical AMD hardware publishes an episode. No Ollama,
no sidecar process, no extra port.

```bash
pip install llama-cpp-python==0.3.33 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Pin the version. Do not take the latest.** `0.3.35` dies with
`STATUS_ILLEGAL_INSTRUCTION` (`WinError -1073741795`) inside
`llama_init_from_model`, before a single token. It reproduces at
`n_gpu_layers=0`, so the fault is in the **CPU backend** and no GPU avoids it.
`0.3.33` loads and generates on the same machine, and the two builds were
confirmed byte-identical across two different machines by SHA-256. An unpinned
`pip install llama-cpp-python` resolves to the broken one today.

On Windows, CUDA wheels also need an importable CUDA 12 runtime:

```bash
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
```

These coexist safely with a CUDA 13 torch — measured on both Blackwell and Ada,
loading llama.cpp first and then running a real CUDA matmul through torch.

**Not on Windows?** The line above is the Windows CUDA recipe and the only one this
project has measured. The GGUF lane is the only local writer the Mac, AMD and CPU
profiles allow, so use upstream llama-cpp-python's own build flags for your backend
(these are upstream's documented commands, not something this pack has proven yet;
please report what worked):

| platform | install |
|---|---|
| Linux, NVIDIA | try the same `--extra-index-url .../whl/cu124` wheel line as above first (unmeasured here on Linux) |
| macOS, Apple Silicon | `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python==0.3.33 --no-cache-dir` |
| Linux, AMD ROCm | `CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python==0.3.33 --no-cache-dir` (upstream's current flag name) |
| CPU only | `pip install llama-cpp-python==0.3.33 --no-cache-dir` |

The `0.3.33` pin is a Windows measurement. If it will not build on your platform,
the newest release is the next thing to try.

> **Test it the way OTR does.** A bare `import llama_cpp` fails even on a
> *working* install, because OTR preloads the CUDA DLLs and extends the DLL
> search path first. Use the pack's own path instead:
> `from nodes._otr_gguf_backend import _import_llama_cpp; _import_llama_cpp()`

### 2c. Hugging Face token — best practice

**You do not need a token to run OTR.** The 8 GB haunted profile and everything
it pulls are ungated: verified by anonymous download of the real weight files,
with no credential sent. Same for the writer, the voices and the music.

**Set one anyway.** Anonymous downloads are rate-limited, and a multi-gigabyte
pull that gets throttled part-way is a failed install rather than a slow one.
`huggingface_hub` says so itself on every anonymous fetch: *"You are sending
unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher
rate limits and faster downloads."* Get one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) —
**read** scope is enough.

#### The safe ways, in order of preference

**1. Log in once (recommended).** This writes the token to a file only your
account can read, and nothing else ever has to know about it:

```bash
hf auth login
```

**2. Or create the token file yourself.** It is a plain text file containing
**the raw token and nothing else** — no quotes, no `HF_TOKEN=`, no JSON. Create
it at:

```
Windows   C:\Users\<you>\.cache\huggingface\token
macOS     ~/.cache/huggingface/token
Linux     ~/.cache/huggingface/token
```

and the entire contents are one line:

```
hf_your_token_here
```

That is the whole file (your real token is longer; the example is kept short on
purpose, because the registry's publish gate is a secret scanner and a
token-shaped placeholder trips it). OTR reads this location *and* the one your `HF_HOME`
points at, so it works whether you logged in before or after installing OTR.

**3. Or an environment variable**, if you prefer or you are running headless:
`HF_TOKEN=hf_xxxx`. On Windows, ComfyUI Desktop does not inherit user-scope
variables, so OTR also reads `HKCU\Environment` to cover that gap.

#### What NOT to do, and why it matters

**Never paste a token into a node widget** — not into OTR's, not into any other
pack's, no matter what a Note node beside it says. A widget value is written
into `widgets_values` in the workflow JSON, which means it travels with:

* every workflow file you save, share, or attach to a bug report
* the prompt in ComfyUI's queue and `/history`
* **the PNG metadata of every image you generate** (that embedded workflow is
  exactly what lets you drag a PNG back onto the canvas to restore it)
* any traceback or support bundle that prints the node's inputs

"Just don't save the workflow" is an instruction, not a control, and most of
those paths are not a save. **OTR has no token widget anywhere and never will.**

**Do not put it in a `.env` file at the ComfyUI root.** Vanilla ComfyUI does not
read one — there is no dotenv loader in `main.py` and `python-dotenv` is not
even a ComfyUI dependency. It will be silently ignored.

**Be careful with ComfyUI Desktop's environment-variable editor.** It can set
`HF_TOKEN`, but Desktop itself warns those values are stored **unencrypted**.
Prefer the login file above.

### 3. Hugging Face token — only if you pick a gated model

**Most people need no token at all.** The shipped canonical workflow pins
`google/gemma-4-12b-it`, which is Apache-2.0 and ungated, so a normal first run downloads
without any account. You only need a token if you switch the writer dropdown to one of the
gated rows below.

**Gated — you must accept the terms on the model page first, then supply a token:**

| Model / weights | What it is | Accept terms at |
|---|---|---|
| `google/gemma-2-2b-it` | an optional writer LLM | https://huggingface.co/google/gemma-2-2b-it |
| `Lightricks/LTX-2.5` | the LTX 2.5 **video** weights | https://huggingface.co/Lightricks/LTX-2.5 |

**Ungated — nothing required:** `google/gemma-4-12b-it`,
`mistralai/Mistral-Nemo-Instruct-2407`, `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`,
`Lightricks/LTX-Video` (the older 0.9.x line), and the `Comfy-Org` Wan repackages.

> **Note:** Gemma **4** is Apache-2.0 and ungated, unlike Gemma **2** and Gemma **3**. Only
> `gemma-2-2b-it` still requires a terms click (verified against the Hugging Face API, which
> reports `"gated": "manual"` for that repo and `"gated": false` for every other curated row).

> **The LTX 2.5 gate bites late, and that is why it is listed here** (added 2026-08-29 after
> it stopped a clean-machine install). Nothing in a default first run touches it — the
> canonical workflow ships on the procedural video floor — so you meet it only when you
> select an `ltx25_*` row in `OTR_VideoDirector`. Its repo reports `"gated": "auto"`:
> approval is automatic, but the terms click and a token are both still required, and an
> unauthenticated fetch returns **HTTP 401** rather than anything that reads like a licence
> problem. If an `ltx25_*` lane fails to download its weights, this is why.

Accepting the terms is a **manual, one-time click** while signed in to Hugging Face — a token
alone is not enough, and the download fails until you have done both. Get a token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a **read** token is
all you need).

**Where to put it.** OTR reads `HF_TOKEN` from the process environment first, then — on
Windows only — from `HKCU\Environment`. That registry fallback exists because **ComfyUI
Desktop does not inherit user-scope environment variables**, so a token you set in the
System Properties dialog is invisible to it until you either bake it in or reboot.

Windows (PowerShell) — sets it user-wide, where ComfyUI Desktop will find it:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token_here", "User")
```

macOS / Linux — add to `~/.bashrc` or `~/.zshrc`:

```bash
export HF_TOKEN=hf_your_token_here
```

Restart ComfyUI afterwards. If a gated model still fails, the console error names the exact
repo and the two steps it is missing.

### 4. Install the models

**If you are on an 8 GB card, load the 8 GB variant named in "Pick the graph" below;
its downloads are listed there.**

The shipped 16 GB canonical workflow (`otr_canonical`) is lighter than you'd expect: its
video-role dropdowns default to the **procedural still/CRT floor** (`still_flat`, no GPU
video checkpoint required at all), and its image role defaults to **Z-Image-Turbo**
(Apache-2.0, no license friction). So a first run needs only: the local writer LLM
(`gemma-4-12b-it` as shipped, or your own choice), Z-Image-Turbo, and the default voice
and music weights. Two of those fetch themselves on first use (Kokoro, Stable Audio 3).
**One does not: IndexTTS2**, the character voice the canonical graph ships with. It runs
in its own separate Python (3.10 + torch 2.8), so before your first Queue Prompt on
`otr_canonical` either run its one-time installer from a terminal in the pack folder
(`powershell -ExecutionPolicy Bypass -File scripts\_otr_indextts2_install.ps1`, which
builds that environment and downloads its own multi-gigabyte model), or open the
**OTR_CastLock** node and switch `char_voice_engine` to `bark` (or `kokoro` on Python
3.12). Skip both and the render runs the writer, then stops with
`IndexTTS2 Path B not installed`. The heavier local video checkpoints — HuMo, LTX,
Wan, AnimateDiff, MiniMax H3 — are **optional upgrades** you dial in later via the
`OTR_VideoDirector` dropdowns; see [Which video models fit your card](#which-video-models-fit-your-card)
before downloading any of them. If a model is missing, the engine fails **loudly** and stops —
it never silently substitutes another model or quietly produces garbage. There is no automatic
fallback: the procedural CRT path is a route you **select** (and the canonical workflow ships
with it selected), not a net that catches a failed engine. Watch the console on the first run;
it names any missing weight and where it expects it.

### 5. Run it

**Pick the graph that matches your card.** Saved GUI graphs and `--machine`
recipes use the same canonical workflow, but provisioning installs assets only;
it does not silently rewrite the graph currently open in ComfyUI.

| you have | load this | what it renders |
|---|---|---|
| 8 GB card, ready for real video | `workflows/variants/otr_nvidia_8gb_haunted.json` (drag it onto the canvas) | the proven 8 GB matrix row: AnimateDiff haunted video and Kokoro voices, about 16 GB of downloads. Kokoro runs on Python 3.12 (torch) and 3.13 (kokoro-onnx, CPU) alike; only Python 3.14 has no Kokoro backend yet -- there, open **OTR_CastLock** after loading and set `voice_bank` -> `bark_legacy`, `char_voice_engine` -> `bark`, `announcer_voice_engine` -> `bark` before you queue. Needs the AnimateDiff-Evolved pack (section 2b) |
| 8 GB card, Klein stills and LTX 2.5 video | not a shipped graph yet -- see below | measured 2026-09-02 on a physical RTX 4060 under plain stock launch flags: Klein 4B stills at about 21 s each, LTX 2.5 clips at about 14 min each (works, slow). Needs ComfyUI-GGUF (section 2b). A shipped 8 GB profile for this pair is the next item on the plan |
| 16 GB or more, GUI authoring baseline | **the same menu -> `otr_canonical`** (or drag `workflows/otr_canonical.json` onto the canvas) | Gemma-4-12B writer, `still_flat` video for every role, Z-Image-Turbo stills, IndexTTS2 + Kokoro voices, Stable Audio 3 music. Read the IndexTTS2 note in section 4 first. This is **not** the Gemma/Wan/Kokoro/musicgen `--machine 16gb` tuple; use the headless command below to apply that row atomically |
| AMD GPU on Linux (draft, unproven on real hardware) | `workflows/variants/otr_amd8_rocm.json` or `otr_amd16_rocm.json` (drag onto the canvas) | images only: Klein 4B stills with still-motion and visualizer video, Kokoro voices (torch on 3.12, kokoro-onnx on 3.13) or bark via the CastLock dropdowns; needs a ROCm torch and ComfyUI-GGUF. Fully local |
| Apple Silicon Mac (draft, unproven on real hardware) | `workflows/variants/otr_mac_mps.json` (drag onto the canvas) | images only, and as shipped the picture roles use `google_image`, a paid Google API that needs `OTR_GOOGLE_API_KEY` -- the local Klein engine is ruled for Mac but not yet wired for Apple's GPU backend. Switch the three image dropdowns in **OTR_VideoDirector** to a still or visualizer lane if you want a fully local run |

1. Load the graph from the table. (The console prints the Browse Templates path on every
   start, right under the `[OldTimeRadio]` load banner.)
2. For the exact matrix row without hand-editing dropdowns, leave ComfyUI
   running and execute this command, replacing only the exact machine key:

   ```powershell
   # Desktop/source example. Portable users point this at python_embeded\python.exe.
   $ComfyPython = 'C:\path\to\ComfyUI\.venv\Scripts\python.exe'
   & $ComfyPython scripts/otr_canonical_api_run.py --comfyui-url http://127.0.0.1:8188 --machine 8gb --act-count 1 --source-bank original --visual-style sci_fi_radio --timeout 0
   ```

   This loads `workflows/otr_canonical.json` and applies every matrix value
   before Queue. Change the URL only if your running ComfyUI uses another port.
   Use the interpreter that launches ComfyUI, never an arbitrary system
   Python; a standard portable install uses
   `ComfyUI_windows_portable\python_embeded\python.exe`.
   Machine rows select Kokoro, which runs on Python 3.12 (torch) and 3.13
   (kokoro-onnx, CPU). Only Python 3.14 has no Kokoro backend packaged yet: there,
   replace `--machine 8gb` with `--profile otr_4060_floor` for the Bark route
   (the selectors cannot be combined).
3. For a saved GUI graph, hit **Queue Prompt**.
4. Walk away. Script, voices, music, mastering, and video all run automatically. The shipped
   graph rolls a random story bank each run and renders through the procedural still/CRT floor
   — the fast, guaranteed-to-complete path. Swap dropdowns once you're ready for a specific bank
   or a GPU video engine.
5. Find the finished episode in **`output/otr/obs/`**.

---

## Requirements

- **GPU:** an NVIDIA card is recommended for the local video engines. The shipped canonical
  workflow renders through the procedural still/CRT floor by default (no GPU video model
  required); heavier local/cloud routing is opt-in via the `OTR_VideoDirector` dropdowns or
  explicit profile overrides. Episode length is set by act count, not a word target.
- **OS:** Windows or Linux for the proven NVIDIA paths (RTX 4060 8 GB and RTX 5080 16 GB
  have both published episodes); AMD ROCm needs Linux. macOS (Apple Silicon) ships as an
  unverified draft profile (`otr_mac_mps`) whose pictures come from a paid Google API
  today -- read the Mac row in "Pick the graph" before you start.
- **Python:** 3.12 or 3.13. ComfyUI Desktop and the portable build ship 3.13, where the
  Kokoro voice runs through kokoro-onnx on the CPU (section 2b); 3.14 has no Kokoro
  backend yet (bark replaces it with three dropdown changes).
- **Other setups:** per-platform workflow variants + recipes ship in-repo (16 GB NVIDIA
  canonical, cloud-lane variant, Mac, AMD). The Mac/AMD variants are drafts — not yet
  verified on real hardware.
- **RAM:** 32 GB of system memory is the comfortable floor for the video lanes; the 8 GB
  card streams model weights from host RAM. The measured host-RAM peaks so far are on the
  5080 (the H3 clamped run at 27.56 GiB, the HuMo 14B lane at 27.53 GiB); LTX 2.5 on the
  4060 has not had its host-RAM peak measured yet.
- **Disk:** the model set is large (tens of GB). Episodes are a few dozen MB each.

---

<!-- BEGIN GENERATED: machine-matrix -->

## What works on what machine

| your machine | writer | video | voice | music | image | status |
|---|---|---|---|---|---|---|
| **8 GB NVIDIA (RTX 4060, 3070, 2080)** | gemma-4-E2B | animatediff15_v3_haunted_video | kokoro | musicgen | flux2_klein | **EPISODE PATH PROVEN** -- writer/video/voice/music on RTX 4060; image lane (Klein) proven 2026-09-02 on a Python 3.13 clean room |
| **16 GB or more NVIDIA (RTX 5080, 3090, 4090, A4500)** | gemma-4-12b | wan22_high_video | kokoro | musicgen | z_image_turbo | **COMPONENTS PROVEN** -- Wan on named Ampere/Blackwell hardware; exact row tuple and unlisted cards unproven |
| **10-15 GB NVIDIA (RTX 4070, 3080, 3080 Ti 12 GB)** | gemma-4-E2B | animatediff15_v3_haunted_video | kokoro | musicgen | flux2_klein | `draft`, unproven |
| **AMD / ROCm (Linux only)** | gemma-4-E2B | still_motion | kokoro | musicgen | flux2_klein | `draft`, unproven |

**Use the machine key, not an experimental profile name.** Run these with the exact Python executable that launches ComfyUI (shown as `<ComfyUI Python>`). Preview the install plan first, then run the same command without `--list` to install it.

* **8 GB NVIDIA (RTX 4060, 3070, 2080)** -> `<ComfyUI Python> scripts/otr_provision.py --machine 8gb --list`
* **16 GB or more NVIDIA (RTX 5080, 3090, 4090, A4500)** -> `<ComfyUI Python> scripts/otr_provision.py --machine 16gb --list`
* **10-15 GB NVIDIA (RTX 4070, 3080, 3080 Ti 12 GB)** -> `<ComfyUI Python> scripts/otr_provision.py --machine 12gb --list`
* **AMD / ROCm (Linux only)** -> `<ComfyUI Python> scripts/otr_provision.py --machine amd --list`

Provisioning installs and verifies artifacts; it does not rewrite the saved graph. To apply one row atomically to the real canonical workflow on a normal port-8188 ComfyUI server, run `<ComfyUI Python> scripts/otr_canonical_api_run.py --comfyui-url http://127.0.0.1:8188 --machine 8gb --act-count 1 --source-bank original --visual-style sci_fi_radio --timeout 0`, replacing only the exact machine key. To use an explicit profile instead, replace `--machine 8gb` with `--profile <exact-profile-id>`; the two selectors are intentionally exclusive. Every machine row selects the Kokoro voice. On the Python 3.13 that ComfyUI Desktop and the portable build ship it runs through kokoro-onnx on the CPU (the same voices, about six times faster than realtime); on Python 3.12 through the torch kokoro package. Python 3.14 has no kokoro backend packaged yet; there, run `--profile otr_4060_floor` for the bark route or switch the OTR_CastLock voice dropdowns to bark.

Apple Silicon is still the unproven experimental `otr_mac_mps` profile; CPU-only is `cpu_floor`. Neither is promoted to a machine key or PROVEN until a named physical system publishes an episode.

<!-- END GENERATED: machine-matrix -->

## Which video models fit your card

**For the full picture -- writer, video, voice, music and image per machine, with what is PROVEN versus merely shipping -- see [docs/MACHINE_MATRIX.md](docs/MACHINE_MATRIX.md).** Its machine rows and proof receipts come from `config/machine_classes.json`; experimental profile inventory comes from `config/profiles/`. The section below is the video-specific detail behind it.


**This table is the profile.** Pick your card, read the column, choose that name
in the `OTR_VideoDirector` dropdown. Names below are exactly the dropdown
entries.

Every figure is MEASURED, not estimated, and each says where it came from. A
blank verdict means nobody has measured it -- that is recorded as unknown rather
than guessed, because a guessed VRAM number is the one thing a user cannot
recover from.

**Read each 8 GB claim at its stated proof level.** A physical RTX 4060 8 GB
has published 7 documented full OTR episodes (six on AnimateDiff video, one with Klein
stills and still-motion video on a fresh Python 3.13 portable), so the invoked
writer/video/voice/music episode path AND the row's image lane are **PROVEN**. Separately, a
raw MiniMax H3 FL2VA ComfyUI recipe produced
three valid 864x480x90 clips with native audio on the same physical card, with
7.147/6.788/6.788 GiB peaks. That is **LAB-PROVEN true diffusion**, but it is
below OTR's canonical 124-model-frame floor and did not use OTR's silent H3
adapter, so the OTR H3 lanes remain candidates pending their own canonical run.

A 5080 `--reserve-vram 12` H3 MIME render is useful comparative evidence, but
it is a **CLAMPED SIMULATION**, not physical-8-GB proof:

| | measured |
|---|---|
| peak VRAM | **7.28 GiB** (baseline 2.52) |
| peak host RAM | **27.56 GiB** (baseline 17.35) |
| boot lane | `sage-free, no-pinned, reserve-12gb` |
| duration | 178.9 s |
| verdict | `PASS (cold)`, output visually approved |

**The host RAM number is the one people miss.** 27.56 GiB peak against an
8 GB laptop's typical 32 GiB of system memory is a tighter margin than the VRAM
is. A machine with 16 GiB of RAM will struggle with this pipeline regardless of
its GPU, and no VRAM table will warn you.

**Two limits, stated so nobody reads more into this than it carries.** The lab's
own rule is that a recipe is only `PASS` when a **second consecutive warm run**
records it, and both H3 MIME receipts are `run_number 1`, cold -- so this is a
cold pass, not a completed gate. **The physical 8 GB laptop has since rendered, and this section used to say
otherwise.** It reported the card inventoried (**8,188 MiB VRAM, 31.7 GiB host
RAM**) but having "rendered nothing" -- true when written, false within days,
and left standing while an 8 GB RTX 4060 published **five of five source banks**
on `animatediff15_v3_haunted_video` (`docs/4060_DRILL_LOG.md`, steps 7-19). That
is why the row above reads **EPISODE PATH PROVEN** rather than `?`, and why the per-machine table
is now GENERATED -- see
[docs/MACHINE_MATRIX.md](docs/MACHINE_MATRIX.md). A hand-kept compatibility
claim goes stale in the direction that costs a user the most: telling them their
card cannot do a thing it has already done.

`ltx25_high_video` peaks at **14.48 GiB** on the 16 GB card, but a peak is what
the allocator grabbed, not a hard floor: on 2026-09-02 a physical RTX 4060 8 GB
clean-room install rendered real LTX 2.5 clips end to end under plain stock launch
flags, about 14 minutes a clip (the text encoder pinned to CPU by the engine, the
video model half streamed from host RAM). It works on 8 GB; it is slow, and no
shipped 8 GB profile wires it up yet. Receipt: `docs/ship-audit-2026-09-01/
4060_CLEANROOM.md`, Leg C5. The same run fixed why every 8 GB still used to take
about 42 minutes: the writer LLM was still on the card when the image stage began.
Update to commit `da2b7a36` or later if your stills are that slow.

### Local video models

| Dropdown name | Measured VRAM | 8 GB | 12 GB | 16 GB |
|---|---|:--:|:--:|:--:|
| `ltx098_low_video (16:9)` | 6.8 GiB @ 512x288x161 | maybe | yes | yes |
| `h3_low_audio_in (16:9)` | 6.9-7.2 GiB @ 864x480x90 in the raw recipe lab | candidate: raw 90-frame **LAB-PROVEN**, not an OTR episode | yes | yes |
| `h3_low_video (16:9)` | **7.28 GiB under an 8 GB clamp** | unknown: clamp only, not physical 8 GB proof | yes | yes |
| `ltx23_low_audio_in (16:9)` | 7.36 GiB @ 1024x576x193 | maybe | yes | yes |
| `animatediff15_v3_haunted_video (16:9)` | ~3.9 GB of weights, hold-2 cadence | **PROVEN** | yes | **PROVEN** |
| `wan22_high_video (16:9)` | 12.1 GiB @ 832x480x193 | no | maybe | yes |
| `humo17_high_audio_in_portrait (portrait)` | 12.84 GiB @ 480x832x129 | no | maybe | yes |
| `humo14_high_audio_in_wide (16:9)` | 13.06 GiB @ 832x480x97 | no | no | yes |
| `humo14_high_audio_in_portrait (portrait)` | 13.22 GiB @ 480x832x97 | no | no | yes |
| `ltx23_high_video (16:9)` | 13.3 GiB @ 1024x576x169 | no | no | yes |
| `wan22_high_fast (16:9)` | 12.8 GiB measured 2026-08-22 | no | maybe | yes |
| `ltx25_high_video (16:9)` | **14.48 GiB peak on a 16 GB card** (what the allocator grabbed, not a floor) | works, slow: ~14 min a clip on a physical RTX 4060 under stock flags, 2026-09-02; no shipped 8 GB profile yet | yes | **PROVEN on the 5080** |
| `humo17_high_audio_in_wide (16:9)` | not measured at this aspect | ? | ? | yes |
| `mesh_stage (16:9)` | not measured | ? | ? | yes |

`animatediff15_v3_haunted_video` was the ONE surviving AnimateDiff lane after
the 2026-08-23 directive ("delete any animatediff that are not haunted"). Its
former peers — `animatediff15_video`, the hold-3/hold-5 cadence variants,
`animatediff15_v2_video`, and `animatediff15_v3_video` — are retired and
tombstoned in the engine registry; they no longer appear in any dropdown.
**A second AnimateDiff lane joined it on 2026-09-02:**
`animatediff15_v3_stillin_lab_video`, the still-in laboratory peer. Both are
selectable today; the retired peers above are still gone.

**Licensing note on this table:** most engines here are open weights, but two
are not. `h3_low_video` / `h3_low_audio_in` (MiniMax H3) run under a personal,
non-transferable authorization the maintainer obtained directly from MiniMax —
it does not transfer to your install; treat H3 as off unless you have your own
agreement with MiniMax. `animatediff15_v3_haunted_video` loads a
motion module with **no published license grant** (`commercial_clean = False`
in the adapter) — fine for personal/hobby use, not cleared for commercial
redistribution. See [License & Credits](#license--credits) for the full list.

### Procedural and still lanes -- these run anywhere

`still_flat (16:9)`, `still_motion (16:9)`, `still_pan (16:9)`,
`still_word (16:9)`, `viz_camera (16:9)`, `viz_green (16:9)`,
`viz_mxc_cpu (16:9)`, `viz_mxc_mandala (16:9)`.

The four `viz_*` lanes are pure numpy/PIL/ffmpeg with no model at all and no GPU
requirement. The `still_*` lanes cost whatever your chosen IMAGE model costs,
since the video side is a pan or a hold over a still. `still_flat` is the
canonical workflow's shipped default for every video role.

### Cloud lanes -- no local VRAM, but they are paid services

`cloud_kling_avatar (16:9)`, `cloud_seedance_2 (16:9)`,
`cloud_vidu_q2_pro_fast_720p (16:9)`, `cloud_wan_i2v (16:9)`,
`cloud_wan_i2v_audio (16:9)`, `google_omni_video (16:9)`,
`google_veo_video (16:9)`, `word_razzle (16:9)` (a Comfy Cloud partner lane
despite the name — it renders provider-side via Pixverse). All are OFF by
default; this project is offline-first and nothing here is required to make
an episode.

### Two lanes need their own boot

`h3_low_video` and `h3_low_audio_in` require a sage-free boot with pinned memory
disabled and VRAM reserved. They will fail preflight on a standard boot -- and
that is the guard working, not the lane being broken. Both are also the slowest
local lanes by a wide margin.

---

## How it works

**Audio is the source of truth.** The writer produces a script, the voice/music engines render
it, and everything is assembled into a single **frozen 48 kHz master mix**. That master defines
the episode timeline and the per-beat clip budget. Video is rendered to fit the audio and is
**muxed in last, byte-identical, in the archival copy** written to `otr/episodes/` — the audio
there is never re-encoded or altered by the video stage. The published copy in `otr/obs/` (the
one you actually watch) re-encodes that same audio to AAC 320 kbps for player compatibility;
the PCM content is unchanged, only the container codec differs.

```
Story bank → LedgerScriptWriter (LLM) → FreezeCascade → CastLock
     → character voices + announcer + music themes (per-role engine roster)
     → SceneSequencer → AudioEnhance → EpisodeAssembler  ==> 48 kHz MASTER (frozen)
     → VideoDirector / ShotLock (per-role engine + per-beat prompts)
     → VideoRenderBatch (render each beat through its engine)
     → SilentComposite → CaptionBurn → CreditsRoll → MasterAudioMux  ==> final MP4 in otr/obs
```

Only two of the five story banks (`scifi_news_pro`, `media_archive`) actually pull from a news
or archive feed — `public_domain` and `shakespeare` adapt a fixed source text, and `original` is
entropy-seeded with no external input at all. "Story bank" above covers all five; see
[Story sources](#story-sources-source-banks) for what each one actually consumes.

---

## Story sources (source banks)

The writer's `source_bank` dropdown selects where each episode's story comes from. The
shipped canonical workflow rolls randomly across every eligible bank each run
(`scifi_news_pro` is only the code-level fallback for a freshly-dropped, unconfigured node —
pin the dropdown to one bank if you want a fixed lane). Every lane is an INDEPENDENT bank (its
own story pack + story_rules) with no dependency on any other lane.

| Bank | What it does |
|------|--------------|
| `scifi_news_pro` | sci-fi radio drawn from a live science feed; an LLM-first multipass writer using the configured model slots |
| `media_archive` | media RSS / archive items → restoration-adventure episodes |
| `public_domain` | faithful radio adaptation of a public-domain source |
| `shakespeare` | Folger scene adaptation. **The Folger Digital Texts are CC BY-NC 3.0 (noncommercial)** — episodes from this bank inherit that restriction on the source text. |
| `original` | no-source original fiction seeded from an entropy spark draw |

A typed `custom_premise` rides along as an operator hint on the original lanes and as a
source override on the article lanes. Every lane is fail-closed: a bad source, context
overflow, or contract violation stops loudly instead of shipping a degraded story, and
the LLM writes all story text — Python validates, it never rewrites prose.

**Add your own source bank:** every bank is independent, and you can author a sixth
peer to the shipped five — your own feed, archive, or source strategy — running through
the same trusted writer. The requirements contract (above all: the episode ledger must
be COMPLETE for every downstream consumer) lives in
[`docs/EXTENDING_OTR.md`](docs/EXTENDING_OTR.md); read it before authoring.

---

## v2.0-alpha — the Open Video Model Platform

The video layer is **model-agnostic**: a registry of pluggable engine adapters, chosen
**per role**, with no single model treated as "primary." You pick the engine for each kind of
beat, and that pick is honoured exactly: a missing or OOMing engine **fails loudly** and stops
the render rather than swapping in a substitute you did not choose. The frozen audio is never
touched either way. If you want the zero-GPU procedural CRT path, select it — it is the
canonical workflow's shipped default, not a rescue lane.

**Roles** (each selectable in `OTR_VideoDirector`):

| Role | What it is | Canonical default |
|------|------------|-------------------------------|
| `announcer_visual` | the announcer bookends | `still_flat` (image-model still, no video model) |
| `music_visual` | opening/closing theme bookends | `still_flat` (image-model still, no video model) |
| `character_video` | character dialogue beats | `still_flat` (image-model still, no video model) |

(The former `sfx` speaker role and `scene_broll` / `background_abstract` video roles were
removed in the 2026-07-01 cleanbreak — old ledgers using them fail loud by design.)

**Engines available:** HuMo (audio-driven face, 14B + 1.7B tiers), LTX (text/image→video and
audio-in), Wan (TI2V / I2V), AnimateDiff (Ghost Signal, three shipped cadence peers), MiniMax
H3 (personal license only, see the licensing note above), `mesh_stage`, and the cheap CPU
floors (CRT **visualizer**, Ken-Burns, flat still). Audio-driven engines are offered only where
audio exists; engines load one at a time with explicit VRAM reclaim between stages, and
renders are request-hash deterministic. The old VRAM tier system is gone — per-platform
workflow variants are the sizing mechanism now.

### The video model reference — read these two before adding or changing an engine

| doc | what it holds | kept true by |
|---|---|---|
| [`docs/ENGINE_MATRIX.md`](docs/ENGINE_MATRIX.md) | **every per-model number** — clip window, frame ladder, continuity, join mode, segment counts, effective canvas | **generated + drift-gated.** `python tools/engine_matrix.py --check` is a suite test, so it cannot disagree with the adapters |
| [`docs/2026-08-02-FINAL-all-engine-maths-and-stills.md`](docs/2026-08-02-FINAL-all-engine-maths-and-stills.md) | the things a generator cannot derive — still logic and the local/cloud re-mint split, the fix list with per-item status, the open decisions, the padding rule | by hand, with a dated verification stamp |

**The rule between them: a hand-maintained doc must never re-type a number the
generated one already owns.** That is not style. On 2026-08-06 the hand-written
tables were found asserting 3 and 10 segments for HuMo where the live registry
said 5 — a ceiling that had moved four days earlier — while the drift-gated
matrix had been right the whole time. Cite the generated matrix; do not copy it.

Multi-clip coverage itself (how a long beat is partitioned into chained or
jump-cut segments) is settled in `nodes/_otr_video_engines/coverage_plan.py`, and
the arithmetic totals exactly on every engine, local and cloud.

**Padding rule (operator, 2026-08-06):** no mirror and no ping-pong anywhere —
every second of audio gets ORIGINAL video, and a short render fails loud rather
than filling. The single sanctioned exception is the closing-theme backdrop that
holds the last drama clip under the closing theme; the credits roll itself
freezes a frame and never loops.

### Preflight guides -- the checklists that gate a change

Each subsystem has a preflight document: a gate-by-gate checklist run whenever
that subsystem is added to or materially changed, each backed by an enforcement
suite so the doc cannot drift from the code.

| subsystem | guide | enforced by |
|---|---|---|
| video models | [`docs/VIDEO_LANE_PREFLIGHT.md`](docs/VIDEO_LANE_PREFLIGHT.md) | `tests/test_lane_preflight_matrix.py` |
| image models | [`docs/IMAGE_GEN_PREFLIGHT.md`](docs/IMAGE_GEN_PREFLIGHT.md) | `tests/test_image_gen_preflight_matrix.py` |
| TTS voices | [`docs/TTS_VOICE_PREFLIGHT.md`](docs/TTS_VOICE_PREFLIGHT.md) | `tests/test_tts_voice_preflight_matrix.py` |
| source banks / story | [`docs/SOURCE_BANK_PREFLIGHT.md`](docs/SOURCE_BANK_PREFLIGHT.md) | the roster/bijection suites it names |
| LLMs (writer models) | [`docs/LLM_PREFLIGHT_GUIDE.md`](docs/LLM_PREFLIGHT_GUIDE.md) | seven gates before a row joins the dropdown |
| driving a soak leg | [`docs/SOAK_LEG_GUIDE.md`](docs/SOAK_LEG_GUIDE.md) | not a preflight gate -- the widget map + sanctioned-lever rules for varying engines/models/upscalers across legs |
| weights on disk | [`docs/MODEL_INVENTORY.md`](docs/MODEL_INVENTORY.md) | not a preflight gate -- the full model list under `C:\ComfyUI-Models`, what references each file, and the disk-reclaim analysis |

The rule that binds them (operator, 2026-08-21): **every video lane obeys the
per-role image-model dropdowns** -- the picture a `still_*` or motion lane holds
is minted by whichever image engine the operator selected for that role. The
only exemption is the `viz_*` visualizer family, which is procedural and mints
no still -- and each of those lanes declares that exemption out loud
(`accepts_still = False`); staying silent is a test failure. Adding your own
engine? Start at `docs/EXTENDING_OTR.md`, then run the matching preflight.

### Image models

The picture behind every still/motion lane and video role comes from a per-role image-model
dropdown (`announcer_image_model`, `music_image_model`, `character_image_model`), independent
of which video engine you picked. **`z_image_turbo` (Apache-2.0) is the shipped canonical
default for all three roles.** Same open-set "registry IS the menu" story as video: drop in an
adapter, no other edits, and it's selectable everywhere.

| Engine | License | Notes |
|---|---|---|
| `z_image_turbo` | Apache-2.0 | shipped default, all three roles |
| `lumina_image` | Apache-2.0 | ~7-12 GB measured |
| `flux2_klein` | Apache-2.0 | FLUX.2 Klein 4B |
| `flux_gen1` (Flux.1-dev) | **BFL non-commercial** | coded as the in-stack "gen-1" default; the canonical workflow does not select it |
| `ideogram4_local` | **non-commercial weights, opt-in** | heaviest local image engine, 16 GB-class only; typography-first for the `still_word` title card |

Six more engines run through the Comfy Cloud partner bridge (`cloud_flux_pro`,
`cloud_nano_banana_2`, `cloud_seedream_2`, `cloud_krea_2_turbo`,
`cloud_luma_photon_flash`, `ideo`) plus a direct Google Gemini/Nano-Banana adapter
(`google_image`, BYO API key). All are OFF by default — same "the dropdown pick is the
enable" pattern as the cloud video lanes, no local VRAM, budget-estimated per call, fail
loud without credentials. Full contract: [`docs/IMAGE_GEN_PREFLIGHT.md`](docs/IMAGE_GEN_PREFLIGHT.md).

### Headless canonical path

Agents and API tests use exactly one workflow file: `workflows/otr_canonical.json`.
By default, the headless wrapper applies no profile and leaves the saved dropdowns
alone. Explicit profiles are still available for deliberate route testing, such as
`otr_cloud_lanes` for the hosted Partner API path.

Headless/API smoke runs must use the canonical workflow wrapper:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Acts 3
```

For a no-queue validation of the exact API prompt shape:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_canonical_api_run.py --offline-schemas --dry-run --act-count 3
```

Cloud route example:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile otr_cloud_lanes -Acts 3
```

`--words` no longer exists anywhere on this path. It went with the `target_words`
widget: act count is the only episode-shape knob, and length is an observation.

That path always loads `workflows/otr_canonical.json`; engine dropdowns move only
through explicit profiles, and ad-hoc `--set` patches are limited to creative/story
widgets.

### Optional: hosted LLM via OpenRouter (off by default)

The writer runs locally out of the box — the shipped canonical workflow pins Gemma-4-12B for
both the creative and technical slots (Mistral-Nemo is only the code-level fallback for a
freshly-dropped, unconfigured node). You can optionally route either slot to a hosted frontier
model via OpenRouter — it activates as soon as `OPENROUTER_API_KEY` is set (no separate opt-in
flag anymore), is cost-guarded, and fails closed. Full walkthrough:
[`docs/openrouter-setup.md`](docs/openrouter-setup.md).
The docs index is [`docs/README.md`](docs/README.md).

---

## Output layout

Everything for an episode lands under your ComfyUI `output/otr/` tree:

- `output/otr/episodes/<episode>/` — working assets (audio, frames, intermediate clips).
- `output/otr/obs/` — the **finished, playable episodes** (what you watch / publish).

The published file is named after what produced it, so a folder of episodes is
readable at a glance without opening any of them:

```
<title>_<timestamp>__<style>__<video>__<image>__<tts>__<bank>_final.mp4

arms_at_the_ready_20260903_092133__cartoon__wan_ti2v__z_image_turbo__indextts2__public_domain_final.mp4
```

A lane that renders no stills reports `none` in the image field rather than
borrowing another episode's engine. The archival copy under `episodes/` keeps a
different, pipeline-stage name — that one is provenance, not something to read.

Point OBS (or any player) at `otr/obs/` for a continuous broadcast — new finished episodes
appear there as they render.

---

## Troubleshooting

- **"SERVER DID NOT COME UP" on headless boot** — set `PYTHONUTF8=1` and
  `PYTHONIOENCODING=utf-8`; a non-UTF-8 console crashes on the first emoji log line.
  (The shipped launcher `scripts/_otr_soak_server_launch.cmd` already sets both, and a
  regression guard in the sibling survival-guide repo keeps it that way.)
- **An engine "fails loudly" mid-render** — that's by design; check the log for the missing
  model/dependency. The render stops there rather than substituting a different engine, so fix
  the named dependency (or select the procedural CRT path) and run it again.
- **Out of VRAM on a local video tier** — use the canonical procedural path or an
  explicit lighter/cloud profile.
- **No audio under the end credits** — known limitation: the credits scroll can outlast the
  master mix's closing theme. Tracked for a fix.
- **Nodes don't appear after install** — restart ComfyUI; confirm you're on the `v2.0-alpha`
  branch.
- **`neither kokoro backend is installed` (or `kokoro is not installed`) at the first voice
  line** — on Python 3.13 run `python -m pip install kokoro-onnx` with ComfyUI's own
  interpreter (a registry install older than 2.0.0-alpha.17 does not carry it); on 3.12
  `pip install kokoro`. The message names the exact line. Or open **OTR_CastLock** and set
  `voice_bank` -> `bark_legacy`, `char_voice_engine` -> `bark`, `announcer_voice_engine`
  -> `bark`, then queue again.
- **`kokoro ONNX model not found`** — the one-time boot fetch of the 326 MB model did not
  complete (offline boot, or `HF_HUB_OFFLINE=1`). Run the `huggingface-cli download`
  line the message prints, then queue again; nothing downloads during a render.
- **`IndexTTS2 Path B not installed`** — the 16 GB canonical graph's character voice needs its
  own one-time installer (`scripts\_otr_indextts2_install.ps1`, section 4), or switch
  `char_voice_engine` to `bark`.
- **A still takes about 42 minutes on an 8 GB card, and the log says `loaded partially;
  0.00 MB usable`** — the writer LLM was still on the card when the image stage began.
  Fixed in commit `da2b7a36` (2026-09-02); update the pack. Nothing in the launch line
  fixes it.
- **The render says a node class is missing (`UnetLoaderGGUF`, `LTXV*`, `ADE_*`)** — that lane
  needs one of the node packs in section 2b; the message names which.
- **A gated model returns HTTP 401** — the LTX 2.5 weights and `gemma-2-2b-it` need a terms
  click on Hugging Face plus a read token (section 3); every default weight is ungated.

---

## Quality discipline

Development runs under a sibling QA harness — the
[ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide):
a machine-readable Bug Bible (329 entries and growing) distilled from this project's live production
incidents, plus a static regression suite that runs against this pack after every change.
Production bugs are staged in [`docs/PROD_BUG_LOG.md`](docs/PROD_BUG_LOG.md) and promoted
to the Bible in verified batches. Only bugs that actually failed in a live run qualify —
review findings never create entries on their own.

### Soak runs — how a change is actually proven

A green unit suite proves the code parses, not that an episode renders. Nothing here is
called done until it has produced a finished, playable episode in `output/otr/obs/`.

A **soak leg** is one real end-to-end render through the canonical workflow. It passes only
when all three of these hold — any one alone is not a pass:

| Signal | Where | Means |
|---|---|---|
| `RESULT SUCCESS` | leg log | the graph completed |
| `obs_publish OK` | server log | the episode was published |
| the `.mp4` on disk | `output/otr/obs/` | it is actually watchable |

A finished render leaves the server **resident** at ~9–10 GB and 1% GPU — that is the
no-teardown behavior, not a crash. Read the log for `Prompt executed` before declaring a
run dead, and reset the box before the next boot.

**Sweep drivers** run a matrix of legs unattended, one at a time (ComfyUI serializes
prompts, and this project is sequential-execution only). Each waits for the queue to drain
before submitting, so a leg's timeout measures its own render rather than its predecessor's;
a failed leg is logged and the sweep continues; and the receipt JSON is rewritten after
every leg, so killing a run mid-flight still leaves a complete record.

- [`scripts/otr_llm_image_upscale_sweep.py`](scripts/otr_llm_image_upscale_sweep.py) —
  every curated local LLM in both writer slots, across image engines, stills, and upscalers.
- [`scripts/otr_bank_engine_sweep.py`](scripts/otr_bank_engine_sweep.py) — the smallest local
  model (`gemma-4-E2B-it`, 3.0 GB) in **both** writer slots across every runnable source bank
  and all five local image engines. A 2B model is the worst case for structured extraction,
  so a bank that survives it survives everything above it.

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_bank_engine_sweep.py
```

Engine dropdowns move **only** through profile `role_overrides` in `config/profiles/*.json`
— exactly what a human clicking the announcer / music / character dropdowns and saving the
graph would produce. A sweep driver never edits `workflows/otr_canonical.json` and never
pokes `widgets_values`; the image and video engine widgets are managed and refuse ad-hoc
patching by design.

A `--dry-run` validates prompt shape without rendering, and **is not a pass** — it needs
`--offline-schemas`, and a dry leg finishing in 0.1 min has proven nothing about a model.

---

## Known limitation: character drift

**Some episodes will hand a line to the wrong character**, and this pack does not
fix that. You may hear a character claim a job that belongs to someone else, say
they don't know something they plainly do, or — rarest and most obvious — address
themselves by name. It is uncommon, it does not break a render, and the episode
still plays. But it is real and you should know about it before you run this.

Everything else in the acceptance test **is** handled. A post-story clean stage
reads every spoken row with a model and rewrites anything that is not speech —
stage directions, sound cues, narration that would otherwise be read aloud by the
voice actor. That works, and it is measured (see below).

**Why character drift is not fixed, honestly.** It was built and tested, and it
did not pass. On a planted test set where the same line appears twice — once in
the wrong character's mouth and once in the right one — the detector found 3 of 6
planted defects on one run and 1 of 6 on the next, with an identical detector and
identical inputs. It reliably caught only the blatant case (a character naming
themselves) and never caught the subtler ones. A detector that unstable cannot be
trusted to rewrite dialogue, so it ships **disabled** rather than quietly making
episodes worse.

**The constraint is hardware, not design.** This project is deliberately 100%
local and offline — the reference machine is a 16 GB laptop GPU, and the largest
model that comfortably fits is in the 12B class. Judging whether a line belongs to
a particular character means holding the whole cast, and who knows what, in mind
while reading a single sentence. That is a harder ask than spotting a stage
direction, and a 12B is not reliable at it.

**If you want to chase it:** the pass is written, tested and ready — set
`JUDGE_ATTRIBUTION = True` in `nodes/_otr_ledger_clean.py`, and use a **frontier
model well above what a 16 GB card can run**. The measurement rig is included so
you can check whether your model actually does better rather than taking anyone's
word for it:

```bash
python scripts/otr_clean_stage_lab.py --f2 --model <your-model>
```

It scores recall (planted defects caught) *and* false alarms on clean lines, which
is the half a normal render cannot show you. Do not judge a change on recall
alone — it is easy to catch everything by suspecting everything, and that rewrites
good dialogue.

The same rig, without `--f2`, measures the stage-direction cleanup that **is**
shipped and on by default.

---

## The LEMMY easter egg

Every so often a character named **Lemmy** makes a cameo — a small tribute carried across the
project's generations. Born of the machine, still raising hell on the airwaves. 🤘

---

## Changelog

The current line is **v2.0-alpha** (Open Video Model Platform; per-role video AND image
engines; five independent story source banks; per-platform workflow variants; frozen 48 kHz
audio master, byte-identical in the archival copy). Full per-version history is in the git log
and the GitHub Releases page.

## License & Credits

See [`LICENSE`](LICENSE). Built on ComfyUI and the open-source HuMo / LTX / Wan / AnimateDiff /
Z-Image-Turbo / Lumina / FLUX.2 Klein / IndexTTS2 / Kokoro / Stable Audio ecosystems, plus
several optional engines (Chatterbox, Dia, Bark, MusicGen, and others) — thanks to all of
their authors.

**A few optional, off-by-default pieces carry restricted terms, not open licenses:**

- `flux_gen1` (Flux.1-dev) — BFL non-commercial license.
- `ideogram4_local` — non-commercial model agreement; code ships, weights don't.
- `h3_low_video` / `h3_low_audio_in` (MiniMax H3) — a personal, non-transferable
  authorization the maintainer obtained directly from MiniMax; it does not carry over to
  your install.
- `animatediff15_v3_haunted_video` — the shipped motion module publishes no
  license grant at all (`commercial_clean = False`); fine for personal use, not cleared for
  commercial redistribution.
- The `shakespeare` story bank adapts Folger Digital Texts, which are CC BY-NC 3.0
  (noncommercial).

None of these are required for a first run — the canonical workflow's shipped defaults
(Gemma writer, Z-Image-Turbo, IndexTTS2/Kokoro/Stable Audio, procedural video floor) are all
open and commercial-friendly. Check each engine's own license before commercial use of the
others.
