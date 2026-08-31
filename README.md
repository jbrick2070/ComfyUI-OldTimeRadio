# ComfyUI-OldTimeRadio (SIGNAL LOST)

Turn **real news, public-domain stories, Shakespeare, or fully original LLM fiction** into a
finished **radio-drama video** — script, voices, music, and CRT-style video — fully automated
inside ComfyUI. Drop it in, queue one workflow, walk away, and a complete episode lands in
your output folder.

**Pipeline:** story source → LLM script → character voices + announcer + music themes (a
swappable 7-voice / 5-music engine roster; IndexTTS2 + Kokoro + Stable Audio 3 ship as the
defaults) → 48 kHz master mix → model-agnostic video (procedural CRT floor by default, or
HuMo / LTX / Wan / AnimateDiff / MiniMax H3 once you dial a heavier lane in) → final MP4.

100% local by default. No API keys required. Optional hosted LLM and all-cloud
routes exist; they stay off unless you turn them on.

> **Already installed it? Load the show:** **Workflow → Browse Templates →
> EXTENSIONS → comfyui-old-time-radio → otr_canonical**, then **Queue Prompt**.
> The 25 `OTR_` nodes are the parts; that workflow is the thing you actually run.

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

**Renting a GPU instead?** Follow [docs/RUNPOD_INSTALL.md](docs/RUNPOD_INSTALL.md) -- it covers any remote Linux host, not only RunPod.

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

**The ComfyUI Registry route does not currently work, and Manager cannot install
this pack by any route.** Both published versions are `Flagged`
([registry page](https://registry.comfy.org/publishers/fluxus/nodes/comfyui-old-time-radio)),
so `latest_version` resolves to null: `@latest` has no target, and Manager
refuses the `nightly` git path on any network-exposed instance. Checked live
2026-08-31 -- 2 versions, 0 active. If Manager reports "not a CNR node" or
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
| `animatediff15_*` — including **`otr_nvidia_8gb_haunted`**, the 8 GB default | [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) | provides the `ADE_*` classes the haunted lane samples through |

**You do not have to memorise this.** If a pack is missing, the render stops
with a named error that now tells you which pack to install and where to get
it — it does not fail silently or half-render. But it stops at render time,
*after* the model weights have downloaded, which is why it is written here too.

**What the 8 GB haunted profile actually downloads on a clean machine: ~16 GB.**
The video lane itself is only ~3.9 GB (SD1.5 1.99, `v3_sd15_mm` 1.56, the
domain adapter 0.10, kokoro voices 0.30). The rest arrives through the
Hugging Face cache the first time the pipeline runs — the writer
(`gemma-4-E2B-it`, ~9.6 GB), `musicgen-small` (~2.2 GB) and `Kokoro-82M`
(~0.3 GB). Every one of them is ungated and needs no token. Worth knowing
before you start it on a metered connection.

### 2b-ii. The GGUF writer lane — install 0.3.33, not the latest

Only needed if you select a `*-GGUF` writer row. It is the lane that runs a
large writer on a small card, and **it is the only local writer lane that works
off NVIDIA at all** — bitsandbytes NF4 is CUDA-only, so every Mac, AMD and CPU
profile in this pack (`otr_mac_mps`, `otr_amd8_rocm`, `otr_amd16_rocm`,
`cpu_floor`) runs GGUF through in-process llama.cpp. No Ollama, no sidecar
process, no extra port.

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
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

That is the whole file. OTR reads this location *and* the one your `HF_HOME`
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

The shipped canonical workflow is lighter than you'd expect: its video-role dropdowns default
to the **procedural still/CRT floor** (no GPU video checkpoint required at all), and its image
role defaults to **Z-Image-Turbo** (Apache-2.0, no license friction). So a first run needs only:
the local writer LLM (Gemma or your own choice), Z-Image-Turbo, and the default voice/music
weights (IndexTTS2, Kokoro, Stable Audio 3). The heavier local video checkpoints — HuMo, LTX,
Wan, AnimateDiff, MiniMax H3 — are **optional upgrades** you dial in later via the
`OTR_VideoDirector` dropdowns; see [Which video models fit your card](#which-video-models-fit-your-card)
before downloading any of them. If a model is missing, the engine fails **loudly** and stops —
it never silently substitutes another model or quietly produces garbage. There is no automatic
fallback: the procedural CRT path is a route you **select** (and the canonical workflow ships
with it selected), not a net that catches a failed engine. Watch the console on the first run;
it names any missing weight and where it expects it.

### 5. Run it

1. Open **Workflow → Browse Templates**, scroll the left sidebar to **EXTENSIONS**,
   click **comfyui-old-time-radio**, and pick **otr_canonical**. (The console prints
   this same path on every start, right under the `[OldTimeRadio]` load banner.)
   *Prefer files? `workflows/otr_canonical.json` inside the installed pack is the
   same graph — drag it onto the canvas.*
2. Hit **Queue Prompt**.
3. Walk away. Script, voices, music, mastering, and video all run automatically. The shipped
   graph rolls a random story bank each run and renders through the procedural still/CRT floor
   — the fast, guaranteed-to-complete path. Swap dropdowns once you're ready for a specific bank
   or a GPU video engine.
4. Find the finished episode in **`output/otr/obs/`**.

---

## Requirements

- **GPU:** an NVIDIA card is recommended for the local video engines. The shipped canonical
  workflow renders through the procedural still/CRT floor by default (no GPU video model
  required); heavier local/cloud routing is opt-in via the `OTR_VideoDirector` dropdowns or
  explicit profile overrides. Episode length is set by act count, not a word target.
- **OS:** Windows or Linux. Tested heavily on Windows + RTX (Blackwell/sm_120).
- **Other setups:** per-platform workflow variants + recipes ship in-repo (16 GB NVIDIA
  canonical, cloud-lane variant, Mac ~10 GB ceiling, AMD). The Mac/AMD variants are
  drafts — not yet verified on real hardware.
- **Disk:** the model set is large (tens of GB). Episodes are a few dozen MB each.

---

<!-- BEGIN GENERATED: machine-matrix -->

## What works on what machine

| your machine | writer | video | voice | music | image | status |
|---|---|---|---|---|---|---|
| **8 GB NVIDIA (e.g. RTX 4060)** | -- | -- | -- | -- | -- | **no profile yet** |
| **16 GB or more NVIDIA (RTX 5080, rented 24 GB, ...)** | -- | -- | -- | -- | -- | **no profile yet** |
| **10-15 GB NVIDIA (e.g. RTX 4070, 3080 12 GB)** | -- | -- | -- | -- | -- | **no profile yet** |
| **AMD / ROCm** | -- | -- | -- | -- | -- | **no profile yet** |

**Use the profile named for your machine** -- pass it to `--profile`, or pick the matching entries in the dropdowns. The engine names above are exactly the dropdown text.

<!-- END GENERATED: machine-matrix -->

## Which video models fit your card

**For the full picture -- writer, video, voice, music and image per machine, with what is PROVEN versus merely shipping -- see [docs/MACHINE_MATRIX.md](docs/MACHINE_MATRIX.md).** It is generated from the profiles themselves, so it cannot drift from what the code actually offers. The section below is the video-specific detail behind it.


**This table is the profile.** Pick your card, read the column, choose that name
in the `OTR_VideoDirector` dropdown. Names below are exactly the dropdown
entries.

Every figure is MEASURED, not estimated, and each says where it came from. A
blank verdict means nobody has measured it -- that is recorded as unknown rather
than guessed, because a guessed VRAM number is the one thing a user cannot
recover from.

**Read the 8 GB column as CANDIDATES, not promises**, and here is exactly how
far the evidence goes.

The strongest 8 GB evidence is a **CLAMPED SIMULATION**, not a run on an 8 GB
card. `MiniMax H3 MIME I2V` was rendered on the 16 GB card under a
`--reserve-vram 12` clamp, which forces ComfyUI to operate inside roughly an
8 GB budget:

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
is why the row above reads PROVEN rather than `?`, and why the per-machine table
is now GENERATED -- see
[docs/MACHINE_MATRIX.md](docs/MACHINE_MATRIX.md). A hand-kept compatibility
claim goes stale in the direction that costs a user the most: telling them their
card cannot do a thing it has already done.

For contrast, `ltx25_high_video` measured **14.48 GiB** on the 16 GB card --
roughly 6.5 GiB past an 8 GB card's entire capacity, which is why its label says
5080-only rather than "high VRAM".

### Local video models

| Dropdown name | Measured VRAM | 8 GB | 12 GB | 16 GB |
|---|---|:--:|:--:|:--:|
| `ltx098_low_video (16:9)` | 6.8 GiB @ 512x288x161 | maybe | yes | yes |
| `h3_low_audio_in (16:9)` | 6.9-7.2 GiB @ 864x480 | **likely** | yes | yes |
| `h3_low_video (16:9)` | **7.28 GiB under an 8 GB clamp** | **likely** | yes | yes |
| `ltx23_low_audio_in (16:9)` | 7.36 GiB @ 1024x576x193 | maybe | yes | yes |
| `animatediff15_v3_haunted_video (16:9)` | ~3.9 GB of weights, hold-2 cadence | **PROVEN** | yes | **PROVEN** |
| `wan22_high_video (16:9)` | 12.1 GiB @ 832x480x193 | no | maybe | yes |
| `humo17_high_audio_in_portrait (portrait)` | 12.84 GiB @ 480x832x129 | no | maybe | yes |
| `humo14_high_audio_in_wide (16:9)` | 13.06 GiB @ 832x480x97 | no | no | yes |
| `humo14_high_audio_in_portrait (portrait)` | 13.22 GiB @ 480x832x97 | no | no | yes |
| `ltx23_high_video (16:9)` | 13.3 GiB @ 1024x576x169 | no | no | yes |
| `wan22_high_fast (16:9)` | 12.8 GiB measured 2026-08-22 | no | maybe | yes |
| `ltx25_high_video (16:9)` | **14.48 GiB measured** | no | no | **5080-only** |
| `humo17_high_audio_in_wide (16:9)` | not measured at this aspect | ? | ? | yes |
| `mesh_stage (16:9)` | not measured | ? | ? | yes |

`animatediff15_v3_haunted_video` is the ONE surviving AnimateDiff lane
(operator directive 2026-08-23: "delete any animatediff that are not
haunted"). Its former peers — `animatediff15_video`, the hold-3/hold-5
cadence variants, `animatediff15_v2_video`, and `animatediff15_v3_video` —
are retired and tombstoned in the engine registry; they no longer appear in
any dropdown.

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

---

## Quality discipline

Development runs under a sibling QA harness — the
[ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide):
a machine-readable Bug Bible (319 entries and growing) distilled from this project's live production
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
