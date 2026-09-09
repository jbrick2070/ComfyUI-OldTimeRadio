# Running ComfyUI-OldTimeRadio on a Mac (Apple Silicon)

**Status: WORKING.** The shipped canonical renders end to end on Apple Silicon
and publishes to `otr/obs/` (first proven 2026-09-07 on a Mac mini M4, 16 GB,
macOS 26.6.2, ComfyUI Desktop 0.34.6, Python 3.13, torch 2.12.1).

This is the practical guide. The measurements behind it are in
`MAC_LESSONS_LEARNED.md`; the engine-by-engine survey is in
`MAC_ENGINE_TEST_PLAN.md`.

---

## 1. Press Run -- what works with no extra steps

Install the pack, load `otr_canonical` from
**Workflow > Browse Templates > EXTENSIONS > comfyui-old-time-radio**, press Run.

The shipped graph is deliberately the **only fully-local configuration Apple
Silicon has**: writer on Metal, Kokoro voices, Stable Audio 3 music, and three
visualizer video lanes that mint no still, so **no image weights and no API key
are needed**.

| stage | engine | notes |
| --- | --- | --- |
| writer | `Qwen/Qwen3.5-4B` on `mps` | ~6.5 tok/s. **Keep `llm_quant_policy` = `none`** |
| voices | `kokoro` | runs through kokoro-onnx on CPU on Python 3.13, by design -- that is what makes it auto-install |
| music | `stable_audio_3` | ungated, commercially clean, self-fetching |
| video | `viz_mxc_cpu`, `viz_green`, `viz_camera` | zero dependencies, no stills |

### Two settings that will ruin your day if you change them

* **`llm_quant_policy` must stay `none`.** NF4 *works* on Metal but runs at
  0.3-0.5 tok/s against 14.47 on CUDA -- there is no Metal kernel, so
  bitsandbytes falls back to a Python dequantise loop. It is slow enough that
  the 40 s `NewsCurationDeep` budget expires and the run dies with a confusing
  `_LLMTimeoutWorkflowPause`. It does not look like "quantisation is slow".
* **Do not select `viz_mxc_mandala`.** It needs `pycairo`, which has no macOS
  wheel (see section 3).

### Do not run the test suite while a render is running

Twice on 2026-09-08 the known-fail guard reported NEW failures -- four HuMo
beat-session tests once, five audition/citation tests another time -- that did
not reproduce at all on an idle machine minutes later. Both runs happened while
a render had the box 19 GB into swap.

On a machine with separate VRAM the suite and a render coexist. Here they
compete for the same physical memory, and a suite run under that pressure
produces failures that are about the pressure, not about the code. **Take your
before/after comparison on an idle box, or you will spend an hour chasing a
regression you did not write.**

### An OOM here takes the MACHINE down, not the render

**Read this before you select any engine.** On a discrete GPU, running out of
VRAM raises a Python exception, ComfyUI catches it, and you read a traceback. On
Apple Silicon there is no separate VRAM: ComfyUI's "offload device" is the same
physical memory the model is already in, so there is nowhere to spill to. The OS
resolves it by killing processes. In practice the whole machine goes down, and
there is **no traceback afterwards** -- the log simply stops mid-load.

That happened here on 2026-09-08 loading `wan_ti2v` with an fp16 UNET: the log's
last line is `WAN22 ... loaded completely; 9536.40 MB, full load: True`, and then
nothing.

**So do not size a model by trying it.** Two consequences:

* The pack now refuses an oversized local video lane at the gate rather than
  loading it -- see "What protects you, and what does not" below.
* Anything the guard cannot read is on you. Check the weight sizes first
  (`scripts/otr_fetch_lane_weights.py` `LANE_INFO` has them per lane) against
  the **Metal working-set ceiling**, which is about 75% of physical RAM
  (11.8 GiB on a 16 GB machine) -- not the RAM figure on the box.

### Memory: 16 GB is the floor, and it is tight

The writer alone peaks near 14 GB inside ComfyUI on a 16 GB machine. It works,
but close other applications first -- a remote-desktop session competing for RAM
is enough to trigger an OOM kill. Note `ps` under-reports badly on Apple Silicon
(0.33 GB for a 14 GB process); use `footprint -p <pid>` and read
`phys_footprint`.

---

## 2. What CANNOT work locally on Apple Silicon, and why

**CORRECTED 2026-09-07 (this entry previously said Apple Silicon has no local
image engine -- that was wrong).** `z_image_turbo` LOADS AND EXECUTES ON METAL.
It is blocked by MEMORY, not by the device:

```
Requested to load ZImageTEModel_   loaded completely;  7672.25 MB   full load: True
Requested to load Lumina2          loaded completely; 11739.54 MB   full load: True
...
RuntimeError: MPS backend out of memory (MPS allocated: 13.01 GiB,
  other allocations: 5.96 GiB, max allowed: 20.13 GiB). Tried to allocate 1.46 GiB
```

Both models load fully; it dies in the KSampler needing about 20.4 GiB against a
20.13 GiB ceiling -- it misses by roughly 300 MB. The checkpoint alone is
12.3 GB.

**So the honest statement about Z-Image is about RAM, not about Metal.** It is
not CUDA-only, it is large: the `["cuda"]` declaration is wrong in KIND. A 32 GB
Mac would very likely run it, and that is a hardware question rather than a
portability defect anyone can patch.

### The conclusion this section used to draw was WRONG -- read this part

Until 2026-09-08 this section ended: *"no local image engine fits in 16 GB ...
nothing local can mint a still here ... the visualizer lanes are the only
fully-local video path Apple Silicon has. The shipped canonical is not a
conservative choice, it is the only one that runs."*

Every clause of that is now false, and the mistake is worth naming because it is
the one this whole document exists to prevent: **the conclusion was drawn from
ONE engine's failure.** Z-Image did not fit, so "no image engine fits". It was
never a measurement of the set, and the set was never enumerated.

What the set actually contains, measured on the same 16 GB M4:

| | | |
| --- | --- | --- |
| `sd15` | **works** | 2.1 GB, local, mints stills on Metal (section 5) |
| all four `still_*` lanes | **work** | fed by `sd15` (section 9) |
| `ltx_8gb` (LTX 0.9.8) | **works** | real local video diffusion (section 7) |

So the shipped canonical IS a conservative choice: the visualizer lanes are what
you get with **zero downloads**, not what you get because nothing else runs.

**Still declared `["cuda"]`, and here honesty requires three buckets, not one:**

* **Really NVIDIA-only** -- `ideogram4_local` and both `minimax_*` need `nvfp4`
  artifacts; `ltx_audio_in` fails CLOSED without NVML (`eng_ltx_av.py:749-756`),
  and unlike `ltx_8gb` that is a real code gate, not a row. `indextts2`,
  `chatterbox` and `dia` are sidecars pinned to cu128 torch with a PowerShell-only
  installer. None of these is a paperwork problem.
* **Blocked by size or by a missing node pack, not by Metal** -- `z_image_turbo`
  (above), `humo` / `humo_14B_169` (26.7 GB, fp8 text encoder), `ltx_video`
  (22B), the three `ltx25_*` (gated HF repo + ComfyUI-GGUF), both
  `animatediff15_v3_*` (section 8).
* **MEASURED FATAL on 16 GB unified memory** -- `wan_ti2v` and `fastwan_8gb`.
  `wan_ti2v` is not a size guess: it loaded fully on Metal and killed the
  machine. Its shipped configuration is a 9.37 GB GGUF set rather than the fp16
  one that died, so it may fit -- but see the corruption note below before you
  spend the download. `fastwan_8gb` subclasses it and hoists MORE, at a ~21.8 GB
  peak; it is further away, not closer.
* **NEVER TESTED, and nobody should pretend otherwise** -- `flux2_klein`,
  `lumina_image`, `flux_gen1`, `humo_1.7B`, `humo_1.7B_169`, `mesh_stage`,
  `stable_audio_music`, and the whole upscale namespace. `flux2_klein` is the
  interesting name: `config/machine_classes.json` already carries an operator
  ruling naming Klein 4B for Mac, and it has never been run on Metal.
  **It is 10.99 GB, not 2.6 GB** -- three files, and an earlier draft of this
  line quoted only the first: `flux-2-klein-4b-Q4_K_M.gguf` (2.60 GB) +
  `qwen_3_4b.safetensors` (8.04 GB text encoder) + `flux2-vae.safetensors`
  (0.34 GB), per `scripts/otr_provision.py:175-207`. The encoder is sequential,
  so the peak is about 8 GB rather than 11 -- it fits, but it is not a small
  download, and `sd15` at 1.99 GB remains the smallest local image engine here.

### What protects you, and what does not

The pack ships a **unified-memory weight floor**
(`nodes/_otr_video_engines/motion_common.py`, wired at the render gate). Before a
local video engine loads, it computes that engine's PEAK CONCURRENT RESIDENCY
from its actual loader files on disk -- the larger of (its text encoder alone)
and (everything else together), not a sum of all of them -- and refuses if that
will not fit:

```
wan_ti2v needs 10.6 GiB of weights resident and this host has 10.3 GiB of
accelerator budget (11.8 GiB free, less 1.5 GiB reserved for activations and the
OS). On UNIFIED memory there is no separate host RAM to offload into ... so
loading this would not fail the render, it would take the MACHINE down.
```

It is a no-op on CUDA, where a card can offload to host RAM and an oversized model
is merely slow.

**Now the limits, because a guard you over-trust is worse than none:**

* It reads an engine's weight names by duck-typing two adapter methods. It can
  read `ltx_8gb`, `wan_ti2v`, `fastwan_8gb`, `humo` and its variants, and
  `mesh_stage`. **Every other lane is unguarded** -- not blocked, just unchecked.
* **It does not cover image engines at all.** `z_image_turbo`, `flux2_klein`,
  `lumina_image` and `flux_gen1` are on you.
* It is a FLOOR: it weighs the files, not the activations, so it catches "the
  weights alone do not fit" and nothing subtler. `z_image_turbo` died in the
  KSampler needing 20.4 GiB against a 12.3 GB checkpoint -- a floor check would
  not have predicted that gap.
* It fails OPEN on anything it cannot resolve exactly. That is deliberate: a
  false refusal blocks work that succeeds, and only one of those two errors is
  acceptable in a guard nobody has calibrated.
* `OTR_UNIFIED_MEMORY_HEADROOM_MB` overrides the 1.5 GiB reservation ONLY -- the
  weight check itself always runs. A small or zero value weakens the reserve; a
  negative or malformed one falls back to the default rather than disarming it,
  deliberately, so a typo cannot silently turn the guard off.

**Names in that list are INTERNAL ids; the dropdown shows public labels.**
`ltx_8gb` is `ltx098_low_video`, `wan_ti2v` is `wan22_high_video`, `fastwan_8gb`
is `wan22_high_fast`, `ltx_video` is `ltx23_high_video`, `ltx_audio_in` is
`ltx23_low_audio_in`, the `minimax_*` pair is `h3_low_video` / `h3_low_audio_in`.
The mapping is `nodes/_otr_shared/public_engines.py`.

---

## 3. Engines that need manual steps -- and exactly what to do

### `viz_mxc_mandala` -- Windows-only in practice

`pycairo` publishes Windows wheels, an sdist, and **no macOS wheel**. On a stock
Mac `pip install pycairo` falls back to the sdist and fails to build, because
libcairo headers are not present. Measured 2026-09-07.

```bash
brew install cairo pkg-config
pip install pycairo
```

Without Homebrew this engine cannot run. **Use `viz_mxc_cpu`, `viz_green` or
`viz_camera` instead** -- zero dependencies, no cairo. The engine now says this
itself when you select it on a Mac.

Linux has the same problem for a different reason (no Linux wheels):
`apt install libcairo2-dev pkg-config` first.

### `animatediff15_v3_*` -- weights have NO auto-fetcher

**THREE** files are named in the CUDA profiles' `model_requirements` and nothing
downloads any of them. On any platform you must place them by hand:

| file | source repo | destination |
| --- | --- | --- |
| `v1-5-pruned-emaonly-fp16.safetensors` (1.99 GB) | `Comfy-Org/stable-diffusion-v1-5-archive` | `models/checkpoints/` |
| `v3_sd15_mm.ckpt` (1.56 GB) | `guoyww/animatediff` | `models/animatediff_models/` |
| `v3_sd15_adapter.ckpt` (~95 MB) | `guoyww/animatediff` | `models/loras/` |

The third one is the easy one to miss -- it is the v3 DOMAIN ADAPTER, a LoRA on
the image model rather than part of the motion module, and preflight fails
CLOSED on it (`domain_adapter=v3_sd15_adapter.ckpt`). Getting two of three gets
you nothing.

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Comfy-Org/stable-diffusion-v1-5-archive','v1-5-pruned-emaonly-fp16.safetensors'))
print(hf_hub_download('guoyww/animatediff','v3_sd15_mm.ckpt'))
print(hf_hub_download('guoyww/animatediff','v3_sd15_adapter.ckpt'))"
```

**`mm-p_0.5.pth` is NOT one of these, whatever a summary tells you.** That 1.82 GB
file belongs to the RETIRED lane in `eng_ghost_signal.py`
(`GHOST_MOTION_MODULE_NAME`). The shipped `animatediff15_v3_haunted_video` lives
in `eng_ghost_signal_official.py` and uses `MM_V3_NAME` = `v3_sd15_mm.ckpt`. Do
not download it.

### Then just set the dropdowns -- there is no profile for this

Open `otr_canonical`, and on **OTR_VideoDirector** set whichever video roles you
want to `animatediff15_v3_haunted_video (16:9)`. That is the whole selection
step. The image dropdowns are inert for this lane -- it is `text_to_video` and
mints no still -- so leave them.

A Mac-specific profile for this briefly existed and was **deleted**. It encoded
three dropdown settings, a canvas the engine overrules anyway, and a preflight
list, which is a second place to keep in sync for no benefit; it was wrong twice
in its first hour. The dropdowns are the path.

**Do not expect a clip-length knob.** This lane declares `max_frames=0` -- one
unsplittable timeline -- and sits deliberately outside `PLANNING_CAP_ENGINES`,
so `video.max_render_frames` does not govern it and adding it there would split
the beat into jump cuts at roughly twice the latent cost. Section 10.7 has the
arithmetic.

**Install the pack with the PINNED commit, not a fresh clone.** `main` is the
version that issue #576 reports producing colored noise (section 11):

```bash
OTR_COMFY_ROOT=/path/to/ComfyUI \
  <ComfyUI Python> -c "
import importlib.util as u
s=u.spec_from_file_location('p','scripts/otr_provision.py')
m=u.module_from_spec(s); s.loader.exec_module(m)
m.ensure_animatediff_pack('/path/to/ComfyUI')"
```

Calling `ensure_animatediff_pack` directly gets you the pin without the
`--packs-only` run failing on ComfyUI-LTXVideo's git-lfs requirement (10.2).
This pack ships no `requirements.txt`, so it adds NOTHING to the venv -- the
safest install in the whole document.

### THE STEP THAT COSTS AN HOUR: Comfy Desktop does not map `animatediff_models`

The pack will report, at boot:

```
[AnimateDiffEvo] - ERROR - No motion models found. Please download one and
place in: ['.../custom_nodes/ComfyUI-AnimateDiff-Evolved/models']
```

**Your motion module is fine and it is in the right place.** The problem is the
mapping: Comfy Desktop generates its `extra_model_paths` file with `checkpoints`,
`loras`, `vae`, `text_encoders` and a dozen others -- but **no
`animatediff_models` category**. So the adapter LoRA in `loras/` resolves and the
motion module beside it does not, which is a confusing half-failure.

That generated file says "do not edit manually" in its own header, and it means
it. ComfyUI accepts the flag more than once, so add a SECOND file instead:

```yaml
# otr_mac_extra_paths.yaml
otr_mac_addendum:
  base_path: '/path/to/your/shared/models'
  'animatediff_models': 'animatediff_models/'
  'animatediff_motion_lora': 'loras/'
```

```bash
python main.py --extra-model-paths-config "<the Desktop file>" \
               --extra-model-paths-config otr_mac_extra_paths.yaml
```

Confirm it took by asking the API rather than by looking at the folder:

```bash
curl -s http://127.0.0.1:8188/object_info/ADE_LoadAnimateDiffModel \
  | python3 -c "import json,sys; print(json.load(sys.stdin)
      ['ADE_LoadAnimateDiffModel']['input']['required']['model_name'][1]['options'])"
```

None of this is enough on its own -- see section 8 for the node pack, which is
the dependency that actually blocks the lane.

then copy each to the destination above. **Mac status: UNVERIFIED** -- the
engine declares `["cuda"]` and contains no NVIDIA-specific code, so the
declaration may simply be untested. Being tested; see `MAC_ENGINE_TEST_PLAN.md`.

### Cloud / API engines

`elevenlabs`, `google_tts`, `google_lyria`, `sonilo`, every `cloud_*`,
`google_image`, `ideo` and `word_razzle` all already declare `mps`. They need
**credentials, not a port** -- testing them measures an API key rather than
Apple Silicon.

An earlier draft of this paragraph called a cloud image key "the single
highest-leverage addition for a Mac, because it unlocks the `still_*` video
lanes that are otherwise unreachable". That is no longer true. `sd15` is a
LOCAL image engine and it unlocks all four `still_*` lanes for free -- see
section 5 for the engine and section 9 for the proof. A cloud key buys you
different-looking stills, not reachable ones.

---

## 4. If something breaks, read this first

| symptom | cause | fix |
| --- | --- | --- |
| **ComfyUI will not start at all**, exit 1, `ImportError: tokenizers>=0.23.1 ... found 0.22.2` | the shipped `tokenizers` pin in versions `.24`-`.28` | `pip install 'tokenizers>=0.23.1,<0.24'`, and use `2.0.0-alpha.29` or later. **Do NOT reinstall from Manager** -- it serves the broken version, and a bricked boot cannot be repaired from the UI because Manager IS a ComfyUI extension |
| music is noise, "like a broken cassette" | ComfyUI's sub-quadratic attention is wrong on MPS | fixed automatically by the pack's prestartup. If you disabled it, pass `--use-pytorch-cross-attention` |
| render dies at the mp4 encode, `winget install ffmpeg` | ffmpeg/ffprobe were not declared dependencies | `pip install -r requirements.txt` then **`ffdl install`** -- see the ffprobe row below; the pip install alone is not enough |
| `NewsCurationDeep exceeded 40s` | `llm_quant_policy` is not `none` | set it to `none` (section 1) |
| OOM / the machine freezes mid-render | 16 GB shared with macOS and everything else | close other apps; the writer peaks near 14 GB |
| a video/image engine refuses with `EngineUnusable` | **NOT the `["cuda"]` row** -- that row is not a render-time gate and never refuses anything (see `docs/ADDING_IMAGE_AND_VIDEO_LANES.md`). The refusal is the engine's own `assert_usable`: a missing weight, a missing node class, or a missing NVML/vendor probe | read the message -- it names the artifact or class. A `["cuda"]` row is a claim about what has been PROVEN, not a lock |
| a lane you did not pick starts a huge download at Queue | the image dropdowns still say `z_image_turbo` and the video lane you picked consumes a still | set ALL THREE image dropdowns to `sd15` **before** you queue. The validator provisions the selected image engine, so an unchanged `z_image_turbo` fetches ~20 GB of weights that cannot run here |
| render dies at the scopes stage, `Unrecognized option 'vsync'` | this Mac ships **ffmpeg 9.0**, which REMOVED `-vsync` (deprecated since 5.1) -- it dies at argv parse before a frame is read | already handled: `scope_draw.cfr_flags()` probes the installed binary once and returns `-fps_mode` here (verified 2026-09-08 on this machine). If you see this, something bypassed the probe -- do not hardcode either spelling |
| `ffprobe` still not found after `pip install -r requirements.txt` | `ffmpeg-downloader` ships the fetcher, not the binaries | run `ffdl install` once (or `brew install ffmpeg`). `imageio-ffmpeg` supplies ffmpeg only -- ffprobe has no equivalent bundled wheel |

**Error messages that lie on a Mac** (cosmetic, being cleaned up):
`[StoryOrchestrator] CUDA warmup complete` and
`orphan worker still on GPU ... racing the orphan's CUDA kernels` both print on
machines with no CUDA. Neither indicates a CUDA code path.


---

## 5. Local image generation on Apple Silicon -- what works and what does not

**MEASURED 2026-09-08 on a Mac mini M4, 16 GB.**

### Z-Image has NO viable Mac configuration

| variant | size | result |
| --- | --- | --- |
| `z_image_turbo_bf16` | 12.31 GB | **MPS OOM.** Both models loaded (`full load: True`), died in the KSampler needing ~20.4 GiB of a 20.13 GiB ceiling |
| `z_image_turbo_int8_convrot` | 5.78 GB | **`NotImplementedError: 'aten::_int_mm' is not implemented for MPS`.** There is no int8 matmul on Metal at all |
| `z_image_turbo_nvfp4` | 4.51 GB | Blackwell-native fp4 -- NVIDIA only |

Its own docstring recommends `nvfp4` for low VRAM, which is useless on a Mac.
There is no fourth option: the model is either too large, or quantised in a
format Metal cannot execute.

### `sd15` is the Mac image engine

Added 2026-09-08 for exactly this reason. **1.99 GB, ungated, one ordinary
checkpoint** (MODEL + CLIP + VAE in a single file through
`CheckpointLoaderSimple` -- no split loaders, no separate text encoder, no GGUF
pack, no licence click).

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Comfy-Org/stable-diffusion-v1-5-archive',
      'v1-5-pruned-emaonly-fp16.safetensors'))"
```

then copy it into `models/checkpoints/`. Select `sd15` in any image-model
dropdown on `OTR_VideoDirector`.

**It fits every request down to 768 on the long side** (`OTR_SD15_MAX_SIDE`),
preserving aspect and snapping to a multiple of 8 -- the canonical's 832x480
mints as 768x440. That is not tidiness: SD 1.5 duplicates subjects past roughly
768, producing two heads and mirrored torsos with **no error anywhere**.

Knobs: `OTR_SD15_CKPT`, `OTR_SD15_MAX_SIDE`, `OTR_SD15_WIDTH` / `_HEIGHT`
(no-request default only), `OTR_SD15_STEPS`, `OTR_SD15_CFG`, `OTR_SD15_SAMPLER`,
`OTR_SD15_SCHEDULER`.

### AN IMAGE ENGINE IS INERT UNTIL A VIDEO LANE CONSUMES ITS STILL

The three roles in the shipped canonical all select lanes that mint no still
(`viz_mxc_cpu`, `viz_green`, `viz_camera`, all `accepts_still=False`). **So
selecting an image engine and changing nothing else proves nothing** -- the run
goes green having never called it. Flip a role to `still_motion`, `still_flat`
or `still_pan` first; all three already declare `["cuda","cpu","mps"]`.

### Image-to-video is blocked one step removed

`ltx_8gb` (LTX 0.9.8) PASSED its engine gate on `mps` once its `["cuda"]` row
was patched -- its adapter contains no NVIDIA-specific code -- and then failed
at the still it consumes, because the still came from `z_image_turbo`. With
`sd15` supplying stills this becomes testable for the first time.

## 6. Fonts: titles look wrong on macOS

**Confirmed on this machine:** `Consolas` is **absent** from macOS, and
`_otr_captions.py` named it in TWO places -- the `otr_crt` caption style (a QA
variant, not the default) and, the one that actually shipped on every episode,
the hero title card's ASS style row. libass falls back to something proportional
and wider, so the green title overruns its computed width and clips.

The `sdh_standard` style beside it asks for Arial, which macOS DOES have
(`/System/Library/Fonts/Supplemental/Arial.ttf`). That is why the white dialogue
captions look right while the green titles do not -- the bug was never
"fonts on macOS", it was one Windows-only face.

### FIXED 2026-09-08 -- `mono_font()`

`_otr_captions.py` now resolves the monospace face at EMIT time instead of
hard-coding one:

| platform | face | why |
| --- | --- | --- |
| `win32` | `Consolas` | unchanged, so the CUDA machines render byte-identically |
| `darwin` | `Menlo` | ships with every macOS since 10.6 |
| anything else | `DejaVu Sans Mono` | the near-universal Linux monospace |

Two call sites moved onto it: the `otr_crt` caption style (which now stores the
`MONO` sentinel rather than a face name) and the hero title card's style row,
which became `_title_style_line()` -- a function rather than the module constant
it was, because a constant would freeze whatever face the server booted with.

`OTR_CAPTION_MONO_FONT` overrides it outright. That is the escape hatch for a
machine missing the OS font, and it is also how you falsify the fix: set it to a
nonsense name and the captions must visibly change face. If they don't, libass
isn't reading the style you think it is.

**Why this was invisible for so long:** an ASS `Style:` line naming a font the
renderer cannot find does not error. libass asks fontconfig for a substitute,
gets a proportional sans, and draws it. There is no warning in any log. The only
symptom is that the type looks wrong -- which is easy to blame on the theme.

`Monaco.ttf` IS present at `/System/Library/Fonts/Monaco.ttf` on macOS 26.6.2, so
the credits-roll font search does resolve, and the overlapping-columns symptom
has a **different, still-undiagnosed cause**. Do not assume this fix addressed
it.


---

## 7. Local video diffusion on a Mac: LTX 0.9.8, step by step

**PROVEN 2026-09-08** on a Mac mini M4 (16 GB): `ltx098_low_video` rendered all
three beat classes and published a 108 s 1080p episode with zero errors.

### What you need

| item | size | how it arrives |
| --- | --- | --- |
| `ltxv-2b-0.9.8-distilled.safetensors` | 5.91 GB | **auto-fetched** -- already in the visual-asset manifest (`Lightricks/LTX-Video`) |
| `t5xxl_fp16.safetensors` | 9.12 GB | **auto-fetched** (`comfyanonymous/flux_text_encoders`) |
| an image engine for the still | 1.99 GB | **`sd15`, and this is the part people miss** |

**Nothing here is gated and nothing needs a manual download.** The two LTX
weights self-fetch on first use. Total on disk is about 17 GB.

### The three steps

1. **Install the pack and let it fetch.** Both LTX weights are in the manifest.
2. **Select the lanes.** On `OTR_VideoDirector`, set the video model to
   `ltx098_low_video (16:9)` for whichever roles you want, and set the matching
   image model to `sd15`.
3. **Make sure `sd15`'s checkpoint is present** (1.99 GB, ungated):

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Comfy-Org/stable-diffusion-v1-5-archive',
      'v1-5-pruned-emaonly-fp16.safetensors'))"
```

then copy it into `models/checkpoints/`.

### THE STEP THAT IS NOT OBVIOUS

**LTX 0.9.8 is image-to-video: it consumes a still, it does not invent one.** It
had never run on a Mac before -- not because of anything in the engine, but
because every LOCAL image engine on the platform was `["cuda"]`, so nothing could
mint the still it needed. Its first Mac attempt failed at the image step, not the
video step. `sd15` is what unblocked it.

So **selecting LTX without also selecting a working image model gets you a
failed render**, and the error names the image engine rather than LTX.

### What it costs on 16 GB

| configuration | wall clock | notes |
| --- | --- | --- |
| one lane (`still_motion`) | 22:22 | comfortable |
| one lane LTX + 2 visualizers | 39:17 | comfortable |
| **all three lanes on LTX** | **1:07:27** | swapped to ~14 GB, 27% free, zero errors -- the ceiling, not a comfortable setting |

LTXV loads fully on Metal: 3.67 GB + 9.08 GB (t5xxl) + 2.38 GB, every one
`full load: True`. No OOM, no unimplemented operator.

### Why the registry said this was impossible

`ltx_8gb` declared `["cuda"]`. Its adapter contains **zero** NVIDIA-specific code
-- no nvenc, nvml, triton, flash_attn, `torch.cuda` or `.cuda()` -- and pins its
T5 encoder to CPU by design. The row was untested policy, not a measurement. See
the 30-second grep test in `ADDING_IMAGE_AND_VIDEO_LANES.md`.


## 8. `animatediff15_v3_*` -- needs a THIRD-PARTY NODE PACK, not just weights

**Tested 2026-09-08, failed at the gate (not at render), and the cause is not
Apple Silicon.** It would fail identically on Windows without the same pieces.

```
EngineUnusable: video engine 'animatediff15_v3_haunted_video' is not usable for
role 'text_to_video': missing_model -- artifact(s) not found:
  motion_module=v3_sd15_mm.ckpt (folder_paths category 'animatediff_models'),
  domain_adapter=v3_sd15_adapter.ckpt
```

Three things are required and none auto-fetches:

| requirement | where it goes | note |
| --- | --- | --- |
| **`ComfyUI-AnimateDiff-Evolved`** (custom node pack) | `custom_nodes/` | **This is the real dependency.** It registers the `animatediff_models` folder category and provides the nodes. Without it the motion module is invisible even when the file is on disk |
| `v3_sd15_mm.ckpt` (1.56 GB) | `models/animatediff_models/` | `guoyww/animatediff` |
| `v3_sd15_adapter.ckpt` (~95 MB+) | `models/loras/` | the v3 DOMAIN ADAPTER -- a LoRA on the image model, not the motion module |
| `v1-5-pruned-emaonly-fp16.safetensors` (1.99 GB) | `models/checkpoints/` | shared with `sd15`; `Comfy-Org/stable-diffusion-v1-5-archive` |

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('guoyww/animatediff','v3_sd15_mm.ckpt'))
print(hf_hub_download('guoyww/animatediff','v3_sd15_adapter.ckpt'))"
```

**Mac status: UNKNOWN, not FAILED.** The engine never ran, so nothing was learned
about whether its code works on Metal. Its adapters contain zero NVIDIA-specific
code, so the `["cuda"]` row is as likely to be untested policy as
`ltx_8gb`'s was -- but that is a hypothesis, not a result.

**Recommendation: use `ltx_8gb` instead on a Mac.** It is proven, both its
weights auto-fetch, and it needs no third-party node pack. AnimateDiff is worth
revisiting only if you specifically want its look.

---

## 9. The `still_*` lanes: ALL FOUR render on Apple Silicon

**Proven 2026-09-08.** Section 3's "a cloud image key is the single
highest-leverage addition for a Mac, because it unlocks the `still_*` video
lanes that are otherwise unreachable" was written before `sd15` existed. It is
now **out of date**: a local image engine unlocks them too, and for free.

There are **four** `still_*` lanes and there are only three video dropdowns, so
no single run can cover them. It took two, and both published:

| lane | receipt in `otr/obs/` | still supplier |
| --- | --- | --- |
| `still_motion` | `..._20260908_005546__arch__`**`stmo`**`__...` | `sd15` |
| `still_pan`, `still_flat`, `still_word` | `..._20260908_042821__cart__`**`stwo`**`__...` | `sd15` |

(The engine shortcode in an episode filename is the DELIVERED video engine --
`stmo`/`stpa`/`stfl`/`stwo` for the four still lanes, `lx8g` for `ltx_8gb`,
`vcam` for `viz_camera`; the table is `nodes/_otr_shared/shortcodes.py:120-130`.
That slug is the cheapest possible proof of what actually rendered, and it is
worth reading before you believe any claim in this file, including mine.)

The three-lane run, zero errors:

| lane | beats it drew | still supplier |
| --- | --- | --- |
| `still_pan` | `music_opening_001`, `music_closing_001` | `sd15` |
| `still_flat` | `b001`, `b006` | `sd15` |
| `still_word` | `b002`-`b005` | `sd15` |

Set all three still-lane dropdowns plus the image engine and press Run:

```
OTR_VideoDirector  announcer_video_model  ->  still_flat (16:9)
                   music_video_model      ->  still_pan (16:9)
                   character_video_model  ->  still_word (16:9)
                   announcer_image_model  ->  sd15
                   music_image_model      ->  sd15
                   character_image_model  ->  sd15
```

Those are `widgets_values` 0-5 in that order if you are editing the saved JSON
rather than clicking the node. The three video dropdowns come first, then the
three image dropdowns.

The log line to look for, once per beat, is the proof the still actually
reached the lane:

```
[OTR.render_driver] still_flat: beat b001 conditioning on scene still
  still_b001_edf20ec1b43a.png (landscape; portrait never used)
```

If you instead see the LOUD warning `still_flat MISSING-STILL ... beat b001 has
NO scene still in the ledger`, the image phase produced nothing and the lane
will FAIL rather than paint a dark floor. That is deliberate (there are no
fallbacks here). Fix the image engine, not the video lane. In the run above the
warnings appear first and the success lines second, because the ledger is
consulted before the image phase fills it -- warnings in the first half of the
log are not a failure on their own.

**Cost (measured, wall clock, M4/16 GB):**

| run | video lanes | wall clock | episode |
| --- | --- | --- | --- |
| stills only | `still_pan` + `still_flat` + `still_word` | **22:38** | 18 MB |
| stills only | `still_motion` + 2 visualizers | **22:22** | 79 MB |
| one LTX lane | `ltx098_low_video` + 2 visualizers | **39:17** | -- |
| all-LTX | three `ltx098_low_video` roles | **1:07:27** | 81 MB |

Stills are by far the cheapest way to get a watchable episode out of a Mac: a
three-still-lane show costs about a third of the all-LTX one. (An earlier draft
of this paragraph said "about 55 minutes" for LTX. No run took 55 minutes -- the
number was interpolated rather than measured, and the table above is what the
logs actually say.)

### `word_razzle` is a CLOUD lane -- do not chase it as a Metal bug

`word_razzle` sits next to `still_word` in the dropdown and sounds local. It is
not. It is the Pixverse `cloud_pixverse_i2v` row, and it fails like this with no
credential:

```
RenderError: shot shot_music_opening_001 engine 'word_razzle' failed to render;
fallbacks are disabled (FailureKind.CRASH_BEFORE_LOAD) -- fix the engine or its
inputs: cloud media: auth -- no credentials: set OTR_COMFY_API_KEY, or run with
a logged-in Comfy account
```

Its registry row `["cuda", "cpu", "mps"]` with `practical_without_gpu: True` and
an empty `model_requirements` is **correct, not a bug**: the render happens
provider-side, so your local device genuinely does not matter.

Testing it measures a credential, not Apple Silicon -- and note the credentials
are not all the same one:

| video lane | credential it actually measures |
| --- | --- |
| `word_razzle`, every `cloud_*` row | Comfy Cloud (`OTR_COMFY_API_KEY`, or a logged-in Comfy account) |
| `google_omni_video` (Gemini Omni Flash), `google_veo_video` (Veo 3.1) | a **direct BYO Google API key** -- these do not touch Comfy Cloud at all |

(`ideo` is an Ideogram lane in the **image** registry, not a video lane; it
belongs to the same "credentials, not a port" bucket but does not appear in
these dropdowns.)

---

## 10. The GGUF lanes: the install is solved, and it is HIGH FRICTION

**Operator's framing, and it is the right one:** GGUF may well work, the install
method is now known, and that install is high friction. This section is for
someone with decent coding skills, or an AI coder sitting beside them. If that
is not you, stop here -- `sd15` + the `still_*` lanes and `ltx_8gb` need none of
this.

**Be precise about what is proven here: the INSTALL, not a render.** As of
2026-09-08 the ComfyUI-GGUF pack is verified registered on this machine (six
loader classes, boot clean, one added wheel) and **no GGUF lane has completed a
render on Apple Silicon in this repository.** Do not read this section as a
qualification of any lane.

**What GGUF buys you on a Mac.** Quantised weights are how the bigger lanes fit
in unified memory at all. `flux2_klein` is 2.60 GB as a Q4 GGUF against 7.75 GB
bf16; `wan_ti2v`'s shipped set is 9.37 GB GGUF against 21.2 GB in fp16, and the
fp16 route is the one that took this machine down. So on this platform GGUF is
not an optimisation, it is frequently the only version that can run.

**What it costs you.** Everything below was hit in one sitting on 2026-09-08, in
this order, on a machine that already had ComfyUI working.

### 10.0 K_M quants do NOT garble on Metal -- SETTLED 2026-09-08

**The concern was real and it is now answered.** Commit `5f1b94b4` passed over
`flux2_klein` for Apple Silicon partly because *"K_M quants garble on MPS"*, and
city96's ComfyUI-GGUF issue #177 reports GREEN OUTPUT from `Q*_K` Flux weights
on Metal. A web sweep judged #177 to predate a PyTorch MPS integer-operation fix
and to be contradicted by later Flux-family GGUF runs.

**Measured here, and the sweep was right.** `flux-2-klein-4b-Q4_K_M.gguf`
through `UnetLoaderGGUF` minted a clean 1472x832 still on the first attempt --
coherent subject, correct style adherence, no green cast, no smearing, no noise.
Receipt: `otr/episodes/signal_lost_the_dark_sea_beyond_the_glass_20260908_095959/
stills/still_music_opening_001_d94b2a43c4e6.png`, 20 steps, guidance 4.0.

So a K_M quant is not a reason to avoid a lane on this platform. **The habit is
still worth keeping**: a garbled render exits ZERO, so open the first still or
clip and look at it before concluding a lane works. That is how this was
settled, and it is the only way it could have been.

`wan_ti2v`'s shipped set is also K_M (`Wan2.2-TI2V-5B-Q5_K_M.gguf` plus a
`umt5-xxl-encoder-Q5_K_M.gguf`), so this result removes one of the two objections
to it. The other one -- the open Wan temporal-corruption defect on this exact
macOS/torch generation, section 11 -- still stands, and it is the disqualifying
one.

### 10.1 The repo ships the installer -- point it at the right tree

```bash
OTR_COMFY_ROOT=/path/to/ComfyUI \
  <ComfyUI Python> scripts/otr_provision.py --packs-only
```

`scripts/otr_provision.py` clones and PINS the three packs (ComfyUI-GGUF at a
fixed commit plus an in-repo LTX 2.5 patch, ComfyUI-LTXVideo, and
ComfyUI-AnimateDiff-Evolved) and installs their requirements. You do not need to
find them yourself, and you should not: the pins matter.

**Two traps in that one command.**

* **`OTR_COMFY_ROOT` is not optional in practice.** Without it the script
  guessed `/Users/<me>/Documents` here and created a `custom_nodes/` folder
  there, cloning two packs into a directory ComfyUI has never heard of. Nothing
  warned; the receipt said `PATCHED`. Check the `comfy root :` line it prints
  BEFORE walking away, and delete any stray `custom_nodes/` it made elsewhere.
* **`--list` does not dry-run for packs.** The flag is documented as "show what
  would be installed, install nothing". Combined with `--packs-only` it clones
  and installs anyway. Treat `--packs-only` as always live.

### 10.2 `git-lfs` -- the step that stops a clean Mac dead

```
FAILED  required pack/dependency -- git checkout -q --detach FETCH_HEAD failed
in .../custom_nodes/ComfyUI-LTXVideo: git-lfs filter-process: git-lfs: command
not found
fatal: the remote end hung up unexpectedly
```

macOS ships neither `git-lfs` nor Homebrew. `GIT_LFS_SKIP_SMUDGE=1` does **not**
rescue it -- the checkout still fails. Install git-lfs first
(`brew install git-lfs && git lfs install`, which means installing Homebrew
first).

**There is no skip flag, and this matters if you rerun.** `install_node_packs`
requires ComfyUI-LTXVideo unconditionally, so the provisioner will report
INCOMPLETE every time until git-lfs exists -- and moving the broken clone aside
does not settle it, because the next run re-clones and fails again the same way.
What you get by moving it aside is a ComfyUI that boots cleanly in the meantime,
not a finished provision. The GGUF pack DOES land before that failure, which is
why Klein and the Wan lanes are reachable without it.

**And clean up after the failure, because it does not.** A failed checkout
leaves a directory full of files with NO COMMITS -- `git log` says *"your current
branch 'main' does not have any commits yet"* and every file is untracked. That
is an unpinned pack sitting in `custom_nodes/`, which ComfyUI will import at
boot, next to lanes that currently work without it. Move it aside:

```bash
mv custom_nodes/ComfyUI-LTXVideo custom_nodes/.disabled/ComfyUI-LTXVideo-unpinned
```

`ltx_8gb` does NOT need this pack -- it drives stock ComfyUI nodes and has
published episodes here without it. Only the three `ltx25_*` lanes need it, and
they cannot run on 16 GB anyway.

### 10.3 Verify the pack actually registered

A cloned pack that failed to install its wheel registers NOTHING, and the
failure arrives much later as `WrapperNodeMissing` in the middle of a render.
Check at the API instead of at the filesystem:

```bash
curl -s http://127.0.0.1:8188/object_info | python3 -c "
import json,sys; d=json.load(sys.stdin)
print([k for k in d if 'GGUF' in k])"
```

Six classes is right: `UnetLoaderGGUF`, `CLIPLoaderGGUF`, `DualCLIPLoaderGGUF`,
`TripleCLIPLoaderGGUF`, `QuadrupleCLIPLoaderGGUF`, `UnetLoaderGGUFAdvanced`.
An empty list means the `gguf` wheel is missing even though the folder is there.

### 10.4 Watch the venv, because this is how a boot gets bricked

Installing a pack's requirements runs `pip` into the SAME environment ComfyUI
boots from. That is exactly how the `tokenizers` pin bricked this install on
2026-09-07 (section 4). Snapshot before, diff after:

```bash
<ComfyUI Python> -m pip freeze > /tmp/venv_before.txt
# ... run the provisioner ...
<ComfyUI Python> -m pip freeze > /tmp/venv_after.txt
diff /tmp/venv_before.txt /tmp/venv_after.txt
```

A good result is one added line. The ComfyUI-GGUF install here added exactly
`gguf==0.19.0` and changed nothing else. **If that diff shows an existing
package being upgraded or downgraded, stop and read it before restarting
ComfyUI** -- a downgrade of `tokenizers`, `numpy`, `transformers` or `torch` is
the shape of a bricked boot, and you cannot repair it from the Manager UI
because the Manager is itself a ComfyUI extension.

### 10.5 An `OTR_*` knob goes on the SERVER, not on your shell

Every `OTR_*` value an engine reads is read **inside the ComfyUI process**.
Exporting it in the terminal you run an API client from does nothing at all:

```bash
# WRONG -- the client has the value, the renderer does not
OTR_WAN_TI2V_UNET_NAME=... python scripts/otr_canonical_api_run.py ...

# RIGHT -- restart ComfyUI with it
OTR_WAN_TI2V_UNET_NAME=... python ComfyUI/main.py ...
```

That covers the knobs that actually bind: the loader-name and path overrides
(`OTR_WAN_TI2V_UNET_NAME`, `OTR_LTX_8GB_CKPT`, `OTR_SD15_MAX_SIDE`),
`OTR_UNIFIED_MEMORY_HEADROOM_MB`, and `OTR_MPS_PYTORCH_ATTENTION`. If you
changed one and nothing changed, you almost certainly set it on the wrong
process.

**A trap worth naming, because this section originally fell into it.** Some
adapters still carry an `ENABLE_FLAG` constant -- `flux2_klein.ENABLE_FLAG =
"OTR_ENABLE_FLUX2_KLEIN"`, and `eng_wan_ti2v`'s docstring calls
`OTR_ENABLE_WAN_TI2V` a "vestigial opt-in flag". **They gate nothing.** Both
adapters set `requires_flag = None`, `EngineUsabilityReason.GATED_BY_FLAG` is
documented dead, and `tests/test_registry_is_the_menu_guard.py` asserts that no
registered engine carries a live flag. Setting either variable has no effect
whatsoever -- if an engine refuses, the reason is in its `assert_usable`
message, not a missing opt-in. An earlier draft of this section told you to
restart the server to set one, which would have sent you looking in the wrong
place entirely.

---

## 10.6 The writer, not the video lane, is what kills a 16 GB Mac

**This is the most important thing on this page, and it was found by asking the
right question rather than by testing harder.**

Three renders took the whole machine down on 2026-09-08. Two of them I blamed on
the video lane and spent hours shortening clips. The operator's observation
ended that:

> The LTX lane carries **16.1 GB** of weights and published episodes here. The
> AnimateDiff lane carries **2.4 GB** and cannot finish. So the video lane is
> not the variable.

It was not. The writer is common to both, and the heavier lane survived only
because its episode made fewer writer cycles.

### What was actually happening

`Qwen/Qwen3.5-4B` is ~8.7 GB, and one episode loaded it **four times**. The
reason is an ordering bug that is wrong on every platform:
`_otr_writer_vram.unload_writer_llm_after_script`'s own docstring says "evict the
writer LLM after the LAST LLM phase", but the call sat at
`_otr_writer_tail.py:1102`, with three more LLM phases below it -- the brief
reflection, ledger clean/cleanup, and the cast-coverage repair. Each was
appended to the tail *after* the unload was placed, and each got a fresh
`from_pretrained` plus warmup in response. On a discrete card that is 25-40
wasted seconds a time. Here it was fatal, for a reason that took three
independent reviews to pin down:

**`unload_llm` clears the cache but not every reference.** The Slot Drama
Contract's `_sdc_cache` and `_sdc_gen_fn` (`OTR_LedgerScriptWriter.py:4448`) are
locals of the still-live `run()` frame, and the generate closure captures the
cache entry *and* the model directly (`_otr_constrained_generate.py:234-242`).
Those aliases keep the old model alive. `model.to("cpu")` then turns it into an
~8.7 GB CPU-resident copy that cannot be reaped -- and on unified memory "cpu"
is the same physical RAM -- so the reload builds a second copy beside the first.
**Two writers, ~17 GB, on a 16 GB machine.** The kill lands mid-`from_pretrained`
with no traceback.

### Two fixes, and the honest order of importance

1. **The unload moved to the real boundary** (`_otr_writer_tail.py`, after the
   cast-coverage repair). Four loads become two, and no reload happens while
   those aliases are live. **This is the fix.**
2. **`torch.mps.empty_cache()` in the teardown** (`_otr_model_loader.py`). The
   teardown implemented its documented six-step sequence only for CUDA; steps
   4-6 had no Metal counterpart, so `.to("cpu")` left the Metal pool held until
   the next load washed it. Useful -- it closes a 2x window across the gap where
   the video models load -- but it cannot free a referenced CPU copy, so on its
   own it does not stop the kill.

### What to check if this comes back

```bash
grep -c "Loading LLM model" <server log>     # 2 per episode, not 4
grep -c "proceeding with caution" <log>      # informational, see below
```

**`vram_fit=WARN@4.3 GB` is lying to you by 2x.** `_estimate_resident_gb`
divides every non-GGUF model's disk size by two
(`_otr_model_catalog.py:1828, 1898`), i.e. it assumes NF4 regardless of
`quant_policy`. The Mac profile runs `quant_policy: "none"`, so the true
residency is >= 8.68 GB plus KV -- 87% of the 10 GB ceiling, not 43%. The gate
only refuses at 1.5x the ceiling, so it admits either way; but do not read that
number as headroom.

### The general lesson, which cost the most time

Every kill looked like a video-lane problem because a video lane was on screen
when it happened. The thing to measure on this platform is not the lane you
selected -- it is **how many times the biggest model in the pipeline is loaded**,
and whether anything still holds a reference when it is.

---

## 11. What the wider world reports, and why we stopped testing by trying

**Method note, and it is the point of this section.** Everything above section 10
was learned by running things. That stopped being acceptable when a `wan_ti2v`
attempt took the whole machine down (section 1) -- on unified memory a bad guess
costs a hard reboot, not a stack trace. So the remaining candidates were
researched instead: a web sweep for REPORTED experience, with sources, dates,
and a hard distinction between exact-16 GB reports and larger-Mac ones, since
32 GB and 64 GB results do not transfer.

Researched 2026-09-08. **Treat every row as evidence about somebody else's
machine unless this repo carries a receipt for it.**

| candidate | evidence | verdict for a 16 GB M4 |
| --- | --- | --- |
| **LTX-Video 0.9.8 distilled** | strong -- plus our own published episodes | **KEEP.** Proven here; nothing further needed unless torch/ComfyUI move |
| **FLUX.2 Klein 4B Q6** | strong -- an exact 16 GB completion exists | **Viable.** Preserve the known-good model/encoder combination |
| **FLUX.2 Klein 4B Q4_K_M** | moderate -- no exact 16 GB report found | Plausible, and the one to try first on a Mac. Note `scripts/otr_provision.py` does NOT fetch it -- `flux2_klein` sits in `MANUAL_TIERS`, so its three files are a manual download. See the K_M warning below |
| **AnimateDiff-Evolved + SD1.5 v3** | moderate on Metal (2023 completions); STRONG on cost | **The cheapest lane in the pack, and the best untested candidate.** Measured 4.9 GB with SD1.5 FULLY RESIDENT and no offload on the 4060 -- it fits 16 GB unified with enormous headroom. It is `text_to_video`, so it needs NO scene still and sidesteps the image-engine question entirely. Use the PINNED commit, not `main` (see below) |
| CogVideoX-2B | anecdote -- one Mac walkthrough, no hardware named | Unproven at this size |
| Wan 2.1 Fun InP 1.3B | anecdote -- an exact M4/16 GB completion EXISTS | **Do not follow it as written:** it required `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, which REMOVES the MPS allocation ceiling. On a host where OOM kills the machine, that setting is the opposite of a mitigation |
| LTX 2.x (`ltx25_*`) | strong NEGATIVE | **Avoid.** Open MPS BF16 attention NaNs produce all-black video |
| **Wan 2.2 TI2V-5B** | strong NEGATIVE | **Dead here**, and not only on memory -- see below |
| Stable Video Diffusion, Mochi, HunyuanVideo, FramePack | none found | No qualifying 16 GB Apple Silicon completion reports at all |

### The three findings that change what you should do

**1. Wan is not merely too big -- it renders WRONG.** An open ComfyUI issue dated
2026-08-21 reproduces Wan 2.1/2.2 TEMPORAL CORRUPTION on an M4 Pro running
macOS 26.6 with torch 2.12.1 -- the same software generation as this host -- and
GGUF Q8 and fp16 corrupt IDENTICALLY. That rules out quantisation as the cause
and makes it an MPS kernel defect. So the 9.37 GB GGUF set fitting is beside the
point: it would fit and still produce garbage. `wan_ti2v` and `fastwan_8gb` are
both off the table on this platform until that lands a fix -- and note this is a
DIFFERENT reason from the memory one that killed the machine here. Either alone
is disqualifying.

**2. LTX 2.5's black-video defect has a published workaround, and it is worth
knowing even though we cannot run the lane.** ComfyUI issues from 2026-08-22 and
-23 report frequent all-black LTX 2.5 output from MPS BF16 attention NaNs, and
`--use-split-cross-attention` reliably avoiding it across 49, 121 and 241
frames. Note the tension with this pack's own prestartup, which FORCES PyTorch
attention on MPS to fix the sub-quadratic `baddbmm` bug (section 4). Both are
real; they want different attention backends. If you ever get an LTX 2.x lane
running here, that is the first knob to try -- and `OTR_MPS_PYTORCH_ATTENTION=0`
is how you get our forcing out of the way.

**3. AnimateDiff: the PIN is why the haunted lane is fine, and it is a good
advert for pinning.** AnimateDiff-Evolved issue #576 (opened 2026-06-18, still
open) reports COLORED NOISE from both `mm_sd_v15_v2` and `v3_sd15_mm` across
Gen1 and Gen2 loaders, multiple schedules and FP32 -- on an **RTX 4070**, so it
is a core/pack version incompatibility rather than anything to do with Metal.

**It does not touch this pack's lane, and the dates say so.** `otr_provision.py`
pins `ANIMATEDIFF_PIN = 92576512...` rather than tracking `main`, and the 4060
published SIX episodes on the haunted lane on 2026-08-29 -- more than two months
AFTER #576 was filed (`config/machine_classes.json` carries that receipt). A
pinned commit that keeps working while `main` breaks is the pin doing its job.

The practical reading for a Mac reader is therefore narrow: **use the
provisioner, not a hand-rolled `git clone` of `main`** -- a hand-cloned latest is
exactly the configuration #576 describes. Beyond that the lane is simply
untested on Metal, like everything else in the NEVER TESTED bucket.

### What "no reports found" means here

It means nobody has published a qualifying result, not that the thing fails.
Every "none found" row above is an invitation, not a verdict -- and the cheap
way to change one is to run it and record the peak, not to reason about it.

---

## 10.7 Why the AnimateDiff clip is 125 latents and cannot be capped

**Because three different frame counts look alike in a log, and only one of them
is the clip.** This cost three wrong attempts before a review panel settled it,
so the arithmetic is written out rather than summarised.

```
[AnimateDiffEvo] Sliding context window sampling activated --
latents passed in (125) greater than context_length 16.
```

125 is **not** the clip length.

| quantity | value on a 10-second beat | where it comes from |
| --- | --- | --- |
| `target_frame_count` -- DELIVERED frames | 250 | ShotLock, from the beat's AUDIO budget |
| `source_request` -- SAMPLED frames | 125 | `ceil(250 / hold)`, hold = 2 |
| ADE sliding windows | ~8 | `source_request / 16` |

`ghost_unique_source_count = ceil(T / hold)` (`eng_ghost_signal.py:323`), floored
at 16 by `ghost_source_request` (`:341`), lands as `EmptyLatentImage`'s
`batch_size` (`:1016`) and feeds the sampler (`:1028`). At delivery each decoded
frame is held twice (`:354-360`), restoring all 250. **The picture is
audio-locked; the model is only asked for half of it.**

### The clip length is not yours to set, and that is deliberate

`video.max_render_frames` is applied to the widget -- verified by dumping the
submitted graph -- and does nothing here. The lane declares:

```python
frame_contract = FrameContract(min_frames=1, max_frames=0, quantum=1, ...)
#: A beat is ONE timeline even when it spans several internal context windows
#: -- the 16 is a context window, never an OTR clip duration -- so there is no
#: ceiling to split on and continuity is explicitly NONE.
```
(`eng_ghost_signal.py:523-533`)

`PLANNING_CAP_ENGINES` is `("ltx_8gb", "fastwan_8gb", "wan_ti2v")`
(`frame_contract.py:319`), and `effective_frame_contract` returns the contract
UNCHANGED for anything else.

**Adding this lane to that tuple would make it WORSE, not shorter.** With
`continuity=NONE` the planner joins segments with `join_mode="jump"`
(`coverage_plan.py:195-197`), so a 17-frame cap would split a 250-frame beat into
about fifteen segments, each still flooring at the 16-frame context window --
roughly 240 sampled latents instead of 125, and fifteen jump cuts where there was
one continuous beat.

(That fifteen is specific to a SEVENTEEN-frame cap, not to capping in general --
a cap near 248 would split the same beat only once. The conclusion survives the
correction: on a `continuity=NONE` lane any cap introduces a jump and extra
floor/alignment work, so the lane stays out of `PLANNING_CAP_ENGINES`. But the
number was doing more argumentative work than it had earned.)

Capping at the beat level is out too: `validate_coverage_plan` refuses any plan
whose visible frames differ from the audio-derived target
(`coverage_plan.py:478-482`). Shorter picture means picture that no longer
matches its sound.

### MEASURED 2026-09-09: AnimateDiff-Lightning is 3.83x on this M4

The first numbers for `animatediff15_lightning_video`, taken with ISOLATED
ComfyUI graphs -- checkpoint -> `ADE_AnimateDiffLoaderGen1` -> KSampler ->
VAEDecode, no OTR pipeline, no writer, no TTS -- so each number is the lane's
sampler cost and nothing else's. Same machine, same canvas, same frame count,
back to back:

| arm (512x288, 16 frames, one window) | wall clock |
|---|---|
| golden v3, 20 steps, cfg 8.0, `autoselect` | **230.3 s** |
| **Lightning 8-step, 8 steps, cfg 1.0, `sgm_uniform` / `sqrt_linear`** | **60.1 s** |

(The 55.1 s quoted further down is the SAME recipe on a DIFFERENT prompt --
the safari scene rather than `recur_frac`. Two prompts, not two measurements
of one arm. Cold checkpoint load is in both.)
| Lightning 8-step, 8 steps, **cfg 2.0** (negative LIVE) | 100.1 s |

**3.83x, not the 5x the pass count predicts.** The gap is real and was called in
review before the run: VAE decode and checkpoint load do not shrink with the
step count, so they become a larger fraction of a shorter render. Treat pass-count
arithmetic as a floor, never as the estimate.

### cfg 1.0 WINS on this lane, on both axes -- and the lettering fear did not land

This was the lane's one genuinely open question, because cfg 1.0 does not weaken
the negative prompt, it DELETES the pass (`comfy/samplers.py:610`), and this repo
had already refused AnimateLCM for exactly that. The test was built to be the
worst case for it: a bar counter with rows of bottles, which is precisely what
SD1.5 volunteers labels and signage onto.

* **No lettering at cfg 1.0.** None on the bottles, none anywhere.
* **cfg 2.0 looked WORSE**, not better: magenta/green chromatic fringing, halos
  around the figures, oversaturation. Lightning is distilled FOR cfg 1.0, so
  raising guidance fights the distillation instead of helping it. It also cost
  1.8x the time (100.1 s against 55.1 s).

So the AnimateLCM precedent does NOT transfer, and `OTR_LIGHTNING_CFG` stays a
sweep knob rather than a fix waiting to be applied.

**What this evidence is NOT.** Two images, one seed each, 16 frames at 512x288.
Not a sweep. If lettering ever appears it will be on a beat whose prompt names a
sign or a dial, and the env knob is already there.

### The DECODER A/B: `vae-ft-mse-840000` is better, and is NOT in the lane

Run at the operator's request after he judged the cfg-1.0 render "most
realistic". Identical prompt, seed 42, cfg 1.0, 8 steps, 512x288, 16 frames --
**only the decoder changed**:

| arm | wall clock | picture |
|---|---|---|
| A, the SD1.5 checkpoint's baked VAE | 55.1 s (cold) | more milky haze |
| **B, `vae-ft-mse-840000-ema-pruned`** | 28.8 s (WARM) | cleaner glass on the bottles, better foliage separation |

**B's 28.8 s is a WARM-CACHE ARTIFACT, not the VAE being faster.** The
checkpoint was already resident from arm A. Decode cost is essentially
identical; treat ~55 s as the cold figure for both. The win here is quality
only.

Pin data, verified against the file on disk and the Hub API rather than taken
from an agent's summary:

```
stabilityai/sd-vae-ft-mse-original
  revision 629b3ad3030ce36e15e70c5db7d91df0d60c627f
  vae-ft-mse-840000-ema-pruned.safetensors
  334,641,190 bytes
  sha256 735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75
  licence MIT
```

MIT is worth noting: it is **more permissive than anything else this lane
loads** -- the SD1.5 checkpoint and the Lightning module are both CreativeML
Open RAIL-M. Adopting it would not weaken the lane's licence position.

**SUPERSEDED THE SAME DAY -- THE LANE NOW DECODES WITH ft-mse.** This section
first said the decoder was deliberately NOT wired, because three judges were
asked and the two that answered agreed on sequencing (adopt only AFTER a clip
lands in `otr/obs/`), even though they used opposite verdict words. The operator
then overrode that -- *"you may as well wire it in, if it fails it fails -- label
it as an exp lane"* -- which was the documented flip condition one judge had
named. So as of `c3212e1c` the lane declares
`vae_name = VAE_FT_MSE_NAME` and `prepare` rebinds `prepared["vae"]` from its own
`VAELoader`; only lanes that declare no `vae_name` still take `ckpt_out[2]`, which
is both published siblings. The sequencing argument is kept here because it is
still the right reasoning The reasoning is the same in both: tonight's proof was hand-built graphs
that never touched `prepare` / `render_clip` / `canonicalize` / the cadence
receipts, and adding a third artifact before that first real leg gives a failure
two suspects instead of one. A decoder swap is also the one change that can be
A/B'd after the fact, because it re-decodes the same latents -- unlike
`align_source_to_context_window`, which altered what was sampled and therefore
had to land first.

So this is recorded as a MEASURED, PINNED, READY option and deliberately not
wired. When the adapter-path proof lands, the seam is a `vae_name` /
`vae_min_bytes` pair on the parent defaulting to `None` (mirroring `lora_name`
cell for cell so the two published siblings stay byte-identical), the only graph
change being a second one-node `VAELoader` in `prepare` so `prepared["vae"]`
comes from it instead of `ckpt_out[2]`. The recipe receipt id must REPOINT
(`..._ftmse_...`) rather than be edited in place, because a proof clip will
already exist under the current id.

### CLOSED 2026-09-09: the lane published an episode, and the row moved

The section below is kept as written, because the distinction it draws is the
reason this took three commits instead of one -- and then it was satisfied.

**`device_backends` is now `["mps", "cuda"]`.** A COMPLETE episode rendered
through the real OTR adapter path on the M4/16 GB:

| | |
|---|---|
| beats | 23 |
| delivered frames | 2,736 |
| wall clock | 02:32:34 |
| output | 157 MB, 2:16.24, 1920x1080 h264 + AAC stereo |
| published | `otr/obs/lightning_mac_proof_2_20260909_100958__arch__adlt__none__koko__news__q354b__sa3_final.mp4` |

Two tokens in that filename are the load-bearing ones. **`adlt`** is this lane's
shortcode, so the file names the engine that made it. **`none`** is the IMAGE
slot -- proof that `accepts_still = False` held and no image engine was invoked
for any of the three video roles.

No OOM. The machine stayed up (3h28m at the end of the run), which is the
difference between this attempt and the one before it.

WHAT IS STILL NOT PROVEN, and it is worth naming rather than letting the green
tick cover it: every beat in this episode fell UNDER the 136-latent ceiling, so
**the adaptive-hold guard never fired.** That path is unit-tested and has never
run under live fire. The episode that forced it into existence had beats of
239-267 delivered frames; this one had 91-250, because the writer produced a
different script. A deliberately long beat is still owed.

#### What the earlier numbers did NOT prove (kept for the distinction)

`device_backends` was `["cuda"]` through three commits and that was correct. The
isolated runs proved the RECIPE executes on Metal. They did not prove the LANE
does -- they were hand-built graphs submitted to `/prompt`, not the OTR adapter
path, so nothing exercised `prepare`/`render_clip`, the cadence receipts,
`canonicalize`'s exact-canvas refusal, or the delivered-frame contract. Those are
different claims and only the second earns the row.

### A STALE INSTALLED PACK MAKES TESTS LIE, and the failures look real

`custom_nodes/comfyui-old-time-radio` is a SEPARATE COPY of this repo, not a
symlink to it. ComfyUI needs it there; the tests import from the repo. When the
two drift, a FULL-SUITE run can resolve `nodes.*` to the INSTALLED copy while a
single-file run resolves it to the repo -- so tests pass alone and fail together.

It bit on 2026-09-09 and cost real time, because the failures are perfectly
plausible: four tests that assert on `inspect.getsource` reported that code
present in the repo was missing. It was missing -- from the stale copy they had
actually imported.

**Symptom:** a test passes when you run its file and fails in the full suite,
and the assertion is about source text or a newly added attribute.

**Check:**

```bash
diff -rq --exclude=__pycache__ \
  ~/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes \
  <repo>/nodes
```

**Fix:** copy the changed files over and drop `__pycache__`, then re-run. Do this
after EVERY edit you intend to render with, or the render exercises different
code from the one the tests just proved.

### The Mac setup that made it run at all

Two things, both of which cost time to find:

1. **`animatediff_models` is not in Comfy Desktop's generated mapping.** It is a
   category AnimateDiff-Evolved registers itself
   (`utils_model.py:343-344`), pointing at BOTH its own pack directory and
   `<comfy>/models/animatediff_models`. A motion module in the shared models
   root is therefore invisible. Either hardlink the module into
   `<comfy>/models/animatediff_models/` (no extra disk, same volume), or pass
   `config/otr_mac_extra_model_paths.yaml` ALONGSIDE the Desktop file.
2. **A bare `python main.py` has NO checkpoints.** The Desktop app's mapping
   lives at `~/Library/Application Support/Comfy Desktop/instance-model-paths/inst-*.yaml`
   and must be passed explicitly, or `ckpt_name` validates against an empty
   list and every prompt is rejected:

```bash
python main.py --extra-model-paths-config \
  "$HOME/Library/Application Support/Comfy Desktop/instance-model-paths/inst-<id>.yaml"
```

### The math the lane never did -- and 125 is not a legal count

Everything above is about the clip LENGTH, which is audio-derived and not yours
to set. There is a separate question the lane went years without asking, and the
operator raised it on 2026-09-08: *"like all other video lanes we need to do the
math -- maybe we pad to the min or somehow calculate the beat into a legal
segmentation."* Every other video lane resolves a beat into lengths its model
actually accepts. This family never did, because its constraint is a CONTEXT
WINDOW rather than a VRAM ceiling, and the window scheduler absorbs an illegal
count in silence.

**The rule**, read off `create_windows_static_standard` in the pinned
AnimateDiff-Evolved (not off its docs): windows start at 0, 12, 24 ... where the
stride is `context_length - context_overlap = 16 - 4 = 12`. When a window would
run past the end, the scheduler BACKS THE FINAL WINDOW UP so it still spans a
full 16. Legal counts are therefore:

```
N = 16 + 12k          # 16, 28, 40, 52 ... 124, 136 ...
```

which is exactly the shape `FrameContract.quantum` already describes -- *"legal
lengths are `min_frames + k * quantum`"* -- i.e. `min_frames=16, quantum=12`.

**125 is one past a legal count**, and 125 is what a 250-frame hold-2 beat asks
for. Its final window overlaps the previous by **15 of 16 frames** instead of 4:

| N | windows | overlaps |
|---|---|---|
| 124 | 10 | {4} |
| **125** | 11 | **{4, 15}** |
| 136 | 11 | {4} |
| **250** | 21 | **{4, 10}** |

So the tail re-denoises almost entirely covered ground and the pyramid fuse
weights it unevenly. **This is NOT a crash and never has been.** The back-up is
upstream's deliberate clamp -- it keeps every window full-length, which is what
the motion module wants -- and the operator reports never having seen it fail,
with the procgen background layer covering any small gap. It is simply not the
arithmetic the rest of the pack does.

**Rounding up is FREE, which is why it is worth doing at all.**
`sampling.py:843-844` invokes the model ONCE PER WINDOW on that window's slice,
so cost tracks the window COUNT, not the latent count. That count is
`ceil((n-16)/12) + 1`, and the rounded-up value has the identical count by
construction -- rounding up IS taking that ceiling. **125 and 136 are both
eleven windows.** The only real cost is decoding the surplus frames, which the
lane already discards and already reports as
`model_frame_count - cadence_source_frame_count`.

That surplus machinery is not new either: `ghost_source_request`'s `max(U, 16)`
is the same pad-up-discard-report idea applied to the FLOOR. Alignment applies
it to the STRIDE. The operator's "pad to the min" was already half-built.

**Where it is switched on.** `align_source_to_context_window` defaults to
**False** on `GhostSignalEngine`, so `animatediff15_v3_haunted_video` and the
still-in lab peer are untouched -- swept across all 1199 beat lengths, both
return source counts byte-identical to the pre-seam function. Only
`animatediff15_lightning_video` opts in, because it has never rendered and can
start correct rather than be corrected. **Turning it on for a lane with
published episodes changes its latent count and therefore its picture**, so that
is an operator decision backed by a 5080 comparison, not a driver one.

Verified rather than argued: swept `n = 1..599` against the REAL upstream
scheduler -- the window-count formula is exact in every case, rounding up never
adds a window in any case, and every aligned count tiles with a uniform overlap
of 4.

### The one real lever, and why it is not a speed knob

`OTR_GHOST_HOLD_FACTOR` (`eng_ghost_signal.py:119`, range 1-5, unset = unchanged)
is the only mechanism that reduces sampled latents without changing delivered
duration. Its own documentation gives the arithmetic: hold 2 generates **12.5
unique source positions per displayed second**, hold 3 gives **8.33**.

Three things to know before reaching for it:

* **It is a SHELL environment variable on the machine running the server.** A
  profile cannot carry it -- the `video` section accepts only `device_policy`,
  `dtype_policy` and optional `max_render_frames`
  (`capability_profiles.py:135-153`), and an unknown key raises. `launch.env`
  does not reach an already-booted server either.
* **It changes how the show LOOKS.** Fewer fresh positions per second is slower
  motion, not a free optimisation. Hold 2 is pinned as the golden contract by
  `test_the_golden_lane_still_declares_hold_2`.
* **Setting it on this machine breaks four tests** in
  `tests/test_ghost_signal_cadence.py`. That is the golden contract doing its
  job, not a bug.

### So what IS the Mac's cost here

About 132 s per sampler step, ~44 minutes for a 10-second beat, against the
4060's 3-3.6 minutes for the same lane. Nothing is misconfigured: an M4 samples
125 latents across eight sliding windows more slowly than an Ada card does.
Treat that as the measured price of this lane on this hardware, and pick a
cheaper lane if it matters -- the `still_*` family renders a whole episode in
about 22 minutes (section 9).
