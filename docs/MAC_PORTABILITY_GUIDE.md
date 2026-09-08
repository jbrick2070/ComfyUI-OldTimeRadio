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
* **NEVER TESTED, and nobody should pretend otherwise** -- `flux2_klein`
  (2.6 GB Q4 GGUF, and `config/machine_classes.json` already carries an operator
  ruling naming Klein for Mac), `lumina_image`, `flux_gen1`, `wan_ti2v`,
  `fastwan_8gb`, `humo_1.7B`, `humo_1.7B_169`, `mesh_stage`,
  `stable_audio_music`, and the whole upscale namespace. `flux2_klein` is the
  most interesting name on that list: it is the second-smallest local image
  engine in the pack and it has never been run on Metal.

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
