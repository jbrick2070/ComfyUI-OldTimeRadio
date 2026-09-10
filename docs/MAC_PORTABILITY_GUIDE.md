# Running ComfyUI-OldTimeRadio on a Mac (Apple Silicon)

**Status: WORKING.** The shipped canonical renders end to end on Apple Silicon
and publishes to `otr/obs/`. First proven 2026-09-07 on a Mac mini M4, 16 GB,
macOS 26.6.2, ComfyUI Desktop 0.34.6, Python 3.13, torch 2.12.1. Seven episodes
had published from that machine by 2026-09-09, covering local stills, local
video diffusion and a distilled AnimateDiff lane.

This is the practical guide: what works, what will reboot your machine, what to
install, and where to look next. It is written for the 16 GB machine it was
measured on; a 32 GB Mac has more headroom and nothing above has been measured
there.

| you want | read |
| --- | --- |
| a verdict for every one of the 61 engines (PROVEN / LIKELY / OOM RISK / WILL NOT RUN) | `MAC_COMPLIANCE_MATRIX.md` |
| the measurements behind the numbers here | `MAC_LESSONS_LEARNED.md` |
| the engine-by-engine survey and test plan | `MAC_ENGINE_TEST_PLAN.md` |
| what is still open from the Mac campaign | `MAC_PUNCH_LIST.md` |
| each defect with its evidence, by id (`PBUG-2026090x-nn`) | `PROD_BUG_LOG.md` |
| how to read a `device_backends` row, and the 30-second grep for an untested one | `ADDING_IMAGE_AND_VIDEO_LANES.md` |

**Three things to know before anything else.**

1. **An out-of-memory on a Mac reboots the MACHINE.** Not the render -- the
   machine, with no traceback. Section 2 is about nothing else.
2. **A `device_backends` row without `mps` means nobody has run it here.** It
   does not mean it cannot run. `ltx_8gb` was `["cuda"]` until the day it
   published two episodes on this machine; `bark` was `["cuda","cpu"]` until it
   ran on `mps`.
3. **Gated or manual weight downloads are friction, not failure.** They are
   recorded so you know what a first run costs, never as a mark against a lane.

---

## 0. What works, at a glance

Everything in this table has been run on the M4 / 16 GB. "PROVEN" means a
full episode published to `otr/obs/`; the bracketed token is the engine
shortcode in that episode's filename, which is the cheapest proof of what
actually rendered (`nodes/_otr_shared/shortcodes.py`).

| stage | engine | status on M4 / 16 GB | measured cost | section |
| --- | --- | --- | --- | --- |
| writer | `Qwen/Qwen3.5-4B` on `mps`, `llm_quant_policy` = `none` | PROVEN, every episode | ~6.5 tok/s; peaks near 14 GB | 1, 12 |
| voices | `kokoro` | PROVEN, every episode | kokoro-onnx on CPU under Python 3.13, by design | 1 |
| music | `stable_audio_3` | PROVEN, every episode | -- | 1 |
| voices / music, alternatives | `bark`, `musicgen` | PROVEN on `mps` (measured runs, not episodes) | see the matrix | 3 |
| video, zero downloads | `viz_mxc_cpu`, `viz_green`, `viz_camera` | PROVEN (`vcam`), 2 episodes | ~24 min end to end | 1 |
| stills, local | `sd15` | PROVEN (`sd15`), 3 episodes | 1.99 GB, ungated | 4 |
| video from a still | `still_motion`, `still_pan`, `still_flat`, `still_word` | PROVEN (`stmo`, `stwo`), 2 episodes | 22:22 -- 22:38 per episode | 6 |
| video diffusion, local | `ltx_8gb` (LTX 0.9.8), fed by `sd15` | PROVEN (`lx8g`), 2 episodes | 39:17 (one lane) to 1:07:27 (all three) | 5 |
| video diffusion, local, **EXPERIMENTAL** | `animatediff15_lightning_video` | PROVEN (`adlt`), 1 episode | 2:32:34 for 23 beats | 7 |
| upscale | `spandrel_esrgan` | `mps` accepted 2026-09-09 on a bit-exact receipt; no episode has used it yet | 67 MB, auto-fetched | 8 |

What that table does NOT say is just as load-bearing:

* **No lane past those has run here.** The compliance matrix classifies the
  other 45 engines from their code and weight sizes; a LIKELY there is a
  prediction, not a receipt.
* **Every cost above was measured with the pack's forced SDPA attention on
  MPS**, which is what makes Stable Audio 3 produce music rather than noise.
  The end-to-end penalty of that forcing measured ~1.15x on a diffusion lane
  (PBUG-20260908-05: 119-189 s/it forced against 105-106 s/it stock), so the
  figures stand as measured -- they are not loose upper bounds.

---

## 1. Press Run -- the zero-download path

Install the pack, load `otr_canonical` from
**Workflow > Browse Templates > EXTENSIONS > comfyui-old-time-radio**, press Run.

The shipped graph is wired from the `otr_mac_mps` profile
(`config/profiles/otr_mac_mps.json`): writer on Metal, Kokoro voices, Stable
Audio 3 music, and three visualizer video lanes that mint no still. **No image
weights and no API key are needed.**

| stage | engine | notes |
| --- | --- | --- |
| writer | `Qwen/Qwen3.5-4B` on `mps` | ~6.5 tok/s. **Keep `llm_quant_policy` = `none`** |
| voices | `kokoro` | runs through kokoro-onnx on CPU on Python 3.13, by design -- that is what makes it auto-install |
| music | `stable_audio_3` | ungated, commercially clean, self-fetching |
| video | `viz_mxc_cpu`, `viz_green`, `viz_camera` | zero dependencies, no stills |
| image (dropdowns) | `sd15` | **inert under this wiring** -- no `viz_*` lane consumes a still. It mints one the moment you switch a role to a `still_*` lane or `ltx_8gb`; fetch its checkpoint first (section 4) |

This is a conservative choice, not the only one that runs: the visualizer lanes
are what you get with zero downloads. `sd15` plus the `still_*` lanes cost one
1.99 GB file; `ltx_8gb` costs about 17 GB on disk and self-fetches.

### Two settings that will ruin your day if you change them

* **`llm_quant_policy` must stay `none`.** NF4 *works* on Metal but runs at
  0.3-0.5 tok/s against 14.47 on CUDA -- there is no Metal kernel, so
  bitsandbytes falls back to a Python dequantise loop. It is slow enough that
  the 40 s `NewsCurationDeep` budget expires and the run dies with a confusing
  `_LLMTimeoutWorkflowPause`. It does not look like "quantisation is slow".
* **Do not select `viz_mxc_mandala`.** It needs `pycairo`, which has no macOS
  wheel (section 8). The engine says so itself when you select it on a Mac.

### Memory: 16 GB is the floor, and it is tight

The writer alone peaks near 14 GB inside ComfyUI on a 16 GB machine. **"It
works" was too generous and is corrected here: it works SOMETIMES, and the
difference is whatever else happens to be resident.** The same combination
published a full 23-beat episode at 10:09 on 2026-09-09 and hard-rebooted the
machine at 16:23 the same day, ~19 minutes in, at the writer -> video
transition. 14 GB of a 16 GB machine has no margin for a second thing, and the
second thing does not have to be large. Close other applications first -- a
remote-desktop session, or a test suite that imports torch, is enough.

Note `ps` under-reports badly on Apple Silicon (0.33 GB for a 14 GB process);
use `footprint -p <pid>` and read `phys_footprint`.

**THE WRITER DROPDOWN UNDER-STATES THIS BY 3x, and that is the trap.** The
picker shows `Qwen/Qwen3.5-4B (4.3 GB)`. That badge reports
`_estimate_resident_gb`, which halves a row's download size on the stated
assumption that OTR loads at 8-bit/NF4 -- true on NVIDIA, false here, because
bitsandbytes is excluded on darwin by declared intent. The honest Apple Silicon
number for that row is the measured **14 GB**. Read the badge as an NVIDIA
figure until the picker carries per-device rows. Nothing refuses the load: the
menu offers it, and the machine pays.

The writer is also what took the machine down three times on 2026-09-08 -- it
was being loaded four times per episode, and on unified memory the second copy
landed beside a first that could not be freed. That is fixed (two loads per
episode now) and section 12 explains it, because the same shape will be the
first thing to suspect if a kill comes back.

---

## 2. An OOM here takes the MACHINE down, not the render

**Read this before you select any engine.** On a discrete GPU, running out of
VRAM raises a Python exception, ComfyUI catches it, and you read a traceback. On
Apple Silicon there is no separate VRAM: ComfyUI's "offload device" is the same
physical memory the model is already in, so there is nowhere to spill to. The OS
resolves it by killing processes. In practice the whole machine goes down, and
there is **no traceback afterwards** -- the log simply stops mid-load.

That happened here on 2026-09-08 loading `wan_ti2v` with an fp16 UNET: the log's
last line is `WAN22 ... loaded completely; 9536.40 MB, full load: True`, and then
nothing. It happened again on 2026-09-09 when an AnimateDiff beat asked for 160
latents (section 7): five beats rendered, the sixth rebooted the box, `uptime`
read 1 minute.

**So do not size a model by trying it.** What stands between you and a reboot:

### Guard 1: the unified-memory weight floor

`nodes/_otr_video_engines/motion_common.py`,
`refuse_if_weights_exceed_unified_memory`, wired at the render gate in
`render_driver.py`. Before a local video engine loads, it resolves that
engine's loader files on disk and refuses if the weights alone will not fit:

```
<engine> needs N GiB of weights resident and this host has M GiB of
accelerator budget (F GiB free, less 1.5 GiB reserved for activations and the
OS). On UNIFIED memory there is no separate host RAM to offload into -- the
offload device is the same physical memory -- so loading this would not fail
the render, it would take the MACHINE down. Refusing at the gate instead.
```

It is a no-op on CUDA, where a card can offload to host RAM and an oversized
model is merely slow.

**How it counts on a Mac, which is not how it counts on CUDA.** On CUDA the
peak is the larger of (the text encoder alone) and (everything else together),
because ComfyUI's loader really does run those as two phases. **On unified
memory every artifact is charged as concurrently resident.** The two-phase
split is a fiction there: `free_after_use` does not evict on MPS
(PBUG-20260908-02 -- `wan_ti2v` logged `0 models unloaded.` immediately before
the load that killed the machine; `flux2_klein` held a 7.67 GB encoder through
sampling; `ltx_8gb` never attempted an unload at all). "offload device: cpu" is
ComfyUI working as designed, and on unified memory host RAM IS the accelerator's
memory, so the move frees nothing. Charging the sum is the conservative
direction and it is the measured one.

**The budget it checks against is physical RAM x 1.15, less the 1.5 GiB
reservation, and that multiplier rests on TWO receipts and nothing else:**

| lane | concurrent weights | outcome |
| --- | --- | --- |
| `ltx_8gb` | 16.1 GiB | survived, on swap, slowly -- and published episodes |
| `wan_ti2v` (fp16) | 21.2 GiB | killed the machine |

15.9 GiB of physical RAM sits below both, so a bare-RAM threshold would refuse a
lane with receipts. 1.15x puts the line at about 18.3 GiB, between the two
observations and closer to the survivor. Two points do not make a curve; the
code says so in the same words and asks to be widened the moment a third receipt
lands, in either direction.

**Its limits, because a guard you over-trust is worse than none:**

* It reads an engine's weight names by duck-typing two adapter methods. It can
  read `ltx_8gb`, `wan_ti2v`, `fastwan_8gb`, `humo` and its variants, and
  `mesh_stage`. **Every other lane is unguarded** -- not blocked, just
  unchecked. The Lightning AnimateDiff lane has a guard of its own (below);
  its v3 siblings do not.
* **It does not cover image engines at all.** `z_image_turbo`, `flux2_klein`,
  `lumina_image` and `flux_gen1` are on you.
* It is a FLOOR: it weighs the files, not the activations, so it catches "the
  weights alone do not fit" and nothing subtler. `z_image_turbo` died in the
  KSampler needing 20.4 GiB against a 12.3 GB checkpoint -- a floor check
  would not have predicted that gap.
* It fails OPEN on anything it cannot resolve exactly. That is deliberate: a
  false refusal blocks work that succeeds, and only one of those two errors is
  acceptable in a guard nobody has calibrated.
* `OTR_UNIFIED_MEMORY_HEADROOM_MB` overrides the 1.5 GiB reservation ONLY --
  the weight check itself always runs. A small or zero value weakens the
  reserve; a negative or malformed one falls back to the default rather than
  disarming it, deliberately, so a typo cannot silently turn the guard off.

### Guard 2: the AnimateDiff latent ceiling -- ONE measured point

`latent_ceiling_for_host` in the same module. The whole calibration is one
bracket taken on 2026-09-09: **136 latents at 512x288 rendered; 160 rebooted
the machine** (PBUG-20260909-01). The ceiling therefore lies somewhere in
(136, 160], and the guard uses 136 -- the largest count observed to SURVIVE,
never the smallest observed to fail -- so the untested gap is treated as unsafe.

The code calls this **single-point calibration, out loud**: it scales that one
anchor linearly by physical RAM (after subtracting a ~4.7 GB fixed resident
cost that does not shrink with the machine) and inversely by canvas pixels.
**It is not a memory model of the sampler.** The memory that actually killed the
machine is attention activation across the sliding windows, not the latents
themselves, and nobody has measured that curve. Treat it as a bracket, not a
capacity figure.

Only `animatediff15_lightning_video` opts in (`adaptive_hold_for_memory =
True`). The v3 haunted lane and its still-in peer do **not**, so a long beat
can reach the fatal count on them unguarded -- that is why the compliance
matrix rates them OOM RISK despite 3.8 GB of weights. Section 7.2 has what the
guard has and has not proven.

### Check the weights yourself for anything unguarded

`scripts/otr_fetch_lane_weights.py` `LANE_INFO` has the sizes per lane. Sum
every artifact -- on this platform nothing is evicted between phases -- and read
the total against two lines:

* **~11.8 GiB, the Metal working-set ceiling** (about 75% of physical RAM).
  Above it you are into swap. `ltx_8gb` at 16.1 GiB ran from swap and
  published, slowly.
* **The guard's line: physical RAM x 1.15 (about 18.3 GiB here), less the
  1.5 GiB reservation.** The one lane measured past it took the machine down.

Neither line is a model. The 16 GB figure on the box is not the number to size
against.

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

---

## 3. What cannot run locally on Apple Silicon, and why

`MAC_COMPLIANCE_MATRIX.md` carries the per-engine verdict for all 61 engines
and the reasoning behind each; this section is the shape of the answer, not a
second copy of it. Four buckets, and they must not blur:

**Really NVIDIA-only (WILL NOT RUN).** `ideogram4_local` and both `minimax_*`
need `nvfp4` artifacts, and their int8 fallback hits `aten::_int_mm`, which is
unimplemented on MPS. `ltx_audio_in` fails CLOSED without NVML
(`eng_ltx_av.py:749-756`) -- a real code gate, not a registry row. None of these
is a paperwork problem.

**Blocked by size, not by Metal (OOM RISK).** `z_image_turbo` loads fully on
Metal and dies in the KSampler about 300 MB short of a 20.13 GiB ceiling --
measured, section 4. `humo` / `humo_14B_169` (26.7 GB fully resident by
contract), `ltx_video` (the 22B stack), the three `ltx25_*` (~19.6 GB, and an
open MPS black-video defect on top, section 13). A 32 GB Mac would change some
of these answers; that is a hardware question, not a portability defect.

**Measured fatal here, AND renders wrong.** `wan_ti2v` loaded fully on Metal
with its fp16 UNET and killed the machine (section 2). Its shipped GGUF set is
9.37 GB and might fit -- but an open ComfyUI issue reproduces Wan 2.1/2.2
temporal corruption on this exact macOS/torch generation with GGUF Q8 and fp16
corrupting identically, so it would fit and still render garbage (section 13).
Either reason alone is disqualifying. `fastwan_8gb` subclasses it and hoists
MORE, so it inherits both reasons and is further from fitting, not closer; the
matrix rates it WILL NOT RUN.

**Never run here.** `lumina_image`, `flux_gen1`, `humo_1.7B`, `humo_1.7B_169`,
`mesh_stage`, `stable_audio_music`, the sidecar voices (`indextts2`,
`chatterbox`, `dia` -- blocked by a PowerShell-only installer, not by Metal;
`indextts2` by size as well).
The matrix reads each one's code and sizes it; **an "unsafe at 16 GB" for any
of them is a size estimate, not a measurement**, and the reason not to try
anyway is the cost of being wrong (a reboot), not confidence that the estimate
is right. `stable_audio_music` additionally needs the `stable-audio-tools`
package, which nothing declares, and gated weights -- friction, not failure.

### Rows that gained `mps` this campaign, each on a receipt

| engine | row moved | receipt |
| --- | --- | --- |
| `bark` | 2026-09-07 | 40.8 s for 4.6 s of speech on `mps`, finite, flatness 0.070 |
| `sd15` | added 2026-09-08 with `mps` | three published episodes |
| `ltx_8gb` | 2026-09-08, `["cuda"]` -> `["cuda","mps"]` | two published episodes |
| `animatediff15_lightning_video` | 2026-09-09, `["cuda"]` -> `["mps","cuda"]` | one published episode |
| `spandrel_esrgan`, `off` (upscale) | 2026-09-09, `["cuda","cpu"]` -> `["cuda","cpu","mps"]` | RealESRGAN_x2plus 128->256 on `mps` in 0.51 s, **max \|mps - cpu\| = 0.00000** -- bit-exact against the CPU path, which is what makes it evidence; a wrong kernel returns too |

Every row that is still `["cuda"]` after that is either in the first three
buckets above or has simply never been run here.

### Names in that list are INTERNAL ids; the dropdown shows public labels

`ltx_8gb` is `ltx098_low_video`, `wan_ti2v` is `wan22_high_video`, `fastwan_8gb`
is `wan22_high_fast`, `ltx_video` is `ltx23_high_video`, `ltx_audio_in` is
`ltx23_low_audio_in`, the `minimax_*` pair is `h3_low_video` / `h3_low_audio_in`.
The mapping is `nodes/_otr_shared/public_engines.py`.

### History: what this section said until 2026-09-08, and why it was wrong

Until 2026-09-08 this section ended: *"no local image engine fits in 16 GB ...
nothing local can mint a still here ... the visualizer lanes are the only
fully-local video path Apple Silicon has. The shipped canonical is not a
conservative choice, it is the only one that runs."* Before that, on
2026-09-07, it said Apple Silicon had no local image engine at all, which was
corrected the same day: `z_image_turbo` executes on Metal and is blocked by
RAM, not by the device -- its `["cuda"]` declaration was wrong in KIND.

Every clause of the 09-08 version is now false, and the mistake is worth naming
because it is the one this whole document exists to prevent: **the conclusion
was drawn from ONE engine's failure.** Z-Image did not fit, so "no image engine
fits". It was never a measurement of the set, and the set was never enumerated.
Enumerated, the set contained `sd15` (works), all four `still_*` lanes (work,
fed by `sd15`), and `ltx_8gb` (works). The compliance matrix exists so that
this cannot happen again by omission.

---

## 4. Local image generation on Apple Silicon

**Measured 2026-09-08 on the M4 / 16 GB.**

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
dropdown on `OTR_VideoDirector` (the shipped canonical already does).

**It fits every request down to 768 on the long side** (`OTR_SD15_MAX_SIDE`),
preserving aspect and snapping to a multiple of 8 -- the canonical's 832x480
mints as 768x440. That is not tidiness: SD 1.5 duplicates subjects past roughly
768, producing two heads and mirrored torsos with **no error anywhere**.

Knobs: `OTR_SD15_CKPT`, `OTR_SD15_MAX_SIDE`, `OTR_SD15_WIDTH` / `_HEIGHT`
(no-request default only), `OTR_SD15_STEPS`, `OTR_SD15_CFG`, `OTR_SD15_SAMPLER`,
`OTR_SD15_SCHEDULER`. They are read on the SERVER (section 14).

### `z_image_turbo` has NO viable Mac configuration

| variant | size | result |
| --- | --- | --- |
| `z_image_turbo_bf16` | 12.31 GB | **MPS OOM.** Both models loaded (`full load: True`), died in the KSampler needing ~20.4 GiB of a 20.13 GiB ceiling |
| `z_image_turbo_int8_convrot` | 5.78 GB | **`NotImplementedError: 'aten::_int_mm' is not implemented for MPS`.** There is no int8 matmul on Metal at all |
| `z_image_turbo_nvfp4` | 4.51 GB | Blackwell-native fp4 -- NVIDIA only |

```
Requested to load ZImageTEModel_   loaded completely;  7672.25 MB   full load: True
Requested to load Lumina2          loaded completely; 11739.54 MB   full load: True
...
RuntimeError: MPS backend out of memory (MPS allocated: 13.01 GiB,
  other allocations: 5.96 GiB, max allowed: 20.13 GiB). Tried to allocate 1.46 GiB
```

Its own docstring recommends `nvfp4` for low VRAM, which is useless on a Mac.
There is no fourth option: the model is either too large, or quantised in a
format Metal cannot execute. It is not CUDA-only; it is large.

### `flux2_klein`: minted a clean still, has not finished an episode

`flux-2-klein-4b-Q4_K_M.gguf` through `UnetLoaderGGUF` minted a clean 1472x832
still on the first attempt -- coherent subject, correct style adherence, no
green cast, no smearing. Receipt:
`otr/episodes/signal_lost_the_dark_sea_beyond_the_glass_20260908_095959/stills/still_music_opening_001_d94b2a43c4e6.png`,
20 steps, guidance 4.0, ~8 min per still. That settles the "K_M quants garble
on Metal" concern in the good direction (section 11).

What it does not settle: **no episode has published with it.** The run was
killed when swap ate 14 GB of disk. The reason is PBUG-20260908-02 -- on Metal
its 7.67 GB text encoder is NOT evicted before sampling, so the whole
**10.99 GB** set is resident at once (an earlier draft of this guide quoted
2.6 GB, which is only the first of three files):

| file | size | source |
| --- | --- | --- |
| `flux-2-klein-4b-Q4_K_M.gguf` | 2.60 GB | manual -- `flux2_klein` sits in `scripts/otr_provision.py` `MANUAL_TIERS` |
| `qwen_3_4b.safetensors` (text encoder) | 8.04 GB | manual |
| `flux2-vae.safetensors` | 0.34 GB | manual |

It ran, from swap, at 23.5 s/step. Its row is still `["cuda"]` because no
episode has earned `mps`. `config/machine_classes.json` names Klein 4B as the
Mac image ruling; on this 16 GB box `sd15` at 1.99 GB is the one that publishes.
The punch list carries the open item.

### An image engine is INERT until a video lane consumes its still

The three roles in the shipped canonical all select lanes that mint no still
(`viz_mxc_cpu`, `viz_green`, `viz_camera`, all `accepts_still=False`). **So
selecting an image engine and changing nothing else proves nothing** -- the run
goes green having never called it. Flip a role to `still_motion`, `still_flat`,
`still_pan` or `still_word` first (section 6), or to `ltx_8gb` (section 5).

The reverse trap is worse: a still-consuming video lane with the image dropdown
left on `z_image_turbo` starts a ~20 GB download at Queue for weights that
cannot run here. **Set the image dropdowns before you queue.**

### Image-to-video was blocked one step removed, and `sd15` unblocked it

`ltx_8gb` is image-to-video: it consumes a still. Until `sd15` existed every
LOCAL image engine on the platform was `["cuda"]`, so its first Mac attempt
failed at the image step, not the video step. With `sd15` supplying stills it
published two episodes. Section 5.

---

## 5. Local video diffusion on a Mac: LTX 0.9.8, step by step

**PROVEN 2026-09-08** on the M4 / 16 GB: `ltx098_low_video` rendered all three
beat classes and published a 108 s 1080p episode with zero errors.

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
3. **Make sure `sd15`'s checkpoint is present** (section 4).

### The step that is not obvious

**LTX 0.9.8 is image-to-video: it consumes a still, it does not invent one.**
Selecting LTX without a working image model gets you a failed render, and the
error names the image engine rather than LTX.

### What it costs on 16 GB

| configuration | wall clock | notes |
| --- | --- | --- |
| one lane (`still_motion`) | 22:22 | comfortable |
| one lane LTX + 2 visualizers | 39:17 | comfortable |
| **all three lanes on LTX** | **1:07:27** | swapped to ~14 GB, 27% free, zero errors -- the ceiling, not a comfortable setting |

LTXV loads fully on Metal: 3.67 GB + 9.08 GB (t5xxl) + 2.38 GB, every one
`full load: True`. No OOM, no unimplemented operator. On this platform the
guard charges all 16.1 GiB as concurrently resident (section 2) -- that figure
is one of the two receipts it is calibrated on -- and the lane survives it by
swapping.

### Why the registry said this was impossible

`ltx_8gb` declared `["cuda"]`. Its adapter contains **zero** NVIDIA-specific code
-- no nvenc, nvml, triton, flash_attn, `torch.cuda` or `.cuda()` -- and pins its
T5 encoder to CPU by design. The row was untested policy, not a measurement. See
the 30-second grep test in `ADDING_IMAGE_AND_VIDEO_LANES.md`.

`ltx_8gb` does NOT need the ComfyUI-LTXVideo node pack -- it drives stock
ComfyUI nodes. Only the three `ltx25_*` lanes need that pack, and they cannot run
on 16 GB anyway (section 11.2).

---

## 6. The `still_*` lanes: ALL FOUR render on Apple Silicon

**Proven 2026-09-08**, every still supplied locally by `sd15`. There are
**four** `still_*` lanes and only three video dropdowns, so no single run can
cover them. It took two, and both published:

| lane | receipt in `otr/obs/` | still supplier |
| --- | --- | --- |
| `still_motion` | `..._20260908_005546__arch__`**`stmo`**`__...` | `sd15` |
| `still_pan`, `still_flat`, `still_word` | `..._20260908_042821__cart__`**`stwo`**`__...` | `sd15` |

(The engine shortcode in an episode filename is the DELIVERED video engine --
`stmo`/`stpa`/`stfl`/`stwo` for the four still lanes, `lx8g` for `ltx_8gb`,
`vcam` for `viz_camera`, `adlt` for the Lightning lane; the table is
`nodes/_otr_shared/shortcodes.py`. That slug is the cheapest possible proof of
what actually rendered, and it is worth reading before you believe any claim in
this file, including mine.)

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
said "about 55 minutes" for LTX. No run took 55 minutes -- the number was
interpolated rather than measured; the table is what the logs say.)

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
provider-side, so your local device genuinely does not matter. Section 8 has
which credential each cloud lane actually measures.

---

## 7. AnimateDiff on a Mac

Two AnimateDiff families are registered, and on this platform they are in very
different places. Read 7.1 and 7.2 before the setup, because the setup is the
same for both and the outcome is not.

### 7.1 Where it stands

| lane | status on M4 / 16 GB | row |
| --- | --- | --- |
| `animatediff15_lightning_video` | **PROVEN 2026-09-09** -- a complete 23-beat episode through the real OTR adapter path, published. **EXPERIMENTAL by operator label**: `default_roles` is empty so it can only be chosen per role from the director dropdown, and it carries no qualified cost row | `["mps", "cuda"]` |
| `animatediff15_v3_haunted_video` | **renders correctly on Metal** (clips faithful to the `recur_frac` style, judged by eye 2026-09-08) and is **unusable here on cost**: 122-134 s/it, and the golden recipe is 20 steps at cfg 8.0 = 40 UNet passes per beat -- about three quarters of an hour for one beat. No clip has reached `otr/obs/` from a Mac. It has **no adaptive-hold guard**, so a long beat can hit the fatal latent count (7.2); the matrix rates it OOM RISK for that reason, not for its 3.8 GB of weights | `["cuda"]` |
| `animatediff15_v3_stillin_lab_video` | subclasses the haunted lane; same cost, same missing guard | `["cuda"]` |

The proof episode:

| | |
| --- | --- |
| beats | 23 |
| delivered frames | 2,736 |
| wall clock | 02:32:34 |
| output | 157 MB, 2:16.24, 1920x1080 h264 + AAC stereo |
| published | `otr/obs/lightning_mac_proof_2_20260909_100958__arch__adlt__none__koko__news__q354b__sa3_final.mp4` |

Two tokens in that filename are the load-bearing ones. **`adlt`** is this lane's
shortcode, so the file names the engine that made it. **`none`** is the IMAGE
slot -- proof that `accepts_still = False` held and no image engine was invoked
for any of the three video roles. No OOM; the machine stayed up (3h28m at the
end of the run), which is the difference between this attempt and the one
before it.

**What the Lightning lane is, as shipped on 2026-09-09:** the Ghost Signal graph
on ByteDance's distilled 8-step module -- 8 steps at cfg 1.0, `euler`,
`sgm_uniform`, `sqrt_linear` beta schedule; **hold 3** as its default cadence
(operator ruling, 7.5); the external `vae-ft-mse-840000` decoder in place of the
checkpoint's baked VAE (7.4); source counts aligned to the 16 + 12k context
window; and the adaptive-hold guard on. The recipe receipt id is
`animatediff_sd15_lightning8_ftmse_static16_hold3_512x288_v1`. It is
CreativeML Open RAIL-M like the SD1.5 checkpoint -- a real grant, but with use
restrictions nobody has reviewed, so `commercial_clean = False`.

### 7.2 What is NOT proven, said before the green tick covers it

* **The adaptive-hold guard has never fired under live fire.** Every beat in
  the proof episode fell UNDER the 136-latent ceiling, so the guard did not have
  to act. The attempt that forced it into existence had beats of 239-267
  delivered frames and rebooted on one of ~320; the proof episode had 91-250,
  because the writer produced a different script. The escalation logic is
  unit-tested against the real selector (`_beat_hold(500)` resolves 3 -> 4 and
  asks for 136 latents rather than 172), and rendering 136 latents is proven
  repeatedly -- but those two COMPOSED, a single beat escalating and rendering
  through `prepare` / `render_clip` / `canonicalize`, has never run. With hold 3
  now the default, the guard fires only past roughly T=480-500, so an ordinary
  episode will not reach it.
* **It cannot be faked, and that was tested.** A harness built from the
  canonical with one number changed (`target_frame_count` forced to 500) was
  REFUSED by `validate_coverage_plan`, correctly: the coverage plan still said
  250. Every route to a synthetic long beat requires defeating the validator
  that makes the result meaningful. The only honest live-fire path is a real
  episode whose audio produces a ~20-second beat. Closed with that gap stated
  (PBUG-20260909-01).
* **The ceiling is one bracket, not a model** -- section 2, guard 2.
* **The two published v3 lanes are untouched.** Alignment, the decoder, hold 3
  and the guard are all Lightning-only, because changing a published lane's
  latent count or decoder changes its pictures. Turning any of them on for a
  lane with episodes is an operator decision backed by a 5080 comparison, not a
  driver one.

### 7.3 Setup -- the same for every AnimateDiff lane

**Nothing here auto-fetches, and the first item is the one that actually blocks
the lane.** Without the node pack the motion module is invisible even when the
file is on disk, and the gate fails with `missing_model` naming
`animatediff_models` -- which looks like a weights problem and is not.

**1. The node pack, at the PINNED commit.** `main` is the version that
AnimateDiff-Evolved issue #576 reports producing colored noise (section 13).
`scripts/otr_provision.py` pins `ANIMATEDIFF_PIN`; call the function directly
so the `--packs-only` run cannot stall on ComfyUI-LTXVideo's git-lfs
requirement (section 11.2):

```bash
OTR_COMFY_ROOT=/path/to/ComfyUI \
  <ComfyUI Python> -c "
import importlib.util as u
s=u.spec_from_file_location('p','scripts/otr_provision.py')
m=u.module_from_spec(s); s.loader.exec_module(m)
m.ensure_animatediff_pack('/path/to/ComfyUI')"
```

This pack ships no `requirements.txt`, so it adds NOTHING to the venv -- the
safest install in the whole document.

**2. The weights, by hand.**

| file | size | source | destination | needed by |
| --- | --- | --- | --- | --- |
| `v1-5-pruned-emaonly-fp16.safetensors` | 1.99 GB | `Comfy-Org/stable-diffusion-v1-5-archive` | `models/checkpoints/` | every lane (shared with `sd15`) |
| `animatediff_lightning_8step_comfyui.safetensors` | 908,929,664 bytes | `ByteDance/AnimateDiff-Lightning` | `models/animatediff_models/` | Lightning |
| `vae-ft-mse-840000-ema-pruned.safetensors` | 334,641,190 bytes, sha256 `735e4c3a...58f3c75`, MIT | `stabilityai/sd-vae-ft-mse-original` @ `629b3ad3` | `models/vae/` | Lightning |
| `v3_sd15_mm.ckpt` | 1.56 GB | `guoyww/animatediff` | `models/animatediff_models/` | v3 lanes |
| `v3_sd15_adapter.ckpt` | ~95 MB | `guoyww/animatediff` | `models/loras/` | v3 lanes -- the DOMAIN ADAPTER, a LoRA on the image model, and preflight fails CLOSED on it. Getting two of three gets you nothing |

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Comfy-Org/stable-diffusion-v1-5-archive','v1-5-pruned-emaonly-fp16.safetensors'))
print(hf_hub_download('guoyww/animatediff','v3_sd15_mm.ckpt'))
print(hf_hub_download('guoyww/animatediff','v3_sd15_adapter.ckpt'))"
```

**`mm-p_0.5.pth` is NOT one of these, whatever a summary tells you.** That
1.82 GB file belongs to the RETIRED lane in `eng_ghost_signal.py`
(`GHOST_MOTION_MODULE_NAME`). The shipped v3 lanes live in
`eng_ghost_signal_official.py` and use `v3_sd15_mm.ckpt`. Do not download it.

**3. THE STEP THAT COSTS AN HOUR: Comfy Desktop does not map
`animatediff_models`.** The pack will report at boot:

```
[AnimateDiffEvo] - ERROR - No motion models found. Please download one and
place in: ['.../custom_nodes/ComfyUI-AnimateDiff-Evolved/models']
```

**Your motion module is fine.** `animatediff_models` is a category
AnimateDiff-Evolved registers itself (`utils_model.py:343-344`), pointing at its
own pack directory and `<comfy>/models/animatediff_models`; Comfy Desktop's
generated `extra_model_paths` file maps `checkpoints`, `loras`, `vae` and a dozen
others but not that one. So the adapter LoRA in `loras/` resolves and the motion
module beside it does not, which is a confusing half-failure.

That generated file says "do not edit manually" in its own header, and it means
it. Two ways out: hardlink the module into `<comfy>/models/animatediff_models/`
(no extra disk, same volume), or pass the pack's addendum ALONGSIDE the Desktop
file -- ComfyUI accepts the flag more than once:

```bash
python main.py \
  --extra-model-paths-config "$HOME/Library/Application Support/Comfy Desktop/instance-model-paths/inst-<id>.yaml" \
  --extra-model-paths-config config/otr_mac_extra_model_paths.yaml
```

(Edit `base_path` in `config/otr_mac_extra_model_paths.yaml` first.) A bare
`python main.py` without the Desktop file has NO checkpoints at all --
`ckpt_name` validates against an empty list and every prompt is rejected.

Confirm it took by asking the API rather than by looking at the folder:

```bash
curl -s http://127.0.0.1:8188/object_info/ADE_LoadAnimateDiffModel \
  | python3 -c "import json,sys; print(json.load(sys.stdin)
      ['ADE_LoadAnimateDiffModel']['input']['required']['model_name'][1]['options'])"
```

**4. Then just set the dropdowns -- there is no profile for this.** Open
`otr_canonical`, and on **OTR_VideoDirector** set whichever video roles you want
to `animatediff15_lightning_video (16:9)` (or the haunted lane). The image
dropdowns are inert for these lanes -- they are `text_to_video` and mint no
still -- so leave them. Also consider the WRITER. `Qwen/Qwen3.5-4B` is the
shipped default for good reason -- ungated, Apache-2.0, and measured at 2.99 GiB
resident / 14.47 tok/s on an 8 GB NVIDIA card, the smallest and fastest row
there is. But that figure is **under NF4**, and bitsandbytes is not installed on
macOS, so here the same row loads full bf16 at ~14 GB. Its entry is tagged
`mac16-tight` -- and `tight` is not `avoid`. Every one of the SEVEN episodes
ever published on this machine used this writer, across five video lanes and
four source banks; it is the only writer with a Mac receipt at all. The one
reboot came at the writer-to-video handover with a test suite competing for RAM.
Close other applications before changing writer. `google/gemma-4-E2B-it` (6.0 GB,
plain `mac16`, equally ungated) is the headroom option, but nothing has been
rendered with it here -- that tag is arithmetic, not a receipt.

A separate `otr_mac_lightning.json` held exactly this arrangement and was
**deleted on 2026-09-09**, for the same reason the profile below was: measured
against the canonical it differed by zero nodes, zero links and five widget
values. Five dropdown settings do not justify a second graph to keep in step --
and while it existed it sat in `workflows/variants/`, which is generated-only,
so it crashed `build_variants.py --check` outright. The five settings are in
README's "Other setups".

A Mac-specific profile for this briefly existed and was **deleted**: it encoded
three dropdown settings, a canvas the engine overrules anyway, and a preflight
list -- a second place to keep in sync for no benefit, and it was wrong twice in
its first hour. The dropdowns are the path.

**Do not expect a clip-length knob.** These lanes declare `max_frames=0` -- one
unsplittable timeline -- and sit deliberately outside `PLANNING_CAP_ENGINES`,
so `video.max_render_frames` does not govern them. 7.5 has the arithmetic.

**5. Sync the installed copy before every render** (section 14). The pack under
`custom_nodes/` is a COPY of the repo; restarting the server does not sync it.

### 7.4 What it costs, measured

**Isolated sampler cost, 2026-09-09.** Hand-built ComfyUI graphs -- checkpoint
-> `ADE_AnimateDiffLoaderGen1` -> KSampler -> VAEDecode, no OTR pipeline, no
writer, no TTS -- so each number is the lane's sampler cost and nothing else's.
Same machine, same canvas, same frame count, back to back:

| arm (512x288, 16 frames, one window) | wall clock |
| --- | --- |
| golden v3, 20 steps, cfg 8.0, `autoselect` | **230.3 s** |
| **Lightning 8-step, cfg 1.0, `sgm_uniform` / `sqrt_linear`** | **60.1 s** |
| Lightning 8-step, **cfg 2.0** (negative LIVE) | 100.1 s |

(A 55.1 s figure elsewhere in the repo is the same cfg-1.0 recipe on a
different prompt -- the safari scene rather than `recur_frac`. Two prompts, not
two measurements of one arm. Cold checkpoint load is in all of them.)

**3.83x, not the 5x the pass count predicts.** VAE decode and checkpoint load do
not shrink with the step count, so they become a larger fraction of a shorter
render. Treat pass-count arithmetic as a floor, never as the estimate.

**cfg 1.0 wins on both axes.** This was the lane's one genuinely open question,
because cfg 1.0 does not weaken the negative prompt, it DELETES the pass
(`comfy/samplers.py:610`), and this repo had already refused AnimateLCM for
exactly that. The test was built to be the worst case: a bar counter with rows
of bottles, which is precisely what SD1.5 volunteers labels and signage onto.
No lettering at cfg 1.0, none anywhere. cfg 2.0 looked WORSE -- magenta/green
chromatic fringing, halos around the figures, oversaturation -- because
Lightning is distilled FOR cfg 1.0, and it cost 1.8x the time. So the AnimateLCM
precedent does not transfer, and `OTR_LIGHTNING_CFG` stays a sweep knob. **What
this evidence is not:** two images, one seed each, 16 frames at 512x288. If
lettering ever appears it will be on a beat whose prompt names a sign or a dial,
and the knob is there.

**The decoder A/B, and why it is wired.** Identical prompt, seed 42, cfg 1.0,
8 steps, 512x288, 16 frames -- only the decoder changed:

| arm | wall clock | picture |
| --- | --- | --- |
| A, the SD1.5 checkpoint's baked VAE | 55.1 s (cold) | more milky haze |
| **B, `vae-ft-mse-840000-ema-pruned`** | 28.8 s (WARM) | cleaner glass on the bottles, better foliage separation |

**B's 28.8 s is a warm-cache artifact, not the VAE being faster** -- the
checkpoint was already resident from arm A; treat ~55 s as the cold figure for
both. The win is quality only. The decoder is MIT, more permissive than anything
else this lane loads. It is wired as of `c3212e1c` (7.6 has the sequencing
argument that was overruled).

**The v3 lane's cost, for comparison.** About 132 s per sampler step, ~44
minutes for a 10-second beat, against the 4060's 3-3.6 minutes for the same
lane. Nothing is misconfigured: an M4 samples 125 latents across eight sliding
windows more slowly than an Ada card does. That is the measured price, and it
is why the Lightning lane exists.

### 7.5 The frame arithmetic -- what the numbers in the log mean

**125 is not the clip.** Three different frame counts look alike in a log:

```
[AnimateDiffEvo] Sliding context window sampling activated --
latents passed in (125) greater than context_length 16.
```

| quantity | value on a 10-second beat at hold 2 | where it comes from |
| --- | --- | --- |
| `target_frame_count` -- DELIVERED frames | 250 | ShotLock, from the beat's AUDIO budget |
| `source_request` -- SAMPLED frames | 125 | `ceil(250 / hold)` |
| ADE sliding windows | ~8 | `source_request / 16` |

`ghost_unique_source_count = ceil(T / hold)` (`eng_ghost_signal.py`), floored
at 16 by `ghost_source_request`, lands as `EmptyLatentImage`'s `batch_size` and
feeds the sampler. At delivery each decoded frame is held `hold` times,
restoring all 250. **The picture is audio-locked; the model is only asked for a
fraction of it.**

**The clip length is not yours to set, and that is deliberate.** The lane
declares `FrameContract(min_frames=1, max_frames=0, quantum=1, ...)` -- a beat
is ONE timeline even when it spans several internal context windows.
`PLANNING_CAP_ENGINES` is `("ltx_8gb", "fastwan_8gb", "wan_ti2v")`
(`frame_contract.py:319`) and `effective_frame_contract` returns the contract
unchanged for anything else. **Adding this lane to that tuple would make it
WORSE, not shorter:** with `continuity=NONE` the planner joins segments with
`join_mode="jump"`, so a 17-frame cap would split a 250-frame beat into about
fifteen segments, each still flooring at the 16-frame context window -- roughly
240 sampled latents instead of 125, and fifteen jump cuts where there was one
continuous beat. (That fifteen is specific to a 17-frame cap; a cap near 248
would split the same beat only once. The conclusion survives: on a
`continuity=NONE` lane any cap introduces a jump and extra floor/alignment
work.) Capping at the beat level is out too: `validate_coverage_plan` refuses
any plan whose visible frames differ from the audio-derived target.

**Legal counts are `16 + 12k`, and the Lightning lane rounds up to one.** Read
off `create_windows_static_standard` in the pinned pack: windows start at 0, 12,
24 ... (stride = `context_length - context_overlap` = 16 - 4 = 12), and when a
window would run past the end the scheduler BACKS THE FINAL WINDOW UP so it
still spans a full 16. So 125 -- what a 250-frame hold-2 beat asks for -- is
one past a legal count, and its final window overlaps the previous by 15 of 16
frames instead of 4:

| N | windows | overlaps |
| --- | --- | --- |
| 124 | 10 | {4} |
| **125** | 11 | **{4, 15}** |
| 136 | 11 | {4} |
| **250** | 21 | **{4, 10}** |

Not a crash, and never was -- upstream's clamp keeps every window full-length,
which is what the motion module wants. But rounding up is FREE: the model is
invoked once per window, and 125 and 136 are both eleven windows. The only cost
is decoding the surplus frames, which the lane already discards and reports as
`model_frame_count - cadence_source_frame_count`. `align_source_to_context_window`
is True on the Lightning lane and False on the parent, so the two published v3
lanes are byte-identical to before (swept across all 1199 beat lengths). Verified
against the real upstream scheduler for n = 1..599: the window-count formula
`ceil((n-16)/12) + 1` is exact in every case, rounding up never adds a window,
and every aligned count tiles with a uniform overlap of 4.

**Hold 3 is the Lightning lane's default cadence (operator ruling 2026-09-09,
after an A/B he judged: "i like hold 3").** Not a memory compromise -- the
default. The argument, which the 2026-08-22 rate ruling never weighed: AnimateDiff
is TRAINED at 8 fps (ByteDance's own workflow encodes at 8; the 16-frame context
is 2.000 s at that rate). Hold 3 is 8.33 fps of fresh picture, very nearly the
rate the module learned; hold 2's 12.5 fps asks for motion outside that
distribution.

The 2026-08-22 ruling rejected 8 fps on UNIFORMITY grounds -- a true 8 fps
inside 25 needs runs of 3.125 and comes out ragged 3-3-3-4. That objection does
not reach hold 3. The selector emits **runs of exactly 3 with one tail of
`T % 3`**, which is arithmetic, not raggedness, and the tail is already reported
as `cadence_tail_trim`:

| delivered frames T | runs of 3 | tail |
| --- | --- | --- |
| 250 (a 10-second beat) | 83 | 1 frame |
| 320 (the beat that rebooted the machine) | 106 | 2 frames |

`test_the_lane_runs_at_hold_3_by_operator_ruling` asserts both rows and the
general rule. (One caution for anyone reading older text: the commit message and
the lane's own class comment attached T=320's counts to a 250-frame beat. The
table above is the correct statement.)

The look call and the memory call point the same way, which is unusual. Every
beat from the episode that rebooted the machine costs less at hold 3: 250 -> 88
latents (was 136), 243 -> 88 (124), 267 -> 100 (136), 239 -> 88 (124), and the
fatal T~320 beat needs 112 with no adaptation at all. The adaptive guard remains
as a backstop past roughly T=500, escalating 3 -> 4 (7.2). The receipt id was
REPOINTED to `..._hold3_...` rather than edited, because the proof episode ran
at hold 2 under the old string.

`GHOST_DEFAULT_HOLD` stays 2 and the two v3 lanes keep the cadence that made
their episodes; `test_the_golden_lane_still_declares_hold_2` pins it.

**`OTR_GHOST_HOLD_FACTOR`** (range 1-5, unset = unchanged) overrides the hold on
any Ghost lane. Three things before reaching for it: it is a SHELL variable on
the machine running the server (a profile cannot carry it -- the `video`
section accepts only `device_policy`, `dtype_policy` and `max_render_frames`,
and an unknown key raises); it changes how the show LOOKS, not just what it
costs; and setting it on this machine breaks four tests in
`tests/test_ghost_signal_cadence.py`, which is the golden contract doing its
job.

### 7.6 History -- the reasoning that was overruled or superseded, kept because it was sound

* **The decoder was measured, pinned and deliberately NOT wired at first.**
  Three judges were asked; the two that answered agreed on sequencing (adopt
  only AFTER a clip lands in `otr/obs/`), because the isolated proofs never
  touched `prepare` / `render_clip` / `canonicalize` / the cadence receipts, and
  adding a third artifact before that first real leg gives a failure two
  suspects instead of one. The operator then overrode that the same day --
  *"you may as well wire it in, if it fails it fails -- label it as an exp
  lane"* -- which was the documented flip condition one judge had named. Hence
  `c3212e1c` and the EXPERIMENTAL label. A decoder swap is also the one change
  that can be A/B'd after the fact, because it re-decodes the same latents --
  unlike alignment, which altered what was sampled and therefore had to land
  first.
* **`device_backends` stayed `["cuda"]` through three commits, and that was
  correct.** The isolated runs proved the RECIPE executes on Metal. They did not
  prove the LANE does -- hand-built graphs submitted to `/prompt` exercise none
  of `prepare` / `render_clip`, the cadence receipts, `canonicalize`'s
  exact-canvas refusal or the delivered-frame contract. Those are different
  claims, and only the second earns the row. It earned it on 2026-09-09.
* **On 2026-09-08 this guide said AnimateDiff was "BLOCKED, not failed" on a
  Mac.** True at the time: it had failed at the gate for want of the node pack
  and the domain adapter, never reaching its own code. The next day it rendered.
* **The first Lightning attempt is the one that rebooted the machine**
  (PBUG-20260909-01) -- five beats through the real adapter, then 160 latents.
  That episode directory (`..._lightning_mac_proof_20260909_080251`) never
  published. The guard, alignment and then hold 3 followed in that order.

---

## 8. Engines that need manual steps, and what those steps measure

### `viz_mxc_mandala` -- needs Homebrew

`pycairo` publishes Windows wheels, an sdist, and **no macOS wheel**. On a stock
Mac `pip install pycairo` falls back to the sdist and fails to build, because
libcairo headers are not present. Measured 2026-09-07.

```bash
brew install cairo pkg-config
pip install pycairo
```

Without Homebrew this engine cannot run. **Use `viz_mxc_cpu`, `viz_green` or
`viz_camera` instead** -- zero dependencies, no cairo. Linux has the same
problem for a different reason (no Linux wheels): `apt install libcairo2-dev
pkg-config` first.

### `ffmpeg` and `ffprobe` -- one command after `pip install`

`pip install -r requirements.txt` installs `ffmpeg-downloader`, which ships the
FETCHER, not the binaries. Run **`ffdl install`** once (or `brew install
ffmpeg`). `imageio-ffmpeg` supplies ffmpeg only -- ffprobe has no equivalent
bundled wheel, and the visualizer probes back every clip it encodes. This Mac
ships ffmpeg 9.0, which removed `-vsync`; the pack probes the binary and uses
`-fps_mode` there (section 9).

### AnimateDiff -- section 7.3

Node pack at the pinned commit, weights by hand, and the Comfy Desktop mapping.

### GGUF lanes -- section 11

High friction, and the install is solved.

### Upscale -- nothing manual, and one receipt

`spandrel_esrgan`'s weights auto-fetch ungated from the Real-ESRGAN releases
with a SHA check (`scripts/ensure_upscale_models.py`). `_resolve.resolve_device`
rejected `"mps"` by name from the first ship, with its docstring stating the
lifting condition -- *"when a Mac user provides an integration receipt"*. That
receipt was taken 2026-09-09 (section 3's table); `mps` is accepted, and asking
for Metal on a box without it is a named refusal, not a quiet fall-through to
CPU. No episode has rendered through the upscale stage on a Mac yet.

### Cloud / API engines -- credentials, not a port

`elevenlabs`, `google_tts`, `google_lyria`, `sonilo`, every `cloud_*`,
`google_image`, `ideo` and `word_razzle` all already declare `mps`. Testing
them measures a credential rather than Apple Silicon, and the credentials are
not all the same one:

| lane | credential it actually measures |
| --- | --- |
| `word_razzle`, every `cloud_*` row, `elevenlabs`, `sonilo` | Comfy Cloud -- `OTR_COMFY_API_KEY`, or a **logged-in ComfyUI Desktop session** |
| `google_omni_video`, `google_veo_video`, `google_image`, `google_tts`, `google_lyria` | a direct BYO Google API key -- these do not touch Comfy Cloud at all |

The compliance matrix's "Cloud lanes" section records exactly how far this was
taken: the local half of every cloud video lane (the ffmpeg gate, the 1080p
conform, the provider-audio strip) is PROVEN on this Mac with a real episode mp4
standing in for a provider response, and **the POST itself is untested**. Comfy
Cloud auth comes from a logged-in Desktop session, and every Mac render in this
campaign ran through a headless `main.py` where those inputs do not exist, so it
cannot be exercised here regardless of credits. Neither half is a Metal
question.

(`ideo` is the cloud Ideogram row in the **image** registry, not the local
`ideogram4_local`, which is NVIDIA-only.)

---

## 9. If something breaks, read this first

| symptom | cause | fix |
| --- | --- | --- |
| **the machine rebooted mid-render, log stops mid-load, no traceback** | an OOM on unified memory | section 2. Check `grep -c "Loading LLM model" <log>` -- 2 per episode is right, 4 means the writer double-load is back (section 12). On an AnimateDiff lane check the beat's latent count against 136 (section 7) |
| **ComfyUI will not start at all**, exit 1, `ImportError: tokenizers>=0.23.1 ... found 0.22.2` | the shipped `tokenizers` pin in versions `.24`-`.28` | `pip install 'tokenizers>=0.23.1,<0.24'`, and use `2.0.0-alpha.29` or later. **Do NOT reinstall from Manager** -- it serves the broken version, and a bricked boot cannot be repaired from the UI because Manager IS a ComfyUI extension |
| music is noise, "like a broken cassette" | ComfyUI's sub-quadratic attention is wrong on MPS | fixed automatically by the pack's prestartup, which forces PyTorch attention (~1.15x cost on diffusion lanes, measured). If you disabled it with `OTR_MPS_PYTORCH_ATTENTION=0`, turn it back on |
| render dies at the mp4 encode, `winget install ffmpeg` | ffmpeg/ffprobe were not declared dependencies | `pip install -r requirements.txt` then **`ffdl install`** -- the pip install alone is not enough (section 8) |
| `ffprobe` still not found after `pip install -r requirements.txt` | `ffmpeg-downloader` ships the fetcher, not the binaries | `ffdl install` once, or `brew install ffmpeg` |
| render dies at the scopes stage, `Unrecognized option 'vsync'` | ffmpeg 9.0 REMOVED `-vsync` and dies at argv parse | already handled: `scope_draw.cfr_flags()` probes the installed binary once and returns `-fps_mode` here (verified 2026-09-08). If you see this, something bypassed the probe -- do not hardcode either spelling |
| `NewsCurationDeep exceeded 40s` | `llm_quant_policy` is not `none` | set it to `none` (section 1) |
| OOM / the machine freezes mid-render | 16 GB shared with macOS and everything else | close other apps; the writer peaks near 14 GB; section 2 |
| a video/image engine refuses with `EngineUnusable` | **NOT the `["cuda"]` row** -- that row is not a render-time gate and never refuses anything (`ADDING_IMAGE_AND_VIDEO_LANES.md`). The refusal is the engine's own `assert_usable`: a missing weight, a missing node class, or a missing NVML/vendor probe | read the message -- it names the artifact or class. A `["cuda"]` row is a claim about what has been PROVEN, not a lock |
| a lane you did not pick starts a huge download at Queue | the image dropdowns still say `z_image_turbo` and the video lane you picked consumes a still | set ALL THREE image dropdowns to `sd15` **before** you queue. The validator provisions the selected image engine, so an unchanged `z_image_turbo` fetches ~20 GB of weights that cannot run here |
| `[AnimateDiffEvo] ... No motion models found` with the file on disk | Comfy Desktop's generated paths file has no `animatediff_models` category | section 7.3, step 3 |
| `EngineUnusable ... missing_model -- motion_module=... domain_adapter=...` | the node pack, the module or the adapter is absent -- usually the node pack | section 7.3, steps 1-2 |
| `vram_fit=WARN@4.3 GB` on a model that is 8.68 GB on disk | `_estimate_resident_gb` halves every non-GGUF model regardless of `quant_policy` | it is lying by 2x; do not read it as headroom (section 12). Open on the punch list |
| a test passes alone and fails in the full suite, asserting on source text | the installed pack under `custom_nodes/` is a stale COPY of the repo | section 14 |
| tests fail during a render and pass afterwards | memory pressure, not code | section 2; compare on an idle box |

**Error messages that lie on a Mac** (cosmetic, not yet cleaned up):
`[StoryOrchestrator] CUDA warmup complete` and
`orphan worker still on GPU ... racing the orphan's CUDA kernels` both print on
machines with no CUDA. Neither indicates a CUDA code path.

---

## 10. Fonts: titles look wrong on macOS -- FIXED 2026-09-08

`Consolas` is **absent** from macOS, and `_otr_captions.py` named it in TWO
places -- the `otr_crt` caption style (a QA variant) and, the one that shipped
on every episode, the hero title card's ASS style row. libass falls back to
something proportional and wider, so the green title overruns its computed width
and clips. The `sdh_standard` style beside it asks for Arial, which macOS DOES
have, which is why the white dialogue captions looked right while the green
titles did not -- the bug was never "fonts on macOS", it was one Windows-only
face.

`mono_font()` now resolves the monospace face at EMIT time:

| platform | face | why |
| --- | --- | --- |
| `win32` | `Consolas` | unchanged, so the CUDA machines render byte-identically |
| `darwin` | `Menlo` | ships with every macOS since 10.6 |
| anything else | `DejaVu Sans Mono` | the near-universal Linux monospace |

Two call sites moved onto it: the `otr_crt` style (which now stores the `MONO`
sentinel rather than a face name) and the hero title card's style row, which
became `_title_style_line()` -- a function rather than a module constant,
because a constant would freeze whatever face the server booted with.

`OTR_CAPTION_MONO_FONT` overrides it outright. That is the escape hatch for a
machine missing the OS font, and it is also how you falsify the fix: set it to a
nonsense name and the captions must visibly change face. If they don't, libass
isn't reading the style you think it is.

**Why this was invisible for so long:** an ASS `Style:` line naming a font the
renderer cannot find does not error. libass asks fontconfig for a substitute,
gets a proportional sans, and draws it. There is no warning in any log.

**Still open:** `Monaco.ttf` IS present at `/System/Library/Fonts/Monaco.ttf` on
macOS 26.6.2, so the credits-roll font search does resolve, and the
overlapping-columns symptom has a **different, still-undiagnosed cause**. Do not
assume this fix addressed it.

---

## 11. The GGUF lanes: the install is solved, and it is HIGH FRICTION

**Operator's framing, and it is the right one:** GGUF may well work, the install
method is now known, and that install is high friction. This section is for
someone with decent coding skills, or an AI coder sitting beside them. If that
is not you, stop here -- `sd15` + the `still_*` lanes and `ltx_8gb` need none of
this.

**Be precise about what is proven here: the INSTALL and one STILL, not an
episode.** The ComfyUI-GGUF pack is verified registered on this machine (six
loader classes, boot clean, one added wheel); `flux2_klein` minted one clean
still through it (section 4); **no GGUF lane has completed an episode on Apple
Silicon in this repository.**

**What GGUF buys you on a Mac.** Quantised weights are how the bigger lanes fit
in unified memory at all. `flux2_klein` is 2.60 GB as a Q4 GGUF against 7.75 GB
bf16; `wan_ti2v`'s shipped set is 9.37 GB GGUF against 21.2 GB in fp16, and the
fp16 route is the one that took this machine down. So on this platform GGUF is
not an optimisation, it is frequently the only version that can run.

### 11.0 K_M quants do NOT garble on Metal -- SETTLED 2026-09-08

Commit `5f1b94b4` passed over `flux2_klein` for Apple Silicon partly because
*"K_M quants garble on MPS"*, and city96's ComfyUI-GGUF issue #177 reports GREEN
OUTPUT from `Q*_K` Flux weights on Metal. A web sweep judged #177 to predate a
PyTorch MPS integer-operation fix and to be contradicted by later Flux-family
GGUF runs. **Measured here, and the sweep was right** -- the Klein still in
section 4 is the receipt.

So a K_M quant is not a reason to avoid a lane on this platform. **The habit is
still worth keeping**: a garbled render exits ZERO, so open the first still or
clip and look at it before concluding a lane works. That is how this was
settled, and it is the only way it could have been. `wan_ti2v`'s shipped set is
also K_M, so this removes one of its two objections; the other -- the open Wan
temporal-corruption defect (section 13) -- still stands, and it is the
disqualifying one.

### 11.1 The repo ships the installer -- point it at the right tree

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

### 11.2 `git-lfs` -- the step that stops a clean Mac dead

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

`ltx_8gb` does NOT need this pack. Only the three `ltx25_*` lanes need it, and
they cannot run on 16 GB anyway.

### 11.3 Verify the pack actually registered

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

### 11.4 Watch the venv, because this is how a boot gets bricked

Installing a pack's requirements runs `pip` into the SAME environment ComfyUI
boots from. That is exactly how the `tokenizers` pin bricked this install on
2026-09-07 (section 9). Snapshot before, diff after:

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

---

## 12. The writer, not the video lane, is what kills a 16 GB Mac

**This is the most important thing on this page, and it was found by asking the
right question rather than by testing harder.**

Three renders took the whole machine down on 2026-09-08. Two of them were blamed
on the video lane, and hours went into shortening clips. The operator's
observation ended that:

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

**What is NOT validated:** this is shared code, and the 5080 has not confirmed
it -- writer loads per episode (4 -> 2), byte-identical ledger and output, and
pre-audio allocated memory. The safety argument is that nothing between the
removed unloads and the new one loads a different model. Open on the punch list
(C1).

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
number as headroom. Still open (punch list B2).

### The general lesson, which cost the most time

Every kill looked like a video-lane problem because a video lane was on screen
when it happened. The thing to measure on this platform is not the lane you
selected -- it is **how many times the biggest model in the pipeline is loaded**,
and whether anything still holds a reference when it is.

---

### The Metal writer lane: use GGUF, not bf16

**On Apple Silicon the transformers lane at `llm_quant_policy: none` is the one
combination that cannot be made to fit, and no dropdown fixes it.** That is
worth stating plainly because the obvious lever looks like it should work and
does not:

* `llm_quant_policy` offers `bnb_nf4`, `bnb_8bit` and `none`. Both bnb lanes
  need bitsandbytes, and `requirements.txt` carries
  `bitsandbytes>=0.42.0; sys_platform != 'darwin'` -- it is deliberately not
  installed on macOS. Installed by hand it *runs*, at 0.3-0.5 tok/s against
  14.47 on CUDA, because there is no Metal NF4 kernel. So `none` is the only
  reachable quant, and `none` is exactly the setting that peaks at 14 GB.

**What fits: the GGUF writer row at `gguf_quant: Q4_K_M`** -- roughly 2.5 GB for
a 4B writer, against 8.7 GB of bf16 safetensors. `nodes/_otr_gguf_backend.py`
is genuinely device-aware here (`default_layers = DEFAULT_N_GPU_LAYERS if
policy.device in ("cuda", "mps") else 0`), so llama.cpp's Metal backend takes
`n_gpu_layers` exactly as CUDA does rather than silently falling back to CPU.
Section 11 covers the install, which is high-friction but solved.

So Apple Silicon wants its own writer lane. **This is stated here and enforced
nowhere** -- operator directive, 2026-09-09: no gated dropdowns and no
capability matrix in code, only documentation. A guard that refused the bf16
lane on Metal was written and then removed for exactly that reason.

**NOT YET SELECTABLE, and this is the honest status.** `GGUF_ROWS` in
`nodes/_otr_gguf_backend.py` is currently EMPTY, so no GGUF writer appears in
the picker at all -- there is nothing to point a Metal-named row at until a
pinned artifact is added. The backend is ready and device-aware; the row is
missing.

**The arithmetic, so you can size it yourself.** Resident runs about **1.61x**
the bf16 download size -- `14 / 8.68`, the one measurement this repo has
(PBUG-20260907-06). On a 16 GB machine, minus what macOS and the video stack
need, a bf16 writer above roughly 7 GB of safetensors has no margin. Q4_K_M on
a 4B writer is roughly 2.5 GB and is not close to the edge.

---

## 13. What the wider world reports, and why we stopped testing by trying

**Method note.** Everything above was learned by running things. That stopped
being acceptable when a `wan_ti2v` attempt took the whole machine down -- on
unified memory a bad guess costs a hard reboot, not a stack trace. So the
remaining candidates were researched instead: a web sweep for REPORTED
experience, with sources, dates, and a hard distinction between exact-16 GB
reports and larger-Mac ones, since 32 GB and 64 GB results do not transfer.

Researched 2026-09-08; the "here" column updated 2026-09-09. **Treat every row
as evidence about somebody else's machine unless this repo carries a receipt
for it.**

| candidate | external evidence | here, on the 16 GB M4 |
| --- | --- | --- |
| **LTX-Video 0.9.8 distilled** | strong | **PROVEN** (section 5). Nothing further needed unless torch/ComfyUI move |
| **FLUX.2 Klein 4B Q6** | strong -- an exact 16 GB completion exists | not tried; the Q4_K_M row below is the one in the registry |
| **FLUX.2 Klein 4B Q4_K_M** | moderate -- no exact 16 GB report found | **minted a clean still** at ~8 min each, from swap; no episode (section 4). K_M garbling: settled, does not happen (section 11.0) |
| **AnimateDiff-Evolved + SD1.5** | moderate on Metal (2023 completions); STRONG on cost -- 4.9 GB with SD1.5 fully resident on the 4060 | **PROVEN with the Lightning module** (section 7); the v3 golden recipe renders correctly and is ~45 min a beat. Use the PINNED commit, not `main` (below) |
| CogVideoX-2B | anecdote -- one Mac walkthrough, no hardware named | unproven at this size, not in the registry |
| Wan 2.1 Fun InP 1.3B | anecdote -- an exact M4/16 GB completion EXISTS | **Do not follow it as written:** it required `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, which REMOVES the MPS allocation ceiling. On a host where OOM reboots the machine, that setting is the opposite of a mitigation |
| LTX 2.x (`ltx25_*`) | strong NEGATIVE | **Avoid.** ~19.6 GB, and open MPS BF16 attention NaNs produce all-black video |
| **Wan 2.2 TI2V-5B** (`wan_ti2v`) | strong NEGATIVE | **Dead here**, and not only on memory -- see below |
| Stable Video Diffusion, Mochi, HunyuanVideo, FramePack | none found | no qualifying 16 GB Apple Silicon completion reports at all |

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
attention on MPS to fix the sub-quadratic `baddbmm` bug (section 9). Both are
real; they want different attention backends. If you ever get an LTX 2.x lane
running here, that is the first knob to try -- and `OTR_MPS_PYTORCH_ATTENTION=0`
is how you get our forcing out of the way.

**3. AnimateDiff: the PIN is why the lanes are fine, and it is a good advert
for pinning.** AnimateDiff-Evolved issue #576 (opened 2026-06-18, still open)
reports COLORED NOISE from both `mm_sd_v15_v2` and `v3_sd15_mm` across Gen1 and
Gen2 loaders, multiple schedules and FP32 -- on an **RTX 4070**, so it is a
core/pack version incompatibility rather than anything to do with Metal.
`otr_provision.py` pins `ANIMATEDIFF_PIN` rather than tracking `main`, the 4060
published SIX episodes on the haunted lane on 2026-08-29 -- more than two months
AFTER #576 was filed -- and this Mac published on the same pin on 2026-09-09. A
pinned commit that keeps working while `main` breaks is the pin doing its job.
**Use the provisioner, not a hand-rolled `git clone` of `main`** -- a
hand-cloned latest is exactly the configuration #576 describes.

### What "no reports found" means here

It means nobody has published a qualifying result, not that the thing fails.
Every "none found" row above is an invitation, not a verdict -- and the cheap
way to change one is to run it and record the peak, not to reason about it.

---

## 14. Operational traps that produce plausible wrong results

### An `OTR_*` knob goes on the SERVER, not on your shell

Every `OTR_*` value an engine reads is read **inside the ComfyUI process**.
Exporting it in the terminal you run an API client from does nothing at all:

```bash
# WRONG -- the client has the value, the renderer does not
OTR_WAN_TI2V_UNET_NAME=... python scripts/otr_canonical_api_run.py ...

# RIGHT -- restart ComfyUI with it
OTR_WAN_TI2V_UNET_NAME=... python ComfyUI/main.py ...
```

That covers every knob that binds: the loader-name and path overrides
(`OTR_WAN_TI2V_UNET_NAME`, `OTR_LTX_8GB_CKPT`, `OTR_SD15_*`),
`OTR_UNIFIED_MEMORY_HEADROOM_MB`, `OTR_MPS_PYTORCH_ATTENTION`,
`OTR_GHOST_HOLD_FACTOR`, `OTR_CAPTION_MONO_FONT`. If you changed one and nothing
changed, you almost certainly set it on the wrong process.

**`ENABLE_FLAG` constants gate nothing.** Some adapters still carry one --
`flux2_klein.ENABLE_FLAG = "OTR_ENABLE_FLUX2_KLEIN"`, and `eng_wan_ti2v`'s
docstring calls `OTR_ENABLE_WAN_TI2V` a "vestigial opt-in flag". Both adapters
set `requires_flag = None`, `EngineUsabilityReason.GATED_BY_FLAG` is documented
dead, and `tests/test_registry_is_the_menu_guard.py` asserts that no registered
engine carries a live flag. If an engine refuses, the reason is in its
`assert_usable` message, not a missing opt-in.

### A stale installed pack makes tests lie, and the failures look real

`custom_nodes/comfyui-old-time-radio` is a SEPARATE COPY of this repo, not a
symlink to it. ComfyUI needs it there; the tests import from the repo. When the
two drift, a FULL-SUITE run can resolve `nodes.*` to the INSTALLED copy while a
single-file run resolves it to the repo -- so tests pass alone and fail together.

It bit on 2026-09-09 and cost real time: four tests that assert on
`inspect.getsource` reported that code present in the repo was missing. It was
missing -- from the stale copy they had actually imported.

**Check, and do this after EVERY edit you intend to render with:**

```bash
diff -rq --exclude=__pycache__ \
  ~/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes \
  <repo>/nodes
```

**Fix:** copy the changed files over and drop `__pycache__`, then re-run.
Restarting the server does not sync it; only copying does. Otherwise the render
exercises different code from the one the tests just proved.

### Stale harnesses: build it from the canonical

Operator, 2026-09-09, twice in one hour: *"be careful of stale harnesses"* and
*"i would always build a harness from scratch using the canonical"*. Both
warnings landed on live mistakes.

`scripts/otr_visual_smoke.py` looked like exactly the right instrument for a
live-fire test of the adaptive-hold guard: it bakes a rendered episode's planned
ledger into a bundle and replays ONLY the render tail, skipping the ~30-minute
writer and TTS stages. Three things were wrong with reaching for it:

1. **It submits to the SERVER**, so the code under test is whatever the running
   ComfyUI loaded at startup -- not what is in the repo. The pack was three files
   stale at that moment, which meant the "guard" being exercised was the
   previous commit's.
2. **Its own defaults have rotted.** It targets port 8000; this host runs 8188.
   The replay died on a connection error before doing anything.
3. **Making it work required faking state.** Its bake refused because a
   `pending_<ts>` directory had been renamed to the final episode name on
   publication. The fix was a symlink from the old pending name to the published
   episode -- exactly the kind of thing that later makes "weird episode title
   stuff happen". It was removed as soon as the harness was abandoned.

A canonical-built harness then overwrote
`output/otr/episodes/_shared/state/node_episode_input.json` -- node 92 captures
its input there, so a synthetic single-shot ledger replaced the real 23-shot one
until it was restored from the published episode's own ledger. Same failure
class: a scratch file that later runs read.

**The rule.** Build the harness from the canonical workflow and the canonical
runner, the way a real episode runs, rather than restoring an older accelerator.
A stale accelerator does not fail loudly -- it produces a plausible result from
the wrong code.

---

## 15. Corrections ledger

Each of these was stated in this guide and later found wrong. They are kept here
so the trail is visible without interrupting the current answer.

| date | said | corrected to |
| --- | --- | --- |
| 2026-09-07 | Apple Silicon has no local image engine | `z_image_turbo` executes on Metal and is blocked by RAM; the `["cuda"]` row was wrong in kind |
| 2026-09-08 | no local image engine fits 16 GB; the visualizer lanes are the only fully-local video path | `sd15`, all four `still_*` lanes and `ltx_8gb` work. Conclusion had been drawn from one engine's failure (section 3) |
| 2026-09-08 | a cloud image key is "the single highest-leverage addition for a Mac" because it unlocks the `still_*` lanes | `sd15` unlocks them locally and for free; a cloud key buys different-looking stills, not reachable ones |
| 2026-09-08 | `flux2_klein` is 2.6 GB | 10.99 GB across three files; and on Metal the encoder is not evicted, so all of it is resident (section 4) |
| 2026-09-08 | the all-LTX episode took "about 55 minutes" | 1:07:27 measured; the 55 was interpolated |
| 2026-09-08 | the unified-memory guard charges the larger of encoder vs. the rest | on unified memory it charges every artifact, because eviction does not happen on MPS (section 2) |
| 2026-09-08 | forcing SDPA attention on MPS makes every diffusion lane ~14x slower | ~1.15x end to end; the microbenchmark could not be reconciled with the step time it sat inside (PBUG-20260908-05) |
| 2026-09-08 | AnimateDiff is "BLOCKED, not failed" on a Mac, and the best untested candidate | Lightning published 2026-09-09; the v3 recipe renders correctly and is too slow (section 7) |
| 2026-09-08 | to set an engine's `ENABLE_FLAG`, restart the server with it | the flags gate nothing (section 14) |
| 2026-09-09 | a planning cap would produce "~15 jump cuts" | that is the count for a 17-frame cap specifically; the conclusion holds (7.5) |
| 2026-09-09 | the ft-mse decoder is deliberately not wired | wired the same day by operator ruling, lane labelled EXPERIMENTAL (7.6) |
| 2026-09-09 | a 250-frame beat at hold 3 is "106 runs of 3 plus a 2-frame tail" | those are T=320's counts; T=250 is 83 runs plus a 1-frame tail; the tail is always `T % 3` (7.5) |
| 2026-09-08 | the guard's budget is the 11.8 GiB Metal working set (the refusal example quoted "11.8 GiB free") | the budget is physical RAM x 1.15 (about 18.3 GiB here) less 1.5 GiB; the working-set figure is what `free_vram_mb()` reports, and a model parked on "cpu" is invisible to it (section 2) |
