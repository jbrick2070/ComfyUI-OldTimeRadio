# Adding your own image engine or video/still lane

Companion to `EXTENDING_OTR.md` (which covers **source banks**). This one covers
the **engines** behind the dropdowns on `OTR_VideoDirector`.

Three levels of effort. **Start at level 1 -- most people never need level 2.**

---

## Level 1: point an EXISTING engine at a different checkpoint (NO CODE)

The shipped engines read their weight filenames from environment variables. To
run a smaller or different checkpoint of a model OTR already supports, set the
variable and restart ComfyUI. Nothing is edited, nothing is registered.

`z_image_turbo` (`nodes/_otr_image_engines/z_image_turbo.py`) is the worked
example:

| variable | default | what it selects |
| --- | --- | --- |
| `OTR_ZIMAGE_UNET` | `z_image_turbo_bf16.safetensors` | the diffusion model |
| `OTR_ZIMAGE_CLIP` | `qwen_3_4b.safetensors` | the Qwen3-4B text encoder |
| `OTR_ZIMAGE_VAE` | `ae.safetensors` | the VAE |
| `OTR_ZIMAGE_CLIP_TYPE` | `qwen_image` | the `CLIPLoader` type |
| `OTR_ZIMAGE_UNET_DTYPE` | `default` | `UNETLoader` weight dtype |
| `OTR_ZIMAGE_CFG` | engine default | guidance |

All the variants live in the same official repo,
`Comfy-Org/z_image_turbo` under `split_files/`:

| file | size | note |
| --- | --- | --- |
| `z_image_turbo_bf16.safetensors` | 12.31 GB | the default |
| `z_image_turbo_int8_convrot.safetensors` | 5.78 GB | **the low-RAM choice on Apple Silicon** |
| `z_image_turbo_nvfp4.safetensors` | 4.51 GB | Blackwell-native fp4 -- NVIDIA only |
| `qwen_3_4b.safetensors` | 8.04 GB | the default TE |
| `qwen_3_4b_fp8_mixed.safetensors` | 5.63 GB | low-VRAM TE |
| `qwen_3_4b_fp4_mixed.safetensors` | 3.48 GB | low-VRAM TE, fp4 |

```bash
# example: the low-RAM pairing
export OTR_ZIMAGE_UNET=z_image_turbo_int8_convrot.safetensors
export OTR_ZIMAGE_CLIP=qwen_3_4b_fp8_mixed.safetensors
```

Fetch a variant with:

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Comfy-Org/z_image_turbo',
      'split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors'))"
```

then copy it into `models/diffusion_models/` (text encoders go in
`models/text_encoders/`, VAEs in `models/vae/`).

**Apple Silicon note.** The engine's own docstring recommends `nvfp4` for low
VRAM -- that is Blackwell-native and useless on a Mac. **`int8_convrot` is the
Apple Silicon equivalent**, and it matters: the 12.31 GB bf16 model needs about
20.4 GiB at sampling and OOMs on a 16 GB Mac against a 20.13 GiB ceiling. See
`MAC_PORTABILITY_GUIDE.md`.

---

## Level 2: add a new IMAGE engine

The authoritative contract is the docstring at the top of
`nodes/_otr_image_engines/__init__.py`; the gates are `IMAGE_GEN_PREFLIGHT.md`,
enforced by `tests/test_image_gen_preflight_matrix.py`, which sweeps the LIVE
registry -- **your engine is covered the moment it registers, with no test
edits.**

1. **Copy the closest adapter** into `nodes/_otr_image_engines/<yourname>.py`.
   `z_image_turbo.py` for a local diffusion checkpoint; `eng_cloud_image.py`'s
   `_CloudImageBase` for a partner API.

   > **WARNING, and it makes this guide bite itself:** `z_image_turbo.py` --
   > the adapter this step tells you to copy -- declares `["cuda"]` in
   > `registry.py:157`. Copy it verbatim and you inherit that, and **your engine
   > will register fine and simply never be offered on a Mac.** The failure is
   > silent. Set `device_backends` from what YOU measured (see the section at
   > the end), not from whatever the template happened to carry. Flagged by the
   > 5080 window, 2026-09-07. Rename the class and its `name`, and
   decorate with `@register`. **Registering IS joining the dropdown** -- there
   is no separate allow-list.

2. **Declare, never inherit by silence:**
   * `name` -- the dropdown id. **Permanent once shipped.**
   * `engine_version` -- part of the still cache key. A silent engine falls
     back to `"1"`, which means you could never invalidate your own cached
     stills. Bump it whenever output should stop being reused.
   * `commercial_clean` -- a real bool, honestly. `flux_gen1` is `False` (BFL
     non-commercial); the Apache locals are `True`.
   * `roles` -- all three (`announcer_visual`, `music_visual`,
     `character_video`). The dropdown offers every engine in every slot, so
     serving fewer fails at render *after* the episode is written and voiced.
   * `required_inputs` -- `("text_prompt",)`.

3. **Implement the lifecycle:** `assert_usable` (raise a NAMED error when
   weights or a key are missing -- never let render discover it), `prepare`,
   `render_image`, `teardown`. **Lazy-import torch/comfy INSIDE `render_image`,
   never at module scope** -- the cold-import test (invariant V-12) fails
   otherwise.

4. **Add one CAPABILITIES row** in `_otr_image_engines/registry.py`. One row per
   engine and vice versa; `tests/test_capability_profiles.py` holds that
   bijection. Set `device_backends` HONESTLY -- see the warning below.

5. **Run the preflight.** If your provider can REFUSE a prompt, read Gate IG4
   first: a refusal can arrive as a normal SUCCESS with a valid PNG (measured on
   Ideogram 4, 2026-08-21), so classify it in the adapter rather than trusting
   the status code.

---

## Level 3: add a VIDEO / STILL lane

Full checklist: `VIDEO_LANE_PREFLIGHT.md`, enforced by
`tests/test_lane_preflight_matrix.py`. Two declarations deserve calling out.

**`accepts_still` -- declare it explicitly, True or False (preflight G3.6).**
Every video lane is expected to render the still minted by whichever IMAGE
engine the operator picked for that role. Motion lanes inherit `True` from
`MotionEngineBase`; the procedural `viz_*` family declares `False` out loud. An
engine that declares NEITHER and lists no `init_image` resolves to `False`
through a getattr fallback -- it mints no still, **the operator's chosen image
model is never invoked**, and the episode renders anyway with nothing reporting
it. `tests/test_still_spine_engine_coverage.py` fails any engine that stays
silent.

**Every per-artifact constant must travel with the lane (preflight G1.3).**
If you build your lane as a SIBLING of an existing one, make the model filename,
byte floor, recipe receipt and quant token **class attributes your sibling can
override** -- never module-level constants read from inside a method. A method
reading the module constant loads the PARENT's weights while stamping its own
receipt: wrong pixels under a confident label. This has bitten twice already
(the WAN recipe accessors, and Ghost -- where the module name was overridable
and the byte floor beside it was not, so a byte-perfect 1.67 GB module was
refused as "truncated" against a floor sized for a 1.82 GB one). **When you
subclass, ask what ELSE was sized for the parent.**

---

## An image engine is INERT until a video lane consumes its still

**Register an image engine, download its weights, and it can still never be
invoked.** The image dropdowns on `OTR_VideoDirector` are consumed per role by
the VIDEO lane selected beside them, and a lane that declares
`accepts_still = False` never asks for one.

The shipped canonical currently selects three such lanes:

```
announcer   viz_mxc_cpu    accepts_still = False
music       viz_green      accepts_still = False
character   viz_camera     accepts_still = False
```

So testing a new image engine against the canonical AS SHIPPED proves nothing --
the leg goes green having never called your code. **Flip a role to a still lane
first** (`still_motion`, `still_flat` or `still_pan`, all of which already
declare `["cuda","cpu","mps"]`). Raised by the 5080 window, 2026-09-07, and it
would otherwise have cost a full render cycle to discover.

## A resolution note if your model is 512-native

SD 1.5 and its relatives are 512x512 native while the canonical canvas is
832x480. **At 832 wide SD 1.5 duplicates subjects** -- two heads, mirrored
torsos. That is confidently-wrong output rather than an error, so decide before
your first render whether to mint at 512 and let the still lane handle framing.

## Verify the repo is UNGATED before promising it in a dropdown

`stabilityai/*` has historically required accepting terms on the Hub. A gated
repo breaks the auto-install property the pack depends on -- see the engine
selection criteria in `MAC_LESSONS_LEARNED.md` section 9.
`Comfy-Org/stable-diffusion-v1-5-archive` (2.13 GB, ordinary checkpoint loader,
no GGUF pack) is verified ungated.

## Declaring `device_backends` honestly

`device_backends` is **enforced before any work runs**, so a `["cuda"]` row
raises `EngineUnusable` on a Mac regardless of whether the code would have
worked. The 2026-09-07 Apple Silicon session found several rows that were
untested policy rather than measured fact:

* `bark` declared `["cuda", "cpu"]` and ran fine on `mps` once measured.
* `z_image_turbo`, `ltx_8gb` and both `animatediff15_v3_*` adapters declare
  `["cuda"]` and contain **zero** NVIDIA-specific code -- `z_image_turbo` was
  then observed loading and executing on Metal, and failing on RAM alone.

**So: declare what you have MEASURED.** If you have not tried a backend, say so
in a comment rather than excluding it silently -- an omitted backend is
indistinguishable from a tested-and-failed one, and the next person cannot tell
which they are looking at. A genuine NVIDIA dependency (nvenc, NVML, triton,
flash-attn, bitsandbytes CUDA kernels, an `nvfp4` artifact) is a real reason;
"never tried it" is not.

### The 30-second test that settles it

Before believing a `["cuda"]` row, grep the adapter:

```bash
grep -cE 'nvenc|nvml|triton|flash_attn|torch\.cuda|\.cuda\(\)|sm_[0-9]|nvfp4' <adapter>.py
```

**Zero hits means the row is probably untested policy, not a hardware fact.**
That single check was right three times on 2026-09-08:

| engine | grep | outcome |
| --- | --- | --- |
| `ltx_8gb` | 0 hits | **runs on Metal.** Published a full episode; the row was wrong |
| `eng_ghost_signal*` (animatediff) | 0 hits | under test |
| `bark` | had a literal `cuda if available else cpu` | **runs on mps** once given the chance |

And the counter-example that keeps the rule honest -- `z_image_turbo` is
genuinely unusable on a Mac, but **not for the reason its row implies**: bf16 is
simply too large for 16 GB, and its int8 variant dies on
`aten::_int_mm`, which PyTorch's MPS backend does not implement. Real limits
exist; they are just rarely the ones a bare `["cuda"]` is standing in for.

**A row that was never tested should say so:**

```python
# device_backends: ["cuda"] -- mps UNTESTED, no hardware available. The adapter
# contains no NVIDIA-specific code, so this row may simply be untried.
```
