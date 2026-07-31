# WAN 8-GB (`wan_ti2v`) -- parameter analysis for the low-VRAM tier

Operator question, 2026-07-31: *"methodically analyze what the parameters should
be for the wan 8gb video engine -- think just like the other engines. What are
the inputs? Recommended steps or temperatures? What is the minimum frame count,
max frame count, what's safe in those ranges on an 8 GB generic video card? So
we give our 8 GB VRAM friends a good experience."*

Every number below is read from the real tree at HEAD `6da72c92`. Where a number
is a GUESS rather than a measurement, it says so.

---

## THE HEADLINE: on a real 8 GB card this tier cannot render at all today

`compute_real_frame_budget` (`motion_common.py:323-379`) predicts:

    required_free_vram = (overhead + per_frame_at_res * frames) / margin

with `FRAME_COST_MODEL["wan_ti2v"] = (7000.0, 185.0)` (`:263-265`) and
`_BUDGET_MARGIN = 0.85` (`:279`).

At the intended 832x480 the pixel ratio to the cost reference is
`399,360 / 1,224,704 = 0.3261`, so `per_frame_at_res = 60.33 MB`:

| canvas | frames | required FREE VRAM |
|---|---:|---:|
| 832x480 | 17 (the motion floor) | **9,442 MB** |
| 832x480 | 1 (arithmetic floor) | **8,306 MB** |
| 1472x832 (the ACTUAL default -- see below) | 17 | **11,935 MB** |

**An 8 GB card has 8,192 MB TOTAL, so free is always less.** The overhead term
alone is `7000 / 0.85 = 8,235 MB`. The engine raises `MotionBudgetError`
unconditionally before it renders a single frame. This is not a theory -- it is
the recorded 2026-07-23 production failure, quoted in the engine's own source
(`eng_wan_ti2v.py:713-716`).

So "make WAN 8-GB ready" is not a ceiling question. The ceiling was the symptom.

---

## THE THREE REAL DEFECTS, in priority order

### 1. The engine does not declare its canvas -- it renders at 3.07x the pixels

`WanTi2vEngine` has **no `render_canvas` attribute**. `ltx_8gb` declares
`render_canvas = (512, 288)` statically (`eng_ltx_8gb.py:518`); WAN declares
nothing, so `render_driver.py:2494` falls through to:

    _lc = os.environ.get("OTR_VIDEO_LANDSCAPE_CANVAS", "1472x832")

There is no `wan_ti2v` branch (branches exist for `ltx_video` and
`ltx_audio_in`). The profile's `render.canvas_w/h = 832x480` is read by NOTHING
on the render path, and `launch.env.OTR_VIDEO_LANDSCAPE_CANVAS` only binds if the
server was booted with it -- the dead channel `PBUG-20260723-02` is about.

**A plain canonical WAN run renders at 1,224,704 pixels instead of 399,360.**
That is 3.07x the per-frame activation cost. This is the single largest defect
and it is a one-line class of fix -- the same fix ltx already shipped in B5.

### 2. The text encoder is bigger than the UNET, and WAN has no lever for it

On-disk sizes of the three frozen defaults:

| file | GiB |
|---|---:|
| `Wan2.2-TI2V-5B-Q5_K_M.gguf` (UNET) | 3.549 |
| `umt5-xxl-encoder-Q5_K_M.gguf` (text encoder) | **3.861** |
| `wan2.2_vae.safetensors` | 1.313 |
| **total if co-resident** | **8.722** |

The text encoder is the LARGEST single file, and all three together already
exceed an 8 GB card. WAN's only mitigation is ordering (`free_after_use=True`,
and the beat hoist deliberately excludes CLIP so it does not pin resident --
`eng_wan_ti2v.py:352-358`). Peak co-resident weights are then UNET + VAE
= 4.86 GiB, leaving ~3.1 GiB for activations, fragmentation and CUDA context.

**`ltx_8gb` MEASURED that this is the decisive knob**: with `t5_device` on GPU
the peak lands at 16.0-16.1 GB on a 16.3 GB card, i.e. *"an 8 GB box does not
render at all"*. Its recipe therefore pins `t5_device: "cpu"`.

**WAN has no `t5_device` knob at all.** Adding one is the highest-value change
available and is a straight port of a proven, measured decision.

### 3. The cost model is a single 2026-06 data point taken on a 16 GB card

The entire `(7000.0, 185.0)` row traces to ONE comment
(`motion_common.py:253-254`): *"wan_ti2v render-phase peak 10277 MB @ 17 frames
@ 1472x832"*. No canvas sweep, no length sweep, no tiled/untiled comparison, no
clamped run, and `VramPeakProbe` samples MACHINE-WIDE NVML -- so 10,277 MB
includes whatever else was resident on the 5080 that day.

Note the internal disagreement nobody has explained: the model charges **7,000 MB
overhead** for a path whose peak resident weights are ~4,980 MB. The ~2 GB gap
has no recorded provenance. If the true overhead is nearer 5,000 MB, an 8 GB card
becomes arithmetically possible; if it is really 7,000 MB, this tier is
misnamed. **We do not currently know which.**

---

## WHAT THE PARAMETERS SHOULD BE

### Inputs (what an operator actually sets)

The engine takes an `init_image` (`required_inputs = ("init_image",)`) and is
driven by the frozen recipe plus three live channels: the render canvas, the
frame ceiling (`video.max_render_frames` -> ledger -> `profile_max_render_frames()`),
and the per-beat target frame count from coverage planning. Everything else is
frozen and only movable under the consent act `OTR_WAN_TI2V_PREQUALIFICATION`.

### Canvas -- RECOMMEND 768x432, not 832x480

| candidate | pixels | aspect | /16 clean | note |
|---|---:|---|---|---|
| 1472x832 (today's actual) | 1,224,704 | 23:13 | yes | 3.07x cost. Wrong for this tier. |
| 832x480 (documented intent) | 399,360 | **26:15** | yes | Not 16:9 -- pillarboxes, exactly the objection ltx raised at `eng_ltx_8gb.py:508-509`. |
| **768x432 (RECOMMENDED)** | **331,776** | **exactly 16:9** | yes (48x27) | **17% cheaper than 832x480 AND geometrically correct.** |

768x432 drops `per_frame_at_res` from 60.33 MB to **50.12 MB**. It is the only
candidate that is both cheaper and correctly shaped. Declare it STATICALLY on the
engine as `render_canvas`, and pin `render.canvas_w/h` equal to it with a test,
exactly as ltx does -- so it can never drift back to an env-only channel.

### Frame range

- **Minimum: 17. Keep it.** `_TI2V_MIN_FRAMES = _TI2V_DEFAULT_FRAMES = 17`,
  matching `FRAME_MOTION_FLOOR["wan_ti2v"] = 17`. That is 0.68 s at 25 fps and
  it is the motion floor -- below it the clip reads as a stutter, and the
  ping-pong mirror-fills the rest of the beat. 17 = 4*4+1 is the smallest
  sensible 4n+1 rung above a still.
- **Maximum: 177 is the ENGINE limit and is fine as an absolute. The TIER
  ceiling should be raised from 17 to 33 -- IF the fit is proven.** At 768x432
  with a 5,000 MB overhead, an 8 GB card allows
  `(8192*0.85 - 5000) / 50.12 = 33 frames`. 33 = 4*8+1 is a legal rung and gives
  1.32 s of real motion instead of 0.68 s -- a visibly better experience, with
  half as much mirror-fill. **Do not ship 33 until it is measured**; ship 17
  until then.
- **The 4n+1 rule is not negotiable** -- the Wan VAE compresses 4 frames to 1
  latent. Enforced by `quantize_frames_4n1` on the single-clip path and by
  `FrameContract.is_legal_length` on the planned path (which REFUSES rather than
  snapping -- correct, leave it).

### Sampler settings

| param | frozen | verdict |
|---|---|---|
| `steps` | 30 | **A TIME knob, not a VRAM knob.** It does not change the fit. ltx runs 8 because it is a distilled model; Wan 2.2 TI2V-5B is not distilled and 30 is a defensible quality point. Leave at 30 for now; sweep 20/25/30 for wall-clock later. Lowering it will NOT make the tier fit. |
| `cfg` | 5.0 | Correct for a non-distilled Wan. (ltx's 1.0 is a distilled-model artifact -- do not copy it.) |
| `shift` | 5.0 | Correct for the 5B (the 14B uses 8.0). Leave. |
| `sampler` | `euler` | Leave. Note `_PORTABLE_SAMPLERS` is a whitelist of ONE, which makes an opposing-override test structurally impossible -- widen the whitelist before any sampler sweep. |
| `scheduler` | `simple` | Leave. |
| `negative` | the shared default | Leave. |

### VAE / decode -- the second-biggest lever

| param | frozen | verdict |
|---|---|---|
| `tiled_vae` | `True` | **Keep, but it is INHERITED, not measured.** ltx's sweep is the evidence that matters: tiled decode flattened its peak to 8241-8278 MB across 17-161 frames, versus 8662-10859 MB untiled -- for ~zero wall clock (824 s vs 842 s). *"An 8 GB tier needs a ceiling a long beat cannot grow through."* WAN almost certainly benefits the same way; it has simply never been checked. |
| `vae_tile` | 256 | ltx measured **512**. WAN's 256 has no recorded reason. Smaller tile = lower peak, more overhead. For an 8 GB tier 256 is the safer default -- **keep it and sweep 256 vs 512.** |
| `vae_overlap` / `vae_temporal` / `vae_temporal_overlap` | 64 / 16 / 8 | Identical to ltx's measured values. Leave. |

### dtype

Profile says `fp8_ok`; ltx's 8 GB tier says `no_fp8_no_fp4`. The WAN weights are
already GGUF Q5_K_M, so fp8 has little left to give and adds a numerical-stability
risk on a tier that cannot afford a retry. **Recommend aligning to
`no_fp8_no_fp4`** unless a sweep shows fp8 buys real headroom.

---

## THE SWEEP THAT SETTLES IT

Copy ltx's B6 shape exactly -- it is the only WAN-adjacent method in this repo
that produced trustworthy numbers. Four cells, every cell a full canonical leg
(`RESULT SUCCESS` + `obs_publish OK` + asset on disk), at the declared canvas:

| cell | `t5_device` | `tiled_vae` |
|---|---|---|
| A | cpu | off |
| B | gpu | off |
| C | cpu | ON |
| D | gpu | ON |

**Judge on the SPREAD, not the minimum** -- that was ltx's decisive column. Then
re-fit `FRAME_COST_MODEL["wan_ti2v"]` from the real `(frames, peak_mb)` pairs
instead of the single 2026-06 point, and set the tier ceiling from the re-fit.

**Run it CLAMPED** (`OTR_HEADLESS_RESERVE_VRAM_GB=8`). `VramPeakProbe` is
machine-wide, so an unclamped sweep ranks recipes but proves nothing about 8 GB
fit -- the exact honest limit already recorded for ltx.

---

## RECOMMENDED ORDER

1. **Declare `render_canvas = (768, 432)`** on the engine + pin the profile equal
   to it with a test. Without this every other number is measured against the
   wrong canvas. Fixes the live 3.07x defect.
2. **Add a `t5_device` knob** to the WAN recipe, default `"cpu"` for this tier.
   Straight port of ltx's measured decision, and the largest single lever.
3. **Run the 4-cell clamped sweep** and re-fit the cost model from real pairs.
4. **Set the tier ceiling from the re-fit** (17 stays until the data says 33).
5. Align `dtype_policy` if the sweep supports it.

Steps 1 and 2 are code and can land offline. Steps 3-4 need the GPU.

**Answering the original ownership question:** once the engine declares its own
canvas and its own `t5_device`, the tier's identity lives in the ADAPTER, which
is where the per-adapter-ownership doctrine wants it, and the
`config/profiles/*.json` channel can retire without taking the 8 GB tier with
it. `max_render_frames` stays an operator OVERRIDE on top of an
adapter-declared default -- widget `0` should mean "use the adapter's contract",
never "unlimited".

**One inconsistency to fix while in here:** `_cost_model_for`
(`motion_common.py:312-313`) silently swallows a malformed
`OTR_VIDEO_COST_OVERHEAD_MB` / `_PER_FRAME_MB`, which is the opposite failure
mode from `wan_recipe.config_number`'s fail-closed parsing three files away.
