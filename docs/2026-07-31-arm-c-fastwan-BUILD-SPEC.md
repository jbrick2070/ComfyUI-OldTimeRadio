# ARM C (FastWan 5B) -- build spec, r1 judgment, and the EXECUTED result

**Status:** the blocking question is ANSWERED and the build is BLOCKED on a new,
narrower, PROVEN reason. Arm C stays CUT. An operator decision is owed on the
substrate before it can be re-opened.
**Written:** 2026-07-31 at HEAD `4872b1f6`. **Superseded/extended:** 2026-07-31
at HEAD `04ae4f0c` with the r1 judgment and the load probe result.
**Owns:** standing arm C up in `scripts/run_video_arm_bakeoff.py` so it produces
a render and a VRAM number under the same clamp as arms A / B-partial / D.
**Does NOT own:** shipping FastWan into an engine, the estimator refit, or the
8 GB tier's qualification.

---

## 0. WHAT CHANGED, IN ORDER

| # | Claim | Verdict |
|---|---|---|
| 1 | The 3-step DMD schedule is not expressible in ComfyUI | **FALSE.** Core `ManualSigmas` + `SamplerCustom` express it exactly. |
| 2 | Arm C needs `ModelSamplingSD3 {shift: 1.0}` | **FALSE.** `timestep()` is `sigma * 1000` and never reads `shift`. Keep 5.0. |
| 3 | q6_k is ~660 MB heavier and dead on arrival | **FALSE.** Decimal-GB vs binary-GiB error. Real delta 381.1 MiB against 599.8 MiB headroom. |
| 4 | cfg is an open question | **ANSWERED: cfg 1.0.** Section 3 below. |
| 5 | Adding arm C is "one ArmSpec row plus one graph file" | **FALSE.** Section 5. |
| 6 | The GGUF repack will load, because tensor keys match | **FALSE, AND THIS IS THE LIVE BLOCKER.** Section 4. |

Rows 1-3 and 5 are the r1 panel's work (Codex `gpt-5.6-sol` + Antigravity,
Claude anchor and sole judge); the full judgment log is local-only at
`kibitz-runs/2026-07-31-arm-c-fastwan/r1/final.md`. Rows 4 and 6 are this
document's own executed work.

## 1. WHY ARM C WAS CUT, AND WHY THAT REASON IS NOW WRONG

`run_video_arm_bakeoff.py` cut arm C with:

> Its licence PASSES (apache-2.0 at both levels) but no ComfyUI base graph exists
> at any priority -- the weights are Diffusers layout and the 3-step DMD schedule
> (timesteps 1000,757,522) is not ordinary `KSampler steps=3`.

**Both halves are false.** The weights half died when a GGUF repack appeared
(`Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF`, apache-2.0). The schedule half
died when core ComfyUI turned out to ship `ManualSigmas` (section 2). The cut
reason in the runner has been rewritten to the executed one, and
`tests/test_video_arm_bakeoff.py` now pins it so the superseded justification
cannot be re-inherited.

## 2. THE SCHEDULE -- COORDINATES SOLVED, TRANSITION *NOT* SOLVED

> **r2 CORRECTION (2026-07-31, codex `gpt-5.6-sol`).** This section originally
> read "SOLVED, IN STOCK CORE NODES" and it over-claimed. `ManualSigmas` fixes
> **where** the model is evaluated. It says nothing about **how the latent moves
> between evaluations**, and that is a separate contract this spec never
> checked. See section 2A. The node plan below is necessary and NOT sufficient.

FastWan is DMD-distilled: 3 steps at timesteps 1000, 757, 522, i.e. normalised
sigmas `[1.000, 0.757, 0.522]` then 0.

`KSampler{steps: 3, scheduler: simple}` cannot reach that. `simple_scheduler`
samples a 1000-entry table at roughly `1.000, 0.667, 0.334, 0`, and
`ModelSamplingSD3(shift)` warps it by `s' = shift*s / (1 + (shift-1)*s)`.
Solving the first interior step for the target gives `shift = 1.556`, and that
same shift puts the second interior step at `0.437` against a target of `0.522`
-- one degree of freedom against two constraints. (Figures are discretized
approximations; the conclusion does not depend on the rounding.)

Core ComfyUI ships `ManualSigmas` in `comfy_extras/nodes_custom_sampler.py`
(category `model/sampling/sigmas`, one multiline-off String input, SIGMAS out,
parsed with `re.findall` into a FloatTensor). So the literal
`"1.0, 0.757, 0.522, 0.0"` IS the schedule. `SamplerCustom` in the same file
takes `model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas,
latent_image` and its slot 0 is a drop-in for the LATENT node `10` consumes.
It has no `denoise` input, which is correct here: the sigma list is the schedule.

**Node `2` stays at `ModelSamplingSD3 {shift: 5.0}`.** `comfy/model_sampling.py`,
`ModelSamplingDiscreteFlow`:

    def timestep(self, sigma):
        return sigma * self.multiplier          # multiplier = 1000, no shift
    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

`shift` appears only in schedule GENERATION and in `percent_to_sigma`. The
sigma -> timestep path used during inference ignores it, so explicit sigmas
yield timesteps `[1000, 757, 522, 0]` regardless. Independently corroborated by
the official `scheduler/scheduler_config.json`, which sets `flow_shift: 5.0`
alongside `UniPCMultistepScheduler`, `flow_prediction`, `num_train_timesteps
1000` and `final_sigmas_type: zero`, and by `FastWan2_2_TI2V_5B_Config` in
FastVideo, which sets `flow_shift = 5.0` and
`dmd_denoising_steps = [1000, 757, 522]`.

In-tree precedent: `nodes/_otr_video_engines/eng_ltx_av.py:299-312` already
ships a `manual_sigmas` recipe switch injecting a fixed distilled ladder. Arm C
is the same shape applied to Wan.

Planned replacement for node `9`, three nodes for one:

    "9a": KSamplerSelect { sampler_name: "euler" }                 -> SAMPLER
    "9b": ManualSigmas   { sigmas: "1.0, 0.757, 0.522, 0.0" }      -> SIGMAS
    "9" : SamplerCustom  { model ["2",0], add_noise true,
                           noise_seed 42, cfg 1.0,
                           positive ["4",0], negative ["5",0],
                           sampler ["9a",0], sigmas ["9b",0],
                           latent_image ["13",0] }                 -> LATENT

`ManualSigmas` is flagged `is_experimental=True`. Contain that by pinning the
ComfyUI revision and validating the live `/object_info` schema before submit --
NOT by taking a KJNodes dependency for a node core already provides.

## 2A. THE TRANSITION -- NO STOCK SAMPLER REPRODUCES DMD (r2, CONFIRMED)

DMD's multi-step loop is a **full restart**, not an ODE march. FastVideo
`DmdDenoisingStage.forward`: predict x0, then for every non-terminal step draw
**fresh** noise and re-noise x0 to the NEXT timestep --

    latents = self.scheduler.add_noise(pred_video, torch.randn(...), next_timestep)

with **zero carry-over of the previous latent**. Against
`comfy/k_diffusion/sampling.py`:

- **`sample_euler` (:189)** -- `d = to_d(x, sigma_hat, denoised); x = x + d*dt`.
  Deterministic. Its only noise injection sits behind `s_churn > 0`, and the
  default is 0. **No restart at all.**
- **`sample_euler_ancestral` (:215)** dispatches to `sample_euler_ancestral_RF`
  (:239) for `CONST` model sampling, which `ModelSamplingDiscreteFlow` is. At
  eta=1 it gives `sigma_down = sigma_{i+1}^2 / sigma_i` and
  `x = (sigma_down/sigma_i)*x + (1 - sigma_down/sigma_i)*denoised` plus a
  partial re-noise -- it **retains a fraction of the previous x**, reaching
  `x = denoised` only at the terminal `sigmas[i+1] == 0`.

So the correct sigma coordinates fed to `euler` produce a DIFFERENT trajectory
than the reference. This is the exact failure class the bench forbids: it does
not error, it renders something plausible and hands back a VRAM number for a
recipe nobody ran.

**Requirement.** Arm C's sampler must be PROVED, not assumed -- either a
bench-helper SAMPLER implementing the reference transition exactly (owning
generator device, seed and draw order, and logging its own evaluated timesteps
and transition mode), or documented numerical parity against the pinned
reference implementation. The vendored `otr_bakeoff_helper` is already
SHA-pinned and installed by the runner, so a helper SAMPLER is admissible under
the s0A carve-out without touching an engine module.

**Corollary that may decide the substrate.** This requirement binds a DMD
recipe. If the C-2 candidate's reference recipe is a plain deterministic
few-step Euler at cfg 1 -- common for step/CFG-distilled models -- then C-2 is
expressible in stock nodes and C-1 is not. That asymmetry is an engineering
reason of a different order than anything in section 6, and settling it is the
next action.

## 3. CFG -- PINNED AT 1.0 (the r1 open question, now closed)

The official README script passes no `--guidance-scale`, so the answer lives in
the FastVideo repo. Read from the published package (`fastvideo` 0.2.0, the
current PyPI release):

1. **`SamplingParam.guidance_scale` defaults to `1.0`**
   (`fastvideo/api/sampling_param.py:86`), and the `--guidance-scale` CLI flag
   defaults to that same field. A script that passes no flag runs at 1.0.
2. **The DMD path applies no classifier-free guidance at all.**
   `DmdDenoisingStage.forward` (`fastvideo/pipelines/stages/denoising.py:1124`
   onward) runs the transformer EXACTLY ONCE per timestep, on positive
   `prompt_embeds` only. There is no `do_classifier_free_guidance` branch, it
   never reads `batch.guidance_scale`, and it never consumes
   `negative_prompt_embeds`. Both `wan_dmd_pipeline.py` and
   `wan_i2v_dmd_pipeline.py` mount that exact stage.
3. **The routing is confirmed.** `fastvideo/registry.py:730-740` maps
   `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` to
   `FastWan2_2_TI2V_5B_Config` with `default_preset="fast_wan_2_2_ti2v_5b"`.

**A trap worth naming, because it points the other way and is wrong.** That
preset (`fastvideo/pipelines/basic/wan/presets.py`) declares
`guidance_scale: 5.0` and `num_inference_steps: 50`. Both are INERT on the DMD
path: the stage overrides its timesteps from
`pipeline_config.dmd_denoising_steps` and never reads `guidance_scale`. The
entry is a verbatim copy of the non-distilled `wan_2_2_ti2v_5b` preset --
50 steps is self-evidently not a 3-step contract. The sibling
`FAST_WAN_T2V_480P` preset carries `guidance_scale: 3.0` with
`num_inference_steps: 3`, inert for the same reason. **The code path is the
authority; the preset table is not.**

**Why cfg 1.0 is the faithful ComfyUI translation, not an approximation.**
`comfy/samplers.py`:

    def sampling_function(model, x, timestep, uncond, cond, cond_scale, ...):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None

At cfg 1.0 ComfyUI drops the uncond batch entirely and runs one forward pass on
the positive conditioning -- structurally the same thing `DmdDenoisingStage`
does. Any cfg > 1.0 would add a second forward pass upstream never performs.

**Consequence for the graph, and it is not cosmetic.** Node `5` (the negative
`CLIPTextEncode`) stays wired, as a same-graph control, and is documented as
EXECUTING but SAMPLER-INERT. That matches upstream, which also encodes a real
negative prompt the DMD stage never consumes. Arm C is therefore a RECIPE
candidate -- model + distilled schedule + guidance contract -- not a
model-only swap, and the original section 2's "differs only in model and
schedule" was too narrow.

## 4. THE LIVE BLOCKER -- THE REPACK DOES NOT LOAD

r1 accepted Antigravity's SHOULD-FIX that nothing yet proved `UnetLoaderGGUF`
could key-map a third-party repack, and made a load-only probe the first
action. **It fired.** Driving the real production path --
`gguf_sd_loader` -> `comfy.sd.load_diffusion_model_state_dict` with `GGMLOps`,
the exact code `UnetLoaderGGUF.load_unet` runs -- against
`C:\ComfyUI-Models\diffusion_models\FastWan2.2-TI2V-5B-q6_k.gguf`
(4,210,247,200 bytes, sha256
`416A87E30F2328DBEFD7666AC90B395EAD74F443748FF31C83483AC4AC6121CC`):

    INCUMBENT Wan2.2-TI2V-5B-Q5_K_M.gguf   modulation ranks [3]   -> LOADED OK
    CANDIDATE FastWan2.2-TI2V-5B-q6_k.gguf modulation ranks [2]   -> RAISED

Three defects, all structural:

1. **Zero kv fields.** The repack carries NO `general.architecture` (the
   incumbent carries `wan`) and no `comfy.gguf.orig_shape.*` entries.
   ComfyUI-GGUF therefore takes the `arch_str is None` branch, logs
   `loaded in compatibility mode 'sd.cpp'`, and recovers the arch only by
   `detect_arch` over the tensor names. That branch is a survivable warning, not
   the failure -- but it identifies the toolchain.
2. **`modulation` is rank-2.** With no `orig_shape` metadata the loader derives
   the shape from the GGUF dims, giving `(6, 3072)` for all 31
   `blocks.N.modulation` and `(2, 3072)` for `head.modulation`. ComfyUI declares
   `nn.Parameter(torch.empty(1, 6, dim))` at `comfy/ldm/wan/model.py:215`. The
   incumbent's dims reverse to `(1, 6, 3072)` and load clean.
3. **The norm vectors are the wrong length.** Every `self_attn`/`cross_attn`
   `norm_q`/`norm_k` weight arrives 2520-long against the model's 3072. The
   repack quantized 1-D norm tensors that ComfyUI-GGUF's converter leaves in
   F32/F16: its type histogram is 342 F32 + 483 Q6_K where the incumbent's is
   1 F32 + 524 F16 + 210 Q5_K + 90 Q6_K.

**The banked assumption that killed the estimate was "825 tensors each, ZERO key
differences, so UnetLoaderGGUF will map it."** The key parity is real. It is
also not sufficient: shapes, ranks and quant plan all differ, and the loader
needs those. **Key parity is not shape parity.** A cheap header diff -- 32 shape
mismatches and an absent arch field -- would have shown this before any GPU
time, and now does, in seconds.

The provenance caveat r1 recorded (~358 downloads at the time, 1.1K now; a
31-byte README with no conversion notes, against 366K downloads on the official
Diffusers weights) was the correct instinct. r1 framed the open question as
conversion IDENTITY versus arm D's shipping RIGHTS, and noted that a malformed
conversion can load, render plausibly, and still produce a valid VRAM number.
The real outcome is one notch better for us: it does not load at all, so there
was never a plausible-but-wrong number to guard against.

## 5. "ONE ARMSPEC ROW PLUS ONE GRAPH FILE" IS FALSE

`_WAN_CLASSES` contains `KSampler` and omits `KSamplerSelect`, `ManualSigmas`
and `SamplerCustom`, and `offline_preflight` rejects any graph using a class the
ArmSpec does not declare. The real change set for arm C is: a new graph file, a
new required-class tuple, an `ArmSpec` row, a graph SHA pin, an `ARM_LICENCE`
row, the arm-C-absence tests inverted, recipe-contract tests, and the cut reason
rewritten -- one commit.

`docs/GO_FORWARD_PLAN.md` NEXT item 4 has been softened accordingly: data entry
holds for a candidate reusing an existing recipe shape, not for one bringing a
new sampling contract, and a file on disk is not an admitted candidate until it
load-probes.

## 6. WHAT ARM C NEEDS NEXT -- AN OPERATOR DECISION

cfg is pinned; the schedule's COORDINATES are solved and its TRANSITION is not
(section 2A). Two candidates were acquired and **LOAD-PROBED**, both apache-2.0
on the quant repo.

> **Header verification does not count, by operator directive 2026-07-31.**
> Header verification is exactly what gave the Green-Sky repack a false pass:
> 825 tensor keys, zero differences, and it does not load. Every candidate below
> was driven through the real path -- `gguf_sd_loader` ->
> `comfy.sd.load_diffusion_model_state_dict` with `GGMLOps`, i.e. what
> `UnetLoaderGGUF.load_unet` runs -- and survived it. A name-only or header-only
> check is not a load probe.

### Load-probe results (2026-07-31)

| | incumbent | C-2 Turbo Q5_K_M | C-1 FastWan LoRA |
|---|---|---|---|
| bytes on disk | 3,810,603,360 | 3,815,414,496 | 660,874,456 |
| state-dict entries | 825 | **825** | 1099 tensors |
| `blocks.0.modulation` | (1, 6, 3072) | **(1, 6, 3072)** | n/a |
| `self_attn.norm_q.weight` | (3072,) | **(3072,)** | n/a |
| result | LOADED OK | **LOADED OK** | **793 patched, 0 unmatched** |

C-2 matches the incumbent in every dimension that killed the repack, measured on
the state dict BEFORE `load_diffusion_model_state_dict` consumes it. C-1's LoRA
resolves 306 low-rank pairs + 487 dense diffs = 793 keys through core
`load_lora_for_models` with **zero** unmatched, so its `diffusion_model.`
namespace is correct and BUG-070's KJ-wrapper exclusion is not triggered.
"No exception raised" is not the pass condition -- `load_lora_for_models` warns
on unmatched keys and still returns a model. The pass condition is the coverage
invariant: 793 patched, 0 unmatched.

### The LoRA VRAM question, measured -- and the earlier claim withdrawn

A matched control (separate processes, identical path, `patch_on_device=False`):

| | incumbent only | + FastWan LoRA | delta |
|---|---|---|---|
| GGML layers carrying patches | 0 | 300 | |
| NVML delta after load | 3954.0 MiB | 3968.0 MiB | **+14.0 MiB** |
| torch peak | 3634.1 MiB | 3647.2 MiB | +13.1 MiB |

**WITHDRAWN:** any claim that the LoRA's 630 MiB goes resident and consumes the
599.8 MiB headroom. It does not -- `move_patch_to_device` keeps patch data off
the device until needed, and the measured resident cost is +14.0 MiB.

**CONFIRMED by execution:** the patches do NOT fold once at load. After a device
load, `get_weight` on `diffusion_model.blocks.0.self_attn.q` (stored
`tensor_type=13` Q5_K, `dtype=uint8`) returns `float16 (3072, 3072)` -- the
patch is applied to a DEQUANTIZED tensor inside the on-the-fly dequant path
(`ComfyUI-GGUF/ops.py:166-190`), per call. So C-1 carries a SAMPLING-TIME term
that a load measurement cannot see and that lands on the whole-window NVML peak
this bench grades. It cannot be settled by argument in either direction. **One
clamped 17-frame render is the gate**, recording NVML, torch
allocated/reserved, system RAM and pagefile.

### The candidates

**C-1. FastWan as a LoRA over the incumbent.**
`Kijai/WanVideo_comfy/FastWan/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors`,
660,874,456 bytes. Keeps arm C = FastWan exactly as briefed, and keeps every
finding in sections 2 and 3 (sigmas, shift 5.0, cfg 1.0) intact. The base
weights stay BIT-IDENTICAL to arm A, which is the cleanest single-axis read this
bench can produce. Costs one extra `LoraLoaderModelOnly` node, and a
LoRA-patching VRAM profile that is not how a distilled model would ship --
that partly confounds "what does this model cost". Kijai's WanVideo_comfy is the
canonical ComfyUI Wan repo. The same repo also holds the merged
`Wan2_2-TI2V-5B-FastWanFullAttn_bf16.safetensors` (9.3 GiB), which is the proper
INPUT to ComfyUI-GGUF's own `tools/convert.py` if we ever want to make the
Q5_K_M ourselves and settle conversion identity permanently.

**C-2. A different distillation that is already a clean GGUF.**
`hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF`, `Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf`,
3,815,414,496 bytes. LOAD-PROBED (above), and additionally header-consistent:
`general.architecture = wan`, `blocks.0.modulation` -> `(1, 6, 3072)`,
`head.modulation` -> `(1, 2, 3072)`, norms unquantized, type histogram
519 F32 + 6 F16 + 210 Q5_K + 90 Q6_K -- structurally the incumbent's twin. **It is +4.6 MiB against the incumbent, so
the headroom question disappears entirely** (versus +381.1 MiB for q6_k against
599.8 MiB of measured headroom). 36.4K downloads. But it is Turbo, not FastWan:
a different distillation on `quanhaol/Wan2.2-TI2V-5B-Turbo`, 4 steps at cfg 1,
whose upstream licence this project has already flagged as UNSTATED. Its recipe
would have to be pinned from scratch -- none of section 3 transfers.

**NO RECOMMENDATION YET, and the earlier one is withdrawn.** This section
originally recommended C-1 on the grounds that "it is what the operator asked
for". That is deference, not engineering, and r2 dissolved the technical case on
both sides of it: the LoRA headroom objection is refuted (+14.0 MiB), and the
single-axis claim is weakened by the compute-time patch term rather than by
residency.

**The question is no longer "which model file".** Section 2A changed it to
"which recipe can this bench express honestly, and at what measured cost". Two
gates decide it, in order:

1. **Which candidate's reference TRANSITION is expressible?** C-1 is DMD and
   needs a full-restart sampler no stock ComfyUI sampler provides. If C-2's
   reference recipe is a plain deterministic few-step Euler at cfg 1, C-2 needs
   no custom sampler and C-1 does. Settle it from the `quanhaol` source the same
   way cfg was settled -- from the code path, not the model card.
2. **What does each actually cost under the clamp?** One 17-frame clamped render
   per surviving candidate. For C-1 that is the only instrument that prices the
   per-forward patch term; for either it is the first honest `peak_delta_mib`.

Standing costs, unchanged by r2:
- C-1 is FastWan and keeps sections 2/3 intact; C-2 is a DIFFERENT distillation
  (`quanhaol/Wan2.2-TI2V-5B-Turbo`, 4 steps at cfg 1) whose upstream licence
  this project has already flagged UNSTATED, so it must carry
  `shippable: None` in `ARM_LICENCE` exactly like arm D.
- C-2 is +4.6 MiB against the incumbent, so its headroom question is settled
  before it starts; C-1's is not, pending gate 2.
- Kijai's merged `Wan2_2-TI2V-5B-FastWanFullAttn_bf16.safetensors` (9.3 GiB)
  through `ComfyUI-GGUF/tools/convert.py` remains the named FALLBACK if gate 2
  shows C-1 thrashing: it removes the patch term entirely by folding the
  distillation into the weights. Note `tools/convert.py:160` asserts
  reference-vs-Diffusers layout and Wan's 5-D `patch_embedding` needs the
  `fix_5d_tensors` path, so it is not an afternoon.

Until both gates are answered, **arm C stays CUT with the section 4 reason**,
and no `ArmSpec`, graph file or licence row is written for it -- a blocked arm
must not add dead branching (spec 1).

## 7. CONSTRAINTS THAT DO NOT BEND

- **No engine module is touched.** Arm C is an isolated bench graph under the
  CLAUDE.md s0A carve-out (O6), pinned by SHA-256 in the campaign manifest, same
  as arms A / B-partial / D. `wan_ti2v` stays sacred.
- **No fallbacks, no silent degrade.** If the schedule cannot be expressed, the
  arm refuses to run rather than running approximately and reporting a number.
  When arm C is built, `offline_preflight` gains an arm-owned recipe contract
  validating the sigma literal, the sigma count, the sampler and cfg OFFLINE,
  plus a SIGMAS passthrough probe asserting `[1000, 757, 522, 0]` into the
  server log. Without that probe the server-log claim comes out of the spec.
- **The bench grades on `peak_delta_mib`** against each cell's own desktop
  baseline, bar 7168 MiB = 8192 - 1024 display allowance. Arm C changes nothing
  about grading.
- **An automated visual discriminator was CUT, unanimously.** Schedule
  correctness is deterministic telemetry, not a picture. Visual review stays a
  separate quality gate, never the schedule gate.
- Licence: apache-2.0 at both levels for FastWan itself. Do not re-litigate it.
  Conversion IDENTITY is a separate question from RIGHTS, and it is the one that
  bit here.
