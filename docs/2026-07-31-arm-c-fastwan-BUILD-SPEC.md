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
| 7 | `ManualSigmas` expresses the DMD schedule | **FALSE.** It fixes the evaluation coordinates, not the transition. Section 2A. |
| 8 | C-2 (Turbo) is a plain few-step Euler, so it needs no custom sampler | **FALSE.** It is Self-Forcing/DMD with the same restart loop, plus a warp step. Section 2B. |
| 9 | C-2's licence is merely "unstated" | **FALSE, AND WORSE.** The author publishes **CC BY-NC-SA 4.0** (NonCommercial) in the GitHub `LICENSE.md`; the HF weights repo grants nothing. C-2 is SHIPPING-BLOCKED. Section 6A. |

Rows 1-3 and 5 are the r1 panel's work; row 7 is r2's (codex MUST-FIX 1). Panel
= Codex `gpt-5.6-sol` + Antigravity, Claude anchor and sole judge throughout;
judgment logs are local-only at
`kibitz-runs/2026-07-31-arm-c-fastwan/{r1,r2}/final.md`. Rows 4, 6 and 8 are
this document's own executed work.

**Note that row 1 and row 7 are the same mistake at two scales, made by me both
times: check the part that is easy to check, then write the conclusion for the
part you did not check.** Row 6 is a third instance from the other direction
(tensor NAMES compared, shape parity concluded). The standing correction is in
section 2A and in `docs/GO_FORWARD_PLAN.md` NEXT item 4.

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

**Corollary that might have decided the substrate -- CHECKED, AND IT DOES NOT.**
The hope was that C-2's reference recipe would be a plain deterministic few-step
Euler at cfg 1, making C-2 expressible in stock nodes while C-1 is not. **It is
not.** See section 2B.

## 2B. GATE 1 ANSWERED -- BOTH CANDIDATES ARE RESTART SAMPLERS

Pinned from the reference code path, not the model card and not community
practice.

`quanhaol/Wan2.2-TI2V-5B-Turbo` is "efficient step distillation and CFG
distillation ... **leveraging the Self-Forcing framework**", trained by
`running_scripts/train/Wan2.2/dmd.sh`. Its shipped inference config
`configs/inference/wan22.yaml` is:

    warp_denoising_step: true
    model_name: Wan2.2-TI2V-5B
    denoising_step_list: [1000, 750, 500, 250]

and its own `pipeline/wan22_fewstep_inference.py` loop is:

    _, pred_image_or_video = self.generator(...)
    if index < len(self.denoising_step_list) - 1:
        next_timestep = self.denoising_step_list[index + 1] * ...
        noisy_image_or_video = self.scheduler.add_noise(
            pred_image_or_video.flatten(0, 1),
            torch.randn_like(pred_image_or_video.flatten(0, 1)),
            next_timestep.flatten(0, 1)).unflatten(0, noise.shape[:2])

Predict x0, re-noise with **fresh** `randn_like` to the next timestep, zero
carry-over -- **the identical restart transition FastWan uses**, and the same
structure as the acknowledged upstream `guandeh17/Self-Forcing`
`pipeline/causal_inference.py`.

**Consequences, and one runs against C-2:**

1. **Neither candidate is expressible in stock ComfyUI nodes.** The custom
   SAMPLER of section 2A is required either way. Cheaper than it sounds: ONE
   helper implementation serves both, differing only in the timestep list.
2. **C-2 carries an EXTRA contract C-1 does not: `warp_denoising_step: true`.**
   The listed timesteps are not used raw -- they are warped through the
   scheduler's own timestep table (`timesteps[1000 - denoising_steps]` in the
   Self-Forcing lineage). So the naive `sigma = t/1000` mapping that is correct
   for FastWan is **wrong for Turbo**, and reproducing Turbo means reproducing
   the warp exactly.
3. Turbo is 4 steps `[1000, 750, 500, 250]`; FastWan is 3 steps
   `[1000, 757, 522]`.

**TRAP, named because it is the one that would tempt a shortcut.** The GGUF
repacker's README quotes community advice: "4 steps is enough. CFG 1, sampler is
Euler or SA_Solver or Uni_PC, scheduler is simple or normal or beta." That is
what people run and like -- it is NOT the reference recipe, and the reference
code above contradicts it. Plausible output is not recipe fidelity. Adopting the
civitai line would hand this bench a VRAM number for a recipe the model's
authors never ran, which is precisely the failure section 2A exists to prevent.

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

**GATE 1 -- which candidate's reference TRANSITION is expressible? ANSWERED
(section 2B): NEITHER.** Turbo is Self-Forcing/DMD with the identical
predict-x0-then-restart loop, so the custom SAMPLER is required either way and
one helper implementation serves both. This gate was expected to favour C-2 and
instead **runs mildly against it**: Turbo adds `warp_denoising_step: true`, an
extra contract C-1 does not carry, which invalidates the naive `sigma = t/1000`
mapping for Turbo alone.

**GATE 2 -- what does each actually cost under the clamp? OPEN.** One 17-frame
clamped render per surviving candidate, recording NVML, torch
allocated/reserved, system RAM and pagefile. For C-1 this is the only instrument
that prices the per-forward patch term; for either it is the first honest
`peak_delta_mib`.

**Where that leaves the balance.** C-1's two objections have both weakened: the
headroom claim is refuted (+14.0 MiB resident) and the sampler cost is now
shared rather than C-1-specific. C-2's two advantages have both narrowed: the
recipe is no longer cheaper to express, it is dearer by one warp contract, and
its remaining edge is the settled headroom (+4.6 MiB). **Gate 2 is now the only
thing that can disqualify C-1**, and it is a measurement, not an argument. Run
it before recommending anything -- I have already reversed this recommendation
once on reasoning that a measurement then refuted.

### 6A. LICENCE -- C-2 IS SHIPPING-BLOCKED. This is not a tiebreaker, it is a gate.

**Correction to this document's own earlier framing.** C-2 was filed as
"provenance best, licence worse", as though those were comparable axes on one
scale. They are not. A missing licence field is not a weaker grant than
apache-2.0; it is the ABSENCE of a grant. Default copyright reserves rights, and
a downstream repacker cannot convey rights the upstream never conveyed --
`hum-ma`'s apache-2.0 tag is a claim ABOUT `quanhaol`, not a licence FROM
`quanhaol`. Excellent file provenance with no licence chain is a candidate you
can MEASURE and cannot SHIP.

Chains traced to the source, file by file, not by tag (2026-08-01):

| level | C-1 (FastWan) | C-2 (Turbo) |
|---|---|---|
| base model | `Wan-AI/Wan2.2-TI2V-5B` **apache-2.0** | `Wan-AI/Wan2.2-TI2V-5B` **apache-2.0** |
| distillation framework | FastVideo's own DMD | `guandeh17/Self-Forcing` **Apache-2.0** |
| distilled model | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` **apache-2.0** (verified at the repo, not from a note) | `quanhaol/Wan2.2-TI2V-5B-Turbo` -- **HF weights repo states NOTHING**; GitHub repo's `LICENSE.md` is **CC BY-NC-SA 4.0** |
| ComfyUI conversion | `Kijai/WanVideo_comfy` -- no repo-level licence | `Kijai/WanVideo_comfy` -- no repo-level licence |
| GGUF repack | n/a | `hum-ma` tags **apache-2.0** |

**The decisive facts for C-2:**

1. The HF weights repo `quanhaol/Wan2.2-TI2V-5B-Turbo` is FOUR files --
   `.gitattributes`, `README.md`, `config.json`, `model.pt`. **No LICENSE file.
   No `license:` field in the card frontmatter. No licence section in the README
   body.** Checked the tree, not just the card.
2. The author's GitHub repo DOES publish terms, and they are
   **CC BY-NC-SA 4.0 -- Attribution-NonCommercial-ShareAlike**. Cards lie by
   omission; files usually do not.
3. That NonCommercial term is **`quanhaol`'s own choice, not inherited**: the
   acknowledged framework `guandeh17/Self-Forcing` is Apache-2.0 and the base is
   apache-2.0. Nothing upstream imposed NC. It is a deliberate, bespoke
   restriction by the model's author.
4. So `hum-ma`'s apache-2.0 tag does not merely lack a foundation -- it
   **contradicts the only terms the author actually published**.

Against the standing operator rule -- *MIT preferred, apache/BSD acceptable, no
bespoke or unstated terms* -- C-2 fails twice: the weights are unstated, and the
author's published terms are both bespoke and NonCommercial. **C-2 is
SHIPPING-BLOCKED, recorded here up front so that a gate-2 win cannot quietly
promote it.** If C-2 wins on measurement and still has no licence chain, that is
a finding for the research brief, NOT a candidate for the workflow. Its
`ARM_LICENCE` row is `shippable: False` -- not arm D's `None`, because arm D's
question is an ambiguous clause while C-2's is an explicit NC term plus an
ungranted artifact.

**C-1's chain, stated precisely** (correcting a loose earlier claim of
"apache-2.0 verified at both levels"): the MODEL is apache-2.0 at both levels
and that is now verified at the source -- `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`
carries `license:apache-2.0`, and so does the Wan base. The FILE we load-probed
is Kijai's rank-128 extraction, and `Kijai/WanVideo_comfy` states no repo-level
licence (its only LICENSE file is `LoRAs/Ditto/ditto_LICENSE.txt`, a per-artifact
one for an unrelated LoRA). The distinction that matters: for C-1 a grant EXISTS
upstream and apache-2.0 expressly permits preparing and redistributing
derivatives, so Kijai's silence is a notice-compliance gap on Kijai's side
rather than a missing grant. For C-2 there is no grant at the origin at all, so
nothing can flow. Every level of C-1 that states terms states apache-2.0; no
level states restrictive or bespoke terms.

*This is a policy determination against the operator's stated rule, not legal
advice.*

**Consequence for gate 2, and it is a simplification.** If C-2 can never be the
substrate, a gate-2 render on C-2 is no longer decision-relevant -- it would be
a research data point, not a choice between candidates. **The decision-relevant
gate 2 is C-1 alone: does FastWan-via-LoRA fit under the clamp?** Run C-2 only
if the operator wants the research number.

### 6B. Conversion identity -- separate question, and C-2 still scores well

Worth keeping distinct from rights, because conflating them is what produced the
"provenance best, licence worse" error above. `hum-ma` documents the whole
conversion chain: converted with `city96/ComfyUI-GGUF/tools` from Kijai's
`Wan22-Turbo/Wan2_2-TI2V-5B-Turbo_fp16.safetensors`, quantized, "and finally
fixed 5d tensors" -- named converter, named source, named post-step, which is
exactly what the Green-Sky 31-byte README lacked, and it is corroborated by the
load probe. Conversion IDENTITY gates admission to the comparative matrix;
shipping RIGHTS gate the workflow. C-2 passes the first and fails the second.

### 6C. Remaining standing facts

- C-2 is +4.6 MiB against the incumbent, so its headroom question is settled
  before it starts; C-1's is not, pending gate 2.
- Kijai's merged `Wan2_2-TI2V-5B-FastWanFullAttn_bf16.safetensors` (9.3 GiB)
  through `ComfyUI-GGUF/tools/convert.py` remains the named FALLBACK if gate 2
  shows C-1 thrashing: it removes the patch term entirely by folding the
  distillation into the weights. Note `tools/convert.py:160` asserts
  reference-vs-Diffusers layout and Wan's 5-D `patch_embedding` needs the
  `fix_5d_tensors` path, so it is not an afternoon. Its licence position is
  C-1's -- apache-2.0 at the model level -- so this fallback does not inherit
  C-2's problem.

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
