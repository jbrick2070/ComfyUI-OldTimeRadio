# LTX-AV talk lane -- de-blur: roundtable answer + panel request (one doc)

Generated 2026-06-17. Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro (live OpenRouter pass, ~$0.21). Claude = sole judge/grounder.

How to read this: PART 1 is the actionable answer; PART 2 shows what survived grounding; PART 3 is the unedited panel; PART 4 is the exact request so you can paste it into ChatGPT / Gemini / DeepSeek / Grok yourself.


======================================================================
# PART 1 -- SYNTHESIZED ANSWER (recommended settings)
======================================================================

# LTX-AV talk lane -- de-blur settings (roundtable pass01, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro (~$0.21). Claude grounded
every claim against `eng_ltx_av.py` + both mini JSONs. 4/4 converged on the root cause.

## Root cause (CONFIRMED, 4/4 + grounding)
The blur is the **non-distilled 8-step configuration**. The golden video mini is sharp
at 8 steps **only because it applies the distilled LoRA @0.70** (a few-step-distilled
model). The AV talk graph has **NO LoRA** (grounded: `_node_candidates()` has no
`LoraLoaderModelOnly`; `_build_graph` wires unet->ModelSamplingLTXV->guider) yet still
runs **8 steps of plain `euler` at cfg 3.0 with i2v strength 1.0 and scheduler
terminal 0.1** -> under-converged + over-guided -> soft. Contributing: i2v strength 1.0
(golden 0.75), terminal 0.1 (golden sigmas end at 0.0), and the 832x480 canvas that is
**over the 14.5 GB ceiling** (measured 15.75 GB).

## TIER 1 -- no-LoRA, no-code A/B ladder (do FIRST; edit the mini's widgets directly)
All at **512x288** (the ceiling-safe canvas; the standalone mini takes raw widget edits,
no engine change needed). Change ONE variable at a time, same seed/prompt/image/frames,
judge on RAW frames (see methodology). Ranked by expected impact:

1. **Canvas 832x480 -> 512x288.** MUST. 832x480 = 15.75 GB > cap; 512x288 ~13.7 GB. Edit
   `LTXVImgToVideo` width/height (and keep length). Everything below assumes 512x288.
2. **Steps 8 -> 24** (`LTXVScheduler.steps`, env `OTR_LTX_AV_STEPS`). The #1 fix without
   the LoRA. Cost: ~linear time (8->24 ~ 3x sampling time). A/B 12 / 16 / 24.
3. **i2v strength 1.0 -> 0.8** (`LTXVImgToVideo.strength`, env `OTR_LTX_AV_I2V_STRENGTH`).
   Golden uses 0.75; 1.0 likely costs portrait micro-detail. A/B 0.75 / 0.85. Watch:
   lower strength may reduce head/mouth motion -- judge sharpness AND motion separately.
4. **cfg 3.0 -> 2.0** (`CFGGuider.cfg`, env `OTR_LTX_AV_CFG`). 3.0 over-guides a
   non-distilled base. A/B 1.5 / 2.0 / 2.5. (Do NOT use cfg 1.0 here -- 1.0 only works
   with a cfg-distilled model/LoRA; without it 1.0 = no guidance.)
5. **terminal 0.1 -> 0.0** (`LTXVScheduler.terminal`). Golden sigmas end at 0.0 (full
   denoise); 0.1 leaves residual noise = softness.
6. **sampler `euler` -> `euler_cfg_pp`** (`KSamplerSelect.sampler_name`). Zero-cost; the
   golden uses it; often higher micro-contrast on LTX even without the LoRA.
7. (empirical) **max_shift 2.05 -> ~1.5** at 512x288 (`ModelSamplingLTXV` +
   `LTXVScheduler`). Flow-matching wants less shift at low res. VERIFY-AT-BUILD.
8. (low) **VAEDecodeTiled temporal_size 64 -> 128** only if VRAM allows; judge on raw
   frames. Do NOT jump to the golden's 4096 (decoding 105 frames at once OOMs). Spatial
   softness is more likely steps/cfg/strength than temporal tiling (GPT, grounded).

Recommended first single shot to eyeball: **512x288, steps 24, cfg 2.0, strength 0.8,
terminal 0.0, euler_cfg_pp** -- then peel back one knob at a time.

## TIER 2 -- distilled LoRA path (RISKY; test SECOND, on a tiny clip)
Matching the golden exactly (LoRA@0.70 + `euler_cfg_pp` + fixed `ManualSigmas` ending
0.0 + cfg 1.0 + strength 0.75) would recover the ~122 s speed AND sharpness -- IF it
transfers. **Central unverified risk (4/4):** the distilled LoRA was trained on the
video-only t2v/i2v transformer, NOT the **audio+video concat latent** that
`LTXVConcatAVLatent` feeds the unet. It may suppress the audio cross-attention (breaking
lip-sync) or mis-shape. Guards before adoption:
- **Test on a 9-frame A2V clip first**; confirm it renders AND lip-sync survives.
- If using `ManualSigmas`, **BYPASS `ModelSamplingLTXV`** (Gemini, grounded: golden has
  no ModelSamplingLTXV; the distilled sigmas are already shifted -> applying it too
  double-shifts -> blur).
- **License-check** `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (Apache/LTX-2
  Community only) before it enters the AV lane.
- Define a non-LoRA fallback = the Tier-1 winner.

## Methodology (so we measure blur, not artifacts)
- Judge sharpness on **raw decoded PNG frames** (or a face-crop Laplacian-variance),
  NOT the VP9/h264 outputs -- codec compression can masquerade as model blur.
- One variable per run; identical seed / prompt / image / canvas / frame count.
- The standalone **mini is the A/B vehicle** (direct widget edits = no code). Folding the
  winner into `eng_ltx_av.py` is a follow-on (sampler/scheduler/terminal/decode/LoRA are
  currently HARDCODED there -> needs new env vars `OTR_LTX_AV_SAMPLER` /
  `OTR_LTX_AV_TERMINAL` / `OTR_LTX_AV_SCHEDULER` / optional LoRA before the engine can run
  the matrix; steps/cfg/strength are ALREADY env-overridable).
- For LIP-SYNC eval (not blur), match video length to the audio (13.28 s line -> ~337
  frames @25fps, 8n+1) or trim the audio to the clip; for pure sharpness 105 frames is
  fine.

## Guards (reject anything that violates these)
- All tests at 512x288 stay <= 14.5 GB; never default the lane to native 1472x832.
- 100% local; no VHS/GPL, no RES4LYF/AGPL; LoRA license verified before use.
- Determinism (fixed seed); UTF-8/ASCII; SFW.

## VERIFY-AT-BUILD (unverifiable from code)
- i2v `strength` semantics direction (panel disagreed); just A/B 0.75-1.0 and look.
- resolution-aware shift values at 512x288.
- temporal_size effect on spatial sharpness.
- distilled-LoRA transfer to the A2V concat latent (the Tier-2 gating test).


======================================================================
# PART 2 -- JUDGMENT LOG (confirmed / rejected / unverifiable)
======================================================================

# Judgment log -- pass01 (Claude grounded each claim vs eng_ltx_av.py + both JSONs)

## CONFIRMED (code supports -> folded into pass01_plan.md)
- **832x480 over the 14.5 GB ceiling** -- measured 15.75 GB live; use 512x288. [all 4]
- **Blur = non-distilled 8-step config.** Grounded: AV `_build_graph` has no LoRA and
  runs 8 steps; the golden is sharp at 8 steps only via the distilled LoRA. [GPT/Gemini/DeepSeek]
- **i2v strength 1.0 vs golden 0.75** -- `eng_ltx_av.py:56` `_LTX_AV_I2V_STRENGTH="1.0"`. [all]
- **cfg 3.0 vs golden 1.0** -- `eng_ltx_av.py:55` (env-overridable `OTR_LTX_AV_CFG`). [GPT/Gemini]
- **terminal 0.1 vs golden sigmas end 0.0** -- `eng_ltx_av.py:295` `terminal:0.1`; golden
  `ManualSigmas` ends "...0.421875, 0.0". [DeepSeek]
- **ManualSigmas + ModelSamplingLTXV = double-shift.** Grounded: the golden JSON has NO
  `ModelSamplingLTXV` node; AV adds it -> if distilled sigmas are adopted later, bypass it. [Gemini]
- **Sampler/scheduler/shift/terminal/decode are HARDCODED in the engine**
  (`eng_ltx_av.py` lines 271, 293-305) -> engine A/B needs new env vars; steps/cfg/strength
  already env-overridable. The standalone mini sidesteps this (direct widget edits). [GPT/DeepSeek/Grok]
- **My "differs in exactly these ways" was overstated** -- the golden also differs in
  TOPOLOGY (LTXVImgToVideoConditionOnly vs LTXVImgToVideo, no audio concat, no
  ModelSamplingLTXV). Ablate one variable at a time. [GPT]
- **Audio/video duration mismatch** -- 105 frames (4.2 s) vs the 13.28 s line; fine for a
  sharpness A/B, but match length to audio for a real lip-sync eval. [GPT]
- **Judge on raw frames** -- VP9/h264 compression can masquerade as model blur. [GPT]

## MISREAD / REJECTED (cite why)
- Grok: "negative prompt is static, make it overridable" -- it ALREADY is:
  `eng_ltx_av.py:249` `OTR_LTX_AV_NEGATIVE`. Rejected.
- GPT: "`OTR_LTX_AV_RENDER_CANVAS` doesn't exist" -- partial misread: it exists in
  render_driver (the engine's `_render_dims` doesn't read it). For the mini, edit the
  width/height widgets directly. Downgraded to a note.
- DeepSeek/Grok: raise `temporal_size` toward the golden's 4096 -- would OOM decoding 105
  frames at once (GPT/Gemini, consistent with the measured VRAM). Capped at 128/256, judged
  on raw frames. Rejected the 4096 jump.

## UNVERIFIABLE (verify-at-build; NOT adopted as fact)
- **Distilled-LoRA transfer to the A2V concat latent** -- the Tier-2 gating test (tiny clip first).
- i2v `strength` mechanism direction (panel disagreed) -- settle empirically.
- resolution-aware shift values at 512x288.
- `temporal_size` effect on spatial sharpness.

## Convergence
One grounded pass; 4/4 converged on root cause + the Tier-1 ladder. No conflicting
must-fix survived grounding. A pass02 is optional (diminishing returns) -- the actionable
signal is strong. Recommended next = run the Tier-1 first shot on the 5080 and eyeball.


======================================================================
# PART 3 -- RAW PANEL REVIEWS (unedited, model headers preserved)
======================================================================



### GPT-5.5

<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The proposed A/B is not build-ready: the repro is time-mismatched, over VRAM cap at 832x480, and most proposed “knobs” are hardcoded or unimplemented in `eng_ltx_av.py`.

MUST-FIX BEFORE BUILD:
1. [Symptom] / [KNOBS/7] 105 frames at 25 fps is only 4.2 s, but the stated audio slice is ~13 s. The grounded JSON note says `c02_b002_line.wav` is 13.28 s, while `LTXVImgToVideo` length is 105. That makes the A2V/lip-sync test invalid and can confound any blur diagnosis. Concrete fix: either trim the audio to 4.2 s or set video length from audio duration, snapped to 8n+1: for 13.28 s at 25 fps, target about 337 frames. Then re-measure VRAM/time.

2. [Constraints/Invariants] / [Symptom] The observed 832x480 run peaks ~15.75 GB, above the stated 14.5 GB ceiling. The plan still treats 832x480 as a main A/B canvas. Concrete fix: make 512x288 the default build/test canvas for ceiling-respecting A/B, and treat 832x480 only as an explicit over-ceiling diagnostic run. Do not accept any “sharpness” setting until it is measured under 14.5 GB at the chosen production canvas.

3. [eng_ltx_av.py:_render_dims] If request canvas is absent, `_render_dims()` defaults to native 1472x832. Under the stated 14.5 GB ceiling this is almost certainly unsafe given 832x480 already measured over cap. Concrete fix: change the fallback default for this lane to the VRAM-safe production canvas, or fail closed when canvas is absent. Do not default to 1472x832 on a 16 GB laptop profile.

4. [ltx_av_talk_mini_repro_gguf_mit.json/Note] The note says to drop to 512x288 via `OTR_LTX_AV_RENDER_CANVAS`, but `eng_ltx_av.py` excerpt shows no such env var; dimensions come from request canvas or native fallback. Concrete fix: either implement `OTR_LTX_AV_RENDER_CANVAS` in `_render_dims()` with validation, or remove that instruction and require request.canvas.

5. [Questions/3] / [eng_ltx_av.py:_build_graph] Most requested knobs are not actually configurable in the grounded engine: sampler is hardcoded `"euler"`, scheduler class is hardcoded `LTXVScheduler`, shifts are hardcoded `2.05/0.95`, `stretch=True`, `terminal=0.1`, tiled decode values are hardcoded, and no LoRA/ManualSigmas nodes exist in `_node_candidates()`. Only steps/cfg/i2v strength are env-overridable. Concrete fix: add explicit env/config-controlled switches for sampler, scheduler mode, manual sigma string, shifts, terminal, stretch, decode tiling, and optional LoRA before claiming an A/B matrix can be run through the engine.

6. [CONTRAST] The statement “differs from the AV talk graph in exactly these ways” is false against the grounded JSON. The golden graph also differs in topology and inputs: `EmptyLTXVLatentVideo + LTXVImgToVideoConditionOnly` vs `LTXVImgToVideo`, no audio concat, different image, different prompt, different text encoder device, different conditioning order, and no `ModelSamplingLTXV` in the golden JSON. Concrete fix: rewrite the comparison as “known differences,” not “exactly,” and build controlled ablations that change one variable at a time on the same prompt/image/canvas/frame count.

7. [Hypothesis to test] The hypothesis over-attributes blur to “non-distilled configuration” without controlling for the frame/audio mismatch, canvas/encoding, A2V concat path, and topology differences above. Concrete fix: run minimum controlled ladder first: same 512x288 canvas, matched audio/video duration, same seed, same prompt/image, then vary only steps/cfg/sampler/scheduler/i2v strength.

8. [Questions/2] Applying `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` to the A2V graph is not buildable from the grounded engine: `_node_candidates()` lacks `LoraLoaderModelOnly`, `_weight_paths()` does not check the LoRA file, and there is no env/config for LoRA name/strength. Concrete fix: if testing LoRA, add gated optional LoRA support with node resolution, weight existence/floor check, license check, and an explicit graph order. Verify whether the LoRA output is compatible with `ModelSamplingLTXV` and GGUF patching before production.

9. [Questions/2] The proposed distilled chain has an unresolved sequencing dependency: golden JSON uses `UnetLoaderGGUF -> LoraLoaderModelOnly -> CFGGuider` and does not show `ModelSamplingLTXV`; AV uses `UnetLoaderGGUF -> ModelSamplingLTXV -> CFGGuider`. It is not specified whether AV+LoRA should be `unet -> lora -> ModelSamplingLTXV -> guider` or `unet -> ModelSamplingLTXV -> lora -> guider`. Concrete fix: verify node compatibility and pick one graph order explicitly; do not silently insert LoRA “between unet and guider.”

10. [Questions/2] `ManualSigmas` is not available in `_node_candidates()`. Replacing `LTXVScheduler` with fixed sigmas will fail through `wrapper_bridge.resolve_graph_classes()` unless the node candidate is added. Concrete fix: add `ManualSigmas` as a gated candidate and graph branch, or keep `LTXVScheduler`.

11. [Constraints/Invariants] License handling is underspecified for the distilled LoRA. The document requires Apache GGUF + LTX-2 Community only, but the LoRA is an additional artifact not checked in `_weight_paths()` and not license-validated in the excerpt. Concrete fix: verify the LoRA license and source before adding it; otherwise exclude it from build-ready settings.

12. [Questions/3] “Render high then downscale” conflicts with the VRAM evidence. Native 1472x832 is far larger than 832x480, and 832x480 already exceeds the cap. Concrete fix: reject native/high-res generation on this profile unless a measured run stays under 14.5 GB. If using downscale, generate at the highest measured under-cap canvas only.

SHOULD-FIX:
1. [Questions/3] Do not prioritize `VAEDecodeTiled` as the likely source of microdetail blur without an A/B. Grounding shows both graphs use `tile_size=512` and `overlap=64`; the main decode difference is temporal_size 64 vs 4096. Spatial softness is therefore not explained by spatial tiling alone. Concrete fix: test decode temporal_size 64 vs 4096 on the same already-sampled latent if possible, and compare raw frames.

2. [eng_ltx_av.py:render_clip] Decode happens before `_wb.reclaim_idle_models()`. Increasing `VAEDecodeTiled temporal_size` to 4096 may raise peak VRAM while model patchers are still resident. [ASSUMPTION] Concrete fix: measure decode-only peak with temporal_size 4096 at 512x288 before adopting; consider reclaim/offload before decode if supported.

3. [ltx_av_talk_mini_repro_gguf_mit.json] The repro uses `SaveWEBM` VP9 quality 18 and also `SaveVideo` h264 for muxed output. Encoder artifacts can be mistaken for model blur. Concrete fix: judge sharpness on raw decoded frames or lossless/high-quality image sequence before comparing WEBM/MP4 outputs.

4. [Questions/3] Step-count cost must be stated as roughly linear sampler time. Moving from 8 to 12/16 steps should not materially increase model weights VRAM, but can increase activation/runtime memory slightly depending implementation. [ASSUMPTION] Concrete fix: A/B 8/12/16 at 512x288 and record both peak NVML and wall time.

5. [Questions/3] CFG change should be tested independently from LoRA. Current engine exposes `OTR_LTX_AV_CFG`; safe first tests are cfg 1.0, 1.5, 2.0, 3.0 with the current non-LoRA graph. Concrete fix: rank low-CFG tests before LoRA because they require no new nodes or weights.

6. [Questions/3] I2V strength is env-overridable via `OTR_LTX_AV_I2V_STRENGTH`, so test 0.65/0.75/0.85/1.0 independently. Strength 1.0 may over-denoise away portrait detail, but lowering it may also reduce mouth/head motion. [ASSUMPTION] Concrete fix: evaluate sharpness and lip motion separately.

7. [Questions/4] The document asks for LTX-2.3-specific shift/terminal/stretcher advice, but no grounded source excerpt provides official recommended values. Concrete fix: mark any shift/terminal recommendations as empirical only unless backed by upstream LTX docs or code.

8. [Constraints/Invariants] Text encoder device differs between graphs: AV uses `"cpu"`; golden uses `"default"`. Moving it to GPU may blow VRAM and is not necessary to test blur. Concrete fix: hold text encoder on CPU for all ceiling-respecting A2V tests unless specifically measuring encoder-device impact.

9. [Questions/1] The likely blur cause should be reframed as a bundle, not one cause: under-sampled 8-step non-distilled A2V, high cfg, i2v strength 1.0, duration mismatch, and compressed output are all plausible. Concrete fix: state the ranked suspicion but do not claim causality until ablations isolate it.

10. [eng_ltx_av.py:assert_usable] The VRAM ceiling is checked after render, not before sampling. That catches violations too late to protect the build profile. Concrete fix: add preflight canvas/frame-count policy or empirical denylist for known-over-cap dimensions, especially 832x480x105 and native fallback.

OPTIONAL / NICE-TO-HAVE:
- Add a small “quality probe” harness that renders 512x288, fixed 81 or 105 frames, matched short audio, and dumps raw PNG frames plus NVML peak JSON.
- Add objective sharpness metrics only as secondary signals: Laplacian variance/edge energy on face crop, plus human review for lip-sync and warping.
- Store A/B graph metadata in output filenames: canvas, frames, steps, cfg, sampler, scheduler, strength, seed.

CUT THESE (over-engineering):
1. [Questions/3] Cut native 1472x832 testing on the 16 GB laptop profile. It is not compatible with the observed 832x480 peak over cap, and it will not answer the production 512x288 question.

2. [Questions/2] Cut LoRA from the first A/B round. It requires new node gating, weight checks, license validation, and graph sequencing decisions. First test existing exposed knobs: cfg, steps, i2v strength, and maybe sampler if implemented.

3. [Questions/3] Cut non-tiled `VAEDecode` as an early test unless a measured decode-only run proves it fits. Tiled decode is already used by the sharp golden graph; the immediate risk is sampler/config, not spatial tiling.

4. [Questions/4] Cut prompt/negative-prompt tuning from the initial sharpening pass. The current negative already includes “blurry,” and prompt changes will confound sampler/CFG/strength ablations. Keep prompt fixed until the graph settings are isolated.

### Gemini-3.1-pro

<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The graph will execute, but the default parameters guarantee a blurry output and mathematically violate your 14.5 GB VRAM ceiling.

MUST-FIX BEFORE BUILD (Severity Order):
1. [Constraints / VRAM Ceiling] The prompt states 832x480 hits 15.75 GB, breaking the 14.5 GB invariant. 
   - **Fix**: You MUST drop the render canvas to 512x288 (`OTR_LTX_AV_RENDER_CANVAS` or request template). Do not adopt the golden recipe's `temporal_size=4096`; it will immediately OOM.
2. [eng_ltx_av.py / _LTX_AV_STEPS] 8 steps on a non-distilled base model causes massive under-convergence (the primary cause of your blur). The golden recipe is only sharp at 8 steps because it uses a distilled LoRA. 
   - **Fix**: Change `_LTX_AV_STEPS` default to `25`. (Time cost will increase to ~350s, but it is mandatory for sharpness without the LoRA).
3. [eng_ltx_av.py / _LTX_AV_I2V_STRENGTH] Strength `1.0` in `LTXVImgToVideo` overwrites 100% of the init image with pure noise, destroying the sharp portrait details before denoising even begins.
   - **Fix**: Change `_LTX_AV_I2V_STRENGTH` default to `0.75` or `0.80`.

SHOULD-FIX (Ranked A/B Settings for Sharpness & Panel Answers):
1. **A/B Test 1 (Safe Base)**: `euler` + `LTXVScheduler` (25 steps) + CFG `2.5` + Strength `0.75`. 
   - *Rationale*: Properly converges the base model and retains init image high-frequencies without LoRA risks. 
   - *Risk*: High time cost (+200s). VRAM safe at 512x288.
2. **A/B Test 2 (Risky Distilled)**: `euler_cfg_pp` + `ManualSigmas` (8 steps) + CFG `1.0` + LoRA @ `0.70`. 
   - *Rationale*: Matches golden sharp recipe exactly to recover the ~122s render time. 
   - *Risk*: High correctness risk. The A2V Unet processes a concatenated audio+video latent (`LTXVConcatAVLatent`). The T2V-distilled LoRA was NOT trained on this audio channel depth/distribution. It will likely suppress the audio cross-attention, breaking lip-sync entirely, or cause tensor shape crashes.
3. **A/B Test 3 (Hybrid)**: `euler_cfg_pp` + `LTXVScheduler` (20 steps) + CFG `2.0` + Strength `0.80` (NO LoRA).
   - *Rationale*: `euler_cfg_pp` often yields higher micro-contrast than plain `euler` on LTX even without distillation. 
   - *Risk*: Moderate time cost.
4. **[Graph Sequencing]** If you test the distilled `ManualSigmas` (A/B Test 2), you MUST bypass `ModelSamplingLTXV`. 
   - *Rationale*: `ManualSigmas` already contains the explicitly shifted schedule. Applying `ModelSamplingLTXV` on top double-shifts the sigmas, causing severe blur.

OPTIONAL / NICE-TO-HAVE:
- **Resolution-aware Shift**: Flow-matching models require less noise shift at lower resolutions. If you drop to 512x288, lower `max_shift` in `ModelSamplingLTXV` and `LTXVScheduler` from `2.05` to `~1.50`. This prevents over-smoothing at low resolutions.
- **Negative Prompt**: The golden recipe removes "frozen pose, still image" from the negative prompt. If your talking head lacks motion, remove these; they can inadvertently suppress natural micro-movements.

CUT THESE (Over-engineering / Traps):
1. [Golden Recipe / `VAEDecodeTiled`] `temporal_size=4096`. Safe to cut/ignore. Decoding 105 frames at once will instantly blow the 16 GB VRAM limit. Keep `temporal_size=64` (or `32` for 512x288).
2. [Golden Recipe / `LTXVImgToVideoConditionOnly`] Do not swap `LTXVImgToVideo` for `ConditionOnly` in the A2V graph. The A2V concat logic expects the fully noised/scaled latent from `LTXVImgToVideo`, not just the conditioning output.

[ASSUMPTION] I am assuming the `LTXVConcatAVLatent` node alters the latent channel dimension or sequence length in a way that a standard T2V LoRA will either ignore (breaking audio sync) or misinterpret (causing visual artifacts).

### Grok-4.3

<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Hypothesis rests on untested distilled-LoRA transfer to A2V concat latent while current code+JSON hard-wires the blurry non-distilled path and already exceeds VRAM ceiling at target canvas.

MUST-FIX BEFORE BUILD:
1. [Symptom + eng_ltx_av.py:133] 832x480 already peaks 15.75 GB (>14.5 GB cap); any sharpness change (steps, sampler, or decode tile) must be gated by explicit downscale to 512x288 or it will OOM on the 5080. Add runtime canvas clamp before LTXVImgToVideo.
2. [Hypothesis + eng_ltx_av.py:180 (LTXVScheduler) + ltx_av_talk_mini...json:15] Distilled LoRA + ManualSigmas + euler_cfg_pp are absent from the A2V graph; inserting them risks latent-shape mismatch on LTXVConcatAVLatent output. Replace hypothesis with measured A/B that keeps LoRA off until proven.
3. [Knobs item 11 + eng_ltx_av.py:216] VAEDecodeTiled temporal_size=64 (vs 4096 in sharp recipe) directly truncates temporal context and produces softness; change default to 256 (or higher) when VRAM permits and document the seam risk.
4. [Constraints + eng_ltx_av.py:99] Negative prompt is static and contains "blurry"; make it an overridable env var so the A/B test can actually remove the word that the sampler is told to aim for.

SHOULD-FIX:
1. [Knobs item 9 + ltx_av_talk...json:14] KSamplerSelect stays on "euler"; switch default to "euler_cfg_pp" (no LoRA) as zero-cost partial adoption that the sharp recipe already validates.
2. [eng_ltx_av.py:172] i2v strength hard-coded to 1.0; expose as OTR_LTX_AV_I2V_STRENGTH and default to 0.85 to reduce copy-through of the soft init portrait.
3. [ltx_av_talk...json:20] No ModelSamplingLTXV shift retune for 832x480; add explicit 1.85/0.85 pair when resolution < native and record measured shift impact.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line "distilled_safe" boolean in _build_graph that swaps only sampler+cfg+steps while leaving LoRA out.

CUT THESE (over-engineering):
1. ManualSigmas block (sharp JSON) - cannot be used until A2V latent compatibility proven; safe to drop from first A/B pass.
2. Any non-tiled VAEDecode path - exceeds ceiling at 832x480 and is not license-clean under current node list.

### DeepSeek-v4-pro

<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? **no**. The document is a research hypothesis, not a build-ready spec; it lacks concrete code changes, a VRAM-safe resolution, license verification, and any validation of the distilled LoRA’s transfer to the audio-concatenated A2V latent. It cannot be implemented directly.

MUST-FIX BEFORE BUILD:
1. **[Hypothesis to test]** Proposes adopting the distilled LoRA on the A2V graph without evidence it works on audio+video concatenated latents. The LoRA was trained on video-only t2v/i2v; its behaviour on the extended A2V input channels is unknown. Fix: Add an explicit validation step—test the LoRA on a minimal A2V clip (e.g., 9 frames) and define a non-LoRA fallback (e.g., more steps, adjusted sampler) if the output breaks.
2. **[eng_ltx_av.py Implementation]** The engine code has no provision for a `LoraLoaderModelOnly` node, `ManualSigmas`, or `euler_cfg_pp` sampler. The plan does not detail the required modifications. Fix: Specify all code changes to `_node_candidates()`, `_build_graph()`, and new env vars (`OTR_LTX_AV_DISTILLED_LORA`, `OTR_LTX_AV_SAMPLER`, etc.) with backward-compatible defaults.
3. **[VAEDecodeTiled / temporal_size]** The current `temporal_size=64` strongly contributes to temporal blur; the sharp video mini uses `4096`. The plan only asks about this, without ordering a test. Fix: Prioritize increasing `temporal_size` to the full clip length (or at least 256) in an A/B test; it’s a low-cost sharpness improvement.
4. **[VRAM Ceiling]** The current 832×480 already uses ~15.75 GB, exceeding the 14.5 GB limit. Any sharpening addition (LoRA, larger temporal chunk, more steps) will worsen this. The plan does not propose a safe resolution or a measured high-res downscale path. Fix: Determine a resolution (e.g., 640×368) that stays ≤14.5 GB with all sharpened settings, or implement a profiled tiled high-res render→downscale chain.
5. **[Licensing]** The file `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` used in the sharp recipe is not license-checked. The project demands Apache+Community-only, no extra restrictions. Fix: Verify the LoRA’s license matches LTX-2 Community terms; if not, reject it and rely solely on non-LoRA sharpening.
6. **[Sampler dependency]** `euler_cfg_pp` requires the model to internalize CFG (cfg-pp mode), which may only be valid after applying the distilled LoRA. If the LoRA fails or is not used, the plan offers no fallback sampler+CFG combination. Fix: Document that if LoRA is absent, fall back to `euler` (or another sampler) with CFG 3.0, and confirm `euler_cfg_pp` works only with the LoRA present.

SHOULD-FIX:
1. **[LTXVScheduler terminal]** The terminal sigma of 0.1 stops denoising early, leaving residual noise and softness. Make terminal configurable and test 0.0.
2. **[i2v strength]** Strength 1.0 forces full image adherence, which can produce a waxy/blurry look. Lowering to 0.8–0.85 often improves motion sharpness and is trivial to test.
3. **[Negative prompt]** Expand the negative prompt to explicitly target softness (e.g., “soft focus, motion blur, low detail”).

OPTIONAL / NICE-TO-HAVE:
- Provide an env-var toggle (`OTR_LTX_AV_EXPERIMENTAL_SHARP=1`) that switches between the current baseline and a pre-configured sharp graph without manual JSON editing.

CUT THESE (over-engineering): None; the plan is under-specified, not bloated.

[ASSUMPTION] Inferred that the distilled LoRA weights will load and apply to the A2V-modified transformer without architectural mismatch—this must be verified.
[ASSUMPTION] Inferred that increasing `temporal_size` won’t OOM at a lowered resolution—needs measurement.

======================================================================
# PART 4 -- PANEL REQUEST (paste to re-run the panel manually)
======================================================================

﻿# LTX-AV settings -- PANEL REQUEST (paste into ChatGPT / Gemini / DeepSeek / Grok)

--- SYSTEM / REVIEWER ROLE ---

ROLE: You are a senior architect and release engineer doing a FINAL, adversarial,
pre-build review of the plan / spec / design document below. You are skeptical by
default. No praise, no padding -- find what breaks.

You are one voice on a panel of independent reviewers. You do not see the other
reviews. A separate judge will verify every claim you make against the real
source code, so vague or hand-wavy criticism is worthless -- be specific, cite the
section you mean, and make each point checkable.

REVIEW THE WHOLE DOCUMENT, weighting:

1. Correctness -- anything that would not work as written, contradicts itself, or
   rests on a false assumption. Name the section.
2. Gaps -- what the plan needs to handle but does not (edge cases, failure modes,
   ordering, missing steps).
3. Risk -- where this is most likely to break at build or run time, and why.
4. Over-engineering -- what is heavier than it needs to be and could be cut
   without losing the goal.
5. Hidden dependencies / sequencing -- steps that secretly depend on each other,
   or that are ordered wrong.

GROUNDING: If grounding excerpts (real source files, JSON, schemas) are provided
below the document, check your claims against them. Do NOT invent file contents,
function names, or APIs you cannot see. If a claim depends on code you were not
shown, say "verify: <what>" instead of asserting it. Confident-but-wrong claims
about the code waste the judge's time and will be discarded.

OUTPUT (strict, plain text, no fluff):
- VERDICT: build-ready as-is? yes / yes-with-fixes / no. One line why.
- MUST-FIX BEFORE BUILD: numbered. Each = [section id] + the defect + the concrete
  fix. Severity order.
- SHOULD-FIX: numbered, same format.
- OPTIONAL / NICE-TO-HAVE: brief.
- CUT THESE (over-engineering): numbered, with why it is safe to cut.
- Mark [ASSUMPTION] anywhere you are inferring beyond the document or the
  grounding excerpts.

Cite section identifiers throughout. Do not restate the document back. Prefer the
smallest change that closes each defect.


--- DOCUMENT TO REVIEW (problem statement) ---

# Problem statement: LTX-AV "talk" lane looks blurry/fuzzy -- what settings sharpen it?

## Goal / use case
OTR is a 100% local, offline sci-fi **radio drama** generator on a single **RTX 5080
Laptop, 16 GB VRAM (14.5 GB usable ceiling, host NVML)**. The **LTX-AV "talk" lane**
(family `audio_driven_face`) drives a **talking-head video** from:
- ONE character **portrait still** (image-to-video init), plus
- a **per-beat speech audio slice** (the character's line),
to get a **lip-synced talking head** that looks **sharp + cinematic**.

Models: **LTX-2.3-22B** transformer as a **GGUF Q3_K_M** unet (the only quant that
fits the ceiling), **Gemma-3-12B fp4** text encoder (run on CPU to save VRAM), LTX-2.3
**video VAE** + **audio VAE**. The graph is an **A2V** (audio-to-video) graph: the
audio is VAE-encoded and **concatenated** to the video latent, sampled jointly, then
**split** back; only the video latent is decoded (the generated audio latent is
discarded -- "V-1", the clip is silent and real audio is muxed later).

## Symptom (what we observed on a real render)
A live render (832x480, 105 frames, character portrait + a 13 s real line) **succeeds**
end to end but the result is **"kind of ok but blurry/fuzzy"** -- acceptable as a
rudimentary talking head, but **soft, low microdetail, not crisp**. We want it sharper
(and, ideally, with believable but not warpy head/mouth motion). Measured: wall ~122 s
(warm loaders), **VRAM peak ~15.75 GB at 832x480 (OVER the 14.5 GB cap; no OOM on the
16 GB board)**.

## The KNOBS / INPUTS of the current (fuzzy) LTX-AV talk graph
(Exact values; this is the graph that produced the blur. Grounding files attached:
`eng_ltx_av.py` builds this; `ltx_av_talk_mini_repro_gguf_mit.json` is this graph.)

1. `UnetLoaderGGUF` unet = `ltx-2.3-22b-dev-Q3_K_M.gguf`  (the 22B transformer, GGUF Q3)
2. `ModelSamplingLTXV`  max_shift = **2.05**, base_shift = **0.95**   (unet -> sampling)
   -- NOTE: **no distilled LoRA** is applied (unet -> ModelSamplingLTXV -> guider).
3. `LTXAVTextEncoderLoader`  text_encoder = gemma_3_12B_it_fp4_mixed,
   ckpt_name (projection) = ltx-2.3-22b-dev.safetensors, device = **cpu**
4. `CLIPTextEncode` positive = the per-beat shot description (a talking-head prompt);
   `CLIPTextEncode` negative = "low quality, worst quality, blurry, jpeg artifacts,
   distorted, deformed, static, frozen pose, still image, watermark, text"
5. `LTXVConditioning` frame_rate = **25.0**
6. Audio path: `LoadAudio` -> `LTXVAudioVAEEncode`(audio_vae) -> audio_latent
7. i2v: `LTXVImgToVideo`  width=**832**, height=**480**, length=**105**, batch_size=1,
   **strength = 1.0**   (init = the character portrait still; outputs pos/neg conditioning + video latent)
8. `LTXVConcatAVLatent`(video_latent, audio_latent) -> av_latent
9. Sampler stack:
   - `KSamplerSelect` sampler_name = **euler**
   - `LTXVScheduler`  steps = **8**, max_shift = **2.05**, base_shift = **0.95**,
     stretch = **true**, terminal = **0.1**   (computes the sigma schedule)
   - `RandomNoise` seed = 1
   - `CFGGuider` cfg = **3.0**   (model from ModelSamplingLTXV; pos/neg from the i2v node)
   - `SamplerCustomAdvanced`(noise, guider, sampler=euler, sigmas=LTXVScheduler, latent=av_latent)
10. `LTXVSeparateAVLatent`(av_latent) -> video_latent (audio_latent dropped)
11. `VAEDecodeTiled`  tile_size=**512**, overlap=**64**, temporal_size=**64**,
    temporal_overlap=**8**   (video VAE)
12. Output canvas tested = **832x480 x105 @ 25fps**. Native LTX-AV default is 1472x832;
    the VRAM-safe production canvas is **512x288**.

## The CONTRAST: the SHARP "video mini" recipe (NOT blurry)
The golden video-only mini (`ltx_bookend_mini_repro_gguf_mit.json`) uses the SAME 22B
Q3_K_M unet and SAME video VAE and renders **SHARP** (i2v Laplacian-sharp; t2v real
motion). Its recipe differs from the AV talk graph in exactly these ways:
- **Distilled LoRA APPLIED**: `LoraLoaderModelOnly` `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`
  @ **0.70** between the unet and the guider.
- Sampler = **`euler_cfg_pp`** (not plain `euler`).
- Sigmas = **fixed `ManualSigmas`** distilled schedule
  "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0" (8 steps),
  **not** a computed `LTXVScheduler`.
- **CFG = 1.0** (not 3.0) -- the distilled LoRA is a cfg-distilled/guidance-baked model,
  so it runs at cfg ~1.0.
- i2v = `LTXVImgToVideoConditionOnly` **strength = 0.75** (not 1.0).
- `VAEDecodeTiled` temporal_size = **4096** (not 64).
- It has **no audio path** (pure t2v/i2v); text encoder device = default (GPU).

## Hypothesis to test (panel: confirm/deny/improve)
The blur most likely comes from running the 22B in a **non-distilled** configuration on
the A2V graph: **plain `euler` + a stretched 8-step `LTXVScheduler` + cfg 3.0 + i2v
strength 1.0 + no distilled LoRA**, whereas the sharp recipe is the **distilled chain**
(LoRA@0.70 + `euler_cfg_pp` + fixed distilled sigmas + cfg 1.0 + strength 0.75). 8 steps
of plain euler at cfg 3.0 on a distilled-style model may be **under-converged /
over-guided -> soft**.

## Constraints / invariants (any proposal MUST respect)
- **100% local/offline; no cloud.** License-clean: Apache GGUF + LTX-2 Community ONLY;
  **no RES4LYF/AGPL, no VHS/GPL** nodes.
- Single resident heavy engine **<= 14.5 GB** (832x480 already exceeds; 512x288 fits
  ~13.7 GB). Sharper settings must not blow the ceiling at the chosen canvas.
- **Determinism** (seed-keyed). UTF-8/ASCII, SFW.
- The **A2V graph is different** from the t2v/i2v graph: the latent is the
  audio+video concat, sampled jointly. **Unknown whether the distilled LoRA (trained on
  the t2v/i2v transformer) and the fixed distilled sigmas transfer cleanly to the A2V
  concat latent.** This is the central uncertainty.
- Keep V-1 (audio latent dropped, silent video, mux later) unless there's a strong reason.

## Questions for the panel
1. **What is the most likely cause of the softness/blur**, given the two recipes above?
2. **Does the distilled LoRA @0.70 + `euler_cfg_pp` + fixed distilled sigmas + cfg 1.0
   transfer to the A2V (audio-concat) graph?** If risky, why, and what's the safe partial
   adoption (e.g. keep cfg low + euler_cfg_pp + more steps without the LoRA)?
3. **Concrete setting changes to sharpen**, ranked, with expected VRAM/time cost:
   step count, sampler (euler vs euler_cfg_pp vs others), scheduler (fixed sigmas vs
   LTXVScheduler stretch/terminal/shift), **cfg**, **i2v strength**, ModelSamplingLTXV
   shift values, **resolution/canvas** (does 832x480 vs native 1472x832 vs 512x288 change
   sharpness? is there an LTX "render-high-then-downscale" path?), and
   **`VAEDecodeTiled` tiling** (could tile_size 512 / temporal_size 64 cause seams or
   softness vs the video mini's 4096? would a non-tiled `VAEDecode` be sharper if VRAM
   allows?).
4. Any **LTX-2.3-specific** guidance for sharp i2v/A2V (recommended sigma shift for the
   resolution, terminal sigma, the role of `stretch`, negative-prompt advice).
5. Anything that would **break the ceiling or licensing** -- call it out so we reject it.

Deliver: a ranked, A2V-aware set of setting changes we can A/B on the 5080, each with a
one-line rationale and a VRAM/time risk note. We (Claude) will ground every claim
against the attached real files before adopting.


--- GROUNDING 1/3: nodes/_otr_video_engines/eng_ltx_av.py ---

"""LTX-2.3 AUDIO-INPUT (A2V) lane -- additive, in-process, default-OFF / dark.

A NEW, DARK, ADDITIVE engine pair that drives video from the per-beat slice of the
FROZEN master audio + a text prompt (+ a FLUX still for the talk lane). It is NOT
the golden prompt-only ``ltx_video`` engine and shares NO code or env with it --
the two lanes diverge on purpose (this lane snaps frames UP via
``av_dims.next_8n1``; ``eng_ltx_video`` snaps DOWN). ``eng_ltx_video.py`` is FROZEN
and is never imported or touched here.

Two adapters over one shared core (M0-GROUNDED graph; GGUF Q3_K_M proven on the
RTX 5080 at 13688 MB peak <= the 14500 ceiling, Gemma-3 encoder offloaded to CPU):

* ``ltx_av_talk``  -- roles (announcer_visual, character_video); family
  ``audio_driven_face``; required text_prompt + audio_ref + init_image; lip-sync
  attempt from the still (I2V) + the audio slice; fallback -> humo.
* ``ltx_av_music`` -- role (music_visual,); family ``audio_conditioned_video``;
  required text_prompt + audio_ref; audio-reactive scene motion (sync-loose);
  fallback -> ltx_video.

V-1 absolute: the lane DISCARDS LTX's audio side entirely -- the graph terminates
at ``LTXVSeparateAVLatent -> video_latent -> VAEDecodeTiled`` (the audio_latent
branch + ``LTXVAudioVAEDecode`` are never wired), the clip is ALWAYS silent
(has_audio False), and only ``OTR_MasterAudioMux`` emits audio.
``test_audio_byte_identical`` stays green.

Cold-import clean (V-12): module scope imports only stdlib + the dep-free shared
helpers + the registry. torch / the LTX wrapper nodes are imported LAZILY inside
``load`` / ``render_clip`` (the GPU slice), never here. NVML is REQUIRED for this
lane (heaviest engine) -- assert_usable fails CLOSED when NVML is absent so the
ceiling guard can never silently no-op. UTF-8, no BOM, ASCII-only source.

Config (env; each resolves via ComfyUI folder_paths so a box never needs a code
edit): OTR_ENABLE_LTX_AV (opt-in flag); OTR_LTX_AV_UNET (GGUF unet in models/unet);
OTR_LTX_AV_TEXT_ENCODER (Gemma-3 in text_encoders); OTR_LTX_AV_PROJECTION_CKPT
(the LTX ckpt supplying the text-projection, in checkpoints); OTR_LTX_AV_VIDEO_VAE
+ OTR_LTX_AV_AUDIO_VAE (in vae). RESTART ComfyUI after any mid-render cancel
(a wedged PID holds the AS-3 lease ~120 s; reclaim only frees dead PIDs).
"""
from __future__ import annotations

import os

from .._otr_shared import av_dims as _AVD
from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

# --- frame + canvas grounding (LTX-AV lane only; snap UP, never copy Lane A) ---
_LTX_AV_MIN_FRAMES = _AVD._LTX_MIN_FRAMES        # 9 (8n+1 floor)
_LTX_AV_MAX_FRAMES = int(os.environ.get("OTR_LTX_AV_MAX_FRAMES", "497"))  # M0 initial
_LTX_AV_NATIVE_W = 1472
_LTX_AV_NATIVE_H = 832
# Default sampler recipe (the M0-proven 8-step distilled-ish base pass; cfg 3.0
# matched the probe). All env-overridable; never shared with eng_ltx_video.
_LTX_AV_STEPS = int(os.environ.get("OTR_LTX_AV_STEPS", "8"))
_LTX_AV_CFG = float(os.environ.get("OTR_LTX_AV_CFG", "3.0"))
_LTX_AV_I2V_STRENGTH = float(os.environ.get("OTR_LTX_AV_I2V_STRENGTH", "1.0"))
# ASCII-only negative (CLAUDE.md). One shared constant; cap 240 in the driver.
_LTX_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, "
    "static, frozen pose, still image, watermark, text")

# Weight sanity floors (GiB) -- catch a truncated / wrong download, NOT exact
# byte checks. Q3_K_M unet ~9.3 GiB; Gemma-3 fp4 ~8.2 GiB; video VAE ~1.35 GiB.
_GiB = 1024 ** 3
_FLOOR_UNET = 8 * _GiB
_FLOOR_ENCODER = 6 * _GiB
_FLOOR_VIDEO_VAE = 1 * _GiB
_FLOOR_AUDIO_VAE = int(0.2 * _GiB)


def _resolve(folder, name):
    """Resolve a model filename to a full path via ComfyUI folder_paths (honors
    extra_model_paths.yaml), with a best-effort join fallback for the headless /
    CPU existence check (no folder_paths registered)."""
    if not name:
        return ""
    try:
        import folder_paths  # type: ignore
        p = folder_paths.get_full_path(folder, name)
        if p:
            return p
    except Exception:  # noqa: BLE001 - headless/CPU
        pass
    here = os.path.abspath(__file__)
    comfy_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(here))))), "models", folder, name)
    return comfy_models


class _LtxAvBase(_MC.MotionEngineBase):
    """Shared LTX-AV core: assert_usable gate, graph spec, render lifecycle.

    Subclasses set the lane identity (name / family / roles / required_inputs /
    fallback_engine) and ``_is_talk``. The I2V-vs-t2v branch is INTERNAL to
    ``_build_graph`` so talk + music share one resident load path."""

    default_roles = ()                  # dark: never a default for any role
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = "OTR_ENABLE_LTX_AV"
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    _is_talk = False
    _TERMINAL = "decode"

    # ---- config resolution (env override -> folder_paths -> join) ----
    def _unet_name(self):
        return os.environ.get("OTR_LTX_AV_UNET", "ltx-2.3-22b-dev-Q3_K_M.gguf")

    def _encoder_name(self):
        return os.environ.get("OTR_LTX_AV_TEXT_ENCODER",
                              "gemma_3_12B_it_fp4_mixed.safetensors")

    def _projection_ckpt(self):
        return os.environ.get("OTR_LTX_AV_PROJECTION_CKPT",
                              "ltx-2.3-22b-dev.safetensors")

    def _video_vae_name(self):
        return os.environ.get("OTR_LTX_AV_VIDEO_VAE",
                              "ltx-2.3-22b-dev_video_vae.safetensors")

    def _audio_vae_name(self):
        return os.environ.get("OTR_LTX_AV_AUDIO_VAE",
                              "ltx-2.3-22b-dev_audio_vae.safetensors")

    def _weight_paths(self):
        """(label, full_path, floor_bytes) for each required weight artifact."""
        return [
            ("transformer GGUF", _resolve("unet", self._unet_name()), _FLOOR_UNET),
            ("Gemma-3 text encoder",
             _resolve("text_encoders", self._encoder_name()), _FLOOR_ENCODER),
            ("video VAE", _resolve("vae", self._video_vae_name()), _FLOOR_VIDEO_VAE),
            ("audio VAE", _resolve("vae", self._audio_vae_name()), _FLOOR_AUDIO_VAE),
        ]

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Ordered, fail-closed-before-GPU gate (six PINNED reasons only):
        1 flag; 2 BUG-070 Sage gate; 3 NVML REQUIRED (this lane only -- fail
        closed so the ceiling guard never silently no-ops); 4 node gate (every
        required ComfyUI class resolves); 5 weight floors (realpath + size);
        6 av_dims on request_template.canvas (None tolerated)."""
        # 1 -- opt-in flag
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "%s is opt-in; set %s=1 and install the LTX-2.3 GGUF + Gemma-3 "
                "encoder + LTX VAEs" % (self.name, self.requires_flag),
                kind="video")
        # 2 -- BUG-070 SageAttention contamination (int8-PV aborts LTX silently)
        _MC.assert_sage_not_patched(self.name, self.family)
        # 3 -- NVML REQUIRED for the heaviest lane (grounded fail-open risk:
        #      probe_used_mb()->0 makes the ceiling asserts no-op)
        from .._otr_shared import gpu_residency as _GR
        if not _GR.nvml_available():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "%s requires NVML to enforce the %d MB ceiling; NVML is "
                "unavailable on this host (the LTX-AV lane fails closed rather "
                "than run an unbounded heavy forward)"
                % (self.name, _MC.dynamic_vram_ceiling_mb()), kind="video")
        # 4 -- node gate: every required ComfyUI class must resolve (lazy read)
        from . import wrapper_bridge as _wb
        missing = []
        mapping = _wb.node_class_mappings()
        for logical, candidates in self._node_candidates().items():
            try:
                _wb.resolve_node_class(candidates, mapping)
            except Exception:  # noqa: BLE001 - collect every missing class
                missing.append("/".join(candidates))
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s missing required ComfyUI node class(es): %s (install/update "
                "ComfyUI-GGUF + ComfyUI-LTXVideo)" % (self.name, ", ".join(missing)),
                kind="video")
        # 5 -- weights present + above the sanity floor (realpath -> broken
        #      symlinks fail)
        for label, path, floor in self._weight_paths():
            real = os.path.realpath(path) if path else ""
            if not real or not os.path.exists(real):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s not found at %r (set the matching OTR_LTX_AV_* env)"
                    % (self.name, label, path), kind="video")
            if os.path.getsize(real) < floor:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s at %r is below the %d-byte floor (truncated/wrong "
                    "file?)" % (self.name, label, real, floor), kind="video")
        # 6 -- dims on the provided canvas (None tolerated); wrap any ValueError
        if request_template is not None:
            try:
                w, h = self._canvas_dims(request_template)
                if w and h:
                    _AVD.assert_ltx_dims(w, h, _LTX_AV_MIN_FRAMES)
            except EngineUnusable:
                raise
            except Exception as exc:  # noqa: BLE001 - no raw ValueError escapes
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s canvas dims invalid for LTX: %s" % (self.name, exc),
                    kind="video")
        return self.name

    # ---- graph spec (M0-grounded; classes resolve via wrapper_bridge) ----
    def _node_candidates(self):
        cands = {
            "unet": ("UnetLoaderGGUF",),
            "te": ("LTXAVTextEncoderLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "cond": ("LTXVConditioning",),
            "modelsampling": ("ModelSamplingLTXV",),
            "videovae": ("VAELoader",),
            "audiovae": ("VAELoader",),
            "loadaudio": ("LoadAudio",),
            "audioenc": ("LTXVAudioVAEEncode",),
            "concat": ("LTXVConcatAVLatent",),
            "noise": ("RandomNoise",),
            "ksel": ("KSamplerSelect",),
            "sched": ("LTXVScheduler",),
            "guider": ("CFGGuider",),
            "sampler": ("SamplerCustomAdvanced",),
            "separate": ("LTXVSeparateAVLatent",),
            "decode": ("VAEDecodeTiled",),
        }
        if self._is_talk:
            cands["loadimage"] = ("LoadImage",)
            cands["i2v"] = ("LTXVImgToVideo",)
        else:
            cands["emptylatent"] = ("EmptyLTXVLatentVideo",)
        return cands

    def _build_graph(self, plan, length, width, height, audio_name, image_name):
        """The declarative LTX-AV A2V graph (wrapper_bridge.run_graph format).

        Common: GGUF unet -> ModelSamplingLTXV; LTXAVTextEncoderLoader (Gemma-3 on
        CPU) -> pos/neg CLIPTextEncode -> LTXVConditioning; VAELoader x2
        (video+audio); LoadAudio -> LTXVAudioVAEEncode. Talk branch: LoadImage ->
        LTXVImgToVideo (I2V conditioning) -> video latent. Music branch:
        EmptyLTXVLatentVideo. Both: LTXVConcatAVLatent(video,audio) ->
        SamplerCustomAdvanced(LTXVScheduler/euler/CFGGuider) -> LTXVSeparateAVLatent
        -> VAEDecodeTiled(video only; audio latent DROPPED, V-1)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        positive = plan.get("text_prompt") or "a vintage radio broadcast scene"
        negative = os.environ.get("OTR_LTX_AV_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        seed = int(plan.get("seed", 0) or 0)
        g = {
            "unet": {"class": "unet", "inputs": {"unet_name": self._unet_name()}},
            "te": {"class": "te", "inputs": {
                "text_encoder": self._encoder_name(),
                "ckpt_name": self._projection_ckpt(), "device": "cpu"}},
            "pos": {"class": "pos", "inputs": {"text": positive, "clip": W("te", 0)}},
            "neg": {"class": "neg", "inputs": {"text": negative, "clip": W("te", 0)}},
            "cond": {"class": "cond", "inputs": {
                "positive": W("pos", 0), "negative": W("neg", 0),
                "frame_rate": float(self.target_fps)}},
            "modelsampling": {"class": "modelsampling", "inputs": {
                "model": W("unet", 0), "max_shift": 2.05, "base_shift": 0.95}},
            "videovae": {"class": "videovae",
                         "inputs": {"vae_name": self._video_vae_name()}},
            "audiovae": {"class": "audiovae",
                         "inputs": {"vae_name": self._audio_vae_name()}},
            "loadaudio": {"class": "loadaudio", "inputs": {"audio": audio_name}},
            "audioenc": {"class": "audioenc", "inputs": {
                "audio": W("loadaudio", 0), "audio_vae": W("audiovae", 0)}},
            "noise": {"class": "noise", "inputs": {"noise_seed": seed}},
            "ksel": {"class": "ksel", "inputs": {"sampler_name": "euler"}},
        }
        if self._is_talk:
            g["loadimage"] = {"class": "loadimage", "inputs": {"image": image_name}}
            g["i2v"] = {"class": "i2v", "inputs": {
                "positive": W("cond", 0), "negative": W("cond", 1),
                "vae": W("videovae", 0), "image": W("loadimage", 0),
                "width": int(width), "height": int(height), "length": int(length),
                "batch_size": 1, "strength": _LTX_AV_I2V_STRENGTH}}
            video_latent = W("i2v", 2)
            guider_pos, guider_neg = W("i2v", 0), W("i2v", 1)
        else:
            g["emptylatent"] = {"class": "emptylatent", "inputs": {
                "width": int(width), "height": int(height),
                "length": int(length), "batch_size": 1}}
            video_latent = W("emptylatent", 0)
            guider_pos, guider_neg = W("cond", 0), W("cond", 1)
        g["concat"] = {"class": "concat", "inputs": {
            "video_latent": video_latent, "audio_latent": W("audioenc", 0)}}
        g["guider"] = {"class": "guider", "inputs": {
            "model": W("modelsampling", 0), "positive": guider_pos,
            "negative": guider_neg, "cfg": _LTX_AV_CFG}}
        g["sched"] = {"class": "sched", "inputs": {
            "steps": _LTX_AV_STEPS, "max_shift": 2.05, "base_shift": 0.95,
            "stretch": True, "terminal": 0.1, "latent": W("concat", 0)}}
        g["sampler"] = {"class": "sampler", "inputs": {
            "noise": W("noise", 0), "guider": W("guider", 0),
            "sampler": W("ksel", 0), "sigmas": W("sched", 0),
            "latent_image": W("concat", 0)}}
        g["separate"] = {"class": "separate",
                         "inputs": {"av_latent": W("sampler", 0)}}
        g["decode"] = {"class": "decode", "inputs": {
            "samples": W("separate", 0), "vae": W("videovae", 0),
            "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 8}}
        return g

    # ---- residency ----
    def load(self):
        """Resolve the installed ComfyUI node classes (fail-closed NAMED if
        absent). The heavy weight load happens when the loader nodes execute in
        render_clip (ComfyUI's own model management); the AS-3 lease brackets the
        real residency."""
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE audio-conditioned clip via the in-process LTX-AV graph and
        encode the decoded IMAGE batch to a SILENT bt709 clip (V-1: the audio
        latent is dropped at LTXVSeparateAVLatent; only the mux adds audio)."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)
        if not plan["audio_path"]:
            raise _wb.GraphExecutionError(
                "%s requires audio_ref (got %r)" % (self.name, plan["audio_path"]))
        if self._is_talk and not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "%s (talk) requires init_image (got %r)"
                % (self.name, plan["init_image"]))
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        audio_name = _wb.stage_into_comfy_input(plan["audio_path"])
        image_name = (_wb.stage_into_comfy_input(plan["init_image"])
                      if self._is_talk and plan["init_image"] else "")
        width, height = self._render_dims(request)
        length = _AVD.next_8n1(plan["target_frame_count"] or self.target_fps)
        if length > _LTX_AV_MAX_FRAMES:
            length = _AVD.next_8n1(_LTX_AV_MAX_FRAMES)
            if (length - 1) % _AVD._LTX_TEMPORAL_BASE != 0:
                length = _LTX_AV_MAX_FRAMES
        _AVD.assert_ltx_dims(width, height, length)
        graph = self._build_graph(plan, length, width, height, audio_name, image_name)
        results = _wb.run_graph(graph, classes)
        images = results[self._TERMINAL][0]
        self._retain_model_patchers(results, prepared)
        frames = _wb.images_to_uint8(images)
        out_path = otr_engine_tmp_mp4("otr_ltx_av_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # BUG-291 reclaim (LOUD; never unload_all): evict the umt5/Gemma encoder +
        # idle patchers so the resident stack drops under the ceiling before the
        # PASS-PM assert and the next beat starts drained.
        _wb.reclaim_idle_models(reason="%s post-decode" % self.name)
        if not os.environ.get("OTR_TEST_MODE"):
            _MC.assert_vram_within_ceiling("%s-render" % self.name)
        return {"out_path": path, "frame_count": n}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)

    def _retain_model_patchers(self, results, prepared):
        """Best-effort V-4: keep the MODEL ModelPatchers the graph produced so
        teardown can detach(unpatch_all=True) them."""
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        for nid in ("unet", "modelsampling"):
            out = results.get(nid)
            if not out:
                continue
            obj = out[0]
            if id(obj) not in seen and callable(getattr(obj, "detach", None)):
                bucket.append(obj)
                seen.add(id(obj))

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    @staticmethod
    def _ref_path(ref):
        """Pull a filesystem path out of an audio_ref / init_image that may be a
        bare string OR a mapping carrying a ``path`` key (the AudioRef shape)."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _init_image_ref(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            return assets.get("init_image") or ""
        return ""

    def _canvas_dims(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        return int(c_get("w", 0) or 0), int(c_get("h", 0) or 0)

    def _render_dims(self, request):
        """The render canvas: request.canvas.w/h when present, else the native
        1472x832. Snapped to LTX's 32-multiple via assert_ltx_dims downstream."""
        w, h = self._canvas_dims(request)
        if w and h:
            return w, h
        return _LTX_AV_NATIVE_W, _LTX_AV_NATIVE_H

    def _build_render_request(self, request):
        """Pure: the normalized inference request the LTX-AV graph consumes."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return {
            "init_image": self._init_image_ref(request),
            "audio_path": self._ref_path(get("audio_ref")),
            "text_prompt": get("text_prompt") or "",
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
        }

    def _clip_from_raw(self, raw, request):
        """Pure: shape a wrapper result into the silent CanonicalClip dict
        (bt709 / yuv420p; has_audio False -- only OTR_MasterAudioMux adds audio)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "ltx_av_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
        }


@register
class LtxAvTalkEngine(_LtxAvBase):
    """LTX-AV talk lane: lip-sync attempt from the FLUX still (I2V) + the audio
    slice. Reuses the ``audio_driven_face`` family; degrades to HuMo (then the
    HuMo 1.7B tier -> latentsync -> still floor)."""

    name = "ltx_av_talk"
    family = "audio_driven_face"
    roles = ("announcer_visual", "character_video")
    required_inputs = ("text_prompt", "audio_ref", "init_image")
    fallback_engine = "humo"
    _is_talk = True


@register
class LtxAvMusicEngine(_LtxAvBase):
    """LTX-AV music/scene lane: audio-reactive scene motion (sync-loose; the
    visuals breathe with the track). NEW ``audio_conditioned_video`` family;
    degrades to the golden prompt-only ``ltx_video`` (then the still floor)."""

    name = "ltx_av_music"
    family = "audio_conditioned_video"
    roles = ("music_visual",)
    required_inputs = ("text_prompt", "audio_ref")
    fallback_engine = "ltx_video"
    _is_talk = False


__all__ = ["LtxAvTalkEngine", "LtxAvMusicEngine"]


--- GROUNDING 2/3: workflows/ltx_av_talk_mini_repro_gguf_mit.json ---

{
  "last_node_id": 26,
  "last_link_id": 32,
  "nodes": [
    {
      "id": 1,
      "type": "UnetLoaderGGUF",
      "pos": [
        80,
        60
      ],
      "size": [
        330,
        80
      ],
      "flags": {},
      "order": 0,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "MODEL",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            1
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "UnetLoaderGGUF"
      },
      "widgets_values": [
        "ltx-2.3-22b-dev-Q3_K_M.gguf"
      ]
    },
    {
      "id": 2,
      "type": "ModelSamplingLTXV",
      "pos": [
        80,
        190
      ],
      "size": [
        330,
        90
      ],
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {
          "name": "model",
          "type": "MODEL",
          "link": 1
        }
      ],
      "outputs": [
        {
          "name": "MODEL",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            2
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "ModelSamplingLTXV"
      },
      "widgets_values": [
        2.05,
        0.95
      ]
    },
    {
      "id": 3,
      "type": "VAELoader",
      "pos": [
        80,
        330
      ],
      "size": [
        330,
        80
      ],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "VAE",
          "type": "VAE",
          "slot_index": 0,
          "links": [
            3,
            4
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "VAELoader"
      },
      "widgets_values": [
        "ltx-2.3-22b-dev_video_vae.safetensors"
      ]
    },
    {
      "id": 4,
      "type": "VAELoader",
      "pos": [
        80,
        460
      ],
      "size": [
        330,
        80
      ],
      "flags": {},
      "order": 3,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "VAE",
          "type": "VAE",
          "slot_index": 0,
          "links": [
            5,
            6
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "VAELoader"
      },
      "widgets_values": [
        "ltx-2.3-22b-dev_audio_vae.safetensors"
      ]
    },
    {
      "id": 5,
      "type": "LTXAVTextEncoderLoader",
      "pos": [
        80,
        590
      ],
      "size": [
        330,
        110
      ],
      "flags": {},
      "order": 4,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "CLIP",
          "type": "CLIP",
          "slot_index": 0,
          "links": [
            7,
            8
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXAVTextEncoderLoader"
      },
      "widgets_values": [
        "gemma_3_12B_it_fp4_mixed.safetensors",
        "ltx-2.3-22b-dev.safetensors",
        "cpu"
      ]
    },
    {
      "id": 6,
      "type": "CLIPTextEncode",
      "pos": [
        470,
        60
      ],
      "size": [
        390,
        140
      ],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {
          "name": "clip",
          "type": "CLIP",
          "link": 7
        }
      ],
      "outputs": [
        {
          "name": "CONDITIONING",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            9
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "CLIPTextEncode"
      },
      "widgets_values": [
        "Medium close-up of HAYES VANCE, a 50s lead researcher -- oval face, heavy brows, aquiline nose, thin lips, strong jawline, a small scar on the left cheek; shoulders hunched, speaking with tense, intrigued unease. Behind him a bustling neon-lit metropolis glows through rain-streaked glass at midnight. He has just received an anonymous tip that NeuroTech is hiding something linking diabetes and dementia. Cinematic 35mm film, moody volumetric lighting, sharp focus, talking head, subtle natural head motion."
      ]
    },
    {
      "id": 7,
      "type": "CLIPTextEncode",
      "pos": [
        470,
        240
      ],
      "size": [
        390,
        130
      ],
      "flags": {},
      "order": 6,
      "mode": 0,
      "inputs": [
        {
          "name": "clip",
          "type": "CLIP",
          "link": 8
        }
      ],
      "outputs": [
        {
          "name": "CONDITIONING",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            10
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "CLIPTextEncode"
      },
      "widgets_values": [
        "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, static, frozen pose, still image, watermark, text"
      ]
    },
    {
      "id": 8,
      "type": "LTXVConditioning",
      "pos": [
        470,
        410
      ],
      "size": [
        320,
        100
      ],
      "flags": {},
      "order": 7,
      "mode": 0,
      "inputs": [
        {
          "name": "positive",
          "type": "CONDITIONING",
          "link": 9
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "link": 10
        }
      ],
      "outputs": [
        {
          "name": "positive",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            11
          ]
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "slot_index": 1,
          "links": [
            12
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVConditioning"
      },
      "widgets_values": [
        25.0
      ]
    },
    {
      "id": 9,
      "type": "LoadImage",
      "pos": [
        80,
        760
      ],
      "size": [
        330,
        314
      ],
      "flags": {},
      "order": 8,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            13
          ]
        },
        {
          "name": "MASK",
          "type": "MASK",
          "slot_index": 1,
          "links": []
        }
      ],
      "properties": {
        "Node name for S&R": "LoadImage"
      },
      "widgets_values": [
        "c02_466a19906ccb.png",
        "image"
      ]
    },
    {
      "id": 10,
      "type": "LoadAudio",
      "pos": [
        80,
        1110
      ],
      "size": [
        330,
        100
      ],
      "flags": {},
      "order": 9,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "AUDIO",
          "type": "AUDIO",
          "slot_index": 0,
          "links": [
            14,
            15
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LoadAudio"
      },
      "widgets_values": [
        "c02_b002_line.wav"
      ]
    },
    {
      "id": 11,
      "type": "LTXVAudioVAEEncode",
      "pos": [
        470,
        560
      ],
      "size": [
        300,
        80
      ],
      "flags": {},
      "order": 10,
      "mode": 0,
      "inputs": [
        {
          "name": "audio",
          "type": "AUDIO",
          "link": 14
        },
        {
          "name": "audio_vae",
          "type": "VAE",
          "link": 5
        }
      ],
      "outputs": [
        {
          "name": "LATENT",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            16
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVAudioVAEEncode"
      },
      "widgets_values": []
    },
    {
      "id": 12,
      "type": "LTXVImgToVideo",
      "pos": [
        900,
        60
      ],
      "size": [
        340,
        200
      ],
      "flags": {},
      "order": 11,
      "mode": 0,
      "inputs": [
        {
          "name": "positive",
          "type": "CONDITIONING",
          "link": 11
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "link": 12
        },
        {
          "name": "vae",
          "type": "VAE",
          "link": 3
        },
        {
          "name": "image",
          "type": "IMAGE",
          "link": 13
        }
      ],
      "outputs": [
        {
          "name": "positive",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            17
          ]
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "slot_index": 1,
          "links": [
            18
          ]
        },
        {
          "name": "latent",
          "type": "LATENT",
          "slot_index": 2,
          "links": [
            19
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVImgToVideo"
      },
      "widgets_values": [
        832,
        480,
        105,
        1,
        1.0
      ]
    },
    {
      "id": 13,
      "type": "LTXVConcatAVLatent",
      "pos": [
        900,
        300
      ],
      "size": [
        320,
        90
      ],
      "flags": {},
      "order": 12,
      "mode": 0,
      "inputs": [
        {
          "name": "video_latent",
          "type": "LATENT",
          "link": 19
        },
        {
          "name": "audio_latent",
          "type": "LATENT",
          "link": 16
        }
      ],
      "outputs": [
        {
          "name": "LATENT",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            20,
            21
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVConcatAVLatent"
      },
      "widgets_values": []
    },
    {
      "id": 14,
      "type": "KSamplerSelect",
      "pos": [
        900,
        430
      ],
      "size": [
        300,
        60
      ],
      "flags": {},
      "order": 13,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "SAMPLER",
          "type": "SAMPLER",
          "slot_index": 0,
          "links": [
            22
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "KSamplerSelect"
      },
      "widgets_values": [
        "euler"
      ]
    },
    {
      "id": 15,
      "type": "LTXVScheduler",
      "pos": [
        900,
        530
      ],
      "size": [
        340,
        150
      ],
      "flags": {},
      "order": 14,
      "mode": 0,
      "inputs": [
        {
          "name": "latent",
          "type": "LATENT",
          "link": 20
        }
      ],
      "outputs": [
        {
          "name": "SIGMAS",
          "type": "SIGMAS",
          "slot_index": 0,
          "links": [
            23
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVScheduler"
      },
      "widgets_values": [
        8,
        2.05,
        0.95,
        true,
        0.1
      ]
    },
    {
      "id": 16,
      "type": "RandomNoise",
      "pos": [
        900,
        720
      ],
      "size": [
        300,
        82
      ],
      "flags": {},
      "order": 15,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "NOISE",
          "type": "NOISE",
          "slot_index": 0,
          "links": [
            24
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "RandomNoise"
      },
      "widgets_values": [
        1,
        "fixed"
      ]
    },
    {
      "id": 17,
      "type": "CFGGuider",
      "pos": [
        1300,
        300
      ],
      "size": [
        300,
        98
      ],
      "flags": {},
      "order": 16,
      "mode": 0,
      "inputs": [
        {
          "name": "model",
          "type": "MODEL",
          "link": 2
        },
        {
          "name": "positive",
          "type": "CONDITIONING",
          "link": 17
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "link": 18
        }
      ],
      "outputs": [
        {
          "name": "GUIDER",
          "type": "GUIDER",
          "slot_index": 0,
          "links": [
            25
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "CFGGuider"
      },
      "widgets_values": [
        3.0
      ]
    },
    {
      "id": 18,
      "type": "SamplerCustomAdvanced",
      "pos": [
        1660,
        300
      ],
      "size": [
        320,
        150
      ],
      "flags": {},
      "order": 17,
      "mode": 0,
      "inputs": [
        {
          "name": "noise",
          "type": "NOISE",
          "link": 24
        },
        {
          "name": "guider",
          "type": "GUIDER",
          "link": 25
        },
        {
          "name": "sampler",
          "type": "SAMPLER",
          "link": 22
        },
        {
          "name": "sigmas",
          "type": "SIGMAS",
          "link": 23
        },
        {
          "name": "latent_image",
          "type": "LATENT",
          "link": 21
        }
      ],
      "outputs": [
        {
          "name": "output",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            26
          ]
        },
        {
          "name": "denoised_output",
          "type": "LATENT",
          "slot_index": 1,
          "links": []
        }
      ],
      "properties": {
        "Node name for S&R": "SamplerCustomAdvanced"
      },
      "widgets_values": []
    },
    {
      "id": 19,
      "type": "LTXVSeparateAVLatent",
      "pos": [
        2040,
        320
      ],
      "size": [
        300,
        80
      ],
      "flags": {},
      "order": 18,
      "mode": 0,
      "inputs": [
        {
          "name": "av_latent",
          "type": "LATENT",
          "link": 26
        }
      ],
      "outputs": [
        {
          "name": "video_latent",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            27
          ]
        },
        {
          "name": "audio_latent",
          "type": "LATENT",
          "slot_index": 1,
          "links": [
            28
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVSeparateAVLatent"
      },
      "widgets_values": []
    },
    {
      "id": 20,
      "type": "VAEDecodeTiled",
      "pos": [
        2380,
        280
      ],
      "size": [
        320,
        130
      ],
      "flags": {},
      "order": 19,
      "mode": 0,
      "inputs": [
        {
          "name": "samples",
          "type": "LATENT",
          "link": 27
        },
        {
          "name": "vae",
          "type": "VAE",
          "link": 4
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            29,
            30
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "VAEDecodeTiled"
      },
      "widgets_values": [
        512,
        64,
        64,
        8
      ]
    },
    {
      "id": 21,
      "type": "SaveWEBM",
      "pos": [
        2760,
        240
      ],
      "size": [
        340,
        150
      ],
      "flags": {},
      "order": 20,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 29
        }
      ],
      "outputs": [],
      "properties": {
        "Node name for S&R": "SaveWEBM"
      },
      "widgets_values": [
        "otr_ltx_av_talk/silent_b002",
        "vp9",
        25.0,
        18.0
      ]
    },
    {
      "id": 22,
      "type": "Note",
      "pos": [
        1300,
        470
      ],
      "size": [
        540,
        470
      ],
      "flags": {},
      "order": 21,
      "mode": 0,
      "inputs": [],
      "outputs": [],
      "properties": {
        "Node name for S&R": "Note"
      },
      "widgets_values": [
        "LTX-AV TALK (audio_driven_face) mini -- CURRENT recipe from eng_ltx_av.py, grounded on a REAL beat for an apples-to-apples lip-sync test.\n\nREAL PRE-BAKED INPUTS (non-generative) from episode signal_lost_keystrokes_of_denial_20260617:\n  LoadImage = c02_466a19906ccb.png  (HAYES VANCE character portrait)\n  LoadAudio = c02_b002_line.wav  (his real beat-b002 line, sliced from the episode master mix at 28.412s for 13.28s -- per-beat audio is NOT stored as a standalone wav, so it was cut from the master). Swap for vz_bill_boerst.wav (indextts2 voice ref) if preferred.\n  positive prompt = the real per-beat shot description (Hayes, anonymous tip).\n\nTHREE OUTPUTS:\n  (1) SaveWEBM 'silent_b002' = the production-faithful SILENT clip (V-1: the audio latent is dropped).\n  (2) SaveVideo 'with_line_audio_b002' = the clip MUXED with the driving line audio so you can HEAR it (core CreateVideo+SaveVideo, license-clean, no VHS).\n  (3) SaveAudio 'model_audio_passthrough_b002' = the LTX-AV model's OWN generated audio, decoded from LTXVSeparateAVLatent's audio_latent via LTXVAudioVAEDecode -- this is what production DISCARDS (V-1). Listen to judge whether the A2V audio is usable.\n\nRECIPE DELTAS vs the video mini: no distilled LoRA; ModelSamplingLTXV 2.05/0.95; in-path audio LoadAudio->LTXVAudioVAEEncode->LTXVConcatAVLatent; KSamplerSelect=euler; LTXVScheduler steps=8; CFG=3.0; LTXVImgToVideo strength=1.0; Gemma device=cpu; VAEDecodeTiled temporal_size=64. Canvas 832x480x105; if VRAM exceeds 14.5GB drop LTXVImgToVideo width/height to 512x288 (OTR_LTX_AV_RENDER_CANVAS)."
      ]
    },
    {
      "id": 23,
      "type": "CreateVideo",
      "pos": [
        2760,
        470
      ],
      "size": [
        320,
        100
      ],
      "flags": {},
      "order": 22,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 30
        },
        {
          "name": "audio",
          "type": "AUDIO",
          "link": 15
        }
      ],
      "outputs": [
        {
          "name": "VIDEO",
          "type": "VIDEO",
          "slot_index": 0,
          "links": [
            31
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "CreateVideo"
      },
      "widgets_values": [
        25.0,
        8
      ]
    },
    {
      "id": 24,
      "type": "SaveVideo",
      "pos": [
        3140,
        450
      ],
      "size": [
        340,
        130
      ],
      "flags": {},
      "order": 23,
      "mode": 0,
      "inputs": [
        {
          "name": "video",
          "type": "VIDEO",
          "link": 31
        }
      ],
      "outputs": [],
      "properties": {
        "Node name for S&R": "SaveVideo"
      },
      "widgets_values": [
        "otr_ltx_av_talk/with_line_audio_b002",
        "mp4",
        "h264"
      ]
    },
    {
      "id": 25,
      "type": "LTXVAudioVAEDecode",
      "pos": [
        2380,
        470
      ],
      "size": [
        320,
        80
      ],
      "flags": {},
      "order": 24,
      "mode": 0,
      "inputs": [
        {
          "name": "samples",
          "type": "LATENT",
          "link": 28
        },
        {
          "name": "audio_vae",
          "type": "VAE",
          "link": 6
        }
      ],
      "outputs": [
        {
          "name": "AUDIO",
          "type": "AUDIO",
          "slot_index": 0,
          "links": [
            32
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVAudioVAEDecode"
      },
      "widgets_values": []
    },
    {
      "id": 26,
      "type": "SaveAudio",
      "pos": [
        2380,
        600
      ],
      "size": [
        340,
        100
      ],
      "flags": {},
      "order": 25,
      "mode": 0,
      "inputs": [
        {
          "name": "audio",
          "type": "AUDIO",
          "link": 32
        }
      ],
      "outputs": [],
      "properties": {
        "Node name for S&R": "SaveAudio"
      },
      "widgets_values": [
        "otr_ltx_av_talk/model_audio_passthrough_b002"
      ]
    }
  ],
  "links": [
    [
      1,
      1,
      0,
      2,
      0,
      "MODEL"
    ],
    [
      2,
      2,
      0,
      17,
      0,
      "MODEL"
    ],
    [
      3,
      3,
      0,
      12,
      2,
      "VAE"
    ],
    [
      4,
      3,
      0,
      20,
      1,
      "VAE"
    ],
    [
      5,
      4,
      0,
      11,
      1,
      "VAE"
    ],
    [
      6,
      4,
      0,
      25,
      1,
      "VAE"
    ],
    [
      7,
      5,
      0,
      6,
      0,
      "CLIP"
    ],
    [
      8,
      5,
      0,
      7,
      0,
      "CLIP"
    ],
    [
      9,
      6,
      0,
      8,
      0,
      "CONDITIONING"
    ],
    [
      10,
      7,
      0,
      8,
      1,
      "CONDITIONING"
    ],
    [
      11,
      8,
      0,
      12,
      0,
      "CONDITIONING"
    ],
    [
      12,
      8,
      1,
      12,
      1,
      "CONDITIONING"
    ],
    [
      13,
      9,
      0,
      12,
      3,
      "IMAGE"
    ],
    [
      14,
      10,
      0,
      11,
      0,
      "AUDIO"
    ],
    [
      15,
      10,
      0,
      23,
      1,
      "AUDIO"
    ],
    [
      16,
      11,
      0,
      13,
      1,
      "LATENT"
    ],
    [
      17,
      12,
      0,
      17,
      1,
      "CONDITIONING"
    ],
    [
      18,
      12,
      1,
      17,
      2,
      "CONDITIONING"
    ],
    [
      19,
      12,
      2,
      13,
      0,
      "LATENT"
    ],
    [
      20,
      13,
      0,
      15,
      0,
      "LATENT"
    ],
    [
      21,
      13,
      0,
      18,
      4,
      "LATENT"
    ],
    [
      22,
      14,
      0,
      18,
      2,
      "SAMPLER"
    ],
    [
      23,
      15,
      0,
      18,
      3,
      "SIGMAS"
    ],
    [
      24,
      16,
      0,
      18,
      0,
      "NOISE"
    ],
    [
      25,
      17,
      0,
      18,
      1,
      "GUIDER"
    ],
    [
      26,
      18,
      0,
      19,
      0,
      "LATENT"
    ],
    [
      27,
      19,
      0,
      20,
      0,
      "LATENT"
    ],
    [
      28,
      19,
      1,
      25,
      0,
      "LATENT"
    ],
    [
      29,
      20,
      0,
      21,
      0,
      "IMAGE"
    ],
    [
      30,
      20,
      0,
      23,
      0,
      "IMAGE"
    ],
    [
      31,
      23,
      0,
      24,
      0,
      "VIDEO"
    ],
    [
      32,
      25,
      0,
      26,
      0,
      "AUDIO"
    ]
  ],
  "groups": [],
  "config": {},
  "extra": {},
  "version": 0.4
}

--- GROUNDING 3/3: workflows/ltx_bookend_mini_repro_gguf_mit.json ---

{
  "last_node_id": 17,
  "last_link_id": 19,
  "nodes": [
    {
      "id": 1,
      "type": "UnetLoaderGGUF",
      "pos": [
        80,
        60
      ],
      "size": [
        330,
        80
      ],
      "flags": {},
      "order": 0,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "MODEL",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            1
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "UnetLoaderGGUF"
      },
      "widgets_values": [
        "ltx-2.3-22b-dev-Q3_K_M.gguf"
      ]
    },
    {
      "id": 2,
      "type": "LoraLoaderModelOnly",
      "pos": [
        80,
        200
      ],
      "size": [
        330,
        110
      ],
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {
          "name": "model",
          "type": "MODEL",
          "link": 1
        }
      ],
      "outputs": [
        {
          "name": "MODEL",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            9
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LoraLoaderModelOnly"
      },
      "widgets_values": [
        "ltxv\\ltx2\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        0.7
      ]
    },
    {
      "id": 3,
      "type": "VAELoader",
      "pos": [
        80,
        360
      ],
      "size": [
        330,
        80
      ],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "VAE",
          "type": "VAE",
          "slot_index": 0,
          "links": [
            4,
            18
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "VAELoader"
      },
      "widgets_values": [
        "ltx-2.3-22b-dev_video_vae.safetensors"
      ]
    },
    {
      "id": 4,
      "type": "LTXAVTextEncoderLoader",
      "pos": [
        80,
        490
      ],
      "size": [
        330,
        110
      ],
      "flags": {},
      "order": 3,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "CLIP",
          "type": "CLIP",
          "slot_index": 0,
          "links": [
            2,
            3
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXAVTextEncoderLoader"
      },
      "widgets_values": [
        "gemma_3_12B_it_fp4_mixed.safetensors",
        "ltx-2.3-22b-dev.safetensors",
        "default"
      ]
    },
    {
      "id": 5,
      "type": "CLIPTextEncode",
      "pos": [
        470,
        60
      ],
      "size": [
        380,
        120
      ],
      "flags": {},
      "order": 4,
      "mode": 0,
      "inputs": [
        {
          "name": "clip",
          "type": "CLIP",
          "link": 2
        }
      ],
      "outputs": [
        {
          "name": "CONDITIONING",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            7
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "CLIPTextEncode"
      },
      "widgets_values": [
        "Continuous shot, a dramatic scene from a sci-fi radio drama"
      ]
    },
    {
      "id": 6,
      "type": "CLIPTextEncode",
      "pos": [
        470,
        220
      ],
      "size": [
        380,
        130
      ],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {
          "name": "clip",
          "type": "CLIP",
          "link": 3
        }
      ],
      "outputs": [
        {
          "name": "CONDITIONING",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            8
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "CLIPTextEncode"
      },
      "widgets_values": [
        "low quality, worst quality, blurry, distorted, watermark, text, static"
      ]
    },
    {
      "id": 7,
      "type": "LoadImage",
      "pos": [
        80,
        640
      ],
      "size": [
        330,
        314
      ],
      "flags": {},
      "order": 6,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            5
          ]
        },
        {
          "name": "MASK",
          "type": "MASK",
          "slot_index": 1,
          "links": []
        }
      ],
      "properties": {
        "Node name for S&R": "LoadImage"
      },
      "widgets_values": [
        "radio_bookend_mimicry_b001.png",
        "image"
      ]
    },
    {
      "id": 8,
      "type": "EmptyLTXVLatentVideo",
      "pos": [
        470,
        400
      ],
      "size": [
        330,
        130
      ],
      "flags": {},
      "order": 7,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "LATENT",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            6
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "EmptyLTXVLatentVideo"
      },
      "widgets_values": [
        832,
        480,
        105,
        1
      ]
    },
    {
      "id": 9,
      "type": "LTXVImgToVideoConditionOnly",
      "pos": [
        900,
        120
      ],
      "size": [
        340,
        120
      ],
      "flags": {},
      "order": 8,
      "mode": 0,
      "inputs": [
        {
          "name": "vae",
          "type": "VAE",
          "link": 4
        },
        {
          "name": "image",
          "type": "IMAGE",
          "link": 5
        },
        {
          "name": "latent",
          "type": "LATENT",
          "link": 6
        }
      ],
      "outputs": [
        {
          "name": "latent",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            16
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVImgToVideoConditionOnly"
      },
      "widgets_values": [
        0.75,
        false
      ]
    },
    {
      "id": 10,
      "type": "LTXVConditioning",
      "pos": [
        900,
        320
      ],
      "size": [
        320,
        100
      ],
      "flags": {},
      "order": 9,
      "mode": 0,
      "inputs": [
        {
          "name": "positive",
          "type": "CONDITIONING",
          "link": 7
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "link": 8
        }
      ],
      "outputs": [
        {
          "name": "positive",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            10
          ]
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "slot_index": 1,
          "links": [
            11
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "LTXVConditioning"
      },
      "widgets_values": [
        25.0
      ]
    },
    {
      "id": 11,
      "type": "KSamplerSelect",
      "pos": [
        900,
        470
      ],
      "size": [
        300,
        60
      ],
      "flags": {},
      "order": 10,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "SAMPLER",
          "type": "SAMPLER",
          "slot_index": 0,
          "links": [
            14
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "KSamplerSelect"
      },
      "widgets_values": [
        "euler_cfg_pp"
      ]
    },
    {
      "id": 12,
      "type": "ManualSigmas",
      "pos": [
        900,
        580
      ],
      "size": [
        340,
        80
      ],
      "flags": {},
      "order": 11,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "SIGMAS",
          "type": "SIGMAS",
          "slot_index": 0,
          "links": [
            15
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "ManualSigmas"
      },
      "widgets_values": [
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
      ]
    },
    {
      "id": 13,
      "type": "RandomNoise",
      "pos": [
        900,
        710
      ],
      "size": [
        300,
        82
      ],
      "flags": {},
      "order": 12,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "NOISE",
          "type": "NOISE",
          "slot_index": 0,
          "links": [
            12
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "RandomNoise"
      },
      "widgets_values": [
        1,
        "fixed"
      ]
    },
    {
      "id": 14,
      "type": "CFGGuider",
      "pos": [
        1300,
        330
      ],
      "size": [
        300,
        98
      ],
      "flags": {},
      "order": 13,
      "mode": 0,
      "inputs": [
        {
          "name": "model",
          "type": "MODEL",
          "link": 9
        },
        {
          "name": "positive",
          "type": "CONDITIONING",
          "link": 10
        },
        {
          "name": "negative",
          "type": "CONDITIONING",
          "link": 11
        }
      ],
      "outputs": [
        {
          "name": "GUIDER",
          "type": "GUIDER",
          "slot_index": 0,
          "links": [
            13
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "CFGGuider"
      },
      "widgets_values": [
        1.0
      ]
    },
    {
      "id": 15,
      "type": "SamplerCustomAdvanced",
      "pos": [
        1660,
        300
      ],
      "size": [
        320,
        150
      ],
      "flags": {},
      "order": 14,
      "mode": 0,
      "inputs": [
        {
          "name": "noise",
          "type": "NOISE",
          "link": 12
        },
        {
          "name": "guider",
          "type": "GUIDER",
          "link": 13
        },
        {
          "name": "sampler",
          "type": "SAMPLER",
          "link": 14
        },
        {
          "name": "sigmas",
          "type": "SIGMAS",
          "link": 15
        },
        {
          "name": "latent_image",
          "type": "LATENT",
          "link": 16
        }
      ],
      "outputs": [
        {
          "name": "output",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            17
          ]
        },
        {
          "name": "denoised_output",
          "type": "LATENT",
          "slot_index": 1,
          "links": []
        }
      ],
      "properties": {
        "Node name for S&R": "SamplerCustomAdvanced"
      },
      "widgets_values": []
    },
    {
      "id": 16,
      "type": "VAEDecodeTiled",
      "pos": [
        2040,
        300
      ],
      "size": [
        320,
        130
      ],
      "flags": {},
      "order": 15,
      "mode": 0,
      "inputs": [
        {
          "name": "samples",
          "type": "LATENT",
          "link": 17
        },
        {
          "name": "vae",
          "type": "VAE",
          "link": 18
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            19
          ]
        }
      ],
      "properties": {
        "Node name for S&R": "VAEDecodeTiled"
      },
      "widgets_values": [
        512,
        64,
        4096,
        8
      ]
    },
    {
      "id": 17,
      "type": "SaveWEBM",
      "pos": [
        2420,
        280
      ],
      "size": [
        340,
        150
      ],
      "flags": {},
      "order": 16,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 19
        }
      ],
      "outputs": [],
      "properties": {
        "Node name for S&R": "SaveWEBM"
      },
      "widgets_values": [
        "otr_ltx_gguf_mit/repro_b001",
        "vp9",
        25.0,
        18.0
      ]
    }
  ],
  "links": [
    [
      1,
      1,
      0,
      2,
      0,
      "MODEL"
    ],
    [
      2,
      4,
      0,
      5,
      0,
      "CLIP"
    ],
    [
      3,
      4,
      0,
      6,
      0,
      "CLIP"
    ],
    [
      4,
      3,
      0,
      9,
      0,
      "VAE"
    ],
    [
      5,
      7,
      0,
      9,
      1,
      "IMAGE"
    ],
    [
      6,
      8,
      0,
      9,
      2,
      "LATENT"
    ],
    [
      7,
      5,
      0,
      10,
      0,
      "CONDITIONING"
    ],
    [
      8,
      6,
      0,
      10,
      1,
      "CONDITIONING"
    ],
    [
      9,
      2,
      0,
      14,
      0,
      "MODEL"
    ],
    [
      10,
      10,
      0,
      14,
      1,
      "CONDITIONING"
    ],
    [
      11,
      10,
      1,
      14,
      2,
      "CONDITIONING"
    ],
    [
      12,
      13,
      0,
      15,
      0,
      "NOISE"
    ],
    [
      13,
      14,
      0,
      15,
      1,
      "GUIDER"
    ],
    [
      14,
      11,
      0,
      15,
      2,
      "SAMPLER"
    ],
    [
      15,
      12,
      0,
      15,
      3,
      "SIGMAS"
    ],
    [
      16,
      9,
      0,
      15,
      4,
      "LATENT"
    ],
    [
      17,
      15,
      0,
      16,
      0,
      "LATENT"
    ],
    [
      18,
      3,
      0,
      16,
      1,
      "VAE"
    ],
    [
      19,
      16,
      0,
      17,
      0,
      "IMAGE"
    ]
  ],
  "groups": [],
  "config": {},
  "extra": {},
  "version": 0.4
}

