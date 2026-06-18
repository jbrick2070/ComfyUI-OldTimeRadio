<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The spec is not build-ready because its required floor is cross-platform/8GB/commercial-clean, but the default loader, sampler set, Lightning artifacts, LoRA wiring, license, and acceptance criteria are still unresolved.

MUST-FIX BEFORE BUILD:
1. [HARD CONSTRAINTS 2] The “system-agnostic” requirement is unresolved while the grounded current default is `UnetLoaderGGUF` for `Wan2.2-TI2V-5B-Q5_K_M.gguf`. The document explicitly asks whether GGUF works on MPS/AMD instead of deciding it. Fix: define the supported default path per backend before build: either prove `UnetLoaderGGUF` works on CUDA/MPS/ROCm/DirectML, or add a safetensors fallback/default with `OTR_WAN_TI2V_LOADER=safetensors`, required model filename, dtype, and usability gating.

2. [Candidate recipes B/C] The Lightning/Distill LoRA candidates cannot be built from the grounded engine as written. `eng_wan_ti2v.py` has no LoRA node, no LoRA path/name env, no strength env, no loader presence check, and no wiring to feed a patched model into `KSampler`. Fix: add an optional default-off LoRA path with explicit `OTR_WAN_TI2V_LORA_NAME`/path, strength envs, object_info-verified node class/input schema, fail-closed model presence checks, and graph wiring that only activates when configured.

3. [HARD CONSTRAINTS 3] License compliance is not satisfied. The spec says “verify LightX2V license” but proposes B/C as candidates. Fix: pin exact artifact names, repository revisions, license files, and checksums for the 4-step and 6-step artifacts; update the model manifest; block B/C unless the artifact license is confirmed Apache-2.0/MIT-compatible.

4. [HARD CONSTRAINTS 1 / Current recipe] The stated 8GB floor is not proven and is contradicted by the only grounded smoke data: “engine vram ~8.2 GB, NVML peak 13.1 GB” on a 5080. Fix: add a required acceptance matrix for real 8GB CUDA, MPS, and AMD/DirectML or ROCm at 832x480, including offload settings, max frame count, success/fail criteria, and measured peak memory. Do not promote any default until it passes.

5. [Candidate recipes / Questions 3] B/C recipe parameters are underspecified. “4-step Lightning LoRA” and “6-step distill” do not define cfg, shift, sampler, scheduler, LoRA strength, negative prompt behavior, or whether CFG should be disabled/near-1.0. The grounded code defaults to `steps=30`, `cfg=5.0`, `shift=5.0`, `sampler=uni_pc`, `scheduler=simple`, which is likely not the intended 4-step distill configuration. Fix: provide a concrete recipe table for A/B/C/E with every env value and artifact required.

6. [HARD CONSTRAINTS 2 / Questions 4] Sampler portability is unresolved and currently unsafe. The grounded code passes `OTR_WAN_TI2V_SAMPLER` and `OTR_WAN_TI2V_SCHEDULER` directly into `KSampler` with no validation. The spec considers `uni_pc`, `sa_solver`, `MoEKSampler`, `euler/beta`, and `lcm/simple` without proving availability on MPS/AMD. Fix: validate sampler/scheduler values against object_info at startup or restrict the floor to a proven core pair; fail closed on unsupported env values.

7. [Current recipe] The frame count is described inconsistently as “length 25 (min 33)”. The grounded code quantizes `plan["target_frame_count"] or target_fps` with `_TI2V_MIN_FRAMES = 33`, so the practical default render is 33 frames, not 25. Fix: update the spec and A/B/C/E test recipes to use the actual default frame count and explicitly state the tested frame count.

8. [HARD CONSTRAINTS 1 / Questions 5] The floor does not define a safe frame-duration envelope. The grounded engine allows `_TI2V_MAX_FRAMES = 177`; memory and runtime risk scale materially with length. Fix: define the 8GB floor as a specific bounded recipe, e.g. 832x480 at 33 frames, and separately gate/validate any longer render lengths before exposing them as floor-safe.

9. [Sources / Candidate recipes] The build is not reproducible from links alone. The spec names repositories but does not pin model revisions, workflow revisions, object_info captures, or checksums. Fix: pin exact download URLs/revisions and expected filenames for every candidate artifact before implementing installer/usability logic.

SHOULD-FIX:
1. [HARD CONSTRAINTS 4] Determinism is asserted but not tested. The grounded code sets `seed`, but backend kernels/samplers may still be nondeterministic. Fix: define determinism acceptance per backend: same seed + same recipe + same backend must reproduce within a stated tolerance or hash target; document any backend exceptions.

2. [eng_wan_ti2v.py grounding / load sequencing] `assert_usable()` checks VAE/CLIP presence and the Wan2.2 VAE guard, but `load()` only checks `_installed()` for the UNET before resolving classes. [ASSUMPTION] If callers can invoke `load()` without `assert_usable()`, aux model failures move later into graph execution. Fix: either guarantee call order in the engine registry or make `load()` reuse the same fail-closed aux checks.

3. [Candidate recipes E] “euler/beta or sa_solver, shift 3.0” mixes two variables at once. Fix: split E into single-variable controls or define the exact comparison matrix; otherwise A/B results will not identify whether sampler, scheduler, or shift caused a difference.

4. [Questions 5] The spec does not define OOM/failure behavior for the low floor. Fix: require named fail-closed errors for unsupported backend, missing node, missing LoRA, unsupported sampler, and OOM; include fallback behavior if the chosen default fails on 8GB.

5. [Questions 3] VAE decode memory risk is not addressed. The grounded graph decodes the sampled latent through `VAEDecode` as a batch. [ASSUMPTION] Longer clips may spike memory during decode. Fix: verify decode peak at 33/177 frames and add chunked decode or a hard frame cap if needed.

6. [HARD CONSTRAINTS 5] “Default preserving current behavior unless promoted” and the request for a new 480p DEFAULT need a promotion gate. Fix: separate “experimental env-only candidates” from “promoted default” and require passing license/portability/8GB tests before changing defaults.

OPTIONAL / NICE-TO-HAVE:
- Add a small backend capability report to logs: loader mode, sampler, scheduler, steps, cfg, shift, frame count, backend, and detected VRAM.
- Add a one-command smoke profile for A/B/C/E that emits timing, peak VRAM, output path, and deterministic seed.
- Add explicit documentation that 720p and audio-in are intentionally rejected for this engine.

CUT THESE (over-engineering):
1. [Questions 4] Cut `MoEKSampler` from the floor plan unless it is proven core, cross-platform, and non-CUDA-specific. It adds dependency and portability risk with no need for the “solid low floor.”

2. [Questions 4 / Candidate recipes E] Cut `sa_solver` from the first floor build unless object_info and backend tests prove it works everywhere. A core `euler` or `lcm` path is enough for the accessibility target.

3. [Candidate recipes C] Cut the separate 6-step distill from the first build pass unless it uses the same artifact family and wiring as B. It adds another artifact/license/parameter/test axis before the basic 4-step LoRA path is even implemented.

4. [Search-first grounding] Do not carry 720p workflow support into this engine’s implementation. The spec already declares 720p out of scope; keeping any 720p knobs or tests in this lane only expands the validation matrix without serving the floor.