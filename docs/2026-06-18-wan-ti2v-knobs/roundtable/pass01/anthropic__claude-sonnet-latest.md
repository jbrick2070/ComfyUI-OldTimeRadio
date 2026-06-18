<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: no. Multiple MUST-FIX items before this spec can drive a build -- the central "8 GB floor" claim is immediately falsified by the only measured data point, two candidate recipes have zero code backing them, and a safety constraint (no CUDA-only ops) is actively violated by a positively-cited option.

---

## MUST-FIX BEFORE BUILD

**1. [GPU smoke / Hard Constraints §1] The 13.1 GB NVML peak directly falsifies the "8 GB VRAM floor" claim.**
The spec declares 8 GB as the target floor then immediately reports NVML peak 13.1 GB on the smoke platform. No mechanism for fitting this into 8 GB is specified — no CPU-offload tuning, no sequential-offload ComfyUI flag, no evidence from an actual 8 GB device. The "engine vram ~8.2 GB" figure appears to be model-weight residency only, not activation peak. Until the spec names *what* achieves 8 GB fit (e.g., `--lowvram`, `cpu_offload`, reduced `length`, a different quant level) and supplies a measured peak from an actual 8 GB device, the floor claim is asserted, not demonstrated. Fix: either profile on a real 8 GB GPU or specify and test an offload configuration; update the spec with the measured peak under that config.

**2. [Current recipe] "length 25 (min 33)" is doubly wrong and will mislead the A/B design.**
The code shows `_TI2V_MIN_FRAMES = 33`, `target_fps = 25`, and in `render_clip`:
```python
length = _wb.quantize_frames_4n1(
    plan["target_frame_count"] or self.target_fps,   # 25 = fps, not frames
    min_frames=_TI2V_MIN_FRAMES, ...)                # clamps to ≥33
```
`target_fps` (25 fps) is being misused as a frame-count fallback, and the result is always clamped to 33. The spec lists "length 25" as the default and "(min 33)" in the same breath — an impossible combination. Fix: correct the spec to state default rendered length is 33 frames (the clamped effective minimum); note `target_fps=25` is Hz, not frame count.

**3. [Candidate recipes B/C] Neither Lightning LoRA candidate has any code backing it.**
`_node_candidates()` has no LoRA node; `_build_graph` has no LoRA wiring. Candidates B and C require at minimum: a `LoraLoader` (or `LoraLoaderModelOnly`) node inserted between `unet` and `modelsampling`, new `_node_candidates` entries, and `OTR_WAN_TI2V_LORA_*` env-var definitions. The spec lists these as A/B candidates without specifying a single line of concrete code change. The A/B test cannot be built from this document as written.

**4. [Hard Constraints §3 / Q3] Lightning LoRA license is unresolved but the candidates depend on it.**
Hard Constraint §3 requires Apache-2.0/MIT. The spec says "verify LightX2V license" in Q3 while simultaneously listing candidates B and C. ModelTC/Wan2.2-Lightning (the base) is Apache-2.0, but `lightx2v/Wan2.2-Distill-Loras` on HuggingFace has a separate model card — its license is not confirmed in this document. If the LoRA weights are not Apache/MIT, candidates B and C violate the hard constraint and must be removed. Fix: resolve the license before the A/B spec is finalized; gate candidates B/C on that result.

**5. [Hard Constraints §2 / Sources] `MoEKSampler` is presented positively without being flagged as non-portable.**
The spec states "some report `MoEKSampler` beats plain `KSampler` for this model" — with no flag that this is a custom CUDA-only extension, not in core ComfyUI, and directly prohibited by Hard Constraint §2. It is not in `_node_candidates()`. Including it in guidance without an explicit NVIDIA-only / out-of-scope call is a contradictory signal. Fix: explicitly exclude `MoEKSampler` and label it CUDA-only / out-of-scope within this section.

**6. [Candidate B / CFG coupling] CFG and step count are not co-validated for Lightning, creating a silent garbage-output path.**
Candidate B states "cfg per distill, often 1.0" while the current code default is `OTR_WAN_TI2V_CFG=5.0`. An operator who sets `OTR_WAN_TI2V_STEPS=4` to enable Lightning without also setting `OTR_WAN_TI2V_CFG=1.0` will get a well-formed render of garbage output with no error. The spec needs to either: (a) define a LoRA-active mode that overrides CFG automatically, or (b) add a validation guard in `assert_usable` / `_build_graph` that requires CFG≤X when steps≤N and a LoRA is configured.

---

## SHOULD-FIX

**7. [Hard Constraints §2 / Open question] GGUF portability on MPS/AMD is left as an "open question" but the current default *is* GGUF.**
`_loader_mode()` defaults to `gguf` for the shipped UNET basename. `UnetLoaderGGUF` comes from ComfyUI-GGUF, a third-party node pack with uneven MPS/AMD support. The spec correctly raises the question but does not answer it. The fallback path (`UNETLoader` via `OTR_WAN_TI2V_LOADER=safetensors`) already exists in code — the spec needs to state which default is used for non-CUDA backends or document the operator's required override. Fix: answer the open question before the build: test `UnetLoaderGGUF` on MPS; if it fails, declare that Mac users must set `OTR_WAN_TI2V_LOADER=safetensors` with an fp8/fp16 weight.

**8. [Candidate E / Q4] `sa_solver` as a KSampler `sampler_name` is unverified.**
The spec lists `sa_solver/beta` (sampler/scheduler) as a Wan2.2 official recommendation. In standard ComfyUI, `sa_solver` appears in the scheduler list, not the sampler list; it may not be a valid `sampler_name` for `KSampler`. If the build emits `sampler_name="sa_solver"` to `KSampler`, it will either silently fall back to a default or raise at runtime. Verify: check ComfyUI `k_diffusion/sampling.py` or `/object_info` for `KSampler.sampler_name` enum → does `sa_solver` appear?

**9. [render_clip] `VramPeakProbe(interval_s=1.0)` is too coarse to measure Lightning candidate peaks.**
A 4-step GGUF render at 832×480 may complete in 3–8 seconds. At a 1-second poll interval there are ≤8 samples, and the activation peak (which is narrow) is likely missed. This means the VRAM data for Candidate B will be unreliable for the 8 GB floor decision. Fix: reduce to `interval_s=0.1` for benchmark runs at minimum.

**10. [Candidate recipes] Candidate D is absent without explanation.**
Candidates are labeled A, B, C, E. The gap may confuse contributors or suggest an incomplete spec. Either document why D was dropped or renumber.

**11. [render_clip] `self.target_fps` misused as frame-count fallback is a latent semantic bug.**
`plan["target_frame_count"] or self.target_fps` passes the fps value (25) as a frame count when no target is specified. Even though `quantize_frames_4n1` clamps this to 33, the intent is wrong and fragile — if `target_fps` changes or `quantize_frames_4n1` semantics change, behavior silently shifts. Fix: introduce `_TI2V_DEFAULT_FRAMES = _TI2V_MIN_FRAMES` and use it as the fallback.

---

## OPTIONAL / NICE-TO-HAVE

- Specify `OTR_WAN_TI2V_LORA_NAME`, `OTR_WAN_TI2V_LORA_STRENGTH` env nomenclature now so operators can prepare before the A/B build lands.
- Add explicit OOM recovery (catch `torch.cuda.OutOfMemoryError`, retry with reduced `length`) given the 13.1 GB measured peak vs 8 GB target.
- Document the exact HuggingFace repo path and filename for the 4-step and 6-step LoRA weights to prevent operator download confusion.

---

## CUT THESE (over-engineering / scope creep)

**1. [Search-first grounding] "The Lightning repo supports both 480P and 720P" — remove this sentence.**
The floor-only mandate is explicit and repeated. Mentioning 720P support in the spec even parenthetically creates an implicit invitation to scope-creep. It adds zero value to the 480p tuning decision and should be cut entirely.

**2. [Q5] "fail-closed gaps" as a freeform open question is too vague to be actionable.**
`assert_usable` already handles the well-defined closed-failure cases (missing UNET, VAE guard M8, aux loaders). The question as written generates discussion without a concrete deliverable. Replace it with a specific checklist item: "does the engine recover OOM on an 8 GB device without crashing the host process?" — that is the one unguarded failure mode the code doesn't address.