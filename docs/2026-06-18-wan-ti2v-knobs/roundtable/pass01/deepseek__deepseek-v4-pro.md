<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? **no**. The plan is a roundtable discussion agenda, not a build-ready spec. It lacks concrete implementation details (exact parameters, new graph nodes, env vars) for the candidate recipes, and leaves critical cross-platform and licensing questions unanswered.

MUST-FIX BEFORE BUILD:
1. **[Candidate recipes]** Candidate B (4-step Lightning LoRA) is not defined with concrete parameters. The plan says “cfg per distill, often 1.0” but does not specify the exact cfg, sampler, scheduler, shift, or LoRA strength. Without these, the recipe cannot be implemented or A/B tested. Fix: specify the full set of knobs (e.g., sampler `euler`, scheduler `simple`, shift `3.0`, cfg `1.0`, and the LoRA file path + strength) for the 480p variant.
2. **[Candidate recipes]** Candidate C (6-step distill) is similarly undefined. Fix: provide the same concrete knobs or drop it if not needed.
3. **[Graph integration]** The current `_build_graph` (in `eng_wan_ti2v.py`) has no LoRA node. To use a LoRA, the graph must include a loader node (e.g., `LoraLoader` or `LoraLoaderModelOnly`) with its inputs (`lora_name`, `strength`). The plan does not mention this. Fix: define the new node, its class, and its wiring in the graph spec.
4. **[HARD CONSTRAINTS / Cross-platform]** The plan asks whether `UnetLoaderGGUF` works on MPS/AMD but does not decide. The floor must be system-agnostic. If GGUF is not portable, the plan must switch to a safetensors path. Fix: determine and document the loader (`gguf` vs `safetensors`) that works on Mac/AMD, and if necessary, change the default or provide a fallback.
5. **[Default-off / additive]** The plan states any new knob ships behind an `OTR_WAN_TI2V_*` env, but does not list the new env vars for the LoRA (path, strength). Fix: define `OTR_WAN_TI2V_LORA_PATH`, `OTR_WAN_TI2V_LORA_STRENGTH`, etc., and ensure they default to off/current behavior unless promoted.
6. **[License]** The plan requires the LoRA to be Apache-2.0/MIT but only says “verify LightX2V license”. No verification step or link is provided. Fix: confirm the license of the specific LoRA file (e.g., from the HuggingFace repo) and document it; if unconfirmed, the LoRA cannot be used in a commercial-clean engine.

SHOULD-FIX:
1. **[VRAM]** Provide a VRAM estimate for the 4-step recipe (including LoRA and any overhead) to prove it fits within 8 GB. The current plan has no budget.
2. **[Sampler portability]** List which samplers are confirmed cross-platform (e.g., `euler`, `lcm`) and which are not (`uni_pc`, `sa_solver`, `MoEKSampler`). If `MoEKSampler` is a custom node not in core ComfyUI, it should be excluded from the floor.
3. **[Determinism]** Verify that the 4-step Lightning LoRA is deterministic with seed (some distilled models may have stochastic components). Document any caveats.
4. **[Fallback]** Provide a fallback mechanism if the Lightning LoRA is missing or fails on a platform (e.g., fall back to the baseline A recipe).

OPTIONAL / NICE-TO-HAVE:
- The roundtable could include a decision matrix to avoid bikeshedding, but the current question set is sufficient if answered.

CUT THESE (over-engineering):
- Candidate C (6-step distill) and E (non-LoRA control) may be over-engineering if the 4-step Lightning is clearly the recommended accelerator. However, they are for A/B testing, so not strictly over-engineering yet. The panel can decide to drop them if the 4-step is proven.

[ASSUMPTION] The Lightning “LoRA” is a LoRA file that requires a `LoraLoader` node; the plan does not state this explicitly but the name and the sources imply it. The current graph lacks this node, so it must be added.