# Round D -- Claude grounded synthesis (BUG-LOCAL-231)

**Date:** 2026-05-18
**Question:** see `__00_question.md`
**Rounds:** ChatGPT gpt-5.5 (36.6s), Gemini gemini-3.1-pro-preview-customtools (38.0s), NVIDIA nvidia/llama-3.3-nemotron-super-49b-v1.5 (171.7s)

## Strong convergence (3 of 3 agree)

1. **First diagnostic is phase-boundary VRAM probing, NOT a code fix.** Log `torch.cuda.memory_allocated()` AND `torch.cuda.memory_reserved()` AND LHM physical at the seven transition points NVIDIA enumerated (Round C §5/§7). No behavior change yet.
2. **Reject candidate (b) -- sampler-time launch flags.** ChatGPT, Gemini, and NVIDIA all explicitly echo the "no flag surgery without proof" lesson from BUG-LOCAL-230.
3. **Reject candidate (d) -- FLUX-schnell -- as RCA.** All three agree it's a workaround, not a root-cause fix. Quality tradeoff (4 vs 20 steps), and the 154 s/step pace is not explained by FLUX-dev's step count.
4. **The 2.13 GiB "cold" baseline at `DeferredCheckpointLoader.fire` is the highest-leverage clue.** On a truly cold ComfyUI, `memory_allocated()` should be <100 MB. 2.13 GiB strongly suggests residual residency from the audio phase or the writer LLM phase.

## Resolved disagreement (factual)

ChatGPT initially claimed: *"If Gemma tensors or KV cache were still live as PyTorch CUDA tensors, they should usually appear in the `2.13 GiB` pre-FLUX allocated number."*

Gemini correctly flagged this as a factual error and NVIDIA concurred: **PyTorch's caching allocator can hold VRAM in `memory_reserved()` even after Python references drop, IF `torch.cuda.empty_cache()` was not explicitly called.** `memory_allocated()` reports only the currently-bound tensors; `memory_reserved()` reports what the allocator is holding back for reuse. LHM sees `memory_reserved` + driver overhead, not `memory_allocated`.

**Implication:** the 2.13 GiB reading does NOT rule out a much larger stale `reserved` block from the LLM or audio phase. The existing `DeferredCheckpointLoader` telemetry uses `memory_allocated`-style accounting (verify in source) and is therefore **incomplete** for diagnosing this defect.

**Resolved against ChatGPT** on this technical point. ChatGPT's broader diagnostic plan stands (phase-boundary probes), but the specific dismissal of candidate (a) was based on a wrong allocator model.

## New candidates surfaced (round-robin completing the original a/b/c/d)

The original candidate set was incomplete. The round-robin surfaced four additional candidates:

- **(e) Residual audio-model residency** (Bark / Kokoro / MusicGen). ChatGPT raised this as a "missed candidate" in Round A; Gemini reinforced ("Bark + Kokoro ~1.5 GiB fits the 2.13 GiB baseline"); NVIDIA concurred. **Plausible primary cause** of the 2.13 GiB cold baseline.
- **(f) T5xxl precision drift.** FLUX requires the T5xxl text encoder. If the workflow's `DualCLIPLoader` loads T5 in fp16 vs fp8 that is a ~5 GiB swing on the active VRAM budget (Gemini estimated 9.8 GiB fp16 / 4.9 GiB fp8; NVIDIA corrected the fp16 figure to ~5-6 GiB but it's still material). The 154 s/step pace + 1 GiB D3D Shared spill fits "T5xxl spills ~1 GiB during text encoding" if T5 was loaded in fp16 by accident.
- **(g) VAE residency.** Depends on whether `flux1-dev-fp8.safetensors` is UNet-only, UNet+VAE, or fully bundled. If VAE is bundled and stays resident during sampling, ~2-3 GiB lost to headroom.
- **(h) Caching-allocator reserve not freed at phase boundaries.** Tightly coupled to (a) + (e). Whether `_otr_model_loader.unload_llm()` and the post-`EpisodeAssembler.audio_done` path actually call `gc.collect()` + `torch.cuda.empty_cache()` after Python references drop. If not, the allocator holds the previous-phase VRAM hostage even though no live tensor references it.

## Prioritized investigation order (probe-only, NO code fix prescribed)

1. **Probe the 2.13 GiB "cold" baseline composition.** Highest leverage; single number that, if explained, likely points to the actual fix. Combines candidates (a), (e), (h).
2. **Probe phase-boundary VRAM behavior** with `memory_reserved()` + `memory_allocated()` + LHM at NVIDIA's seven boundaries (see §"Diagnostic harness" below).
3. **Verify T5xxl precision** in `workflows/otr_scifi_16gb_full.json` -- specifically the `DualCLIPLoader` (or equivalent) widget value for the T5 model. If `fp16` and not `fp8_e4m3fn`, that alone may be the root cause.
4. **Verify CLIP / text encoder residency through sampler.** Whether `BatchFluxRender.load_models_gpu(...)` pins only the diffusion model or also the text encoders. If ComfyUI's default behavior is "offload CLIP to CPU before KSampler", but OTR's pinning code overrides that, we hold ~5 GiB of CLIP+T5 hot during the entire sampler run.
5. **Verify VAE composition** of `flux1-dev-fp8.safetensors`. If bundled (UNet+VAE), check whether VAE is offloaded after encoding.
6. **Defer (b) and (d)** until all of the above are ruled out. Per all 3 rounds, neither is the right primary lever.

## Diagnostic harness (probe-only -- the actual code change for the next smoke)

Per NVIDIA Round C §5/§7, augment telemetry at these seven phase boundaries with `memory_reserved` + `memory_allocated` + LHM-snapshot:

| # | Boundary | File / hook |
|---:|---|---|
| 1 | After audio LLM unload | `nodes/_otr_model_loader.py::unload_llm` / `invalidate_cache_no_gpu_teardown` |
| 2 | After `EpisodeAssembler.audio_done` emit | `nodes/_otr_episode_assembler.py` (or wherever the audio_done signal is fired) |
| 3 | At `DeferredCheckpointLoader.fire` | Existing log marker -- just add `reserved` to the same line |
| 4 | After `DeferredCheckpointLoader.load complete` | Existing log marker -- same |
| 5 | Before text encoding (CLIP/T5) | `visual/batch_flux_render.py` |
| 6 | After text encoding | `visual/batch_flux_render.py` |
| 7 | Before / after sampler step 1 | `visual/batch_flux_render.py` (KSampler call site) |

Estimated diff: ~10-15 lines total, all logging. Zero behavior change. One smoke iter captures the seven-point trace. The probe data then drives the actual fix.

**Decision rule for interpreting the trace (per ChatGPT Round A §2 + Gemini Round B §1):**

- If `allocated` jumps after text encoding and stays high into sampler -> **CLIP / T5 residency (candidate (c) + (f))**.
- If `reserved` is already high at boundary #3 (`DeferredCheckpointLoader.fire`) while `allocated` is low (2.13 GiB) -> **allocator reserve not freed at phase boundaries (candidate (h)), almost certainly hiding audio or LLM cache**.
- If `reserved` jumps while `allocated` does not, between boundaries 5 and 6 -> **T5 / CLIP allocator workspace (candidate (c) -- different cause than tensor residency)**.
- If both are already high before FLUX load -> **stale LLM or audio (candidates (a), (e), (h))**.
- If LHM rises but neither torch metric does -> **non-torch CUDA / D3D allocation, driver-level paging** (less common, deeper investigation).

## What NOT to do (per round-robin + CLAUDE.md)

- Do NOT add `--fast`, `--fast fp8_matrix_mult`, `--lowvram`, `--normalvram`, or any other ComfyUI launch flag without proof. BUG-LOCAL-230 lesson: a flag added without telemetry caused 16 hours of architectural campaign cycles.
- Do NOT chase SageAttention enable, Flash Attention alternatives, or weight streaming.
- Do NOT touch audio generation code -- Rule C7 byte-identity is non-negotiable.
- Do NOT switch to FLUX-schnell as the RCA answer.
- Do NOT prescribe a fix without the seven-point probe trace first.

## Bible candidacy status (revisited)

Pending close. Predicted by fix-type:

- If the fix is "add `gc.collect() + torch.cuda.empty_cache()` at phase boundaries in `_otr_model_loader.unload_llm()` and the audio_done emit path" -> **promotes**. Generalizes to any multi-stage pipeline on a constrained card.
- If the fix is "set T5xxl dtype to fp8 in the workflow's `DualCLIPLoader`" -> **promotes**. FLUX-specific but a recurrent footgun for fp8 workflows (echoes BUG-LOCAL-230's "audit precision on every model in a quantized workflow" pattern).
- If the fix is "switch to FLUX-schnell" or "add a launch flag" -> **does NOT promote**. Workflow-local workaround / would repeat the BUG-LOCAL-230 anti-pattern.

## Open items for the next session

1. Apply the seven-point probe (10-15 line diff across 2-3 files).
2. Run one smoke iter with the probe.
3. Read the trace and decide the fix per the decision rule above.
4. Re-run the BUG-LOCAL-230 7-criteria smoke gate. Gates #5 + #6 must close cleanly.
5. Promote BUG-LOCAL-230 to `[FIXED]` only after gates #5 + #6 close.
6. Decide BUG-LOCAL-231 Bible candidacy at close.

## Process notes (for transcript history)

- All three models flagged candidate (b) as "would repeat BUG-LOCAL-230 anti-pattern." Independent triangulation against the Round 0 instruction.
- Gemini caught one factual error in ChatGPT's allocator model; NVIDIA confirmed. ChatGPT's broader plan stands minus that detail.
- NVIDIA refined Gemini's T5xxl fp16 size estimate (9.8 GiB -> 5-6 GiB). Worth verifying against the actual T5xxl model card before basing the fix on a precise headroom calculation.
- Total round-robin wall time: ~3.7 min (ChatGPT 36s + Gemini 38s + NVIDIA 172s + script overhead). Within the round-robin protocol's expected cost.
- No model hallucinated file paths -- both `visual/batch_flux_render.py` and `nodes/_otr_model_loader.py` are real files in the OTR repo (verified separately during the smoke run).
