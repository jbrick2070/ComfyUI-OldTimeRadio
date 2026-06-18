# wan_ti2v knob roundtable -- pass01 judgment (Claude, grounded vs eng_wan_ti2v.py)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro, Opus-4.8, Sonnet-4.6.
Spend ~$0.90. ALL SIX returned `VERDICT: no` -- strong convergence.

## HEADLINE (unanimous, grounded): the Lightning LoRA is the WRONG lever for this floor -> CUT B/C
- **Step count is not a VRAM lever (Opus, confirmed).** The ~13.1 GB peak is model
  RESIDENCY + the video-VAE DECODE, not iteration count. 30->4 steps makes it
  FASTER, not SMALLER -- it does NOT help the 8GB floor's actual blocker.
- **No LoRA wiring exists (6/6, CONFIRMED).** `_node_candidates()` / `_build_graph()`
  have no LoraLoader; B/C are a graph change, not an env knob, and `assert_usable`
  wouldn't fail-closed on a set-but-missing LoRA.
- **GGUF + safetensors-LoRA needs a custom `LoraLoaderGGUF` (Gemini, CONFIRMED-plausible)**
  -> violates the core-only + cross-platform constraint.
- **License unconfirmed (Sonnet/Opus, VERIFY).** ModelTC/Wan2.2-Lightning base is
  Apache-2.0 but `lightx2v/Wan2.2-Distill-Loras` has a separate card -- gates B/C.
=> The 4-step LoRA belongs (if anywhere) to a HIGHER tier, not the accessible floor.

## CONFIRMED (grounded in the engine) -> fold into the plan
- `_TI2V_MIN_FRAMES = 33` + `target_fps=25` misused as the frame-count fallback ->
  the real default render is 33 frames (spec's "length 25" is wrong); 33f@832x480
  is what hit ~13 GB. Semantic bug: use a real `_TI2V_DEFAULT_FRAMES`.
- Default sampler `uni_pc` is passed straight to KSampler with NO validation.
- 13.1 GB measured on a 5080 -> on a real 8GB card `assert_peak_within_ceiling`
  fail-closes; the "floor" does not run on the floor today.
- safetensors fallback already exists (`_loader_mode` gguf|safetensors + UNETLoader).
- CFG/steps coupling trap: steps=4 without cfg~1.0 = well-formed GARBAGE, no error.

## UNVERIFIABLE -> VERIFY-AT-BUILD (cannot tell from our code; need real backends)
- GGUF (`UnetLoaderGGUF`, ComfyUI-GGUF) viability on MPS / ROCm / DirectX. **Conflict:**
  Gemini argues GGUF is the ONLY viable 8GB Mac path (fp8 safetensors UPCASTS to
  fp16/32 on MPS -> blows 8GB; GGUF dequants per-layer); GPT/Grok/Opus lean
  safetensors-fp16 default for portability. UNRESOLVED -> must test on a real Mac/AMD,
  do NOT assume. The fp8 `umt5` CLIP is the same upcast risk.
- `sa_solver` may be a SCHEDULER, not a valid KSampler `sampler_name` (Sonnet) -> verify object_info.
- `VramPeakProbe(interval_s=1.0)` is too coarse for a 3-8s render -> my 0.7s NVML
  sample of 13.1 GB may UNDERSTATE the true peak. Re-measure at 0.1s.

## CUT (over-engineering / scope creep)
- Candidates B + C (Lightning LoRA / 6-step distill) -- wrong lever for the floor.
- `MoEKSampler` (custom, CUDA-only), `sa_solver` (unverified) from the floor default.
- All 720p references (out of scope; that tier is the LTX audio-in lane).

## DISCARDED / DOWNGRADED panel claims
- "Add a safetensors default for portability" taken as ABSOLUTE -> downgraded: Gemini's
  fp8-upcast-on-MPS point means safetensors is NOT automatically more portable;
  resolve by TEST, not assertion.
