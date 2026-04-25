# Question -- 2026-04-19

# Consult: Is `workflows/otr_scifi_16gb_TEST.json` ready to fire?

## Context

OTR v2.0 ComfyUI custom node pack, branch `v2.0-alpha`. Jeffrey wants to run the TEST workflow against a live ComfyUI instance on Windows (RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, torch 2.10, CUDA 13). Audio is king and must not degrade.

## Current audit + regression state (fresh pass, 2026-04-19)

- AST parse of 102 .py files: **0 violations**
- UTF-8 BOM check: **0 violations**
- Node INPUT/OUTPUT contract audit (INPUT_TYPES ↔ RETURN_TYPES ↔ FUNCTION ↔ CATEGORY): **0 violations**, 21 node classes scanned
- Workflow ↔ NODE_CLASS_MAPPINGS cross-check: **0 violations**
- Regression suite (Bug Bible + dropdown guardrails + core + audio byte-identical): **183 passed, 2 skipped, 2 xfailed** (138s)

## Workflow shape

`workflows/otr_scifi_16gb_TEST.json` — 6 nodes, UI format. Uses 5 `OTR_Visual*` classes:
- `OTR_VisualBridge`
- `OTR_VisualLLMSelector`
- `OTR_VisualPoll`
- `OTR_VisualPromptCoercion`
- `OTR_VisualRenderer`

All 5 are registered in `__init__.py`.

## The open red flag

Task #72 is still open: `visual/backends/ltx_motion.py` line 226 calls:

```python
LTXImageToVideoPipeline.from_pretrained(
    _LTX_PATH,                         # points at models/diffusers/LTX-Video/
    torch_dtype=torch.float8_e4m3fn,
    local_files_only=True,
)
```

That folder is a partial HF snapshot **missing `model_index.json`**. We already confirmed (via `scripts/verify_ltx_hybrid.py`) that the working loader is the hybrid approach:
1. Build T5Config from `models/huggingface/hub/.../LTX-Video/.../text_encoder`
2. Instantiate empty T5EncoderModel + load weights from existing local `models/text_encoders/t5xxl_fp16.safetensors`
3. Build T5Tokenizer from the snapshot `tokenizer/` subfolder
4. `LTXImageToVideoPipeline.from_single_file("models/checkpoints/ltx-video-2b-v0.9.safetensors", text_encoder=t5, tokenizer=tok, torch_dtype=bfloat16)`

The hybrid verify script loads cleanly end-to-end on this machine. The production backend hasn't been patched to match.

When ltx_motion fails to load, it falls back silently to stub-mode (still frames + procgen overlay). That silent fallback is BUG-LOCAL-046, also still open.

## Unknowns I can't resolve without running

- Whether `OTR_VisualRenderer` actually dispatches into `ltx_motion` for this TEST workflow, or whether it short-circuits to `flux_anchor`-only / still-renderer given the TEST offline-asset injection (Task #77 rewrote the TEST JSON to sever LLM + inject offline assets + hardcode telemetry).
- Whether FLUX anchor weights are still present at the expected path.

## The question

Given:
- All static gates are green (audit 0/0/0/0, regression 183 pass).
- The workflow JSON is valid and every `OTR_*` class is registered.
- The ltx_motion loader is known-broken and will fall back to stubs **silently**.
- Audio-path tests are all passing and the audio spine has never been touched by the video work.

**Is it reasonable to fire this workflow as-is, or should we patch Task #72 first?**

Specifically:
1. What's the realistic chance of a clean end-to-end run without the ltx_motion patch?
2. What's the worst-case failure mode if we fire without patching? (silent stub fallback, hang, OOM, corrupted audio?)
3. Is there a cheap pre-flight check that would tell us which backend path `OTR_VisualRenderer` takes in this specific TEST workflow before we commit to firing?
4. If we do fire and get stubs back, is the fix still just the `from_single_file` swap, or should we also close BUG-LOCAL-046 (surface the silent fallback) at the same time?

Please give a grounded recommendation. Disagree freely if you think the regression-green signal is enough to justify firing.
