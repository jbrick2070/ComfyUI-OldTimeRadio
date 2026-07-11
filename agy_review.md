# AGY Review -- OTR Sci-Fi Bake-off

VERDICT: yes-with-fixes

## MUST-FIX BEFORE BUILD

1. **Fix context cap override locking in model catalog**
   - **File:line**: [nodes/_otr_model_catalog.py:1227](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_catalog.py#L1227) [CONFIRMED]
   - **Issue**: `"mistralai/Mistral-Nemo-Instruct-2407"` has its curated override set to `8192`. In `resolve_context_cap`, the returned value is `min(override, limit)`. Since `limit` defaults to `HARD_VRAM_CONTEXT_LIMIT = 8192` but can be scaled via the environment variable `OTR_HARD_VRAM_CONTEXT_LIMIT`, the `min` function locks Mistral-Nemo at `8192` even if the user raises the environment variable to `16384` or `24576` (e.g. `min(8192, 16384) = 8192`).
   - **Patch sketch**: Raise the override in `CURATED_CONTEXT_OVERRIDES` to Mistral-Nemo's actual maximum supported context size (or a high-context ceiling like 32768 or 131072) so that it is properly capped by `limit` (retaining the 8192 default baseline, but allowing scaling up via the env var).
     ```python
     CURATED_CONTEXT_OVERRIDES: dict[str, int] = {
     -    "mistralai/Mistral-Nemo-Instruct-2407": 8192,
     +    "mistralai/Mistral-Nemo-Instruct-2407": 32768,
     ```

2. **Set `prompt_must_fit=True` for whole-script generation passes**
   - **File:line**: [nodes/_otr_scifi_codex.py:1373,1375,1378](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py#L1373) [CONFIRMED]
   - **Issue**: Passes `P5`, `P7`, and `P9` call `invoke_codex_structured` but omit the `prompt_must_fit` parameter, which defaults to `False`. For large word counts (like 720w), the prompt exceeds the context window and silently left-truncates the system prompts and schema definitions. This wastes GPU cycles and results in JSON parsing errors downstream.
   - **Patch sketch**: Add `prompt_must_fit=True` to the structured calls for `P5`, `P7`, and `P9` to force a loud and clear `PromptContextOverflowError` rather than a silent failure.
     ```python
     # P5
     script = invoke_codex_structured(..., prompt_must_fit=True)
     # P7
     script = invoke_codex_structured(..., prompt_must_fit=True)
     # P9
     script = invoke_codex_structured(..., prompt_must_fit=True)
     ```

## SHOULD-FIX

- None. The working tree fixes for Defect A (voice replay crash in content-owned lanes) and Defect B (JSON echoing and dynamic token budgeting) are implemented cleanly, are correct at the root cause, and pass all verification tests.

## 720W CAPACITY VERDICT

### Arithmetic
- **Ceiling Arithmetic**: CONFIRMED. At 720 words split over 40 lines (typical OTR density), the `P7` input prompt alone is **8,284 tokens** (measured using the actual `mistralai/Mistral-Nemo-Instruct-2407` tokenizer). When combined with the dynamic output token reservation of **5,400 tokens** (computed via the dynamic budget formula), the total requirement is **13,684 tokens**. At 48 lines, the prompt alone takes **9,508 tokens**, requiring a total of **14,908 tokens**.
- Both scenarios completely exceed the default 8,192 ceiling. Under the default configuration, this leads to a silent left-truncation that slices off the system prompt/schema instructions.

### Recommendation
- **Recommended Option**: **(a) Raise context_cap for the local transformers path.**
  - **Reasoning**: Mistral-Nemo supports 128k context natively. The KV cache size is $200$ KB per token (calculated using the formula $2 \times L \times N_{kv\_heads} \times D_{head} \times C \times 2 = 2 \times 40 \times 8 \times 160 \times C \times 2 = 204,800$ bytes per token [CONFIRMED]).
    - At 16k context, the KV cache size is ~3.125 GiB. Combined with the 4-bit model weights (~8 GB), the VRAM footprint is ~11.1 GB. Along with baseline OS/DWM overhead (~1.5 GB), the total is ~12.6 GB, which easily fits within the 16 GB RTX 5080 laptop.
    - At 24k context, the KV cache is ~4.69 GiB, totaling ~14.2 GB VRAM. This is close to the 14.5 GB target but still safe.
    - At 32k context, the KV cache reaches ~6.25 GiB, totaling ~15.75 GB. This is extremely tight and risks system thrashing or OOM when other tasks run.
  - Raising the curated override in `CURATED_CONTEXT_OVERRIDES` to `32768` allows users to dynamically scale up the context cap to 16k or 24k by setting the environment variable `OTR_HARD_VRAM_CONTEXT_LIMIT`, while preserving the exact 8192 C7 baseline by default.
  - **Why Option (b) is wrong**: Making `P7/P9` a patch-based pass only reduces the output tokens, but does not decrease the input prompt size, which alone exceeds 8,192. It also breaks three major downstream stages (deterministic repair, validator, assembler) that assume a whole `ScriptArtifactV4`.

## CUT THESE (over-engineering)

1. Do not implement a line-level patch-based retake/rewrite mechanism. It introduces significant complexity and breaks core pipeline assumptions without solving the underlying context window limit.
