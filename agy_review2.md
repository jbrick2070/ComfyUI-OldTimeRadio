# OldTimeRadio Sci-Fi Code Review (Round 2)

This report addresses the Q1–Q4 prompt queries regarding the context window tracing, baseline preservation, KV cache capacity recomputation, and scaling failure points for the 720-word / 40–48 line Sci-Fi configuration.

---

## Q1: Runtime Context Path & Double-Lock Tracing

### 1. End-to-End Tracing of `cache_entry["context_cap"]`
1. **Downstream Consumption**: The prompt truncation logic is defined in `OTR_LedgerScriptWriter._build_truncating_generate_fn` at [nodes/OTR_LedgerScriptWriter.py:647](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L647). It reads the context cap directly from the cached model entry: `cache_entry.get("context_cap")`.
2. **Scheduler Dispatch**: The active model is requested from the slot scheduler via `_SlotScheduler.request_slot` in `nodes/_otr_model_loader.py`.
3. **Context Cap Resolution**: At [nodes/_otr_model_loader.py:941](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_loader.py#L941), `request_slot` calls the **live** helper:
   ```python
   ctx_verdict = _otr_catalog.resolve_context_cap(normalized)
   ```
   `_otr_catalog.resolve_context_cap` is a **live** function in `nodes/_otr_model_catalog.py` that computes the cap verdict based on system limits and overrides.
4. **Cache Insertion**: The resolved verdict value (`ctx_verdict.value`) is passed as the `context_cap` parameter to `load_llm`.
5. **Override Logic**: Inside `load_llm` at [nodes/_otr_model_loader.py:1006](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_loader.py#L1006):
   - Fallback defaults to `_MODEL_CONTEXT_CAPS.get(_resolved_id, 8192)`.
   - If `context_cap is not None`, the fallback is overridden: `_cap = context_cap`.
   - The returned `cache_entry` dict is populated with `"context_cap": _cap`.

### 2. Status of `resolve_context_cap` vs. `compute_effective_context_limit`
- **`resolve_context_cap`**: **LIVE**. It is the active path during slot scheduling and loading.
- **`compute_effective_context_limit`**: **DEAD**. It is only exported in `__all__` and tested in test mocks, but not called anywhere in the active model loading or runtime execution paths.

### 3. Minimal Edit Set to Raise the Cap to 16,384
Because of the model loading preconditions, there is a **double-lock** preventing environment variables alone from scaling up the context cap:
1. **Catalog Override Lock**: `resolve_context_cap` reads `CURATED_CONTEXT_OVERRIDES` to resolve curated models. The entry for Mistral-Nemo is locked at `8192`.
2. **Precondition Lock**: `_otr_loader_backends.check_context_window` is called at load time in `nodes/_otr_model_runtime.py:71`. If the row's native `context_window` is below `HARD_VRAM_CONTEXT_LIMIT`, it raises a `RuntimeError` to prevent silent downstream truncation.

**Minimal Edit Set to unlock**:
- **Edit 1**: In `nodes/_otr_model_catalog.py`, change `context_window=8192` to `context_window=32768` (or the native `131072` limit) in the Mistral-Nemo dataclass definition within `CURATED_LLM_MODELS`.
- **Edit 2**: In `nodes/_otr_model_catalog.py`, change `"mistralai/Mistral-Nemo-Instruct-2407": 8192` to `"mistralai/Mistral-Nemo-Instruct-2407": 32768` (or `131072`) in `CURATED_CONTEXT_OVERRIDES`.
- **Operator Opt-in**: Set the environment variable `OTR_HARD_VRAM_CONTEXT_LIMIT=16384` (or `24576`).

### 4. Tests Pinning These Values
- [tests/test_vram_envelope_c4.py:76-88](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_vram_envelope_c4.py#L76-L88) asserts `catalog.HARD_VRAM_CONTEXT_LIMIT == 8192` under default environment settings.
- [tests/test_effective_context_limit.py:75-86](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_effective_context_limit.py#L75-L86) asserts `compute_effective_context_limit(mistral) == catalog.HARD_VRAM_CONTEXT_LIMIT`.
- [tests/test_context_window_precondition.py:75-85](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_context_window_precondition.py#L75-L85) asserts `check_context_window(mistral)` passes without raising.

---

## Q2: Impact on Audio Byte-Identity Baseline

- **Does raising the cap break the baseline?** No. 
- **Why?** When the `OTR_HARD_VRAM_CONTEXT_LIMIT` environment variable is unset, the system-wide limit `HARD_VRAM_CONTEXT_LIMIT` defaults to `8192`. 
- **Mathematical clamp**:
  $$\text{Effective Context Cap} = \min(\text{override}, \text{limit}) = \min(32768, 8192) = 8192$$
  The context cap resolves to exactly `8192` by default. The input prompts and seed generations remain identical, preserving the "C7 audio byte-identity" baseline.
- **Safety of Opt-in**: The environment opt-in is completely safe. It allows operators on larger hardware to explicitly scale context window size without forcing changes onto the default test environment or other users.

---

## Q3: Model Loading & KV Cache Calculations

### 1. How Mistral-Nemo is Loaded
- **Quantization**: Loaded using **4-bit** quantization (`bnb_nf4`) via `BASELINE_POLICY` in `nodes/_otr_shared/llm_policy.py`.
- **Dtype / Activations**: Attention calculations and key-value cache elements are computed in **BF16** (`bnb_4bit_compute_dtype=torch.bfloat16`, 2 bytes per element).
- **Device**: CUDA (`device="cuda"`).

### 2. KV Cache Recomputation for Mistral-Nemo-12B
*Config: $N_{\text{layers}} = 40$, $N_{\text{kv\_heads}} = 8$, $D_{\text{head}} = 160$, $B_{\text{bytes}} = 2$ (BF16).*

- **Formula**:
  $$\text{KV Cache Size} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times L_{\text{seq}} \times B_{\text{bytes}}$$
- **Size Per Token**:
  $$2 \times 40 \times 8 \times 160 \times 2 = 204,800 \text{ bytes} \approx 200 \text{ KB/token}$$
- **8k (8,192)**:
  $$8192 \times 204,800 \text{ B} = 1,677,721,600 \text{ B} \approx 1.56 \text{ GiB} \approx 1.68 \text{ GB}$$
- **16k (16,384)**:
  $$16384 \times 204,800 \text{ B} = 3,355,443,200 \text{ B} \approx 3.13 \text{ GiB} \approx 3.36 \text{ GB}$$
- **24k (24,576)**:
  $$24576 \times 204,800 \text{ B} = 5,033,164,800 \text{ B} \approx 4.69 \text{ GiB} \approx 5.03 \text{ GB}$$

### 3. VRAM Fit & Go/No-Go Verdict (16 GB Card)
*Baseline VRAM Allocation:*
- OS / DWM / Headroom: ~1.5 GB
- Model Weights (4-bit + unquantized embeddings/head): ~7.5 GB
- Total baseline: ~9.0 GB

*Verdicts:*
- **16,384 (16k)**: **GO**. Total VRAM $\approx 9.0 + 3.36 = 12.36\text{ GB}$. Comfortably below the 14.5 GB target ceiling.
- **24,576 (24k)**: **GO**. Total VRAM $\approx 9.0 + 5.03 = 14.03\text{ GB}$. Below the 14.5 GB target ceiling; highly feasible.
- **32,768 (32k)**: **NO-GO**. Total VRAM $\approx 9.0 + 6.71 = 15.71\text{ GB}$. Exceeds the 14.5 GB ceiling and risks OOM due to CUDA fragmentation.

*Unload Safety*: The writer LLM is unloaded via `unload_llm_if_local_resident()` before the media/render stages run. This releases VRAM back to the baseline level (~1.5 GB) before Bark/Kokoro or LTX-Video load, preventing simultaneous memory pressure.

---

## Q4: Scaling Failures at 720 Words / 48 Lines

We identified several critical scaling limitations at 720 words / 48 lines:

### 1. The Script Token Budget Clamp (Hard Failure)
- **Problem**: `_script_output_token_budget` in `nodes/_otr_scifi_codex.py:946` clamps `script_token_budget` to a hard ceiling of `5400`.
- **Failure Cause**: 
  - For a 720-word, 48-line script, the required JSON output size is estimated at $720 \times 4.5\text{ (dialogue)} + 48 \times 130\text{ (metadata)} + 600\text{ (envelope)} \approx 10,080$ tokens. 
  - Even with a very dense JSON representation, it requires ~5,000–5,500 tokens.
  - Because `max_new_tokens` for the generator passes (P5/P7/P9) is set to `script_token_budget` (clamped to 5400), a verbose model generation will truncate, resulting in invalid JSON and throwing `no decodable top-level JSON object`.
- **Scope**: Affects all backends (both local models and API-based lanes like Gemini/Sonnet) because `max_new_tokens` is forwarded to all slot generators.

### 2. Silent Truncation on Review Passes
- **Problem**: Review passes `P6` (Listener Review) and `P8` (Final Audit) do not pass `prompt_must_fit=True` to `invoke_codex_structured`.
- **Failure Cause**: At 720 words, the input prompts for these passes contain the entire script (with full metadata graph) plus score and fact indexes, totalling ~8k–9k tokens. Without `prompt_must_fit=True`, if the context window is not raised (or if a longer script is run), the prompt will silently left-truncate, causing corrupted inputs to the LLM.

### 3. Media Scaling & Compositing (Safe)
- **Beat counts**: Cap at 12 via `max(3, min(12, len(p2.cast) * 3))`. At 720 words, this equates to ~60 words per beat. The planning logic handles this gracefully.
- **Render plan limits (`render_max_n`)**: The render plan telemetry caps lines to `6` by default. However, the real video generator (`nodes/_otr_video_engines/render_driver.py`) explicitly ignores `meta.render_plan` and renders all scenes/shots/beats from the score graph.
- **Compositing & Captioning**: The compositor (`nodes/otr_silent_composite.py`) and caption builder (`nodes/_otr_captions.py`) handle arbitrary lengths without hardcoded ceilings. Clip underruns are safely loop-filled or frame-held.
