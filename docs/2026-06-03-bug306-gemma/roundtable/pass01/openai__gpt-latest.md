<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Do not ship gemma-4-12b as selectable/PASS; hide/block it first, then add writer fallback only if scoped and observable.

MUST-FIX BEFORE BUILD:
1. [Candidate A / Hard constraint 4] “Hide in catalog” is underspecified because the grounded validator still admits non-curated/local/arbitrary HF ids. In `_otr_model_catalog.py`, removing `google/gemma-4-12b-it` from `CURATED_LLM_MODELS` removes it from the dropdown, but `validate_model_id()` can still admit it via Path 2 if it is already in the HF cache, and via Path 3 when `OTR_MODEL_CATALOG_AUTO_DOWNLOAD` defaults to enabled. Concrete fix: for this bug, either:
   - smallest: remove the `google/gemma-4-12b-it` curated row and verify the canonical workflow does not pin it; accept that manual/stale ids may still reach loader until B exists, or
   - stronger: add a known-unsupported deny/block path for `google/gemma-4-12b-it` / `gemma4_unified` before load or in validation, with an explicit “unsupported by installed transformers” error.
   If the intended guarantee is “cannot be selected from UI,” removal is enough. If the intended guarantee is “cannot crash a stale/manual workflow,” removal alone is not enough.

2. [Candidate A / _otr_model_catalog.py row] The `google/gemma-4-12b-it` row is currently marked `vram_fit_tier="PASS"` and has `loader_backend="transformers_multimodal_text_only"`, so `check_vram_fit()` would report PASS for an unloadable architecture. Concrete fix: do not leave this row active with PASS. Remove it now, or add availability semantics that exclude unavailable rows from dropdown, `_by_repo_id()`, and `check_vram_fit()` PASS treatment.

3. [Candidate D] “Keep the catalog row but mark it not-yet-loadable” does nothing if “mark” means only `notes`. Grounding shows `build_dropdown