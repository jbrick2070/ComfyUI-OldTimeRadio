# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=34.2s

**1. Core Recommendation Verdict**
I **AGREE** with ChatGPT's core recommendation: do not roll back, the BUG-027 dialogue fixes are solid, but the BUG-028 read-discovery path (specifically the wildcard glob) carries residual risk and should be monitored.

**2. Factual Errors in ChatGPT's Answer**
*   **C7 Byte-Identity interpretation:** ChatGPT states, *"if C7 means 'same inputs, same run, byte-identical final audio,' then the LLM path must already be deterministically seeded... The BUG-027 changes do not add new randomness."* This slightly misses the mark on how C7 is evaluated. Changing the prompt (code) *will* and *should* change the LLM output and thus the final audio bytes. C7 guarantees that running the *new* code twice yields the same bytes both times. The fix doesn't violate C7, it just establishes a new baseline.
*   **"Caveat" on per-episode counters:** ChatGPT frames the counter incrementing (e.g., `00001` to `00002` on rerun) as a minor determinism caveat. In a ComfyUI pipeline, this is standard, intended behavior to prevent destructive overwrites of previous batch outputs. It is a feature, not a caveat.

**3. Important Things That Were OMITTED**
*   **The glaring architectural contradiction in `batch_humo_render`:** ChatGPT correctly noted that the wildcard `otr/episodes/*/stills/...` plus an mtime filter isn't "mathematically impossible" to leak. But it missed the bigger question: *If `batch_humo_render` has access to `ledger_mtime`, it must be reading the ledger.* If it's reading the ledger, it knows the `episode_id`. Why use a wildcard `*` at all? It should explicitly target `otr/episodes/{episode_id}/stills/`.
*   **ComfyUI Custom Node `CATEGORY`:** The new `SaveToEpisodeWorkspace` node is missing a `CATEGORY = "OTR/IO"` (or similar) declaration. Without this, ComfyUI dumps it into the root "Add Node" menu, which clutters the UI.
*   **Tensor Shape in `_tensor_to_pil`:** ComfyUI passes images as a 4D tensor `[Batch, Height, Width, Channels]`. The loop `for img in images:` iterates over the batch dimension, yielding 3D tensors. If `_tensor_to_pil` isn't strictly expecting a 3D tensor (or if a weird upstream node passes a 3D tensor causing the loop to iterate over height), it will crash.
*   **VRAM Impact:** Neither the prompt change nor the custom save node hold tensors in memory longer than necessary. VRAM ceiling (14.5 GB) is entirely unaffected by these changes.

**4. My Short Recommendation (Actionable Bullets)**
*   **Keep the BUG-027 fixes exactly as shipped.** The 0.5 threshold is a perfect "catastrophic collapse" guard, and the regex handles your specific writer's output. Do not touch the per-character floor.
*   **Add a loud warning to the BUG-028 fallback.** In `_resolve_target_dir`, if it falls back to `_legacy_stills/`, use `logger.warning()` or `print('\033[93m')` to scream at the console. Silent fallbacks in local pipelines hide regressions.
*   **Refactor the `batch_humo_render` wildcard (Low Priority but Recommended).** If/when you touch this file again, drop the `*` wildcard. Read the `episode_id` from the ledger and construct the exact path. Relying on `ledger_mtime - 60s` is a clever hack, but explicit ID routing is bulletproof.
*   **Restart ComfyUI completely.** Do not rely on hot-reloading for the new `OTR_SaveToEpisodeWorkspace` node or the `__init__.py` mapping changes.
*   **Verify Workflow JSON.** Ensure node 25 in `otr_scifi_16gb_full.json` actually has `"class_type": "SaveToEpisodeWorkspace"` (or whatever key you used in `NODE_CLASS_MAPPINGS`), not just the Python class name.

**5. Uncertainties I Would Want to Verify**
*   **Why was the wildcard used in Site 3?** Is there a specific reason `batch_humo_render.py` cannot resolve the `episode_id` directly, even though it can resolve the `ledger_mtime`? If the ledger is available, the ID should be too.
*   **How does `_tensor_to_pil` handle clamping?** FLUX outputs can sometimes exceed the `[0.0, 1.0]` range. If `_tensor_to_pil` doesn't clamp before multiplying by 255 and casting to `uint8`, you might get visual artifacts (which wouldn't break C7 audio, but ruins the video path).
*   **Are there other workflows?** You updated `otr_scifi_16gb_full.json`. If you have a `_fast.json` or `_test.json` workflow, they will still have the old `SaveImage` node and will dump to the legacy directory.
