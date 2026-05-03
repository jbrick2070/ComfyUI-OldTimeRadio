# Round C -- NVIDIA (mistralai/mistral-nemotron) elapsed=190.7s

### 1. Agreement / Disagreement

**Agreement:**
- **`video_engine.py` Post-Rename Path Fix:** Both agree that recomputing and returning the final path after `rename_episode` is necessary. This is correct.
- **Busting ComfyUI Cache for Singleton:** Gemini’s `IS_CHANGED` method is a critical addition to ensure the singleton is updated per queue item. ChatGPT missed this entirely.
- **Layout-Aware Ledger Discovery:** Both agree on moving away from stem swaps, but Gemini’s directory-based approach is more robust.
- **Strict Cache Fallback:** Both emphasize preserving legacy cache paths for VRAM and C7 compliance.

**Disagreement:**
- **Singleton Use in ComfyUI:** ChatGPT advocates for using a singleton (`_CURRENT`), while Gemini warns against it due to ComfyUI’s node caching. Gemini is correct here—global state in ComfyUI is dangerous.
- **Path Reorg Impact on C7:** ChatGPT assumes path changes won’t affect audio bytes, but Gemini rightly points out that cache misses can violate C7 if audio generation isn’t deterministic.

**Third Path:**
- **Hybrid Approach:** Use the singleton (`_CURRENT`) but enforce cache invalidation (via `IS_CHANGED`) to ensure it’s always fresh. This balances ChatGPT’s simplicity with Gemini’s safety concerns.

### 2. Factual Errors

**ChatGPT:**
- Assumes ComfyUI’s sequential queue prevents cross-episode races. **False:** ComfyUI’s node caching can stale the singleton.
- Overlooks VRAM implications of cache misses in AudioGen/MusicGen.
- Assumes mp4 stem matches `episode_id`, which is broken due to title truncation (`safe_title = ...[:40]`).

**Gemini:**
- No factual errors, but could clarify that `IS_CHANGED` is a ComfyUI-specific method to force node re-execution.

### 3. Important Omissions

**Both Missed:**
- **Ledger Path Validation:** Neither suggests adding a sanity check in `Ledger.save()` to ensure `meta.paths.ledger_path` matches `self.path`. This would catch path-reorg bugs early.
- **Legacy Artifact Handling:** Neither fully addresses how to handle existing artifacts (e.g., radio bookend PNGs) in legacy locations. A migration script might be needed.
- **Downstream Node Parsing:** Neither verifies if any downstream nodes parse `[Video] Saved` log messages. This could break if the path format changes.

### 4. Prioritized Recommendations

1. **Fix `video_engine.py` Post-Rename Path:**
   - Recompute and return the final path after `rename_episode` to avoid stale `pending_<ts>` paths.
   - Example:
     ```python
     final_out_path = Path(ledger.out_dir) / pending_out_path.name
     assert final_out_path.exists()
     return str(final_out_path)
     ```

2. **Bust ComfyUI Cache for Singleton:**
   - Add `IS_CHANGED` to the node that creates `new_ledger()` to ensure the singleton is always fresh.
   - Example:
     ```python
     @classmethod
     def IS_CHANGED(cls, **kwargs):
         return float("NaN")  # Forces ComfyUI to never cache this node
     ```

3. **Layout-Aware Ledger Discovery:**
   - Replace stem swaps with directory-based ledger discovery in `BatchHumoRender` and others.
   - Example:
     ```python
     ep_dir = mp4_path.parent.parent
     ledger_path = mp4_path.parent / f"{ep_dir.name}_ledger.json"
     ```

4. **Strict Cache Fallback:**
   - Ensure MusicGen/AudioGen check `_legacy_audio` first to preserve C7 and prevent VRAM spikes.
   - Example:
     ```python
     if legacy_cache_path.exists():
         return legacy_cache_path  # Preserves C7, prevents VRAM spike
     # Else generate and save to new episodes/<ep>/audio/ dir
     ```

5. **Ledger Path Validation:**
   - Add a sanity check in `Ledger.save()` to ensure `meta.paths.ledger_path` matches `self.path`.
   - Example:
     ```python
     actual = Path(self.path).resolve()
     meta_path = Path(meta.paths.ledger_path).resolve() if present else None
     if meta_path and meta_path != actual:
         raise RuntimeError(f"Ledger path mismatch: {meta_path} != {actual}")
     ```

6. **Verify Audio Determinism:**
   - Test if deleting the cache and re-running an old prompt produces identical audio bytes. If not, Rule C7 is violated.

### 5. Uncertainties for Verification

- **Is `LLMScriptWriter` being cached by ComfyUI?** Test by queuing two identical prompts. If the second run finishes instantly, the singleton is stale.
- **Does `rename_episode` update `_CURRENT.path`?** Verify that the singleton’s internal path string is updated when the directory is moved.
- **Are audio generators deterministic?** If not, Rule C7 relies entirely on the cache surviving the path reorg.
- **Downstream Log Parsing:** Check if any nodes parse `[Video] Saved` log messages. If so, ensure the path format remains compatible.

### Final Answer

**Prioritized Fixes:**
1. **Fix `video_engine.py` Post-Rename Path** to avoid stale `pending_<ts>` paths.
2. **Bust ComfyUI Cache for Singleton** using `IS_CHANGED` to ensure the singleton is always fresh.
3. **Layout-Aware Ledger Discovery** to replace stem swaps with directory-based lookups.
4. **Strict Cache Fallback** for MusicGen/AudioGen to preserve C7 and prevent VRAM spikes.
5. **Ledger Path Validation** in `Ledger.save()` to catch path-reorg bugs early.
6. **Verify Audio Determinism** to ensure Rule C7 compliance.

**Uncertainties to Verify:**
- ComfyUI cache behavior for `LLMScriptWriter`.
- `_CURRENT.path` updates in `rename_episode`.
- Audio generator determinism.
- Downstream log parsing for `[Video] Saved`.

**Key Takeaway:** Use the singleton (`_CURRENT`) but enforce cache invalidation (via `IS_CHANGED`) to ensure it’s always fresh. This balances simplicity with safety in ComfyUI.
