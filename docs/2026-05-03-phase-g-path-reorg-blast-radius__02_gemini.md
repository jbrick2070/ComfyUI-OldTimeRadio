# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=46.1s

### 1. Verdict

**PARTIALLY AGREE.** ChatGPT’s pathing logic and directory restructuring are correct, but replacing the global mtime scan with a global Python singleton (`_CURRENT`) in ComfyUI is a ticking time bomb due to ComfyUI's node execution cache.

### 2. Factual Errors in the ChatGPT Answer

*   **Fatal misunderstanding of ComfyUI execution (The Singleton Trap):** ChatGPT states, *"ComfyUI sequential queue guarantees no cross-episode race."* This ignores **ComfyUI's node caching**. If the node that initializes the singleton (e.g., `LLMScriptWriter`) has unchanged inputs, ComfyUI will cache it and *not execute it*. The `_CURRENT` singleton will either be uninitialized (crashing the run) or stale from the previous run (silently overwriting the wrong episode). The mtime scan you are trying to replace was likely a dirty hack originally written to bypass this exact caching issue.
*   **Ignored VRAM constraints on cache misses:** ChatGPT correctly notes that changing the AudioGen/MusicGen cache directory will cause cache misses, but misses the hardware implication. On your RTX 5080 (14.5 GB VRAM ceiling), an unexpected cache miss mid-pipeline will force ComfyUI to load the audio models into VRAM. If a heavy visual model is already loaded, this unexpected model load will instantly trigger an OOM crash.
*   **Stem swap assumption:** ChatGPT assumes the mp4 stem and `episode_id` match in your live log. However, looking at `nodes/video_engine.py:1449`, you have `safe_title = ...[:40]`. Because the title is truncated to 40 characters, the resulting mp4 stem will almost certainly *not* match the exact `episode_id` if the title is long. Stem swapping is mathematically broken here, not just brittle.

### 3. Important Things That Were Omitted

*   **The DAG is the only safe state:** In ComfyUI, the only safe way to pass state (like the current episode ID or ledger path) is through the graph (node outputs connected to node inputs). Global state in ComfyUI is an anti-pattern that always ends in race conditions or cache-invalidation bugs.
*   **C7 (Byte-Identical Audio) Guarantee:** If Bark, MusicGen, or AudioGen have *any* non-determinism (e.g., floating-point drift on different batch sizes, unseeded samplers), a cache miss permanently violates Rule C7. To guarantee C7, you *must* hit the legacy cache.

### 4. Recommended Final Fix Shape (Smallest Change, Largest Payoff)

1.  **Fix `video_engine.py` Post-Rename Return:** As ChatGPT noted, you must recompute and return the final path after `rename_episode`. Returning the `pending_<ts>` path will break downstream nodes.
2.  **Bust the ComfyUI Cache for the Singleton (The High-Payoff Hack):** If you don't want to rewire all your ComfyUI node ports to pass `ledger_path` explicitly (which is the "correct" but massive architectural fix), you must force the node that creates `new_ledger()` to execute every single time. Add a `classmethod` to that node:
    ```python
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN") # Forces ComfyUI to never cache this node
    ```
    This ensures `_CURRENT` is actually updated per queue item.
3.  **Layout-Aware Ledger Discovery (No Stem Swaps):** Update `BatchHumoRender` and others to derive the ledger from the directory structure, completely ignoring the mp4 filename.
    ```python
    # If mp4 is at episodes/<ep_id>/audio/<anything>.mp4
    ep_dir = mp4_path.parent.parent
    ledger_path = mp4_path.parent / f"{ep_dir.name}_ledger.json"
    ```
4.  **Strict Cache Fallback (Protect VRAM & C7):** For MusicGen/AudioGen (`batch_audiogen_generator.py`, `musicgen_theme.py`), the cache lookup *must* check `_legacy_audio` first before attempting to generate.
    ```python
    if legacy_cache_path.exists():
        return legacy_cache_path # Preserves C7, prevents VRAM spike
    # Else generate and save to new episodes/<ep>/audio/ dir
    ```

### 5. Uncertainties to Verify Before Merging

*   **Is `LLMScriptWriter` (or whatever creates the ledger) currently being cached by ComfyUI?** Test this by queuing two identical prompts in a row. If the second run finishes instantly without hitting the LLM, your singleton is stale.
*   **Are your audio generators 100% deterministic?** If you delete the cache and re-run an old prompt, do the output bytes match exactly? If not, Rule C7 relies entirely on the cache surviving this path reorg.
*   **Does `rename_episode` update `_CURRENT.path`?** Verify that the singleton's internal path string is actually updated when the directory is moved, otherwise `get_ledger().path` points to a ghost directory.
