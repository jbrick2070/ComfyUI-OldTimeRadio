# Round B -- Gemini (gemini-3-pro-preview) elapsed=44.1s

Here is your post-mortem analysis and architectural review. 

As a systems architect looking at a 100% local, VRAM-constrained, strict-audio pipeline, my primary lens is: **defensive programming, deterministic state, and avoiding over-engineering.**

Here are the direct answers to your questions.

---

### Q1: Sanity check BUG-121 (Filesystem fallback for radio still)

**Verdict:** The layered resolver pattern is correct for ComfyUI, but your implementation has three blind spots that will eventually crash a run.

**The Blind Spots:**
1.  **Path Traversal / Bad Types:** If `ledger.episode_id` is somehow parsed as `None`, a float, or contains `../`, `Path(f"output/otr/stills/radio_bookend_{episode_id}.png")` will either throw a `TypeError` or resolve outside your intended directory. 
    *   *Fix:* Cast to string and sanitize: `safe_ep_id = str(ledger.episode_id).replace("/", "_").replace("\\", "_")`.
2.  **The 0-byte File Trap:** ComfyUI/Torch will hard-crash if `Image.open()` hits a 0-byte or corrupted PNG (often left behind by a previous OOM or interrupted run). `Path.exists()` returns `True` for 0-byte files.
    *   *Fix:* Change your existence check from `if path.exists():` to `if path.is_file() and path.stat().st_size > 0:`.
3.  **Directory Resolution Crash:** If `otr_stills_dir()` relies on environment variables or relative paths that aren't set, it will throw an exception, bypassing your graceful `None` return. Wrap the disk-check layer in a tight `try/except Exception`.

### Q2: Sanity check BUG-123 (End-of-run VRAM cleanup)

**Verdict:** It's a brutal, ugly hack. **Keep it.** For a solo developer on a 16GB laptop who doesn't want to write custom ComfyUI execution hooks, this is the highest-ROI solution.

**Addressing your concerns:**
*   **Consumer node calling `unload_all_models()`:** Architecturally, this belongs in a `PromptServer.instance.prompt_queue` completion hook. Doing it in `VideoComposite` assumes `VideoComposite` is *always* the absolute last node to execute. If you ever add a `SaveVideo` or `Notify` node downstream, they will execute with an empty cache and unloaded models. *Accept this tech debt, but document it heavily in the node's UI description.*
*   **Is `cuda.synchronize()` necessary?** **Yes.** `cuda.empty_cache()` does *not* implicitly wait for all async CUDA streams to finish their work. If you drop `synchronize()`, you risk emptying the cache while a kernel is still winding down, which can lead to fragmented memory or a race condition on the next run. Keep the exact order: `gc.collect()` -> `cuda.synchronize()` -> `cuda.empty_cache()`.
*   **Other internal caches:** `comfy.model_management.unload_all_models()` handles the heavy weights. You do not need to manually chase IP-Adapter or CLIP Vision caches unless you are explicitly instantiating them outside of ComfyUI's standard model patcher. Don't go looking for trouble here; 14.5 GB ceiling means you just need the LLM and SDPA attention blocks cleared.

### Q3: Root causes of the 2 open symptoms

**Symptom 1: No scene-boundary cuts in composited mp4.**
*   **Root Cause:** You answered this yourself: *"The Run 1 mp4 was generated with `audio_source=humo_concat` (different mode -- the per_clip_mux failed strict_c7 and fell through)."*
*   **Why it happens:** `per_clip_mux` is the logic that reads `start_s` / `dur_s` and places clips on a timeline. Because it failed the byte-identical audio rule (Rule C7), your pipeline fell back to `humo_concat`. `humo_concat` is almost certainly a "dumb" concatenation—it just stitches the HuMo clips end-to-end and slaps them over the radar base, completely ignoring the ledger's timeline gaps. 
*   **The Fix:** Do not try to make `humo_concat` timeline-aware. Instead, figure out why `per_clip_mux` is violating Rule C7. It is likely resampling the audio, applying a crossfade, or shifting timestamps by a few milliseconds during the muxing phase. Force `per_clip_mux` to use `-c:a copy` in FFmpeg.

**Symptom 2: Radio bookend FLUX render missing.**
*   **Root Cause Hypothesis:** (a) Silent failure in `BatchFluxRender` (lines 481-502).
*   **Why:** Dynamic mode (`""`) means the prompt is being generated on the fly. If the LLM prompt generator outputs a string that breaks the FLUX tokenizer (e.g., weird unicode, or exceeds token limits), the FLUX node will throw an exception. If your `try/except` block catches `Exception` and only logs a warning, the run continues, the ledger is never stamped, and `BatchHumoRender` complains later.
*   **Observability needed to prove it:** Add a `logging.error(f"FLUX render failed: {e}", exc_info=True)` inside that `except` block. Furthermore, check the file modification time of the ledger JSON. If Hypothesis (b) (ledger overwrite) is true, you will see the ledger's `mtime` updating *after* the FLUX node finishes but without the FLUX data.

### Q4: BUG-LOCAL-125 (scene_manifest_json stub) — Fix or deprecate?

**Verdict: Deprecate and remove.**

Do not write code to populate a slot that has no consumers. This violates the "smallest change" rule. The data already exists in the ledger JSON on disk. Serializing it to a ComfyUI string output just wastes CPU cycles and clutters the graph.

**Execution:**
1.  Remove it from `RETURN_NAMES` and `RETURN_TYPES` in `OTR_SceneSequencer`.
2.  If this breaks existing workflow JSONs that have the slot wired (ComfyUI will show a red wire), rename the slot to `DEPRECATED_manifest` and leave it returning `"[]"` for one minor version, then remove it entirely in v2.0-beta.

### Q5: Preventing the BUG-124 false-positive pattern

**Verdict:** Static analysis of ComfyUI workflows is inherently flawed because ComfyUI uses `STRING` as a universal duck-type (it represents paths, JSON payloads, raw text, and base64 data interchangeably). 

Do not build a pre-commit hook or a 30-second dry-run. That is over-engineering for a solo dev.

**The most cost-effective solution is Architectural Self-Documentation:**
The static analyzer (and you) got confused because the input slot on `BatchHumoRender` was named `ledger_json`. When an MP4 path was routed into a slot named `ledger_json`, it looked like a bug. 

*   **Action:** Rename the input slot on `BatchHumoRender` from `ledger_json` to `ledger_path_or_video_fallback`. 

If the variable name explicitly describes the fallback behavior, static analysis tools (especially LLM-based ones) will read the parameter name and realize the routing is intentional. This costs you 10 seconds of refactoring and prevents the human error entirely. 

**Process Rule:** Never trust a static analysis tool that flags a "type mismatch" on a ComfyUI `STRING` port without reading the `def` of the receiving node.
