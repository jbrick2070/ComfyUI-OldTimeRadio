# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=32.0s

### 1. Verdict
I **STRONGLY AGREE** with ChatGPT's core recommendation to drop the timestamp entirely on new writes, as it is the only way to strictly satisfy your Rule C7 (byte-identical MP4s) for clean-cache runs.

### 2. Factual Errors in ChatGPT's Answer
ChatGPT’s answer is exceptionally solid, but there is one minor technical nuance regarding ComfyUI:
*   **Concurrency/Race Conditions:** ChatGPT states, *"If you later introduce parallel cue generation, deterministic filenames create a possible race..."* ComfyUI's core execution engine (`execution.py`) is strictly single-threaded. Nodes execute sequentially. Even if you queue 50 prompts or have a workflow with 10 AudioGen nodes, they will never evaluate concurrently. You do not need to worry about race conditions between ComfyUI nodes, only between entirely separate ComfyUI instances (which you aren't running).

### 3. Important Omissions
*   **Corrupted Cache Hits (The Interruption Problem):** By moving to deterministic filenames (`<prefix>.wav`), you introduce a new risk. If you cancel the ComfyUI queue *while* the model is saving the WAV file, you will leave a half-written, corrupted `<prefix>.wav` on disk. On the next run, `_find_cached` will see it exists, assume a cache hit, and pass garbage to downstream nodes, breaking the run permanently for that seed/prompt. **You must use atomic writes.**
*   **Model Name in Cache Key:** You mentioned `model_revision` and `decode_mode`, but didn't explicitly mention the base model name (e.g., `audiogen-medium` vs `musicgen-small`). If the model name isn't in the digest payload, switching models in the ComfyUI node will incorrectly load the cached output of the previous model. 

### 4. Short Recommendation
1.  **Adopt the deterministic write:** Change new writes to `<prefix>.wav`. This is the highest-payoff change for Rule C7, ensuring FFmpeg metadata remains identical even on a completely fresh machine/cache.
2.  **Implement atomic writes:** When generating a cache miss, save the audio to `<prefix>.tmp`, then use `os.replace(tmp_path, final_path)`. Since you already handled `os.replace` retries in Phase B, you know how Windows file-locking behaves. This prevents corrupted cache hits.
3.  **Use ChatGPT's lookup logic:** Its `iterdir()` + `startswith` approach is much safer than `Path.glob` (which chokes on brackets `[]` in prompts) and avoids the fragility of `mtime` sorting.
4.  **Defer Cache Key V2:** Do not add `model_revision`, `sample_rate`, etc., to the hash payload in Phase D. It violates your "smallest change" constraint and forces you to write complex fallback logic for legacy files. Log it as a Phase E or v2 ticket.

### 5. Uncertainties to Verify
*   **Does your `_load_cached_wav` or `save_wav` already handle atomic writes?** If `save_wav` is just a wrapper around `torchaudio.save` or `soundfile.write`, it is *not* atomic. You will need to implement the `.tmp` -> `.wav` rename at the call site in `musicgen_theme.py` (line ~125) and `batch_audiogen_generator.py` (line ~85).
*   **Are there any other callers of `_cache_key`?** You mentioned you haven't seen any, but a quick `Find in Files` for `_cache_key` across your custom nodes directory is mandatory before changing its signature or behavior.
*   **Is `episode_seed` truly deterministic?** Ensure `episode_seed` isn't accidentally being cast from a float or changing format (e.g., `12345` vs `12345.0`) upstream, as this would silently change the string payload and break the cache hash.
