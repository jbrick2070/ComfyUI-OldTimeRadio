# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=50.9s

Here is the pre-soak peer review of the v2.0-alpha overnight code. 

### Element 1 -- `lock_to_episode` read-and-compare upgrade
*   **Fix-needed probability:** 25%
*   **One-line reasoning:** TOCTOU on read is safely caught, but the final `write_text` is not atomic, risking JSON corruption on crash.
*   **Verdict badge:** AMBER
*   **Concrete failure mode:** If the ComfyUI process is interrupted (user kills it, or a VRAM OOM occurs) exactly during `locked_path.write_text(...)`, it leaves a 0-byte or truncated JSON file. The next run will hit the `json.JSONDecodeError` block and raise a fatal `RuntimeError`, permanently bricking that episode directory until manually cleared. Use a temporary file and `os.replace()`.

### Element 2 -- `OrphanClass` 5-bucket Enum + `apply_classifications`
*   **Fix-needed probability:** 10%
*   **One-line reasoning:** Logic is sound and handles case-insensitivity well, though it intentionally drops 3 out of 5 classification buckets.
*   **Verdict badge:** GREEN
*   **Concrete failure mode:** If the LLM hallucinates a `target_character_id` that doesn't exist in the contract, the function silently skips it. This is safe locally, but pushes the burden of handling that orphan back to the caller (which triggers a trap in Element 3).

### Element 3 -- Plateau-bounded `repair_orphans` loop
*   **Fix-needed probability:** 95%
*   **One-line reasoning:** The plateau detector will infinitely loop/crash if the LLM correctly identifies an orphan as `GENUINELY_NEW` or `DISCARD`.
*   **Verdict badge:** RED
*   **Concrete failure mode:** If a script contains a stray stage direction (e.g., "FOOTSTEPS"), the classifier correctly buckets it as `DISCARD`. `apply_classifications` (Element 2) ignores `DISCARD`s, meaning the orphan is *not* removed from the script or added to the contract. On iteration 2, `residual_orphans` is identical to `prev_residual`, instantly raising `CastContractUnreparable` and crashing the pipeline.

### Element 4 -- `_voice_backends` registry + lazy driver self-registration
*   **Fix-needed probability:** 20%
*   **One-line reasoning:** `get_factory` does not auto-initialize the registry, meaning lazy imports never fire unless manually triggered.
*   **Verdict badge:** AMBER
*   **Concrete failure mode:** The first time `get_factory("bark")` is called, it raises a `KeyError` because `_REGISTRY` is empty. The `_register_default_drivers()` function exists but is never implicitly called by the registry's own getter.

### Element 5 -- `VoiceBackend` runtime-checkable Protocol
*   **Fix-needed probability:** 40%
*   **One-line reasoning:** `@runtime_checkable` on Python Protocols silently ignores attributes (like `engine_name`), checking only for methods.
*   **Verdict badge:** RED
*   **Concrete failure mode:** A stub driver that implements `load`, `generate`, and `unload` but forgets to define the `engine_name` string will perfectly pass `isinstance(obj, VoiceBackend)`. The pipeline will then crash with an `AttributeError` later when the orchestrator tries to read `backend.engine_name`.

### Element 6 -- `OTR_VoiceRender.render` dispatch with try/finally cleanup
*   **Fix-needed probability:** 100%
*   **One-line reasoning:** Violates Rule C7 (byte-identical output) by omitting a seed, and per-line load/unload will cause catastrophic VRAM fragmentation.
*   **Verdict badge:** RED
*   **Concrete failure mode:** Two massive failures here. First, Bark and Kokoro are stochastic; without passing a deterministic `seed` to `generate()`, your audio will change every run, failing Rule C7. Second, loading and unloading a TTS model for *every single line of dialogue* (often 100+ lines per episode) will thrash your disk and fragment your 16GB VRAM until PyTorch throws a CUDA OutOfMemory error around line 40.

### Element 7 -- `render_prompt` few-shot assembly + period system prompt
*   **Fix-needed probability:** 5%
*   **One-line reasoning:** Simple, pure string formatting that correctly separates system and user prompts.
*   **Verdict badge:** GREEN
*   **Concrete failure mode:** If `user_instruction` contains markdown or tokens that mimic your LLM's specific system-prompt boundaries (e.g., `<|system|>`), it could jailbreak the 1940s constraints. Structurally, however, the Python code is completely safe.

***

### Where would you push back?
1. **The "Fresh backend per call" design (Element 6):** I know you don't want low-level VRAM optimization, but instantiating, loading, and unloading a model *per line of dialogue* is a non-starter. It turns a 2-minute generation into a 45-minute disk-thrashing nightmare. You need at least a basic context manager (`with VoiceBackend("bark") as tts:`) at the orchestrator level so the model stays in VRAM for the duration of the batch.
2. **Ignoring valid classifications (Element 2 & 3):** You built a 5-bucket Enum but only process 2 of them. If the LLM finds a `GENUINELY_NEW` character, the system must either add them to the `CastContract` as a new entity or explicitly mark them as "unvoiced" in the script. Ignoring them guarantees a plateau crash.

### What's the load-bearing weak spot?
**Element 6's lack of a `seed` parameter.** You have a hard constraint (Rule C7: byte-identical audio between runs). The current `generate(text, temperature=temperature)` signature makes this impossible. If you don't thread a seed from the ComfyUI node down through the Protocol and into the Bark/Kokoro drivers, the entire project fails its primary acceptance criteria on the very first end-to-end test.
