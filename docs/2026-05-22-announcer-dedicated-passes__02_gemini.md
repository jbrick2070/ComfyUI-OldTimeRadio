# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=28.1s

**1. Agreement Statement**
I PARTIALLY AGREE with the core recommendation; the two-pass architecture (intro early, outro post-loop) is structurally correct, but ChatGPT's suggestion to feed the generated script into the outro prompt is a dangerous trap for your strict 14.5 GB VRAM ceiling.

**2. Factual Errors in the ChatGPT Answer**
*   **VRAM / KV Cache Misunderstanding:** ChatGPT states that feeding the finished script into the outro prompt is "additional inference time, not additional resident model memory." This is factually incorrect for local LLMs. Increasing the prompt context size linearly increases the KV Cache, which consumes VRAM dynamically. On a 16 GB laptop GPU running near a 14.5 GB ceiling, appending a full episode transcript to the prompt right at the end of the generation cycle is a massive OOM (Out of Memory) risk.
*   **C7 (Byte-Identical) Interpretation:** ChatGPT waffles on what "byte-identical" means, suggesting you might need a toggle to preserve old, broken outputs. In continuous generation systems, rule C7 means *deterministic reproducibility* (same seed + same inputs = same audio bytes), not backwards compatibility with deprecated bugs. 

**3. Important Omissions**
*   **Prompt Caching / Prefix Sharing:** To save both VRAM and generation time, the system prompts for `compose_announcer_intro` and `compose_announcer_outro` should share the exact same prefix as the standard character `compose_line`. Local inference engines (like llama.cpp or vLLM backends) can reuse the KV cache for the shared system prompt, drastically reducing the memory spike of adding two new LLM calls.
*   **The Root Cause of the Key-Name Bug:** ChatGPT says to fix the key-name bug but misses the architectural fix: the writer node shouldn't be guessing aliases (`news_close_brief` vs `close_brief`). The upstream news interpreter node must enforce a strict JSON schema output, and the LedgerScriptWriter should strictly parse it. 

**4. Short Recommendation**
*   **Implement two separate passes** in `nodes/_otr_line_composer.py`: `compose_announcer_intro` (called on the first beat) and `compose_announcer_outro` (called post-loop).
*   **Do NOT feed the generated script into the outro.** To protect your VRAM ceiling, feed the outro pass *only* the `news_close_brief` and the `script_brief`. The announcer doesn't need to know exactly what the characters said to deliver a thematic closing frame.
*   **Repurpose the broken helper:** Fix the key-name bug in `override_announcer_close`, rename it to `fallback_announcer_outro`, and use it *only* if the creative LLM outro pass fails JSON validation or times out.
*   **Create a deterministic intro fallback:** Write a simple Python string formatter that injects the `script_brief` into a hardcoded intro template (e.g., `"Tonight on SIGNAL LOST: {script_brief}."`).
*   **Skip generic composition for the outro:** In the `OTR_LedgerScriptWriter` loop, if `speaker_role=="announcer"` and it's the final beat, `continue` the loop and leave the row empty until the post-loop outro pass fills it. This saves a useless LLM call.

**5. Uncertainties to Verify**
*   **Loop Execution Context:** I am assuming `OTR_LedgerScriptWriter` processes all beats in a single Python `for` loop within *one* ComfyUI node execution step. If the beats are being processed across multiple ComfyUI node executions (e.g., via a custom loop mechanism), a "post-loop" pass will require a dedicated ComfyUI node rather than just a function call at the end of the script writer.
*   **LLM Backend Memory Allocation:** I need to know what backend is serving your local model (e.g., ComfyUI-GGUF, Ollama, LM Studio). If the backend pre-allocates a fixed KV cache size at startup, the VRAM spike risk is mitigated (it will just truncate/fail if it exceeds context), but if it allocates dynamically, the OOM risk at the end of the script is high.
