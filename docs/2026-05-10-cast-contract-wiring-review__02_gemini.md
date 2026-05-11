# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=46.6s

### 1. Verdict
I PARTIALLY AGREE; ChatGPT correctly identified the severe Rule C7 (reproducibility) violations regarding LEMMY and the `seed=0` sentinel, but it completely missed ComfyUI's specific frontend/backend seed semantics and introduced a Python type-comparison trap that will cause an infinite retry loop.

### 2. Factual Errors in ChatGPT
*   **ComfyUI Seed Semantics:** ChatGPT suggests generating an `effective_seed` in the backend using `SystemRandom()` if `seed < 0`. This fundamentally misunderstands ComfyUI. ComfyUI's frontend handles "randomize" by generating a random 64-bit integer in JS and sending it to the Python backend. The backend should *never* generate its own random seeds; it should blindly trust the incoming `seed` integer to ensure the ComfyUI workflow graph and UI remain the source of truth for reproducibility.
*   **Type Mismatch in Equality Suggestion:** ChatGPT suggested `tuple(outline.cast) == tuple(req.character_cast)` or exact equality. If `outline.cast` is a list (standard JSON/Pydantic array parsing) and `req.character_cast` is a tuple, `outline.cast == req.character_cast` evaluates to `False` even if the contents match. You must explicitly cast to compare them (e.g., `outline.cast == list(req.character_cast)`).

### 3. Important Omissions
*   **The ComfyUI `seed=0` Anti-Pattern:** Your code `cast_rng = _random.Random(int(seed)) if int(seed) != 0 else None` actively fights ComfyUI. If a user sets the ComfyUI seed widget to "fixed" and types `0`, your backend treats it as "unseeded/random", breaking Rule C7. 
*   **LLM Seeding for Style Generation:** You pass `temperature=resolved["temperature"]` to `_generate_style_via_llm` (Step D.2), but you don't explicitly pass the `seed`. If your local Mistral-Nemo inference engine isn't receiving the episode seed for this call, the style output will drift, violating Rule C7.
*   **`OutlineRequest` Default Trap:** ChatGPT caught this, but missed *why* it's so dangerous. `character_cast: tuple[str, ...] = ()` followed by `if not (1 <= n <= 6): raise ValueError` in `__post_init__` means `OutlineRequest()` will crash on instantiation. Dataclasses evaluate defaults *before* `__post_init__`. Any legacy tests doing `OutlineRequest(news_seed="x", style="y", target_words=350)` will instantly fail.

### 4. Short Recommendation
*   **Fix Rule C7 (Determinism):** In `nodes/OTR_LedgerScriptWriter.py`, delete the `if int(seed) != 0` logic. Always use `cast_rng = _random.Random(int(seed))`. Pass `cast_rng` into `config.cast_pools` so LEMMY's 11% roll uses the deterministic sequence, not `SystemRandom`.
*   **Fix Cast Drift Check:** In `nodes/_otr_outline.py`, change the set equality to `if outline.cast != list(req.character_cast):`. This enforces exact order, exact count, and handles the list vs. tuple type difference without triggering false-positive retry loops.
*   **Fix OutlineRequest Default:** Remove the `= ()` default from `character_cast` in `OutlineRequest` to force explicit passing and prevent instantiation crashes in legacy code/tests.
*   **Add Uniqueness Assertion:** In `OTR_LedgerScriptWriter.py`, add `assert len(char_id_by_name) == len(cast_rows) - 1` (assuming 1 announcer) after building the dict to catch silent duplicate name collapsing.
*   **Verify LLM Seeding:** Ensure `generate_fn` and `_generate_style_via_llm` are actually passing the `seed` parameter to the local LLM backend.

### 5. Uncertainties
*   I am uncertain how `resolved["num_characters"]` is defined in your UI. If the user selects "2", does that mean 2 actors *plus* the announcer, or 2 actors *including* the announcer? If the latter, and they select "1", `OutlineRequest` will receive 0 characters, crashing `__post_init__`.
*   I am uncertain if your `pick_announcer()` guarantees the `char_id` is `"c01"` or `"announcer"`. If the announcer's `char_id` is `"c01"`, but downstream beats hardcode `"announcer"`, the downstream composer might crash looking up `"announcer"` in a ledger that only has `"c01"`.
