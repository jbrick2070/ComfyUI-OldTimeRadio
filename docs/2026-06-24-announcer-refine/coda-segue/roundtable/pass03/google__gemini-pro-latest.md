<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The logic cleanly decouples the news payload from the fiction, but fails on Python process determinism and interface contracts for the validator and LLM API.

MUST-FIX BEFORE BUILD:
1. [FALLBACK FLOOR] **Non-deterministic `hash()` breaks fallback stability.** Python 3.3+ salts `hash()` per-process (`PYTHONHASHSEED`). `POOL[abs(hash(str(cast_seed))) % len(POOL)]` will yield a different fallback on every run for the same seed. 
   *Fix:* Use the exact `hashlib` pattern already established in `select_style` (grounding `_otr_style_catalog.py:718`): `h = int(hashlib.sha256(str(cast_seed).encode("utf-8")).hexdigest(), 16)` then `POOL[h % len(POOL)]`.
2. [VALIDATOR] **Missing argument for n-gram copy guard.** The spec defines `validate_news_coda_bridge(text)`, but the validator requires checking for a ">=5-token verbatim run copied from the cleaned brief". It cannot do this without the brief.
   *Fix:* Change signature to `validate_news_coda_bridge(text, news_close_brief)`. Pass `news_close_brief` into it from `compose_news_coda`.
3. [CALL-SITE WIRING] **Immutable `LineResult` flag stamping.** The spec says "if _style_grammar_on and not nc_brief: also stamp news_coda_no_brief" after calling `compose_announcer_outro`. `LineResult` is a frozen dataclass; you cannot append to `outro_res.compose_flags` in-place.
   *Fix:* At the call site, intercept the result and rebuild it: `outro_res = dataclasses.replace(outro_res, compose_flags=outro_res.compose_flags + ("news_coda_no_brief",))`.

SHOULD-FIX:
4. [REROLL] **Consecutive User messages break strict Chat APIs.** The spec says "Reroll = re-call with an ALTERED PROMPT (append 'Attempt 2...')". If you append a new `{"role": "user"}` message directly after the first `{"role": "user"}` message, strict APIs (like Anthropic or strict OpenAI wrappers) will throw a 400 Bad Request for violating the alternating `user/assistant/user` contract.
   *Fix:* Do not append a second user message. Instead, append the LLM's invalid response `{"role": "assistant", "content": raw}`, THEN append the `{"role": "user", "content": "Attempt 2..."}` message. Alternatively, just rebuild a fresh 2-message array with the retry instruction concatenated to the original user prompt.
5. [ASSEMBLY] **Punctuation collision on normalization.** The spec says "ends with ':' or em dash (normalize if not)". If the LLM generates "And now, the truth.", blindly appending a colon results in "And now, the truth.:".
   *Fix:* `bridge = bridge.rstrip(".!?")` before appending the fallback colon.

OPTIONAL / NICE-TO-HAVE:
- [ASSEMBLY] `fact[:1].upper() + fact[1:]` fails to capitalize the first letter if the brief starts with a quotation mark (e.g., `"The mayor...`). Safe to ignore if briefs are strictly formatted, but an regex or `re.sub(r'[a-zA-Z]', lambda m: m.group(0).upper(), fact, count=1)` is safer.

CUT THESE (over-engineering):
- None. The design is extremely tight and the fallback/reroll mechanics are exactly what is needed for production safety.