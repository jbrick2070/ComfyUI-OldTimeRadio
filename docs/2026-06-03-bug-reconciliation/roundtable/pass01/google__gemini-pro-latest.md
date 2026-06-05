<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Proposed fixes violate PD1 (skipping lines shortens audio) and introduce false positives that will burn retries.

MUST-FIX BEFORE BUILD:
1. [BUG-276] The proposal to "skip-the-line" directly violates PD1 (audio must never shorten or degrade). Dropping a line breaks the narrative audio. Concrete fix: In the routing chokepoint, if `char_id == 'announcer'`, force-route to the Kokoro engine. If a character line routes to Bark without a `v2/*` preset, assign a hardcoded fallback (e.g., `v2/en_speaker_0`) so `eng_bark.py`'s `generate_voice` succeeds and audio is preserved.
2. [BUG-295] Flagging a "bare mid-sentence token" for ALL-CAPS multi-word names will false-positive when a character legitimately yells another's full name (e.g., "Get back here, ERIN SPENDER!"). Concrete fix: Restrict the leak detection strictly to inside `*...*` stage direction blocks or at the absolute start of the string before dialogue begins.
3. [BUG-264] Blindly truncating `script_brief` to exactly 350 characters via string slicing will cut mid-word (e.g., "breaking ne"), feeding malformed garbage downstream. Concrete fix: In the `@model_validator`, slice to 350, then use `.rsplit(' ', 1)[0]` to snap back to the last complete word.

SHOULD-FIX:
1. [BUG-264] When trimming `key_terms`, keep the first `_MAX_KEY_TERMS`. [ASSUMPTION] Weak LLMs typically output the most relevant terms first and hallucinate/drift at the end of the list.

OPTIONAL / NICE-TO-HAVE:
- [BUG-276] Add a unit test that mocks the router output with a preset-less character line and asserts the final audio array length > 0 (proving the fallback preset engaged and `eng_bark.py` did not raise `EngineUnusable`).

CUT THESE (over-engineering):
1. [BUG-295] The compose retry loop for name leaks. Burning an expensive LLM retry for a deterministic formatting artifact is wasteful and risks exhausting the retry ladder. Safe to cut: Replace the retry trigger with an in-place regex scrub that simply strips the ALL-CAPS name from inside the `*...*` block.