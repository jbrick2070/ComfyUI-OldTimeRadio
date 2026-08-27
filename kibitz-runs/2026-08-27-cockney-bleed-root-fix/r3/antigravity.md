VERDICT: build-ready as-is? yes.
The plan's interface contract migration across `_otr_dialogue_policy.py`, `_otr_line_composer.py`, and `_otr_compose_exchange.py` is atomic, type-safe, and fail-closed; the prompt sequencing preserves byte-identity on no-Lemmy paths and across retry/repair cycles without modifying canonical workflow slots or node schemas.

MUST-FIX BEFORE BUILD:
None.

SHOULD-FIX:
1. [P2.1 / P3.1] System prompt `None` type-coercion contract consistency.
   - Defect: In`nodes/_otr_dialogue_policy.py`, `system = system_prompt or ""` ensures a string return when `system_prompt` is `None` or `""`. However, if `append_dialogue_policy(None, active_speakers=("MARLOW",))` is called, returning `""` rather than `""` vs `None` is standard, but callers should not assume `None` is preserved as `None` if they rely on identity checks downstream (`system_prompt is None`).
   - Fix: Grounding in`nodes/_otr_line_composer.py:1040-1051` and`nodes/_otr_compose_exchange.py:389-393` confirms both callers always resolve `system` to a non-null `str` before invoking `append_dialogue_policy`. Keep `return (system_prompt or "")` in `append_dialogue_policy` and document in `tests/test_otr_dialogue_policy.py` that `append_dialogue_policy(None, active_speakers=...)` returns `""` (or `_COCKNEY_ORTHOGRAPHY_RULE`).
2. [P3.2 / P3.3] System message inspection vs user message fixture in tests.
   - Defect: In `tests/test_phase1_composer_prompt.py` and `tests/test_compose_exchange.py`, fake generators inspect captured `messages`. In `compose_line_draft`, `messages[0]` is guaranteed `{"role": "system", ...}` and `messages[1]` is `{"role": "user", ...}`. In `build_exchange_prompt`, `messages[0]` is `{"role": "system", ...}`.
   - Fix: Ensure integration assertions explicitly check `messages[0]["content"]` (role == `"system"`) for `_COCKNEY_ORTHOGRAPHY_RULE` presence/absence, and do not test against joined user prompt text where speaker names naturally appear.
3. [P5.3] Headless qualification port and process reset synchronization.
   - Defect: In `scripts/otr_headless_canonical.ps1:162-171`, selective process reset (`Stop-OtrPython`) and VRAM querying occur before launching the ComfyUI API server, but manual residual processes from ad-hoc runs on non-standard ports could cause transient bind collisions if `-Port` is defaulted.
   - Fix: Follow P5.3 step 1 precisely by running `scripts/otr_headless_canonical.ps1` with default `-Port 0` (ephemeral port resolution via `Resolve-OtrHeadlessPort` at lines 76-93) to avoid colliding with any resident server.

OPTIONAL / NICE-TO-HAVE:
1. [P2.1] Explicit docstring update in`nodes/_otr_dialogue_policy.py` noting that `active_speakers` must be the active dialogue turn speakers (`LineRequest.speaker` or `VoicedSlot.speaker`), explicitly warning future contributors against passing cast dictionaries or full cast lists.

CUT THESE (over-engineering):
1. None in the current plan. The plan already explicitly rejects post-generation dialect scrubbers, regex blacklists, and workflow modifications (P2.4), maintaining minimal surgical scope.
