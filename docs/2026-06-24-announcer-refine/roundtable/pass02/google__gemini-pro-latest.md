<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core pipeline logic holds, but a Python slicing bug and a cyclic dependency on the outline premise will break the build immediately.

MUST-FIX BEFORE BUILD:
1. [1] **Cyclic dependency in `select_style`**: The plan moves `build_story_contract` (which replaces `select_style`) to *before* `OutlineRequest`. However, grounding shows `select_style` (lines 718-733) requires `outline.premise` to evaluate `premise_wants_emergency`. You cannot evaluate the premise before the outline exists. **Fix**: Either compute the emergency flag from `script_brief` pre-outline, OR build the contract *after* the outline macro generates the premise but *before* the phase/beat prompts.
2. [3] **Python negative slicing bug**: The plan dictates truncating the original intent to `_INTENT_MAX - len(enrichment)` FIRST. If `len(enrichment)` > `_INTENT_MAX` (200), this yields a negative number. In Python, `intent[:-5]` does not clamp to zero; it slices 5 characters off the *end* of the string, leaving a massive string that blows past the limit. **Fix**: Clamp the slice index: `intent[:max(0, _INTENT_MAX - len(enrichment))]`.
3. [2/Job 1] **Token overlap false-positives**: Rejecting the open if it "token-overlaps" `ending_change` will trigger near 100% false positives because both strings will share stop words ("the", "a") and character names. **Fix**: The overlap check must explicitly strip stop-words and character names, or require a strict n-gram match (e.g., 3-gram overlap).
4. [2/Job 3] **Double lead-in generation**: If the deterministic news lead-in ("From tonight's headlines:") is injected as a hardcoded text prefix, the LLM will still generate its own conversational lead-in, resulting in a stutter ("From tonight's headlines: The real story tonight is..."). **Fix**: Add an explicit instruction to the flagged `_ANNOUNCER_OUTRO_SYSTEM`: "Start immediately with the concrete facts. Do not write an introductory phrase."

SHOULD-FIX:
1. [2/Job 2] **Climax line lookup gap**: The plan says "pass the CLIMAX beat's line as climax_character_line" citing `_climax_beat_id` (:3266). `_climax_beat_id` is merely a string/integer ID, not the text. **Fix**: Explicitly specify the lookup at the outro call site (:4615): iterate `led.data["lines"]` and extract the text where `line.beat_id == _climax_beat_id`.
2. [2/Job 1] **`compose_announcer_intro` signature**: The call site at :4465 currently only passes `script_brief`. **Fix**: Update the signature of `compose_announcer_intro` to accept `safe_open_brief` as an optional kwarg, and use it exclusively when the `story_scaffold` flag is active.
3. [8] **`opening_status_quo` extraction**: The plan adds `opening_status_quo` to the outline but does not define how it gets mapped into the `SafeOpenBrief` at the call site. **Fix**: Ensure the `Outline` dataclass/schema actually exposes `opening_status_quo` so it can be read at :4465.

OPTIONAL / NICE-TO-HAVE:
- [2/Job 3] Dynamic validation bounds: Since the news coda gets a new length band (18-45 words), pass `min_chars` and `max_chars` overrides to `validate_announcer_line` when the flag is on, rather than hardcoding the old 14-34 word limits.

CUT THESE (over-engineering):
- [1] "delete the late select_style(outline.premise,...)". **Why safe to cut**: If you discover other callers depend on the post-outline style resolution, or if the emergency routing cannot be cleanly decoupled from the premise, it is safer to leave the late resolution as a fallback for off-flag paths rather than blindly deleting it.