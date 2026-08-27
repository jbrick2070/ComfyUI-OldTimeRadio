VERDICT: yes-with-fixes — The root cause analysis and active-speaker scoping architecture are sound and strictly bounded, but the API specification has a silent Python string-iterable failure mode and needs explicit protection for existing voice qualification tests.

MUST-FIX BEFORE BUILD:
1. [P2.1] Silent Python string-iterable trap in `append_dialogue_policy`.
   - Defect: In Python, `str` is an `Iterable[str]`. If a caller passes `active_speakers="LEMMY"` (a string rather than a tuple/list of strings), element-level `isinstance(item, str)` checks pass for each single character (`'L'`, `'E'`, `'M'`, etc.), but none match `"LEMMY"`. The function silently returns the prompt without policy injection, without raising `TypeError`.
   - Fix: In`nodes/_otr_dialogue_policy.py:append_dialogue_policy`, explicitly reject string/bytes before iterating: `if isinstance(active_speakers, (str, bytes)): raise TypeError("active_speakers must be an Iterable of strings, not a single string")`.

2. [P3.1] Risk of overwriting voice qualification test suite in `tests/test_otr_dialogue_policy.py`.
   - Defect: P3.1 specifies "Replace the roster-oriented tests with active-speaker tests" in `tests/test_otr_dialogue_policy.py`. Lines 53–156 of `tests/test_otr_dialogue_policy.py` contain load-bearing qualification receipt tests (`test_the_indextts2_route_is_approved_and_carries_its_receipt`, `test_the_canonical_route_is_routing_not_a_qualification_claim`, etc.) pinned for BUG-12.86. An unconstrained file rewrite instruction risks dropping these qualification tests.
   - Fix: Explicitly restrict the test replacement in `tests/test_otr_dialogue_policy.py` to lines 1–52 (replacing `test_roster_has_lemmy` and `test_append_dialogue_policy`), leaving the voice policy qualification test suite (lines 53–156) intact.

SHOULD-FIX:
1. [P0.3 / P0.5 / P2.1] Unify prompt text narrative between single-line and mixed-exchange specs.
   - Defect: P0.3 specifies a Lemmy-only accent rule for single-line requests, whereas P0.5 specifies a mixed-exchange rule containing an isolation clause for other speakers. P2.1 then provides a single monolithic 3-sentence policy block containing both the Lemmy directive and the non-Lemmy isolation clause for all Lemmy-active prompts. In a solo Lemmy per-line prompt (`active_speakers=("LEMMY",)`), the prompt instructs the model that "Every other character must retain that character's own speech register" even though no other character exists in that turn.
   - Fix: Clarify in P0.3 / P2.1 that a single static 3-sentence rule is intentionally shared across both single-speaker and mixed-exchange callers to keep the prompt construction deterministic and branch-free.

2. [P2.1 / P3.1] Maintain named constant for policy text in`nodes/_otr_dialogue_policy.py`.
   - Defect: P2.1 replaces `_COCKNEY_ORTHOGRAPHY_RULE` with new text but does not specify whether the constant name is preserved or renamed. Tests in P3.1 check for string fragments, which leads to fragile substring matching across test files.
   - Fix: Export the exact scoped text as a module-level constant (e.g. `_COCKNEY_ORTHOGRAPHY_RULE` or `_SCOPED_COCKNEY_POLICY`) in`nodes/_otr_dialogue_policy.py` so unit tests can assert presence/absence against the canonical constant.

OPTIONAL / NICE-TO-HAVE:
1. [P5.1] Add a lightweight test asserting that `LineRequest(speaker="LEMMY", ...)` and `VoicedSlot(speaker="LEMMY", ...)` directly satisfy the `active_speakers` iterable contract without string manipulation.

CUT THESE (scope / over-engineering):
None. The plan is strictly minimal, touches only 3 files in`nodes/`, avoids regex/post-generation scrubbing shims, avoids workflow modifications, and preserves canonical graph integrity.

[ASSUMPTION] Assumes `LineRequest.speaker` and `VoicedSlot.speaker` always arrive as raw character name strings (e.g. `"LEMMY"`), verified agains`nodes/_otr_line_composer.py:248` and`nodes/_otr_compose_exchange.py:109`.
