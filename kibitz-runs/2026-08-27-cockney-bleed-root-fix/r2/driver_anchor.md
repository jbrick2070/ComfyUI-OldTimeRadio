VERDICT: build-ready as-is? yes-with-fixes. R1 closed the architecture and category boundary, but a code-ready handoff still needs an exact helper skeleton and exact prompt-capture assertions so two reasonable implementers cannot choose incompatible behavior.

MUST-FIX BEFORE BUILD:

1. [P2.1] CONFIRMED — the plan specifies a “private active-speaker predicate” but does not name it or give its complete branch order. Pin `_active_speakers_have_lemmy(active_speakers: Sequence[str]) -> bool`: reject top-level `str`/`bytes`, validate each element before testing any name, then compare `speaker.strip().upper() == "LEMMY"`. Validation-before-return matters: `("LEMMY", object())` must not return True while hiding a bad second category.
2. [P3.1-P3.3] CONFIRMED — generic substring checks can become false positives if a routed bank prompt ever mentions an accent. Pin tests to the canonical `_COCKNEY_ORTHOGRAPHY_RULE` for presence/absence, then separately assert the two required scope phrases inside the constant. Integration tests must inspect `recorded_call["messages"][0]["content"]` (exchange) or `recorded_messages[0]["content"]` (line); do not assume a raw-string return or invent a prompt accessor.
3. [P3.3.4] CONFIRMED — the existing `_fake_gen_valid` does not retain messages. State that the prepass invariant test must use a small recording wrapper around that existing fake, not read state that the fake does not expose.

SHOULD-FIX:

1. [P2.1] CONFIRMED — preserve the leading `"\n\n"` on the canonical policy constant so Lemmy-active concatenation retains the current system-prompt separation. The no-Lemmy branch must return the exact original object value after `system_prompt or ""`.
2. [P3.1] CONFIRMED — specify that all existing profile/route imports and tests after the two roster tests remain untouched; remove only the obsolete `roster_has_lemmy` import.
3. [P3.2] CONFIRMED — combine Lemmy retry scope with the existing empty-response correction path rather than adding a second retry framework. Compare the system string across the two captured calls while retaining all current correction-turn assertions.
4. [P3.3] CONFIRMED — define a two-slot Lemmy fixture/raw response locally in `tests/test_compose_exchange.py`; current `_raw_for` is hard-coded for MARLOW/REESE and cannot correctly document the mixed speaker contract.
5. [P4] CONFIRMED — add a frozen-diff check that `git diff -- workflows/otr_canonical.json` is empty. General `git diff --check` cannot prove no workflow change.

OPTIONAL / NICE-TO-HAVE:

- Use `casefold()` instead of `upper()` only if the project already standardizes on it for cast identity. It does not matter for the ASCII reserved name LEMMY, so consistency should decide.

CUT THESE (over-engineering):

1. Do not introduce a policy dataclass, enum, registry, or per-bank policy map. The exact defect has one constant, one predicate, and two callers.
2. Do not add runtime logging for each prompt decision; captured tests and existing production artifacts provide the receipt without exposing prompt content in normal logs.
