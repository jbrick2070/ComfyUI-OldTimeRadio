VERDICT: yes-with-fixes. Three defects will produce import errors or test
failures that contradict the plan's own "keep existing receipts intact" and
"imports resolve" checks. All three are small edits; the core logic is sound.

---

## MUST-FIX BEFORE BUILD

**1. [test_otr_dialogue_policy.py:5] Dead import kills the entire test file**

The plan says to delete `roster_has_lemmy` from `_otr_dialogue_policy.py`
(P2.1: "Delete the dictionary/char_id roster behavior"). The test file at
`tests/test_otr_dialogue_policy.py:5` imports it by name:

```python
from nodes._otr_dialogue_policy import (
    roster_has_lemmy,     # line 5
    append_dialogue_policy,
    _COCKNEY_ORTHOGRAPHY_RULE,
)
```

If `roster_has_lemmy` is removed from the module, this import raises
`ImportError` at collection time. Pytest never runs a single test in the file —
including the qualification-receipt tests the plan explicitly says to leave
intact (P3.1: "Leave every Lemmy profile, route, and BUG-12.86
qualification-receipt test later in the file intact").

Fix: Remove `roster_has_lemmy` from the import at line 5 when deleting it from
the module. The plan must name this edit; it currently does not.

---

**2. [P2.1] `Sequence` is not imported in `_otr_dialogue_policy.py`**

Current imports (`_otr_dialogue_policy.py:3`):
```python
from typing import Any, Dict, Iterable, Union
```

The proposed signature uses `Sequence[str]`:
```python
def append_dialogue_policy(
    system_prompt: str, *, active_speakers: Sequence[str]
) -> str:
```

`Sequence` is not in scope. This raises `NameError` at function definition
time, which happens at module import — crashing ComfyUI startup and every test
that touches the module.

Fix: Add `Sequence` to the import. On Python 3.12 the canonical form is
`from collections.abc import Sequence`; `from typing import Sequence` also
works. Either is fine; just add it. The plan does not mention this import.

---

**3. [P2.1 item 3 vs. P3.1.6] Dict-container guard is underspecified; P3.1.6 test will fail as written**

P2.1 item 3 says to guard against `str` and `bytes` as the top-level
container. P3.1.6 adds a separate expectation: "A cast dictionary or
`_CastShim`-like object raises `TypeError`."

A `dict` passed as `active_speakers` iterates its KEYS (which are strings).
The per-element `isinstance(e, str)` check P2.1 item 3 mandates would see
string keys and NOT raise `TypeError`. So P3.1.6's test would fail against the
implementation the plan describes, and any dict whose key happens to be
`"LEMMY"` would silently fire the policy on a non-speaker value.

Fix: Extend the top-level container guard in P2.1 item 3 to also reject
`dict` (and Mapping types generally), or use an explicit
`isinstance(active_speakers, (str, bytes, dict))` check before iteration.
Alternatively, use `collections.abc.Sequence` and add
`if not isinstance(active_speakers, Sequence): raise TypeError(...)` at entry,
since `dict` is NOT a `Sequence`. Either approach makes P3.1.6 reachable.
The current plan text is not sufficient for an implementor to write code that
passes P3.1.6.

---

## SHOULD-FIX

**4. [P2.1 para 6] "Preserve `_COCKNEY_ORTHOGRAPHY_RULE`" is ambiguous**

The plan says "Preserve `_COCKNEY_ORTHOGRAPHY_RULE` as the one canonical
module-level text constant." The current content of that constant is the
old unscoped text:

```python
_COCKNEY_ORTHOGRAPHY_RULE = (
    "\n\nConvey the Cockney accent through phrasing, idiom, cadence, and rhythm. "
    ...
)
```

The plan also specifies completely different replacement text starting with
"For LEMMY's spoken lines only...". These cannot both be true unless "Preserve"
means "keep the name, replace the content." That is almost certainly the intent,
but the plan never says it explicitly.

The ambiguity matters because P3.1.3 tests raw strings ("For LEMMY's spoken
lines only") rather than the imported constant — so an implementor who keeps
both the name AND the old content would write code that passes P3.1.3 by
checking the raw string separately, while the constant itself holds wrong text.
The plan should say: "Update `_COCKNEY_ORTHOGRAPHY_RULE`'s content to the text
below; keep the name so existing references compile."

---

**5. [P3.2.3 / P4.3] Retry test capture point is ambiguous**

P4.3 confirms that `compose_line_draft` owns prompt construction and retry,
while `compose_line` returns `LineResult`. P3.2.3 instructs to "Force an empty
first reply so the correction retry occurs. Assert the first and second calls
carry identical system-message content."

The spec says to call `compose_line` and inspect captures from
`_recording_creative`. Whether the recording fake is attached to the inner
`generate_fn` slot passed to `compose_line_draft`, or to some wrapper, is not
spelled out. If the fake is wired to the wrong level it will see the correct
number of calls but the messages dict will not include the assembled system
text. The plan should name the exact parameter that receives `_recording_creative`
(verify: `compose_line` signature, `compose_line_draft` signature).

---

## OPTIONAL / NICE-TO-HAVE

- P3.1.1: "result contains neither `Cockney` nor the policy constant" — the
  byte-for-byte equality check already implies both. The Cockney/constant checks
  are redundant but harmless; they add clarity.
- P3.1.3: Explicitly import `_COCKNEY_ORTHOGRAPHY_RULE` in the new test and
  use `result.count(_COCKNEY_ORTHOGRAPHY_RULE) == 1` for the "exactly one copy"
  check rather than a bare `in`, to prevent a double-append regression from
  hiding in future.
- Use `from collections.abc import Sequence` (PEP 585 form) rather than
  `from typing import Sequence` on Python 3.12.

---

## CUT THESE (over-engineering)

None identified. The plan is already minimal. The P4 build-breaker audit steps
are all verifiable grep/parse actions, not ceremony.

---

## GROUNDED CONFIRMATIONS

The following plan claims were verified against the real files:

- `_otr_dialogue_policy.py:6-36`: confirmed — current `_COCKNEY_ORTHOGRAPHY_RULE`
  is the unscoped text; `append_dialogue_policy` takes a positional `roster`
  argument; `roster_has_lemmy` exists and handles str/dict.
- `_otr_line_composer.py:1049-1051`: confirmed — roster is built as
  `list(req.allowed_people or ()) + [req.speaker]` then passed positionally.
- `_otr_compose_exchange.py:391-393`: confirmed — `roster_items = list(cast or [])
  + [slot.speaker for slot in beat_group]` then passed positionally.
- `LineRequest.speaker: str` (`_otr_line_composer.py:248`): confirmed.
- `VoicedSlot.speaker: str` (`_otr_compose_exchange.py:109`): confirmed, "cast
  member name, ALL CAPS".
- Repair path (`_otr_compose_exchange.py:694-706`): confirmed — repair calls
  `_run_once` with the same `beat_group`, which calls `build_exchange_prompt`.
  Fixing line 392-393 in `build_exchange_prompt` fixes both attempt and repair
  in one edit. P4.4's claim is accurate.
- `test_otr_dialogue_policy.py:5`: confirmed — `roster_has_lemmy` is imported
  by name. This is the line that breaks if the function is deleted without
  updating the import.
- `cast` usage in `build_exchange_prompt:335-348`: confirmed — `cast` is used
  for persona lookups (roster block); it is separate from the policy call. The
  P2.3 fix correctly retains `cast` for persona lookup while replacing it as
  the policy input.

verify: `exchange_prepass_audit` ledger field existence (P5.3.5 asserts this
field proves grouped composition; could not confirm without reading
ledger schema files).

verify: `_recording_creative` parameter wiring into `compose_line` /
`compose_line_draft` (P3.2 test harness; relevant for must-fix item 5).
