VERDICT: build-ready as-is? yes-with-fixes. The architecture, implementation, and wiring have converged; two final handoff ambiguities remain in test imports and the live audit's fail-closed uniqueness behavior.

MUST-FIX BEFORE BUILD:

1. [P3.2/P3.3] CONFIRMED — the line and exchange integration files do not currently import `_COCKNEY_ORTHOGRAPHY_RULE`. The plan tells them to assert that constant but only explicitly preserves its import in `test_otr_dialogue_policy.py`. Pin `from nodes._otr_dialogue_policy import _COCKNEY_ORTHOGRAPHY_RULE` in both integration test files, or every new assertion is a `NameError`.
2. [P5.3.6] CONFIRMED — indexing `ledger.lines[]` by `beat_id` must fail closed on blank or duplicate beat IDs. A dictionary comprehension would silently overwrite duplicates and could attach the wrong speaker/slot to a beat, fabricating the same kind of qualification the audit is meant to prevent. Require exactly one dialogue line row per indexed beat ID used in reconstruction; duplicate/blank identity makes the receipt fail.

SHOULD-FIX:

1. [P2.1] CONFIRMED — require the helper docstring to say `active_speakers` means current output speakers only and name the two authoritative values: `LineRequest.speaker` and `VoicedSlot.speaker`. This protects the new category boundary from a future “full cast for safety” regression.
2. [P3.3] CONFIRMED — direct no-Lemmy/mixed prompt tests may call `build_exchange_prompt`; repair identity must call `compose_exchange` with `_CountingGen`. State this separation so an implementer does not overbuild all cases through the retry layer.
3. [P5.3] CONFIRMED — if `media_archive` source acquisition fails for an external/feed reason, stop that leg and record it. Do not silently roll or switch banks inside the same receipt. A separately invoked, explicitly pinned `original` leg may prove the shared inline path, but it must be labeled as the fallback bank rather than a media-archive pass.
4. [P6] CONFIRMED — final project records must update the GO_FORWARD wording that currently recommends a global orthography sentence; otherwise the next implementer sees two contradictory authorities.
5. [P7] CONFIRMED — add startup/import collection failure to the stop conditions explicitly; the focused policy module import is the earliest guard for the removed symbol and new `Sequence` import.

OPTIONAL / NICE-TO-HAVE:

- State an expected source-code/test diff budget: three node helpers/callers and three existing test files, plus admitted records. Any additional production module requires a new grounded reason.

CUT THESE:

1. Do not add a standalone persistent qualification script for a one-off ledger reconstruction unless the implementation discovers the receipt will be reused. A temporary audited probe, deleted before commit, is sufficient.
2. Do not add a fallback policy compatibility alias for `roster_has_lemmy`; it would preserve the wrong category and undermine the root fix.
