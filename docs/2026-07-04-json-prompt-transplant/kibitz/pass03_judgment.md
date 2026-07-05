# pass03_judgment.md -- r3 judgment log

## Codex r3 (raw at ``pass03_raw/codex.md``)

| Codex claim | Status | Rationale |
|---|---|---|
| MF1 Chunk 2 -> Chunk 4 order wrong | ACCEPT | r3 dropped old Chunk 2 (no longer needed). |
| MF2 line_grounding cannot be omitted | ACCEPT | line_grounding stays populated; Phase A never touches it. |
| MF3 Chunk 3 signature needs explicit registry input | ACCEPT | Signature rewritten. |
| MF4 new seams unaddressable through StoryPromptProfile | ACCEPT | Chunk 1 adds 4 fields; extractor still reads pack directly. |
| MF5 TEMPLATE_SEAMS split breaks validators | ACCEPT | Introduced ``ALL_TEMPLATE_SEAMS`` union + back-compat alias. |
| MF6 byte-identity guard lands after risky rewrite | ACCEPT | r3 reordered: Chunk 5 (harness) before Chunk 4 (extractor tests). |
| SF1 ``style_pick_inventor_user_template`` conditional | DEFER | Not blocking Phase A extractor. r4 confirms. |
| SF2 Chunk 5 fixture unspecified | ACCEPT | mirror_nodes fixture named + creative_repo_id=None pinned. |
| SF3 rollback story missing | ACCEPT | Chunk 7 rollback discipline block. |
| SF4 sibling branch discipline undefined | ACCEPT | ``main`` explicit + push verification. |
| CUT operator sign-off at Chunk 4 | ACCEPT | Chunk 4 redefined as extractor coverage tests, no operator decision needed. |

## Fable r3 (raw at ``pass03_raw/fable.md``)

| Fable claim | Status | Rationale |
|---|---|---|
| MF-W1 MF-C6 "OR OMITTED" misread | ACCEPT | Old Chunk 4 (pack rewrite) removed; new Chunk 4 doesn't rewrite science pack. |
| MF-W2 Chunk 2 too narrow | ACCEPT | Chunk 2 dropped; line_grounding stays as-is. |
| MF-W3 Chunk 5 capture mechanics hand-wavy | ACCEPT | Snapshot mechanism from ``test_compat_drift.py:27-52``. Identity pytest goes in OTR-side suite. |
| SF rollback story | ACCEPT | Chunk 7. |
| SF Chunk 7 says push per chunk but chunks 0-6 change no OTR code | ACCEPT | Sibling push per chunk; OTR only receives docs commits. |
| SF doc contradiction 3 vs 4 seams | ACCEPT | Definitive: 4 new seam keys added (macro/phase/beat + line_composer). |
| SF IS_CHANGED moot in Phase A | ACCEPT | Explicit note in pass03. |

## Sonnet r3 (raw at ``pass03_raw/sonnet.md``)

| Sonnet claim | Status | Rationale |
|---|---|---|
| MF1 resolve_profile/StoryPromptProfile untouched but must be | ACCEPT | Chunk 1 adds 4 fields + 4 stage() calls. |
| MF2 Chunk 4 test claim misread | ACCEPT | Old Chunk 4 pack rewrite dropped. New Chunk 4 is extractor coverage tests. |
| MF3 required-seam distinction per-key | ACCEPT | pass03 documents per-key rule: new seams (not in required_seams) are safe absent; existing required seams stay populated. |
| MF4 Chunk 5 snapshot config unclear | ACCEPT | ``creative_repo_id=None`` explicit; snapshots are bare constants. |
| SF rollback story | ACCEPT | Chunk 7. |
| SF sibling branch discipline cross-repo | ACCEPT | Sibling push first, OTR test doesn't import sibling. |

## Rejected / deferred to Phase B

- Old Chunk 2 (line_grounding relaxation) -- unnecessary once line_grounding
  extraction was already deferred to Phase B.
- Old Chunk 4 (science pack rewrite to prove passthrough) -- unnecessary
  since production doesn't consume packs in Phase A. Passthrough is
  proven by extractor's None-return semantics + new Chunk 4 coverage
  tests.
- Style-picker user-template seam add (Codex SF1) -- deferred unless r4
  finds it blocks the extractor.
- Router docstring cleanup (Sonnet + Codex): follow-up doc hygiene, not
  Phase A code.

Zero r3 panel claims rejected. Every panelist's ground-truth folded.

## Panel diversity dividend r3

- **Codex** grounded the TEMPLATE_SEAMS validator ripple (3 consumers
  at :185/:232/:351 that would break under a naive split) and the
  extractor's registry-input requirement.
- **Fable** grounded the ``registry.py:169-170`` "get(seam, '').strip"
  reality that omission = empty-string for load-time semantics; caught
  ``_make_system`` closure implementation issue.
- **Sonnet** grounded the missing StoryPromptProfile fields and the
  Chunk 4 test-name misread ("test_science_profile_leaves..." tests a
  different module than the plan claimed).

Convergent finding across all three: **the extractor cannot wrap
resolve_profile() without adding plumbing that r2 didn't name**. r3
adds Chunk 1's contracts.py + profiles.py hunks explicitly.
