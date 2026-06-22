# STORY-QUALITY LIFT -- build-ready plan (pass03, post-R3 wiring)

Supersedes pass02 forward. R3 panel = GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude anchor/judge,
grounded against the verified wiring facts (grounding_pack_r3.md W1-W5). The unbuildable auto-repair is
removed; the channel/site corrections are folded. This is the coder kickoff (planner window does not
write production code). A few small ordering checks are marked "(verify-at-build)".

## 1. Invariants (unchanged from pass02 sec 1) + corrections
- Ledger ROW schema `l3-2026-05-14` FIXED; the critic REPORT models are not the ledger schema (a typed
  `StanceIssue` is allowed, rides `meta.story_critic_report`).
- ZERO workflow-JSON change (hash before/after, fail on diff). COERCE-NEVER-CRASH in render; CI-only
  asserts gated on `OTR_TEST_MODE` (the conftest sets it). ONE gate path per concern. Three audio lanes.
- **There is NO per-line `meta` dict** (W5). Per-line breadcrumbs ride `compose_flags` (list of
  "kind:detail" strings); aggregates ride episode-level `meta`.

## 2. DEFECT 1 (TOP) -- bare stage-direction leak

**Tier 1 GENERATION (strengthen, not new):** the "spoken words only, never narrate your own actions in
third person" rider ALREADY EXISTS at `_otr_line_composer.py::_build_user_prompt` 1307-1315 and the
corpus still leaks. STRENGTHEN it there (more aggressive/structural wording). Apply ONLY to
`_build_user_prompt` (line text); do NOT add spoken-line hygiene to `_otr_outline._build_beat_user_prompt`
(it writes beat INTENT, not dialogue -- interface mismatch).

**Tier 2 COMPOSER REROLL -- corrected site:** add `detect_stage_business_for_reroll(text, speaker_name="")
-> Tuple[bool, str, str]` (hit, hint, reason_code) in `_otr_line_hygiene`. Call it inside
`compose_line_draft` (1689-1928) IMMEDIATELY AFTER the LLM draft is received and BEFORE
`strip_line_formatting`/quote-normalization (the `compose_line` 2015-2060 site is too late -- the raw
draft + quote boundaries are already gone). Keep the existing one-reroll guard. reason_code in
{leading, trailing_after_quote, embedded_between_quotes, undelimited_action_clause} for fixture
diagnostics. This is the ONLY tier that can reroll (it runs at compose time); it handles the
malformed/undelimited cases (b015, b017).

**Tier 3 DETERMINISTIC FREEZE FLOOR** (`_otr_ledger_scrub._strip_stage_directions` 381-412): deterministic
last-resort strip for the HIGH-CONFIDENCE balanced-quote class only.
- SHARED double-quote segmentation helper (used by BOTH Tier 2 detection and Tier 3 floor -- single
  source, no parse drift): structural double quotes only (`"`+curly U+201C/U+201D); IGNORE single
  quotes/apostrophes (scare-quotes 'The Chronicle').
- If double-quote count is ODD -> `return (text, False)` (leave unscrubbed). **The floor CANNOT route
  back to reroll** (it runs after the composer loop, downstream in the DAG). An odd-quote line that
  reaches the floor stays as the final frozen line -> CI/fixture FAILS (so we know); production ships
  LOUD. (b015 should already have been caught by Tier 2 at compose time; the floor is not its safety net.)
- If balanced: strip an OUTSIDE-quote span only when `is_third_person_action_clause(span)` TRUE: no
  `_PRONOUN_ROOTS` token; an early token in `_NARRATION_VERBS`; lead not a `_DIALOGUE_STARTER`; word-count
  <= a small (configurable) cap. **Extend `_NARRATION_VERBS` (136-144)** with adjusts/clutches/taps/
  tightens/overrides/dances/dancing (+ obvious neighbors). EXCLUDE undelimited no-quote lines (b017) from
  the floor (Tier 2 only).
- Well-formedness (mandatory): after strip, `.strip(" ,;-")`, require final char in `_TERMINAL_PUNCT`,
  non-empty, balanced quotes; else ABORT the strip (return original). Per-line breadcrumb ->
  `compose_flags` ("stage_dir_stripped:<reason>"); episode finding stays the existing ScrubResult/
  CODE_STAGE_DIRECTION.
- Ordering inside `_strip_stage_directions`: existing delimited scrub -> NEW quote-anchored bare scrub ->
  existing leading bare floor; idempotent (2nd pass returns (same, False)). Test interaction with the
  existing unanchored delimited scrub (parentheticals inside quoted dialogue stay).

**Acceptance:** floor GUARANTEES leak=0 for the balanced-quote class (b005/b010/b012); Tier 2 best-efforts
the malformed/undelimited (b015/b017); a leak surviving Tier-2 exhaustion -> CI FAIL, production LOUD.
Negative fixtures REQUIRED: legit first-person action; quoted titles/scare-quotes; he/she/they about
OTHERS; benign lowercase-after-punctuation; an ANNOUNCER line (scrub gates on is_spoken_role
character+announcer). `story_quality_scan.py` imports the SAME `_otr_line_hygiene` helpers (no drift).

## 3. DEFECT 3 -- b011 role mis-stamp: coercion, corrected channels + sites

- Helper `coerce_speaker_role_for_char_id(line, cast_ids, source) -> Tuple[line, changed]`: if
  `char_id in cast_ids` force `speaker_role="character"`. `cast_ids` EXCLUDES the "announcer" sentinel +
  music/sfx roles (so it does NOT fight the legitimate `cast_lock.py:473` announcer re-stamp where
  char_id IS the announcer). Explicit behavior for char_id None/""/"announcer"/music.
- Apply at: (a) `_otr_ledger_reviewer.py:1063` role_mismatch repair -- REJECT `expected="announcer"` when
  the row has a cast char_id (the b011 culprit); (b) `production_ledger.set_lines`; (c) a FINAL pre-freeze
  consistency SWEEP placed inside the freeze cascade AFTER all line-role mutations (incl cast_lock-class)
  and BEFORE the freeze/hash + role-dependent routing (TTS/scrub). NOT at `init_lines_from_outline`
  (char_id is derived FROM role there -- coercion is a no-op). (verify-at-build: confirm the sweep sits
  after cast_lock in the actual phase order; cast_lock's announcer re-stamp is legitimate + char_id=="announcer".)
- Audit: per-line breadcrumb in `compose_flags` ("role_coerce:prev=announcer,new=character,reason=cast_char_id");
  episode aggregate in `meta["role_coercions"]` (count + line_ids). NO per-line meta dict.
- CI-only invariant assert (gated on `OTR_TEST_MODE`; music/sfx separate): `char_id in cast_ids =>
  role=="character"`; `role=="announcer" => char_id=="announcer"`.

## 4. DEFECT 2 -- antagonist stance: GENERATION lever + DETECTION (auto-repair CUT)

Auto-repair via `needs_full_rerun` is CUT (W1: no cross-run channel survives; the verdict is a terminal
string the upstream writer never reads; `new_ledger()` wipes meta; `regeneration_hint` is read by nobody;
JSON frozen forbids a new edge). Removed: the coherence_hints mechanism, the determinism-trap paragraph,
the max-1-rerun bound, and the `"coherence_hints" not in meta` assert.

- **Generation lever (the ROOT fix):** strengthen `_otr_outline._build_beat_user_prompt` (1166-1236) to
  require the antagonist's stance toward the protagonist/central object be CONSISTENT across beats -- a
  reversal needs an explicit turn beat. Beat-intent-level, JSON-free, no cross-run state. (This is where
  the arc coherence is actually decided.)
- **Detection (backstop/telemetry):** add a typed `StanceIssue` to the critic report
  (character_id, target [pass cast/protagonist/central-object context into the critic prompt, or relax
  target to a free-form string validated in tests], prior_stance, new_stance, missing_turn_beat [id OR
  reason string], line_ids, severity); add FailedDimension value `"stance"` AND update the critic
  system-prompt prose (`_otr_story_critic.py:310-329`) so the model emits it, AND tests proving
  `meta.story_critic_report` round-trips it (verify the report model is not strict-reject on the new
  field). LOUD log; telemetry only -- no reroll/gate in v1.

## 5. DEFECT 4 -- CUT (gate + telemetry out of acceptance).

## 6. Acceptance + fixture + sequencing
- `tests/fixtures/clean_strong_ledger.json` (hand-authored clean) + counters as in-memory harness
  returns + `compose_flags` per-line + `meta` aggregates (NOT log scraping). Asserts: clean fixture ->
  ZERO strips/rerolls/coercions; no row carries CODE_STAGE_DIRECTION; the 5 corpus lines reach their
  expected tier outcome (b005/b010/b012 floor-stripped well-formed; b015/b017 Tier-2 rerolled or CI-fail).
- Per chunk: full suite + Bug Bible green; JSON hash unchanged; audio per the three lanes; caught ->
  repaired/stripped -> ABSENT from final frozen ledger OR CI-fail; no-op on the clean fixture.
- Order: 0. manual no-bypass BASELINE re-smoke (operator) -> 1. DEFECT 1 (Tier1 strengthen + Tier2-in-draft
  + Tier3 floor, shared quote helper) -> 2. DEFECT 3 (coercion + sweep) -> 3. DEFECT 2 (beat-prompt stance
  lever + critic stance axis + telemetry) -> 4. DEFECT 4 cut.

## 7. Verify-at-build (small, non-blocking)
1. Exact pre-freeze sweep insertion phase vs cast_lock ordering in the cascade.
2. `compose_flags` has no strict format validation downstream (DeepSeek ASSUMPTION).
3. The critic report model accepts the new `StanceIssue` (lenient vs strict pydantic).
4. `OTR_TEST_MODE` is the right gate for the CI-only invariant assert (conftest convention).
