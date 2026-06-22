# STORY-QUALITY LIFT -- FINAL build-ready plan (pass04, CONVERGED)

R1-R4 roundtable converged (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro panel + Claude code-grounded
anchor/judge). R4: all three panel verdicts "yes-with-fixes" -> converged; the R4 fixes are spec-locks
(below), no architecture change. This is the CODER KICKOFF (planner window does not write production
code). Schema `l3-2026-05-14` FIXED; branch v2.0-alpha. Spend across R1-R4 reported in pass04_judgment.

## 0. Goal
Lift the weak-end story ("Chandra's Echo" was C+). Every gate is one a strong/opus script already passes
(no-op on good input). Three grounded defects + one explicitly out-of-scope. ZERO workflow-JSON change.

## 1. Invariants
Ledger ROW schema FIXED (new per-line signals ride `compose_flags`; aggregates ride episode `meta`; the
critic REPORT models are not the ledger schema). ZERO JSON change (hash before/after). COERCE-NEVER-CRASH
in render; CI-only asserts gated on `OTR_TEST_MODE`. Audio: three lanes (unit/no-golden; existing frozen
baseline byte-identical unless in-scope; Chandra re-smoke = operator-gated recapture OR pre-TTS text
compare). Deterministic, LOUD fallbacks, UTF-8 no BOM, SFW.

## 2. DEFECT 1 (TOP) -- bare stage-direction leak

**Shared quote helper (single source -- Tier 2 + Tier 3 use it identically, no parse drift):**
`segment_double_quotes(text) -> (normalized_text, segments)` in `_otr_line_hygiene`. NORMALIZE curly
U+201C/U+201D to straight `"` BEFORE counting; count normalized `"`; IGNORE single quotes/apostrophes
(scare-quotes 'The Chronicle', contractions). Return the NORMALIZED text + the in-quote / outside spans so
both tiers receive identical input. Odd `"` count => caller treats as unbalanced.

**`is_third_person_action_clause(span) -> bool`:** TRUE iff (a) NO FIRST/SECOND-person pronoun -- the
existing `_PRONOUN_ROOTS` set is 1st/2nd-person ONLY (i/we/you/me/us/my/your/our); THIRD-person
(he/she/his/her/they) is PERMITTED (it is the subject of a stage direction); (b) an early token in the
extended `_NARRATION_VERBS`; (c) lead not a `_DIALOGUE_STARTER`; (d) word-count <= a configurable cap.
Locked unit assertions: `clutches her wedding ring tightly` => TRUE (b010); `taps his cane impatiently`
=> TRUE (b012); a first-person spoken action ("I adjust the dial as I speak") => FALSE.
**Extend `_NARRATION_VERBS` (136-144) with an EXPLICIT closed list:** `adjusts, clutches, taps, tightens,
overrides, dances, dancing` (turns/looks/smiles/sighs already present). No "obvious neighbors".

**Tier 1 GENERATION (strengthen existing):** make the EXISTING rider at `_otr_line_composer._build_user_prompt`
1307-1315 more aggressive (it already says "first person, never narrate your own actions in third person"
and still leaks). Line-prompt ONLY; do NOT touch `_build_beat_user_prompt` (beat intent != dialogue).
Keep the wording change minimal (sample on a frontier model for dialogue-quality regression).

**Tier 2 COMPOSER REROLL (locked control flow):** add `detect_stage_business_for_reroll(text,
speaker_name="") -> (hit: bool, hint: str, reason_code: str)` in `_otr_line_hygiene` (reason_code in
{leading, trailing_after_quote, embedded_between_quotes, undelimited_action_clause}). CALL it inside
`compose_line_draft` (1689-1928) right after the LLM draft, BEFORE `strip_line_formatting`/normalization.
DISABLE the old `compose_line` 2015-2060 stage-business reroll block (or make it delegate to the
draft-level detector) so there is exactly ONE guard; thread a single `_stage_dir_repair_attempted` into
`compose_line_draft`; APPEND `hint` to the existing reroll-hint concatenation in the `_BARE_STAGE_HINT`
format. TEST: one malformed line gets AT MOST one stage-business reroll total. Tier 2 is the ONLY tier
that can reroll (it runs at compose time); it owns the malformed/undelimited cases (b015, b017).

**Tier 3 DETERMINISTIC FREEZE FLOOR** (`_otr_ledger_scrub._strip_stage_directions` 381-412): order =
existing delimited scrub -> NEW quote-anchored bare scrub -> existing leading bare floor; idempotent.
Using the shared helper: if odd `"` count -> `return (text, False)` (leave unscrubbed; the floor CANNOT
route back to reroll -- it is downstream of the composer loop; an odd-quote line that reaches the floor
becomes the final frozen line -> CI-fail / production ships LOUD). If balanced -> strip an OUTSIDE-quote
span iff `is_third_person_action_clause`. EXCLUDE undelimited no-quote lines (b017). Well-formedness after
strip: `.strip(" ,;-")`, then the last SPOKEN char (ignoring an optional trailing closing structural
double-quote) must be in `_TERMINAL_PUNCT`, non-empty, balanced quotes; else ABORT the strip (return
original). So b005 -> `"Not before I amplify it. The world deserves to hear this."` (closing `"` is fine).
Per-line breadcrumb -> `compose_flags` ("stage_dir_stripped:<reason>"); episode finding = existing
CODE_STAGE_DIRECTION/ScrubResult.

**DEFECT-1 byte-identical golden GATE:** run the new floor over the EXISTING golden fixture ledger and
assert ZERO strips; a non-zero strip is the operator-gated recapture trigger, surfaced LOUD (not
discovered by a red byte-identical test).

**Acceptance:** floor GUARANTEES leak=0 for the balanced-quote class (b005/b010/b012); Tier 2 best-efforts
b015/b017; a leak surviving Tier-2 exhaustion -> CI FAIL + an INTENTIONAL v1 production limitation (ship
LOUD + a freeze-warning counter in final metadata; do not imply leak=0 beyond the balanced-quote class).
Negative fixtures REQUIRED: legit first-person action; quoted titles/scare-quotes; he/she/they about
OTHERS; benign lowercase-after-punctuation; an ANNOUNCER line. `story_quality_scan.py` imports the SAME
`_otr_line_hygiene` helpers.

## 3. DEFECT 3 -- b011 role mis-stamp: coercion (channels + sites locked)
Helper `coerce_speaker_role_for_char_id(line, cast_ids, source) -> (line, changed)`: if `char_id in
cast_ids` force `speaker_role="character"`. `cast_ids = set(ledger.get("cast",{}).keys())` MINUS the
"announcer"/music/sfx sentinels (so it never fights the legitimate `cast_lock.py:473` announcer re-stamp
where char_id IS the announcer). Explicit behavior for char_id None/""/"announcer"/music.
Apply at: (a) `_otr_ledger_reviewer.py:1063` role_mismatch repair -- REJECT `expected="announcer"` when the
row has a cast char_id (the b011 culprit); (b) `production_ledger.set_lines` (cast_ids from the ledger
cast table at the call site; if unavailable, no-op and rely on the sweep); (c) **MANDATORY pre-freeze
SWEEP = the final step of `OTR_LedgerFreezeCascade`'s mutation phase, immediately after the cast_lock
call returns and BEFORE the freeze hash + role-dependent routing (scrub/TTS).** NOT at
`init_lines_from_outline` (char_id derived FROM role there -> no-op). Audit: per-line `compose_flags`
("role_coerce:prev=announcer,new=character,reason=cast_char_id") + episode `meta["role_coercions"]`
(count + line_ids). CI-only assert (gated on `OTR_TEST_MODE`; music/sfx separate): `char_id in cast_ids =>
role=="character"`; `role=="announcer" => char_id=="announcer"`.

## 4. DEFECT 2 -- antagonist stance: GENERATION lever + DETECTION telemetry (auto-repair CUT)
Auto-repair via `needs_full_rerun` CUT (no surviving cross-run channel; JSON frozen).
- **Generation lever (root fix):** strengthen `_otr_outline._build_beat_user_prompt` (1166-1236) to require
  the antagonist's stance toward the protagonist/central object be CONSISTENT across beats (a reversal
  needs an explicit turn beat), REFERENCING the pinned antagonist want `DramaticState.character_b_wants`
  (the 1-beat adjacency window alone cannot enforce arc-wide consistency). Best-effort nudge.
- **Detection (telemetry ONLY):** add a typed `StanceIssue` to the critic report -- `character_id: str`,
  `target: str` (FREE-FORM; do NOT wire new cast context into the critic prompt), `prior_stance: str`,
  `new_stance: str`, `missing_turn_beat: str` (id OR reason, single str), `line_ids: list[str]`,
  `severity: str`. Rides `meta.story_critic_report`. **Do NOT add "stance" to `FailedDimension` in v1** and
  do NOT convert `StanceIssue` into a `RerollTarget`/freeze gate/`needs_full_rerun` -- that would silently
  reintroduce a repair path. TEST: a stance-only critic report leaves reroll targets + freeze verdict
  unchanged. (Update the critic system-prompt prose 253-335 so the model emits StanceIssue.)

## 5. DEFECT 4 -- abrupt escalation: OUT OF SCOPE for this build
No gate, no telemetry in acceptance (R1 unanimous cut; symptom of DEFECT 2 + weak model). Optional: a
prompt-only rider in `_build_beat_user_prompt`, no gate, no test. Re-open only if a no-bypass frontier
re-smoke still shows abrupt jumps after DEFECT 2.

## 6. Acceptance + fixture + sequencing
`tests/fixtures/clean_strong_ledger.json` (hand-authored clean) + counters returned by the test harness +
`compose_flags` per-line + `meta` aggregates (NOT log scraping). Clean-fixture asserts: ZERO
strips/rerolls/coercions; no row carries CODE_STAGE_DIRECTION. Corpus asserts: b005/b010/b012 floor-stripped
+ well-formed; b015/b017 Tier-2 rerolled or CI-fail. Per chunk: full suite + Bug Bible green; JSON hash
unchanged; audio per the three lanes. Order: 0. manual no-bypass BASELINE re-smoke (operator) -> 1. DEFECT 1
(shared quote helper + Tier1 strengthen + Tier2-in-draft + Tier3 floor + golden no-op gate) -> 2. DEFECT 3
(coercion + mandatory sweep) -> 3. DEFECT 2 (beat-prompt stance lever + StanceIssue telemetry) -> 4. DEFECT 4 skip.

## 7. VERIFY-AT-BUILD checklist (concrete steps)
1. Open `OTR_LedgerFreezeCascade.py`; find the cast_lock call; place the role sweep immediately after it,
   before the freeze hash; record the line number. Test ledger `char_id=c02, speaker_role=announcer` ->
   frozen row `speaker_role=character`; row `char_id=="announcer"` stays announcer (not coerced).
2. Grep for any strict format validation of `compose_flags` entries; confirm `stage_dir_stripped:*` /
   `role_coerce:*` pass freeze/TTS prep without error.
3. Construct a `StoryCriticReport` with one `StanceIssue`; assert `model_dump`/`model_validate` round-trip
   (if the model is `extra='forbid'`, add `StanceIssue` to the field defs).
4. Confirm `OTR_TEST_MODE` (conftest) is set before the CI-only invariant asserts run.
5. Confirm `segment_double_quotes` straight+curly normalize+count identically; `'The Chronicle'` / `'alive'`
   negatives unaffected; `_strip_stage_directions` idempotent on b005/b010/b012 outputs.
6. Confirm `strip_line_formatting` (scrub step 3a, runs before `_strip_stage_directions` 3b) does not
   destroy the double quotes the floor segments on -- run the quote-anchored bare scrub on text BEFORE any
   quote-mutating normalization, or confirm 3a preserves `"`.
7. Stance telemetry does not trigger reroll/gate (test from sec 4).

## 8. The four defects -> grounded summary (for the coder)
1. LEAK (b005/b010/b012 trailing, b015 embedded-malformed, b017 undelimited): `_otr_line_hygiene`
   detector + shared quote helper + extended `_NARRATION_VERBS`; Tier2 in `compose_line_draft`; Tier3 floor
   in `_otr_ledger_scrub._strip_stage_directions`.
2. STANCE (c02 Manfred reverses): `_build_beat_user_prompt` lever + critic `StanceIssue` telemetry.
3. ROLE (b011 char_id=c02 + role=announcer): coercion helper at the repair guard + set_lines + mandatory
   pre-freeze sweep.
4. ESCALATION (UN jump): out of scope.
