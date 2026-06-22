# STORY-QUALITY LIFT -- hardened CODING plan (pass02, post-R2 implementability)

Supersedes pass01 forward. R2 panel = GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude anchor/judge
(grounded against real `_otr_line_hygiene.py` + the seam map). All three panel verdicts were "no" for
ONE reason: pass01 deferred the detection primitive + the DEFECT-2 repair to R2. R2 closes them. Items
that need live wiring verification are marked "(R3)".

## 1. Invariants (R2 clarifications added)

- Ledger schema `l3-2026-05-14` FIXED. **CLARIFICATION:** this binds the LEDGER row schema only. The
  critic's in-memory report models (`StoryCriticReport` etc.) are NOT the ledger schema -- a new typed
  critic field is allowed and rides `meta.story_critic_report` as the report already does.
- ZERO workflow-JSON change -- hash `otr_scifi_16gb_full.json` before/after suite, fail on diff.
- COERCE, NEVER CRASH in the render path (`logger.warning("ROLE_COERCION ...")`); raising asserts are CI-only.
- ONE gate path per concern. Audio: three lanes (unit/no-golden; existing frozen baseline byte-identical;
  Chandra re-smoke = operator-gated recapture OR pre-TTS frozen-ledger-text compare).
- **No-bypass re-smoke is a MANUAL operator milestone, NOT an automated build gate** (it resets the
  resident :8000 server). Automated gates = unit/fixture tests only.

## 2. DEFECT 1 (TOP) -- bare stage-direction leak: concrete tiered fix

**Tier 1 GENERATION (primary):** add "write ONLY spoken words; first/second person; no third-person
physical-action narration (no stage directions)" to the line-prompt builder. Exact builder function =
**(R3)** -- candidates: the `_otr_line_composer` prompt-assembly seam + `_otr_outline._build_beat_user_prompt`
(1166-1236). No JSON change.

**Tier 2 COMPOSER REROLL** (`_otr_line_composer.compose_line` 2015-2060): add
`detect_stage_business_for_reroll(text, speaker_name="") -> Tuple[bool, str]` in `_otr_line_hygiene`
(new; alongside the existing `detect_leading_stage_business(text)->(bool,str)` and
`detect_narration_self_address(text, speaker_name)` at 148-169). It returns hit + a LOUD hint and
catches LEADING (existing) + TRAILING + EMBEDDED + UNDELIMITED (b017) via the classifier below. Call it
on the RAW draft BEFORE `strip_line_formatting`/quote-normalization (else the quote boundaries are gone).
One reroll (existing `_stage_dir_repair_attempted` guard); on failure keep draft (floor backstop).

**Tier 3 DETERMINISTIC FREEZE FLOOR** (`_otr_ledger_scrub._strip_stage_directions` 381-412):
- **Quote scanner:** STRUCTURAL DOUBLE QUOTES only (`"` + curly `U+201C/U+201D`). IGNORE single quotes /
  apostrophes (b011/b014 carry scare-quotes `'The Chronicle'`, `'alive'`, `'frequency'` and contractions).
- **Hard abort on unbalanced:** if the double-quote count is ODD, `return (text, False)` -- this routes
  b015 (orphan close-quote) to Tier-2 reroll, never the floor.
- **If balanced:** segment into in-quote spans + OUTSIDE spans; strip an OUTSIDE span ONLY when
  `is_third_person_action_clause(span)` is TRUE.
- `is_third_person_action_clause(span) -> bool`: TRUE iff (a) NO `_PRONOUN_ROOTS` token (no
  i/we/you/my/your/our, matched before an apostrophe), (b) an early token is in `_NARRATION_VERBS`,
  (c) the lead is not a `_DIALOGUE_STARTER`, (d) word-count <= a small cap. This REUSES the tested
  guards (they already kill "looks can be deceiving, John").
- **Extend `_NARRATION_VERBS` (136-144)** with the corpus verbs it currently lacks: `adjusts`,
  `clutches`, `taps`, `tightens`, `overrides`, `dances`/`dancing` (+ obvious neighbors). Keep narrow.
- **EXCLUDE the undelimited no-quote case (b017) from the floor** -- no structural anchor; reroll-only.
- **Well-formedness (mandatory):** after a strip, `.strip(" ,;-")`, then REQUIRE final char in
  `_TERMINAL_PUNCT`, non-empty, balanced quotes; if any fails, ABORT the strip (return original). Never
  emit a malformed line.
- **Ordering + idempotence:** delimited scrub -> NEW quote-anchored bare scrub -> existing leading bare
  floor; a second `_strip_stage_directions` pass must return `(same_text, False)`.

**Acceptance (resolves the b015/b017 contradiction):** the deterministic floor GUARANTEES leak=0 for the
balanced-quote class (b005/b010/b012). For the undelimited/malformed class (b015/b017) the composer
reroll is best-effort; if a leak survives reroll exhaustion -> the CI/fixture acceptance FAILS (so we
know), while the production render ships with a LOUD log (no crash). Negative fixtures REQUIRED before
any destructive strip: legit first-person action narration; quoted titles/scare-quotes; he/she/they
references to OTHERS ("They know the code"); benign lowercase clauses after punctuation.

## 3. DEFECT 3 -- b011 role mis-stamp: coercion at the RIGHT sites (Gemini circular-dep catch)

**Root (grounded):** `init_lines_from_outline` DERIVES char_id FROM role (761-766), so role=announcer
there always yields char_id="announcer". b011's char_id=c02 + role=announcer therefore comes from a
LATER mutation -- the `role_mismatch` repair (`_otr_ledger_reviewer.py:1054-1070`) setting
role=announcer while leaving char_id=c02. So DO NOT coerce at init (impossible/no-op there).

- **One helper:** `coerce_speaker_role_for_char_id(line, cast_ids, source) -> Tuple[line, changed]`:
  if `char_id in cast_ids` force `speaker_role="character"`; explicit behavior for char_id None/""/
  "announcer"/music-roles (leave true structural roles unless char_id is a cast id). LOUD on change.
- **Apply at:** (a) the `role_mismatch` repair guard -- REJECT `expected="announcer"` when the row has a
  cast char_id; (b) `set_lines` (external updates); (c) a FINAL pre-freeze consistency sweep over the
  whole ledger (it has `cast`) as the catch-all. NOT at `init_lines_from_outline`.
- **CI-only invariant assert** (allowed roles; music/sfx separate): `char_id in cast_ids => role=="character"`;
  `role=="announcer" => char_id=="announcer"`.
- **Audit (no ledger-schema change):** `meta["role_coercion"]={prev,new,source,reason}` (verify the row
  `meta` dict is mutable/serializable -- (R3)).
- **Audit ALL `speaker_role` write points** (grep nodes/; ensure each is covered or documented none) -- (R3).

## 4. DEFECT 2 -- antagonist stance: DETECTION in v1; auto-repair is an R3-gated stretch

**Detection (v1):** add a typed `StanceIssue` to the critic report (allowed -- not the ledger schema),
emitted by `run_story_critic`: fields `character_id`, `target` (constrained v1 to the episode central
object + protagonist; critic NAMES it), `prior_stance`, `new_stance`, `missing_turn_beat` (a beat/line id
OR a reason string -- do not force an id the model can't map), `line_ids`, `severity`. Add FailedDimension
value `"stance"` (verify it is a Literal with no exhaustive match -- `_otr_reroll._scope_and_hints` folds
it as a `[dim]` prefix, additive/safe). Rides `meta.story_critic_report`.

**Repair (STRETCH, gated on R3):** use the EXISTING `needs_full_rerun` episode escalation ONLY (CUT the
"outline re-intent" alternative -- you cannot re-intent without regenerating lines anyway, and a partial
rewind would add a second gate path). DETERMINISM TRAP: a seed-keyed rerun reproduces the same arc unless
its INPUT changes; and JSON is FROZEN so no new node port. So the coherence hint must ride a reserved
ledger `meta["coherence_hints"]` key that the cascade re-injects into the NEW ledger, and
`_otr_outline` must READ it and append to `_build_beat_user_prompt`. BOUND: max 1 stance full-rerun; if
it still fails, ship LOUD + telemetry (no loop).
- **R3 MAKE-OR-BREAK:** verify (a) the `needs_full_rerun` path carries a `meta` key across the reset into
  the new ledger, and (b) `_otr_outline` can read it WITHOUT a JSON change. If meta does NOT survive the
  reset, the auto-repair is NOT buildable in v1 -> SHIP DETECTION + LOUD + telemetry only; defer auto-repair.
- **Recommended v1 = DETECTION + LOUD + telemetry** (the no-bypass re-smoke surfaces the flag); auto-repair
  ships only if R3 confirms meta-survival cheaply.

## 5. DEFECT 4 -- CUT (gate AND telemetry out of acceptance)

Cut the gate (R1). R2: keep even the scope-jump telemetry OUT of acceptance (read-only at most). Re-open
only if a no-bypass frontier re-smoke still shows abrupt jumps after DEFECT 2.

## 6. Acceptance, fixture, sequencing

- **Strong-model NO-OP fixture:** create `tests/fixtures/clean_strong_ledger.json` (hand-authored, well-
  formed, no leaks/no stance reversal) + counters (composer reroll attempts, deterministic strips, role
  coercions, full reruns). Asserts: on the clean fixture, ZERO strips/rerolls/coercions; and
  `"coherence_hints" not in ledger.meta`; no row carries `CODE_STAGE_DIRECTION`.
- Per chunk: full suite + Bug Bible green; JSON hash unchanged; audio per the three lanes; a caught defect
  is repaired/rerolled and ABSENT from the final frozen ledger OR fails CI (explicit reroll-exhaustion
  behavior); no-op proven on the clean fixture; `story_quality_scan.py` imports the SAME detector helpers
  (no engine/scan drift).
- Order: 0. manual no-bypass BASELINE re-smoke (operator) -> 1. DEFECT 1 (tiers) -> 2. DEFECT 3 (coercion)
  -> 3. DEFECT 2 (detection; repair stretch per R3) -> 4. DEFECT 4 cut.

## 7. R3 remit (wiring / integration)

1. **DEFECT 2 make-or-break:** does `needs_full_rerun` carry `meta["coherence_hints"]` across the reset,
   and can `_otr_outline` read it with NO JSON change? (Decides auto-repair vs detection-only v1.)
2. Exact Tier-1 prompt-builder function (line composer prompt assembly vs `_build_beat_user_prompt`).
3. Audit ALL `speaker_role` write points in nodes/.
4. Confirm `FailedDimension` is a Literal (adding "stance" non-breaking); confirm the critic-report parser
   accepts the new `StanceIssue`.
5. Confirm the ledger row `meta` dict is mutable/serializable (for the coercion audit) + the JSON no-drift hash target.
