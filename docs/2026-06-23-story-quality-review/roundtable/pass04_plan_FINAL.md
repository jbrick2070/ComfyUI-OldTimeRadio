# Story-Quality LIFT (post-R3) -- FINAL build-ready plan (R4 converged)

4-round live roundtable: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Grok-4.3, Claude code-grounded judge.
Spend: A $0.111 + R1 $0.056 + R2 $0.095 + R3 $0.091 + R4 $0.081 = **~$0.434**. Artifacts: `pass00..pass04` +
`pass0N_judgment.md` + `GROUNDING_*.md`. R4 verdicts: all 3 responders "yes-with-fixes", fixes small/specific ->
CONVERGED (no new architecture-level must-fix; lever set stable since pass02).

## What this fixes (grounded)
The episodes are dramatically identical (the "console standoff": every premise -> people fighting over a
lever/key/console with a countdown), the climax happens off-stage (announcer fiat), and characters speak their
own stage directions. The pipeline ALREADY instructs against all of this in soft prose and the weak local model
ignores it (`_otr_outline._build_beat_user_prompt` "ACTION UNDER PRESSURE / RAISE THE STAKE / KEEP STANCE
CONSISTENT"; composer cast cards + dramatic frame + speech_signature). Therefore the only effective fixes are
DETERMINISTIC + UPSTREAM (Python that builds + fills the beat skeleton) -- NOT another QA reroll gate (unanimous
panel + operator).

## Hard constraints
NO flag-and-reroll gate (or disguised one). Weak-local-robust (Python-filled fields + deterministic negative
constraints, never assume schema obedience). Content-only; frozen ledger schema l3-2026-05-14 + workflow JSON
UNCHANGED. Audio spine frozen (`test_audio_byte_identical`); audio-affecting work flag-gated default-off +
deliberate golden re-baseline. Deterministic/seed-keyed; UTF-8 no BOM; SFW. Flag OFF => byte-identical
(no field population, no `meta.story_quality` key).

## Data placement (R4-FINAL -- avoids the Pydantic trap)
`Beat` is strict Pydantic with NO `meta` field (verified `_otr_outline.py:84-135`). DO NOT add defaulted Beat
fields (model_dump() emits them -> JSON drift; even `exclude=True` can leak into the LLM outline schema -- GPT).
**Hold the deterministic per-beat SQ data (`beat_role`, `conflict_object`, `conflict_type`, `personal_cost`,
`sensory_consequence`, `state_change`) in a WRITER-SIDE `dict[beat_id -> sq]`** built AFTER outline parse and
consumed when constructing `LineRequest`. Zero Beat-schema change. `LineRequest` (dataclass) gets optional
`beat_role/conflict_object/conflict_type=""` (dataclass defaults are safe), rendered in DRAMATIC FRAME only when
non-empty AND `OTR_STORY_QUALITY_L12` on. Telemetry rides `meta.story_quality` + `compose_flags` (existing
free-form).

## Flags (all default-OFF)
`OTR_STORY_QUALITY_L12` (L1+L2; read in `_otr_outline.py` selectors/validator/mutation AND
`_otr_line_composer.py` render), `OTR_COMPOSER_ACTION_STRIP` (L3), `OTR_TRANSCRIPT_SANITIZER` (L4).
Telemetry gate: `meta.story_quality` is written iff ANY SQ flag is on; NO key when all off.

---

## THE LEVERS (final)

### L5a -- make measurement trustworthy + stop hiding the best writer (do FIRST; no audio effect)
- **Scale the edit cap** (`compute_edit_cap`, `_otr_ledger_reviewer.py:1136`, currently `min(8,max(3,
  voiced_beats//3))` = 6 for 18 beats). Dense, well-written gemma prose trips >6 doctor edits -> `too_many_edits`
  -> the cascade halts + rolls back BEFORE the critic (`_otr_freeze_cascade.py:599/730`) -> never arc-graded
  ("?"). PIN one formula: `compute_edit_cap = max(3, min(12, ceil(voiced_beats*0.6)))` (6 beats->4, 18->11,
  19->12) with explicit test values. **DROP the "advisory critic before terminal stop" idea** -- `run_story_critic`
  needs `generate_fn` which the reviewer module does not have (Gemini/GPT/Grok, grounded `:756`). Instead, make
  the downstream consumer TOLERATE a missing `meta.story_critic_report` on terminal verdicts.
- **Fix the telemetry undercount** (EP16: rows carried `objective_literal_retry` yet `l1_rerolls=0`). Two parts:
  (a) scrub `_meta.setdefault("story_quality",{}).update({...})` instead of the blind `=` overwrite
  (`_otr_ledger_scrub.py:1006`); (b) aggregate from the FINAL persisted rows (the cascade restore runs before
  scrub -> count the saved ledger, not discarded rows). Telemetry/grading only; never the frozen schema.

### L1 + L2 -- the STRUCTURAL CORE (ship together; scaffolding-first, then flag-on)
Neither alone works: L1 alone = new words, same standoff; L2 alone = same threat-noise in relabelled slots.

**L1 -- premise-anchored conflict, deterministic (no retry).**
- `select_domain(meta, premise) -> str`: ordered keyword map (casefold+NFC, ordered meta-field inspection),
  default `"general"`. No LLM.
- Curated UTF-8 table `domain -> {conflict_objects[], conflict_types[]}`; `"general"` = a generic
  institutional-power palette so no premise is unserved.
- Python assigns each voiced beat a `conflict_object` + `conflict_type` (seed-keyed:
  `sha256(f"{seed}:{beat_index}:{domain}:object")` and `...:type` -- distinct labels; sorted-candidate modulo;
  NOT `hash()`).
- L1a: route `allowed_roster` into the composer's split `allowed_people`/`allowed_things` at the writer call
  site so scenes name the REAL program/agency/place (Chandra, El Nino), filtering "ANNOUNCER"/"NARRATOR" from
  render (phantom gate still gets the union). VERIFY the split fields are populated today.
- Crisis-noun repair: allowed palette = roster + a deterministic noun-token filter over title/premise/logline
  (`[A-Za-z][A-Za-z'-]{2,}` minus stopwords/all-caps -- GPT v0 extractor). Repair ALL ungrounded crisis nouns
  (Gemini: drop the arbitrary cap) by deterministic whole-token substitution from the beat's palette object,
  ONLY in `beat.intent` (mutable allowlist; never speaker/ids/arc_phase/target_words/sfx_cue/roster/title).

**L2 -- phase = dramatic FUNCTION (separate `beat_role` + a new validator).**
- `beat_role in {setup, pressure, personal_stake, irreversible_choice, consequence}` (the climax IS
  `irreversible_choice` = last voiced beat; no separate "climax" token). Held in the writer-side dict, NOT on
  `Beat`, NOT on `arc_phases`.
- Outline build sequence (flag ON): parse outline -> assign roles+conflict slots -> fallback beat factory for
  missing required content (SAME-BEAT: mutate only `intent` (+`mood` if invalid) + the writer-side sq dict;
  preserve beat_id/speaker/speaker_role/arc_phase/target_words/count/order) -> existing budget/arc validators
  UNCHANGED -> NEW `beat_role` validator LAST, preserving the first-failure contract (exactly one
  personal_stake before the first irreversible_choice; exactly one irreversible_choice last).
- `personal_cost` content: NO structured cost/fear field exists -> a deterministic `(speaker, domain)` fallback
  table is the PRIMARY source (Gemini), written to the writer-side dict. Climax beat carries
  `sensory_consequence` + `state_change` (field-presence, not prose regex).
- Carry `beat_role` + `conflict_object` into the composer via the new `LineRequest` fields (update the call
  site that builds LineRequest -- adding fields alone leaves them empty).
- `choice_summary` / seed-keyed outro template family: CUT for v0 (the on-stage climax beat is the fix).

### L3 -- strip narrated-action (flag `OTR_COMPOSER_ACTION_STRIP`; audio-affecting)
Composer marks non-spoken action with an explicit `ACTION:` marker; deterministic regex strips only that
segment, inserted right after compose/polish returns text and before line persistence. Add a `compose_flags`
marker `action_strip:regex` (separate counter; do NOT reuse `l7_splits`). Do not persist `internal_action`.
Sequence after L1/L2. Claim: "strips what the model marks; L4 catches the rest."

### L4 -- minimal transcript sanitizer (flag `OTR_TRANSCRIPT_SANITIZER`; audio-affecting)
Final line TEXT only, after text is final + before freeze/TTS/hash/golden; never speaker-label/identity fields.
Strip prompt-leak ("voice should", director-note patterns) + conservative quote-wrapper balancing (no apostrophe/
measurement damage). Mojibake = verify/no-mutation test ONLY (build artifact; no encoding repair in v0).

### DEFERRED (operator decides)
- **L5b** gemma-12b default -- gate on L5a's cap-fix + a controlled 5-brief bake-off + a side-effect re-soak.
- **L6** best-of-N -- CUT from v0 (it is a select-gate, costs N generations, can't fix structural sameness);
  on record because the operator asked; revisit after L1/L2 establish a new baseline.

---

## Acceptance metric (the goal is a measurably less-samey story)
- `meta.story_quality.ungrounded_crisis = {matches, total}` per episode; `ungrounded_crisis_density =
  matches/total_voiced_words` -> large drop vs the R3 soak baseline.
- Cross-episode distinct `conflict_object`/`conflict_type` counts across a soak (the sameness measure).
- Required-role presence; PREREQUISITE = L5a (trustworthy telemetry + the best writer actually graded).
- Compatibility test: unknown `meta.story_quality` keys + new `compose_flags` ignored by freeze/TTS/serialize/hash.
- Guard: do NOT reintroduce longer-but-monotone (the 430w standoff).

## Build order
1. L5a (cap formula + telemetry merge/source + missing-report tolerance).
2. L1/L2 scaffolding (writer-side sq dict + `select_domain` + palette table + selectors + fallback factory + new
   role validator + LineRequest fields + call-site threading + tests), flag OFF => no-drift JSON + byte-identical
   prompt asserts.
3. L1/L2 render ON (`OTR_STORY_QUALITY_L12`) -> re-soak small matrix -> measure sameness.
4. L3 then L4 (golden re-baseline each, operator-gated GPU render).
5. Evaluate L5b bake-off; (operator) decide L6.
Each chunk: full suite + Bug Bible green, no-drift JSON assert, commit + push to v2.0-alpha.

## Verify-at-build checklist (R4)
1. `Beat` is the only beat row class (no `OutlineBeat`) -- grounding shows only `Beat`. CONFIRMED here.
2. Outline->ledger serialization: confirm the writer-side sq dict (not Beat) reaches the composer and that NO new
   key lands in the frozen ledger except gated `meta.story_quality`/`compose_flags`. No-drift assert, flag ON+OFF.
3. `allowed_people`/`allowed_things` are populated at the writer call site today (L1a).
4. `compute_edit_cap` change yields the pinned values (6->4, 18->11, 19->12); terminal-verdict rate drops on the
   dense-gemma episodes; downstream tolerates a missing `story_critic_report`.
5. Scrub aggregation counts the SAVED rows (test: `objective_literal_retry` before a rollback path -> counts
   match saved, not discarded).
6. Define + test the telemetry flag contract (no `meta.story_quality` key when all SQ flags off).
