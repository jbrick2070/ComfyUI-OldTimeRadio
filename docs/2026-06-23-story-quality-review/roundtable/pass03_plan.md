# Story-Quality LIFT (post-R3) -- Build Plan v3 (after R3: wiring / integration / sequencing)

Grounded against real shapes (`GROUNDING_R3.md`). R3 panel: GPT-5.5 + Gemini-3.1-pro + Grok-4.3 (DeepSeek
returned empty/length) (`pass03/`, ~$0.091). R3 caught build-breaking wiring bugs; all folded. The LEVER SET is
unchanged from pass02; this version pins the WIRING so it is buildable.

## R3 wiring corrections (the build-breakers R3 caught)
1. **L5a critic-grading is impossible as framed.** `too_many_edits` is a TERMINAL reviewer verdict that halts
   the cascade and ROLLS BACK the ledger BEFORE `run_story_critic` runs (`_otr_freeze_cascade.py:730-766`; stop
   at :599). You cannot "grade after." -> two-part fix (below).
2. **Pydantic leak.** Adding `conflict_object: str = ""` to `Beat` (Pydantic, `_otr_outline.py:84`) is NOT
   serialization-free -- `model_dump()` emits defaults and would drift the outline JSON / frozen ledger.
3. **Telemetry blind-overwrite.** `_otr_ledger_scrub.py:1006` does `_meta["story_quality"] = {...}` (replaces).
4. **Flag must gate the MUTATION, not just the prompt** -- if selectors/fallbacks populate Beat/meta with the
   flag OFF, JSON drifts even though prompts don't (GPT #10).
5. **Adding fields != threading them.** The call site that builds `LineRequest` from `Beat` must pass the new
   fields, and the new role validator must be sequenced correctly.

---

## Data model (FINAL placement -- R3-corrected)
- New per-beat story-quality values (`conflict_object`, `conflict_type`, `beat_role`, `personal_cost`,
  `sensory_consequence`, `state_change`, optional `choice_summary`) ride **`beat.meta`** (free-form), NOT new
  top-level `Beat` Pydantic fields. If a top-level field is unavoidable, it MUST be `Field(default="",
  exclude=True)` so `model_dump()` never emits it (Gemini #2). Either way: flag OFF => nothing populated =>
  byte-identical outline JSON (verify with the no-drift assert).
- `LineRequest` (dataclass, `_otr_line_composer.py:581`): add optional `beat_role=""`, `conflict_object=""`,
  `conflict_type=""` (dataclass defaults are safe -- not Pydantic). Render in DRAMATIC FRAME only when non-empty
  AND the L12 flag is on. The **writer/orchestrator call site that constructs `LineRequest` from the beat MUST be
  updated to pass these** (GPT #5, Grok #2) -- adding the fields alone leaves them empty.
- `EpisodeBudget.arc_phases`: UNCHANGED (do not overload -- R2).
- Flags: `OTR_STORY_QUALITY_L12` (read in BOTH `_otr_outline.py` selectors+validator+mutation AND
  `_otr_line_composer.py` render -- Grok SHOULD #2), `OTR_COMPOSER_ACTION_STRIP`, `OTR_TRANSCRIPT_SANITIZER`;
  all default-OFF. Define the scrub flag contract: aggregate `meta.story_quality` if ANY SQ flag is on; write NO
  key when all off (GPT #12).

## Outline build sequence (GPT #3 -- the explicit order; preserves existing validators)
1. generate/parse outline (existing).
2. **(flag ON only)** deterministically assign `beat_role` + `conflict_object`/`conflict_type` to voiced beats
   (seed-keyed: `sha256(f"{episode_seed}:{beat_index}:{domain}:object")` vs `...:type` -- distinct labels, GPT
   #5; casefold+NFC-normalize inputs, ordered meta-field inspection -- GPT #4).
3. **(flag ON only)** run the fallback beat factory for any missing required role content -- a SAME-BEAT
   replacement only: preserve `beat_id`, `speaker_role`, `arc_phase`, beat count, `target_words` range; fill only
   narrative + new meta fields (GPT #4 must-fix). Write `personal_cost` (from the deterministic (speaker,domain)
   table -- MANDATORY, no structured field exists, Gemini #3/GPT #7) and `sensory_consequence`/`state_change`
   for the climax beat into `beat.meta`.
4. run existing budget/arc validators UNCHANGED.
5. run the NEW `beat_role` validator LAST, preserving the "return on first failure" contract (Grok #1): exactly
   one `personal_stake` before the first `irreversible_choice`; exactly one `irreversible_choice` as the last
   voiced beat. (`beat_role` is a separate field, never `arc_phases`.)

## L1 crisis-noun repair (R3-corrected)
- Allowed palette = `allowed_roster` + normalized title/premise/logline nouns. Repair ALL ungrounded crisis
  nouns deterministically from the beat's assigned palette object (Gemini CUT the arbitrary cap -- partial
  repair leaves residual hallucinations). Substitute whole-token (word-boundary, singular/plural) ONLY in an
  explicit allowlist of mutable fields = `beat.intent` + new SQ meta summary fields; NEVER `speaker`, `beat_id`,
  `speaker_role`, `arc_phase`, `target_words`, `sfx_cue`, or roster/title/premise (GPT #8). No LLM, no retry.
- L1a: route `allowed_roster` into the composer's split `allowed_people`/`allowed_things` at the writer call site
  (VERIFY they are populated today -- GPT #6), filtering `"ANNOUNCER"`/`"NARRATOR"` from render while the phantom
  gate still receives the union (Gemini optional).

## L5a (R3-corrected -- two parts)
- **(i) The cap hides the best writer.** `compute_edit_cap = min(8, max(3, voiced_beats//3))`
  (`_otr_ledger_reviewer.py:1136`) caps an 18-beat episode at 6; dense gemma prose trips >6 doctor edits ->
  `too_many_edits` -> terminal stop -> never arc-graded ("?"). FIX: raise/scale the cap ceiling (e.g. scale by
  word count, not just beats) so good-but-heavily-doctored output is not falsely terminated; OR add an ADVISORY
  grade-only `run_story_critic` BEFORE the terminal stop that stamps ONLY `meta.story_critic_report` (no reroll,
  on the pre-rollback snapshot -- GPT #1). Recommend BOTH: scale the cap (smallest change) + make the downstream
  consumer tolerate a missing `story_critic_report` on terminal verdicts (Gemini #1). Grading != editing.
- **(ii) Telemetry undercount (EP16).** Two fixes: (a) `_meta.setdefault("story_quality", {}).update({...})`
  instead of blind replace (Gemini/GPT #11); (b) aggregate from the FINAL persisted rows -- the cascade
  restore/rollback runs before scrub, so the objective_literal_retry flags must be counted from the rolled-
  forward ledger that is actually saved (Grok #3). Telemetry/grading only; never the frozen ledger schema.

## L3 / L4 (R3-corrected)
- L3 (`OTR_COMPOSER_ACTION_STRIP`): composer marks non-spoken action with an explicit `ACTION:` marker; a
  deterministic regex strips only the `ACTION:`-marked segment. Insert immediately after compose/polish returns
  text and BEFORE line persistence/compose_flags (GPT #9). Add a minimal `compose_flags` marker
  (`action_strip:regex`) for telemetry; do NOT persist `internal_action` (GPT #6).
- L4 (`OTR_TRANSCRIPT_SANITIZER`): operate on FINAL line TEXT only, after all text is final but BEFORE
  freeze/TTS/hash/golden (GPT #9); never touch speaker-label/identity fields. Conservative quote balancing
  (leading/trailing wrappers only -- GPT #7). Mojibake = CUT for v0 (verify-only).

## CUT / DEFERRED (R3-confirmed)
- `choice_summary` seed-keyed outro template family -> CUT for v0 (Grok: announcer already routes narration; the
  on-stage climax BEAT is the fix, the outro reference is polish). Revisit later.
- L1 partial-repair cap -> CUT (repair all).
- L4 mojibake repair -> CUT (verify-only).
- Structured personal_stake discovery -> CUT (use the deterministic table directly).
- L5b gemma-12b default -> DEFERRED (bake-off, operator-gated). L6 best-of-N -> DEFERRED (operator asked; revisit
  after L1/L2).

## Acceptance metric (R3-refined)
- `ungrounded_crisis_density = matches / total_voiced_words` per episode (store raw numerator+denominator too --
  GPT optional); cross-episode distinct `conflict_object`/`conflict_type` counts (the sameness measure).
- Required-role presence (personal_stake before irreversible_choice; irreversible_choice last).
- PREREQUISITE: L5a (trustworthy telemetry + the best writer actually graded).
- A compatibility test: inject unknown `meta.story_quality` keys + new `compose_flags` -> freeze/TTS/serialize
  must ignore them (GPT #3 should-fix / R3 target 5).

## Build order (R3-final, scaffolding-first, flag-gated)
1. **L5a** -- cap scaling + advisory grade-only critic + telemetry merge/source fix (enables measurement; no
   audio effect; harness/telemetry only).
2. **L1/L2 scaffolding** -- `beat.meta` fields + `select_domain` + palette table + deterministic selectors +
   fallback beat factory + new role validator + LineRequest fields + call-site threading + tests, ALL flag OFF
   => no population, no drift (no-drift JSON assert + prompt byte-identical assert).
3. **L1/L2 render ON** (`OTR_STORY_QUALITY_L12`) -> re-soak small matrix -> measure sameness.
4. **L3** then **L4** (audio-affecting; golden re-baseline each; operator-gated GPU render).
5. Evaluate **L5b** bake-off; (operator) decide **L6**.
Each chunk: full suite + Bug Bible green, no-drift JSON assert, commit + push.

## Residual verify-at-build (carry to R4)
- Confirm the beat dataclass is `Beat` only (no separate `OutlineBeat` -- Grok assumption; grounding shows only
  `Beat`).
- Confirm the outline->ledger serialization path (does Beat `model_dump()` reach the frozen ledger? decides
  exclude=True vs meta-only).
- Confirm `allowed_people`/`allowed_things` are actually populated at the writer call site today (L1a).
