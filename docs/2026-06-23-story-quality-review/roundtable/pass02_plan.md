# Story-Quality LIFT (post-R3) -- Build Plan v2 (after R2: coding / implementability)

Grounded against real source (`GROUNDING_EXCERPTS.md`). R2 panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro +
Grok-4.3 (`pass02/`, ~$0.095). R2 turned the arc into a build spec and caught real schema/validator wiring
errors; all folded.

## R2 corrections that reshaped the spec
1. The new fields do NOT exist in the code -> specify EXACTLY where they live (below). (Judge note: adding
   optional fields to the internal `LineRequest`/`Beat`/`OutlineBeat` dataclasses with empty defaults is NOT a
   workflow-JSON change and NOT a frozen-ledger-schema change -- R3's spine already added
   `story_quality_v2_enabled` to `LineRequest` the same way. Gemini's "schema contradiction" conflated the
   Pydantic dataclass with the frozen ledger schema l3-2026-05-14 / the workflow JSON. The frozen things stay
   frozen; ledger-VISIBLE values still ride `meta`/`compose_flags`.)
2. Do NOT overload `EpisodeBudget.arc_phases` with `beat_role` -- it breaks the `per_phase_words`/`per_phase_beats`
   zip (`_otr_outline.py:781`) and the monotonic validator (:853). `beat_role` is a SEPARATE beat field + a NEW
   validator. (GPT/Gemini/Grok unanimous.)
3. There is NO domain signal in meta (grounded: only `allowed_roster`). L1b must classify DETERMINISTICALLY
   (keyword map), never an inline LLM. (Unanimous.)
4. L1+L2 is too big for one commit -> SCAFFOLDING-FIRST (fields + selectors + validators + tests, flag OFF =
   byte-identical), THEN enable rendering under one flag. (GPT.)

## Flags (defined now; all default-OFF => byte-identical; named per R2/GPT)
- `OTR_STORY_QUALITY_L12` -- gates the L1+L2 beat-planner changes (prompt/render side).
- `OTR_COMPOSER_ACTION_STRIP` -- gates the L3 composer delimiter + strip (audio-affecting).
- `OTR_TRANSCRIPT_SANITIZER` -- gates the L4 sanitizer (audio-affecting).
Each audio-affecting flag ON requires a deliberate `test_audio_byte_identical` golden re-baseline (operator-gated
GPU render); document the re-baseline command + path in the build PR.

---

## Where every new field lives (the data model -- VERIFY exact dataclass names in R3)
- `Beat`/`OutlineBeat` (outline dataclass): add `conflict_object: str = ""`, `conflict_type: str = ""`,
  `beat_role: str = ""` (defaults empty -> serialization + existing tests unchanged). Ledger-visible copies ride
  `beat.meta` / `compose_flags`, NOT new frozen-ledger Pydantic fields.
- `LineRequest`: add `beat_role: str = ""`, `conflict_object: str = ""`, `conflict_type: str = ""` (optional,
  default empty). Render in the existing DRAMATIC FRAME block ONLY when non-empty -> unset = pre-change prompt
  byte-for-byte (same pattern as the L2 deflect path at `_otr_line_composer.py:1228`).
- `EpisodeBudget.arc_phases`: UNCHANGED (do not touch -- correction #2).
- Acceptance counters ride `meta.story_quality` (the existing gated telemetry dict).

---

## CORE -- L1 + L2 (scaffolding-first, then flag-gated render)

### L1 -- premise-anchored conflict, deterministic (no retry, no gate)
**L1a (data exists today): anchor beats on the real `allowed_roster` proper nouns.** Route the roster into the
beat prompt so scenes name the actual program/agency/place (Chandra, El Nino, US Space Force) instead of generic
"the array/the core". WIRING VERIFY (GPT): the composer renders `allowed_people`/`allowed_things` (the split
fields); `allowed_roster` today feeds `detect_phantom_names`, not rendering -- so route via the writer call site
that already populates the split fields, filtering out "ANNOUNCER" (Gemini).

**L1b (new source): a deterministic domain -> conflict palette.**
- `select_domain(meta: dict, premise: str) -> str`: an ordered keyword map (classroom/education, legal/tribunal,
  climate/disaster, space/orbital, archaeology, medical, ...), default `"general"`. No LLM. (GPT/Gemini/DeepSeek/Grok.)
- A curated UTF-8 data table `domain -> {conflict_objects[], conflict_types[]}` (e.g. classroom -> lesson
  plan/parent board/demo + power-asymmetry; legal -> injunction/leaked memo/testimony). `"general"` gets a
  generic "institutional power" palette (Gemini) so no premise is left without a palette.
- Python deterministically (seed-keyed: `sha256(f"{episode_seed}:{beat_index}:{domain}")` -> sorted-palette
  modulo; NOT Python `hash()` -- GPT) assigns each beat a `conflict_object` + `conflict_type`. The beat prompt
  VERBALIZES the chosen slot; the model does not invent one.

**L1 crisis-noun repair (deterministic substitution, NOT regeneration).**
- Build the ALLOWED palette from `allowed_roster` + normalized title/premise/logline nouns (GPT: "switch"
  appears in a real episode TITLE; don't corrupt legit terms). Crisis denylist counts ONLY ungrounded crisis
  nouns NOT in that allowed set. Cap = `max(1, floor(total_voiced_beats * 0.2))`.
- On exceed: DETERMINISTICALLY substitute the offending beat's conflict noun from its assigned palette object
  (whole-token, word-boundary, singular/plural map; only in GENERATED intent-like fields; never touch proper
  entities). No LLM call, no retry -> not a gate.

### L2 -- phase = dramatic FUNCTION via a separate `beat_role` + a new validator
- `beat_role in {setup, pressure, personal_stake, irreversible_choice, consequence}` (resolve the enum/"climax"
  inconsistency -- GPT #5: the climax IS `irreversible_choice`, last voiced beat; optional display label only).
- A NEW validator (separate from the arc_phase monotonic one; PRESERVE its "return on first failure" contract --
  Grok #3): exactly one `personal_stake` BEFORE the first `irreversible_choice`; exactly one `irreversible_choice`
  as the LAST voiced beat. Role allocation by voiced-beat count (GPT #7): n=1 -> irreversible_choice; n=2 ->
  personal_stake,irreversible_choice; n>=3 -> setup,personal_stake,pressure...,irreversible_choice.
- CONTENT, not labels (Gemini/DeepSeek):
  * `personal_stake` content source: VERIFY a structured character cost/fear field exists (composer only gets
    `all_voice_cards` as a rendered string -- GPT #8). If none, a deterministic fallback table keyed by
    (speaker, domain); store the chosen private-cost text in `beat.meta`.
  * `irreversible_choice`/climax: require a sensory-consequence + a state-change, checked by FIELD-PRESENCE
    (markers stored as beat fields), not prose regex (GPT #4 should-fix). On absence -> a deterministic fallback
    beat factory `make_required_role_beat(role, arc_phase, speaker, target_words, conflict_object, conflict_type,
    choice_summary, ...) -> beat` returning ALL required beat fields (GPT #10) -- not an LLM reroll.
- Carry `beat_role` + `conflict_object` into the composer via the new optional `LineRequest` fields (above).
- Announcer outro references the climax CHOICE: add a `choice_summary` filled by the slot system (GPT #11) +
  a small seed-keyed template family (avoid one fixed "Because X chose Y").

---

## HYGIENE (after the core)
### L3 -- strip narrated-action via a conservative marker + deterministic regex (flag `OTR_COMPOSER_ACTION_STRIP`)
- Composer marks non-spoken action with an explicit `ACTION:` marker (preferred over bare brackets -- GPT/Gemini:
  "[laughs]" / acronyms / measurements are legit and weak models leave brackets unclosed). A deterministic regex
  strips only an `ACTION:`-marked trailing segment (or a segment matching a conservative third-person
  stage-direction pattern). Do NOT persist `internal_action` (CUT -- GPT: stripping is the only required
  behaviour). Claim: "strips what the model marks; L4 catches the rest" -- not "eliminates."

### L4 -- minimal transcript sanitizer (regex, flag `OTR_TRANSCRIPT_SANITIZER`)
- Operates on FINAL transcript line TEXT only, never speaker-label/identity fields (GPT #6). Strip/repair
  prompt-leak ("voice should", director-note patterns) anchored to line start / quoted leakage; balance quotes
  conservatively (don't swallow apostrophes/measurements -- Gemini). Mojibake = VERIFY-ONLY (build-artifact;
  confirm in a real ledger before any encoding repair -- GPT/DeepSeek).

## INDEPENDENT / EARLY
### L5a -- fix critic `too_many_edits -> arc="?"` abort + `meta.story_quality` telemetry under-count
- R3 task: GROUND the exact files/functions (where `too_many_edits` is set, where `arc="?"` aborts grading,
  where `meta.story_quality` aggregates -- EP16 had `objective_literal_retry` flags but `l1_rerolls=0`). Fix is
  telemetry/grading ONLY; never touches the frozen ledger schema or the outline (Grok #2). Do FIRST (enables
  trustworthy measurement -- GPT).

## DEFERRED (documented; operator decides)
- **L5b gemma-12b default** -- gate on L5a root-cause of the abort + a controlled 5-brief bake-off (scored,
  eval-only not a gate -- GPT #9) + a minimal side-effect re-soak. Not leading.
- **L6 best-of-N** -- CUT from v0 (unanimous R1+R2). On record (operator asked); revisit after L1/L2.

---

## Acceptance metric (concrete -- GPT #14)
- `ungrounded_crisis_density = (ungrounded crisis-noun matches) / (total voiced words)` per episode; target a
  large drop vs the R3 soak baseline.
- `distinct_conflict_types / episode` and distinct `conflict_object` n-grams ACROSS a soak (the cross-episode
  sameness measure -- the real goal).
- Required-role presence: personal_stake before irreversible_choice; irreversible_choice = last voiced beat;
  outro references choice_summary.
- Persist per-episode counts under `meta.story_quality`; PREREQUISITE = L5a (trustworthy telemetry).
- Guard: do NOT reintroduce longer-but-monotone (the 430w standoff).

## Build order (R2-revised, scaffolding-first)
1. **L5a** (telemetry/critic-abort fix; enables measurement; no audio effect).
2. **L1/L2 scaffolding** (add the dataclass fields + `select_domain` + palette table + deterministic selectors +
   the new `beat_role` validator + the fallback beat factory + tests, ALL with flag OFF -> byte-identical).
3. **L1/L2 render ON** behind `OTR_STORY_QUALITY_L12` (beat prompt verbalizes slots; composer renders the new
   fields). Re-soak small matrix -> measure sameness.
4. **L3** (`OTR_COMPOSER_ACTION_STRIP`) then **L4** (`OTR_TRANSCRIPT_SANITIZER`) -- audio-affecting, golden
   re-baseline each.
5. Evaluate **L5b** bake-off; (operator) decide **L6**.
Each chunk: full suite + Bug Bible green, no-drift JSON assert, commit + push.

## R3 grounding targets (wiring round)
1. Exact dataclass names/shapes: the outline `Beat`/`OutlineBeat`, `LineRequest`, `EpisodeBudget`, and the writer
   call site that populates `allowed_people`/`allowed_things` + `LineRequest` beat fields.
2. The new `beat_role` validator's insertion point vs `validate_outline_against_budget` (first-failure contract).
3. The critic `too_many_edits`/`arc="?"` + `meta.story_quality` aggregation code (L5a).
4. Whether a structured character cost/fear field exists (L2 personal_stake source) or a fallback table is needed.
5. Confirm unknown `meta`/`compose_flags` keys are ignored by freeze/TTS/serialization (a compatibility test).
