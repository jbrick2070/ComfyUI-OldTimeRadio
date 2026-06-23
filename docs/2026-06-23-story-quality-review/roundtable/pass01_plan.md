# Story-Quality LIFT (post-R3) -- Improvement Plan v1 (after R1: arc/creative)

**Goal:** genuinely BETTER stories on a WEAK LOCAL writer, WITHOUT a flag-and-reroll QA gate. Evidence:
`../STORY_REVIEW.md` + `passA_STORY_CRITIQUE_SYNTHESIS.md`. R1 panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro
+ Grok-4.3 (`pass01/`, ~$0.056). R1 forced two structural corrections (below).

## The two R1 corrections that reshaped the plan
1. **A denylist that "regenerates the beat on exceed" IS a reroll gate** (unanimous). The fix must be a HARD
   deterministic constraint, never a retry. -> L1 is recast as a **required, Python-filled structured slot**, not
   a prompt injection and not a regeneration.
2. **"Required beat slots" are just labels unless mechanically filled** (unanimous). Naming a slot "climax" does
   not make a weak model write one. -> L2 slots must carry **deterministically injected content + a deterministic
   fallback**, and the beat's dramatic FUNCTION must be **carried into the composer** as a tag, or the composer
   flattens it back to threat-noise.

## The core finding (unchanged, now with the honesty fix)
OTR already ships SOFT versions of the right instructions and the weak model ignores them (grounded in
`_otr_outline._build_beat_user_prompt` :1166 -- "ACTION UNDER PRESSURE / RAISE THE STAKE / KEEP STANCE
CONSISTENT"; `_otr_line_composer._build_user_prompt` :1065 -- cast cards, dramatic frame, continuity,
`speech_signature`). So the only things that work are DETERMINISTIC + UPSTREAM (Python that builds the beat
skeleton + fills required structured fields), or model-agnostic SELECTION. (For R2/R3 these code excerpts ship
as grounding so the panel verifies them rather than flagging "unverified" -- R1 correctly noted the packet did
not include the source.)

## Hard constraints (every fix satisfies)
1. NO flag-and-reroll critic/QA gate, and NO disguised one (no "regenerate the beat/line on a quality check").
   Deterministic substitution / required-slot construction / regex repair are allowed; an LLM retry keyed to a
   quality verdict is not.
2. Weak/small LOCAL model robust: prefer Python-filled structured fields + deterministic negative constraints
   over positive style nudges; never assume the model obeys a strict output schema (provide a regex fallback).
3. Content-only; ledger schema `l3-2026-05-14` FIXED (new values ride free-form `meta`/`compose_flags`; unknown
   keys MUST be ignored by freeze/TTS/tests/serialization -- verify); ZERO workflow-JSON change unless a node is
   truly added (then same-commit, and update the arc-phase validators in the SAME change -- R1/Grok).
4. Audio spine FROZEN: `test_audio_byte_identical` green; anything that changes generated dialogue is flag-gated
   default-off or a deliberate operator-gated golden re-baseline (name the exact flag + default + re-baseline
   procedure before building -- R1/Grok).
5. Deterministic/seed-keyed; local/offline; UTF-8 no BOM; SFW.

---

## The lever set (revised; STRUCTURAL CORE first, then hygiene, then deferred)

### CORE -- L1+L2 ship TOGETHER (neither alone works)
Rationale (DeepSeek, decisive): L1 alone changes VOCABULARY but the standoff STRUCTURE persists with new words;
L2 alone puts the same threat-noise in relabelled slots. The pair is the structural fix.

**L1 -- premise conflict as a REQUIRED, Python-filled beat field (deterministic; replaces the denylist-as-gate).**
- Every beat intent carries a structured `conflict_object` + `conflict_type` SLOT, chosen DETERMINISTICALLY by
  Python from the brief-derived entities (reuse `allowed_things` / the meta brief; seed-keyed rotation so beats
  don't all grab the most salient noun). The beat prompt may only VERBALIZE the chosen slot, not invent one.
- The generic-crisis vocabulary (override, purge, lever, console, lockdown, core, vent, scrubber, countdown,
  manual control, switch, drive, keycard) is handled WITHOUT a retry: (a) it is excluded from the allowed
  palette, and (b) a post-outline check COUNTS ungrounded crisis nouns (only those NOT in the brief palette --
  GPT: don't fight legit sci-fi usage) and, on exceed, DETERMINISTICALLY substitutes the beat's conflict noun
  from the palette (no LLM regeneration). Cap expressed as a ratio: `max(1, floor(total_beats*0.2))`.
- Per-beat a `conflict_type` derived from the premise (DeepSeek): e.g. classroom -> student/teacher power
  asymmetry, legal -> injunction/testimony, not just an object noun.
- GROUNDED FINDING (R1 assumption CONFIRMED -- I inspected 4 real R3 ledgers): the entity source is
  `meta.allowed_roster` and it carries ONLY PROPER NOUNS -- people/agencies/places/programs (e.g. "NASA",
  "CHANDRA X-RAY OBSERVATORY", "US SPACE FORCE", "EL NINO", "VICTUS HAZE PUMA"). It does NOT contain dramatic
  conflict objects ("injunction", "lesson plan", "testimony"). So L1 CANNOT just "reuse allowed_things" -- that
  palette does not exist today. L1 therefore SPLITS:
    * **L1a (data exists, deterministic): anchor beats on the REAL `allowed_roster` proper nouns** so the scene
      uses the actual program/agency/place (Chandra, El Nino) instead of generic "the array/the core" -- a
      specificity lever with a live data source.
    * **L1b (needs a new source): a domain -> conflict-object/conflict-type palette.** Build a small curated
      table keyed by news domain (classroom -> lesson plan/parent board/demo; legal -> injunction/testimony;
      climate -> forecast/evacuation order). Requires a domain/category signal -- VERIFY a category field exists
      in meta (else classify from the logline). This is the R2/R3 design question.

**L2 -- phase = dramatic FUNCTION via Python-filled required beats (not bare labels).**
- Define a minimal phase contract (GPT): `beat_role in {setup, pressure, personal_stake, irreversible_choice,
  consequence}`; exactly one `personal_stake` BEFORE the first `irreversible_choice`; exactly one
  `climax/irreversible_choice` as the last voiced beat; map roles onto the existing `arc_phases` positions; define
  behaviour when the beat budget is too small (drop optional pressure beats first, never the climax).
- Mechanical fill, not a label (Gemini/DeepSeek): the `personal_stake` beat injects a character-specific private
  cost/fear from the cast/character sheet; the `climax` beat injects a required sensory consequence + a
  state-change. If a generated intent lacks the marker, substitute a deterministic hand-authored fallback beat
  keyed to (phase, conflict_object) -- NOT an LLM reroll.
- Carry the `beat_role` + `conflict_object` TAG into the composer (DeepSeek) so the line honours the beat's
  function. This is a contract field, not a new reroll; the existing dramatic-frame block carries it.
- Announcer outro references the climax CHOICE as a semantic requirement with a small seed-keyed template family
  (GPT: avoid one fixed "Because X chose Y" every time).
- WIRING (Grok): adding required slots to `EpisodeBudget.arc_phases` MUST update the monotonic arc_phase
  validators in the SAME change or phases silently drop.

### HYGIENE -- after the core, lower blast radius
**L3 -- strip narrated-action/meta via a DELIMITER + deterministic regex (not strict JSON).**
- The composer marks non-spoken action with a simple delimiter (e.g. wrap stage business in [brackets] or after
  a fixed marker); a DETERMINISTIC regex strips anything bracketed before freeze/TTS. Robust to weak-model
  non-compliance (Gemini: don't rely on key-value JSON from a 12B). `internal_action`, if kept, rides `meta`
  (schema fixed). Name the flag + default + re-baseline procedure (audio-affecting). Sequence AFTER L1/L2
  (DeepSeek: L1/L2 reduce the narrated-action volume first). Claim downgraded: "strips whatever the model
  delimits; the sanitizer (L4) catches the rest" -- not "eliminates deterministically."

**L4 -- minimal deterministic transcript sanitizer (regex, NOT an LLM gate).**
- Strip/repair prompt-leak ("voice should", "tone", director-note patterns, lowercase "announcer:" inside a
  character line) + unbalanced quotes at freeze. Conservative (Gemini: don't swallow valid apostrophes/
  measurements). Mojibake -> VERIFY-ONLY (the panel-packet instance was a build artifact; confirm the real
  ledger/TTS path before adding any encoding repair).

### INDEPENDENT / EARLY (safe, decoupled from the structural work)
**L5a -- fix the critic `too_many_edits -> arc="?"` abort + the `meta.story_quality` telemetry under-count.**
- The richest gemma outputs go ungraded (the critic bails); EP16 carried `objective_literal_retry` flags while
  `l1_rerolls=0`. Both are measurement bugs that must be fixed BEFORE any re-soak, or improvement cannot be
  measured (GPT). Harness/telemetry only; never touches the frozen ledger schema (Grok). Do this early.

### DEFERRED (gated; documented, operator decides)
**L5b -- gemma-12b as the default creative writer.** Evidence is thin (3 episodes) and gemma's `too_many_edits`
abort may signal formatting instability that could backfire if defaulted (Gemini). DO NOT lead with it (R1
unanimous). Gate on: (i) L5a's root-cause of the abort, (ii) a controlled bake-off (5 briefs x gemma-12b vs
current default, scored), (iii) a minimal re-soak for side effects (Grok). Decoupled from the structural fix.

**L6 -- best-of-N line selection (operator candidate b).** CUT from v0 (R1 unanimous): it is a generate-N-score-
keep-one SELECTION gate (same operational shape the constraint warns against), costs N local generations, and
cannot fix structural sameness (all N share the beat). KEPT ON RECORD because the operator asked for it --
revisit only after L1/L2 establish a new baseline; operator's call.

---

## Acceptance metric (define BEFORE building -- the goal is "better story", measured)
- PRIMARY: cross-episode SAMENESS reduction -- a deterministic diversity measure over a soak (distinct
  conflict-object / conflict-type n-grams across episodes; ungrounded-crisis-noun density per episode). The
  per-episode critic score is NOT the target.
- SECONDARY: required-slot presence (personal_stake before irreversible_choice; on-stage climax as last voiced
  beat); announcer outro references the climax choice.
- PREREQUISITE: L5a first, so telemetry/grading is trustworthy (GPT).
- Guard against the failure to NOT reintroduce: longer-but-still-monotone (the 430w standoff).

## Build order (revised)
L5a (safe, enables measurement) -> VERIFY L1's palette-source (`allowed_things` quality) -> L1+L2 together
(structural core) -> re-soak small matrix (measure sameness) -> L3 (delimiter+regex) -> L4 (sanitizer) ->
evaluate L5b bake-off -> (operator) decide on L6. Each chunk: full suite + Bug Bible green, no-drift JSON
assert, flag-gated where audio-affecting, commit + push.

## Open code-verify items (carried into R2/R3, shipped as grounding)
1. `allowed_things` real content (palette viability) -- a ledger inspection. GATES L1.
2. The arc_phase skeleton builder + monotonic validators -- where required `beat_role` slots inject.
3. Composer output path + freeze/scrub -- delimiter-strip feasibility + the audio-affecting flag.
4. The critic `too_many_edits` abort path + telemetry aggregation (EP16 undercount).
5. The writer-default selection point.
