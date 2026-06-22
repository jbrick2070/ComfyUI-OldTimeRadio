# STORY-QUALITY LIFT -- hardened plan (pass01, post-R1 arc/creative)

Supersedes pass00 forward. R1 panel = GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude anchor/judge.
Schema `l3-2026-05-14` FIXED; branch v2.0-alpha. This window is planner-only (produces the coder
kickoff, not production code). Mechanism choices marked "(R2)" are deliberately deferred to the
coding-plan round.

## 0. Goal + measurable target (was vague)

Lift the WEAK-end story (local mistral + bypass floor was C+). A fix is valid only if it is a NO-OP on
a strong/opus script. Campaign target (no-bypass, weak local config): story_quality_scan deltas
improve (leak count -> 0, role-consistency violations -> 0, >=1 stance reversal caught+repaired), no
new criticals, and ZERO strips/rerolls fired on a known-good strong-model fixture.

## 1. Hard invariants (R1 added the last three)

- Ledger schema `l3-2026-05-14` FIXED -- new signals ride free-form `meta`, no new Pydantic fields.
- ZERO workflow-JSON change -- hash `workflows/otr_scifi_16gb_full.json` before/after the suite, fail on diff.
- Audio spine FROZEN. THREE acceptance lanes (resolves the byte-identical-vs-frozen-text tension):
  (a) hygiene/unit tests -- no audio golden; (b) existing frozen audio baseline -- byte-identical
  unless the changed fixture is explicitly in scope; (c) Chandra re-smoke -- if spoken text changes,
  operator-gated golden recapture OR compare only the pre-TTS frozen-ledger text.
- **COERCE, NEVER CRASH.** A runtime correctness fix coerces to a safe value + LOUD log; it never
  raises in the render path (a halt on a weak model is worse than the defect). Raising asserts are
  CI/test-only.
- **ONE gate path per concern** (no overlapping quality systems): existing critic/reroll for STORY
  issues; deterministic ledger COERCION for STRUCTURAL correctness; deterministic SCRUB for text
  HYGIENE. Nothing gets two competing mechanisms.
- Model-agnostic, deterministic (seed-keyed), LOUD fallbacks, UTF-8 no BOM, SFW.

## 2. DEFECT 1 (TOP) -- bare stage-direction leak: TIERED fix

Corpus (real frozen ledger): b005/b010/b012 = trailing after close-quote; b015 = embedded between
quoted spans (malformed quoting -- spoken text also sits OUTSIDE quotes); b017 = embedded, NO quotes.

Three tiers, primary -> backstop:

- **Tier 1 GENERATION (primary lever):** the line composer / Stage-3 beat prompt instructs PURE
  SPOKEN WORDS -- first/second person, no third-person physical-action narration. Most of the lift
  comes from not emitting the leak in the first place (a detector-only approach is an arms race).
- **Tier 2 COMPOSER REROLL (`_otr_line_composer.compose_line` 2015-2060):** extend the existing
  one-shot stage-direction reroll to DETECT trailing + embedded + undelimited (b017) cases and reroll
  with a LOUD hint. This is the ONLY tier that touches the unquoted/undelimited and malformed-quote
  (b015) cases. On reroll failure it keeps the draft (floor is the backstop).
- **Tier 3 DETERMINISTIC FREEZE FLOOR (`_otr_ledger_scrub._strip_stage_directions`):** strip ONLY
  HIGH-CONFIDENCE cases -- a third-person present-tense physical-action clause that sits OUTSIDE a
  matched quote pair, i.e. the clean `"<spoken>." <action>` and `"<spoken>." <action> "<spoken>"`
  shapes (b005/b010/b012). **Do NOT** blind-strip all extra-quote text (b015 proves legitimate spoken
  text can sit outside quotes) and **EXCLUDE** the undelimited no-quote case (b017) from the floor
  entirely -- there is no safe structural anchor. The floor must classify the span as action, not
  just locate it.

Detection primitive + exact classification rules (sentence-segment-then-classify vs span regex) =
**(R2)**. Hard requirements regardless of primitive: (i) negative fixtures REQUIRED before any
deterministic strip is accepted -- legitimate first-person action narration, quoted titles,
scare-quotes, benign lowercase clauses after punctuation; (ii) a stripped line must remain WELL-FORMED
(balanced quotes, no orphan punctuation, non-empty) -- add to acceptance; (iii) the deterministic
strip must fire ZERO times on the strong-model fixture.

## 3. DEFECT 2 -- incoherent antagonist arc: ARC-LEVEL, not line reroll

Locked by R1 (unanimous): a stance reversal spanning b003->b008->b011/b014->b017 cannot be repaired by
rerolling one line -- that only rewords the contradiction or thrashes. Split detection from repair:

- **Detection:** add a per-character STANCE / MOTIVATION COHERENCE axis to the existing critic
  (`_otr_story_critic.py` 253-335; reuses the single critic LLM call, no new subsystem). It must NAME:
  character, the object/person the stance is toward (constrained v1 to the episode central object +
  protagonist; critic states the target in `meta`), prior stance, new stance, the missing turn beat,
  and affected line ids.
- **Repair (R2 -- the key coding-round decision):** must act ABOVE the line. Candidate, reusing
  existing machinery: a critical stance reversal escalates through the cascade's EXISTING episode-scope
  structural-failure path (`_otr_freeze_cascade` -> `needs_full_rerun`) with a coherence hint injected
  into the outline/`DramaticState` (the antagonist's binding `character_b_wants` already has a
  `_wants_must_oppose` validator) so the rerun executes a coherent through-line -- NOT a per-line
  reroll. Decide in R2: outline re-intent vs episode escalation, and the determinism/cost of a rerun on
  a weak model. Acceptance = the FINAL frozen ledger has no unresolved critical stance reversal (detect
  alone is insufficient).

## 4. DEFECT 3 -- b011 role mis-stamp: deterministic COERCION (not a raising assert)

b011: char_id=c02 (cast id) but speaker_role=announcer -- internally inconsistent. R1 correction
(Gemini, grounded): a fatal assert would HALT the pipeline on a weak model -> coerce instead.

- **Runtime coercion** at the role write points (`production_ledger.init_lines_from_outline` 684-805,
  `set_lines` 839-906) AND the `role_mismatch` repair guard (`_otr_ledger_reviewer.py` 1054-1070):
  if `char_id` resolves to a known cast id, force `speaker_role="character"`; reject
  `expected="announcer"` from the repair when char_id is a cast id. LOUD log on coercion.
- **Invariant (CI/test-only assert):** cast char_id => role=character; role=announcer => char_id=="announcer".
- **Audit (no schema change):** when `speaker_role` changes, stamp prev/new/source/reason into `meta`
  or test logs, to trace the exact origin (init beat vs repair) at build.

## 5. DEFECT 4 -- abrupt UN escalation: CUT the gate

Unanimous CUT. It is a symptom of DEFECT 2 (incoherent outline) + the weak model; a semantic
proportion/setup gate is the most likely to flake on strong scripts and the least grounded (no seam).
Keep ONLY an optional `story_quality_scan.py` scope-jump telemetry count (not in story-lift
acceptance, no reroll/gate). Re-open ONLY if a no-bypass frontier re-smoke still shows abrupt jumps
after DEFECT 2 ships.

## 6. Sequencing + milestones

0. **No-bypass BASELINE re-smoke FIRST** (before any code): box reset (CLAUDE.md S4 selective CIM
   kill) -> re-smoke the real `otr_scifi_16gb_full.json` with OTR_BYPASS_FREEZE_HALT OFF, so fixes are
   designed against real halt behavior, not bypass-only. Operator-gated (resets the resident :8000
   server / interrupts any live OBS).
1. **DEFECT 1** (most visible + most grounded) -- tiers 1->2->3, negative fixtures, well-formedness.
2. **DEFECT 3** (contained correctness coercion).
3. **DEFECT 2** (deepest; mechanism finalized in R2).
4. DEFECT 4 -- cut (optional scan telemetry only).
Each chunk: full suite + Bug Bible green; JSON hash unchanged; audio lane per the three lanes above;
caught -> repaired/rerolled -> ABSENT from final frozen ledger (explicit behavior on reroll
exhaustion); no-op proven on the strong-model fixture.

## 7. Open questions to resolve in R2 (coding plan)

1. DEFECT 1 detection primitive + classification rules (segmenter vs span regex; the 3rd-person
   present-tense physical-action test; how the quote-boundary tier stays safe on b015's malformed quoting).
2. DEFECT 2 repair mechanism: outline re-intent vs the existing `needs_full_rerun` episode escalation;
   determinism + cost on a weak model; how the coherence hint reaches the outline/DramaticState.
3. DEFECT 3 exact origin of the b011 stamp (outline beat vs role_mismatch repair) -- trace at build via
   the meta audit; confirm the single correct coercion point.
4. The strong-model NO-OP fixture: which known-good ledger, and the assert that zero strips/rerolls fire.
