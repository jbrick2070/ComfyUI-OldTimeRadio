# STORY-QUALITY LIFT -- problem statement to harden (pass00)

**Status:** seed document for a 4-round roundtable (R1 arc/creative -> R2 coding plan ->
R3 wiring -> R4 convergence). Claude is grounded anchor panelist + sole judge.
**Date:** 2026-06-22. **Branch:** v2.0-alpha (HEAD `223877a`). **Schema:** ledger `l3-2026-05-14` (FIXED).
**Source evidence:** live FLOOR smoke "Chandra's Echo"
(`output/otr/episodes/signal_lost_chandras_echo_20260622_141546/`), local mistral-nemo writer,
320 words, 3 chars, `OTR_BYPASS_FREEZE_HALT=1`. Findings:
`docs/2026-06-22-voice-casting-arch/SMOKE_CHANDRAS_ECHO_FINDINGS.md`. Story grade **C+ (~6/10)** --
floor for this config (weak local model + bypass), not the ceiling. Goal: lift the FLOOR.

## 0. Goal (one line)

Make the weak-end story measurably better by closing four grounded defects from the smoke --
WITHOUT rewriting the strong end (opus). Every gate must be one a strong model already passes;
it lifts the weak/local model and is a no-op on a good script.

## 1. Hard invariants (reject any fix that breaks one)

- **Ledger schema `l3-2026-05-14` is FIXED.** New signals ride the free-form `meta` dict; NO new
  Pydantic fields, NO new line-row columns.
- **ZERO workflow-JSON change.** All fixes are internal node code (node 1 `OTR_LedgerScriptWriter`
  and its modules + the freeze cascade host node 62). No node/wiring/widget edit to
  `workflows/otr_scifi_16gb_full.json`. Add a no-drift regression assert.
- **Audio spine FROZEN, byte-identical.** `test_audio_byte_identical` stays green. The frozen
  baseline voice is indextts2. A craft fix that changes generated dialogue may need a DELIBERATE
  golden recapture (operator-gated) -- flag it, never silently shift the golden.
- **Model-agnostic + deterministic** (seed-keyed). Every in-render fallback is LOUD. UTF-8 no BOM. SFW.
- **Reuse existing machinery.** The compose-time reroll loop (`_otr_line_composer.compose_line`,
  `reroll_hint`) and the scoped critic/reroll convergence (STEP 4/5, `MAX_REROLL_CYCLES=2`) already
  exist -- extend them, do not build a parallel reroll system.

## 2. The four defects (grounded in real code + the real frozen ledger)

### DEFECT 1 (TOP FIX) -- bare stage directions LEAK into spoken text, indextts2 speaks them

The pre-freeze scrub catches LEADING and DELIMITED directions only; bare directions that are
**trailing** or **embedded** survive into the frozen text and are spoken aloud. Real corpus:

- b005 (c03): `"Not before I amplify it. The world deserves to hear this." adjusts dials on the console`
- b010 (c03): `"...My husband's theories have no bearing here." clutches her wedding ring tightly`
- b012 (c04): `"...is purely theoretical." taps his cane impatiently`
- b015 (c03): `...presenting my findings to the UN sooner than expected." tightens her scarf, a nervous gesture "I do hope...`
- b017 (c02): `Sherlock, stop this at once! overrides systems, fingers dancing on the console I won't let you...`

**Three sub-patterns** (the fix must cover all three; the corpus proves each):
1. **Trailing after a closing quote** -- `"<spoken>." <bare action clause>` (b005, b010, b012).
2. **Embedded between quoted spans** -- `"<spoken>." <bare action> "<spoken>` (b015).
3. **Embedded undelimited, NO quotes at all** -- `<spoken>! <bare action> <spoken>.` (b017).

**Grounded code seams (verified):**
- `nodes/_otr_line_hygiene.py` -- `_leading_stage_strip` (246-312) is **LEADING-ONLY**: it requires
  `body[0].islower()` (263-266) and scans from the start, so it returns trailing/embedded directions
  UNCHANGED. `scrub_leading_stage_direction` (315-318), `detect_leading_stage_business` (321-327),
  hint `_BARE_STAGE_HINT` (232-235). Delimited scrubs (`_STAGE_DIRECTION_RES`, brackets/asterisks/
  cue-verb parens) ARE unanchored but only catch DELIMITED text.
- `nodes/_otr_ledger_scrub.py` -- `_strip_stage_directions` (381-412) freeze floor: applies the
  delimited regexes anywhere + the LEADING bare floor; returns `Tuple[str,bool]`; stamps
  `CODE_STAGE_DIRECTION` ("stage_direction_stripped"). No trailing/embedded bare handling.
- `nodes/_otr_line_composer.py` -- `compose_line` (1931-2261): stage-direction reroll at 2015-2060,
  guarded by `_stage_dir_repair_attempted` (one reroll level), runs `detect_leading_stage_business`
  + cliche/stage-business/on-the-nose flags, concatenates `reroll_hint`, freeze floor as backstop.

**Proposed approach (to harden):** add a `detect_trailing_embedded_stage_direction` (or extend the
existing detector) covering the three sub-patterns, with a tight false-positive guard (distinguish a
3rd-person physical-action clause from legitimate spoken narration); wire it into all three layers in
ONE change -- detector (`_otr_line_hygiene`) -> compose-line reroll (`_otr_line_composer`) ->
deterministic freeze floor (`_otr_ledger_scrub._strip_stage_directions`). Corpus case per line above.
Re-smoke WITHOUT `OTR_BYPASS_FREEZE_HALT`.

### DEFECT 2 -- incoherent antagonist arc (Manfred flip-flops)

c02 (Manfred) reverses stance with no turn beat: supportive (b003) -> dismissive (b008, "nothing
extraordinary") -> betrays Mali by leaking her research to the press (b011, b014) -> defends her
life's work (b017, "I won't let you jeopardize Mali's life's work"). Motivation does not track.

**Grounded seam (verified):** `nodes/_otr_story_critic.py` `run_story_critic` (505-605) has NO
per-character arc-coherence / stance-reversal axis. The 5 craft dimensions are knowledge / pressure /
relationship / decision / obstacle (253-335) + tension fit; arc verdict is only
`strong|uneven|flat|mid_collapse`. The nearest hooks: SECTION 1 CONTINUITY (`ContinuityIssue`,
factual/line-scoped) and SECTION 2 VOICE DRIFT (`VoiceDriftNote`, per-character but voice/register,
not plot stance). Reroll convergence exists (`_otr_reroll.py`, scoped, `failed_dimension`).

**Proposed approach (to harden):** add a per-character STANCE-COHERENCE signal -- flag a character
whose stance toward the central object/another character reverses without an intervening turn beat;
route the flagged line(s) through the existing scoped reroll with a `failed_dimension`-style hint.
Open question for the panel: critic axis vs a deterministic stance-tracker vs an outline-stage guard.

### DEFECT 3 -- b011 character line mis-stamped `speaker_role=announcer`

b011 is `char_id=c02` (a cast character, Manfred) but `speaker_role=announcer`, and the text is
plainly character dialogue. The row is internally INCONSISTENT: `init_lines_from_outline`
(`production_ledger.py:684-805`) sets `char_id="announcer"` when role is announcer (761-766), yet
here char_id=c02 -- so role was set to announcer on a character-charid row.

**Grounded seam (verified):** the only writer of `speaker_role=announcer` onto a non-announcer row is
the `role_mismatch` repair in `nodes/_otr_ledger_reviewer.py:1054-1070` (honors an LLM
`expected="announcer"` if it passes `_ALLOWED_SPEAKER_ROLES`), unless the outline beat itself stamped
the role. **Proposed approach (to harden):** add a role<->char_id CONSISTENCY assert at the role
source (a `char_id` that is a cast id => role must be `character`; never `announcer`); decide whether
to fix at init, at set_lines, or in the role_mismatch repair guard. Trace the exact origin at build.

### DEFECT 4 -- abrupt UN escalation (no setup / proportion gate)

The observatory two-hander jumps to global stakes with no setup: b015 "presenting my findings to the
UN", b016 "I'm overriding the UN's block. Prepare to transmit." Scale leaps from a 2-3 person
observatory scene to UN-level with no intervening beat.

**Grounded seam (verified):** `nodes/_otr_outline.py` `_build_beat_user_prompt` (1166-1236) enforces
escalation ONLY by a soft prompt directive ("escalate, never tread water", 1226-1234);
`compute_beat_tension_ramp` (`nodes/_otr_slot_drama_contract.py:762-792`) is a deterministic ORDINAL
ramp (1->5 by beat position) decoupled from semantic scope; `intent_is_action_under_pressure`
(1251-1257) is measurement-only, non-binding. There is NO proportion/setup gate.

**Proposed approach (to harden):** lightest viable lever -- strengthen the Stage-3 beat prompt to
require a setup beat before a scope jump, and/or a measurement signal in `story_quality_scan.py`. Open
question: is this worth a gate at all, or is it a symptom of the weak local model that a frontier
writer + DEFECT-2 coherence already covers? (Candidate CUT -- panel to judge.)

## 3. Acceptance (per chunk + campaign)

- Per chunk: full suite + Bug Bible green; ZERO workflow-JSON change (no-drift assert); audio
  byte-identical holds (or a deliberate, operator-gated golden recapture); UTF-8 no BOM; SFW.
- DEFECT 1: the 5 corpus lines (b005/b010/b012/b015/b017) are stripped from the FROZEN text; a
  no-bypass re-smoke shows the freeze gate + reroll firing (not shipping via repair-then-ship bypass).
- DEFECT 2/3: the mis-stamp and at least one stance-reversal are caught by the gate on a re-smoke.
- DEFECT 4: measured (scan signal) even if not gated.
- Campaign output: a build-ready, dependency-ordered chunk plan a coder window executes (this window
  is planner-only; it produces the coder kickoff, not production code).

## 4. Open questions for the panel

1. DEFECT 1 false-positive risk: how to reliably separate a bare 3rd-person ACTION clause from
   legitimate spoken narration of action ("I adjust the dials," said in-character)? Is detection
   safe enough to put in the DETERMINISTIC freeze floor, or reroll-only with the floor as a
   conservative backstop?
2. DEFECT 1: is the right primitive a sentence-segmenter (split on `."`/`!`/`?` then classify each
   segment) or a span regex? What is the smallest robust approach?
3. DEFECT 2: critic axis vs deterministic stance-tracker vs outline-stage guard -- which is the
   minimal lever that lifts the weak end without flaking on the strong end?
4. DEFECT 3: fix at the role source (init/set_lines) vs the repair guard -- where does the
   inconsistency actually originate, and what is the single correct stamp point?
5. DEFECT 4: gate vs measure-only vs CUT. Is it a distinct defect or a symptom of DEFECT 2 + weak model?
6. Sequencing: which defect first? (Proposed: DEFECT 1 top -- it is the most visible and the most
   grounded; DEFECT 3 is a contained correctness fix; DEFECT 2 is the deepest; DEFECT 4 may be CUT.)
