# STORY-LEDGER INTEGRITY -- R1-converged DRIFT plan (2026-06-25)

Synthesized from Claude's grounded anchor + a 3-model panel (gpt-5.5,
gemini-3.1-pro, deepseek-v4-pro), every claim checked against the real
`_otr_story_critic.py` / `_otr_freeze_cascade.py`. Operator steer: FOCUS ON DRIFT
(broad already hashed). Governing principle (unanimous): an accuracy GUARD must be
DETERMINISTIC + offline + CI-runnable -- never an LLM that can fail-open. LLMs stay
ADVISORY (findings/repair hints); binary gates are deterministic or degrade to an
explicit "unchecked", NEVER silently "pass".

## Priority order (the build sequence)

### 1. Kill the critic FAIL-OPEN (the #1 drift) -- CONFIRMED in code
`StoryCriticReport.clean()` (lines ~189-197) returns `arc_verdict="strong"` + empty
issues when the LLM ladder exhausts/raises (~445-455): a CRASH reads as a
masterpiece. Fix:
- Add `"unverified"` to `ArcVerdict`; `clean()` returns `arc_verdict="unverified"`
  (NOT "strong"). Also stamp `meta.story_critic_status = {ran, validated, failure}`
  outside the frozen report so a reader never treats unrun == clean (keeps the
  report schema stable; the enum add is the one tiny schema touch).
- The freeze cascade maps `"unverified"` to a NON-clean state (never
  `frozen_clean`); observable + restampable, not a hard block (don't gate ship on a
  flaky LLM).
- A3 mechanical floor (Gemini, ~567-590): if it appends anti-loop reroll targets to
  a report whose `arc_verdict` is `strong`/`unverified`, deterministically downgrade
  to `"uneven"` (a report can't be "strong" AND demand rerolls).

### 2. Deterministic CROSS-STAGE consistency assertion (THE core drift fix)
The `sound_palette` bug proved nothing asserts the ledger/canon ingested its
upstream. Build a SOURCE-OF-TRUTH matrix + a pure offline CI parity test
(`tests/test_ledger_canon_parity.py`): dynamically reflect the `StoryContract` +
`CastLock` (+ outline) and assert every non-optional upstream field has a mapped,
populated equivalent in frozen `ledger.meta` + `episode_canon`. Matrix columns:
`field | source | canon/ledger path | normalizer | required? | assert-timing`.
Minimum rows: `sound_palette<-contract.sound_world`, title, premise, setting,
time_of_day, style<-contract.slug, cast ids/names/roles<-CastLock, outline beat
ids, line `beat_id`. PURE python (no LLM -> cannot fail-open). Runs as a CI test
AND a pre-freeze deterministic assertion. (verify-at-build, DeepSeek/Gemini
[ASSUMPTION]: contract+outline+CastLock are in scope at freeze + the schemas are
importable; the workflow JSON is committed.)

### 3. CI drift guards (make the existing checks STANDING, not ad-hoc)
- **Widget positional drift (BUG-LOCAL-097):** a mandatory offline CI test loads
  the canonical `otr_scifi_16gb_full.json`, zips `widgets_values` against live
  `INPUT_TYPES` per node, fails on any non-append misalignment. (verify: exact
  `OTR_WorkflowValidator` name + whether a test already invokes it -- if so, just
  gate it.)
- **Schema-version drift:** migration/compat for `l3-2026-05-14`: a vintage-ledger
  fixture test; any field whose default changes SEMANTICS must fail-loud or have a
  deterministic derivation, never silently default to wrong.

### 4. Freeze WARN taxonomy (stop shipping accuracy defects as "warns")
The cascade labels an arc/critic failure `structural` then ships "the best
candidate" (contradictory). Define: `structural_error` -> BLOCKS at Phase 10;
`story_accuracy_warning` (continuity/`unverified`/canon-divergence) -> ships ONLY
as non-clean with operator-visible meta; `cosmetic_warning` -> clean-with-warns.
Wire critic findings + gap-audit warns into this taxonomy instead of raw counts.

### 5. Make the critic actually WHOLE-story (grounded gap)
`_critic_character_lines` (line ~394) filters to `speaker_role=="character"` -->
announcer/music/SFX/title framing are INVISIBLE to it, yet drift lives there. Fix:
give the critic READ-ONLY context for ALL story-bearing lines; keep actionable
`reroll_targets` restricted to character lines (deterministic validator rejects
targets on non-rerollable lines). Also pass the original outline `beat_intent` into
the critic prompt so a Script-Doctor rewrite that drifts off the beat is caught
(the doctor runs BEFORE the critic -> currently judged against itself, not intent).

## CUT (unanimous)
- **Multi-LLM voting for binary gates** -- conflicts with the deterministic-guard
  invariant; adds cost + model-dependence, still can't prove correctness. (This is
  the honest answer to "multiple LLMs on binary decisions": use them for advisory
  findings, NOT as the gate.)
- **`StanceIssue`** (lines ~150-166) -- self-described "TELEMETRY ONLY / dead-end
  repair path"; it burns the critic's attention budget for zero gating value. Delete
  the model + remove from the prompt.
- **An LLM "positive-evidence engine" that second-guesses the critic** -- the
  deterministic status stamp + parity test is simpler and sufficient.
- Do NOT reopen the settled binary-leak lane / leak gates / structured-call
  tolerance.

## Invariants
Guards deterministic + offline + CI-runnable; ledger schema frozen except the one
`ArcVerdict` enum add (an enum value, not a field); no workflow-JSON node churn;
byte-identical audio spine + canonical happy path untouched; UTF-8 no BOM; SFW.
