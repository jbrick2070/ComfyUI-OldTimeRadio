# OTR Story Architecture -- Increment 1 SUBAGENT SPRINT (overnight, code-ready)

Source of truth: `docs/2026-06-23-story-architecture/SPEC.md` (CONVERGED, 4-round roundtable).
This file packages that SPEC into discrete, default-OFF tickets a coder window can execute
sequentially overnight, with the verify-at-build symbols already grounded. Every ticket ships
**dark** (byte-identical when its flag is off) so nothing changes the live bake until the operator
flips flags in the morning.

## Grounding catches already resolved (do NOT re-derive; the SPEC half-flagged these)

- **Conflict palette symbol = `DOMAIN_PALETTE`** (module-level `Dict[str, Dict[str, Tuple[str,...]]]`
  in `nodes/_otr_story_quality_l12.py` ~L87) + a domain getter ~L363
  (`return DOMAIN_PALETTE.get(domain) or DOMAIN_PALETTE["general"]` -- confirm its exact def name and
  import THAT, do not copy the dict). The SPEC's assumed `load_conflict_palette()` does **not** exist.
- **Climax role = `BEAT_ROLE_IRREVERSIBLE_CHOICE = "irreversible_choice"`** (+ `BEAT_ROLES` tuple,
  ~L51-62). The role-sequence helper already guarantees it is the LAST voiced character beat (~L450-471).
  Import the symbol for C4; don't hardcode the string.
- **CRITIC FIELD MISMATCH (C2 Tier 1's main risk).** `StoryCriticReport` (`nodes/_otr_story_critic.py`
  L241) exposes `arc_verdict` (strong|uneven|flat|mid_collapse), `reroll_targets[]` (each with
  `line_id` / `hint` / `failed_dimension`), `flat_lines`, `continuity_issues`, `render_priority` --
  **NOT** `failing_axes` / `regeneration_hint`. C2 MUST build an adapter:
  `failing_axes` <- derive from `arc_verdict` + the escalation router's `STRUCTURAL_AXES` decision
  (`_otr_reroll_escalation.decide_escalation_scope`); `regeneration_hint` <- synthesize from the
  `reroll_targets[].hint` set (or arc_verdict). Verify-at-build #7 is REAL.

## Global discipline (every ticket -- from CLAUDE.md)

- Edit the canonical `workflows/otr_scifi_16gb_full.json` IN THE SAME commit as any node/widget change
  (append new widgets at the END of `widgets_values`, BUG-LOCAL-097; re-validate with
  `OTR_WorkflowValidator` + the widget-count audit). Unwired code is dead.
- After EVERY code change: full suite (`.venv` python, `$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`)
  + the Bug Bible regression (separate repo, cd + relative `tests\bug_bible_regression.py`). Both green.
- Default-OFF => byte-identical: assert the off-path call-count is unchanged and no new `meta` key
  appears when the flag is off. UTF-8 no BOM, SFW, no "dummy".
- Commit AND push every green ticket to `v2.0-alpha` same session; verify HEAD == origin, AST parse,
  no 0-byte/BOM on touched files.
- Desktop Commander for git/venv/tests; file tools for edits. Reset the box before any headless GPU run
  (selective CIM kill of `ComfyUI*main.py`, never a blanket python kill).

## Overnight execution order (the coder window runs these top-to-bottom)

`T0 (probe) -> T3 (use_exchange GPU check, independent) -> T4 (staging penalty) -> T1 (pitch room) ->
T2 (critic-axes refine)`. T0 and T3 are GPU; T4/T1/T2 are CPU-testable (LLM mocked) and ship dark.
**STOP-and-leave-for-operator points:** (a) T0 writes a recommendation doc -- do NOT auto-enable
frontier; (b) every build flag stays OFF -- the operator reviews + flips in the morning. If a ticket's
acceptance fails, STOP, leave it red with a note, move to the next independent ticket.

---

## T0 -- Local-ceiling probe (GATE; GPU; produces an operator decision)

- **Goal:** answer "can the local model compose a 75+ (B) story at all?" Cheap, throwaway.
- **Build:** a standalone temp script `scripts/_otr_ceiling_probe.py` (delete after; results persist).
  Seed a temp `generate_pitches()` (NOT C1's node) from the raw `script_brief` + divergence seeds ->
  outline -> compose 5 episodes at a REALISTIC word budget (>=200; short stories grade low on pacing --
  if shortened, inject a "short-format test, do not penalize length" note into the `grade_story` prompt
  for the probe ONLY) -> `grade_story`. Grade the 3 best; near the 75 line grade TWICE (pass = both >=75).
  Log model id / seeds / temperature / word budget per episode to a JSONL + a summary.
- **Output:** `docs/2026-06-23-story-architecture/CEILING_PROBE.md` -- the numbers + a recommendation:
  local-viable (some ep >= 75) vs needs-frontier vs accept-B. Name the env the operator would set:
  `OTR_ENABLE_FRONTIER_GREENLIGHT` + `OTR_GREENLIGHT_MODEL` (HKCU pattern). **Do NOT auto-enable frontier.**
- **Acceptance:** the doc exists with real grades; box reset after. No production code touched.

## T3 -- Flip `use_exchange` (GPU; independent; config-only)

- **Goal:** prove the already-built grouped-exchange composer is safe to default ON.
- **Verify first:** the exact `use_exchange` JSON field + that the writer runtime does not override it
  (`OTR_LedgerScriptWriter` INPUT_TYPES ~L1838; canonical workflow node 1).
- **Build:** an N=3 harness that runs the canonical workflow with ONLY `use_exchange=True` (separate run
  from any other change -- one variable). Assert: effective `use_exchange=True` reaches the composer,
  VRAM <= 14.5 GB, zero slot drift (identical slot count/order/ids before vs after).
- **Acceptance:** pass -> a config-only change flipping the JSON default to True + the assertion test,
  suite+BugBible green, commit+push. Fail -> leave OFF, write the failure note, move on.

## T4 -- Deterministic staging penalty (CPU; default None = byte-identical)

- **Goal:** make `select_best_outline` prefer outlines whose climax lands on-mic.
- **Build:** `_otr_staging_penalty(outline) -> float` in `nodes/_otr_story_select.py`: import
  `BEAT_ROLE_IRREVERSIBLE_CHOICE` from `_otr_story_quality_l12`; penalty (e.g. 50.0) when the
  irreversible_choice beat is NOT the final voiced beat (character/announcer) OR has an empty/indecisive
  `intent`; else 0.0. Add optional `penalty: float | None = None` to `score_outline` AND
  `select_best_outline`; subtract from the final score; `None` => byte-identical.
- **Verify first:** audit EVERY `score_outline` / `select_best_outline` caller; add a regression test
  proving identity when `penalty=None`. Compute the penalty INSIDE the best-of-N candidate loop
  (post-outline, pre-composition) so it steers selection.
- **Acceptance:** suite+BugBible green; off-path (penalty None) byte-identical proven; commit+push.

## T1 -- Pitch room + greenlight (CPU-testable; ships dark behind a flag)

- **Goal:** generate 3 forcibly-divergent premises, taste-select one, hand the winner to the outliner.
  THE primary lever (changes WHICH story gets told).
- **Build:** `nodes/_otr_pitch_room.py`. `run_pitch_room(outline_req, *, generate_fn, local_model,
  frontier_cfg, seed_context, meta) -> (OutlineRequest, PitchMeta)`. Called in
  `OTR_LedgerScriptWriter.run()` AFTER news briefs, BEFORE `generate_outline` (the `news_interpreter`
  injected-`generate_fn` pattern). **GATE behind `OTR_ENABLE_PITCH_ROOM` (default OFF) AND/OR a
  `pitch_room` widget -- when off, run() is byte-identical (no pitch call, no `meta.story_quality.pitch`).**
- **Divergence:** each of 3 pitches seeds from `DOMAIN_PALETTE` (the real symbol) + one genre from
  {thriller, drama, sci-fi, noir} + one archetype from {reluctant hero, anti-hero, naive idealist,
  jaded veteran}, in a templated logline prompt.
- **Schema (`structured_call`, Pydantic):** `PitchCandidate(id:int in [1,2,3] unique, logline,
  protagonist, antagonist_or_pressure, genre_mode, emotional_core, theme_sentence, final_20_seconds,
  conflict_type, setting_class, surprise:int, human_want:int, stageability:int,
  console_standoff_risk:int, why_different)`; `GreenlightDecision(selected_id:int, ranking:list[int]
  permutation, rationale)`. Validate selected_id in ids + ranking is a permutation + >= 3 valid
  candidates; <= 2 regenerations then fall back to the original `script_brief` and stamp
  `pitch.status=failed_fallback`.
- **Greenlight:** LOCAL greenlight (same rubric on the local model) is the DEFAULT. When
  `OTR_ENABLE_FRONTIER_GREENLIGHT` + `OTR_GREENLIGHT_MODEL` set, upgrade the taste call to frontier
  (timeout 30s, 1 retry, fail-CLOSED to local). Tie-break: lower `console_standoff_risk`, then lower id.
- **Handoff:** build a concise `script_brief` from the winner via the fixed template
  ("{logline} Protagonist: {protagonist}. Conflict: {conflict_type}, {setting_class}. Emotional core:
  {emotional_core}. Final 20s: {final_20_seconds}."), hard-truncate ~200 tokens, then
  `dataclasses.replace(outline_req, script_brief=brief)` (OutlineRequest is FROZEN). Stamp raw seed +
  full pitch in `meta.story_quality.pitch`.
- **Verify first:** `OutlineRequest` is `@dataclass(frozen=True)` with optional `script_brief` that
  takes precedence (confirmed) + the macro-prompt length tolerance for the richer brief (set the hard
  cap before build, checklist #4).
- **Acceptance:** flag-OFF byte-identical (no pitch call, no meta.pitch key, asserted); flag-ON unit
  tests with a mocked `generate_fn` (divergence, validation, fallback, tie-break, frozen-replace);
  suite+BugBible green; widget (if added) IN the JSON same commit + re-validated; commit+push.

## T2 -- Critic axes drive the refine loop (CPU-testable; ships dark)

- **Goal:** make the 5B critic the pipeline already runs actually buy a better re-plan, instead of the
  refine loop revising against only `grade_story.biggest_weakness`.
- **Build the ADAPTER first** (the grounding catch): a small pure helper that maps the real
  `StoryCriticReport` (`arc_verdict` + escalation router `STRUCTURAL_AXES` decision) -> `failing_axes`,
  and the `reroll_targets[].hint` set -> a `regeneration_hint` string. Persist both to
  `meta.story_quality.critic_failing_axes` / `critic_regeneration_hint`.
- **Wire:** in `_refine_loop` (OTR_LedgerScriptWriter ~L2050), after each pass read the stamped 5B
  output (verify-at-build #2: confirm the per-pass `last` exposes `meta.story_critic_report`; if not,
  add the meta plumbing) and build `prior_critique` from the adapter output, falling back to
  `grade_story.biggest_weakness` when the critic is absent. Same premise; the T4 staging penalty rides
  the re-outline selection. Turn `enable_critic_escalation` ON in the canonical workflow as part of this.
- **Increment-1 premise handling:** `decide_escalation_scope` keeps returning EPISODE for
  `premise_clarity` -- do NOT add `EscalationScope.PREMISE` yet (that is Increment 2; keeps exhaustive
  switches safe).
- **Guard:** add a keep-best **monotonicity smoke test** (the loop is known non-monotonic) so the
  critique-source swap does not worsen the drift.
- **Acceptance:** flag/critic-absent path byte-identical to today's refine loop; adapter unit-tested
  against a synthetic StoryCriticReport; monotonicity smoke present; suite+BugBible green; JSON change
  (enable_critic_escalation) in the same commit + re-validated; commit+push.

## Deferred to Increment 2 (do NOT build overnight)

C2 Tier 2 premise re-pitch (`EscalationScope.PREMISE` end-to-end, `console_standoff` rubric axis,
fingerprints, re-pitch caps). The whole-scene/episode prose->ledger parser spike (real risk = SILENT
speaker mis-attribution -> needs a deterministic-attribution gate, not built tonight). Multi-seed
assignment desk, character interviews, listener-taste critic. Refine-loop non-monotonicity is a REAL
regression (fresh compose each pass) -- revisit AFTER the levers land.

## Morning operator review (what the human decides)

1. Read `CEILING_PROBE.md` -> decide local vs frontier-greenlight vs accept-B; set the env if frontier.
2. Flip the Increment-1 flags on for a live N=3 sameness/grade re-soak (pitch-room ON; use_exchange per
   T3; critic-escalation ON) and eyeball the stories. Promote to default only after that eyeball.
