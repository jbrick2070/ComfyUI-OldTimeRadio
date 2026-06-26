# DRIFT BUILD SPEC -- story-ledger integrity (R1-converged, Claude-judge-authored)

Converged at R1 (panel gpt-5.5 + gemini-3.1 + deepseek-v4, ~$0.15; operator steered
to lean on code-grounding over more panel passes). Build order = priority. Guards
are DETERMINISTIC + offline + CI-runnable (an accuracy guard must never be an LLM
that fail-opens). Each chunk: regression suite + Bug Bible green, commit+push to
v2.0-alpha.

## CHUNK 1 -- kill the critic FAIL-OPEN (`nodes/_otr_story_critic.py`)
Grounded: `ArcVerdict = Literal["strong","uneven","flat","mid_collapse"]` (line 238);
`arc_verdict: ArcVerdict = "strong"` (261); `clean()` (~189-197) + the exhaust path
(~445-455) return "strong" on failure -> a crash reads as a masterpiece.
- Extend the Literal -> add `"unverified"` (line 238). Low-risk (Literal, no
  exhaustive match) -- grep consumers that branch on `arc_verdict` and handle the
  new value.
- `clean()` returns `arc_verdict="unverified"` (NOT "strong").
- Stamp `meta.story_critic_status = {"ran": bool, "validated": bool, "failure": str}`
  in the writer/freeze meta (outside the frozen report -> schema stays put).
- `_otr_freeze_cascade.py`: map `"unverified"` -> a NON-clean freeze verdict
  (observable + restampable, NOT a hard ship-block -- don't gate on a flaky LLM).
- A3 mechanical floor (~567-590): if it appends anti-loop reroll targets to a report
  whose `arc_verdict` is `strong`/`unverified`, deterministically downgrade to
  `"uneven"` (a report can't be clean AND demand rerolls).

## CHUNK 2 -- deterministic CROSS-STAGE consistency (THE core drift fix) -- NEW
No canon-parity test exists today (that is why the sound_palette class shipped).
`StoryContract` + `CastLock` are importable (defined in `OTR_LedgerScriptWriter.py`
/ `_otr_story_select.py`).
- `nodes/_otr_ledger_consistency.py`: a PURE function
  `assert_ledger_consistency(contract, outline, castlock, canon, ledger) ->
  list[Defect]`. Drives off an explicit SOURCE-OF-TRUTH matrix
  (`field | source | canon/ledger path | normalizer | required?`). Min rows:
  `sound_palette<-contract.sound_world`, title, premise, setting, time_of_day,
  `style<-contract.slug`, cast ids/names/roles `<-CastLock`, outline beat ids,
  line `beat_id`. No LLM.
- Call it at PRE-FREEZE: in test/CI mode -> raise; in production -> LOUD warn +
  `meta.consistency_status` (never silently pass).
- `tests/test_ledger_canon_parity.py`: reflect the `StoryContract`/`CastLock`
  pydantic models; assert every non-optional field has a populated mapped canon/
  ledger equivalent. + a GOLDEN fixture for the sound_palette regression.
- (verify-at-build: contract+outline+CastLock are in scope at the freeze call site.)

## CHUNK 3 -- CI drift guards
- **Widget positional drift (BUG-LOCAL-097).** `tests/test_workflow_json_guardrails.py`
  today checks widgets_values TYPING + link integrity + stale dropdown literals, but
  NOT positional order vs LIVE `INPUT_TYPES`. ADD a test that imports the node
  classes and zips each node's `widgets_values` against its `INPUT_TYPES` widget
  order, failing on any non-append misalignment. (Extend that file; don't duplicate.)
- **Schema-version drift.** A vintage-`l3-2026-05-14`-ledger fixture test: any field
  whose default changes SEMANTICS must fail-loud or have a deterministic derivation,
  never silently default to wrong.

## CHUNK 4 -- freeze WARN taxonomy (`nodes/_otr_freeze_cascade.py`)
Stop labeling a shipped arc/critic failure "structural." Define:
`structural_error` -> BLOCKS at Phase 10; `story_accuracy_warning`
(continuity / `unverified` / canon-divergence) -> ships ONLY as non-clean with
operator-visible meta; `cosmetic_warning` -> clean-with-warns. Wire critic findings +
gap-audit warns into this taxonomy instead of raw warn counts.

## CHUNK 5 -- make the critic actually WHOLE-story + cut dead telemetry
- `_critic_character_lines` (line ~394) filters `speaker_role=="character"` ->
  announcer/music/SFX/title framing are invisible, yet drift lives there. Give the
  critic READ-ONLY context for ALL story-bearing lines; keep `reroll_targets`
  character-only (the post-validator already rejects non-rerollable targets).
- Pass the original outline `beat_intent` into `_render_critic_user_prompt` so a
  Script-Doctor rewrite (doctor runs BEFORE the critic) that drifts off the beat is
  caught.
- CUT `StanceIssue` (~150-166): self-described "TELEMETRY ONLY / dead-end repair
  path" -- delete the model + remove from the prompt (frees the critic's attention
  budget).

## CUT (do not build)
- Multi-LLM voting for binary gates (conflicts with the deterministic-guard
  invariant; LLMs advisory only).
- An LLM "positive-evidence engine" second-guessing the critic (the status stamp +
  parity test suffice).
- Reopening the settled binary-leak lane / leak gates / structured-call tolerance.

## Invariants
Deterministic offline CI guards; ledger schema frozen except the one `ArcVerdict`
enum-value add; no workflow-JSON node churn; byte-identical audio spine + canonical
happy path untouched; UTF-8 no BOM; SFW. Sequence: 1 -> 2 -> 3 -> 4 -> 5.
