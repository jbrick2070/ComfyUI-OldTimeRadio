# Ticket: Spine / Outline ledger-stamping + structural validation

**Filed:** 2026-04-29
**Status:** Scheduled, NOT deferred
**Branch base:** v2.0-alpha
**Owner:** Jeffrey

## Why this ticket exists

Today's prompt-engineering audit (commit-shipping concurrent with this file)
confirmed that the structured Outline / Open-Close Spine output produced by
`_open_close_expansion()` in `nodes/story_orchestrator.py` is consumed in
exactly two places:

1. The body user_prompt (line ~4040), as text spliced into the LLM's writer prompt
2. The auto-title generator (line ~5303)

It is NEVER stamped to the ledger. There is no `ledger.outline`,
`ledger.beats[]`, or `ledger.spine_meta` field. SceneSequencer cannot
validate parsed prose against a structural contract because no contract
exists in the ledger -- it parses prose and trusts the parse.

This is the architectural lever the prompt-review audit identified as the
real win. Today's voice-consistency soft-warnings collection (in the same
commit as the audit findings) is the data-gathering precursor. Once 2-3
real-episode runs have produced `ledger.voice_warnings[]` data, this
ticket unblocks: we will know empirically how often the LLM drifts
versus how often it complies, which sizes the structural validation work.

## Scope (concrete)

### Schema changes

- Bump `nodes/_otr_ledger.py` `CURRENT_SCHEMA_VERSION` from `l3-2026-04-28` to `l4-YYYY-MM-DD` (date-stamped at ship time)
- Add three new top-level ledger fields:
  - `outline` (string): the winning outline / spine text returned from `_open_close_expansion()`. ~500-1500 chars typical.
  - `beats[]` (array of dicts): one entry per scene-level beat, shape:
    ```json
    {"beat_id": "b001", "scene_id": "s01", "summary": "...", "characters_present": ["c01", "c02"], "expected_dialogue_lines": 4}
    ```
  - `spine_meta` (dict): structured open/close skeleton, shape:
    ```json
    {"open_hook": "...", "close_payoff": "...", "character_arcs": [{"char_id": "c01", "start_state": "...", "end_state": "..."}]}
    ```

### story_orchestrator.py changes

- Inside `_open_close_expansion()` after the winner is selected, parse the winning_outline into structured beats via a small LLM-side JSON-emit pass (cleanup_model_id, low temperature). Stamp `ledger.outline`, `ledger.beats[]`, `ledger.spine_meta` before return.
- Migration path: if outline structuring fails, save raw `outline` string and leave `beats[]` empty. The legacy ledger merge mechanism preserves arbitrary keys so legacy ledgers without these fields continue to load.

### SceneSequencer changes

- After parsing the body script, validate parsed scenes against `ledger.beats[]`:
  - Soft warning if a beat's `characters_present` set has zero matching dialogue lines
  - Soft warning if `expected_dialogue_lines` deviates by >50% from actual count for that beat's scene
  - Soft warning if scene order in script differs from beat order in spine
- Warnings stamped to `ledger.structural_warnings[]` (same pattern as today's `voice_warnings[]`)
- Hard error reserved for: scene markers in script that have no corresponding beat in spine (i.e., LLM invented an extra scene)

### Test coverage

- New `tests/test_spine_ledger.py` — round-trip a sample winning_outline through the structurer, assert beats[] contract
- Extend `tests/test_production_ledger.py` schema bump assertion (`l3-` → `l4-`)
- Extend `tests/test_core.py` SceneSequencer regression with a beats[] mismatch fixture

### Bug Bible candidate

Yes. Generalisable rule: **structured-pass output that ONLY lives in the LLM context window is a wasted artifact; if a downstream phase needs to validate against it, the structured output must be stamped to the durable ledger.**

## Unblock conditions

Reopen this ticket when **all three** are true:

1. The voice-consistency soft-warnings collection (shipped 2026-04-29) has accumulated `ledger.voice_warnings[]` data on at least **2-3 real-episode runs**. The data shape after 2-3 runs answers: do RP fine-tunes drift more than base-instruct? Are mismatches concentrated in (gender, age) or (tone, energy)? This sizes the structural validation work.
2. Mistral-Nemo + Gemma 4 E4B both have at least one PASS verdict in the LLM edge-case matrix (`docs/2026-04-29-llm-edge-case-matrix.md`). We do not refactor the spine path on top of an unvalidated base model.
3. v2.0-alpha video stack is feature-complete and a full episode renders end-to-end without surfacing any new BUG-LOCAL-1xx in the audio path. Schema bumps during active video work increase the risk surface.

## Estimated cost

- Schema bump + helper additions: 1-2 hours
- SceneSequencer validation: 2-3 hours
- Test coverage: 1-2 hours
- Round-robin consult before kickoff (per CLAUDE.md, schema bumps qualify): 30 min

Total: half a day to a day. Not a 30-min change.

## Why this is NOT deferred

This is a real architectural lever. The prompt-review audit identified it
as the highest-value change in the writer-prompt surface. It is on the
roadmap, just sequenced behind the data collection that lets us size it
correctly.

Today's smaller work (per-character voice priming + AISM bullets +
voice_warnings collection) ships in the same commit as this ticket file,
and the voice_warnings data starts accumulating immediately on the next
real-episode run.

## References

- Prompt-review audit: chat transcript 2026-04-29 (round-robin synthesis)
- Today's commit (voice priming + AISM + voice_warnings): see git log on v2.0-alpha after this ticket file is committed
- Existing two-LLM split feature: nodes/story_orchestrator.py `cleanup_model_id` widget; `_resolve_cleanup_model_id` resolver in tests/test_two_llm_split.py
- Existing live ledger streaming: nodes/story_orchestrator.py L1.5 hook (search "live_ledger=True")
