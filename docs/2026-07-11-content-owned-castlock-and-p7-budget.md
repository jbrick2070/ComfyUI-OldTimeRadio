# 2026-07-11 -- Content-owned CastLock seed contract + P7 envelope echo / output cap

Status: PROPOSED FIX (pre-code). Grounded against the real Windows tree at
commit `e679b754` (v2.0-alpha) and the live canonical 30w Codex roll 12
(server log `C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log`, 2026-07-11 08:10-08:25).

## Context: what is already fixed and proven live

PBUG-20260711-13 (Codex P5 typed repair retained legacy metadata) is FIXED and
PROVEN LIVE at `e679b754`. In roll 12 the P5 base call again emitted
`boundary="beat_end"`; `repair_script_artifact_metadata` normalized it from the
accepted score graph and the ladder logged
`repair factory resolved the failure deterministically; no LLM repair call made`.
Codex then cleared P6-P9, stamped 13 delivery lines, and entered the tail.

Suite at that commit: 7581 passed / 31 skipped / 1 xfailed. Bug Bible: 17 passed.

## Defect A (BLOCKER) -- content-owned lane dies in CastLock

### Live trace
```
[delivery-stamp] stamped=13 (delivery-differs=0)
!!! Exception during processing !!! num_characters must be 1-6, got 0
  cast_lock.py:189   lock -> self._assign_bark_voices(cast, meta, report)
  cast_lock.py:353   _assign_bark_voices -> _OTRCAST.replay_voice_assignment(...)
  _otr_casting.py:1029  replay_voice_assignment -> assemble_pre_locked_rows(...)
  _otr_casting.py:1211  ValueError: num_characters must be 1-6, got 0
Prompt executed in 00:14:27  (FAILED -- no obs_publish)
```

### Root cause
`cast_lock._assign_bark_voices` treats `meta.cast_contract.cast_seed` as the
signal "the writer's seeded cast picker produced this cast, so replay it":

```python
contract = (meta or {}).get("cast_contract") or {}
cast_seed = contract.get("cast_seed")
if cast_seed is None:
    ... # preserve voice_preset, no replay   <-- the escape hatch
num_characters = int(contract.get("num_characters_request") or 0)
voices = _OTRCAST.replay_voice_assignment(cast_seed=..., num_characters=..., ...)
```

Content-owned lanes (scifi_codex / gemini / sonnet / fable2) build their OWN cast
rows and stamp their own presets in the lane runner
(`_otr_scifi_codex._assemble_ledger`: announcer -> kokoro/`bm_george`;
c01/c02/c03 -> bark `v2/en_speaker_6` / `_3` / `_0`). They never run the writer's
picker, so there is no `num_characters_request` in the contract and nothing to
replay. Historically they fell through the `cast_seed is None` escape hatch.

The credits fix that landed in `e679b754` (for the earlier PBUG-20260711-15
credits receipt failure) stamped `cast_contract.cast_seed` for content-owned
lanes -- which CLOSED that escape hatch and made CastLock attempt a replay of a
cast that was never rolled: `num_characters_request` absent -> `0` -> ValueError
deep inside `assemble_pre_locked_rows`.

So the stamp asserted something FALSE. `cast_contract.cast_seed` is not a generic
"episode seed" receipt -- it is a claim of *replayability*.

### Proposed fix (two edits, both at the real boundary)

1. `nodes/OTR_LedgerScriptWriter.py` (shared writer tail, content-owned branch):
   stamp `meta["episode_seed"]` ONLY. `otr_credits_roll.py:279-284` already
   accepts either receipt:
   ```python
   seed = (meta.get("cast_contract") or {}).get("cast_seed", meta.get("episode_seed"))
   if seed is None: raise CreditsDataError(...)
   ```
   so `episode_seed` satisfies the no-fallback credits provenance contract
   WITHOUT claiming a replayable writer cast. Do not write `cast_contract.cast_seed`
   / `cast_seed_source` / `cast_contract_version` for a lane that owns its cast.

2. `nodes/cast_lock.py::_assign_bark_voices`: do not replay the writer picker for
   a lane that owns its own cast. Use the SAME family discriminator the freeze
   cascade and the voice lane already use --
   `_otr_text_delivery.delivery_mode_for_meta(meta) == CONTENT_OWNED` -- to
   preserve the lane's `voice_preset` values, and STILL run the Gate 1 invariants
   (`_assert_unique_bark_voices`, `_assert_voice_preset_invariant`) so a
   content-owned lane can never ship duplicate or non-`v2/` bark voices.
   This is defense in depth: even with edit 1, a future lane that stamps
   `cast_seed` must not be able to detonate the same way.

Grounding check: the codex announcer row is locked to the literal name
`ANNOUNCER` (`_validate_cast_plan`), which is exactly the row both Gate 1
invariants exclude, and c01-c03 carry unique `v2/*` presets -- so running the
invariants on a lane-owned cast passes on real data and still catches a future
regression.

### Explicitly NOT the fix
- Do not default `num_characters` to the observed cast length in
  `assemble_pre_locked_rows` -- that fabricates a replay of a picker sequence
  that never ran and would silently overwrite the lane's chosen voices.
- Do not drop the credits receipt requirement.

## Defect B (LATENT, will bite at 720w) -- P7 echoes the request envelope and hits the output cap

### Live trace
```
[OTR_LedgerScriptWriter] OUTPUT_CAP: prompt_tokens=4543 generated_tokens=2800 max_new_tokens=2800
'scifi_codex:P7' attempt 1 failed: no decodable top-level JSON object found: line 1 column 1 (char 0)
  | raw head: { "artifact_inputs": { "accepted_line_count": 13, "accepted_line_graph": [ {...
'scifi_codex:P7' attempt 2/3: structural retry at temperature=0.300 -> PASSED
```
Roll 12 survived only because the structural retry happened to comply.

### Root cause (two compounding faults)
1. The model re-emitted the *request envelope* (`{"artifact_inputs": {...}}`)
   instead of the ScriptArtifactV4 root. `_SCRIPT_ARTIFACT_ROOT_INSTRUCTION`
   forbids returning "a score, a scene, a beat, or a patch" but never forbids
   echoing the input envelope keys (`pass_id`, `artifact_inputs`,
   `result_json_schema`).
2. `_script_output_token_budget(requested_words)` scales the reservation from the
   WORD STEER only: `min(5400, max(2800, words*4.5 + 1200))`. But a
   ScriptArtifactV4's size is dominated by the per-line metadata of the accepted
   line graph (13 lines x ~11 fields), not by the dialogue word count. A 30-word
   and a 720-word script can carry the same line count; the budget does not know
   the line count at all.

### Proposed fix
1. Extend the whole-script root contract to name the forbidden envelope keys and
   require the response to begin at the artifact root
   (`{"schema_version": "scifi_codex.script_artifact.v4", ...}`).
2. Make `_script_output_token_budget` a function of BOTH the requested words and
   the accepted line count (the real driver of serialized size), keeping the
   existing floor/ceiling behavior and the token-budget receipts. Fail loudly if
   the reservation cannot fit the accepted graph inside the context cap rather
   than silently truncating.

## Acceptance
1. Focused tests for both defects (content-owned CastLock preserve+verify path;
   whole-script budget scales with line count; root contract forbids envelope echo).
2. Full suite + Bug Bible green.
3. Canonical Codex 30w publishes to `otr\obs\` (`obs_publish OK`, asset Test-Path).
4. Gemini + Sonnet 30w publish.
5. Only then the 720w bake-off across the media packs.
