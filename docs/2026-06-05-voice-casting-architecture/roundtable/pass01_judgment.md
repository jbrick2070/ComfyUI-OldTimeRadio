# Pass 01 judgment log

Panel: GPT-5.5 / Gemini-3.1-pro / Grok-4.3 / DeepSeek-v4-pro (OpenRouter). Spend
$0.29. All four: "not build-ready" -- consensus that the casting->engine contract
has real gaps. Judge: Claude (Opus), grounded against the real files.

## CONFIRMED (grounded, folded into the plan)
- `preserve_ledger` default blocks the promoted IndexTTS2 -> the bark you saw
  (`cast_lock.py` _CAST_POLICIES default). [all 4] -> fixed at the workflow level.
- `_stamp` sets `voice_ref_id` but not `voice_ref_path`; clip-engine dispatch
  reads `voice_ref_path` (`cast_lock.py:390` vs `_otr_voice_node_common.py` ref
  resolution). [DeepSeek, GPT]
- `eng_indextts2.commercial_clean=False` (bilibili license, line 47) vs CC0 bank
  refs `=true`; request uses `profile.commercial_clean`. Effective = engine AND
  ref. [GPT, Grok] -- verified the adapter flag directly.
- Kokoro `ANNOUNCER_VOICE_POOL` has 4 voices, only `bm_george` installed;
  `begin_episode` random-picks + os.path.exists-checks (`eng_kokoro.py:26,85`).
  [GPT] -- verified directly.
- Resample only covers the bark fallback; primary clips appended unchanged.
  [GPT, Grok, Gemini] -- matches the shipped fix's scope.
- `_OTR_CLONE_ENGINES` hard-coded tuple + membership branches violate the
  model-agnostic invariant. [GPT]
- Gender re-derived at render with no population guarantee (the observed
  bark-despite-refs). [all 4]

## CORRECTED / SCOPED
- "Flip the global default policy in code" [Grok, DeepSeek]: I flipped it at the
  WORKFLOW level (node 80) instead -- same effect for your runs, lower blast
  radius, keeps the byte-safe `preserve_ledger` default available for the frozen
  Bark legacy path (which the plan and tests rely on). GPT's "auto_registry only
  when the engine requires a ref" is the right code-level form if we change the
  schema default later.
- "Collapse to a single canonical voice_ref_id + resolver" [GPT] vs "the mixed
  voice_ref_field contract is fine, just always stamp" [Grok, DeepSeek]: I sided
  with the latter -- stamp `voice_ref_path` for clip engines (MUST-FIX 1); less
  indirection, no determinism gain from the full collapse.

## UNVERIFIED -> verify-at-build
- Gemini flagged the indextts2 worker hard-crashes if torch prints a warning to
  stdout before the readiness JSON line (`eng_indextts2.py` stdout readline ->
  json.loads). Plausible and worth a guard, but not grounded this pass; noted.
- DeepSeek's ASSUMPTION that `legacy_first_engines("char_voice")` returns
  indextts2 first -- the auto_registry target-engine pick depends on it; verify in
  `_otr_engine_profiles`.

## Convergence
One pass. The panel agreed on shape + gaps; grounding closed the specifics. The
immediate fix is applied; the rest is a reviewed backlog (each item needs a
ComfyUI restart + live render), so a second fan-out would be premature -- harden
the synthesized plan only if you want adversarial review before building it.
