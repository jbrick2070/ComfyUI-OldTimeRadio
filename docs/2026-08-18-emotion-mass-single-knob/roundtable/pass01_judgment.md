# R1 judgment -- Claude as sole judge

Panel: `openai/gpt-5.6-sol` + `google/gemini-3.1-pro-preview` (both via
OpenRouter). `deepseek/deepseek-v4-pro` FAILED both attempts -- empty content,
`finish_reason=length`, the model spent its whole budget on hidden reasoning.
A first pass at 2000 max-tokens truncated both surviving reviews mid-sentence
and was re-run at 7000; the truncated run is preserved under
`pass01_truncated_2000tok/` rather than overwritten.

Substitution note: this is a CLOUD panel standing in for the local kibitz
lanes. Codex is not installed on this box and Antigravity returned
`RESOURCE_EXHAUSTED (429)` at the account level for every model. Per the
2026-08-17 directive a held lane never blocks the arc.

## ACCEPTED (verified against the real files)

| # | Claim | Source | Grounding |
|---|---|---|---|
| A1 | The audition's `"shipped"` arm hardcodes alpha 0.4 / cap 0.4, so re-qualifying with it would certify the WRONG build | GPT 2, Claude anchor Q6 | CONFIRMED `scripts/otr_lemmy_production_audition.py:58`. Fix is GPT's, and better than mine: the shipped arm CLEARS the env overrides so it can never drift from the constants |
| A2 | Docstring edits move the fingerprint too -- the WHOLE file is hashed, not just the constants | GPT 6 | CONFIRMED `live_engine_impl_version` sha256s the file bytes. Freeze **every** edit to `eng_indextts2.py` before rendering. My anchor said "after the constants land", which was half right and would have produced a stale record |
| A3 | Overwrite protection is incomplete: it guards only `MANIFEST.json`, so WAVs in a manifest-less dir and the whole `_KEY` dir are unprotected | GPT 7, Gemini 2 | CONFIRMED `otr_lemmy_production_audition.py:167-176` -- `key_dir.mkdir(exist_ok=True)` then unconditional `write_text`. Directly strengthens the Bible entry |
| A4 | `effective mass = min(alpha*sum, cap)` is FALSE as stated -- truncation and the 3-decimal floor land under it | GPT 5 | CONFIRMED by measurement: the emotional line gives 0.5590, not 0.5600; the ladder's cap-0.4 rung measured 0.398. State it as an approximation; assert on `emotion_payload()["effective_mass"]` |
| A5 | "Collapse onto ONE knob" is not what the plan actually builds -- the real knob (mass cap) is absent from the profile while alpha stays in it | GPT 4 | CONFIRMED. Add `emo_mass_cap: 0.56` to `char_indextts2_v1.default_params` so the profile declares the knob that matters. Safe: `audio_engine_profiles.yaml` is NOT sha256-pinned anywhere |
| A6 | The acceptance harness will report false failures | Gemini 1 | CONFIRMED and worse than claimed: `scripts/otr_voice_identity_2x2.ps1:53-56` hardcodes `Alpha='0.4'; Cap='0.4'` as the FIX arms, and line 24 documents `--expect-alpha 0.4 --expect-mass-cap 0.4`. The 2x2 would keep proving a build that no longer ships |
| A7 | The manifest cannot prove which runtime made the evidence | GPT 8 | CONFIRMED -- it records clips, seeds and effective mass but no alpha, cap, fingerprint or reference hash. The qualification record cites this manifest BY HASH, so the runtime facts belong inside it |
| A8 | Flipping 8 route assertions is not acceptance for the new behaviour | GPT 9 | CONFIRMED as a gap. Add direct tests: default alpha, default cap, profile/adapter agreement, a production vector binding at the cap, a below-cap vector staying below, env overrides, cache-key movement |
| A9 | "100% of real production lines" overstates a 57-line sample | GPT SF1/SF2 | CONFIRMED. Reword to "100% of the sampled 57". Below-cap vectors stay valid behaviour and get a test rather than being designed away (GPT CUT 3) |
| A10 | Pin alpha, do not delete it | GPT CUT 1, Gemini SF1, Claude anchor SF4 | CONFIRMED unanimously and independently |
| A11 | The stale narrative is wider than two docstrings | GPT SF4 | CONFIRMED -- grep `alpha 0.4`, `ceiling 0.4`, `leaves 0.6`, `0.6 of him` across adapter, YAML, audition prose and arm labels |

## MODIFIED (right instinct, wrong mechanism)

* **Gemini 3 / CUT 1 -- "the synthetic stale record is wasted; pass the real
  superseded 2026-08-10 record to the selection test."** MISREAD of the
  record's SHAPE. The superseded copy is a flat evidence dict (`route_id`,
  `record_id`, `engine_impl_version`, `audition_manifest`, `operator_verdict`)
  with **no `qualification_record` key at all**, so `select_policy_route` would
  reject it for missing structure, not for a stale fingerprint -- the test
  would pass through the wrong path, which is precisely GPT SF5's warning.
  **But the instinct is right**, so: the synthetic record now takes its rotted
  fingerprint FROM the superseded record (`b965453f355661a3`, the value history
  actually withdrew) instead of an invented `deadbeefdeadbeef`. Real history,
  correct structure.

## REJECTED

* **GPT 1 -- "scope 0.560 to Lemmy, or audition a representative set of
  IndexTTS2 characters."** The factual half is CONFIRMED: `OTR_CastLock` and
  `OTR_BatchCharacterVoices` both sit on `indextts2` in
  `workflows/otr_canonical.json`, so this changes every character. But scoping
  it to one character would be a shim, and the operator's recorded verdict is
  explicitly general -- he "volunteered that this is a general complaint, not a
  quirk of this test." A global taste setting is what was asked for. The
  actionable half of the claim is already satisfied: I verified
  `approved_native_routes` carries exactly ONE fingerprint-bound record
  (Lemmy's), so "re-qualify every affected route" means re-qualify his.

* **GPT 3 -- "the A/B confounds the emotion change with the seed change."**
  CONFIRMED as fact (`OTR_VOICE_CHARACTER_SEED` 1 vs 0 across the arms), and
  ACCEPTED as a defect, but not by GPT's fix of dropping the historical
  control. Resolved with a THIRD arm instead, so both variables separate and
  the historical comparison survives.

## VERIFY-AT-BUILD

* Bump `char_indextts2_v1.engine_impl_version` 2 -> 3. Not strictly required --
  `render_time_params` already puts both knobs in the cache key -- but the
  1 -> 2 bump for the seed change set the convention, and it is free.

## Convergence call

Not converged. R1 opened enough real defects that the coding plan has changed
shape, so R2 runs on the revised plan rather than on `pass00`.

**Spend: $0.0906 (truncated run) + $0.1492 (7000-token re-run) = ~$0.2398.**
