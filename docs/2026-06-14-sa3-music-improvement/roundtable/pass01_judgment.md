# pass01 judgment — SA3 music (Claude as judge)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro (~$0.17). Verdicts: GPT no, DeepSeek no,
Grok no, Gemini yes-with-fixes — all because pass00 was a problem-statement of QUESTIONS, not a
buildable plan. The grounded pass turned them into concrete answers. CONVERGED → stop (no new material
unknowns; remaining sampler-value tuning is an empirical A/B authoring task, not a spec gap).

## ACCEPTED (grounded → folded into pass01_plan.md)
- **Gemini M1/M2 — `seconds_total` context + per-cue `seconds_start`, keep latent=dur.** The headline
  fix. CONFIRMED vs `generate_clip` (it passes `dur` to both `ConditioningStableAudio.seconds_total`
  and `EmptyLatentAudio`). Improves structure without trimming or breaking length/determinism.
- **GPT#2 / Grok#4 — keep `compose_music_prompt` signature; put the negative prompt + SA3 shaping
  INSIDE `generate_clip`.** CONFIRMED: signature `(str,int)`, `_render_clips` unpacks `prompt,
  duration_s`, legacy musicgen also imports it. Changing the return shape would break callers.
- **GPT#4 / DeepSeek SF#1 / Grok#2 — verify the live engine.** GROUNDED THIS PASS: JSON node 83 =
  `stable_audio_3`. Ambiguity resolved → the change lands. (The stale `_LEGACY_FIRST_FALLBACK`/docstring
  in `stable_audio_theme.py` should be corrected for UI honesty — folded as a cleanup.)
- **GPT#6 / DeepSeek#7 — `test_audio_byte_identical` scope.** GROUNDED: it sha256s the full episode
  audio vs a golden. So this work REQUIRES a deliberate operator-gated golden re-baseline. Folded as a
  hard sequencing constraint (the biggest correction to pass00's "keep it green" framing).
- **GPT#3 / Grok#1 — the §2 brief-reader constraint was overstated.** CONFIRMED: `compose_music_prompt`
  reads `setting/atmosphere/script_brief/period_voice` DIRECTLY and only uses `_read_brief_field` for
  `music_mood_terms`. Relaxed the constraint; SA3 genre/instrument shaping is static cue/period-derived
  text (not "poking meta").
- **GPT SF#1/#2/#3/#7 — cap prompt length; deterministic period→genre fallback map; targeted negatives
  (avoid blanket "dissonant"/"noisy"); log sampler+prompt hash in render_log.** All folded.
- **Cut best-of-N; cut larger-model migration; keep small_music** (unanimous). Folded.

## REJECTED / DOWNGRADED
- **"Render 30-45s then trim"** (my pass00 §3.C option; some panelists entertained it) — REJECTED:
  Grok#3 + GPT#5 correctly note it changes latent size + the determinism contract. Gemini's
  conditioning-only approach supersedes it (same structural benefit, no trim).
- **Grok#1 "only reorder existing fields, add nothing"** — DOWNGRADED: too strict. Adding STATIC
  period→genre/instrument/production descriptors is deterministic and code-side, not a brief-protocol
  violation. Kept the richer template but with GPT's length caps + no fabricated key.
- **Elaborate musical-key derivation from brief fields** (implied by pass00's template) — CUT (GPT/Grok):
  use a simple cue/mood default or omit key.

## UNVERIFIABLE → verify-at-build (in the coder ticket)
- SA3-correct `steps/cfg/sampler/scheduler` values — no panelist had proven numbers; treat as the prime
  A/B knobs, keep as env-overridable constants, decide from the listen test.
- Whether `_read_brief_field` supports nested paths (`story_brief_terms.setting`,
  `gen_params_initial.period_voice.descriptor`) — only matters if we route more fields through it.
- Whether a larger ungated SA3 music checkpoint exists + fits 16GB — out of scope this build (opt-in
  via `OTR_SA3_CKPT` only).
