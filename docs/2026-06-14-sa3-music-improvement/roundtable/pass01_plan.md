# SA3 Music Quality — CONVERGED PLAN (pass01, grounded)

**Bug:** BUG-LOCAL-408 | **Branch:** `v2.0-alpha` | **Panel:** GPT-5.5, Gemini-3.1-pro, Grok-4.3,
DeepSeek-v4-pro (~$0.17). Claude judged + grounded every claim against the real code. This is the
buildable plan; raw reviews in `pass01/`, accept/reject log in `pass01_judgment.md`.

## Grounded facts (verified this pass — resolve the open ambiguities)
- **SA3 IS the live engine.** `workflows/otr_scifi_16gb_full.json` node 83 `OTR_StableAudioTheme`
  widgets = `['stable_audio_3', 'mono_safe']`. The operator's music IS SA3 — the change will land.
  (Disregard the stale `_LEGACY_FIRST_FALLBACK=("musicgen","stable_audio_music")` default in
  `stable_audio_theme.py`; the SAVED graph pins `stable_audio_3`.)
- **`test_audio_byte_identical` covers the FULL audio bytes.** `tests/test_audio_byte_identical.py`
  re-runs the episode with `FIXED_SEEDS` and sha256-compares to a golden hash. Any SA3 prompt/sampler
  change WILL change the music bytes and FAIL this test. => This work REQUIRES a deliberate, operator-
  gated **golden re-baseline** (capture mode at the bottom of that test file). The "frozen spine" we
  keep is the MUX ARCHITECTURE (mux-LAST, no `-shortest`, byte-identical *assembly*), NOT the literal
  music content — changing the music is the whole point.
- **Caller contract:** `compose_music_prompt(meta, cue_id) -> (str, int)`; `_render_clips` unpacks
  `prompt, duration_s`; `generate_clip(prompt, duration_s, seed)` internally builds `neg=encode("")`,
  `ConditioningStableAudio().append(pos, neg, 0.0, dur)`, `EmptyLatentAudio().generate(dur, 1)`,
  `KSampler(..., 100, 6.0, "dpmpp_3m_sde_gpu", "exponential", ..., 1.0)`. The legacy musicgen consumer
  also imports `compose_music_prompt`. **Do NOT change its return shape.**

## The fix — three coordinated, grounded changes

### A. Structural conditioning (the headline fix — Gemini, grounded) — `eng_stable_audio_3.py::generate_clip`
The 4-12s cues sound unstructured because `seconds_total == dur` tells SA3 the whole song is 4-12s.
Fix WITHOUT trimming or changing length/determinism:
- Keep `EmptyLatentAudio().generate(dur, 1)` at EXACTLY `dur` (latent size unchanged → length +
  determinism contract preserved; seed stays the sole carrier).
- Set the `ConditioningStableAudio` `seconds_total` to a fixed structural context `SA3_CONTEXT_S = 30.0`
  (env `OTR_SA3_CONTEXT_S`), NOT `dur`.
- Set `seconds_start` PER CUE so the clip is a real slice of a 30s arc:
  - opening → `seconds_start = 0.0` (intro/build)
  - interstitial → `seconds_start = (SA3_CONTEXT_S - dur) / 2` (a middle, unresolved bridge)
  - closing → `seconds_start = SA3_CONTEXT_S - dur` (the resolving outro/tail)
  Derive the cue from the cue_id, not by sniffing prompt text. Plumb `cue_id` (or a small
  `cue_window` arg) into `generate_clip` — its only caller is `_render_clips`, so extend the call
  site. Append `seconds_start, seconds_total` to the signature with safe defaults (0.0, dur) so the
  legacy/other callers are unaffected.

### B. SA3-shaped prompt + real negative prompt — keep `compose_music_prompt` signature; shape in SA3 only
- The POSITIVE prompt stays produced by `compose_music_prompt` (signature unchanged), but ADD an
  SA3-only augmentation that prepends a deterministic **genre + instrumentation + production** clause
  derived from the cue + the period/setting the brief already carries. Put this in a new pure helper
  (e.g. `sa3_music_prompt_augment(meta, cue_id, base_prompt) -> str`) called inside `generate_clip`
  (or just before it in `_render_clips` for the SA3 engine), NOT by changing `compose_music_prompt`'s
  return shape. CAP length (GPT): one genre phrase, ≤5 instrument/production tags, ≤3 mood terms, one
  cue-arc phrase, keep the existing `, instrumental only, no dialogue, no vocals` tail.
- **Deterministic period→genre/instrument map** (static, code-side — not "poking meta"; cue/period
  derived). Example rows (final map in the coder ticket): contains `1950|radio|atomic|pulp|sci-fi` →
  `"vintage orchestral sci-fi score, theremin, eerie strings, brass, timpani, analog tape warmth"`;
  default → `"cinematic instrumental underscore, small orchestra, low strings, soft brass, light
  percussion, analog tape warmth"`. Do NOT fabricate a musical KEY from noisy brief fields (panel
  consensus); use a simple cue/mood default (minor for opening/interstitial; minor→major resolve
  language for closing) or omit key.
- **Negative prompt** lives as a constant INSIDE `generate_clip` (replacing `encode("")`). Use TARGETED
  negatives that don't kill the eerie/tape texture (GPT — avoid blanket "dissonant"/"noisy"):
  `"vocals, singing, speech, spoken words, lyrics, voiceover, crowd noise, harsh clipping, digital
  distortion, muddy mix, out of tune, low quality"`. Env-overridable `OTR_SA3_NEG_PROMPT`.

### C. Sampler inputs — one-line constants, made traceable (no new config surface)
- The panel had no SA3-proven numbers; current `steps=100, cfg=6.0, dpmpp_3m_sde_gpu, exponential,
  denoise=1.0` are unverified for SA3. Treat these as the PRIME A/B knobs. For the first build keep them
  as named constants (so a one-line tweak is possible) and expose env overrides
  (`OTR_SA3_STEPS/CFG/SAMPLER/SCHEDULER`). A conservative starting point to A/B: `cfg 6.0→7.0`,
  steps 100 (SA3 small tolerates fewer; try 50 in the sweep). Decide final values from the listen test.
- Add a runtime guard that the sampler/scheduler names exist in the installed ComfyUI registry before
  render (fail LOUD, not a silent bad-name). Log the final `steps/cfg/sampler/scheduler` + a hash of the
  positive+negative prompt into `render_log` so each A/B clip traces to exact settings (no determinism
  change).

## Decisions (converged — close the open questions)
- **Durations:** keep `CUE_DURATIONS = {12,8,4}` (latent length unchanged). Structure comes from the
  `seconds_total`/`seconds_start` conditioning (A), not from rendering-longer-and-trimming (rejected:
  breaks length/determinism — Grok/GPT).
- **Model:** keep `stable_audio_3_small_music.safetensors` default; larger checkpoint stays opt-in via
  the existing `OTR_SA3_CKPT` env ONLY after local availability + 16GB residency are verified. No new
  download in this build (cut — all 4 models).
- **Seed:** strictly SINGLE per-cue seed (existing `_seed_to_int64(...,slot)` under
  `deterministic_inference`). Best-of-N CUT (all 4 — no headless persistence, breaks reproducibility).

## Constraints / invariants
- Keep `compose_music_prompt(meta, cue_id) -> (str,int)` signature (legacy musicgen consumer).
- Determinism: latent length = cue duration; seed-int the sole carrier; render-twice waveform compare
  in the smoke.
- **Golden re-baseline is REQUIRED + operator-gated** (this change intentionally alters music bytes).
  Sequence: land A/B/C → operator listens + approves → re-capture the `test_audio_byte_identical`
  golden (capture mode) in the SAME commit → test green again on the NEW golden. Until approved, expect
  that test RED (it is the intended change, not a regression).
- 100% local, no new pip dep, ≤14.5GB (small model is light), UTF-8 no BOM / ASCII in
  `_otr_music_prompt.py`, SFW.

## Test / verify plan
- Build-time smoke (mockable): `generate_clip` receives non-empty positive + non-empty negative + int
  seed + expected `dur`, and returns `{"waveform","sample_rate"}`; assert the prompt still ends with
  `_PROMPT_TAIL`.
- Determinism: render a cue twice with the same seed → waveforms identical.
- Sampler-name guard test (names exist in registry).
- A/B LISTEN (operator): 3 cues (opening/closing/interstitial), OLD vs NEW prompt+conditioning, fixed
  seed, blind pick. Source the verdict from the operator; then re-baseline the golden.
- Full suite + Bug Bible green (excluding the intentionally-rebaselined `test_audio_byte_identical`
  until the operator approves the new golden).

## CUT (panel consensus — do not build)
Render-longer-then-trim; best-of-N seeds; larger-model migration; a new sampler config surface / grid
search in the pipeline; elaborate musical-key derivation from brief fields; changing
`compose_music_prompt`'s return shape.
