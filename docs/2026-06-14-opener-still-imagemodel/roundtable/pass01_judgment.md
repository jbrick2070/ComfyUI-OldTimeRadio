# Roundtable pass01 judgment — opener-still / image-model / title-card

Panel: Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro (pass01) + GPT-5.5 (pass01b_gpt,
re-run at 12k tokens — the 2k cap made the 3 reasoning models return empty).
Spend: ~$0.27 (pass01) + ~$0.14 (gpt) = ~$0.41. Claude = sole judge/grounder.
**CONVERGED in one grounded pass** (all 4 agreed on the architecture; grounding
flipped FIX 1 off the manifest and refuted the BUG-405 dispatcher fork).

## ACCEPTED (CONFIRMED against the real code)
- **FIX 1 seam was wrong (DeepSeek+GPT+Gemini, grounded).** `build_clip_manifest`'s
  `init_image` is not read by the composite (`plan_timeline_segments` reads clip
  path/start_s/tfc/exists). Real seam = `build_request_from_shot`. GROUNDED ROOT
  (Claude): `flux_still.family=="static_image_gen"` ∉ `_SCENE_INIT_FAMILIES`
  (L413) and ≠ ltx_video, so neither scene-still branch fires → opener black.
- **BUG-405 is not a fallback bug (GPT, grounded).** Dispatcher SKIPS on
  `assert_usable` failure; no silent flux fallback. ⇒ policy carried flux_gen1.
  Capture the policy first; fix is workflow-config (saved JSON widgets) and/or
  registry/weights, LOUD on missing. Cut the dispatcher rewrite.
- **FIX 3 concrete (DeepSeek+GPT, grounded).** Call
  `otr_shot_lock.overlay_audio_timing(led)` in `SignalLostVideoRenderer.render_video`
  after `load_ledger`, before `_resolve_title_timing`. Remove/guard the 1s envelope
  fallback. Verify the OTR_SignalLostVideo `script_json` wiring (pre vs post audio)
  — may be a JSON wiring change, not code-only.
- **Defensive (GPT):** make the `plan_timeline_segments` `all()`-positioned flip
  LOUD in the composite report (latent guard, non-blocking).
- **CUTS (consensus):** no second music_open line; no dispatcher rewrite; no
  composite-floor title card.

## REJECTED / DOWNGRADED
- DeepSeek "the still image itself may be black" → downgraded to a verify step
  (the still is minted by flux_gen1; the black is the missing init wiring, not a
  black PNG). Cheap to confirm on disk.
- "Synthesize a music_open line in EpisodeAssembler" (the PRIOR roundtable's
  converged plan) → REJECTED: ShotLock already makes the beat; a start_s=0 line
  breaks `derive_opening_music_beat`. The prior pass had misread the code.

## VERIFY-AT-BUILD (unverifiable from excerpts)
- `_still_index` key for the opener still (`b000_music_open` vs `still_b000_…`).
- Actual `image_policy_json` contents at render (flux_gen1 vs lumina/qwen/hidream).
- `_inprocess_gen_fn` dispatch-by-engine_id (GPT-claimed).
- `OTR_SignalLostVideo` `script_json` wiring; post-audio `speaker_role` values.

## Convergence
One grounded pass; the panel agreed on the architecture and grounding resolved the
two material misreads. A second paid pass would not add material direction — STOP.
Hardened plan: `pass01_plan.md`.
