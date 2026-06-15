# OTR opener-still + image-model + title-card — HARDENED fix plan (roundtable pass01)

Converged in ONE grounded pass. Panel: Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro,
GPT-5.5 (3 reasoning models returned empty at the 2k cap; GPT re-run at 12k = OK).
Claude grounded every claim against the real code; only CONFIRMED items below.
Raw reviews: `pass01/` + `pass01b_gpt/`. Judgment: `pass01_judgment.md`.

## Headline (grounded)
- **BUG-403 placement is NOT a bug** (instrumented: positioned=True, b000 clip at
  `[0,9.5s)`); the earlier all-black opener was a BUG-402 casualty (now fixed).
- **The pass00 FIX 1 seam was DEAD CODE** (panel catch): `build_clip_manifest`'s
  `init_image` field is NOT read by the composite (`plan_timeline_segments` reads
  clip path/start_s/tfc/exists only). The real seam is `build_request_from_shot`.
- **BUG-405 is NOT a dispatcher fallback bug** (panel catch): the dispatcher reads
  the per-role engine and SKIPS on `assert_usable` failure (no silent flux fallback).
  6 flux_gen1 stills => the IMAGE POLICY carried flux_gen1. Capture it first.

## Invariants (unchanged; reject any fix that breaks one)
Audio spine FROZEN (`test_audio_byte_identical` green); workflow source of truth =
`otr_scifi_16gb_full.json` (wire JSON changes in same commit); LOUD fallbacks;
single resident heavy ≤14.5 GB; 100% local; UTF-8 no BOM; SFW.

## FIX 1 — opener centre still is black (the real root)
**Grounded root:** `render_driver.build_request_from_shot` resolves the beat's
`init_image` to the SCENE STILL only when `engine_family ∈ _SCENE_INIT_FAMILIES`
(`{"image_to_video","static_motion"}`, L413/L667) or `engine_id=="ltx_video"`
(LTX-I2V branch, L686). **`flux_still`'s family is `"static_image_gen"`** — in
NEITHER path — so the opener (operator picked flux_still for the music slot) keeps
`init_image = portrait = ""` (b000 char_id="") → flux_still renders a black centre.
Dialogue beats show because they carry real char portraits.
**Fix:** make `flux_still` condition on the beat's scene still in
`build_request_from_shot` — add `"static_image_gen"` to `_SCENE_INIT_FAMILIES`
(cleanest) OR an explicit flux_still scene-still branch — resolving via
`_still_index(ledger).get(beat_id)`; LOUD warning + trace stamp if the still is
absent (mirror the LTX-I2V missing-still pattern, never a silent black).
**Verify-at-build:** the opener still row IS in `_still_index` under
`beat_id="b000_music_open"` (`_still_index` filters `kind.startswith("scene_")`;
the dispatcher logs `scene_open b000` — confirm the row's `kind`/`beat_id`; if the
key is `still_b000_music_open` not `b000_music_open`, broaden `_still_index`
narrowly for `object_id == "still_"+beat_id`). Regression: a unit test on
`build_request_from_shot` for the synthetic opener asserts
`request["asset_refs"]["init_image"]` non-empty + `init_source=="scene_still"`.

## FIX 2 — per-role image-model selection ignored (capture-first)
**Grounded:** `otr_image_gen_dispatcher.resolve_engine_for_role` reads
`image_policy["image_models"][slot]`; `assert_usable` failure → warning + SKIP
(no flux fallback). So 6 flux_gen1 mints ⇒ the policy carried flux_gen1, NOT a
silent fallback and NOT (per GPT, verify-at-build) a hardcoded gen_fn
(`_inprocess_gen_fn` dispatches by `request["engine_id"]`).
**Step 1 (diagnostic, do FIRST):** add a LOUD one-line log in `dispatch_images`
before `assert_usable`: `object_id, role, slot, resolved engine_id`. Re-render →
read whether the policy carries `flux_gen1` or `lumina/qwen/hidream`.
**Step 2 (fix per the capture):**
- If policy = `flux_gen1`: the operator's GUI picks did not reach the policy →
  fix the source (the saved `otr_scifi_16gb_full.json` OTR_ImageDirector widgets
  are all flux_gen1; update them to the intended engines IN THE JSON, source-of-
  truth invariant) and/or confirm `OTR_ImageDirector.direct()` emits the live
  per-role model into `image_policy_json`.
- If policy = `lumina/qwen/hidream` but flux still minted: verify those engines
  are registered with a real `render_image` AND their weights are on disk; a
  missing engine/weight must fail LOUD (named), never silently mint flux.
**Cut:** no dispatcher routing rewrite (the engine-id dispatch is correct).

## FIX 3 — title-card window collapses to ~1s
**Grounded root:** `video_engine.SignalLostVideoRenderer.render_video` does
`led = _OTRLC.load_ledger(script_json)` (PRE-audio-timing) then calls
`_resolve_title_timing(led, …)` → `first_dialogue_f=None` (lines have no
`start_s`) → envelope fallback → `[0,25)` (~1s).
**Fix:** in `render_video`, immediately after `load_ledger`, overlay audio timing
before any title/HUD consumes `led` — call `otr_shot_lock.overlay_audio_timing(led)`
(or factor it into a shared dep-free helper if importing the node module is
unacceptable) so `_resolve_title_timing` sees `start_s` → window
`[0, first_dialogue)`. Remove/guard the `_envelope_intro_end` 1s fallback so it
cannot silently recur. Regression: pre-audio `script_json` (no start_s) + disk
ledger first-dialogue `start_s≈9.5` ⇒ `music_open_end_f ≈ first_dialogue_f`, not 25.
**Verify-at-build:** (a) `OTR_SignalLostVideo` `script_json` wiring in
`otr_scifi_16gb_full.json` — if it is fed the pre-audio ledger, this is a JSON
wiring change (rewire to the timed ledger OR add a timed-ledger input), NOT code-
only. (b) post-audio `speaker_role` values — if they are `char_voice`/`dialogue`
rather than `announcer`/`character`, extend `_SPEECH_ROLES_VIDEO`.

## DEFENSIVE (should-fix, non-blocking)
`plan_timeline_segments`: when `target_total_frames is not None` and SOME-but-not-
ALL rows carry `start_s`, log LOUD the missing `shot_id`/`beat_id` (the composite
report, not just the temporary instr) instead of silently flipping the whole
timeline to sequential. (Not the root of any current bug; a latent guard.)

## CUTS (panel consensus — do NOT do)
- No second `music_open` ledger line (ShotLock `derive_opening_music_beat` already
  makes the beat from first-dialogue `start_s`; a `start_s=0` line also breaks it).
- No dispatcher routing rewrite (engine-id dispatch is correct).
- No drawing the title card on the composite floor (the defect is localized in
  `_resolve_title_timing` input timing — fix that, smaller).
- Remove the `[BUG-403/404 instr]` logging once these land + regressions pass.

## Build order
FIX 1 (clear, code-only) → FIX 3 (small; confirm JSON wiring) → FIX 2 (diagnostic
log first, then the capture decides code vs JSON). Suite + Bug Bible after each;
commit+push per green chunk; re-render to verify opener still + title + per-role
image engines.
