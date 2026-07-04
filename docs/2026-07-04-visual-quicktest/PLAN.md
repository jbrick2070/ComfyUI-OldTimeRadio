# CODING PLAN -- visual QUICK-TEST harness (frozen ledger + baked audio -> skip writer/audio, sweep every cloud image+video model)

Date: 2026-07-04  Branch: v2.0-alpha
Goal: iterate on IMAGE / VIDEO engines in ~1 min instead of ~8-10 min by reusing a
frozen episode (ledger + master WAV) and running ONLY the visual tail. Then sweep
every published cloud model and RIP the ones that don't render.

## 0. WHY THIS WORKS (grounded)
- `OTR_ImageGenDispatcher` (node 91) reads the ledger from its `script_json`
  **forceInput** (`otr_image_gen_dispatcher.py:783-784`, `led = self._loads(script_json, {})`
  :820). Feed it a FROZEN ledger and the writer is bypassed.
- Downstream (video render, composite, credits, MUX) resolves the episode dir + master
  WAV by the ledger's `episode_id` (the log shows LOUD re-resolve when a dir is renamed).
  So if the FROZEN episode dir (ledger + `audio/<ep>_master.wav`) is on disk, the audio
  is already "baked in" -- no writer, no voice/music/assembler nodes needed.
- The whole audio half (nodes 1 writer, 80/81/82/83 cast+voice+music, EpisodeAssembler)
  is REPLACED by "load the frozen ledger". The tail runs against the pre-computed master.

## 1. BUILD -- three pieces

### A. A tiny loader node: `OTR_LoadLedgerFixture`  (new, ~40 lines)
- INPUT: `ledger_path` (STRING) = path to a frozen `<ep>_ledger.json`.
  optional `episode_dir` (STRING, default "") to pin the audio dir if the ledger's
  episode_id dir moved.
- OUTPUT: `(script_json, audio_done, gate)` --
  - `script_json` = the ledger file's raw text (so ImageGenDispatcher consumes it verbatim).
  - `audio_done` = the sentinel the video stage waits on, rebuilt from the ledger's
    `total_episode_dur_s`/master (mirror `[EpisodeAssembler] emit audio_done signal:
    audio_done:length_sec=..;sample_rate=..;length_samples=..;segments=..`).
  - `gate` = a non-empty STRING for the gate_in chain.
- It must ALSO stamp the durable ledger singleton (`production_ledger.get_ledger()` /
  `stamp_durable`) from the frozen data so `OTR_CreditsRoll` (which reads the singleton,
  see docs/credits) has its col-1/2/3 content -- otherwise credits render blank.
- Register in the 5 standard sites (NODE_CLASS_MAPPINGS etc.).

### B. A fixture episode (frozen, checked-in-ish)
- Pick a GOOD completed episode (e.g. `output/otr/episodes/signal_lost_the_human_data_
  point_20260704_115654/`). Copy its `audio/<ep>_master.wav` + `<ep>_ledger.json` to a
  stable fixture dir (e.g. `tests/fixtures/quicktest_episode/`), OR just reference it in
  place. The ledger already carries lines + cast + dramatic_state + engines, so credits +
  video timing work. (Master WAV ~ a few MB; keep ONE.)

### C. Quick-test workflow `workflows/otr_visual_quicktest.json`
- Wiring: `OTR_LoadLedgerFixture` -> `OTR_ImageGenDispatcher` (91) -> `OTR_VideoRenderBatch`
  (87/92) -> `OTR_SilentComposite` (84) -> `OTR_CaptionBurn` (86) -> `OTR_PostUpscaleProcgenBlend`
  (93) -> `OTR_CreditsRoll` (95) -> `OTR_MasterAudioMux` (85). NO node 1, NO 80/81/82/83,
  NO EpisodeAssembler. Keep `OTR_VideoDirector`/`OTR_ImageDirector` (87/88) so the model
  widgets are settable per run. Reuse the exact node params from
  `workflows/otr_scifi_16gb_full.json` (copy the tail nodes + relink to the loader).
- Validate: `OTR_WorkflowValidator` + JSON round-trip + link/widget audit.

## 2. THE SWEEP (test all published cloud models, RIP the dead)
Driver (headless, direct-submit like `scripts/_otr_cloud_audio_babysit.py`, OR a Desktop
checklist): for each engine, set the director widget + run the quick-test workflow, record
PASS (obs mp4) / FAIL(root-cause) into `docs/CLOUD_ENGINE_COVERAGE.md`.
- IMAGE (per role via ImageDirector): `cloud_recraft` (PASS), `cloud_flux_pro` (PASS),
  `cloud_nano_banana_2` (fixed @606dc7f1, retest), `cloud_seedream_2`, `ideo`, `ideo_word`.
- VIDEO (character_video-scoped -- face engines only on character; see the RADIO-IS-HOST
  rule in CLOUD_ENGINE_COVERAGE.md): `word_razzle` (PASS), `cloud_kling_avatar`,
  `cloud_seedance_2` (dark row), `cloud_kling_lipsync` (needs base clip).
- RIP POLICY (operator): an engine that FAILS LOUD after a real root-cause attempt and has
  no fix path gets its adapter + partner row + dropdown entry removed (like the reasoning-
  LLM cull), so the dropdown only ever offers models that actually render. Record the rip.

## 3. VERIFY
- The quick-test itself is the harness; unit-test the loader (`OTR_LoadLedgerFixture` emits
  a well-formed audio_done + stamps the durable singleton from a fixture ledger).
- Full suite + Bug Bible after the loader node; workflow JSON in the same commit; push per chunk.
- Acceptance: a quick-test run with image=recraft completes to an obs mp4 in < ~90s (no
  writer, no audio) with the credits col-3 populated (proves the durable-singleton stamp).

## 4. NOTE
This is orthogonal to the `other_beats -> character` rename plan (that plan's image-role
rename should land first or be respected here -- use `character_image` role keys, not
`other_beats_image`, when the rename ships).
