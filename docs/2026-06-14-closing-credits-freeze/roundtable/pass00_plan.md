# BUG-LOCAL-406 — closing FREEZE + dropped rolling-credits/HUD post-roll (regression). Fix plan to harden.

## One-paragraph problem
The published episode used to end with a ~20 s rolling-credits / Telemetry-HUD
post-roll (over its own closing music). It worked **2 days ago (2026-06-12)**.
**Now** the video FREEZES on the last drama frame while the closing theme plays and
the credits/HUD never appear. This is a REGRESSION introduced by the §4D procgen
pipeline (committed 2026-06-13: `336fb41` 4C floor/title, `39aa6c9` 4D scopes +
3-input blend, `eb64cd1` wired 4D into the workflow JSON). We must RESTORE the
post-roll through the new pipeline WITHOUT touching the frozen master audio.

## Grounded evidence (measured, not assumed)
- Floor node `OTR_SignalLostVideo` (nodes/video_engine.py) is SELF-CONTAINED for
  the whole episode incl. the post-roll:
  - `total_encode_frames = total_frames + _hud_frames` (video_engine.py L2001) —
    `total_frames` = master length, `_hud_frames` = the ~20 s HUD post-roll.
  - floor AUDIO = `pcm_out = np.concatenate([pcm, hud_pcm])` (L1918) — master mix
    PLUS the HUD closing-music (`closing_audio` from MusicGen, else a gentle decay).
  - So for a 61 s master the floor mp4 is ~81 s (61 s drama + 20 s credits post-roll,
    with audio for both). Measured: 06-12 floor `audio\...192436.mp4` = **81.08 s, AUD**.
- ffprobe, 06-12 (WORKING) episode `signal_lost_black_eyes_gambit_20260612_192436`:
  - `pending_..._master.wav` = **61.08 s**; `_silent.mp4` (composite) = 61.04 s;
    `_silent_procgen_blended_final.mp4` = **61.08 s**; floor `audio\...mp4` = **81.08 s**.
- ffprobe, 06-14 (BROKEN) episode `signal_lost_plunging_depths_20260614_185229`:
  - master.wav = 39.705 s; composite `_silent.mp4` = 39.68 s; `_..._blended_final.mp4`
    = 39.705 s. (No post-roll segment present at all.)
- Production video chain (from the workflow JSON, verified):
  `EpisodeAssembler(7)` → `SignalLostVideo(12)` (floor, ~master+hud) →
  `SilentComposite(84)` (base_video_path = floor) → `CaptionBurn(86)` →
  `PostUpscaleProcgenBlend(93)` (source=captioned composite, procgen=floor, scopes) →
  `MasterAudioMux(85)` → publish.
- `OTR_SilentComposite.assemble_silent_timeline` (nodes/otr_silent_composite.py
  L402-444) sets `target_total` from `manifest['total_target_frames']` (the
  beats-only budget = sum of per-shot `target_frame_count`, render_driver.py L1380-1440),
  then — only when `floor_ok` — extends it to the MASTER-MIX WAV duration, and the
  comment EXPLICITLY caps at "the MASTER MIX duration -- NOT the base's video
  length (the procgen runs ~20s past the master with its own silent post-roll)".
  So the composite is master-capped BY DESIGN → it never carries the [master,
  master+hud] post-roll.
- `OTR_PostUpscaleProcgenBlend` (nodes/otr_post_upscale_procgen_blend.py) blends
  `[0:v]` source (composite, master-len) with `[1:v]` procgen (floor, ~master+hud)
  using `blend=...:shortest=1` (L442/L481) → output clamps to the SHORTER input
  (the master-len source) → the floor's post-roll tail is discarded. Audio is
  `-map 0:a?` from the source (master-len). The muxer-level `-shortest` was
  deliberately removed earlier (C7); only the FILTER-level `shortest=1` remains.
- `OTR_MasterAudioMux(85)` then muxes the frozen master mix (master length).

## Root cause (grounded)
The credits/HUD post-roll exists only in the FLOOR (`SignalLostVideo`), in the
region `[master, master+hud_frames]` with its own closing-music audio. Every
downstream node in the §4D publish chain caps at the MASTER length:
the composite extends only to the master WAV (by design), and the §4D blend
`shortest=1` clamps to the master-len source. So the post-roll is dropped, and
because the floor (procgen_mp4) is longer than the source, the final ends at the
last master-aligned frame → the closing-theme tail of the master shows a held
last frame = the FREEZE the operator sees. 2 days ago the post-roll reached the
final (pre-§4D publish path); the §4D rewiring dropped it.

## Invariants (reject any fix that breaks one)
- AUDIO SPINE FROZEN: the `[0, master]` master mix must stay byte-identical
  (`test_audio_byte_identical` GREEN). The post-roll closing-music for `[master,
  master+hud]` is NOT part of the frozen master and already lives in the floor's
  audio — it may be appended, but the master region must not be re-encoded/shifted.
- NO muxer-level `-shortest` (C7: it cut the master audio tail). Keep the
  framesync/clone behavior the codebase chose; do not reintroduce `-shortest`.
- 100% local; single resident heavy ≤14.5 GB; LOUD fallbacks; UTF-8 no BOM; SFW.
- Workflow source of truth = `workflows/otr_scifi_16gb_full.json`: any node/
  wiring/widget change goes IN that file in the same change + re-validate.

## Candidate fixes (for the panel to critique / improve — do NOT assume one)
- **A. Length-carry through §4D.** Make the §4D blend output the LONGER input:
  pad/scale `[0:v]` source to the procgen (floor) length (tpad clone its last
  frame to `floor_len`) so `blend` covers `[0, floor_len]`; the `[master,
  floor_len]` region is then floor-only = the credits post-roll. Then the
  MasterAudioMux must supply audio for `[master, floor_len]` = the floor's
  post-roll closing music (NOT the frozen master). Question: cleanest way to get
  that tail audio without touching the `[0, master]` frozen mix — e.g. mux the
  floor's audio for the tail only, or have the blend pass through the procgen's
  audio for the tail.
- **B. Composite carries the post-roll.** Extend `assemble_silent_timeline`
  `target_total` to the FLOOR length (master+hud), tail-filled from the floor's
  post-roll slice, and carry the floor's post-roll audio. (Changes the "cap at
  master, not base length" design — assess against the C7 history + the frozen mix.)
- **C. Publish the floor for the tail only / concat.** Keep §4D over `[0,master]`,
  then concat the floor's `[master, floor_len]` post-roll (video+audio) onto the
  final. (A concat avoids re-encoding the master region; assess sync + the mux.)
- Identify the MINIMAL restore that matches the 2-days-ago behavior, name the
  exact node(s)/lines to change, and the regression test to add (e.g. assert the
  final's duration ≈ `round((master_dur+hud_dur)*fps)` and the tail segment is the
  floor post-roll, not a clone-hold).

## Verify-at-build
- Confirm what the PRE-§4D publish path emitted 06-12 (floor direct vs extended
  chain) from git (`eb64cd1` workflow delta) — to match the restore to the proven-good behavior.
- Confirm `_hud_frames`/closing-music duration is deterministic + the floor's
  post-roll audio is present in its mp4 (it is: floor `audio\...mp4` carries audio).
