# Claude anchor review -- r1 (black-clips root-cause) -- GROUNDED

VERDICT: the doc's prime suspect is WRONG; refocus the hunt. Grounded vs the c1132196 diff + the seam.

## CONFIRMED
- SYMPTOM real: per-beat clip files are the dark floor (0x0A0E14); finals = procgen-only at drama.
- IMAGE GEN WORKS: 3425 episode stills + 2997 cache, 223 minted in the last 14h, real ~1MB images.
  So the regression is NOT "stills not generated."
- MECHANISM real: build_request_from_shot (render_driver ~938) sets still_pan/still_flat init_image =
  `_still_index(ledger).get(still_pool_key or beat_id, "")`; empty -> "missing_scene_still" -> the
  cheap family's dark floor. `_still_index` (~414) keys `ledger['images']['images']` rows whose `kind`
  starts `scene_`, by `beat_id` (scene_background_plate wins).

## MISREAD / CLEARED
- **c1132196 (flux_still->still_pan) is a PURE RENAME -- NOT the cause.** I read the diff: the
  still->video conditioning is byte-identical (same `_still_index(...).get(still_pool_key or beat_id)`
  lookup, same dark-floor-on-missing). The CAPABILITIES de-flux only flips commercial_clean/heavy->cpu;
  the comment itself says it "never loads Flux." So the rename did not change generation or routing.
  DROP c1132196 as the suspect.

## REFOCUSED ROOT-CAUSE (the real break to find)
The minted scene still exists on disk (still_b003_<hash>.png) but is NOT found by `_still_index` under
the beat's `still_pool_key`/`beat_id`. So ONE of:
1. The dispatcher (otr_image_gen_dispatcher.py) does NOT write a `scene_*` `images.images` row for the
   still-carrier beats (announcer/music bookends + still_pan-filled beats), or writes it under a
   `beat_id`/`still_pool_key` that does NOT match the shot's lookup key.
2. The image POLICY/granularity does not REQUEST a scene_* still for those beats (so none is minted for
   them) -- e.g. the dispatcher mints scene_character for character beats but no scene_open for the
   still_pan-filled announcer/music beats.
3. A ~48h render-driver still-routing edit broke the key/kind: **e8fa941d** + **a30f5945** (the
   ltx_audio_in role-driven still routing, +80/+25 lines in render_driver) -- did they change which
   beats resolve a scene_* still, or the still_pool_key derivation?

## WHAT THE PANEL MUST GROUND (vs the code)
A. Trace dispatcher mint -> the EXACT `images.images` row written for a still-carrier beat (its `kind`
   + `beat_id` + any `still_pool_key`) vs the lookup key in build_request_from_shot. Find the mismatch.
B. Diff e8fa941d + a30f5945 against the still-routing + the dispatcher's scene-still decision; identify
   the line that stopped a scene_* still being written/keyed for the still-carrier beats.
C. The "before we removed fallbacks" link: a prior fallback may have masked the empty still (resolved
   an alternate still source); identify it.
D. Confirm scope: do CHARACTER beats (scene_character minted) also go black, or only announcer/music
   (scene_open)? That localizes whether it's the policy (which kinds are minted) or the lookup key.

## INVARIANT
Output the grounded ROOT CAUSE + minimal fix locus; no shim; no broad rewrite; master audio untouched.
