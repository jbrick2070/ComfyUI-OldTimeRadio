# The 2D-still spine -- pass01 plan (panel-hardened; 6/5-restoration north star)

OPERATOR NORTH STAR (binding): the 6/5 pipeline produced top-notch FLUX radio
stills + awesome motion. This plan RESTORES that contract on the new platform;
where panel advice invents instead of restores, restoration wins. Reference:
`docs/2026-06-10-brief-downstream-gaps/legacy_otr_video_plan_e74a3ce.py.txt`
(the deleted otr_video_plan.py -- PASS-1 portraits / PASS-2 scene stills /
PASS-3 5-layer composite prompts).

## ST-1 Shared subject + still-prompt helpers (`nodes/_otr_story_brief_helpers.py`)

- `get_open_subject(role, synthetic)` -> the concrete radio subject strings
  (today's render_driver wording moves HERE; the driver refactors to call it).
- `compose_still_prompt(meta, *, kind, role, beat_id, char_entry=None)` --
  the legacy LAYER ORDER: subject (macro radio for opens; portrait_prompt for
  characters) + setting terms top-2 + framing hint ("full-frame macro,
  centered subject" opens / three-quarter chars) + TRIMMED era tail + style
  tail. Era-tail profiles: `era_tail_profile="still"` = atmosphere line +
  palette top-2 + lighting top-2 (~120 chars); existing video/portrait call
  sites unchanged this slice except portraits adopt the still profile.
- Parity test: driver LTX text prompt and the open's still prompt share the
  same leading subject string.

## ST-2 Scene-still objects (`nodes/otr_meta_brief_image_prompt.py`)

- After character prompts, derive SCENE-STILL objects WITHOUT graph reorder:
  the open via the same pure helper ShotLock uses
  (`derive_opening_music_beat(ledger, fps)`), announcer/outro via the role
  mapping over lines. Emit
  `{object_id: "still_<beat_id>", kind: "scene_open"|"scene_beat", role,
  beat_id, w, h, prompt, prompt_hash, source}` alongside portrait rows
  (portraits keep their schema + person guard + gear scrub; scene stills get
  the no-text clause and SKIP the person guard).
- Scene stills are LANDSCAPE (canvas-derived /32); portraits stay 832x1216.
- v1 scope: open + announcer + outro beats only (panel cut: not every beat).

## ST-3 Dispatcher (`nodes/otr_image_gen_dispatcher.py`)

- role/slot from the payload (kills the hardcoded character_video): announcer
  stills -> announcer_image_model, music/open -> music_image_model, scene ->
  other_beats (ImageDirector slots finally honored).
- `episode_id` input (workflow json relink, IN PLACE per operator directive):
  every still saves to `output/otr/episodes/<ep>/stills/`; ledger images[]
  rows carry `{object_id, kind, beat_id?, char_id?, path(episode-local),
  content_hash, provenance}`; `stills_manifest.json` written beside them.
- Cache: key gains kind/w/h; a cache hit still materializes into the CURRENT
  episode's stills/ + a fresh ledger row. Global pool: kept this slice;
  retired in a follow-up after the tracked-reader sweep proves zero readers.
- Seeds: V-7 request-hash scheme (determinism preserved).

## ST-4 Driver consumption (`nodes/_otr_video_engines/render_driver.py`)

- `_still_index(ledger)` -> `{beat_id: path}` for kind=scene_*.
- Init selection by ENGINE FAMILY: `audio_driven_face` -> portrait(char_id)
  (unchanged); `image_to_video` + `static_motion` -> scene still(beat_id),
  LOUD fallback to today's behavior when absent; text engines unchanged.
- Trace rows gain `init_source` (portrait|scene_still|none) + `init_image`
  basename for EVERY beat (the mechanical acceptance check).

## ST-5 Conditioned motion, v1 order (the look ships even if gates stay shut)

1. `still_kenburns` drifts the scene still (VERIFY it accepts external init;
   add if not) -- the literal 6/5 look, zero new GPU risk.
2. `wan_i2v` init = scene still where OTR_ENABLE_WAN_I2V=1 (one probe clip
   first: dimension snapping on landscape stills).
3. LTX img2vid: CUT from v1 (wrapper version-band risk proven by the 121f
   VAEDecode failure). LTX keeps the round-5 text path. Future probe ticket.

## ST-6 Sequencing + json

- Stills mint in the image phase; the image_done gate wired so the video
  render node cannot start first (VERIFY the gate input; add if missing).
- Workflow json edits (episode_id wire, gate wire) land IN
  `workflows/otr_scifi_16gb_full.json` in place; no new widgets beyond the
  relink; no runner patches.

## ST-7 Tests (CPU, suite-resident)

Schema emission (open/announcer/outro objects from a fixture ledger; b000
present WITHOUT ShotLock); dispatcher role/slot resolution + episode-dir save
+ manifest; driver family-based init selection + LOUD missing-still fallback;
parity (driver/still subject); era-tail profiles; determinism (same ledger ->
same seeds/hashes); trace init stamps; existing pins stay green.

## Out of scope (tickets)

M4->HuMo creative seam (S4); LTX img2vid probe; global-pool retirement sweep;
3D mesh routing (schema is already compatible: object_id/char_id/beat_id keys).

## Acceptance (ONE 30w production render)

`episodes/<ep>/stills/` contains the open still + portraits + manifest; the
open still is macro-radio 6/5-style and its prompt leads with the shared
subject (string assert + operator eyeball); every still exists BEFORE the
video phase (ordering assert); trace shows `init_source=scene_still` on every
static_motion/image_to_video beat and `portrait` on every talking head; all
round-5 gates stay green; suite + Bug Bible green; audio byte-identical;
commit+push per the operator git policy.
