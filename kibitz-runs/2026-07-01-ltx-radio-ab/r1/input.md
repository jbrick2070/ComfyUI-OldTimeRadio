# ADDENDUM: ltx_audio_in radio-still A/B (faceless console vs HuMo-face) -- SEPARATE, POST-MAIN

**Status: PARKED addendum.** The MAIN feature (brief-driven HuMo radio-host,
`PLAN_HARDENED.md`) is ALREADY being coded in another window. This A/B is a SEPARATE plan to
run AFTER that lands + is green -- do NOT fold it into the in-flight main build. Operator
2026-07-01.

## Preconditions (from the main feature)
Do not start until the main feature is merged + green. It provides the two stills this A/B
switches between:
- a FACELESS brief-driven radio still (from the now-brief-driven `get_open_subject` /
  `radio_form_from_meta`), and
- the HuMo `radio_host_portrait` FACE still.

## Goal
`ltx_audio_in` is I2V on an `init_image` -- confirmed: its docstring
(`eng_ltx_av.py:18`) says "I2V on the beat's WIDE scene still + the audio slice", and
`render_clip` (`:627`) stages `plan["init_image"]` (`:656-659`, comment :657 "the scene still
for music/announcer"). So the A/B is purely WHICH still becomes that init_image on
announcer/music bookends:
- **(a) faceless brief-driven radio-console still** -> ltx animates a "living instrument"
  (dials/glow/needle motion + camera drift, audio-reactive). The safe-cool default.
- **(b) HuMo-style radio-FACE still** -> ltx animates an AMBIENT face: it breathes / shimmers /
  drifts with the audio but does NOT lip-sync (ltx does motion, not face animation). The
  atmospheric wildcard -- eerie-cool for tense/paranoid briefs, or slightly uncanny.
Pick the cooler default by eyeball.

## Design
- **Toggle `OTR_LTX_RADIO_FACE` (env, default 0 = faceless console).** 0 = today's/main behavior
  (faceless brief-driven scene still). 1 = feed the face-radio still as ltx's init_image on
  announcer/music bookends only.
- Selection happens where the ltx bookend `init_image` is resolved (render_driver
  `build_request_from_shot`, the same place the main feature routes HuMo's init). Only
  `ltx_audio_in` + announcer/music; NEVER touches HuMo (main), mesh, viz, or character beats.
- **ASPECT (must-handle):** `ltx_audio_in` renders WIDE (16:9), but the main feature's
  `radio_host_portrait` FACE still is PORTRAIT (for HuMo). Option (b) therefore needs a
  WIDE-framed face-radio still (face set in a wide console), not the portrait HuMo asset --
  either a wide variant of `radio_host_portrait` or aspect-follow at mint time. Do NOT feed a
  portrait still to a wide engine (pillarbox -- the exact trap the main-feature kibitz flagged).
- **Non-lip-sync disclosure:** document that (b) is ambient motion, not talking, so the A/B
  eyeball is not misjudged as "broken lip-sync". HuMo remains the only true talking/singing host.

## A/B protocol
Render the SAME baked episode/brief twice (identical story+audio, e.g. the S-F visual smoke
fixture) -- once `OTR_LTX_RADIO_FACE=0`, once `=1` -- and eyeball the announcer/music bookends
side by side. Operator picks the default.

## Invariants
- Audio byte-identical (`test_audio_byte_identical` green). Determinism seed-keyed. Default 0 =
  no change to the current ltx look. LOUD fallback if the chosen still is missing (never black).
  UTF-8 no BOM; SFW/adult. Wide init for a wide engine. SEPARATE from the main feature; no
  overlap in the same commit.

## Open questions (for kibitz)
- Wide face-radio asset: mint a wide `radio_host_portrait` variant, or aspect-follow the
  existing mint by engine slot? (Prefer reuse of the main feature's aspect-follow.)
- Toggle granularity: global env vs per-episode vs per-role. (Start global env; simplest A/B.)
- Does `ltx_audio_in`'s I2V motion preserve enough of the still that a face reads as a face,
  or does the motion smear it? (Empirical -- the whole point of the A/B.)
