# ADDENDUM: ltx_audio_in radio-still A/B (faceless console vs HuMo-face) -- SEPARATE, POST-MAIN

**Status: BUILT & GREEN (code lane, 2026-07-01).** The MAIN feature (brief-driven HuMo
radio-host, `PLAN_HARDENED.md`) is merged + green, so this A/B was coded on top of it.
Shipped in commit `1cdef2b7` on `v2.0-alpha` (suite 5940 pass, Bug Bible + B7 green):
`OTR_LTX_RADIO_FACE` env toggle (default 0 = faceless, byte-identical); MetaBrief mints the WIDE
per-role `still_<role>_radio_face_169` stills only when the toggle is ON (seed-pinned to the
bookend seed, no-baby negative); `render_driver.build_request_from_shot` swaps the ltx_audio_in
announcer/music bookend init to that wide face still, failing LOUD if it is absent OR not wide,
and REJECTING LOUD if `OTR_ENABLE_HUMO_HOSTS` is also on (HuMo owns the bookends). The A/B is
NAMED as an LTX init asset ("radio_face", not "humo") -- LTX does ambient motion, not lip-sync.
Tests: `tests/test_ltx_radio_face_ab.py`.
REMAINING (operator, GPU): the actual frozen A/B render harness -- force the bookends to
`ltx_audio_in` (e.g. `OTR_FORCE_ENGINE_MAP=announcer_visual=ltx_audio_in,music_visual=ltx_audio_in`),
render the same baked episode twice (`OTR_LTX_RADIO_FACE=0` then `=1`) on a TENSE and a neutral
brief, eyeball, and pick the default. A persistent default = a separate gated
`otr_scifi_16gb_full.json` edit.

--- ORIGINAL PLAN (kibitz-hardened) BELOW ---

**Status: PARKED addendum.** The MAIN feature (brief-driven HuMo radio-host,
`PLAN_HARDENED.md`) is ALREADY being coded in another window. This A/B is a SEPARATE plan to
run AFTER that lands + is green -- do NOT fold it into the in-flight main build. Operator
2026-07-01.

## Preconditions (from the main feature)
Do not start until the main feature is merged + green. It provides the two stills this A/B
switches between:
- a FACELESS brief-driven radio still (from the now-brief-driven `get_open_subject` /
  `radio_form_from_meta`), and
- the HuMo `still_radio_face_169` FACE still.

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
  `still_radio_face_169` FACE still is PORTRAIT (for HuMo). Option (b) therefore needs a
  WIDE-framed face-radio still (face set in a wide console), not the portrait HuMo asset --
  either a wide variant of `still_radio_face_169` or aspect-follow at mint time. Do NOT feed a
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

## HARDENED (kibitz r1 -- Codex + Claude anchor, grounded; Antigravity/Claude-Code ran slow, folded what landed)
- **The A/B harness MUST force ltx_audio_in on the bookends.** Node-87 saves announcer/music =
  `viz_green`, so ltx is not the live bookend engine -- the toggle would be DEAD. Both A/B legs
  must override announcer/music to `ltx_audio_in` (e.g. `OTR_FORCE_ENGINE_MAP=announcer_visual=
  ltx_audio_in,music_visual=ltx_audio_in`, or a slot override). A persistent default change =
  a same-commit `otr_scifi_16gb_full.json` edit; the A/B itself stays override-only.
- **Mode matrix (two toggles must not fight).** `OTR_LTX_RADIO_FACE` applies ONLY when the final
  routed bookend engine is `ltx_audio_in`. If the main feature's `OTR_ENABLE_HUMO_HOSTS` is ON,
  HuMo owns the bookends -> `OTR_LTX_RADIO_FACE` is ignored / rejected LOUD (never a silent
  double-route). Document the precedence in one place.
- **Wide-still resolution branch + fail-loud.** `ltx_audio_in` conditions wide engines on the
  SCENE-still branch and clears portrait leakage (`render_driver.py` ~:1001-1024). Add a bookend
  branch: when `OTR_LTX_RADIO_FACE=1` AND role in announcer/music, resolve the WIDE
  `still_<role>_radio_face_169` as init_image INSTEAD of the scene still; fail LOUD if that row is
  absent or not wide (never black, never a silent portrait->pillarbox).
- **Provenance/naming (Codex): option (b) is a "wide radio-host still for LTX init", NOT "HuMo".**
  The RUNTIME engine is LTX (ambient motion), HuMo is only the still's STYLE source. Reserve
  "HuMo" for actual `audio_driven_face` rendering. Name = `still_<role>_radio_face_169`.
- **Frozen A/B (both stills pre-exist).** Freeze ONE ledger/audio/story (the S-F visual smoke
  fixture) with BOTH candidate stills already minted, then vary ONLY `OTR_LTX_RADIO_FACE` and
  render the video tail twice -- so the comparison is init-still-only, not image-gen differences.
  Run it on a TENSE brief AND a neutral one (don't pick the default from the wildcard's best case).
- **Decision rule.** Faceless (`still_<role>_radio`) stays the default UNLESS the face leg wins on
  readable radio identity + non-uncanny motion + no false lip-sync expectation across both briefs.
- **Manifest stamp per leg** (forensics): routed engine, init path/kind/dims, `OTR_LTX_RADIO_FACE`,
  and whether HuMo-host routing was active.
- **CUTS (Codex):** global env toggle only -- CUT per-episode/per-role granularity for this A/B.
  CUT any touch of mesh / viz / HuMo routing / character beats -- the surface is ONE
  `ltx_audio_in` announcer/music init-image branch + the frozen harness.

## Open questions (for kibitz)
- Wide face-radio asset: mint a wide `still_radio_face_169` variant, or aspect-follow the
  existing mint by engine slot? (Prefer reuse of the main feature's aspect-follow.)
- Toggle granularity: global env vs per-episode vs per-role. (Start global env; simplest A/B.)
- Does `ltx_audio_in`'s I2V motion preserve enough of the still that a face reads as a face,
  or does the motion smear it? (Empirical -- the whole point of the A/B.)
