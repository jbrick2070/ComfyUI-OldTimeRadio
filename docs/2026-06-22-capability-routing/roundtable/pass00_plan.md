# Capability-based engine routing -- problem statement (declare once, model-agnostic downstream)

## Goal (operator, 2026-06-22)
Declare each model's CAPABILITIES ONCE -- does it take image-in? text-prompt? audio-in?
video/base-clip-in? -- and route MODEL-AGNOSTICALLY downstream. NO per-engine role whitelists
scattered in multiple places. Any model serves any role whose inputs it can consume. HuMo + LTX-AV
(audio-in) are the only "specials" -- and that should FALL OUT of the capability match, not a hardcoded
list. Extends to IMAGE engines too ("the image architecture as well").

## Current state (code-grounded)
- `_otr_shared/role_compat.py::engine_fits_role` gates on TWO things: (1) `role in engine.roles`
  (a hardcoded per-engine whitelist) AND (2) `required_inputs <= ROLE_AVAILABLE_INPUTS[role]`
  (the input-capability match). Both must pass.
- `ROLE_AVAILABLE_INPUTS`: announcer_visual + music_visual + the other-beats roles each supply
  {text_prompt, init_image, audio_ref, base_clip_ref}; background_abstract supplies {text_prompt} only.
- Engines declare `required_inputs` (wan_i2v=("init_image",), ltx_video=("text_prompt",),
  character_3d=("audio_ref","init_image"), visualizer=("audio_ref",)) + `default_roles` (mostly empty)
  + a `roles` list (the whitelist read by engine_fits_role).
- render-side gate `render_driver._assert_family_inputs_satisfiable` re-checks FAMILY_REQUIRED_INPUTS
  vs the request's present tokens (text_prompt / init_image / audio_ref / base_clip_ref) -- a SECOND
  place the capability logic lives.

## The wall hit (live, 2026-06-22)
`OTR_VideoDirector: engine 'wan_i2v' does not fit any role for slot 'announcer_video_model'
('announcer_visual',) (incompatible required inputs)`. Root cause: wan_i2v IS input-compatible with
announcer_visual (needs init_image, which the announcer supplies) but its `roles` whitelist OMITS
announcer_visual -> rejected. ltx_video (text_prompt) fits everywhere. The whitelist over-restricts
capability-compatible engines. The no-silent-swap LOUD behavior is CORRECT; excluding wan is the bug.

## Proposed direction (to harden)
- ONE capability declaration per engine = the inputs it CONSUMES (image_in / text_prompt / audio_in /
  base_clip_in), a single source of truth. Role-fit derives PURELY from capability vs the role's
  available inputs -- drop the separate `roles` whitelist gate.
- An engine fits a role IFF its consumed inputs are a subset of the role's available inputs:
  - HuMo / LTX-AV / character_3d (require audio_ref) -> only audio-bearing roles (announcer/music).
    Stay "special" BY CAPABILITY, not a list.
  - still+prompt engines (wan_i2v, ltx_video, flux*, ...) -> all roles.
- Apply the same capability-once model to IMAGE engines.

## Hard questions for the panel
1. Is `roles`/`default_roles` purely redundant with the input-capability match, or does it encode
   something the match can't (creative-appropriateness, 16:9-vs-portrait aspect, "never-auto-default"
   vs "selectable")? Note `default_roles` looks like AUTO-DEFAULT, a SEPARATE concern from eligibility.
2. How to declare capabilities ONCE without breaking the existing consumers (engine_fits_role,
   render_driver FAMILY_REQUIRED_INPUTS, ShotLock, the director descriptor builder)? Single source ->
   both gates derive from it.
3. Safety: with the whitelist gone, does ANY engine wrongly fit a role (the no-silent-swap guarantee)?
   Enumerate engine x role.
4. Image engines: same whitelist gap? Unify.
5. Migration: change in ONE place, keep the no-fallbacks LOUD, regression suite + a re-render.

## Invariants
- **NON-REGRESSION (HARD, operator 2026-06-22 -- "make sure whatever fix doesn't break the existing
  models that are working well"): the fix MUST be a strict SUPERSET of current routing.** Every engine
  that fits a role TODAY must STILL fit it; the working models (HuMo audio-announcer, ltx_av_music
  bookends, ltx_video b-roll, flux stills, ...) keep their EXACT current eligibility. Only ADD
  previously-blocked-but-capable engines (e.g. wan_i2v -> announcer). PROVE it with a per-engine x
  per-role BEFORE/AFTER table -- zero working routes may change; the only deltas are additive.
- No-silent-swap safety preserved (LOUD on a genuine capability mismatch); audio specials
  (HuMo/LTX-AV/character_3d) stay gated to audio roles BY CAPABILITY; model-agnostic downstream;
  minimal/zero scattered whitelists; deterministic CPU tests; UTF-8 no BOM; SFW.
