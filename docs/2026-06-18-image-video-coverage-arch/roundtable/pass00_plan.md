# Coverage architecture: any image model -> any video/3D model (one place)

## The ask (operator, 2026-06-18)
"We need a better architecture for all video models to ensure they accept all image
models as inputs -- a future-proof init_image-consuming lane. Ideally approval to
use an image model is in ONE place and not stored uniquely for every video model or
3D model."

## Current state (grounded against the code, 2026-06-18)
- Image engines (`nodes/_otr_image_engines/`) mint a still: `prompt -> .png`,
  `required_inputs = ("text_prompt",)`. flux_gen1 (default), lumina_image,
  z_image_turbo, flux2_klein (new), + opt-in peers.
- Video engines (`nodes/_otr_video_engines/`) each declare `required_inputs`, and
  whether they consume a still is implicit in whether `"init_image"` is in that
  tuple:
  - `ltx_av_talk`: ("text_prompt","audio_ref","init_image")  -> consumes a still
  - `humo` family: audio_ref + init_image                    -> consumes a still
  - `ltx_video`: ("text_prompt",)                            -> can use init_image
    but does NOT declare it required -> the still-skip gate SKIPS its still
  - `ltx_av_music`: ("text_prompt","audio_ref")              -> no still
  - `visualizer` / abstract floor: no init_image             -> no still (by design)
- The dispatcher's still-skip gate (`_still_needed_for_role`, shipped `b2f07e0`)
  reads the role's selected video engine `required_inputs` and SKIPS the image
  render when `"init_image"` is absent. This is correct for the procedural floor,
  but it ALSO silently skips engines that COULD use a still (ltx_video) -- coverage
  is an accident of each engine's `required_inputs` tuple.
- Image-MODEL selection now lives in ONE node (OTR_VideoDirector, `b8bb388`). But
  image<->video COMPATIBILITY (which video lanes accept a still, which image models
  are approved/usable) is still scattered: per-engine `required_inputs`, per-engine
  `assert_usable`, role_compat membership, the validated/opt-in display gate, and
  capability profiles. There is no single "image model X is approved + this is how a
  still is fed to any video lane" declaration.

## What "good" looks like (design goals)
1. **Universal init_image lane.** Every video/3D engine can OPT to receive a still
   from any image engine via ONE typed contract (e.g. a `StillInput` capability),
   not a bespoke per-engine wire. Audio-only lanes (ltx_av_music) and the procedural
   floor (visualizer) opt OUT explicitly, not by omission.
2. **Approval in one place.** Whether an image model is allowed/usable is declared
   ONCE (a central image-model capability/approval registry), not re-stored on every
   video or 3D adapter. A video engine references "accepts stills: yes/no + how",
   never a private list of approved image models.
3. **Future-proof.** Adding a new image model or a new video model is a single
   registration; coverage (image x video) is derived, not hand-wired. No N*M matrix
   to maintain.
4. **Fail-loud, no silent skips of capable lanes.** A lane that CAN use a still
   should either use it or LOUDLY declare it doesn't -- never an accidental skip.
5. **Honors the OTR invariants:** model-agnostic / no "primary"; role_compat as the
   shared filter; cold-import clean; the workflow JSON
   (`workflows/otr_scifi_16gb_full.json`) is the source of truth; positional widget
   safety; single-resident VRAM discipline; fail-closed on missing weights.

## Open design questions (for the roundtable panel)
- Where does the still<->video contract live? A capability field on the video
  adapter (`accepts_still: True`, `still_input_name: "init_image"`), a shared mixin,
  or a separate coverage table? How does it stay DRY vs `required_inputs`?
- How is "image model approved/usable" centralized? A single approval/registry
  surface both directors + dispatcher read -- replacing the scattered
  required_inputs/assert_usable/validated-gate logic for the still path.
- How does the dispatcher decide to render a still: from the video engine's
  capability declaration (not its required_inputs tuple)? What's the migration so
  the still-skip gate keys on the capability, not the implicit tuple?
- 3D engines: a still -> mesh-portrait path differs from still -> 2D init_image. Is
  "accepts still" one capability with a kind (`init_image` vs `mesh_portrait`), so
  the 3D granularity lock + the 2D init lane share one declaration?
- Wiring in the litegraph JSON: one optional still input per video node vs a single
  routed still bus? Keep it provider-before-consumer + positional-widget-safe.
- Back-compat / migration: how to land this without breaking the shipped
  required_inputs consumers (humo/ltx_av_talk) or the procedural floor opt-out.

## Constraints
- 16 GB Blackwell laptop, single-resident heavy <= 14.5 GB, torch 2.10/cu130/sm_120.
- No silent fallbacks (operator directive 2026-06-16): a lane fails LOUD.
- Roundtable panel for ideas; Claude grounds every claim vs the real code and is the
  sole judge/synthesizer.
