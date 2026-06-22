<!-- Claude (Opus) grounded panelist -- R2 coding plan. -->

# R2 grounded panelist review (Claude) -- capability-routing coding plan

## The plan is right + minimal; three code-level risks to nail

1. **wan's `roles` source must be EMPTIED, not just "optional-override"-bypassed.** The descriptor
   `roles = getattr(eng, "roles", ())` (director l.132). wan today is rejected from announcer, so its
   resolved `roles` is NON-EMPTY and excludes announcer_visual. Gemini's "enforce only if non-empty"
   would THEN STILL exclude wan. So the fix must set wan's `roles = ()` (capability-only) -- AND confirm
   where that non-empty value comes from (a class attr? a mixin? the registry?). Action: grep all
   `roles =` on the wan class + its mixins/base; set it empty (capability governs).

2. **i2v + a still-less role = a real edge.** Once capability-only, wan (now `required=("text_prompt",)`)
   becomes eligible for `background_abstract`, which supplies ONLY `{text_prompt}` (no init_image). But
   wan is image-to-video -- it needs a still. The operator's model ("the still is always derived from the
   prompt") must be WIRED: either `background_abstract` also derives+supplies an init_image, or wan's
   render path generates one from the prompt when none is provided. R3 must resolve this, or
   `optional_inputs` is a lie for an engine that truly needs a still. (Safer v1: keep wan out of
   background_abstract by NOT adding init_image to that role -- and accept wan fits announcer/music/
   scene_broll/character_video which DO supply a still.)

3. **ASPECT.** wan/ltx = wide; some roles/HuMo = portrait. Capability-match ignores aspect. Confirm the
   director's per-role aspect derivation (`_role_aspects`) enforces it DOWNSTREAM and that `roles` was NOT
   silently the aspect gate. Add a test: a wide engine is not auto-picked into a portrait-only slot.

## Accept
- wan `required_inputs=("text_prompt",)` + `optional_inputs=("init_image",)`, matching ltx_video.
- `optional_inputs` on MotionEngineBase (default ()); engine_fits_role ignores optional (only
  `required <= available`).
- FAMILY_REQUIRED_INPUTS must equal the engine `required_inputs` (assert-equal test; keeps the render
  gate single-sourced).
- Generated before/after eligibility table as the non-regression proof (True->False forbidden).
- Auto-selection non-regression test (eligibility growth must not change existing slot picks).

## Bottom line
The change is ~3 small edits (wan attrs + engine_fits_role conditional + MotionEngineBase default) + the
two safety tests. The non-regression bar is met BY CONSTRUCTION (dropping a gate only adds fits) PROVIDED
wan's `roles` is the only thing emptied and aspect stays enforced downstream.
