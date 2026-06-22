# R2 -- coding plan: capability-based routing (operator-corrected)

## Corrected model (R1 + operator correction 2026-06-22)
`required_inputs` = each engine's TRUE MINIMUM. B-roll (wan_i2v, ltx_video, flux stills) = `("text_prompt",)`;
audio-driven specials (HuMo, LTX-AV, character_3d) = `("audio_ref", ...)`. `optional_inputs` = what it ALSO
consumes if present (e.g. `init_image` for b-roll -- the still is always DERIVED from the prompt per beat).
Role-fit = `required_inputs <= role_available_inputs`. -> B-roll fits EVERY role; audio-driven limited to
audio-supplying roles BY CAPABILITY. The `roles` whitelist becomes an OPTIONAL override (empty -> capability
only; non-empty -> a genuine creative restriction).

## Grounded code points
- `_otr_shared/role_compat.py::engine_fits_role` -- gates on `role in roles` AND `required <= available`.
- `otr_video_director.py::_registry_descriptors` (l.130-134): builds `roles = getattr(eng, "roles", ())`
  and `required_inputs = getattr(eng, "required_inputs", ())`. (CONFIRM what wan_i2v's `roles` resolves to --
  it currently excludes announcer_visual; class attr / mixin / empty?)
- `nodes/_otr_video_engines/eng_wan_i2v.py`: `required_inputs = ("init_image",)` (the mis-declaration).
- `nodes/_otr_video_engines/eng_ltx_video.py`: `required_inputs = ("text_prompt",)` (already correct -- the model).
- `nodes/_otr_video_engines/schemas.py::FAMILY_REQUIRED_INPUTS` (l.56) + a FAMILIES<->FAMILY_REQUIRED_INPUTS
  sync assert (l.72); the render gate `render_driver._assert_family_inputs_satisfiable` (l.1326-1342) uses it
  BY FAMILY. This is the SECOND place capability lives.

## Changes (minimal, additive, non-regressive)
1. **`eng_wan_i2v`**: `required_inputs = ("text_prompt",)`, `optional_inputs = ("init_image",)`. Align to
   ltx_video. Ensure no `roles` attr excludes announcer (set/leave `roles = ()` -> capability-only).
   Apply the same to any other i2v b-roll engine.
2. **`role_compat.engine_fits_role`**: make the `roles` check apply ONLY when `roles` is non-empty
   (optional override); empty/None -> capability-only (`required <= available`). KEEP the
   `required <= INPUT_TOKENS` fail-closed + the available-subset match.
3. **MotionEngineBase**: add `optional_inputs = ()` + `roles = ()` defaults (capability-only by default).
4. **Declare-once for the render gate**: `FAMILY_REQUIRED_INPUTS` for wan's family must = its `required_inputs`
   (`("text_prompt",)`). Add an assert-equal test (each engine's `required_inputs` == its family's
   FAMILY_REQUIRED_INPUTS) so the two never drift. Keep the render gate (it checks the concrete request's
   PRESENT tokens; role-fit checks the role's theoretical SUPPLY).

## Tests (the NON-REGRESSION proof -- operator's hard bar)
- **Generated before/after eligibility table**: for EVERY registry engine x EVERY role, assert
  `before(engine_fits_role)=True => after=True`; print the additive `False->True` deltas (expect
  wan_i2v -> announcer_visual + background_abstract + scene_broll, etc.). ZERO `True->False`.
- Audio specials (HuMo/LTX-AV/character_3d/visualizer) still fit ONLY audio-supplying roles after (NOT
  background_abstract).
- Auto-SELECTION non-regression: the default engine PICK for each existing slot is unchanged (eligibility
  expansion must not change auto-picks).
- Suite + Bug Bible green; re-render the 100% Wan -> wan now drives the announcer (b-roll) AND HuMo still does.

## Open -> R3 (wiring)
- Confirm wan_i2v's descriptor `roles` source (the exact thing excluding announcer today).
- ASPECT (wide vs portrait): enforced downstream already (director per-role aspect), or needs an explicit
  capability dimension? Must NOT be silently encoded in `roles`.
- FAMILY_REQUIRED_INPUTS: derive-from-engine vs assert-equal (are there multiple engines per family with
  different required inputs?).
- IMAGE engines: same model, deferred to a separate build item.

## Invariants
Strict SUPERSET (before/after test proves it); no-silent-swap LOUD on a real mismatch; audio specials gated
by capability; deterministic CPU tests; UTF-8 no BOM; SFW; NO workflow-JSON change (engine attrs + role_compat
only).
