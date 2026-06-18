# Coverage architecture v1 (pass01 -- grounded synthesis)

Panel: gemini-3.1-pro + grok-4.3 returned full critiques; opus/sonnet/gpt/deepseek
errored (finish_reason=length -- reasoning ate the token budget; raise --max-tokens
or --reasoning-effort none next pass). Claude grounded every surviving claim against
the real code and is the sole judge. Both panel verdicts were "no" (the problem
statement posed open questions without decisions) -- this pass RESOLVES them.

## Decision 1 -- the still<->video contract is TWO fields, not a new type
Add to the VideoEngine adapter protocol (registry.py), NOT a new `StillInput` class
and NOT a separate coverage table (both CUT as over-engineering -- `required_inputs`
already encodes the same information):
- `accepts_still: bool` -- does this lane take a generated still as init input.
- `still_input_name: str = "init_image"` -- the input name to feed it under.
Audio-only lanes (ltx_av_music) and the procedural floor (visualizer) set
`accepts_still = False` EXPLICITLY (opt-out is declared, never an accident of an
omitted required_inputs entry). [grok MUST-FIX #1; CUT #1/#2 -- CONFIRMED against
registry.py: adapters are duck-typed, adding two attrs is non-breaking.]

## Decision 2 -- the dispatcher keys on the capability, with a dual-read migration
`_still_needed_for_role` (otr_image_gen_dispatcher.py ~line 155) currently keys on
`"init_image" in required_inputs`. Migrate to: read `accepts_still` if the adapter
declares it; else fall back to `"init_image" in required_inputs` (required_inputs
wins for existing names during transition). So humo / ltx_av_talk / ltx_video keep
working unchanged; new lanes use the explicit flag. [grok MUST-FIX #3/#4 -- CONFIRMED:
the gate is a single function, dual-read is a 3-line change.]
- FIX the bare `except: return True` in that function (otr_image_gen_dispatcher.py
  ~158) -- an unknown engine must log LOUD, not silently force a still. [grok
  SHOULD-FIX #1 -- CONFIRMED, violates the no-silent-fallback invariant.]

## Decision 3 -- "approval in one place" = ONE usability surface (the real ask)
Image-MODEL *selection* is already one place (OTR_VideoDirector, shipped b8bb388).
What is still scattered is image-model *usability/approval*: role_compat.engine_fits_
role, registry.VALIDATED_ENGINES, _still_needed_for_role, and each adapter's
assert_usable all encode pieces of "may this image model be used here". Centralize
the APPROVAL decision behind ONE registry helper --
`image_engines.registry.usable(name, role) -> (ok, reason)` -- that the two
directors AND the dispatcher call. Video/3D adapters NEVER store a private list of
approved image models; they only declare `accepts_still`. Coverage (image x video)
is then DERIVED: usable-image x accepts_still-video, filtered by role_compat. No NxM
table to maintain. [grok MUST-FIX #2 -- CONFIRMED the scatter; this is the operator's
"one place, not per video/3D model" requirement.]

## Decision 4 -- do NOT conflate engine capability with role capability
The dispatcher must still respect role_compat (roles gate cross-role coverage); a
video lane that `accepts_still` does not mean every image model fits every role.
"Any image -> any video" is specifically the *still-input* axis; the *role* axis
stays governed by role_compat. Keep the two axes separate or the procedural floor
breaks. [gemini MUST-FIX #1 -- CONFIRMED; this is the guardrail on Decision 1.]

## Decision 5 -- 3D reuses the same declaration (one capability, a kind)
`requires_mesh_portrait` (the 3D lock) and `accepts_still` are the same family:
a 3D lane "accepts a still" as a `mesh_portrait` kind; a 2D lane accepts it as an
`init_image` kind. Model it as `accepts_still: bool` + `still_kind:
{"init_image","mesh_portrait"}` (default init_image). `three_d_locked_slots` /
`enforce_3d_granularity_lock` then read `still_kind == "mesh_portrait"` instead of a
separate `requires_mesh_portrait` flag -- ONE declaration drives both the 2D init
lane and the 3D granularity lock. [grok MUST-FIX #5 -- CONFIRMED two parallel
mechanisms exist today; unify them.]

## Invariants preserved
Model-agnostic / no primary; role_compat stays the role filter; the new fields are
plain attrs (cold-import clean, no NVML at import); workflow JSON stays source of
truth; no silent fallback (Decision 2 fix); single-resident VRAM unchanged (this is
wiring/metadata only, no new resident model).

## Build order (small, reversible chunks)
1. Add `accepts_still` + `still_input_name` + `still_kind` to the video adapters
   (default-derived from required_inputs so it's a no-op at first), + a protocol doc.
2. Add `image_engines.registry.usable(name, role)` central helper; point the two
   directors + dispatcher at it (behavior-preserving refactor + tests).
3. Migrate `_still_needed_for_role` to dual-read + fix the silent-True except.
4. Unify the 3D lock onto `still_kind == "mesh_portrait"`; delete the parallel
   `requires_mesh_portrait` reads once equivalent (cleanbreak, same change).
5. Regression + Bug Bible after each; the all-procedural + flux2->LTX soaks both stay
   green.

## Still open (verify-at-build / next pass)
- Exact signature of `usable()` (tuple vs raise) -- align with EngineUnusable.
- Whether `still_input_name` ever differs from "init_image" in practice (LTX-AV
  stages it as image_name; confirm the wrapper input name).
- Re-run the panel with raised max-tokens so opus/sonnet/gpt/deepseek also weigh in
  (4/6 errored this pass; the 2 that answered converged with the grounding).
