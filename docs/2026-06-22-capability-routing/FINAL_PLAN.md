# FINAL build-ready spec -- capability-based routing (R2+R3 converged)

3-round roundtable (gpt-5.5 + gemini-3.1-pro + deepseek-v4-pro + grok-4.3 + Claude judge) + operator
correction. The fix is TINY (~3 lines + a test) and PROVABLY non-regressive.

## Root cause
`nodes/_otr_video_engines/eng_wan_i2v.py:85` -> `roles = ("scene_broll","music_visual","character_video")`
-- a hand-maintained whitelist that OMITS `announcer_visual`. `engine_fits_role` gates on `role in roles`
AND `required_inputs <= role_available_inputs`, so eligibility today = **roles INTERSECT capability**.
wan's input match for announcer PASSES (announcer supplies init_image) but `roles` forgot it -> blocked.

## The fix
**Drop the `roles` gate -> eligibility = capability-only.** Provably non-regressive: capability is a
SUPERSET of (roles INTERSECT capability), so dropping it only ADDS fits, never removes. This is the
operator's "declare capabilities once, model-agnostic downstream."

### Changes (3 lines + a test)
1. `nodes/_otr_shared/role_compat.py::engine_fits_role`:
   - `if required is None: return False`  (was `if roles is None or required is None: return False`).
   - DELETE the `if role not in tuple(roles): return False` gate.
   - KEEP `if not required_set <= INPUT_TOKENS: return False` + `return required_set <= available`.
2. KEEP every engine's `required_inputs` (wan stays `("init_image",)` -- correct i2v; it still WON'T fit
   the still-less `background_abstract`). KEEP the `roles` attrs (now ignored by the gate -- DEFER the
   dead-attr cleanup to a later patch, reduces blast radius). `default_roles` UNCHANGED (auto-defaults).
3. DO NOT add `optional_inputs` (dead -- nothing consumes it) and DO NOT touch FAMILY_REQUIRED_INPUTS
   (wan keeps init_image -> the family gate is unchanged).

## Aspect -- RESOLVED (confirmed in code)
`otr_video_director._role_aspects` (l.306) derives each role's still aspect from the SELECTED engine's
`render_aspect` (l.316, wide vs portrait). Picking wan (wide) -> a WIDE still -> wide video; picking HuMo
(portrait) -> portrait. Aspect is SELF-CONSISTENT per pick -- the `roles` whitelist was NEVER an aspect
gate. So dropping it introduces no aspect mismatch.

## Tests (`tests/test_capability_routing.py`, deterministic CPU) -- the non-regression proof
- STATIC parameterized eligibility matrix (capability-only). Key asserts:
  - `engine_fits_role(wan_desc, "announcer_visual")` -> TRUE (the live-wall fix).
  - `engine_fits_role(wan_desc, "background_abstract")` -> FALSE (no still for the i2v).
  - audio specials (humo / ltx_av / character_3d / visualizer -- require `audio_ref`) fit ONLY
    audio-supplying roles; NOT `character_video` (no audio_ref) / `background_abstract`.
  - a descriptor with NO `roles` key fits by capability (gate is gone; no fail-closed).
- render-gate: a synthesized announcer request passes `_assert_family_inputs_satisfiable` for wan
  (init_image present).
- Auto-selection: the default engine PICK per existing slot is unchanged (default_roles drives defaults).
- Full suite + Bug Bible green; re-render the 100% Wan -> wan now drives the announcer (b-roll) AND HuMo
  still does (operator eyeball).

## Non-regression (provable)
Eligibility: `roles INTERSECT capability` (old) ⊆ `capability` (new) = strict SUPERSET -> zero routes
removed. Auto-defaults: `default_roles` untouched. Aspect: self-consistent per pick (downstream). The
static matrix test is the CI proof.

## Invariants
NO workflow-JSON change (role_compat.py + one test file); audio specials gated by capability; deterministic
CPU tests; UTF-8 no BOM; SFW; strict superset.
