# R3 -- wiring plan: capability-routing (converged fix + wiring opens)

## ROOT CAUSE (grounding-confirmed)
`nodes/_otr_video_engines/eng_wan_i2v.py:85` -> `roles = ("scene_broll", "music_visual",
"character_video")` -- a hand-maintained whitelist that OMITS `announcer_visual`. `engine_fits_role`
gates on `role in roles` AND `required <= available`, so today's eligibility = **roles INTERSECT
capability**. wan's input match for announcer PASSES (announcer supplies init_image) but its `roles`
omits announcer -> blocked. (Gemini's "optional override / enforce-if-non-empty" does NOT fix this --
wan's `roles` IS non-empty, so it'd stay enforced and wan stays blocked.)

## THE FIX (R2 converged + grounding)
**Drop the `roles` gate in `engine_fits_role` -> eligibility = capability-only (`required <= available`).**
PROVABLY NON-REGRESSIVE: current = roles INTERSECT capability, which is a SUBSET of capability, so dropping
the gate only ADDS fits (superset) -- ZERO removed. KEEP every engine's `required_inputs` (wan stays
`("init_image",)` -- correct i2v; it still WON'T fit the still-less `background_abstract`). `default_roles`
stays for AUTO-DEFAULTS (separate concern) -> existing default picks unchanged. The per-engine `roles`
attrs become DEAD -> the "declare once" cleanup.

## Changes
1. `role_compat.engine_fits_role`: remove the `role in tuple(roles)` gate; keep `required is None ->
   False`, `required <= INPUT_TOKENS` (fail-closed), `required <= available`.
2. Remove the now-dead `roles` attrs from the engine classes (or leave as comments). default_roles stays.
3. KEEP all `required_inputs` (wan = init_image). NO FAMILY_REQUIRED_INPUTS change. CUT optional_inputs.

## Wiring opens (R3 panel resolves)
1. **ASPECT (the main one).** Capability ignores aspect (wan/ltx = wide; HuMo = portrait). With the
   `roles` gate gone, a wide engine is eligible for any capability-matching role, incl. potentially-
   portrait ones (`character_video`). CONFIRM aspect is enforced DOWNSTREAM (the director derives per-role
   aspect via `_role_aspects`; the still is rendered in the role aspect; the engine renders from that
   still). Was aspect IMPLICITLY hiding in `roles`? The before/after test surfaces any fit that depended
   on it. If a wide engine in a portrait role genuinely breaks, add an explicit `supported_aspects`
   capability + role aspect check (else rely on downstream).
2. **AUTO-SELECTION.** Confirm the director's auto-PICK uses `default_roles` (UNCHANGED) so existing slot
   defaults are preserved despite the larger eligible pool. Golden test per existing slot.
3. **before/after eligibility test mechanism.** Snapshot the CURRENT `engine_fits_role` result for every
   (engine, role) BEFORE the edit (committed fixture or an old-algorithm helper); assert `before=True =>
   after=True`; print additive deltas (wan -> announcer_visual expected).
4. **Dead `roles` attrs:** remove (the declare-once cleanup) vs keep-as-docs -- the before/after test guards
   either way.

## Non-regression (provable)
current = roles INTERSECT capability; new = capability; capability ⊇ (roles INTERSECT capability) =>
strict SUPERSET => no working route removed. default_roles unchanged => no auto-pick change. ASPECT is the
one thing to confirm is enforced downstream (not silently in `roles`).

## Invariants
Strict superset (before/after test); audio specials gated by capability (require audio_ref); no
workflow-JSON change; deterministic CPU tests; UTF-8 no BOM; SFW.
