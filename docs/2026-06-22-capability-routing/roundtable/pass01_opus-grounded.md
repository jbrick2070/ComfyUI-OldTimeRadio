<!-- Claude (Opus) grounded code-aware panelist -- R1, written before reading the panel. -->

# R1 grounded panelist review (Claude) -- capability-based routing

## Headline
The capability declaration ALREADY EXISTS: each engine's `required_inputs` IS "what inputs it
consumes." "Declare once, model-agnostic downstream" is therefore mostly **subtractive**: drop the
redundant `roles`-whitelist gate in `engine_fits_role`, and let role-fit be PURELY
`required_inputs <= ROLE_AVAILABLE_INPUTS[role]`. That single change is, by construction, **additive**
to eligibility -- it can only STOP rejecting capability-compatible engines, never remove an existing
fit -- which is exactly the operator's non-regression bar.

## Grounded points
1. **Two gates, one redundant.** `engine_fits_role` requires `role in engine.roles` AND
   `required_inputs <= available`. The input match is the real safety (an audio engine still needs
   `audio_ref`). The `roles` whitelist is the over-restriction that blocked wan_i2v from the announcer.
2. **`default_roles` is a DIFFERENT concern** -- the AUTO-DEFAULT pick when the operator doesn't choose,
   NOT eligibility. Must confirm the descriptor's `roles` isn't just `default_roles` (which is empty for
   wan/ltx) -- if it is, that's the whole bug, and the fix is to make eligibility = capability, leaving
   `default_roles` only for the default pick.
3. **Audio specials fall out for free.** HuMo / ltx_av / character_3d / visualizer require `audio_ref`
   -> only announcer/music supply it -> they stay gated to those roles by capability, no list needed.
   This is precisely the operator's "HuMo/LTX-AV are specials, the rest is still+prompt."
4. **Non-regression is provable.** Dropping the `roles` gate cannot remove a fit (the input match is
   unchanged). Build the per-engine x per-role BEFORE/AFTER table from the real registry -- every BEFORE
   fit must remain; deltas are additive only.

## Risks the panel must pressure-test (where non-regression could actually break)
1. **ASPECT.** wan_i2v / ltx_video are `render_aspect="wide"`; HuMo is portrait. The input-capability
   model does NOT encode aspect. If a role/slot implies a portrait still and a wide engine is now
   eligible, capability-match alone could allow a visually-wrong pairing. MUST confirm aspect is handled
   downstream (the director already derives per-role aspect) and is NOT what `roles` was silently
   encoding.
2. **Is `roles` encoding anything non-input?** Before dropping it, enumerate each engine's current
   `roles` vs its capability-derived set. If any engine's `roles` is NARROWER than capability for a
   deliberate reason (creative-appropriateness, a known-bad pairing), that intent must move into an
   explicit capability/constraint, not be lost.
3. **"Declare once" spans TWO modules.** Capability lives in `role_compat` (engine_fits_role) AND
   `render_driver` (FAMILY_REQUIRED_INPUTS, by family). To truly declare once, FAMILY_REQUIRED_INPUTS
   should DERIVE from the engine `required_inputs` (or be asserted equal in a test) -- else we have the
   same scatter the operator is trying to kill.
4. **Image engines** -- same principle; confirm they have the analogous whitelist gap and unify.

## Recommended v1 (minimal, non-regressive)
Make `engine_fits_role` capability-only (drop the `role in roles` gate); keep `default_roles` for the
auto-default pick; keep the render_driver FamilyInputGap as the LOUD backstop; add the before/after
table as a test; assert FAMILY_REQUIRED_INPUTS == engine required_inputs. Re-render the 100% Wan to
confirm wan now drives the announcer AND HuMo still does too.
