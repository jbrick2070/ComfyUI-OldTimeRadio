# Claude anchor review -- r1 (all-engines-all-slots audit) -- GROUNDED

VERDICT: the architecture is RIGHT in production; the defects are (1) a soak/registry eligibility
DRIFT and (3) the soak not driving the canonical JSON (the drift's root), plus (2) stills rendering
black. All three must be in the plan. Grounded.

## CONFIRMED (grounded)
- DEFECT 1 (drift) REAL: `role_compat.engine_fits_role` = capability-only, model-agnostic (the legacy
  `roles` whitelist is "NO LONGER a gate", 2026-06-22) -- used by PRODUCTION (OTR_VideoDirector
  `_resolve_and_validate`, ShotLock, image director). `registry.engines_for_role` STILL filters
  `role in engine.roles` (the stale whitelist) -- used by the soak `build_profile`. PROVEN divergence:
  ltx_video(required_inputs=('text_prompt',)) -> engine_fits_role(character_video)=TRUE but
  engines_for_role('character_video') EXCLUDES it. => production allows it; the soak fills still_flat.
- DEFECT 3 (canonical JSON) REAL + is the ROOT: the soak synthesizes a profile + uses engines_for_role
  instead of loading otr_scifi_16gb_full.json through the real director path. Rebuilding the soak on
  the canonical JSON (set node-87 picks via the REAL director/role_compat) makes soak-eligibility ==
  production BY CONSTRUCTION, and satisfies CLAUDE.md S0. SUBSUMES build_profile. Merge with the
  already-converged docs/2026-06-29-coverage-soak/COMBO_SOAK_CONVERGED_PLAN.md (bake story+audio from
  the canonical JSON, vary node-87 picks, render stills+video+upscale->obs).
- DEFECT 2 (black stills) REAL: still_pan/still_flat dark-floor when the minted scene still doesn't
  reach them. Eligibility != correct rendering -- the audit must verify BOTH (eligible AND renders
  content). Fold docs/2026-06-30-black-clips/.

## MUST-FIX (arc)
M1. The fix must be at the SOURCE, not just the soak: kill the eligibility drift so `engines_for_role`
    (and every consumer) agrees with `role_compat` (capability). Otherwise a future non-soak consumer
    of the whitelist re-drifts. Preferred: route engines_for_role through
    `role_compat.filter_engines_for_role`; keep `default_roles` ONLY for the auto-default pick.
M2. Rebuild the soak on the canonical JSON (Defect 3) -- the structural fix; it makes the audit's
    "all engines all slots" claim PROVABLE through the real path.
M3. The audit MATRIX must be capability-derived: enumerate EACH engine's `required_inputs` and compute
    eligible(engine, role) = required_inputs <= role_available_inputs[role]. Flag the LEGIT incompat
    cells (capability-grounded, e.g. a base_clip_ref-requiring engine in background_abstract which
    supplies only text_prompt) vs the false whitelist exclusions.
M4. "Used CORRECTLY" = eligible (capability) AND renders real content in the slot (Defect 2 still->init
    + aspect handling: humo portrait pillarbox in a wide slot must not break; LTX-REGR motion).

## SHOULD-FIX
S1. A parametrized matrix TEST: every registered video engine eligible (capability) in all 3 user
    slots + the director accepts it + (post-rebuild) the canonical-JSON soak fills it (no silent
    still/floor swap). This is the regression guard for "no preferred path".
S2. Enumerate every consumer of engines_for_role / the `roles` whitelist (soak, combos, image director,
    ShotLock, capability_profiles, tests) for the migration.

## VERIFY-AT-BUILD (-> r2/r3)
The exact required_inputs per engine; the legit incompat cells; the canonical-JSON soak seam (node-87
pick via the director vs profile-applier); the Defect-2 still->ledger key.
