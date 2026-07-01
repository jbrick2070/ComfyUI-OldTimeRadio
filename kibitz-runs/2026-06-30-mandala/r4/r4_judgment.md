# r4 JUDGMENT -- mandala engine (final convergence)

Panel: Codex + Antigravity, both "yes-with-fixes", NO new ARC issues. r4 did its job -- it surfaced 3
real coding locks r1-r3 missed. All grounded. ARC CONVERGED -> plan is build-ready after these folds.

## NEW LOCKS (r4 convergence delta)
1. **`render_aspect = "wide"`** (Codex #2, GROUNDED eng_viz_rainbow.py:53). Every registered video engine
   must declare it; test_still_aspect_and_labels.py enforces. Also mirror
   `declared_isolation = _MC.ISOLATION_IN_PROCESS` (L54). ADD both class attrs.
2. **`surface.flush()` before `get_data()`** (Codex #4 + Antigravity #2, unanimous; the proto does it at
   L182 but the r3 handoff omitted it). Without flush some backends return blank/partial frames.
3. **`apply_crt_post_rgb` signature = `(rgb, scanlines, vignette, fi, rng_key, vol=0.0)`** (Codex #3 +
   Antigravity #1). Needs `fi` (scope_draw._rng keys on (key, fi, salt) -> without it grain is FROZEN)
   and `vol` (dynamic noise intensity ~ `int(4 + vol*10)`). Deterministic on (rng_key, fi); HxWx3 uint8;
   no in-place mutation. Rename scan/vig -> scanlines/vignette (no naming divergence).

## HOUSEKEEPING (fold into the plan; both flagged)
- DELETE the stale "OPEN FOR r2" section -- it still says "pick by the measured budget" and contradicts
  the locked PIL-roundtrip + numeric budget. Its content is already resolved in the locked sections.
- Retitle "production engine" -> "selectable mandala engine" (r3 locked it opt-in, not a saved default).
- Clarify DECISIONS #2: "NO module-scope cairo import; import-probe in assert_usable + import inside
  render_clip." No cairo in module-level type annotations (use string literal 'cairo.Context').
- Version pins (pycairo 1.29.0 / cairo 1.18.4) = provenance only, NOT a build pin.

## RADIUS -- DO NOT hard-freeze the coefficients (judge override of Antigravity SHOULD #1)
Antigravity proposed exact 0.5x-scaled ring/spoke/band formulas. I am NOT locking specific numbers: the
operator is ACTIVELY tuning the look (asked 2026-06-30 for denser/thicker bands). Lock only the INVARIANT
-- core rings + tuning-eye must not clip on 1472x832 (outer core <= ~0.33*min(w,h)); the outer spectrum
band MAY bleed. Exact coefficients = a build-time look pass WITH the operator, not a frozen constant.

## CONVERGED. No r5. The plan (MANDALA_ENGINE_PLAN.md) after these folds is build-ready for a coder window.
Build gated behind the operator's GO_FORWARD ordering (still_parallax rip-out, visualizer->viz_green
rename, HuMo/mesh work) unless the operator pulls it forward.
