# pass01 judgment (Claude = grounded panelist + judge)

## Panel
~openai/gpt-latest -> openai/gpt-5.5-20260423; ~google/gemini-pro-latest ->
google/gemini-3.1-pro-preview-20260219. deepseek-v4-pro FAILED (empty content,
finish_reason=length -- reasoning model spent the budget on hidden tokens; the known
gotcha, needs --reasoning-effort none). Two diverse frontier families is enough -- the
value is the grounding, not the count. Spend ~$0.15.

## CONVERGENCE (both models, independently) -> CONFIRMED against the real code
- **[C] env-knob cache trap.** `mesh_cache_key(subject_id, portrait_content_hash(still))`
  carries NO geometry-affecting gen params (verified eng_mesh_stage.py:106-119); a new
  `OTR_HY3D_VOXEL_THRESHOLD` is ignored on a cache HIT and collides under the same key on a
  MISS. The pass00 claim "A/B without a code change" is FALSE. -> **CUT C for v1.1.**
- **[B/V2] headless `bpy.ops` context crash.** `_import_glb()` returns objects but sets no
  active object / selection; `main()` manages none (verified). `bpy.ops.object.shade_smooth()`
  needs a context. -> smooth via mesh DATA (`poly.use_smooth=True`), NO operator.

## ACCEPTED (GPT, code-verified)
- **[B] `--surface` default vs "byte-identical."** Stage with no `--portrait` renders flat gray;
  defaulting `--surface=gradient` changes render behavior even if command bytes match. -> parser
  default = **flat** (preserve omitted-arg legacy); `eng_mesh_stage` passes `--surface gradient`
  explicitly; drop the byte-identical claim for the engine's NEW default (it intentionally moves
  to gradient).
- **[B] surface-mode state machine** underspecified. -> define + enforce with LOUD errors:
  `gradient` paints gradient & ignores `--portrait`; `portrait` REQUIRES `--portrait` (p.error if
  missing) & projects; `flat` paints nothing (gray matcap).
- **[B] gradient world-space coords.** `_normalize_meshes()` rewrites object loc/scale and the
  projection samples `obj.matrix_world @ v.co` -> `_paint_gradient_onto_meshes` MUST do the same
  (`co = matrix_world @ v.co`, pass `co.z`), not raw local `v.co.z`.
- **[B] clamp `gradient_color`.** The "~[-0.5,0.5]" normalized range is an assumption, not a
  guarantee -> clamp `co_z` to [-0.5,0.5] inside `gradient_color` before the lerp; test the
  out-of-range case.

## CUT / REJECTED
- **C entirely** (the threshold knob). Cache-unsafe and speculative. The lumpy-artifact root
  cause is the FODDER (busy detail meshed as geometry) -> fixed by A; the FACETED look -> fixed by
  B's per-poly smooth. No mesher-param change ships in v1.1. (A future cache-aware mesher-tuning
  sprint can include gen params in the cache identity if needed.)
- Gemini's "append threshold to mesher_version" fix: technically correct but it would force a full
  re-mesh on every threshold value and balloon the cache -- only worth it inside a dedicated
  tuning sprint, not v1.1. Deferred with C.

## DECISIONS on the pass00 open questions
- V1 (smooth gradient across faces): per-poly `use_smooth=True` interpolates vertex colours
  smoothly in WORKBENCH VERTEX -> a soft ramp, not bands. Verify on the GPU re-smoke.
- V3 (gradient-only vs fodder-tint): operator said "simpler / basic gradients" -> **gradient-only**,
  no fodder tint. Drop the photo decal from the default path; keep projection behind
  `OTR_MESH_PROJECT_PORTRAIT=1`.
- V4 (geometry smooth modifier to melt lumps): NOT in v1.1 -- a real fused-mass lump is a fodder
  problem (A), and a smooth modifier erodes real features; per-poly normal smoothing is the safe
  visual win. Deferred.
- V5 (gradient direction): vertical Z, light-over-dark "museum sculpt". Simple + deterministic.

## Convergence verdict
One grounded pass; both models found the SAME two core defects (cache trap, headless context),
both CONFIRMED, no internal contradictions. Remaining items are build-time specifics, not open
design forks. **Converged -- no second live pass needed.** Build A + B (C cut).
