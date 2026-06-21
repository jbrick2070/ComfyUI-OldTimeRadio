# pass02 judgment (coding + wiring + comfy-quirks; Claude = grounded judge)

## Panel
GPT-5.5 (full), Grok-4.3 (full), Gemini-3.1-pro (fragmented output -- token issue; usable point
only). ~$0.23.

## CONVERGENCE
GPT + Grok independently CONFIRM: the pass01 plan is correct but NOT YET implemented in the
grounded files (true -- mid-build), and IS code-ready once the implementation matches the plan.
Both list the same build set: add `--surface` to `parse_stage_args` + `surface=` to
`build_blender_cmd` + flip `render_clip` to default `surface="gradient"` (portrait behind
`OTR_MESH_PROJECT_PORTRAIT=1`) + add `_paint_gradient_onto_meshes` + `_smooth_mesh_normals` + the
`main()` state machine. The parser arg MUST land in the SAME change as the `--surface` append (else
argparse error). -> exactly the plan.

## ACCEPTED -- folded into the build
- [GPT/Grok] Implement parser-arg + cmd-param + render_clip flip TOGETHER (one chunk). Tests:
  default cmd has `--surface gradient` and NO `--portrait`; opt-in has `--surface portrait
  --portrait <still>`; omitted `surface` is byte-identical legacy.
- [Gemini] `main()` checks `mode=="selftest"` FIRST (projection + non-uniformity gate untouched),
  THEN the surface state machine. (Already the plan.)
- [Grok SHOULD] Re-entrancy guard: REMOVE any pre-existing `otr_proj` color attribute before
  `color_attributes.new(...)` -- fold into a shared `_activate_render_color`/paint helper used by
  BOTH the gradient and portrait paths.
- [Grok SHOULD/Q4] Defensive `mesh.update()` after setting `poly.use_smooth=True` so headless
  WORKBENCH reliably shades smooth (cheap; avoids a "did it actually smooth?" GPU surprise).
- [Grok OPTIONAL] `gradient_color` CPU test at EXACTLY +/-0.5 (boundary) + out-of-range clamp.

## VERIFIED BY CLAUDE (panel could not -- JSON not in grounding)
- Q1 wiring: `workflows/otr_scifi_16gb_full.json` has ZERO `mesh_stage`/`surface`/`portrait`/
  `fodder` references. mesh_stage is a RUNTIME registry pick via the OTR_VideoDirector dropdown.
  -> NO workflow-JSON change needed; change is adapter-internal + content. CONFIRMED.

## DOWNGRADED to verify-at-build (GPU smoke), not code
- [Grok Q2] premultiplied-alpha probe in `validate_frame_dir`: `film_transparent=True` produces
  STRAIGHT alpha and the prior v1.0 smoke composited the mesh frames cleanly; `validate_frame_dir`
  already asserts RGBA. A reliable premultiply probe is fragile -> verify visually on the re-smoke,
  don't add a brittle check.

## REJECTED (with rationale)
- [Grok CUT] Drop the `MESH_FODDER_NEG_SCAFFOLD`: KEEP. It is checked-in canon (a test asserts it)
  documenting what fodder must NOT contain; the POSITIVE scaffold already absorbs the load-bearing
  cues ("smooth solid form", "short tight neat hair") since only POSITIVE reaches the engine.
- [Grok Q6] Replace per-vertex gradient with a WORKBENCH STUDIO+matcap ramp: KEEP per-vertex. It is
  deterministic, directly controllable, and reuses the PROVEN vertex-colour mechanism the
  projection already ships; STUDIO lighting is less deterministic and a bigger unknown.

## Convergence verdict
Coding + wiring + comfy-quirks all covered. No open design forks; the remaining items are the
implementation itself + two defensive guards (attr-remove, mesh.update) + the GPU visual check.
CODE-READY. Build chunk B.
