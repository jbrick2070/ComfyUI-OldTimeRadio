# R2 judgment + convergence call (Claude, sole judge)

Panel: gpt-5.5-20260423, gemini-3.1-pro-preview-20260219, deepseek-v4-pro-20260423.
Spend this pass ~$0.3227. Campaign total R1+R2 ~$0.6067.

## Accepted (grounded CONFIRMED -> folded into pass02)
- **render_aspect refinement (Gemini #1) -- the headline.** ltx_audio_in is
  render_aspect="wide" (CONFIRMED eng_ltx_av:153); a wide engine must take the
  beat's WIDE scene still, never the vertical portrait (pillarbox bug @873-883).
  So ltx_audio_in routes like flux_still/flat_still (wide, char-aware), NOT a
  portrait route. This LOWERS risk: join the existing wide-still branch instead of
  a full rewrite. Folded (pass02 design + C1b).
- **Classifier needs line for char_id (GPT #1, Gemini #2):** CONFIRMED char_id =
  shot.get OR line.get. Signature `_is_character_face_beat(shot, line)`; role
  primary; char_id fallback only when role missing (GPT #9). Folded (C1e).
- **Prompt fallback is family-gated (GPT #8):** CONFIRMED the char-fallback is under
  `elif _fam=="audio_driven_face"`; ltx_audio_in won't hit it. Re-gate on is_char_face.
  Folded (C1d).
- **_uses_ambient_master_audio family-keyed (GPT #6, DeepSeek #1):** CONFIRMED @730.
  Pass is_char_face; False for char beats. Folded (C1c).
- **_LTX_OPEN_ENGINES must REMOVE the old names (DeepSeek #3):** else the health
  check never fires for ltx_audio_in opens. frozenset({ltx_video,ltx_audio_in}).
  Folded (C2c).
- **VALIDATED_ENGINES two-step is a hidden-engine hazard (GPT CUT #1):** add
  ltx_audio_in in the SAME commit (provenance) rather than a follow-up. Folded (C2b).
- **grep ALL runtime refs before delete (GPT #11); render_single clamp gap (GPT #12);
  default_roles assignment (DeepSeek #4).** Folded (verify-at-build + C2a).

## Rejected / reframed
- **GPT #5 "extend _assert_family_inputs_satisfiable for required init_image" --
  REFRAMED, not blocking.** CONFIRMED the family check (FAMILY_REQUIRED_INPUTS by
  family, @1326) won't catch ltx_audio_in's engine-level init_image. BUT render_clip
  ALREADY raises GraphExecutionError on empty init_image + NO fallbacks -> the
  missing-required-still is ALREADY terminal+LOUD. A pre-GPU check is a nicer
  message, not a correctness gap. The real fix is to MINT+ROUTE the still (C1a/C1b)
  so it never triggers. Pre-GPU gap-check = nice-to-have, not a build blocker.
- **GPT full-capability route enum / runtime enumeration (Verify-at-build #1 as
  runtime) -- CUT from runtime, kept as a TEST.** The enumeration pins behavior as a
  table-driven test fixture, not production logic (GPT CUT #2). Folded as C1 test.

## Convergence -- STOP at R2; build now
R1 fixed the ARCHITECTURE (role-driven). R2 fixed the IMPLEMENTATION (render_aspect ->
join the wide-still branch; classifier signature; the three call sites; the
input-gap reality). The design is now implementation-ready and LOWER-risk than
pass00 (no full render_driver rewrite). The residual R2 items are all
"verify-at-build" / wiring mechanics -- exactly what R3 (wiring) and R4 (residual
defects) would re-derive, and they are covered concretely by: the verify-at-build
grep+enumeration, the table-driven routing-matrix test, the workflow validator, and
the full regression suite + Bug Bible. Per the roundtable rule "stop at convergence,
don't grind passes," R3/R4 are folded into the build gates rather than run as
separate fan-outs. Proceeding to CHUNK 1.
