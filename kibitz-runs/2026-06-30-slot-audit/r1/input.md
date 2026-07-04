# ALL ENGINES x ALL SLOTS -- AUDIT + SPRINT FIX PLAN (for kibitz r1-r4)

> Operator architecture invariant (2026-06-30): "ANY video model INCLUDING STILLS can work in ANY of
> the THREE slots. There is NOT one preferred path." Audit that this holds end-to-end and produce a
> sprint-ready fix. NO retest until r1-r4 converges. The panel crawls the REAL code; Claude grounds +
> judges. Output: a complete eligibility+rendering audit matrix + a sprint plan that makes every video
> + still engine usable CORRECTLY (eligible AND renders real content, not black/frozen) in all 3 slots.

## THE 3 USER SLOTS (node-87 OTR_VideoDirector) -> roles
- announcer_video_model  -> announcer_visual
- music_video_model      -> music_visual
- other_beats_video_model (+ the 3 Route-A per-role: character_video / scene_broll / background_abstract)

## CONFIRMED DEFECT 1 -- TWO eligibility rules that DISAGREE (the drift)
- `role_compat.engine_fits_role(desc, role)` (nodes/_otr_shared/role_compat.py): CAPABILITY-only
  (`required_inputs <= role_available_inputs[role]`); the legacy per-engine `roles` whitelist is
  EXPLICITLY "NO LONGER a gate" (operator 2026-06-22, model-agnostic). PRODUCTION (OTR_VideoDirector
  `_resolve_and_validate`, ShotLock, image director) uses THIS. -> correct, model-agnostic.
- `registry.engines_for_role(role)` (nodes/_otr_video_engines/registry.py): filters
  `role in getattr(engine, "roles", ())` -- the STALE per-engine `roles` WHITELIST.
- PROOF of drift: ltx_video.required_inputs=('text_prompt',) ->
  engine_fits_role(ltx_video,'character_video')=TRUE, but engines_for_role('character_video') EXCLUDES
  ltx_video (its `roles` omit character_video). The soak `scripts/_otr_cov_runner.build_profile` uses
  `engines_for_role` -> char_ok=False -> fills `still_flat` (frozen) -> the QA "character beats are
  stills". Production would have ALLOWED ltx_video. So the soak result mis-represents production.

## CONFIRMED DEFECT 2 -- still-carrier engines render BLACK in a slot (image legs)
Even when an engine is ELIGIBLE, it must RENDER CONTENT in the slot. `still_pan`/`still_flat` pan a
PROVIDED still; if the minted scene still does not reach them (`init_image` empty) they synthesize the
dark floor (0x0A0E14) -> BLACK. The image-engine legs (lumina/flux/qwen/z_image) rendered black; image
GEN works (3425 stills minted) so the break is the still -> carrier seam (dispatcher scene-still ledger
write-back / `still_pool_key`/`beat_id` key match). (Separate kibitz: docs/2026-06-30-black-clips/.)

## THE AUDIT TO PRODUCE (every video + still engine x every slot)
For EACH registered video engine {humo, humo_1.7B, humo_1.7B_169, humo_14B_169, ltx_video,
ltx_audio_in, wan_i2v, wan_ti2v, mesh_stage, still_pan, still_flat, still_motion, still_parallax,
station_card, abstract, visualizer} and EACH slot/role, determine:
- A. ELIGIBLE? per role_compat (capability) -- should be TRUE for all (required_inputs are subsets of
  every role's available inputs; verify each engine's required_inputs).
- B. ENGINES_FOR_ROLE agrees? (the whitelist) -- map the DRIFT cells (eligible-but-whitelist-excludes).
- C. RENDERS CONTENT? does the engine produce real video/animation in that slot, or fall to a still /
  black / frozen (the still->init seam; the aspect handling portrait-vs-wide; the LTX-REGR motion).
- D. The per-engine `roles` + `default_roles` declarations (the stale whitelist) -- list every gap.

## SPRINT FIX (to be hardened r2-r4)
1. KILL THE DRIFT: make `registry.engines_for_role` (and the soak `build_profile`) use the SAME
   capability rule as `role_compat` -- ONE model-agnostic source of truth. Either route engines_for_role
   through role_compat.filter_engines_for_role, or deprecate the per-engine `roles` whitelist entirely
   (keep `default_roles` for the auto-default pick only).
2. SOAK build_profile: put the engine-under-test in ALL 3 slots (the operator's "all 3 slots" intent),
   gated by CAPABILITY not the whitelist; the still carrier is only for a genuinely incapable engine
   (none, for the user set).
3. Ensure stills RENDER in every slot (Defect 2): the minted scene still must reach still_pan/still_flat
   (fix the dispatcher scene-still write-back / key match) so a still slot shows the image, never black.
4. Tests: a parametrized matrix test asserting EVERY registered video engine is eligible (capability)
   in all 3 user slots + the director accepts it + the soak fills it (no silent still/floor swap).
5. NO workflow-JSON change for eligibility (combo is the full registry already); audio byte-identical;
   no shim; registry IS the menu.

## OPEN QUESTIONS FOR THE PANEL (ground vs the code)
1. Enumerate EVERY consumer of `engines_for_role` / the per-engine `roles` whitelist (soak, director
   combos, image director, ShotLock, capability_profiles, tests) -- which must migrate to role_compat?
2. Are there engines whose required_inputs make a role genuinely incompatible (e.g. base_clip_ref for
   background_abstract which supplies only text_prompt)? List the LEGITIMATE incompat cells so the
   "all engines all slots" rule has the right, capability-grounded exceptions (NOT whitelist ones).
3. Aspect: humo is render_aspect=portrait (pillarbox); stills/ltx are wide. Does "works in any slot"
   require any aspect handling so a portrait engine in a wide slot is not broken (it pillarboxes today)?
4. Defect-2 exact seam: where the minted scene still fails to land in `images.images` under the lookup
   key for still-carrier beats (fold the black-clips diagnosis).
5. default_roles after the whitelist is killed -- does any auto-default break?

## INVARIANTS
Model-agnostic (no preferred path); 100% local; master audio byte-identical; UTF-8 no BOM; SFW; no
shim; registry IS the menu; per green chunk push. The plan must be SPRINT-READY after r4 (coder can
build it without re-deciding).
