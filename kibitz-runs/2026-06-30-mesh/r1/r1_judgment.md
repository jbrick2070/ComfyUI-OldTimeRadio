# r1 JUDGMENT -- mesh_stage MIN-ACCEPT (grounded; r1-only per operator)

Panel: Codex (grounded "no"). Claude anchor. Antigravity/Claude-Code absent at synth -- r1-only scope
(operator: "maybe I just do an r1"), Codex + anchor are sufficient.

## LOCKED (build-ready direction)
1. **Radio subject is minted in PROMPT-GEN, not render_driver** (Codex #1, grounded). The no-character
   music fodder becomes "a single emblematic object..." in `otr_meta_brief_image_prompt.py:553,569-573`
   (`_subj_id` @:887); `render_driver.build_request_from_shot:863-878` only RESOLVES existing fodder by
   id/path. FIX: make the music/bookend role branch in prompt-gen emit a VINTAGE RADIO mesh_fodder object;
   render_driver then consumes it. CUT "canned radio mesh" -- prompt-level radio fodder is the smallest
   root fix (keep the LOUD missing-fodder guard).
2. **Headroom needs a MEASURABLE contract** (Codex #2, grounded). Blender normalizes mesh longest-dim ->
   1.0 + centers at origin (`scripts/otr_mesh_stage_blender.py:224-240`), fixed radius/elevation
   (:57-58,:366-383); compositor scales fg -> output centered (`otr_silent_composite.py:621-637`). "Camera
   back / composite fit" is too vague. DEFINE: foreground alpha bbox max height <= N% of frame + >= N px
   top margin; IMPLEMENT in the Blender camera radius/elevation (the RIGHT layer -- a composite crop just
   zooms a full-height render + loses detail); ADD a proof-frame bbox test.
3. **Routing** (Codex #3): the workflow has visualizer/humo_14B_169, not mesh_stage. State the exact
   force-map path while LOADING the real JSON (or a gated default promotion) -- don't assume mesh_stage is
   selected.
4. **Bookend detection** (Codex #4): define music_open + pure-music close by role/beat id; test BOTH;
   decide whether bookends share a canonical `radio_host` mesh id for identity continuity (cache keys are
   per-beat `obj_<beat>` today).

## SCOPE (operator r1-only)
- CUT Trellis/WorldMirror discussion (operator-gated, separate). CUT broad material/lighting/turntable/
  background exploration -- those paths already work; only accept optional changes that DIRECTLY serve
  radio readability or headroom. Add a before/after proof artifact (opener + close + one character frame,
  each with alpha bbox stats). No r2-r4.
