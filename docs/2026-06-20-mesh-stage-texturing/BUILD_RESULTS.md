# Textured 3D hero beat -- BUILD RESULTS (2026-06-20 night)

## Shipped + pushed: `8de1057` on `v2.0-alpha` (HEAD == origin)

Core PoC capability landed green: `mesh_stage` now renders a TEXTURED
turntable mesh (single-view vertex-color projection) and composites it over a
GENERATED, image-model-AGNOSTIC scene plate. Suite 4625 pass / 33 skip, Bug
Bible 16/7/3, +13 unit tests, audio spine byte-identical untouched.

### What changed (chunks 2 + 4 + 5, paired so no dead wiring)
- **Projection (`scripts/otr_mesh_stage_blender.py`):** `--portrait` loaded via
  `bpy.data.images.load`; a per-VERTEX (point-domain) `otr_proj` color attribute
  painted from the front Y/Z projection, set active+render so WORKBENCH
  `color_type='VERTEX'` draws it. GLB stays geometry-only (MESHER_VERSION NOT
  bumped). Bounded hero arc (`_build_turntable` start_angle+arc, clamped to
  `MAX_ARC_DEGREES=45`, `frames==1` -> one keyframe). Selftest projects a
  deterministic gradient and fails nonzero unless a non-uniform attribute exists.
  Pure helpers (`arc_keyframes`, `project_uv`, `sample_image`, `clamp_arc_degrees`)
  are CPU-unit-tested.
- **Command plumbing (`eng_mesh_stage.build_blender_cmd`):** `--portrait` /
  `--start-angle` / `--arc-degrees` appended only when set (legacy invocation
  byte-identical); `render_clip` passes the resolved still.
- **C1 stamp (`render_driver.build_clip_manifest`):** mesh_stage directory rows
  get `bg_still_path` from the per-beat scene still (the existing per-beat
  coverage already mints one via the per-role image engine -> AGNOSTIC),
  fail-closed `os.path.isfile` + LOUD warn on absence.
- **C1 composite (`otr_silent_composite.py`):** `bg_still_path` carried through
  `plan_timeline_segments` into the segment dict; `_encode_segment_from_dir`
  gains a still-aware `-loop 1` background branch. Zero new graph links/widgets
  (rides the existing 92->84 manifest channel). Every non-mesh beat omits the
  field -> floor/black background byte-identical.

### REAL-BLENDER VALIDATION (CPU-only, no GPU, did NOT touch the running soak)
Ran the stage selftest through the pinned Blender 4.5.10
(`C:\ComfyUI-Models\tools\blender-4.5.10\blender.exe`), WORKBENCH:
- exit 0; 3 RGBA frames of DIFFERING sizes (47025 / 49224 / 50279 bytes) ->
  the bounded arc actually moves the camera AND the projection is non-uniform.
- `selftest_proof_frame_0001.png` (in this dir) shows the cube TEXTURED with the
  projected gradient (green/teal/blue/magenta), NOT flat gray -> the
  vertex-color projection works visually in real Blender.

This validates the projection + bounded arc + Workbench VERTEX render
end-to-end. The only unvalidated piece left for the full GPU smoke is the hy3d
mesher (GPU) + projecting a REAL portrait onto a REAL mesh.

## GPU SMOKE -- PASSED (2026-06-20, box reset clean, FLOOR lane on :8000)
After the 864-word soak hit its 8h cap and freed the GPU, reset the box (killed
the two :8011 soak servers, port empty, GPU 1.8 GB baseline) and booted a fresh
FLOOR-lane headless server with `OTR_ENABLE_MESH_STAGE=1`. Ran the
single-engine smoke against a REAL FLUX portrait (`c02_portrait.png`):

    python scripts/_otr_single_engine_smoke.py --engine mesh_stage --frames 25 \
        --portrait <c02_portrait.png>

Result: **`ok:true`**, elapsed 42.2 s, a 25-frame straight-alpha directory clip
(5.9 MB, exists=true), **`vram_used_mb`: 2587 << 14500 ceiling**. The proof frame
`gpu_smoke_meshstage_frame.png` (this dir) shows the hy3d mesh TEXTURED with the
projected portrait (the face is clearly visible) on a transparent background --
NOT the flat-gray "plaster of paris" blob. The untested GPU chain (portrait ->
hy3d-2mv mesher -> vertex-color projection -> Blender turntable -> RGBA frames)
works end to end. NOTE: the hy3d single-view mesh is irregular and the
camera-arc framing wants tuning (a look-QA follow-up), but the PROJECTION is
proven on a real mesh + real portrait.

## REMAINING (the activation wiring -- now GPU-unblocked)
These intentionally were NOT landed blind tonight -- they need the GPU render
loop, which the all-night 864-word soak occupies until it finishes:

1. **Chunk 6 -- ledger trigger:** route the beat whose dialogue-slot id ==
   `meta.dramatic_state.costly_choice_beat` (pattern `d\d{3}`) to `mesh_stage`.
   The no-portrait -> `still_parallax` fallback is ALREADY in place (render_clip
   raises FileNotFoundError; `fallback_engine="still_parallax"`; classify_failure
   = DEPENDENCY_MISSING). The remaining piece is the per-beat engine-selection
   override seam (ShotLock / VideoDirector / render_driver) -- ground it with the
   files open, build pure + unit-test the selection, then GPU-validate.
2. **Chunk 7 -- JSON wiring:** verify `mesh_stage` is selectable (not `_HIDDEN`)
   on OTR_VideoDirector node 87; set the 3D-beat role engine + image_models;
   check node 88 granularity for the role. APPEND widgets only (BUG-LOCAL-097);
   re-validate (OTR_WorkflowValidator + round-trip + link/widget audit).
3. **Chunk 3 -- 2D-ellipse contact shadow:** deferred to the GPU-tuning pass
   (aesthetic sizing/opacity wants the real render to tune). Default-off,
   byte-identical when off.
4. **GPU smoke acceptance** (pass02 ACCEPTANCE): reset box (CLAUDE.md sec 4),
   `OTR_ENABLE_MESH_STAGE=1`, render the costly-choice hero beat textured over a
   generated plate, fits <= 14.5 GB single-resident, `test_audio_byte_identical`
   green at the OTR_MasterAudioMux boundary, obs_publish OK, confirm the asset
   on disk.

## PINs resolved
- mesh_stage `required_inputs==("init_image",)`, `uses_still=True`,
  `family="image_to_video"` (already correct -- the portrait-agnostic test holds).
- The scene plate is the per-beat scene still keyed by beat_id in
  `ledger["images"]["images"]` (kind `scene_*`), read by `_still_index`.
- SaveGLB pattern + cache root unchanged (geometry-only GLB).
- Blender color-attribute API confirmed working on 4.5.10 (FLOAT_COLOR / POINT /
  active_color + render index; WORKBENCH color_type=VERTEX renders it).
