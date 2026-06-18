# Coverage architecture -- wiring v1 (pass01, hardened + SHIPPED slice)

Panel: gpt-4.1 + gemini-pro-latest (direct APIs -- OpenRouter launcher stalled).
Both converged: REJECT Candidate B (pure convention) on the fixed-path data-loss
bug; adopt Hybrid-A (capability with a base default). Claude grounded every claim
against the real code (judgment: pass01_judgment.md). This is the final design;
the slice marked SHIPPED is implemented + green.

## Decision: Hybrid-A -- ONE capability, base default, decided in one place
A video/3D lane consumes the role's SELECTED image still iff it consumes a still
-- decided by ONE attribute the dispatcher reads, never a per-(image,video)
whitelist. Coverage is DERIVED (selected-image x accepts_still-video), filtered by
the existing role_compat. New video engines inherit `accepts_still=True` from the
base -> they accept the chosen image automatically ("one and done").

## SHIPPED slice (implemented, suite + Bug Bible green)
1. `MotionEngineBase.accepts_still = True` (motion_common.py) -- every in-process
   motion lane (ltx_video, humo*, wan*, ltx_av_talk) accepts the selected still by
   default. THIS is what lets a flux2/flux still drive a SILENT ltx_video i2v clip
   (the operator's "flux2 images on LTX"): ltx_video already reads an optional init
   image (eng_ltx_video.py:431/829) but did not declare it -> the dispatcher skipped
   the still. Now it accepts.
2. Explicit opt-OUTs: `LtxAvMusicEngine.accepts_still = False` (audio-reactive, no
   still) + `VisualizerEngine.accepts_still = False` (CRT-scope floor) -- preserves
   the accessible "all-procedural episode invokes NO image model" path.
3. `engine_consumes_still(eng)` (otr_image_gen_dispatcher.py): capability wins
   (`accepts_still` if declared), else DUAL-READ fallback to `"init_image" in
   required_inputs` (humo/wan/ltx_av_talk/still_parallax/mesh_stage + the dark 3D
   talkers keep working unchanged with no edit). `_still_needed_for_role` delegates
   to it; the bare `except: return True` is now a SPECIFIC catch + LOUD
   `log.warning` (no silent fallback). Tests added: ltx_video->True,
   ltx_av_music->False, capability-vs-dual-read.

## Grounded panel items folded
- [gpt#3 / gem#4] LOUD except -- DONE (no silent force-True).
- [gem#1 / gpt#4] Candidate B fixed-path overwrite = data loss -> REJECT B. The
  dispatcher already passes per-beat stills explicitly (still_b000/b001/b005 keyed
  per object), so A keeps beat granularity. CONFIRMED.
- [gpt CUT#1 / gem CUT#2] Drop `still_input_name` + `still_kind` -- CUT. The init
  input name is always "init_image"; the 3D lock keeps the working
  `requires_mesh_portrait` boolean. No 20-engine string-enum migration.
- [gem CUT#1] Drop central `registry.usable()` -- role_compat already is the role
  filter; no redundant matrix.

## Rejected / corrected (panel MISREADS, grounded)
- [gem#3] "_ROLE_TO_VIDEO_SLOT missing announcer/character_3d" -- MISREAD: the real
  role keys are announcer_visual / character_video (both present, dispatcher.py:283).
  No change.
- [gem SHOULD#2] "dark 3D scaffolds lack a base -> crash on the default" -- not a
  crash: `getattr(eng,"accepts_still",None)` returns None -> dual-read falls back to
  their `("audio_ref","init_image")` required_inputs -> True. They work unchanged.

## Deferred follow-ups (additive; NOT needed for the deliverable)
- [gem#2 verify-at-build] `optional_inputs` so role_compat sees an OPTIONAL
  init_image. Not blocking now: ltx_video's roles (music/announcer/scene_broll)
  already supply init_image because co-resident required-init engines (humo/wan)
  share those roles, so role_compat already admits ltx_video. Revisit only if a
  future lane needs a role that supplies no init_image.
- Set `accepts_still=True` on the static-still cheap families (flux_still /
  station_card / still_kenburns) if the operator wants them to show the SELECTED
  image too (today they keep their own still behaviour, unchanged).
- Full Decision-3/5 (central usable(), retire requires_mesh_portrait onto a kind).

## Invariants preserved
No silent fallback (LOUD except); role_compat unchanged (still the role filter);
model-agnostic; single-resident unchanged (metadata only); cold-import clean (plain
attrs); workflow JSON untouched (no node/widget change); UTF-8 no BOM; determinism.
