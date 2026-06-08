# Subproject B (M5) -- B-SHIP: HuMo-2D character path; character_3d deferred

Date: 2026-06-08. Branch: v2.0-alpha. Milestone: M5 / B-ship (DONE).

## Decision

The Subproject-B 3D-mesh keystone (mesh -> ARKit-52 WRAP) is BLOCKED on real
assets that do not exist on this machine. Per operator decision (2026-06-08), M5
is rescoped to the already-shipped **HuMo-2D** character path (the sanctioned
keystone fallback). Full 3D-mesh `character_3d` (Hunyuan3D / TRELLIS +
ARKit blendshapes) is DEFERRED as a future opt-in engine; it drops into the same
model-agnostic platform as a peer when its build-probes pass.

## B 3D de-risk keystone -- run headless on the RTX 5080 (scripts/_otr_b_spikes/)

- selftest_harness: PASS (35/35 CPU harness checks).
- probe_b (manifold pre-screen): PASS (synthetic; no real meshes on disk).
- probe_c (KEYSTONE, mesh -> ARKit-52 WRAP): SYNTHETIC-smoke GO (2/25 = 0.08 <
  0.20 NO-GO bar) -- NOT BINDING. A binding verdict needs real meshes
  (`OTR_B_MESH_DIR`) + a real ARKit-52 template (`OTR_B_ARKIT_TEMPLATE_NPZ`:
  verts/faces/mouth_idx + `delta_<name>` x52); BOTH are absent across the machine.
- probe_d (A2F-3D onset): SMOKE-ONLY (no onsets supplied).
- probe_a (cu128 CUDA-ext sm_120): NO-GO -- sm_120 RTX 5080 detected, but ninja +
  a cu128 toolkit (`OTR_CU128_HOME`) are not installed (no ABI isolation possible).
- probe_e (render-spawn VRAM): PASS on the 5080 -- the AS-3 NVML lease +
  cross-process reclaim are proven (reduced 8000 MB footprint to honor the
  ceiling; peak 12134 MB; fully reclaimed below floor).

character_3d remains a future opt-in engine, blocked on: a cu128 build toolchain
(ninja + cu128 nvcc + `OTR_CU128_HOME` + a cu128 sidecar venv), ~25 real
generated meshes, and a real ARKit-52 blendshape template. Re-run
`scripts/_otr_b_spikes/probe_c_arkit_wrap.py` for the binding verdict once they
exist; a GO opens the `character_3d` adapter (B1).

## HuMo-2D shipping character path (confirmed -- no code change)

`humo` (audio_driven_face) is registered, selectable for `character_video` +
`announcer_visual`, the sanctioned fallback head for the deferred `character_3d`
(`resolver.py` `character_3d -> humo`; chain `humo -> latentsync ->
still_kenburns`), and opt-in via `OTR_ENABLE_HUMO`. `hunyuan3d_talk` is
unregistered (deferred). Nothing in live code defaults to the deferred
`character_3d` engine; making HuMo a flag-bypassing `default_role` would weaken
the fail-closed gate, so HuMo stays an opt-in peer (no model "primary"). The
platform code is UNCHANGED at 6a8892f.

## B-ship certification (RTX 5080, 2026-06-08)

- Full repo suite: 3705 passed / 25 skipped / 1 baselined `eng_chatterbox`
  env-fail (audio sidecar dep conflict, in EXPECTED_FAILED).
  `tests/test_audio_byte_identical.py` GREEN (9 passed / 1 skipped). Repo clean
  at 6a8892f.
- Live A-S7.5 full-episode soak via the `OTR_VideoRenderBatch` node (`/prompt`,
  ComfyUI executor thread), HuMo-2D as the character engine, run as TWO
  independent invocations:
  - run1 (oom_index=20): ok=True; episode_1 + episode_2 each = 40/40 real on-disk
    clips, 6 HuMo in-process renders, `character_3d` OOM -> humo -> latentsync ->
    still_kenburns (3 LOUD restamps @rev1), vram_peak 3874 / 3839 MB, frozen
    audio sha unchanged (`21aa71f6a4e5master_audio_pcm_marker`), render-twice
    deterministic.
  - run2 (oom_index=19, cache-bust): ok=True; same shape, vram_peak 3957 / 3848
    MB, audio sha unchanged.
- VRAM: the soak's reported `vram_peak` (the inter-engine-boundary / reclaimed
  measure) is <= 3957 MB (~3.9 GB) <= 14.0 GB across all 4 episodes. The
  transient machine-wide peak while a single HuMo-1.7B forward is resident is
  ~14.5 GB (the proven single-resident-heavy-engine budget; the 14.0 GB figure
  was the now-deferred 3D-sidecar sub-ceiling).
- The soak code (`render_driver` / `eng_humo` / `registry` / `motion_common` /
  `fallback` / `retry_taxonomy` / `resolver` / `otr_video_render_batch`) is
  byte-identical between the A-ship tag (30a428b) and HEAD (6a8892f) -- the
  cleanbreak touched only deleted legacy nodes, additive tests, and docs
  (verified via `git diff`). The live headless server ran that identical code.

Note: the soak is a render-path stress test, not a full-episode assembly. It
writes per-beat clips to TEMP + its report to `output/otr/aship/node_soak.json`;
it does NOT write to `output/otr/obs` (the full-pipeline baseline folder).

## Result

M5 ships via HuMo-2D. M0-M5 green = DONE. `character_3d` (3D mesh) deferred to a
future opt-in engine. Tag: `B-ship`.
