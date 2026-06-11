# 0-E Phase B runbook -- GPU gates for the on-ramp lane (agent-executable)

**Written 2026-06-11 by the 0-E follow-through agent after Phase A shipped.
Phase B starts ONLY when the planner creates `scripts\_otr_0e_gpu_go.txt`
(CS-4 fix + sweep remainder + supervised wan batch land first). Never touch
a server you did not start; never the Wan superset env (CS-3).**

## Phase A state (all pushed; HEAD b477415..9f66fac on v2.0-alpha)

- Blender **4.5.10 LTS** at `C:\ComfyUI-Models\tools\blender-4.5.10\`;
  `OTR_BLENDER_EXE` (User env) points at its exe; zip sha256 in
  LICENSE_RECORD.md. E-3 cube self-test **PASSED** (3-frame WORKBENCH).
- hy3d-2mv ALL-IN-ONE checkpoint at
  `C:\ComfyUI-Models\checkpoints\hunyuan3d-dit-v2-mv.safetensors`
  (sha256 == HF LFS oid, recorded). Embeds DiT + ShapeVAE + DINO encoder.
- DA-V2-SMALL snapshot in the HF cache (`HF_HOME=C:\ComfyUI-Models\huggingface`);
  still_parallax HF-cache snapshot resolution FIXED + live-proven offline.
- mesh_stage graph audited vs the INSTALLED core + official blueprint:
  ImageOnlyCheckpointLoader (slot1=CLIP_VISION, slot2=VAE) ->
  ModelSamplingAuraFlow(shift=1) -> KSampler(30/5.0/euler/normal) ->
  VAEDecodeHunyuan3D(8000/256) -> VoxelToMesh("surface net"/0.6) -> SaveGLB
  (`<prefix>_<counter:05>_.glb`; prefix must live under the comfy output dir).
  Widgets are EXPLICIT (V3 nodes backfill nothing). Suite 4100/0 green.

## B-0 -- CLAIM GUARD (collision safety; multiple watchers may exist)

Before ANY Phase B action: if `scripts\_otr_0e_phase_b_claim.txt` EXISTS,
another session already runs Phase B -- STOP and exit quietly. Otherwise
write it (session id + timestamp) FIRST, then proceed. Delete it only on
Phase B completion (leave it on failure, with a FAILED line appended, so
the next watcher escalates to the operator instead of re-running blind).

## B-1 -- E-1 VRAM probe (replaces the DRAFT 8000MB registry row)

1. Confirm the go-file exists. If a stale SOAK server still holds :8000
   (`python ... main.py --port 8000`), stop ONLY that process (it is the
   soak server, not Comfy Desktop).
2. Launch YOUR server detached with the launcher (it pins :8000 and the
   real output tree): env first
   `OTR_ENABLE_LTX_VIDEO=1 OTR_ENABLE_LTX_ORBIT=1 OTR_ENABLE_STILL_PARALLAX=1
   OTR_ENABLE_MESH_STAGE=1`, then
   `scripts\_otr_soak_server_launch.cmd <log>` (default lane; NEVER `WAN`).
3. Run `scripts\_otr_0e_probes\otr_b1_server_probe.py [portrait] [8000]`
   (ComfyUI venv python). It submits the EXACT adapter graph via /prompt
   (the server validator = the definitive INPUT_TYPES proof), samples host
   NVML at 10Hz, and reports: baseline / host peak / delta / post floor /
   ceiling_ok_14_5gb / SaveGLB outputs. Fallback (no server):
   `otr_b1_vram_probe.py` runs MeshStageEngine.render_clip standalone --
   set OTR_MESH_CACHE_DIR under the INSTALL output dir or call
   folder_paths.set_output_directory first (SaveGLB containment).
4. Replace `vram_estimate_mb: 8000` in
   `nodes/_otr_video_engines/registry.py` (mesh_stage row) with the
   measured peak delta rounded UP to the next 500MB; drop the DRAFT
   comment; suite + Bug Bible; commit+push; RESTART your server (module
   cache -- .py edits never hot-load).

## B-2/B-3 -- acceptance render == ONE sweep leg per engine (pass02 E-6)

Per engine, with your server up and the env flags set in BOTH server and
sweep process:

    python scripts\otr_coverage_sweep.py --only music_visual_ltx_orbit
    python scripts\otr_coverage_sweep.py --only music_visual_still_parallax
    python scripts\otr_coverage_sweep.py --only music_visual_mesh_stage

(`--only` is a substring filter on `sweep_<slot>_<engine>`; music slot
first for mesh_stage per the on-ramp table -- character beats wait for
look-QA.) The sweep gates each leg: playable obs final, byte-identical
master audio, output hygiene, VRAM ceiling. Then per leg output mp4 build
a contact sheet (first/mid/last frame, ffmpeg select+tile) for the
operator's look-QA; verify frames == ledger target and 1472x832 in the
mesh_stage leg trace (canonicalize enforces it -- look for the
directory-clip validator line, no fallback restamps).

## B-4 -- close out

Update `3D_TOOLKIT_PLAN.md` section-0 LIVE STATUS (0-E line), the
otr-build-tracker artifact, and the handoff doc (COORDINATE: the planner
window holds uncommitted live edits to VIDEO_BUILD_HANDOFF.md -- do not
clobber). Commit+push docs. Verify HEAD==origin, no 0-byte, no BOM, AST.

## LEFT FOR THE OPERATOR (tee up, never do)

Look-QA on the three engines' renders + contact sheets; LICENSE_RECORD
sign-off boxes (license + NOTICE text are on file beside the record);
any default-on flip; Comfy Desktop relaunch; v2.0-alpha-stable tag.
