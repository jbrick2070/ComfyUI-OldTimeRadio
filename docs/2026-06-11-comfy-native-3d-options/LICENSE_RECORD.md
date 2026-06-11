# 0-E License Record (ticket E-7) -- operator-visible, gates any default-on

**Status: OPEN -- awaiting the operator's review sign-off.** Per the 0-E
spec, NO 0-E engine may become a DEFAULT for any role until the operator
reviews this record and signs the box at the bottom. All three engines are
shipped SELECTABLE-NOT-DEFAULT (empty `default_roles` + opt-in env flags),
so nothing here blocks selecting them for look-QA.

## mesh_stage -- Hunyuan3D-2mv (THE GATING REVIEW)

- Model: `hunyuan3d-dit-v2-mv` (Tencent Hunyuan3D 2.x, multi-view DiT),
  driven through ComfyUI CORE nodes (compile-free, in-process).
- License: **Tencent Hunyuan community license** (NOT OSI; NOT
  commercial-clean by default). `commercial_clean=False` in the registry
  until this review closes.
- The two clause families the operator must verify against the EXACT
  license text shipped with the checkpoint (VERIFY-AT-REVIEW -- do not
  trust this summary):
  1. **Scale threshold** -- the community grant typically excludes
     licensees above a monthly-active-user threshold (100M MAU in other
     Tencent Hunyuan community licenses). OTR is a personal/open project,
     expected FAR below any threshold, but confirm the figure in the
     license file that ships with the v2-mv weights.
  2. **Territory clauses** -- Tencent Hunyuan community licenses have
     carried territory exclusions (e.g. EU/UK/South Korea in prior
     Hunyuan releases). Confirm whether the v2-mv text carries them and
     whether distribution of OUTPUTS (rendered episodes) vs the MODEL
     differs. Operator is in the US (Los Angeles); the concern is only
     for redistribution/publication scope.
- Outputs: rendered FRAMES of a generated mesh appear in episodes; the
  mesh itself (GLB cache) stays local and is never redistributed.
- Acceptable-use policy: the Tencent AUP applies; OTR's SFW invariant
  already constrains content harder than the AUP.
- **CHECKPOINT PINNED (2026-06-11, ticket A4):**
  `C:\ComfyUI-Models\checkpoints\hunyuan3d-dit-v2-mv.safetensors`
  (4,928,151,562 bytes), sha256
  `d36f5881bcdc56726b73e517cd444c13c60732431622da7268145355c8d38e9c`
  (verified MATCH vs the HF LFS oid). Source:
  `Comfy-Org/hunyuan3D_2.0_repackaged` `split_files/
  hunyuan3d-dit-v2-mv_fp16.safetensors` -- BYTE-IDENTICAL (same LFS oid)
  to `tencent/Hunyuan3D-2mv` `hunyuan3d-dit-v2-mv/model.fp16.safetensors`,
  so the Tencent license applies unchanged. ALL-IN-ONE verified from the
  safetensors header: `model.*` (DiT) + `vae.*` (ShapeVAE) +
  `conditioner.main_image_encoder.*` (DINO image encoder) -- loads via
  core `ImageOnlyCheckpointLoader`, no separate CLIP-vision file.
- **LICENSE TEXT ON FILE for the sign-off:** `hy3d-2mv-LICENSE.txt` +
  `hy3d-2mv-NOTICE.txt` beside this record (fetched from the
  tencent/Hunyuan3D-2mv repo, sizes match the repo listing). Headline
  facts visible on its face (operator still VERIFIES the full text):
  "TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT" (release
  2025-01-21); territory exclusion stated up front ("DOES NOT APPLY IN
  THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"); the MAU
  threshold clause is in the grant conditions.

## License hedge (recorded, unprobed)

- **TripoSR (MIT)** is the recorded hedge mesher: wheel-clean sidecar venv
  + `skimage.measure.marching_cubes` (the S-3D-0-blessed technique),
  vertex color via Blender Color Attribute. Lower human-likeness (2/5).
  The E-2 cache key carries `mesher_id/version`, so swapping meshers
  never collides cache rows and never touches the Blender stage.
- **Step1X-3D (Apache)** = recorded candidate, unprobed (from the 0-E
  roundtable spec).

## The other 0-E engines (no review needed)

- **ltx_orbit**: prompt preset over the EXISTING ltx_video adapter -- same
  checkpoint, same license posture as ltx_video (`commercial_clean=False`,
  license is profile data; nothing new to review for 0-E).
- **still_parallax**: **DepthAnythingV2-SMALL** pinned, **Apache-2.0**
  (`commercial_clean=True`). The larger DA-V2 Base/Large checkpoints are
  **CC-BY-NC** and are BANNED from this engine -- never swap them in
  without reopening this record. Pinned HF repo:
  `depth-anything/Depth-Anything-V2-Small-hf`.
  - **SNAPSHOT FETCHED (2026-06-11, ticket A3):** revision
    `5426e4f0f36572d16453bbda7a8389317b1bef99` in the HF cache under
    `C:\ComfyUI-Models\huggingface` (SMALL only; Base/Large NOT fetched,
    per the ban above).
- **Blender (portable, pinned)**: GPL-3.0 application; invoking it as a
  subprocess does not affect OTR code licensing; rendered output frames
  carry no Blender license burden.
  - **PINNED BUILD (recorded 2026-06-11, ticket A1):** Blender **4.5.10
    LTS** (build date 2026-05-19), official portable zip
    `blender-4.5.10-windows-x64.zip` from
    `download.blender.org/release/Blender4.5/`.
  - zip sha256
    `ef6d846b8015f47ade6df3f9322ce17419080a5d922fa562b6c966064fe30dce`
    (verified MATCH against the official `blender-4.5.10.sha256`
    manifest at download time).
  - Installed at `C:\ComfyUI-Models\tools\blender-4.5.10\`;
    `OTR_BLENDER_EXE` (User env) points at its `blender.exe`.
  - GPL boundary note: Blender is a SPAWNED application binary
    (`--background` subprocess); OTR links no Blender code, so GPL-3.0
    imposes nothing on OTR sources; frames rendered with it are ours.

## Sign-off (operator only)

- [ ] I read the license text shipped with the hy3d-2mv weights; the
      threshold + territory clauses are compatible with OTR's use.
- [ ] mesh_stage may be promoted from selectable to a role DEFAULT.
      (Until BOTH boxes are checked, `default_roles` stays empty and
      `commercial_clean` stays False.)
