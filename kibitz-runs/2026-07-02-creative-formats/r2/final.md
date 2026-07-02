# r2 synthesis (anchor + codex + claude)

All major claims CONFIRMED by line citation (schemas.py:78/139 extras
forbidden; director:291/307; render_driver:1310/1696 + retry taxonomy
:33; capability test tests/test_capability_profiles.py:211;
eng_mesh_stage key :106 + Blender selftest :338; portrait_ledger:84
pixel-byte hashing; KLING concurrency default 1). No material
misreads. Codex "S3 adapter not present" = true-by-design (sequencing
already gates on S3) -- folded as an explicit dependency statement.

ADOPTED INTO THE PLAN:
1. Registration completeness checklist: family=`format_composite`
   (new value), required_inputs, @register, __init__ import,
   CAPABILITIES row (cpu_ok True, 0 VRAM -- the engine is local;
   its cloud spend is governed per-invoke by the S0 budget machinery,
   documented), ENGINE_FAMILY entry, capability-set test extension.
2. FORMAT CONTEXT plumbing: a planning/stamping phase (ShotLock /
   ImageGenDispatcher) writes format assets + BOARD MANIFEST
   (`otr/episodes/<ep>/evidence_board/board_manifest.json`:
   {cast:[{char_id, portrait_hash, x,y,w,h}], layers, z_order,
   layout_seed}) and a versioned optional `format_ctx` block is ADDED
   TO THE VideoRequest SCHEMA (episode_dir, manifest paths, lines[]
   {speaker,start_s,end_s,audio_path}) -- extras stay forbidden;
   format_ctx is schema'd, not smuggled.
3. visual_format semantics (concrete): !=standard sets all three role
   slots' DEFAULTS to the format row; a per-role pick differing from
   the slot's current default = explicit and preserved. Precedence:
   explicit pick > visual_format (widget/env) > profile
   DEFAULT-OVERRIDE > registry default. Append-only widget +
   widget-vector tests.
4. Cloud calls ONLY via the S0 invoke bridge from inside render_clip
   (may block for poll duration; errors classified into the
   render_driver retry taxonomy). render_clip RETURN = single mp4 path
   (composited); per-line lipsync happens inside render_clip.
5. Asset placement plumbing: episode asset root rides format_ctx;
   engines write STRAIGHT to canonical episode paths (no tmp staging
   -- repo sec-6 rule). 
6. Cache keys: mesh = eng_mesh_stage pattern (subject id + portrait
   SHA + row id + adapter/export version + tin_toy profile version);
   Blender plate = (mesh_hash, camera_preset_or_path_hash,
   frame_count) -- duration derived; sepia/polaroid = LOCAL PIL post
   on the RAW portrait (portrait_hash preserved; face-sim comparable)
   -- NEVER a re-minted sepia still.
7. Kling latency reality: concurrency default 1 -> per-line jobs
   serialize. MVP mitigations: lipsync only CLOSE-UP lines (board
   wides/pans are format-local mute shots, stamped); per-episode Kling
   call count printed in the estimate report; concurrency raise is a
   ToS/pricing question recorded in verify probes.
8. Face-similarity check: reuse portrait_ledger machinery; threshold +
   failure action = LOUD ledger stamp + that line stays still (no
   paste), never a silent bad paste.
9. Blender: version gate in assert_usable (--version >= 4.5) or drop
   the exact-version claim; timeout/corrupt_output classification per
   S0 taxonomy; per-plate render manifest (mesh hash, camera, frames,
   fps, lighting, path).
10. tin_toy_v1 mints front + 3/4 as SEPARATE gens (multi-image mesher
    input); single-sheet multiview generation is not assumed.
CUTS: photoreal-CG probe out of V1 (scoped to tin face + readability
+ still-frame-video acceptance -- the Prop Shot mouth gate owns the
photoreal question); whole-episode tin-toy language removed;
"re-run 100% CACHED" demoted to smoke assertion; episode dressing
layer NOT cached for MVP (regenerate; negligible vs Kling);
symlink alternative noted, operator copy directive stands.
