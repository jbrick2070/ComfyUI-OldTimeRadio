# CODER-WINDOW KICKOFF -- Wan 2.2 S2V double-duty engine (smoke first)

Paste this as message #1 of a fresh CODER window. Goal: prove **Wan 2.2 S2V**
(audio-driven image->video) in the FAST smoke harness -- one talking-head clip and
one b-roll clip for Jeffrey's eyeball -- BEFORE any episode-engine wiring. This
window produces the proof; the episode integration is a later step.

## Why (decided 2026-06-12, roundtable + operator)
- LTX-2.3 motion lives in the 22B, which does NOT fit OTR's 14.5GB resident ceiling
  (panel-unanimous: gpt-5.5/gemini-3.1/grok-4.3). v0.9 fits but its motion warps
  (operator eyeball). Dead end both ways.
- **Wan 2.2 S2V-14B does DOUBLE DUTY**: lip-synced talking heads + cinematic camera
  motion in ONE audio-driven pass. It CONSUMES audio (does not generate it) -> it
  drives lips from the FROZEN master mix; the byte-identical mux stays untouched
  (audio spine invariant preserved). One S2V engine can replace LTX-broll +
  HuMo-talkinghead -> fewer resident heavies, simpler VRAM staging.
- GGUF quants map onto OTR's existing 8gb/16gb profile tiers: Q5_K_M (16GB) /
  Q4_K_M (8GB).

## Grounded facts (verified on this box 2026-06-12)
- GGUF loader **`UnetLoaderGGUF` is INSTALLED** (ComfyUI-GGUF). Good -- GGUF works.
- `C:\ComfyUI-Models\vae\wan_2.1_vae.safetensors` is on disk (S2V is Wan-family; it
  likely reuses the Wan VAE -- VERIFY the exact VAE S2V wants).
- OTR's talking-head engine is ALREADY Wan-family (`Wan2_1-HuMo-14B_fp8`,
  `Wan2_1-HuMo-17B_Q5_K_M.gguf` on disk) -> S2V is a consolidation, not a new stack.
- **NOT on disk:** any Wan 2.2 S2V model -> one fetch needed.
- **Wan S2V sampler nodes are NOT installed** (`WanVideoSampler` absent) -> the S2V
  graph needs either ComfyUI-native Wan2.2 S2V nodes (check the ComfyUI 0.24.1
  build) or the ComfyUI-WanVideoWrapper (KJNodes/Kijai) pack. RESOLVE THIS FIRST.

## Tasks (in order)
1. **Node support:** determine the correct S2V node path -- native Wan2.2 S2V in
   ComfyUI 0.24.1, OR install ComfyUI-WanVideoWrapper. Confirm the audio-driven S2V
   nodes load (object_info). Record exact node class names + input names.
2. **Model fetch (operator lifted strict-local):** pull **Wan 2.2 S2V-14B GGUF
   Q5_K_M** (16GB tier) and **Q4_K_M** (8GB tier) into the right models dir. Record
   the HF repo + sha256 + license; fail-closed if absent (no runtime download).
   VERIFY the operator's claimed sizes (~10-11GB Q5, ~8-9GB Q4) against the actual
   files -- treat the pasted numbers as a hypothesis until the files are on disk.
   Also confirm the S2V text encoder + VAE + audio encoder it needs are present.
3. **Smoke harness:** clone the pattern in `scripts/otr_ltx_motion_smoke.py`
   (one still in -> short clip out via /prompt, ~fast, SaveWEBM, MAD via
   `scripts/otr_ltx_mad.py`). Build `scripts/otr_wan_s2v_smoke.py`: a still + a
   short slice of the FROZEN master WAV -> S2V -> webm. Two runs:
   (a) talking-head still + a voiced audio slice (expect lip-sync + subtle motion),
   (b) a radio-console still + a music slice (expect camera/object motion = the
   b-roll job). Save both clips tagged; print render time + peak NVML + MAD.
4. **VRAM gate:** record peak host NVML; the Q5 16GB tier must stay <=14.5GB
   resident (encode->free-encoder->load-transformer->tiled VAE decode sequencing,
   per the roundtable). If Q5 busts it, fall back to Q4 or the split (Wan2.2
   TI2V-5B motion + LatentSync overlay -- OTR already has a latentsync engine).
5. **Eyeball gate:** present both webms to Jeffrey. The bar is VISUAL (real motion
   + lip-sync + the still preserved + no warp), NOT MAD alone (MAD oversold the LTX
   warp this session). Do NOT lock anything until Jeffrey confirms.
6. **Only after eyeball PASS:** scope the episode integration as a SEPARATE step --
   a new `eng_wan_s2v` MotionEngine that consumes the per-beat sliced master audio
   (the render_driver already slices it for HuMo), maps to the 8gb/16gb profile
   tiers, and (optionally) consolidates the HuMo-talkinghead + LTX-broll roles.
   Optional: layer an IC-LoRA Cameraman (dolly/pan/orbit) on b-roll beats for
   explicit camera control IF it exists for S2V.

## Hard rules (unchanged)
- Single resident heavy <=14.5GB (host NVML). 100% local after the one fetch.
  Frozen audio spine: S2V CONSUMES the master, never regenerates it; mux stays
  byte-identical (`test_audio_byte_identical` green). Determinism (seed-keyed).
  UTF-8 no BOM, SFW. Commit per green chunk, do NOT push unprompted (operator gate).
- Run full tests/ + Bug Bible after any code change. Use the canonical headless
  launcher (`scripts/_otr_soak_server_launch.cmd`, UTF-8 + the model-paths yaml)
  and the auto render-launcher + watchdog (`scripts/otr_run_leg.ps1`) patterns.

## Sources the operator provided (verify at build, do not trust blind)
- Wan 2.2 S2V audio-driven workflow: https://comfyui-wiki.com/en/tutorial/advanced/video/wan2.2/wan2-2-s2v
- S2V lip-sync: https://sbcode.net/genai/wan2.2-S2V/
- Pose-control lip-sync S2V: https://www.runcomfy.com/comfyui-workflows/pose-control-lipsync-with-wan2-2-s2v-in-comfyui-audio2video
- S2V GGUF low-VRAM: https://learn.thinkdiffusion.com/latest-in-lipsync-infinitetalk-video2video-comfyui-guide/
