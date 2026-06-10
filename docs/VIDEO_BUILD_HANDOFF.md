
**Last updated:** 2026-06-09  
**Branch:** `v2.0-alpha`  
**HEAD:** `ea024c7` (feat(humo): bake the FAST 14B Kijai+lightx2v 6-step tier in as the default keystone — PUSHED, origin == HEAD)  
**Commits ahead of origin:** 0 (in sync)

## HuMo KEYSTONE LIVE-VERIFIED on the 14B FAST tier (2026-06-09)

Run `ceef5e1b` / episode_id `pending_20260609_170145`. Booted env (echoed):
`OTR_ENABLE_HUMO=1`, `OTR_HUMO_UNET_NAME=Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`,
`OTR_HUMO_LORA_NAME=lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`,
`OTR_HUMO_STEPS=6`, `OTR_HUMO_CFG=1.0`. These are now the BAKED-IN defaults in
`eng_humo.py` (commit `ea024c7`) — only the enable flag is needed; the UNET resolves
via `folder_paths`/extra_model_paths. Live proof: `Requested to load WAN21_HuMo` +
WhisperLargeV3/WanTEModel/WanVAE, KSampler bar `0/6` (6 steps), ~2 min/beat,
VRAM peak **13.8 GB ≤ 14.5**. `engine_histogram={"humo":3,"still_kenburns":2}`;
final mp4 PCM `093e3d7eb129` == master (byte-identical). NB clip files at
`C:\ComfyUI-Models\diffusion_models\Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`
+ `C:\ComfyUI-Models\loras\lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`.

DONE follow-up: the **hard auto-downgrade tier** is wired (commit `dc2d539`). The
chain is now **humo -> humo_1.7B -> latentsync -> still_kenburns** (registry-verified
deterministically); on an OOM/VRAM (or any HARD) failure the 14B keystone degrades
FIRST to the 1.7B HuMo tier (a real talking face) before latentsync/still, with a LOUD
restamp (the swap log now also carries `detail=`). `humo_1.7B` is a registered engine
(`HuMo17BEngine`, `requires_flag=OTR_ENABLE_HUMO`) with env-isolated config
(`OTR_HUMO_17B_*`: own UNET `humo_1.7B_fp16.safetensors`, NO LoRA, 20 steps / cfg 5.0)
so the 14B `OTR_HUMO_STEPS/CFG` never bleed in. Soak trail is now 4 hops; suite 3777.

OPEN follow-up: 2 talking beats (b001/b005) still fell back with
`GraphExecutionError: humo ... audio=''` — a per-beat audio-slice timing edge case
(intro/outro beats get no sliceable clip), NOT a config issue. The 1.7B tier won't
fix audio='' (same missing input), so those still degrade to the floor; the real fix
is the per-beat slice timing for intro/outro beats.

---

## ACTIVE MISSION

Build the OTR Video Engine per the execution plan:

- Repo: `docs/OTR_VIDEO_ENGINE__EXECUTION-PLAN_v1.4.md`
- Canonical: `C:\Users\jeffr\Documents\otr-video-roundtable\OTR_VIDEO_ENGINE__EXECUTION-PLAN_v1.4.md`

**Current step: FULL SMOKE PASSES end-to-end (2026-06-09, commit 317a295). The procgen-video full pipeline renders writer → indextts2/kokoro/SA3 audio → Flux portraits (c01/c02/c03) → HuMo→latentsync→still_kenburns LOUD fallbacks → SilentComposite → MasterAudioMux → `_final.mp4`. Gate verified: `v=41.72s a=49.24s` audio-longer-than-video PASSES. Audio byte-identical (PCM SHA `b90837aae107`, final mp4 audio codec `pcm_s16le`). Suite 3777 passed. NEXT: Subproject C — note Flux portrait gen_fn already runs live in the smoke (832x1216, seed=42, steps=20, cfg=1.0); confirm it meets the Subproject C spec, otherwise proceed to HuMo keystone verify (HuMo needs its weights on disk — currently dependency_missing → still_kenburns fallback).**

---

## HARD RULES

- Do NOT start / resume / "continue" any other sprint — NOT story-spine, NOT story-pipeline, NOT any audio sprint, NOT any other ROADMAP item. They are PARKED.
- Audio is SHIPPED; the audio script ledger is FROZEN (read-only). Never reopen or modify it.
- Ignore any stale `session_handoff.md` and any memory / ROADMAP entry implying other "active" work. The video engine is the ONLY active build until the operator says otherwise.
- Invariants in force at all times:
  - Byte-identical master audio + mux-LAST
  - Single resident heavy engine, VRAM peak ≤ 14.5 GB (3D engines: 14.0 GB)
  - Cloud / OpenRouter allowed (Jeffrey lifted 100%-local rule 2026-06-04; `feedback_cloud_lanes_ok.md`)
  - Determinism via seed-keyed cache
  - Every in-render fallback LOUD (log + ledger restamp; never silent)
  - V-6: all engines unconditionally imported in `__init__.py`; usability gated in `assert_usable` only
  - V-12: cold-import clean (no torch/diffusers/comfy at module scope)
  - UTF-8 no BOM; SFW; no "dummy" → use "placeholder"
- Commit per green chunk. Do NOT push unprompted.
- UPDATE otr-build-tracker artifact every session — preserve gauge + lanes styling.
- PRIME DIRECTIVE: never hand the operator a script/cmd/PowerShell block to run. Use Desktop Commander first, then Windows MCP. YOU run everything.

---

## HEADLESS SERVER ENV (MACHINE SETUP — NOT in git; reproduce on a fresh box)

The headless ComfyUI on :8000 runs from the `ComfyUI-Installs` tree but needs the
canonical model store + sidecars wired in. These are machine-local and CANNOT be
pushed — recreate them before any live smoke or they fail mid-render:

1. **Python:** launch with the deps venv, NOT the bare uv python (the uv
   interpreter lacks `sqlalchemy`/torch). Exact launch (YOU run it via Desktop
   Commander, persistent process):
   ```
   set HF_HOME=C:\ComfyUI-Models\huggingface
   C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
     C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py ^
     --port 8000 --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI
   ```
2. **`HF_HOME=C:\ComfyUI-Models\huggingface`** — without it the server resolves
   HF to the install dir's INCOMPLETE cache and hangs re-downloading mistral-nemo
   (the real 5×4.57GB shards live under `C:\ComfyUI-Models\huggingface`).
3. **`extra_model_paths.yaml`** at `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\`
   pointing `base_path: C:/ComfyUI-Models/` (checkpoints, clip, unet, vae,
   diffusion_models, loras, text_encoders, …). Fixes `folder_paths`-based lookups
   (SA3 music ckpt, Flux, etc.). Confirmed by the startup "Adding extra search
   path …" lines.
4. **Two junctions into the install** (OTR resolves these by `models_dir` join, so
   `extra_model_paths` does NOT cover them):
   - `...\ComfyUI\custom_nodes\...\index-tts`  →  `C:\Users\jeffr\Documents\ComfyUI\index-tts` (indextts2 sidecar venv + checkpoints)
   - `...\ComfyUI\models\TTS`  →  `C:\ComfyUI-Models\TTS` (kokoro voices incl. `bm_fable.pt`, indextts2 refs)
   ```
   mklink /J "...\ComfyUI\index-tts"  "C:\Users\jeffr\Documents\ComfyUI\index-tts"
   mklink /J "...\ComfyUI\models\TTS" "C:\ComfyUI-Models\TTS"
   ```
   (the OTR custom_node itself is already a junction → the Documents repo.)

Non-fatal startup noise (ignore): prestartup emoji `charmap` error, sqlite
"unable to open database file" → "Using RAM pressure cache".

---

## WHERE WE ARE

### This session: MasterAudioMux gate fix (3 commits, NOT pushed)

Root cause found and fixed: the MasterAudioMux duration gate was using
`abs(v_dur - a_dur) > tol`, which fired because the master WAV (45.75 s) includes
opening/closing themes (10 s + 8 s) that are NOT in the drama-only SilentComposite
(38.28 s). The gate has been changed to `v_dur > a_dur + tol` — audio longer than
video is intentional and safe. Three commits landed:

| Commit | File | Fix |
|--------|------|-----|
| `5dbe334` | `nodes/otr_silent_composite.py` | Prefer `manifest["total_target_frames"]` over `_probe_duration()` for frame budget |
| `76633eb` | `scripts/queue_smoke.py` | Resolve OpenRouter/Comfy slot pickers via live `/object_info` first-choice instead of stale sentinel string |
| `2ac76a6` | `nodes/otr_master_audio_mux.py` | Gate direction fix: `v_dur > a_dur + tol` (was `abs(v-a) > tol`); added comment explaining intentional gap |

Suite: 3777 pass / 0 fail after all 3 fixes.

**Infrastructure:** junction created at
`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
→ repo so headless server loads OTR nodes. Headless server was confirmed running
on :8000 (venv python, PID 15840-ish) with OTR import log visible.

### Pending: smoke re-run on :8000

The fix is loaded in the running server. The smoke has NOT been re-queued since
the gate fix. This is the immediate next step:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
  C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\queue_smoke.py
```

Expected: `_final.mp4` appears in node 85 outputs; `v_dur=38.28s < a_dur=45.75s`
gate passes; PCM SHA256 of `_final.mp4` audio == PCM SHA256 of master WAV.

Polling script: `outputs/poll_8000.py` (prompt_id must be updated after re-queue).

### After smoke passes

1. **Push 3 commits** (`5dbe334`, `76633eb`, `2ac76a6`) — Desktop Commander git push.
2. **Subproject C — Flux portrait gen_fn:** ComfyUI-load Flux.1-dev UNET+CLIP+VAE
   → CLIPTextEncode MetaBrief prompt → KSampler → VAEDecode → uint8 →
   `dispatch_images`, under residency lease + single-resident-heavy budget.
3. **HuMo keystone verify:** CLIPS > 0, HuMo gets init_image, audio byte-identical,
   VRAM ≤ 14.5 GB.

### Prior milestones (this branch)

| Commit | What |
|--------|------|
| `24e171b` | Phase 3: character_3d dark scaffold (18 tests, schemas len=8) |
| `19afaea` | Phase 1: LTX GPU-verify (CLIPLoader+T5-XXL split, OTR_TEST_MODE VRAM guard) |
| `1c88c69` | Chunk E cleanbreak: EpisodeAssembler WAV save + link surgery |
| `f2e603e` | M1 first watchable episode (tag: m1-first-episode) |
| `f003978` | B-SHIP (tag: B-ship, pushed; origin == local) |

---

## OPEN OPERATOR / GPU GATES

| Gate | Status | Unblocks |
|------|--------|----------|
| **Smoke re-run on :8000** | **READY — server live with gate fix loaded** | push 3 commits + Subproject C |
| Wan-i2v ckpt on disk (`OTR_WAN_I2V_CKPT`) | BLOCKED: no ckpt | Phase 2 Wan live-verify |
| cu128 toolkit + latentsync sidecar venv | BLOCKED: no cu128 toolkit | Phase 4 latentsync live-verify |
| ~25 real meshes + ARKit-52 .npz + cu128 | BLOCKED: no assets | Phase 5 character_3d LIVE keystone |

---

## FIRST ACTIONS FOR NEXT SESSION

1. Verify headless server is still running on :8000 (`curl http://127.0.0.1:8000/system_stats` via DC).
   If not: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m comfy ...` — see prior session for exact launch command.
2. Queue smoke: run `scripts/queue_smoke.py` via DC.
3. Poll `/history/<prompt_id>` until `completed=true`; confirm `_final.mp4` in node 85 outputs + PCM hash match.
4. If smoke passes: push 3 commits, then proceed to Subproject C Flux portrait gen_fn.
5. Wait for operator go before writing any new production code.

---

## PARKED — NOT NOW

- Story-spine sprint
- Story-pipeline v4
- Any audio sprint (audio is SHIPPED)
- Any other ROADMAP item not in the GO-FORWARD video lanes above
