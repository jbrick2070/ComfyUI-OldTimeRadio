# OTR Phase 1 -- WIRE the video platform into the real episode workflow (BUILD SPEC)

Date 2026-06-08. Branch v2.0-alpha. Operator-APPROVED; build in the order below;
CLEANBREAK LAST; report a checkpoint AFTER wire+render proves green, BEFORE the cleanbreak.
This sheet is the grounded, code-level executable plan (the hard investigation is done).

## 0. Proven this session (grounding -- do not re-derive)
- `workflows/otr_scifi_16gb_full.json` (14 nodes) renders ONLY the SignalLost procgen visual
  (`video_engine.py::SignalLostVideoRenderer`); ledger CLIPS=0; the platform is NOT wired in.
  Evidence: a REAL episode (`signal_lost_dancing_particles_20260608_160818`) rendered end-to-end on
  the live :8199 server; `OTR_MasterAudioMux: audio_byte_identical OK (9b3ec037372a)`,
  `duration_check v=67.200 a=67.204 OK`; ffprobe h264 1920x1080 + aac 48k mono -- but ZERO per-beat
  platform clips (procgen only). So Part A (real episode + byte-identical mux) already holds; the
  gap is exactly the missing per-beat platform render path.
- REUSE: `nodes/_otr_video_engines/render_driver.py::run_episode(ledger, *, fallback_of,
  oom_shot_id=None, oom_engines, assets, frame_count, canvas)` is ALREADY a generic per-beat render
  loop: walks `ledger['video']['shots']`, drives each engine
  (assert_usable->prepare->render_clip->canonicalize->teardown) through the LOUD fallback chain
  (`render_shot`), returns `{ledger, clips, trace, vram_peak_mb}`, never touches frozen audio.
  The soak only feeds it a SYNTHETIC section (`build_soak_fixture`). `render_single`/`run_gpu_soak`
  are the other two entries.
- `nodes/otr_shot_lock.py::OTR_ShotLock.lock(script_json, audio_done, video_policy_json)` ALREADY
  stamps `ledger['video']` gated on `audio_done`: `{video_revision, execution_groups, shots[]}`.
  Shot rows (lines ~498-517): `shot_id="shot_{beat_id}", beat_id, role, engine_id=engine_for(role)
  [from video_policy_json], family="", target_frame_count=budget["per_beat"][beat_id],
  creative={expression,motion,camera,text_prompt}` (request_hash stripped). ShotLock OWNS the
  audio-derived clip budget + the M4 per-beat prompts and SUPERSEDES OTR_VideoPlan. So there is NO
  separate ShotDurationCalculator in the modern path -- the budget is in ShotLock.
- `nodes/otr_video_director.py::OTR_VideoDirector` -> `video_policy_json` (per-role A/B/C engine pick
  from the registry). Feeds ShotLock's `engine_for()`.
- `nodes/otr_image_gen_dispatcher.py::dispatch_images(ledger, image_policy, image_prompts)` mints one
  portrait per character at `output/otr/stills/{portrait_content_hash}.png`, stamps
  `ledger['images'] = {..., 'images':[{char_id?, path, portrait_content_hash, request_hash}...]}`,
  returns `(patched_ledger, image_done, report)`. Upstream image-gen for HuMo `init_image`.
- `nodes/otr_silent_composite.py::normalize_to_silent_canonical(in_path, out_path, w=1472,h=832,
  fps=25)` today normalizes ONE base video (silent/CFR/yuv420p/bt709). MUST be extended to assemble a
  per-beat clip manifest.
- Post nodes EXIST but are ABSENT from the workflow: `OTR_CaptionBurn` (video_path,report),
  `OTR_RTXUpscale` (upscaled_mp4_path,report), `OTR_PostUpscaleProcgenBlend` (final_mp4_path,report),
  `OTR_MasterAudioMux` (final_video_path,report).

## 1. NEW code (small; additive; cold-import clean; UTF-8 no BOM; ASCII)
A. `render_driver`: add optional `request_builder=None` param to `run_episode` (default keeps the
   current global-assets `build_request`). Add `build_request_from_shot(shot, ledger, *, canvas=None)`:
   - `init_image` = portrait path for the shot's char_id from `ledger['images']` (character beats); "" else.
   - `audio_ref`  = the beat's audio-segment path (CONFIRM the per-beat audio field in the audio ledger).
   - `text_prompt`= `shot['creative'].get('text_prompt')` or the build_request default.
   - `timing.target_frame_count` = `shot['target_frame_count']`.
   - `seed` = deterministic from `shot['creative']['request_hash']`/shot_id (V-7 request-hash).
   Add `run_real_episode(ledger, *, fallback_of=None, canvas=None)` = `run_episode` with
   `request_builder=build_request_from_shot`, NO oom forcing, real per-shot assets.
B. Node entry: add `mode="episode"` to `OTR_VideoRenderBatch` (or a new `OTR_EpisodeVideoRender`):
   reads `patched_ledger_json` -> `run_real_episode` -> writes per-beat clips to the episode dir +
   emits a clip-manifest JSON (shot_id -> clip path, beat order) as a STRING out for SilentComposite.
   MUST run in ComfyUI's executor thread (via /prompt). Single resident heavy <=14.5 GB; BUG-291
   detach reclaim between beats (`wrapper_bridge.reclaim_idle_models`, NO `unload_all_models`).
C. `OTR_SilentComposite` assemble: accept the clip manifest + the audio-derived budget -> a
   frame-accurate CFR timeline (frame counts, NOT seconds), gap-fill with the procgen/still floor,
   assert assembled length == frozen master length (pre-mux A/V sync guard). Keep the single-base
   path for the floor.

## 2. Workflow rewire (otr_scifi_16gb_full.json) -- order
1. `audio_done(7)` + `script_json(62)` -> `OTR_VideoDirector` -> `OTR_ShotLock(script_json,audio_done,
   video_policy_json)`.
2. `ShotLock.patched_ledger_json` -> `OTR_MetaBriefImagePrompt` -> `OTR_ImageGenDispatcher`
   (image BEFORE video) -> patched_ledger.
3. -> [new render node, mode=episode] -> clip manifest.
4. -> `OTR_SilentComposite`(assemble) -> `OTR_CaptionBurn` -> `OTR_RTXUpscale`
   -> `OTR_PostUpscaleProcgenBlend` -> `OTR_MasterAudioMux`.
5. GATE 1 (audio spine, do BEFORE cutting any procgen edge): repoint
   `MasterAudioMux.master_audio_path` from `SignalLostVideo(12).video_path` to the REAL master mix
   (EpisodeAssembler `episode_audio` / the master-mix file). NEVER leave the mux sourceless. Keep
   SignalLostVideo as the `ProcgenBlend` floor input.
NOTE: workflow JSON edits must satisfy `scripts/otr_api.py`'s strict converter (serialized slot
layout; seed `control_after_generate` companions; `forceInput`). Reset the 4 OpenRouter/Comfy slot
pickers on node 1 to their `(enable ...)` sentinels before any submit (ComfyUI rejects out-of-list
COMBO). Validate via `otr_api.workflow_to_api_prompt` + `OTR_WorkflowValidator` before submit.

## 3. Gates / invariants (every chunk)
- Audio FROZEN + mux-LAST (`-c:a copy`, no `-shortest`); `test_audio_byte_identical` GREEN +
  `audio_byte_identical OK` at EVERY chunk -- if it drifts, STOP.
- Image readiness (gate 2): confirm `OTR_ImageGenDispatcher` renders real Flux portraits feeding
  HuMo `init_image`; if the image path isn't live, HuMo falls back LOUD (acceptable floor) -- but
  "real HuMo on talking beats" REQUIRES it. Report which.
- Executor thread + single-resident <=14.5 GB + BUG-291 reclaim between beats.
- Determinism: relaunch :8199 with seeds (section 5).
- character_3d stays dark (registry/policy SUPPORTS it, never REQUIRES it).
- Per-change: targeted cold-import/schema/fallback/changed-engine + `test_audio_byte_identical`;
  full Bug Bible + core before each commit; commit per chunk; do NOT push.

## 4. KEYSTONE (the pre-cleanbreak checkpoint gate)
A real episode renders REAL per-beat platform clips (ledger CLIPS>0, HuMo on announcer/character
talking beats + procgen/family on others, NOT all-procgen), final mp4 in `output/otr/obs`; audio
byte-identical (test_audio_byte_identical GREEN + mux check); render-twice deterministic
(`OTR_CAST_SEED`/`OTR_STYLE_SEED`+request-hash); VRAM <=14.5 GB; floor still renders with all heavy
engines OFF. Report to the operator BEFORE the cleanbreak.

## 5. Seeded relaunch (gate 5) -- replicate the live :8199 env + add seeds (cmd shell)
Kill the live :8199 python (PIDs vary; find via cmdline contains main.py + 8199), then launch
(NO --highvram per BUG-291; cuda-malloc; user_aship):
```
set "HF_HOME=C:\ComfyUI-Models\huggingface"
set "OTR_ENABLE_HUMO=1" & set "OTR_HUMO_CKPT=C:\ComfyUI-Models\diffusion_models\humo_1.7B_fp16.safetensors"
set "OTR_HUMO_UNET_NAME=humo_1.7B_fp16.safetensors" & set "OTR_HUMO_CFG=5.0" & set "OTR_HUMO_STEPS=10"
set "OTR_HUMO_LORA_NAME=none" & set "OTR_OUTPUT_DIR=C:\Users\jeffr\Documents\ComfyUI\output"
set "OTR_CAST_SEED=42" & set "OTR_STYLE_SEED=42" & set "PYTHONHASHSEED=0"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8" & set "NVIDIA_TF32_OVERRIDE=0" & set "TOKENIZERS_PARALLELISM=false"
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py --base-directory C:\Users\jeffr\Documents\ComfyUI --user-directory C:\Users\jeffr\Documents\ComfyUI\user_aship --port 8199 --cuda-malloc --extra-model-paths-config "C:\Users\jeffr\AppData\Roaming\Comfy Desktop\shared_model_paths.yaml" --input-directory C:\Users\jeffr\Documents\ComfyUI\input --output-directory C:\Users\jeffr\Documents\ComfyUI\output
```
Poll `http://127.0.0.1:8199/system_stats` until ready. Drive via the `output/otr/_lane1/lane1.py`
pattern (submit/poll/logtail/inspect/graph/probe already written) or `scripts/queue_smoke.py`.

## 6. CLEANBREAK (LAST -- only after the keystone is GREEN)
Cut the procgen-ONLY edges: `SignalLostVideo(12)->SilentComposite.base_video_path` and
`SignalLostVideo(12)->MasterAudioMux.master_audio_path`. Retire `OTR_VideoPlan` (superseded by
ShotLock) + `OTR_BatchLTXRender` (superseded by render_driver); confirm/retire
`_otr_render_plan`/`OTR_RenderPlan`; rewire/retire `OTR_ShotDurationCalculator` off the VideoPlan
envelope (ShotLock owns the budget). Tombstone + unregister; NO runtime gates. procgen/still STAYS a
selectable FLOOR engine + the ProcgenBlend input -- episodes ALWAYS render.

## 7. Confirm at build start (cheap reads, before writing code)
- The shot-row `char_id` field (shot_lock 498-517) + the per-beat audio-segment path source (audio ledger).
- `OTR_VideoDirector` INPUT_TYPES (does it need `OTR_VideoProbe` usable-engines?).
- `OTR_ImageGenDispatcher` image entry char_id<->path mapping for the init_image lookup.
- The SilentComposite assemble input contract (clip-manifest JSON shape) you add.
- mode="episode" on OTR_VideoRenderBatch vs a new node (the batch node writes to otr/aship; the
  episode output must land in the episode dir + output/otr/obs).
