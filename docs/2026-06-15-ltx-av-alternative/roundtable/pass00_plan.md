# LTX AUDIO-INPUT (A2V) ALTERNATIVE PATH -- Plan to Harden (code-ready)

## Goal
A unified **audio-input-assisted-by-prompt** LTX path for **ALL** beats (announcer, music, character):
the FROZEN per-beat audio + the text prompt + the FLUX still drive a synchronized video. This is an
ALTERNATIVE to today's split (prompt-only `ltx_video` for announcer/music + HuMo/LatentSync for character
lip-sync). Harden it to code-ready across FOUR levels: **(A) architecture, (B) coding/wiring, (C) bugs/risks,
(D) final polish.** Panel: 3 frontier models + Claude as a code-grounded panelist AND judge.

## Two lanes (operator's fact-checked framing -- honest version)
- **LANE A -- SHIP NOW (production, already mostly shipped today).** Current `ltx_video` I2V @ 832x480 +
  audio-reactive ledger/prompt injection + the boomerang loop + composite/upscale. This is the SAFE fix for
  "LTX doesn't move enough": keeps the proven low-VRAM 2B path, adds motion WITHOUT loading the heavier
  LTX-2.3 audio-video model. (Today shipped: ksampler default + music_open + boomerang + 832x480 -- the
  audio-reactive PROMPT injection from the ledger is the remaining Lane-A polish.)
- **LANE B -- EXPERIMENTAL BRANCH ONLY (gated, dark, must be PROVEN on this box).** LTX-2.3 is a real
  audio-video foundation model (synchronized A+V, multimodal inputs text/image/video/audio/depth; checkpoints
  `ltx-2.3-22b-dev` full, distilled 8-step/CFG=1, distilled v1.1 + LoRAs, spatial/temporal upscalers;
  two-stage `A2VidPipelineTwoStage` = Stage1 base + Stage2 upscale freezing audio). ComfyUI has native 2.3
  templates. **VRAM reality (corrected):** the official comfortable target is ~32 GB+; a 16 GB 5080 MAY work
  ONLY with FP8 or community GGUF (Unsloth -- NOT a first-party stable path), small Stage-1 res, CPU/offload
  of the Gemma-3-12B text encoder, batch 1, short clips. **Do NOT call the low-VRAM path "stable" -- it must
  survive repeated short-clip tests on THIS 5080 without OOM/offload-thrash before it is anything but
  experimental.**

## Prior art to BUILD ON (not duplicate)
A CONVERGED build-ready plan already exists: `docs/2026-06-10-ltx-av-lane/LTX_AV_SPRINT_PLAN.md` (8-pass
roundtable, grounded vs the now-STALE HEAD `56caa5b`). Its locked design:
- ONE new file `nodes/_otr_video_engines/eng_ltx_av.py`, private shared core + TWO thin MotionEngineBase
  adapters: **`ltx_av_talk`** (roles announcer_visual + character_video; family `audio_driven_face`;
  required text_prompt+audio_ref+init_image; fallback humo->humo_1.7B->latentsync->still_kenburns) and
  **`ltx_av_music`** (role music_visual; family `audio_conditioned_video` NEW; required text_prompt+audio_ref;
  fallback ltx_video->still_kenburns).
- Dark by default (`default_roles ()`), ONE flag `OTR_ENABLE_LTX_AV`, `@register` unconditional (dropdown-
  visible, fails closed at render). ISOLATION_IN_PROCESS (with a freeze-identical STOP rule ->
  ISOLATION_SIDECAR_REQUIRED if the pip env changes). BUG-070 Sage gate, AS-3 single-resident lease, BUG-291
  reclaim_idle_models, V-12 lazy heavy imports, heavy forward in the EXECUTOR THREAD. Envs OTR_LTX_AV_CKPT /
  _TEXT_ENCODER / _VAE.
- **V-1 ABSOLUTE: the new lane DISCARDS LTX's audio side entirely; only OTR_MasterAudioMux emits audio;
  `test_audio_byte_identical` stays GREEN at every milestone.** (LTX-2.3 GENERATES audio; we must throw its
  audio away and keep the frozen master -- audio is INPUT-conditioning only.)
**This plan is STALE in 3 ways the roundtable must re-ground:** (1) HEAD moved 56caa5b -> 9633e1e (cleanbreak
deleted batch_ltx_render.py; eng_ltx_video.py is the new prompt-only engine; the boomerang + ksampler +
music_open shipped today). (2) The fallback `audio_driven_face` family + humo chain may have changed. (3) It
predates the LTX-2.3 two-stage + VRAM facts in the operator's starter.

## (A) ARCHITECTURE questions
1. Is the 2-adapter design (talk + music) still right, or does LTX-2.3 A2V unify into ONE `ltx_av` engine
   serving all three roles (announcer/music/character) with role-keyed prompt + the same audio-conditioning?
2. The FROZEN-AUDIO contract: LTX-2.3 emits A+V; how do we guarantee we discard its audio and keep the
   byte-identical master (V-1) -- decode video-only stream, or generate-then-drop? Where in render_clip?
3. Two-stage (Stage1 base low-res + Stage2 upscale freeze-audio): does Stage2 belong in the engine, or reuse
   OTR's existing composite/upscale? VRAM lease across two stages on 16 GB.
4. Lane B graduation gate: what objective bar (optical-flow/framediff vs the 5/30 gold b005 + no OOM over N
   short clips) promotes Lane B from dark to selectable?

## (B) CODING / WIRING questions
1. Engine registry: `@register` + CAPABILITIES row (vram_class heavy, vram_estimate); how it slots beside the
   current `ltx_video`/`humo`/`latentsync` without breaking the role_compat / dropdown / profile applier.
2. The prod JSON `otr_scifi_16gb_full.json`: today `OTR_VideoDirector` routes announcer/music->ltx_video,
   character->humo, and `OTR_VideoRenderBatch` carries master_audio_path but `ltx_video` IGNORES audio_ref
   by family. Lane B must CONSUME audio_ref. What JSON wiring changes (if any) -- new dropdown options only,
   no new widgets (V-11)? Per CLAUDE.md any node/widget change goes IN the JSON in the same commit.
3. The audio slice: per-beat audio of the frozen master fed as the A2V conditioning input -- reuse the
   existing per-beat audio slice (Subproject-C) plumbing; staging path; format/sample-rate LTX-2.3 expects.
4. Determinism + the boomerang: does the boomerang (half-render+mirror) even apply to an audio-conditioned
   clip (the audio defines the length -> mirroring would desync audio)? Likely boomerang OFF for the AV lane.

## (C) BUGS / RISKS to pre-empt
1. **VRAM on 16 GB (the dominant risk).** 22B busts 14.5 GB; FP8 ~? ; Gemma-3-12B encoder is huge -> offload.
   Two-stage doubles residency pressure. The single-resident lease + reclaim_idle_models must hold; NVML
   REQUIRED (fail-closed) for the heaviest lane. Quantify FP8 vs GGUF-Q4 footprints; prove batch1 short-clip.
2. **GGUF is community/Unsloth, not first-party** -- loader (UnetLoaderGGUF) availability, node-gate it,
   fail-closed if absent; do NOT claim "stable".
3. **Audio-sync drift / the frozen-audio V-1** -- the clip length must match the beat audio exactly; the
   mux stays OTR_MasterAudioMux; LTX's generated audio MUST be dropped; byte-identical test green.
4. **Offload thrash / OOM stalls** -- repeated short-clip soak; the "render finished leaves resident" trap;
   reset-before-run; the ~60s/heartbeat watchdog.
5. **Determinism** seed-keyed; **isolation** (cu130 freeze STOP rule -> sidecar if deps change); SFW; UTF-8.

## (D) FINAL POLISH
Spatial upscaler only AFTER the base pass is stable; the eyeball/optical-flow gate vs the 5/30 gold b005;
README "what each video model gives" (LTX prompt-only vs LTX-AV audio-driven vs HuMo); operator look-QA;
the Lane-A audio-reactive prompt injection (ledger tempo/energy -> motion verbs) as the cheap win that may
make Lane B unnecessary for music/announcer.

## (E) MODEL VARIANT SELECTION (operator delegated: "you make best choice")
Operator: maybe a 12/8 GB-friendly low-res LTX-2.3 (or distilled v1.1) fits this box -- pick the best.
Candidates to rank for the 16 GB 5080 (Lane-B first test; NONE assumed stable until VRAM-probed on THIS box,
since the official comfortable target is ~32 GB):
- **Distilled v1.1 / distilled 8-step CFG=1** -- smallest first-party option, fastest, the most likely 16 GB
  fit. PREFER first.
- **FP8 22B (official)** + gemma-3-12B fp4-mixed encoder (offloaded) -- official but heavier; try if distilled
  underperforms on motion.
- **Community GGUF Q4/Q5 (Unsloth, NOT first-party)** -- the lowest-VRAM fallback (WaveSpeed anecdote: Q4 for
  12-16 GB, offload Gemma on 12 GB, 16 GB smoother); node-gate UnetLoaderGGUF, fail-closed if absent, do NOT
  call it "stable".
- **Low Stage-1 resolution** (256x144 / 384x216 / 512x288) + batch 1 + short 4-6 s clips + no Stage 2 first.
Claude (judge) recommends the concrete first-test variant grounded against the HF model-card sizes + the
panel; the build-time VRAM probe on the 5080 is the final arbiter (prove-it-or-park).

## INVARIANTS (reject any path that breaks one)
test_audio_byte_identical GREEN (V-1); single resident heavy <= 14.5 GB host NVML (or proven on the 5080's
16 GB with margin); 100% local/offline; determinism; LOUD fallbacks; UTF-8 no BOM; SFW; the shipped
ltx_video / humo / latentsync engines stay EXACTLY as-is (additive only); no new static workflow widgets
(V-11); per CLAUDE.md any JSON node/wiring change lands in otr_scifi_16gb_full.json in the same commit.

## QUESTIONS FOR THE PANEL (all four levels)
Rank the dominant risks; decide 1-engine-vs-2-adapter; nail the frozen-audio discard mechanism; the 16 GB
VRAM strategy (FP8-first vs GGUF, offload, two-stage lease); the JSON wiring (dropdown-only?); whether the
boomerang is incompatible with audio-conditioning; the Lane-B graduation bar; and the smallest Lane-A-only
win (audio-reactive prompt injection) that might defer Lane B entirely. Flag anything in the stale 6/10 plan
that no longer matches the current code.
