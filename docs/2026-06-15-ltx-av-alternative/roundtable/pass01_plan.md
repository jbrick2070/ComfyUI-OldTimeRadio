# LTX AUDIO-INPUT (A2V) ALTERNATIVE PATH -- CODE-READY (pass01)

> **Panel status:** the live 3-model pass was BLOCKED -- OpenRouter balance depleted (HTTP 402, "can afford
> 740 tokens"). Top up at openrouter.ai/settings/credits to re-run GPT+Gemini+DeepSeek for an independent
> re-harden. This pass01 is Claude's **panelist (code-grounded) + judge** synthesis, built on the ALREADY-
> CONVERGED 8-pass plan (`docs/2026-06-10-ltx-av-lane/LTX_AV_SPRINT_PLAN.md`) + the operator's LTX-2.3
> research, re-grounded against current HEAD `9633e1e`.

## DECISION SUMMARY (judge)
1. **Lane A is production NOW; Lane B is a DARK experimental branch (prove-it-or-park).** Do NOT make Lane B
   prod until it survives repeated short-clip tests on THIS 5080 16 GB without OOM/offload-thrash. Official
   LTX-2.3 comfortable target is ~32 GB.
2. **Keep the prior 2-adapter design** (`ltx_av_talk` = announcer+character / `ltx_av_music` = music) in ONE
   new file `eng_ltx_av.py` -- it maps to DIFFERENT fallbacks (humo-chain vs ltx_video) and families, which a
   single unified engine would blur. (LTX-2.3 A2V *could* serve all roles from one weights load; share the
   model via a private core, expose two thin adapters.)
3. **Frozen-audio V-1 is the hard gate.** LTX-2.3 GENERATES audio; we DECODE THE VIDEO STREAM ONLY and drop
   its audio entirely. OTR_MasterAudioMux remains the sole audio emitter; `test_audio_byte_identical` stays
   GREEN at every milestone. Audio is INPUT-conditioning only.
4. **Model pick (operator delegated):** start Lane B on the **distilled v1.1 / 8-step CFG=1** checkpoint --
   smallest first-party, fastest, the most likely 16 GB fit -- at **low Stage-1 res (384x216 or 512x288),
   batch 1, 4-6 s clips, NO Stage 2, Gemma-3-12B encoder OFFLOADED to CPU**. FP8-22B is the fallback if
   distilled motion is too weak; community GGUF-Q4 (Unsloth, node-gated `UnetLoaderGGUF`, fail-closed, NOT
   "stable") is the last-resort low-VRAM lever. The **build-time VRAM probe on the 5080 is the final arbiter.**
5. **Boomerang OFF for the AV lane.** The audio defines the clip length; the half-render+mirror would desync
   the audio. The boomerang stays a `ltx_video` (prompt-only) feature only.

## CLAUDE'S PANELIST CRITIQUE (code-grounded vs HEAD 9633e1e -- what the stale 6/10 plan gets wrong now)
- **The plan was grounded vs `56caa5b` (pre-cleanbreak).** `batch_ltx_render.py` is DELETED; the prompt-only
  engine is now `eng_ltx_video.py`. Any "reuse the LTX render core" assumption must re-point at the new file
  (or stay fully independent in `eng_ltx_av.py`).
- **Today's shipped state changes the Lane-A calculus.** ksampler-default + music_open + boomerang + 832x480
  already gave ~9x motion (0.84->7.85, near the 5/30 gold). So **Lane A may already be "good enough" for
  announcer/music** -- Lane B's unique value is CHARACTER lip-sync (where HuMo is the current path). Frame
  Lane B's real win as: does LTX-2.3 A2V beat HuMo on character lip-sync at acceptable 16 GB cost? If not,
  Lane B is research-only.
- **The fallback families must be re-verified.** The prior plan's `audio_driven_face` (talk) + the
  `humo->humo_1.7B->latentsync->still_kenburns` chain need re-grounding against the current registry +
  `_otr_shared/fallback.py` resolver (CW-7 added that). The NEW `audio_conditioned_video` family must be
  registered without breaking role_compat / the dropdown / the profile applier.
- **The frozen-audio discard mechanism is under-specified for LTX-2.3 specifically.** LTX-2.3's two-stage
  pipeline emits an AV latent; the decode must take the VIDEO branch only (LTXVSeparateAVLatent or the
  video-only VAE decode) BEFORE any save -- never write LTX's audio to disk. Add a test that the engine's
  raw return has NO audio path and the clip is `has_audio False` (mirrors the M7 ffprobe contract for Wan).
- **VRAM: the 22B is a non-starter at 16 GB; even FP8 + Gemma-12B is tight.** The single-resident lease +
  BUG-291 reclaim must hold across BOTH stages; NVML REQUIRED / fail-closed for this heaviest lane (the
  prior plan got this right). Two-stage on 16 GB likely means Stage-2 upscale is OUT for v1 (use OTR's
  existing composite/upscale instead).

## ARCHITECTURE (A)
ONE file `nodes/_otr_video_engines/eng_ltx_av.py`: private `_LtxAvCore` (lazy heavy imports, the two-stage
graph, the video-only decode) + two `MotionEngineBase` adapters. Dark (`default_roles ()`), one flag
`OTR_ENABLE_LTX_AV`, `@register` unconditional (dropdown-visible, fails closed). ISOLATION_IN_PROCESS with the
cu130 freeze-identical STOP rule (-> ISOLATION_SIDECAR_REQUIRED if deps shift). Heavy forward in the EXECUTOR
THREAD. Envs `OTR_LTX_AV_CKPT/_TEXT_ENCODER/_VAE`. Additive only -- ltx_video/humo/latentsync UNTOUCHED.

## WIRING (B)
- Registry `@register` + a CAPABILITIES row (vram_class heavy, vram_estimate from the probe). role_compat:
  talk serves announcer_visual+character_video, music serves music_visual.
- Prod JSON `otr_scifi_16gb_full.json`: **dropdown OPTIONS only, NO new widgets (V-11).** The lane appears in
  the `OTR_VideoDirector` per-role dropdowns; selecting it routes that role to ltx_av. Per CLAUDE.md any JSON
  change lands in the same commit + re-validate (OTR_WorkflowValidator + link/widget audit). Default routing
  UNCHANGED (lane stays dark until the operator selects it).
- Audio: reuse the per-beat frozen-master audio slice (Subproject-C) as the A2V conditioning input; stage it
  under a shot/seed name; confirm LTX-2.3's expected sample-rate/format; never re-encode the master.

## BUGS/RISKS (C) -- pre-empt
VRAM OOM/thrash (dominant) -> distilled-first + offload + small-res + batch1 + reset-before-run + watchdog;
GGUF loader absence -> node-gate fail-closed; audio-sync drift -> clip length == beat audio, V-1 byte-identical
test; "render-finished-leaves-resident" -> reclaim + reset; determinism seed-keyed; UTF-8/SFW.

## POLISH (D)
Stage-2 spatial upscaler only AFTER base is stable (likely v2); optical-flow/framediff + eyeball gate vs the
5/30 gold b005; README "what each engine gives" (LTX prompt-only vs LTX-AV audio-driven vs HuMo lip-sync);
the cheap Lane-A win (audio-reactive ledger->prompt motion verbs) that may DEFER Lane B for announcer/music.

## TICKETS (coder windows; each: suite + Bug Bible + audio-byte-identical green per chunk)
- **M0 (probe-or-park).** Fetch distilled-v1.1 + Gemma encoder + the LTX-2.3 VAE to `C:\ComfyUI-Models`
  (sha+license manifest); a BARE-GRAPH A2V smoke at 384x216 / 4 s / batch1 / Gemma-offloaded on the 5080.
  Record peak NVML. **If it OOMs or thrashes -> PARK Lane B, write the finding, Lane A stands.** No engine
  code until M0 passes.
- **M1.** `eng_ltx_av.py` skeleton: core + 2 adapters, assert_usable (flag/Sage/NVML-required/node/weights/
  dims), dark @register, CAPABILITIES row, CPU unit tests (no GPU).
- **M2.** Frozen-audio V-1: video-only decode + the has_audio-False / no-audio-path test + byte-identical
  green; the per-beat audio-slice conditioning input.
- **M3.** Wiring: dropdown option in the prod JSON (same commit, re-validate), role_compat, fallback chains
  re-grounded; a forced `OTR_FORCE_ENGINE_MAP=*=ltx_av_*` smoke.
- **M4.** Graduation soak: N short clips no-OOM + optical-flow/framediff vs 5/30 gold + character-lip-sync
  vs HuMo eyeball. Promotes Lane B from dark to selectable ONLY if it beats Lane A/HuMo at acceptable cost.

## INVARIANTS
test_audio_byte_identical GREEN (V-1); single heavy resident proven on 16 GB with margin; 100% local;
determinism; LOUD fallbacks; UTF-8 no BOM; SFW; additive-only (shipped engines untouched); no new static
widgets (V-11); JSON node/wiring changes in `otr_scifi_16gb_full.json` same commit.
