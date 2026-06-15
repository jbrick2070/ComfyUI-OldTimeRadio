# CODER KICKOFF -- LTX (AUDIO INPUT) LANE  [ltx_av]  -- ADDITIVE, M0-FIRST

> This is the **LTX AUDIO-INPUT (A2V) lane** -- a NEW, DARK, additive engine that drives video from the
> per-beat FROZEN audio + prompt + still. It is NOT the prompt-only `ltx_video` engine.

## #1 HARD GUARDRAIL -- DO NOT TOUCH THE GOLDEN PROMPT-ONLY LTX
The shipped **`ltx_video`** engine (`nodes/_otr_video_engines/eng_ltx_video.py`) is **GOLDEN / FROZEN**. It
carries today's shipped motion work (boomerang `loop_via_reverse`, ksampler default, `music_open` opener,
832x480) and is in production. The audio lane is **100% ADDITIVE**:
- **NEW file only:** `nodes/_otr_video_engines/eng_ltx_av.py`. **NEW engines only:** `ltx_av_talk`,
  `ltx_av_music`.
- **DO NOT edit `eng_ltx_video.py`.** DO NOT touch `OTR_LTX_SAMPLER`, `OTR_LTX_OPEN_MOTION_KEY`,
  `OTR_LTX_LOOP_VIA_REVERSE`, `_LOOP_VIA_REVERSE_DEFAULT`, `_sampler_mode`, the `_LTX_MOTION_PROMPT_BY_ROLE`
  templates, or the `ltx_video` dropdown options. The prompt-only LTX recipe is the operator's golden look.
- The two lanes diverge on purpose (e.g. frame math: `ltx_video` snaps DOWN, `ltx_av` snaps UP) -- never
  "unify" them, never copy one's defaults into the other.
- `test_audio_byte_identical` and every existing `ltx_video` test must stay GREEN, untouched.

## THE PLAN (read these, in order)
1. **`docs/2026-06-15-ltx-av-alternative/roundtable/pass03_plan_FINAL.md`** -- the CONVERGED refresh (3 live
   panel passes + Claude judge, 2026-06-15): the decisions, the M0-gate, the model/VRAM reality, the deltas.
2. **`docs/2026-06-10-ltx-av-lane/LTX_AV_SPRINT_PLAN.md`** -- the AUTHORITATIVE detail spec (8-pass converged
   6/10). Owns the locked internals (snap-UP `next_8n1`, the assert_usable gate, canonicalize-to-T, the
   encoder->reclaim->transformer phasing, schemas family, V-6 auto-dropdown = NO Director/JSON edit). Its
   LINE REFS ARE STALE vs HEAD `334e002` -- re-locate every reference before any delta.

## START HERE -- M0 GRAPH SPIKE (probe-or-park; NO engine code yet)
The LTX-2.3 A2V graph is UNKNOWN. Before any engine code, capture it on THIS 5080:
- Read the official LTX-2.3 ComfyUI template; capture a live `/object_info` (Desktop AND headless) -> record
  the exact node classes (audio loader, conditioning, sampler, and the **TERMINAL video-only decode**), the
  viable LOW-VRAM artifact (A2V is 22B-class; fp8 likely dead under 14.5 GB -> rank GGUF Q3_K_S/Q3_K_M +
  Gemma-encoder CPU-offload + block-swap; NVFP4 is CUT), the VAE decode floor at 384x216 / 512x288, and the
  PEAK NVML at batch1 / 4-6 s / Gemma offloaded.
- Write the GRAPH SPEC into the pass03 doc. **If it OOMs / thrashes / can't prove <= 14500 MB on the 5080 ->
  PARK Lane B, write the finding. Lane A (the golden prompt-only LTX) stands as production. Nothing lost.**
- Only AFTER M0 GO: M1 skeleton (singleton `_ltx_av_core` + 2 adapters + ordered assert_usable + schemas
  family) -> M2 frozen-audio V-1 (video-only decode, zero-audio-stream ffprobe test, 8n+1-padded audio
  input, canonicalize to exactly T) -> M3 wiring (NO JSON edit; explicit force-map
  `announcer_visual=ltx_av_talk,character_video=ltx_av_talk,music_visual=ltx_av_music`) -> M4 graduation
  (lip-sync vs HuMo A/B + N=3 no-OOM, promote-or-park).

## INVARIANTS / RULES (CLAUDE.md + the plan)
V-1 frozen audio byte-identical (drop LTX's generated audio; only OTR_MasterAudioMux emits audio); single
heavy resident ONLY if M0 proves <= 14500 MB else PARK; 100% local/offline; determinism seed-keyed; every
in-render fallback LOUD; UTF-8 no BOM; SFW; V-12 lazy heavy imports + cu130 freeze STOP rule. Run the full
suite + Bug Bible + `test_audio_byte_identical` after EVERY code change; commit per green chunk + push (one
push attempt, then a PowerShell block); do NOT touch the operator's CLAUDE.md edit; prod/main GATED. EVERY
session updates GO_FORWARD_PLAN.md + the otr-build-tracker.

## DOES NOT DISPLACE THE FORWARD ORDER
This is an OPERATOR-GATED OPTIONAL track. The §3 forward order (Wan / 3D / distribution) is unchanged. Start
the LTX-AV M0 spike only on the operator's explicit GO.
