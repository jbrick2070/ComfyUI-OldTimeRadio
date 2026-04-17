# OTR Roadmap

**Last updated:** 2026-04-17 (video stack consult folded in, supporting files retired)
**Branch:** `v2.0-alpha` (+ sprint fork `v2.0-alpha-video-stack`)
**Owner:** Jeffrey A. Brick

**This file is the single source of truth.** Canonical going-forward plan. Three horizons: **v1.7 audio pipeline** (shipped, live-test cycle ongoing), **v2.0 video stack sprint** (14-day build, drives the next two weeks), and **v2.0 continuity layer** (Scene-Geometry-Vault + Style-Anchor cache, post-sprint). Everything shipped or discarded stays in source docs — this file is open items only.

---

## Platform Pins

Lock these. Any work item that contradicts this list is wrong.

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA.
- Flash Attention 2/3: NOT AVAILABLE. Do not chase.
- 100% local, offline-first, open source, no API keys.
- VRAM ceiling: **14.5 GB audio** / **15.5 GB video** (lifted 2026-04-17 for the video stack sprint only — audio stays at 14.5 GB).
- Audio is king. Full narrative output must never break, shorten, or degrade.

---

## P0 — Video Stack Sprint (14-day build)

Sprint fork: `v2.0-alpha-video-stack` off `v2.0-alpha`. Tag target: `v2.0-alpha-video-full`.
Supersedes the retired HY-World 2.0 Gate 0 probe. The HyworldBridge → HyworldPoll → HyworldRenderer trio (shipped) stays as the harness; the backends swap.

### Locked stack

| # | Stage | Pick | Runtime | Peak VRAM | Canonical repo |
|---|---|---|---|---|---|
| 1 | Style anchors | FLUX.1-dev FP8 + ControlNet Union Pro 2.0 | diffusers | 12.5 GB | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` |
| 2 | Scene keyframes | FLUX.1-dev + Depth/Canny | diffusers | 13.5 GB | `XLabs-AI/x-flux` (weights) |
| 3 | Character lock | PuLID for FLUX | diffusers | 14.0 GB | `ToTheMoon/PuLID` *(verify Day 3)* |
| 4 | Hero motion | LTX-Video 2.3 | existing sidecar | 14.5 GB | `Lightricks/LTX-Video` |
| 5 | Long motion / VJ loops | Wan2.1 1.3B I2V | diffusers | 8-10 GB | `Wan-Video/Wan2.1` |
| 6 | Compositing | Florence-2 + SDXL Inpainting | diffusers | 8 GB | `microsoft/Florence-2-large` (HF) |
| 7 | Final mux | HyworldRenderer (shipped `86bfeae`) | ffmpeg | — | in-repo |

Post-processing: ffmpeg + OpenCV VHS stylizer (scanlines, chroma bleed, HUDs, lower-thirds).

### Fallbacks (real, reserved — do not promote without cause)

- Stage 1: SDXL 1.0 + 1980s VHS LoRA stack
- Stage 3: InstantCharacter (`Tencent-Hunyuan/InstantCharacter`)
- Stage 4: HunyuanVideo via Nunchaku INT4 (`mit-han-lab/nunchaku`) if LTX quality ceiling hit
- Stage 5: FramePack (`lllyasviel/FramePack`)
- Stage 6: Insert Anything (`song-wensong/insert-anything`)
- FP8 spike escape across any FLUX stage: GGUF Q8/Q5 via `city96/ComfyUI-GGUF` logic ported into sidecar

### 14-day sprint

Every day ends with: `pytest tests/bug_bible_regression.py`, `pytest tests/test_dropdown_guardrails.py`, `pytest tests/v2/test_audio_byte_identical.py`. No exceptions. C7 failure halts and reverts the day's work.

| Day | Task | Gate |
|---|---|---|
| 1 | **[DONE 2026-04-17]** `backends/` harness, `_base.py`, STATUS.json schema, `placeholder_test.py`. Wire Bridge `backend=` arg + LHM cooldown gate. Fixed bridge.py:296-299 PIPE deadlock (stdout/stderr → per-job log files). | ✅ 14/14 new dispatch tests green; 26/26 Bug Bible; 56/56 dropdown guardrails; 22/22 anchor_gen. C7 unchanged. Pre-existing BUG-LOCAL-042 vram_sentinel errors surviving (not caused by Day 1). |
| 2 | `flux_anchor.py` — diffusers FP8 FLUX baseline. Pin diffusers version in `requirements.video.txt`. | 1024² renders ≤ 12.5 GB. No `flash_attn` imports in venv trace. |
| 3 | `WebFetch` verify PuLID canonical upstream. `pulid_portrait.py` — 3 refs → portrait. | Identity locked across 5 renders, same refs, different prompts. ≤ 14 GB. |
| 4 | `flux_keyframe.py` — FLUX + Depth/Canny ControlNet. | Same layout preserved across 3 prompt variations. ≤ 13.5 GB. |
| 5 | Bridge orchestration: FLUX still → LTX-2.3 handoff. | No VRAM fragmentation FLUX→LTX. Clip renders clean. |
| 6 | `wan21_loop.py` — Wan2.1 1.3B I2V. | 10s loop from FLUX still. ≤ 10 GB. |
| 7 | `florence2_sdxl_comp.py` — text-prompt mask → SDXL inpaint insert. | "cockpit window" mask → CRT overlay insert. ≤ 8 GB. |
| 8 | ffmpeg/OpenCV VHS post-processor as final filter before Renderer. | C7 byte-identical gate passes. |
| 9 | `otr_v2/hyworld/planner.py` — orchestration timeline planner. | Given `outline_json`, emits non-repeating sidecar job list covering full runtime. 3-min dry run clean. |
| 10 | Cold-open canary: "SCENE 01 — Cockpit, Baba boots up the radio." Full Stage 1→7 pass. | End-to-end render, audio aligned, no black frames. |
| 11 | 3-min continuous scene at full res. | No stagnation, no duplicate stills. Wall clock < 45 min. |
| 12 | Character regression: Baba + Booey between Scene 1 and Scene 3. | SSIM > 0.85 on cropped faces. |
| 13 | Full 20-min episode dry run overnight. LHM polled every 60 s via scheduled task. | No OOM, no pagefile thrash, no shared-memory fallback. |
| 14 | Freeze stack. Tag `v2.0-alpha-video-full`. Update BUG_LOG, MEMORY, this ROADMAP. | All three test suites pass. Zero `video-stack` blockers in BUG_LOG. |

### Kill criteria (fail-fast; do not hero-fix)

| Stage | Kill trigger | Fallback |
|---|---|---|
| 1 Anchors | FLUX FP8 peak > 14 GB on 1024² OR `flash_attn` import detected | SDXL 1.0 + 1980s VHS LoRA |
| 2 Keyframes | Union Pro 2.0 fails on diffusers + torch 2.10 | FLUX + single Depth CN |
| 3 Portraits | PuLID no identity lock after 10 attempts × 3 ref packs | InstantCharacter |
| 4 Motion | LTX-2.3 quality regression vs current baseline | HunyuanVideo via Nunchaku INT4 |
| 5 Loops | Wan2.1 peak > 12 GB OR visible VAE temporal drift | FramePack |
| 6 Comp | SDXL inpaint seams > 5 px | Insert Anything |
| 7 Mux | Any C7 regression | Revert, hold day's work |

**Overarching kill rule:** audio degrades in any way → revert immediately. Audio is king.

### Video-stack risks

1. **VRAM fragmentation on Windows spawn.** PyTorch doesn't always release VRAM to OS. Mitigation: Bridge cooldown gate — `libre_tail.snapshot()` must show GPU free ≥ 2 GB before spawn, else 3-s wait + `WORKER_VRAM_BLOCKED` fail-fast; 2-s sleep + `torch.cuda.empty_cache()` post-exit.
2. **FP8 scaling bugs on sm_120 without FA3.** Mitigation: pre-pin GGUF Q8 variant as instant fallback; do not chase FA3.
3. **I2V temporal drift on chained gens.** Mitigation: planner NEVER chains video-to-video; always regenerates motion from a pristine FLUX still.
4. **diffusers + torch 2.10 + FLUX FP8 incompatibility.** Mitigation: pin exact diffusers version Day 2 in `requirements.video.txt`.
5. **Bark/audio interference from co-running video sidecars.** Mitigation: daily C7 gate, separate process trees.
6. **Windows PIPE backpressure deadlock** (already flagged in `bridge.py:296-299`). Mitigation: stderr → tempfile, never `stderr=PIPE` undrained.

### Sanity pass findings (2026-04-17)

1. PuLID upstream uncertain — Day 3 `WebFetch` verification before clone.
2. diffusers version must be pinned Day 2 in `requirements.video.txt`.
3. C7 audio regression runs at end of EVERY day, not only Day 10.
4. Bridge cooldown gate is non-negotiable (LHM free ≥ 2 GB).
5. Both consultants had errors — trust the verified repos above, not either consult raw output.
6. No `ComfyUI-*-Wrapper` as primary runtime (pulls flash_attn, wraps add overhead) — diffusers native or raw model code only.
7. Hotfixes on `v2.0-alpha` during sprint → rebase `v2.0-alpha-video-stack` daily.

### Definition of Done (Day 14)

- `v2.0-alpha-video-full` tagged on origin.
- 20-min episode renders end-to-end with no manual steps.
- C7 audio byte-identical to v1.5 baseline.
- No `flash_attn` imports in venv trace.
- No `CheckpointLoaderSimple` in any live workflow (C2).
- All visual generation in subprocesses (C3).
- Character identity SSIM > 0.85 on face crops between Scene 1 and Scene 3.
- BUG_LOG has zero open `video-stack` blockers.
- ROADMAP.md updated; items 3/4/6/8 unblocked into P2.
- MEMORY.md gets a project memory summarizing the shipped stack.

### New backend module layout

```
otr_v2/hyworld/backends/
  _base.py               # write_status(), STATUS.json schema, cooldown helper
  placeholder_test.py    # Day 1 spawn/cleanup canary
  flux_anchor.py         # Stage 1 — diffusers FP8 FLUX + Shakker-Labs ControlNet
  flux_keyframe.py       # Stage 2 — FLUX + Depth/Canny
  pulid_portrait.py      # Stage 3 — FLUX + PuLID identity insertion
  ltx_motion.py          # Stage 4 — wraps existing LTX sidecar under uniform STATUS contract
  wan21_loop.py          # Stage 5 — Wan2.1 1.3B I2V
  florence2_sdxl_comp.py # Stage 6 — Florence-2 mask + SDXL inpaint
```

Bridge contract additions: `backend=<name>` arg; pre-spawn LHM cooldown gate; post-exit `empty_cache()` + 2 s sleep; STATUS.json adds `peak_vram_gb` field for learned ceilings.

---

## P1 — Audio pipeline (shipped, live-test cycle)

All items code-complete and on `v2.0-alpha`; awaiting real-soak verification as episodes run.

| Item | Summary | Status |
|---|---|---|
| `min_line_count_per_character` self-critique guard | Injected floor=2 into `_critique_and_revise()`; rejects revision if any character drops below. Falls back to pre-critique draft. | Shipped, needs live test |
| Director JSON schema + validator | `_DIRECTOR_SCHEMA` + `_validate_director_plan()` in LLMDirector; repairs missing entries, validates voice_preset strings, filters broken sfx, clamps duration. Wired in `direct()`. | Shipped, needs live test |
| Length-sorted Bark batching | Sort by line length within preset group; script order restored at assembly. Pure throughput win. | Shipped, needs live test |
| VRAM-Sentinel decorator | `vram_sentinel(phase_label, max_entry_gb)` on `BatchBarkGenerator.generate_batch()` at 6 GB ceiling. CUDA-absent safe. | Shipped, needs live test |
| High-creativity soak profile | `"maximum chaos"` re-added to CREATIVITIES pool (~10% weighted). Catches temperature-sensitive regressions. | Shipped, needs live test |
| Per-LLM-call VRAM snapshots | `vram_snapshot("llm_generate_entry"/"exit")` inside `_generate_with_llm()`. Logs tokens + inference time. | Shipped, needs live test |

---

## P2 — Continuity layer (unblocks after video stack sprint ships)

Previously blocked on the retired Gate 0. Now blocked on video stack sprint Day 14. Design begins once stack empirics exist.

| Item | Summary |
|---|---|
| Scene-Geometry-Vault | Series-scale persistent geometry vault so Act 3's bridge matches Act 1's bridge across episodes. Seeded by FLUX anchor outputs from Stage 1. |
| Style-Anchor cache (World Seed + Lighting/Mood split) | Reuse engine over the vault. Same geometry, N relight passes. `style_anchor_hash` in Director schema keys the split. |
| Head-Start async pre-bake (Phase B.5) | Kick off HyworldBridge on `outline_json` while ScriptWriter + Director run. Wall-clock win. Blocked on vault stability. |
| ASCII sanitizer in prompt_compiler | Strip non-ASCII before Tencent text encoders. Preserve case. Collapse whitespace. Fold into `flux_anchor.py` prompt compiler on video-stack Day 2. |

---

## P3 — Experiments & polish

| Item | Summary |
|---|---|
| `torch.compile` on Bark sub-models | `mode="reduce-overhead"` on semantic, coarse, fine acoustic. Needs isolated A/B timing; variable-length loops may fight the compiler. |
| Skip/shorten Bark fine acoustic pass | Fine pass detail that AudioEnhance destroys via tape emu / LPF / Haas. Needs listening test, not spectrogram. |
| `episode_title` socket input on OTR_SignalLostVideo | Replace implicit `script_json` title-token read with explicit socket from ScriptWriter. v2.1 cleanup. |
| Rename `workflows/soak_target_api.json` → `workflows/helpers/antigrav_api_scratch.json` | Antigravity API-conversion helper; keep but move out of top-level workflows to reduce confusion. |

---

## Recently shipped

| Item | Summary | Status |
|---|---|---|
| v1.7 | Tagged and merged to `main` (`0aa6d6e`) | Shipped |
| BUG-LOCAL-034–040 | Parser resilience, title fixes, JSON repair | Shipped with v1.7 |
| HyWorld sidecar trio | HyworldBridge + HyworldPoll + HyworldRenderer wired into `workflows/otr_scifi_16gb_full.json` | Shipped |
| HyworldRenderer audio-length exact-match | `-t audio_duration` + `tpad` for C7 safety; stderr → tempfile | Shipped (`86bfeae`) |
| Phase A race-free sidecar contract | Atomic writes + Windows `os.replace` retry (`_atomic.py`) | Shipped (`ed4c44f` + `5e795a0`) |
| Phase B v0 SD 1.5 anchor generator | `anchor_gen.py` behind `OTR_HYWORLD_ANCHOR=sd15` flag; 27 unit tests | Shipped (`c46a013`) |
| Round-robin consult infrastructure | `scripts/_consult_round_robin.py` (ChatGPT → Gemini → Claude synth) | Shipped |

---

## Discarded (do not revisit)

- Flash Attention 2/3 on sm_120
- Pinning torch < 2.10 (stale by multiple minor versions)
- Weight streaming from system RAM via ComfyUI-Manager
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)
- Speculating on unreleased HyWorld unified latent space
- **HY-World 2.0 Gate 0 probe** (WorldMirror / HunyuanWorld / WorldStereo / WorldPlay-5B) — retired 2026-04-17. HyworldBridge + Poll + Renderer harness stays; the backends are the P0 video stack above.
- `ComfyUI-*-Wrapper` repos as primary runtime (pull flash_attn, wrap overhead)
- v2v chaining (deep-fries output by 3rd generation)
- Single-image LoRA training on the laptop during live orchestration (thrash risk)
- SD 1.5 anchors as final style — did not read as 1980s VHS (pivoted to SDXL + period LoRA, now superseded by FLUX-native anchors under P0)

---

## References

- `CLAUDE.md` — project rules, platform pins, Desktop Commander git pattern
- `docs/BUG_LOG.md` — live bug tracking
- `docs/HANDOFF_2026-04-16.md` — last handoff (Phase A + Phase B v0)
- `docs/2026-04-15-hyworld-integration-plan-review.md` — external review triage (HY-World 2.0 probe now retired)
- `docs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/2026-04-14-otr-v2.1-spec.md` — v2.1 spec
- `docs/2026-04-14-green-zone-guardrail-decision.md` — guardrail decision
- Survival guide: `https://github.com/jbrick2070/comfyui-custom-node-survival-guide`

---

## Daily operating cadence

- First thing: read this file, `CLAUDE.md`, `BUG_LOG.md` header, `git log --oneline -5` on current branch.
- LHM is always on — poll `http://localhost:8085/data.json` (or `outputs/libre_tail.py`) before asking Jeffrey for system status.
- After every code change: AST parse + three regression suites. Do not report "done" until green.
- One `git push` attempt max — if it fails, hand Jeffrey a cmd block with `cd /d` included.
- Verify every push: local HEAD == origin HEAD, no 0-byte files, no BOM, workflow JSONs valid.
- Log bugs the moment they surface. Don't batch. Promote `Bible candidate: yes` to the survival guide only after the fix is verified.
