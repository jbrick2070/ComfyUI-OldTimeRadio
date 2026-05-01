# OTR Roadmap

**Branch:** `v2.0-alpha` | **Owner:** Jeffrey A. Brick | **Last refactored:** 2026-04-30

This file is the **canonical going-forward plan**. Forward-only. Historical session logs and "what shipped" archives are in `docs/ROADMAP_HISTORY.md`.

**Canonical narrative hierarchy** — every ledger, workflow, and doc in this repo follows this:

```
Scene  >  Shot  >  Beat  >  Clip
```

- **Scene** — high-level narrative location (`AstroTech Research Facility`, `Control Room`, ...). One per `scene_id`.
- **Shot** — continuous visual unit. Same framing, same lighting. May contain multiple speakers.
- **Beat** — single-speaker continuous turn within a shot. The unit at which the 7 s clip-fill rule applies — beats never cross speakers, so HuMo audio windows align to one voice.
- **Clip** — one HuMo render call. Length must be `4n + 1` frames (Wan VAE temporal compression of 4) and ≤ 177 (verified ceiling on 16 GB).

Every consumer of `ledger.json` must understand all four levels.

---

## Platform Pins

Lock these. Any work item that contradicts this list is wrong.

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA.
- Flash Attention 2/3: NOT AVAILABLE. Do not chase.
- 100% local, offline-first, open source, no API keys for the shipped pipeline. Cloud LLMs (OpenAI / Gemini / NVIDIA NIM) are for **internal QA round-robins only**, never shipped output.
- VRAM ceiling: **14.5 GB audio** / **15.5 GB video** (lifted 2026-04-17 for the video stack only — audio stays at 14.5 GB).
- Audio is king (rule **C7**). Full narrative output must never break, shorten, or degrade. If video breaks audio, revert immediately. Audio output must remain byte-identical to v1.5 baseline at every gate.

---

## P0 — Active focus

### Live-test verification of the radio-coverage + bit-perfect-audio architecture

The code path is in place; what's open is real-run verification. Every item below is something to confirm or surface during the next ComfyUI episode test, NOT new design work.

**Verify on the next clean run:**

- `ledger.lines[]` carries a `speaker_role` on every entry. No nulls, no missing rows. Roles: `character` / `announcer` / `music_open` / `music_close` / `music_inter` / `sfx`.
- `ledger.meta.audio_path_selected = "master_mix_per_clip_mux"` and `audio_path_reason = "ok (zero audio re-encodes downstream of SignalLostVideo)"`.
- For every line where `speaker_role ∈ {announcer, music_*, sfx}`, the `BatchHumoRender` log line shows `ref=radio_bookend_<ep>.png source=radio-still (...)` — radio still I2V dispatch confirmed.
- `ledger.meta.radio_bookend_prompt_source` populated with the dynamic-build branch tag (e.g. `"dynamic (genre=sci-fi)"`).
- Music tracks > 7s show up as multiple chunked entries (`music_open_001`, `music_open_002`, ...) — chunking math fired.
- ffprobe on the final mp4: video + audio streams both present; final mp4 audio `codec_name == aac` (passthrough from procgen).
- No `[VideoComposite] master_mix_per_clip_mux FAILED` in the log. With `strict_c7=True` (default), any failure would have raised.

**Open follow-ups (in priority order):**

- **Audio codec ffprobe pre-flight** (P2) — confirm procgen audio stream is AAC before `-c:a copy` mux. One-line subprocess.run + assertion. Trivial; deferred until first run confirms procgen output codec.
- **Post-mux audio stream identity validation** (P3) — extract per-stream packet hash on procgen vs final mp4, fail tier on mismatch. Concrete proof of bit-identity. Ship as a separate validation node since the ffmpeg incantation needs care on Windows.
- **Low-motion observability for radio HuMo clips** (P3) — frame-difference metric on non-dialogue clips so "static" failures (Whisper OOD producing flat frames) surface as warnings instead of going unnoticed. No behavior change.
- **HuMo continuity layer for >7s narrative beats** (v2.0-beta) — hybrid blending across HuMo windows so 30s narrative beats don't show 7s jump-cuts. Decoupled from the audio path; gates "production unattended."
- **Per-scene environment FLUX still + LTX/zoompan animated background** (v2.0-beta) — bottom layer under the HuMo center pillarbox in dialogue windows.
- **Procgen-CRT lighten layer on top** (v2.0-beta) — audio-reactive scanlines + flicker as the SIGNAL LOST signature.
- **Drifted-filename smoke for BUG-LOCAL-118** — force an underscore-drifted .mp4 stem to verify the fallback chain fires before relying on it in a long soak.
- **Reconcile `16294df` ROADMAP-vs-git-log mismatch** — git log says "BUG-LOCAL-112 news-history reset"; prior narrative had it as "Wire ScriptCritic." Likely a rebase artifact. Decide canonical message before the next QA pass walks the history.

#### Hardware floor (locked 2026-04-25, do not relitigate)

- HuMo 14B fp8 e4m3fn scaled (Kijai) — `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`. Stock `UNETLoader`. Tuned by Kijai for 16 GB cards.
- Fallback ladder (kept on disk, do NOT delete): `humo_17B_fp8_e4m3fn.safetensors` (highest quality, slower ~6 min/clip), `Wan2_1-HuMo-17B_Q5_K_M.gguf` (speed-tuned).
- Stable shape: `length=97` (3.88 s @ 25 fps), 480x832, batch=1. Or `length=177` at 640x640 (7 s, OOD but verified working).
- Frame count must be `4n + 1`. Helper `humo_length_for_dur(dur_s)` snaps. Cap mirrored to `7.0s` in EpisodeAssembler music chunking.
- Per-step: 42 s. Per-clip: ~4:30 native, ~6:15 in TEST_humo. Cold load: ~50 s.

---

## P1 — Audio pipeline (live-test cycle)

All items code-complete and on `v2.0-alpha`; awaiting real-soak verification as episodes run.

| Item | Summary | State |
|---|---|---|
| `min_line_count_per_character` self-critique guard | Floor=2 in `_critique_and_revise()`; rejects revision if any character drops below; falls back to pre-critique draft | Live-test |
| Director JSON schema + validator | `_DIRECTOR_SCHEMA` + `_validate_director_plan()`; repairs missing entries, validates voice_preset, filters broken sfx, clamps duration | Live-test |
| Length-sorted Bark batching | Sort by line length within preset group; script order restored at assembly. Pure throughput win | Live-test |
| VRAM-Sentinel decorator | `vram_sentinel(phase_label, max_entry_gb)` on `BatchBarkGenerator.generate_batch()` at 6 GB ceiling. CUDA-absent safe | Live-test |
| High-creativity soak profile | `"maximum chaos"` re-added to CREATIVITIES pool (~10% weighted). Catches temperature-sensitive regressions | Live-test |
| Per-LLM-call VRAM snapshots | `vram_snapshot("llm_generate_entry"/"exit")` inside `_generate_with_llm()`; logs tokens + inference time | Live-test |
| ScriptCritic + Reviser advisory gate | Wired into `otr_scifi_16gb_full.json` (id=53). Dynamic anti-slop rubric with `[applies_when]` gates filtered per-run from ledger `gen_params_initial`. Default `block_on_reject=False` | Live-test (3-5 runs before flipping `block_on_reject=True`) |

---

## P2 — Continuity layer

Blocked on video-stack maturity. Design begins once stack empirics exist from the live-test cycle.

| Item | Summary |
|---|---|
| Scene-Geometry-Vault | Series-scale persistent geometry vault so Act 3's bridge matches Act 1's bridge across episodes. Seeded by FLUX anchor outputs |
| Style-Anchor cache | Reuse engine over the vault. Same geometry, N relight passes. `style_anchor_hash` in Director schema keys the split |
| Head-Start async pre-bake (Phase B.5) | Kick off VisualBridge on `outline_json` while ScriptWriter + Director run. Wall-clock win. Blocked on vault stability |
| ASCII sanitizer in prompt_compiler | Strip non-ASCII before Tencent text encoders. Preserve case. Collapse whitespace |
| Diff 3 — spine ledger-stamping + schema bump l3 → l4 | New ledger fields (`outline`, `beats[]`, `spine_meta`) + bundled metadata (`episode_title`, `meta.gen_params`, `meta.news_seed`, `meta.bug_109_retries`, `meta.word_ratio_pct`, `meta.title_source`, `meta.episode_breakdown_s`). See `docs/2026-04-29-spine-ledger-stamping-ticket.md`. **Unblocked by:** 2-3 real-episode runs of `voice_warnings[]` + Mistral-Nemo + Gemma 4 E4B both PASSing the LLM edge-case matrix + v2.0-alpha video stack feature-complete |

---

## P3 — Experiments & polish

| Item | Summary |
|---|---|
| `torch.compile` on Bark sub-models | `mode="reduce-overhead"` on semantic, coarse, fine acoustic. Needs isolated A/B timing; variable-length loops may fight the compiler |
| Skip/shorten Bark fine acoustic pass | Fine pass detail that AudioEnhance destroys via tape emu / LPF / Haas. Needs listening test, not spectrogram |
| `episode_title` socket on `OTR_SignalLostVideo` | Replace implicit `script_json` title-token read with explicit socket. v2.1 cleanup |
| News-history fuzzy dedup for syndication edge case | URL dedup catches direct repeats; same content with different URLs needs a fuzzy headline match |
| Empty-section pruning in filtered rubric | 1-character runs keep `### Ensemble-voice collapse` heading after all 3 rules filter out. Wastes tokens, doesn't break anything |
| VideoComposite cleanup deletion logic | Widget shipped (`cleanup_clips_after_assembly`), no-op for now. Wire actual deletion when stable enough to trust |
| Auto-update `OTR-CANON.md` from passing critic verdicts | `_canon_update()` helper exists in `script_critic.py` but is intentionally not called yet. Wire in once 3-5 runs of critic data accumulate |
| Tune `_MODEL_CONTEXT_CAPS` from real `OTR_VRAMContextTest` data | Currently conservative defaults |
| Update stale dropdown-guardrail tests in same commit as widget changes | Lesson from 2026-04-30: when widget mins/defaults change, update `tests/test_dropdown_guardrails.py` in the same commit so the test suite never drifts behind production |

---

## v2.0 release blockers

### B1 — Generic / relative paths (no Windows-hardcoded absolutes)

**Status:** Step 0 paths refactor shipped 2026-04-28 (`70f4a5c`) — `nodes/_otr_paths.py` helper module with resolution order: `OTR_OUTPUT_DIR` env → `folder_paths.get_output_directory()` → walk-up to ComfyUI root → cwd fallback. ~12-15 hardcoded `r"C:\Users\jeffr\..."` strings replaced.

**Remaining:**
- Workflow JSON path scrub: `workflows/otr_scifi_16gb_full.json` (`LoadAudio` widget hard-pinned `C:\...` mp4) and `workflows/otr_humo_smoke.json` (Resonance Chamber fixture path).
- Lean toward auto-discover with placeholder defaults for the smoke (preserves drag-and-queue UX); empty defaults for the FULL workflow (audio comes from upstream nodes anyway).
- Validation: `OTR_OUTPUT_DIR=/tmp/otr_test pytest tests/` runs cleanly with the override; documented in `README.md` for cloud / non-default installs.

**Why it's a release blocker:** every Windows-absolute path is a portability blocker for any non-Jeffrey user (Linux/Mac/RunPod/cloud) and a portability blocker for the 8GB-tier work. v2.0 cannot ship while paths are user-and-OS-specific.

### B2 — 8GB-VRAM-class user experience

**Stance:** v2.0 doesn't release until 8GB-class users get an enhanced visual output too.

**Architecture (Jeffrey 2026-04-28):** Single master JSON with bypassable video-stack groups. Shared audio chain → procgen, then multiple side-by-side render groups — each group bypassable via Ctrl+B. Final VideoComposite takes whichever group is active. 8 GB users bypass the HuMo + FLUX-fp8 (16 GB) groups and enable the GGUF (8 GB) groups.

**Locked picks (2026-04-30, after evaluating LTX 2.3, LTX-2 19B, ERNIE Image, NVIDIA CES 2026 NVFP4 announcement):**

| Component | 16 GB tier | 8 GB tier | Why |
|---|---|---|---|
| **Stills** | **NVFP4 FLUX.2** (RTX 50 Series, ~5 GB; falls back to FLUX-fp8 ~12 GB if NVFP4 unavailable) | **FLUX.1-dev Q4_K_S** (city96 GGUF, ~5-6 GB) | NVFP4 is the new official quantization NVIDIA announced at CES 2026 — 3x faster, 60% less VRAM than fp8 on RTX 50 Series. Q4_K_S is the safe 8GB GGUF option. |
| **Video** | **HuMo 14B fp8** + master_mix_per_clip_mux | **Wan 2.2 5B TI2V** (native ComfyUI template) | HuMo for character lip-sync (drives video from OUR Bark/Kokoro audio — the whole reason it exists). Wan 5B for 8GB atmospheric B-roll. |
| **Upscale — Speed option** | **RTX Video Super Resolution ULTRA** (~0 GB, HW-accelerated, target 4K, real-time) | **RTX VSR ULTRA** (same node, same zero VRAM cost) | Default. NVIDIA CES 2026 ComfyUI node. Whole-episode upscale, near-real-time, ships with RTX driver. Use this when speed matters more than maximum diffusion-based detail. |
| **Upscale — Quality option** | **SeedVR2 v2.5 NVFP4** (7B, ~6 GB on RTX 50 NVFP4, ~78 s per 65-frame 720p→1080p clip — full episode ~2-3 h on a 5-min run) | not viable on 8 GB | Whole-episode upscale via the diffusion upscaler. Quality king for AI-generated content. SeedVR2 v2.5 NVFP4 support landed via [PR #486](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/pull/486). On RTX 50 NVFP4: 3x faster + 60% less VRAM vs fp16 baseline. |

Both upscale options run on the WHOLE episode (every clip), exposed as a workflow toggle so the user picks per-run. Default = RTX VSR (fast). Quality run = SeedVR2 v2.5 (slow but state-of-the-art on AI content). Either can be toggled off entirely for raw 480x832 / 720p output.
| **TTS / Audio** | Bark + Kokoro + MusicGen + AudioGen → master mix (canonical) | Same | OTR's TTS pipeline is the project. NEVER replaced by model-internal A/V generation (LTX-2's prompt-driven audio is a paradigm mismatch). |

**Picks REJECTED after evaluation:**
- **LTX 2.3 22B distilled** — smallest GGUF (Q5_K_M ~14 GB) doesn't fit 8GB; "distilled" = step-distilled NOT param-distilled; 22B is the only param size Lightricks publishes.
- **LTX-2 19B distilled (Kijai)** — Q4_K_M ~12 GB still over 8GB.
- **LTX-2's built-in audio for character dialogue** — model GENERATES speech from text prompt, doesn't accept input audio. Replacing OTR's TTS would lose Bark/Kokoro voice control + script→voice mapping. Unless an audio-input ControlNet/LoRA ships, LTX-2 is visuals-only for OTR.
- **Wan 2.2 14B GGUF Q3/Q4** — RAM-thrashes on Windows under aggressive offload; support-ticket bait.
- **FLUX.1-dev Q5_K_S** — over 8GB budget once T5 + VAE + OS overhead added.
- **Z-Image Turbo / PixArt-Sigma** — weaker prompt-adherence than FLUX for radio-drama series consistency.
- **ERNIE Image 8B** — parked pending model card review (Jeffrey to provide spec link).

**Acceptance for 8GB path:**
- Full audio pipeline (LLM + Bark + AudioGen + MusicGen + SceneSequencer + EpisodeAssembler) — same as 16 GB.
- SignalLostVideo procgen base — same.
- Stills via FLUX.1-dev Q4_K_S; video via Wan 2.2 5B (atmospheric B-roll / scene loops).
- Final mp4 lands in `output/episodes_for_obs/<ep>/<ep>.mp4` same as 16 GB.
- Wall-clock expectation: FLUX still ~45-90 s/still, Wan 5B clip ~4-8 min/clip (significantly slower than 16GB tier; document upfront).

**Distribution requirements before tagging v2.0:**
- Pin exact ComfyUI version + GGUF model versions in README; include checksums for the GGUF files.
- README must set time expectations explicitly so 8GB users don't think the run hung.
- Both tier workflows live in the same JSON; the README screenshot shows the "8GB mode" group toggles to enable.

**Related:** flip default `optimization_profile` to `Pro (Ultra Quality)` once 16 GB FULL has shipped clean — Jeffrey: *"I almost feel we should default to Pro Ultra"*.

---

## v2.0-beta candidates

### LTX-Video animated backgrounds (3-layer composite)

Promotes the current 2-layer composite (procgen-base + HuMo-overlay, BUG-092) into a 3-layer composite when LTX is wired in:

```
TOP:    Procgen / CRT audio-reactive overlay -- `lighten` blend, ~0.3 opacity
MID:    HuMo lip-sync portrait -- center pillarbox during dialogue, opaque
BOTTOM: LTX animated background -- full canvas, opaque
```

**Why CRT-on-top in lighten mode is more truthful:** a failing broadcast's scanlines + audio-peak flicker should cover the WHOLE frame including the speaker's face — the interference doesn't politely stop at the pillarbox edges. Lighten mode takes max(CRT, underlying) per channel so artifacts ride on top without erasing detail.

**LTX render budget (locked 2026-04-29 PM — render-native + slow-mo):**
- Render at LTX's trained native 24 fps, then slow to 12 fps via ffmpeg `setpts=PTS*2,fps=12`. The slow-mo IS the SIGNAL LOST broadcast-degraded aesthetic.
- 1-2 LTX clips per SCENE (not per shot). Loop across the scene's duration via `-stream_loop -1` with optional crossfade or ping-pong reverse.
- 193 frames per clip = 8 sec native = 16 sec apparent after 2× slow-mo. Math: LTX uses 8x temporal VAE compression so frame counts must be `8n + 1`. 193 = 24*8 + 1.
- Optional dial-up to 241 frames (10 sec native, 20 sec apparent) for long scenes. Documented LTX max is 257.
- Distilled 4-8 steps (default 6).

**Per-episode wall-clock:** smoke (1 scene) ~50 s; short (3 scenes) ~2.5 min; medium (5 scenes) ~4 min. Negligible vs HuMo (~10 min per dialogue line).

**Frame-count widget on `OTR_BatchLTXRender`:**
```
ltx_frames:    [97, 145, 193, 241]   (8n+1 dropdown, default 193)
ltx_steps:     [4, 6, 8]             (distilled, default 6)
slow_mo_factor: float (default 2.0)  (1.0 = no slow, 2.0 = half-speed)
target_fps:    int (default 12)      (post-slow display rate)
```

**Bonus for 8 GB tier:** LTX 0.9 fp16 fits on 8 GB cards. Same `OTR_BatchLTXRender` node serves both 16 GB tier (background layer) and 8 GB tier (primary visual; HuMo bypassed). Single model, two roles via workflow toggle.

### LLM character normalize pass

Currently cast cleanup is two layers: (1) regex blocklist `_SFX_CAST_BLOCKLIST_PATTERNS` (BUG-091 + BUG-097), (2) fuzzy `_consolidate_similar_cast_rows_with_aliases` (BUG-098). Both deterministic, limited to KNOWN patterns. An LLM-based normalize after fuzzy dedup could catch semantic aliases neither layer sees: `KEVIN VOICEOVER` → `KEVIN STENDAHL`, `(captain)` lowercase → `CAPTAIN`, `DR. AMELIA HARTFIELD` → `AMELIA`.

**Constraints:** conservative prompt ("ONLY merge when names CLEARLY refer to the same character; when in doubt, do NOT merge"); hard-cap merge-set ≤50% of cast (flags hallucination); only run on `optimization_profile = "Pro (Ultra Quality)"` (adds 2-5 min wall time); feed first 1500 chars of script_text + first sentence of each character's first line.

**Defer to v2.0-beta** — by then we have a real corpus of run logs showing common emission patterns, so the prompt can be data-informed instead of guesswork-driven.

---

## Discarded — do not revisit

- Flash Attention 2/3 on sm_120
- Pinning torch < 2.10
- Weight streaming from system RAM via ComfyUI-Manager
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)
- Speculating on unreleased Visual unified latent space
- Visual 2.0 Gate 0 probe (WorldMirror / HunyuanWorld / WorldStereo / WorldPlay-5B) — retired 2026-04-17. VisualBridge + Poll + Renderer harness stays as the harness; the backends are the active video stack
- `ComfyUI-*-Wrapper` repos as primary runtime (pull flash_attn, wrap overhead)
- v2v chaining (deep-fries output by 3rd generation)
- Single-image LoRA training on the laptop during live orchestration (thrash risk)
- SD 1.5 anchors as final style — pivoted to FLUX-native
- Subprocess pattern for HuMo orchestration (BUG-076 OTR_PostAudioVideoPipeline + render_humo_batch.py orchestrator) — superseded 2026-04-27 by in-graph nodes (BUG-078). Subprocess scripts remain as ad-hoc CLI smoke tools but the production path is in-graph. `OTR_PostAudioVideoPipeline` class kept registered with `(retired)` title for back-compat with old workflow JSONs
- Blanket `git clean -fX` — the existing `scripts/_*.py` ignore is too broad and would nuke `_consult_*.py`, `yoga_watchdog.py`, and other legitimately-local files. Use targeted `git clean -fX -- <pattern>` instead

---

## References

- `CLAUDE.md` — project rules, platform pins, Desktop Commander git pattern
- `docs/BUG_LOG.md` — live bug tracking
- `docs/ROADMAP_HISTORY.md` — historical session logs and shipped-work archive (everything that used to live in this file)
- `docs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/2026-04-30-project-qa__04_synthesis.md` — most recent round-robin QA pass (OpenAI + Gemini + NVIDIA — NVIDIA was a non-vote due to context overflow)
- Survival guide / Bug Bible: https://github.com/jbrick2070/comfyui-custom-node-survival-guide

---

## Daily operating cadence

- First thing: read this file, `CLAUDE.md`, `docs/BUG_LOG.md` header, `git log --oneline -5` on current branch.
- LHM is always on — poll `http://localhost:8085/data.json` (or `outputs/libre_tail.py`) before asking Jeffrey for system status.
- After every code change: AST parse + three regression suites (Bug Bible regression in survival-guide repo, `tests/test_dropdown_guardrails.py`, `tests/test_core.py`). Don't report "done" until green.
- One `git push` attempt max — if it fails, hand a cmd block with `cd /d` included.
- Verify every push: local HEAD == origin HEAD, no 0-byte files, no BOM, workflow JSONs valid, all node classes registered in `__init__.py`.
- Log bugs the moment they surface. Don't batch. Promote `Bible candidate: yes` to the survival guide only after the fix is verified AND a real run confirms the behavioural fix.
- Round-robin consult before non-trivial design decisions (CLAUDE.md "Round-Robin Consultation" rule). Save transcripts under `docs/<date>-<topic>/`.
- Never use PowerShell for git operations — always cmd shell via Desktop Commander (PowerShell mangles `&&` and commit message quoting).
