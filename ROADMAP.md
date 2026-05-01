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
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0.

**Canonical stack (do not downgrade):**
- CUDA 13.x / cu130
- PyTorch cu130 matching the ComfyUI environment
- SDPA as guaranteed fallback
- SageAttention only when the cu130 wheel/source build matches Python + Torch exactly
- FlashAttention not required for shipped OTR

CUDA 13 is non-negotiable because (1) Blackwell sm_120 support is the point, (2) NVFP4 / FP4 model support in ComfyUI requires `comfy-kitchen` which requires CUDA 13+, (3) Task #2 SeedVR2 v2.5 NVFP4 path needs cu130 downstream. The cu128 SageAttention path exists in the wild and is the easier wheel target, but it belongs in a SEPARATE experimental ComfyUI folder if needed for sandbox work — never in the production OTR pipeline.

**Attention backend policy:**
- Default: PyTorch SDPA (boring, safe, in-tree).
- Preferred acceleration: SageAttention via KJNodes "Patch Sage Attention" node, tested per-workflow only.
- Do NOT use global `--use-sage-attention` unless a specific model/workflow has passed smoke testing — Triton route can produce black outputs with some models.
- FlashAttention 2/3: out of scope on Windows Blackwell. Do not chase community wheels for the shipped pipeline.
- FlashAttention 4: real and worth tracking (`pip install flash-attn-4`, exposes `flash_attn.cute` namespace), but NOT a ComfyUI production dependency yet. Older FA2-style custom nodes hard-coding the top-level import won't see it. FA4 is the future-looking transformer/training answer; SageAttention is the practical diffusion/ComfyUI answer today.
- Any third-party attention wheel must pass before shipping: import test → one FLUX smoke → one Wan/HuMo smoke → no black frames → no VRAM regression → no audio-path impact. Then it's blessed.
- Note on SageAttention wheel sourcing: `mobcat40/sageattention-blackwell` is the leading prebuilt wheel repo for sm_120, but its primary build line is PyTorch 2.11 nightly + CUDA 12.8. A cu130 build exists in that repo, but verify with smoke workflow on our pinned torch 2.10.0 / CUDA 13.0 stack before blessing.
- 100% local, offline-first, open source, no API keys for the shipped pipeline. Cloud LLMs (OpenAI / Gemini / NVIDIA NIM) are for **internal QA round-robins only**, never shipped output.
- VRAM ceiling: **14.5 GB audio** / **15.5 GB video** (lifted 2026-04-17 for the video stack only — audio stays at 14.5 GB).
- Audio is king (rule **C7**). Full narrative output must never break, shorten, or degrade. If video breaks audio, revert immediately. Audio output must remain byte-identical to v1.5 baseline at every gate.

---

## P0 — Active focus

### BUG-128 + BUG-129 fix sequence (locked 2026-05-01 after round-robin + external code review)

Two coupled bugs surfaced during QA of episode `signal_lost_..._20260501_110019`:
- **BUG-LOCAL-128:** `_render_master_mix_per_clip_mux_mode` truncates ~70 s of dialogue audio. Two layered causes — (1) sum(clip.dur_s)=137.52 s vs ledger.total_episode_dur_s=207.39 s gap (HuMo clip < line audio + line gaps + bookend not in lines[]); (2) tail-pad `last_idx` computed before the pillarbox loop (line 556), so when `pb_failures` prunes the genuine last clip the surviving last clip has no tail pad and `-shortest` truncates trailing audio.
- **BUG-LOCAL-129:** ANNOUNCER lines render as random women instead of the radio still. Root cause is HuMo's finetuned weights, not the conditioning code (`reference_latents` injection in `nodes_wan.py:1077` is generic Wan2.1 I2V). HuMo will not animate non-face references; the architectural premise in `_otr_speaker_role.py:12-22` ("the radio is the visual performer for non-dialogue") is wrong for HuMo as the renderer.

**Locked plan: three commits, in this order** (mux-first per external review — small blast radius, every downstream change inherits the fixed mux baseline):

1. **Commit 1 — BUG-128 mux fix** (single-file: `nodes/video_composite.py::_render_master_mix_per_clip_mux_mode`):
   - Compute `last_idx` from `pillarboxed[-1]` after the pillarbox loop (or drop `-shortest` and post-trim with ffprobe-measured durations). Either fix is a few lines.
   - Add post-mux audio packet-hash check vs procgen as a separate validator (the C7 contract is byte-identity *downstream of procgen*, not vs a WAV master — current `-c:a copy` from procgen already preserves that; the hash test just proves it).
   - No behavior change for clean runs (where `pb_failures == 0`); fix only fires when a tail clip dies.

2. **Commit 2 — BUG-129a static-segment generator** (additive: `nodes/video_composite.py` only, BatchHumoRender unchanged):
   - When `_render_master_mix_per_clip_mux_mode` finds a ledger line with no clip on disk (`clip_path.is_file() == False`), instead of silently skipping (current line 512-519), generate a deterministic static segment via ffmpeg subprocess: `-loop 1 -i radio_bookend.png -frames:v <int(round(dur_s * 25))> -r 25 -c:v libx264 -pix_fmt yuv420p -an out.mp4` with `-video_track_timescale 12800` to match HuMo's container timebase (timebase mismatch breaks concat-demuxer with `-c copy`). Use `-frames:v` not `-t` for exact frame counts.
   - Backward-compat: if all clips on disk, static path doesn't fire — output identical to current.

3. **Commit 3 — BUG-129b role policy flip** (`nodes/_otr_speaker_role.py` + `nodes/batch_humo_render.py` + LLM cast schema):
   - Add ANNOUNCER as a real cast member in the LLM story-writer schema with description "1940s radio drama host at vintage broadcast microphone, period suit, dim studio lighting." `BatchFluxRender` produces the portrait through the existing pipeline.
   - Reroute speaker_role: ANNOUNCER → existing portrait chain (BUG-088 resolver). `music_*` and standalone `sfx` → no HuMo render at all (the static-segment path from commit 2 fires instead).
   - Hard assertion in BatchHumoRender dispatch: if `speaker_role in {music_open, music_close, music_inter, sfx}` and target == HuMo, fail fast.
   - SFX disambiguation: SFX concurrent with dialogue stays on the speaking character's HuMo clip (no separate visual). Mark via `is_concurrent_with_dialogue` boolean on the ledger line; static-segment path skips concurrent SFX.
   - Verify FLUX render width/height matches HuMo workflow's width/height (announcer portrait must match HuMo's expected resolution to avoid distortion).

**Acceptance criteria (all must hold before declaring done):**
1. No HuMo render job ever receives the radio still (assertion in dispatch).
2. ANNOUNCER clips l001 and l021 in a regression episode resolve to the same announcer portrait family — no generic-blonde drift.
3. `music_*` / standalone-`sfx` segments render through the static-video path (ledger `clips[].source_kind == "static_ffmpeg"` vs `"humo"`).
4. Final mp4's extracted audio packet-hash matches procgen's audio stream byte-for-byte.
5. Peak VRAM stays below 14.5 GB.
6. Final video duration ≈ master mix duration (no `-shortest` truncation).
7. `tests/test_dropdown_guardrails.py`, `tests/test_core.py`, and the Bug Bible regression all pass.

**Post-Option-1 enrichment (v2.0-beta, not blocking):** ffmpeg audio-reactive overlay on the static radio segments — `showwaves` / `showspectrum` / `avectorscope` / `showvolume` filter passes composited as needle-meter or oscilloscope inset on the radio still. Pure ffmpeg, frame-deterministic, CPU only. Period-correct (1940s radios had needle meters). No model needed.

**Round-robin transcripts:** `docs/2026-05-01-humo-radio-architecture__01_chatgpt.md`, `__02_gemini.md`, `__03_nvidia.md`, `__04_synthesis.md`, `__06_jeffrey_review.md`. All three external models converged on Option 1; external review pushed the sequencing from 2 commits to 3 and surfaced the tail-pad pointer bug as a separate latent issue.

---

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

**Architecture (Locked 2026-04-30):** Single master JSON with bypassable video-stack groups. Shared audio chain → procgen, then multiple side-by-side render groups — each group bypassable via Ctrl+B. Final VideoComposite takes whichever group is active.

**Stance:** 8 GB tier does NOT get "full animated backgrounds" or generative character video. They get an **enhanced visual mode** optimized for their VRAM limits: still + parallax + interpolation for motion, with optional Wan 2.2 5B B-roll for users who want to gamble on render time.

**Do NOT offer:** HuMo, LTX-2, LTX-2.3, or 14B Wan to 8 GB users. The support burden and OOM risk are too high.

**Locked picks (2026-04-30, after evaluating LTX 2.3, LTX-2 19B, ERNIE Image, NVIDIA CES 2026 NVFP4, and round-robin consult on background models):**

| Component | 16 GB tier | 8 GB tier | Why |
|---|---|---|---|
| **Stills** | **NVFP4 FLUX.2** (RTX 50 Series, ~5 GB; falls back to FLUX-fp8 ~12 GB if NVFP4 unavailable) | **FLUX.1-dev Q4_K_S** (city96 GGUF, ~5-6 GB) | FLUX is the visual anchor for both tiers. NVFP4 is the new official quantization NVIDIA announced at CES 2026 — 3x faster, 60% less VRAM than fp8 on RTX 50 Series. Q4_K_S is the safe 8GB GGUF option. |
| **Motion** | **HuMo 14B fp8** + master_mix_per_clip_mux + LTXV background layer | **Still + Parallax + Interpolation** (deterministic Ken-Burns + frame interp on FLUX stills) | HuMo for 16 GB character lip-sync. 8 GB gets safest, fastest, most deterministic motion — high quality, zero VRAM spikes, no diffusion-per-beat. |
| **Optional B-roll** | n/a (HuMo covers all character beats; LTXV covers backgrounds) | **Wan 2.2 5B TI2V** (native ComfyUI template, optional toggle) | Strictly optional B-roll lane for 8 GB users who want generative motion on non-dialogue beats. Slow, not guaranteed; document expectation upfront. |
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

### Animated backgrounds (3-layer composite, 16 GB only)

Promotes the current 2-layer composite (procgen-base + HuMo-overlay, BUG-092) into a 3-layer composite. **8 GB tier does NOT get a background layer** (procgen sides only — keeps 8 GB lean).

```
TOP:    Procgen / CRT audio-reactive overlay -- `lighten` blend, ~0.3 opacity
MID:    HuMo lip-sync portrait -- center pillarbox during dialogue, opaque
BOTTOM: Animated background (model TBD) -- full canvas, opaque
```

**Why CRT-on-top in lighten mode is more truthful:** a failing broadcast's scanlines + audio-peak flicker should cover the WHOLE frame including the speaker's face — the interference doesn't politely stop at the pillarbox edges. Lighten mode takes max(CRT, underlying) per channel so artifacts ride on top without erasing detail.

**Render budget (locked 2026-04-29 PM — render-native + slow-mo, model-agnostic):**
- Render at the chosen model's native fps, then slow to 12 fps via ffmpeg `setpts=PTS*2,fps=12`. The slow-mo IS the SIGNAL LOST broadcast-degraded aesthetic.
- 1-2 clips per SCENE (not per shot). Loop across the scene's duration via `-stream_loop -1` with optional crossfade or ping-pong reverse.
- For LTX: 193 frames per clip = 8 sec native = 16 sec apparent after 2× slow-mo. LTX uses 8× temporal VAE compression so frame counts must be `8n + 1`. 193 = 24*8 + 1. Max 257.
- For Wan: frame-count math TBD per model card during implementation.
- Distilled 4-8 steps (default 6 for LTX; Wan TBD).

**Per-episode wall-clock estimate:** smoke (1 scene) ~50 s; short (3 scenes) ~2.5 min; medium (5 scenes) ~4 min. Negligible vs HuMo (~10 min per dialogue line).

**Frame-count widget shape (model-specific names locked at impl):**
```
frames:         dropdown of valid frame counts for chosen model
steps:          distilled step dropdown
slow_mo_factor: float (default 2.0)
target_fps:    int (default 12)
```

#### Background-model selection — LOCKED 2026-04-30

**Round-robin verdict:** Keep the background layer cheap, stable, and visually appropriate for being blurred/degraded under the HuMo dialogue pillarbox. Foundation-model chasing for a layer that gets slowed to 12 fps and composited under a foreground is the wrong engineering bet.

| Candidate | Size on disk | Peak VRAM | Role | Verdict |
|---|---|---|---|---|
| **LTXV 0.9.x 2B distilled fp16** | ~5 GB | ~7-8 GB w/ VAE | **Default (16 GB)** | **LOCK.** Fits the degraded-broadcast aesthetic perfectly. 193 frames (8n+1), 4-8 distilled steps, then ffmpeg slow-mo to 12 fps. Both ChatGPT + Gemini endorsed. |
| **Still + Parallax + Interpolation** | ~5-6 GB (FLUX still only) | ~7 GB | **Default (8 GB)** | **PLAN B / 8 GB PATH.** Lowest risk, highly deterministic Ken-Burns + frame interp on FLUX stills. Likely enough motion for radio drama without diffusion overhead. ChatGPT's smallest-change biggest-payoff suggestion. |
| **Wan 2.2 5B native FP8** | ~6 GB | ~8-9 GB w/ VAE | Fallback | Keep as a fallback if LTXV introduces unacceptable motion artifacts during live-test. Also serves 8 GB tier as optional B-roll lane. |
| **LTX-2 19B / 2.3 22B GGUF** | 12-14 GB | 14-17 GB w/ VAE decode spike | **REJECTED** | **DO NOT USE FOR BACKGROUNDS.** Audio-video foundation models are a paradigm mismatch and too heavy for a sidecar background layer on a 16 GB VRAM ceiling. VAE temporal decode adds 2-3 GB at decode → OOM. ChatGPT also flagged "1.1" version label as community packaging, not a confirmed upstream tag. |
| **HunyuanVideo distilled** | varies | varies | Not recommended | ChatGPT mentions; operationally heavier than LTXV. Skip. |
| **Stable Video 3 (8B)** | unknown | unknown | Suspect | NVIDIA round suggested with hallucinated specifics; do not pursue without independent verification. |

**Quantization gotchas on Blackwell sm_120 (both ChatGPT + Gemini):** Don't depend on FP8 / NVFP4 paths for video models yet — Blackwell support arrives in layers (PyTorch → CUDA kernels → custom ops → quant backends → custom nodes), and ComfyUI custom video nodes are exactly where "advertised support" and "production-safe support" diverge. Prefer fp16 / bf16 paths that already work.

**Pin format locked:**
```yaml
background_video:
  family: "ltxv"
  upstream_repo: "Lightricks/LTX-Video"
  model_file: "<exact 0.9.x safetensors filename to confirm at impl>"
  upstream_commit: "<HF commit SHA at impl>"
  comfyui_node_repo: "<exact custom node repo>"
  comfyui_node_commit: "<SHA at impl>"
  precision: "fp16"   # prefer over fp8 for stability on this layer
  frames_rule: "8n+1"
  target_frames: 193
  sampler_steps: 6
  postprocess: "setpts=PTS*2,fps=12"
```

#### TTS palette expansion — LOCKED LADDER 2026-04-30

NOT replacing the canonical pipeline (Bark + Kokoro + MusicGen + AudioGen → master mix). EXPANDING the per-character voice palette. Round-robin consult 2026-04-30 produced strong agreement on direction.

**Production add-order ladder (Parler-TTS REJECTED — owner pref; vintage sound stays in the deterministic DSP chain):**

| Priority | Engine | License | Peak VRAM | C7-deterministic? | Verdict |
|---|---|---|---|---|---|
| **1** | **Kokoro** (current) | MIT | ~1 GB | Yes | **KEEP.** Undisputed workhorse for strict lip-sync and clean narration. Gemini calls "undisputed king of low-VRAM deterministic phoneme TTS." |
| **2** | **Bark** (current) | MIT | ~6 GB | Yes (vram_sentinel + length-sort batching shipped) | **KEEP.** Unmatched for period vibe, character texture, and emotional color. |
| **3** | **CosyVoice 2** | Apache-2.0 | ~3-4 GB | Yes (flow-matching ODE solver + fixed seed = byte-identical) | **ADD NEXT.** Strongest production candidate for expanding the dramatic voice palette. Both ChatGPT + Gemini endorsed. |
| **4** | **Piper** | MIT | ~1 GB | Yes | **8 GB / UTILITY FALLBACK.** Tiny, deterministic, fast. Ideal for minor announcer roles or 8 GB emergency fallback. ChatGPT's recommendation for utility voices. |
| **5** | **CosyVoice 3** | Apache-2.0 | unknown | Unverified | **RESEARCH LANE.** Both flag as too new for production. Needs strict C7 hash proof before promotion. NVIDIA round claimed v3.2.1 production-ready with hallucinated commit SHA; ignore that signal. |
| **6** | **Qwen3-TTS** | needs license audit | unknown | **C7 RISK** | **RESEARCH LANE.** Gemini flags autoregressive + flow-matching hybrid as hard to make byte-identical. Highly expressive but requires deep C7 verification before any merge. |

**REJECTED candidates:**
- **Parler-TTS Mini** — owner preference; vintage broadcast sound stays in the deterministic DSP mastering chain (band-limit + tube saturation + plate flavor + noise floor + AM EQ). Pinned exactly there, model-side stylization rejected.
- **Fish Speech** — license incompatible with MIT downstream.
- **XTTS / Tortoise / StyleTTS family** — license ambiguity, Windows friction, C7 determinism risk. Evaluate only if a specific gap appears that priorities 1-4 don't fill.

**C7 qualification protocol (apply to any new TTS before merge):**
1. Same prompt + same seed + same model revision + same driver/torch/CUDA/cuDNN + same batch size + same output format.
2. Run 10 repeated generations across cold start, warm start, and process restarts.
3. Hash final WAV bytes. If any hashes differ → engine is NOT qualified for OTR.

**Period-style controls — locked position:** Vintage broadcast sound lives in the deterministic DSP mastering chain (band-limit, tube saturation, plate flavor, noise floor, AM EQ shaping). TTS engines provide diction / cadence / timbre baseline only. Any model offering "1940s radio" as a text-prompted style is out of scope — we own the vintage sound, the model doesn't get to drift it.

**Pin format to lock once each engine ships:**
```yaml
tts_palette:
  engines:
    - name: "kokoro" / "bark" / "cosyvoice2" / "piper"
      upstream_repo: "<exact repo>"
      model_revision: "<tag/SHA>"
      tokenizer_revision: "<tag/SHA>"
      vocoder_revision: "<tag/SHA>"
      decode_mode: "<greedy|ode_solver|other>"
      sample_rate: "<Hz>"
      wav_hash_test: true
      role: "<character|announcer|narrator|utility>"
```

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
- `docs/2026-04-30-project-qa__04_synthesis.md` — earlier round-robin QA pass (OpenAI + Gemini + NVIDIA — NVIDIA was a non-vote due to context overflow)
- `docs/2026-04-30-ltx-tts-april2026__01_chatgpt.md` — ChatGPT (gpt-5.4) round on background-model + TTS-palette decisions
- `docs/2026-04-30-ltx-tts-april2026-gemini__02_gemini.md` — Gemini (gemini-3-pro-preview) follow-up
- `docs/2026-04-30-ltx-tts-april2026-nvidia__03_nvidia.md` — NVIDIA (mistral-nemotron) round; **discounted** for hallucinated specifics (fabricated `voxpopuli/tts-v2.1`, fake CosyVoice commit SHA, line numbers that don't match codebase)
- **Pending:** next round-robin pass on the v2.0-beta `Background-model selection` and `TTS palette expansion` matrices in this file, using the gpt-5.5 ladder (added to `scripts/_consult_round_robin.py` 2026-04-30)
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
