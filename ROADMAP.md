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

### Locked architecture (2026-04-30): 100% HuMo video coverage + bit-perfect master audio mux

**Decision lock:** every audio second of every episode is a HuMo-rendered video clip. Master audio mix is sliced at sample-accurate per-clip boundaries and muxed onto each HuMo clip with `-c:v copy -c:a copy` (zero re-encode). Final concat is `-c copy` (zero re-encode). Audio passes through ONE ffmpeg mux per clip, never re-encoded.

**Why this lock:**
- Confirmed 2026-04-30 audit of `nodes/video_composite.py`: the current `humo_concat` path re-encodes audio to AAC 192k **three times** (gap-filler segment encode line 318, per-clip normalize encode line 350, final concat encode line 522), plus once more upstream at HuMo's `CreateVideo` step. Audio you hear has been through 3-4 lossy encodes. C7 byte-identity is **broken** on this path.
- Gemini's 2026-04-30 round-robin catch was correct.
- The lossless path requires master audio to be the canonical source, not HuMo's output audio.

**The flow:**

```
1. Bark / Kokoro / MusicGen / AudioGen        → per-line audio stems
2. SceneSequencer + AudioEnhance + EpisodeAssembler
                                                → master mix WAV
                                                + per-line (start_s, dur_s) populated in ledger.lines[]
3. BatchHumoRender per line:
   ─ snap dur_s to nearest 4n+1 frames via humo_length_for_dur()
   ─ I2V reference image picked by speaker_role:
       character  → PASS3 cast portrait
       announcer  → sci-fi radio FLUX still
       music_*    → sci-fi radio FLUX still
       sfx        → sci-fi radio FLUX still
   ─ feed HuMo: ref_image + line_audio (Whisper conditions video motion)
   ─ KEEP video frames, DISCARD HuMo's CreateVideo audio output
   ─ stamp ledger.lines[].humo_dur_s = (frames / 25)
4. AudioSlicer (new node):
   ─ for each line, slice master mix at [start_s, start_s + humo_dur_s]
   ─ ffmpeg -ss <start> -t <humo_dur> -c:a copy
   ─ sample-accurate, no re-encode
5. VideoComposite per-clip mux:
   ─ ffmpeg -i humo_silent.mp4 -i master_slice.wav -c:v copy -c:a copy clip.mp4
   ─ one mux per clip, no re-encode
6. Concat all clips:
   ─ ffmpeg -f concat -i clips.txt -c copy out.mp4
   ─ no re-encode
```

**Why drift is zero by construction:**
- HuMo clip duration = `(4n + 1) / 25` seconds, exact.
- Audio slice duration = exactly that value, sample-accurate cut.
- All mux steps are `-c copy` so no resample, no codec round-trip.
- BUG-102 warmup-pad already trims HuMo's freeze-frame artifact before save, so HuMo video starts on the first phoneme — aligns naturally with the master audio slice.

**Why this beats every other architecture:**
- C7 byte-identity preserved — master mix is the canonical audio, never re-encoded.
- HuMo never has to carry OOD music or SFX audio — Whisper still runs internally for motion conditioning, but the audio output we keep is the master slice (clean Kokoro / MusicGen / AudioGen).
- Wall-to-wall video coverage — every audio second has a HuMo silent clip glued under it.
- Whisper OOD doesn't matter — even if HuMo produces glitchy motion on music/SFX, that **is** the SIGNAL LOST aesthetic (broadcast distress / failing radio) and the audio you hear is master-mix clean.
- Lip-sync drift = zero by construction.
- 100% audio coverage, no missing audio, no lip-sync drift.

**The radio is the performer, not a static prop.** The sci-fi radio FLUX still serves as the I2V reference for any non-dialogue HuMo clip. The speaker grille pulses to the announcer voice, the vacuum tubes flicker on SFX hits, the dial bounces with music peaks — that's the visual spine of the OTR identity.

#### Build order (sequenced — each step gates the next)

1. **SceneSequencer per-line timing population.** Populate `ledger.lines[].start_s` + `dur_s` from the assembled audio timeline. This is the gating blocker — no slice boundaries, no per-clip mux. Also tagged Task #20 in earlier roadmap.
2. **AudioSlicer node.** New node `OTR_AudioSlicer`. Reads ledger, slices master mix at `(start_s, start_s + humo_dur_s)` per line with sample accuracy. Writes `output/otr/audio_slices/<ep_id>/line_NNN.wav`. ffmpeg `-c:a copy` (no re-encode). Stamps `ledger.lines[].audio_slice_path`.
3. **Sci-fi radio FLUX still.** Add a widget on `OTR_BatchFluxRender` (or sibling node) to render one sci-fi radio still per episode. Saved to `output/otr/stills/radio_<ep_id>.png`. Path written to `ledger.radio_still_path`. Aesthetic prompt baseline (refine to taste): *"sci-fi retrofuturistic radio broadcast unit, glowing CRT frequency display, copper vacuum tubes haloed in plasma, brushed steel chassis with art-deco engraving, dim amber + cyan rim lighting, dust-mote atmosphere, 35mm film grain, broadcast-distressed cinematic aesthetic, 1080p"*.
4. **`speaker_role` field on every ledger line.** ScriptParser + SceneSequencer stamp `speaker_role ∈ {character, announcer, music_open, music_close, music_inter, sfx}` on every `ledger.lines[]` entry. Existing dialogue lines default to `character`. Announcer windows (Kokoro output), music windows (MusicGen open/close/interstitial), and standalone SFX cues each get their own role tag.
5. **`BatchHumoRender` reference-image swap.** Branch the I2V reference lookup by `speaker_role`: character → existing PASS3 portrait resolver; announcer/music/sfx → `ledger.radio_still_path`. Keep all existing BUG-088 / BUG-118 fallback chains intact for the character branch. New regression test: render one of each role end-to-end with deterministic seeds.
6. **`VideoComposite` per-clip-mux mode.** New `audio_source = "master_mix_per_clip_mux"` (becomes default). Path:
   - For each line in ledger, mux `humo_silent.mp4` + `audio_slice_path` with `-c:v copy -c:a copy`.
   - Concat all per-clip mp4s with `-f concat -c copy`.
   - No filter graph, no re-encode, no procgen overlay at this stage (procgen lighten layer ships as a v2.0-beta enhancement).
7. **End-to-end smoke episode.** Run a smoke preset (1 act, target_words=100) against the full new pipeline. Acceptance below.

#### Acceptance

- `ledger.lines[].speaker_role` populated for 100% of lines (no nulls).
- Every line has a HuMo-rendered video clip on disk. No gaps in coverage.
- Every line has an `audio_slice_path` on disk, duration exactly matching its HuMo clip duration to sample accuracy.
- Per-clip mux + final concat use `-c copy` exclusively. ffprobe confirms identical AAC frame boundaries vs master mix WAV (or PCM passthrough if container allows).
- `ledger.lines[].speaker_role = announcer / music_* / sfx` lines render with the sci-fi radio still as I2V reference (logs confirm `ref_image_source = radio`).
- Final episode mp4 audio bit-compares against master mix WAV after demux + decode (allowing for one canonical AAC encode if container requires it).
- ffprobe on the final mp4: video stream codec_type=video AND audio stream codec_type=audio AND duration matches `sum(humo_dur_s)` to <1 frame.

#### Hardware floor (locked 2026-04-25, do not relitigate)

- HuMo 14B fp8 e4m3fn scaled (Kijai) — `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`. Stock `UNETLoader`. Tuned by Kijai for 16 GB cards.
- Fallback ladder (kept on disk, do NOT delete): `humo_17B_fp8_e4m3fn.safetensors` (highest quality, slower ~6 min/clip), `Wan2_1-HuMo-17B_Q5_K_M.gguf` (speed-tuned, fast iteration).
- Stable shape: `length=97` (3.88 s @ 25 fps), 480x832, batch=1. Or `length=177` at 640x640 (7 s, OOD but verified working).
- Frame count must be `4n + 1`. Wan 2.1 VAE temporal compression. Helper `humo_length_for_dur(dur_s)` snaps any duration up.
- Per-step: 42 s. Per-clip: ~4:30 native, ~6:15 in TEST_humo. Cold load: ~50 s.

#### HuMo coverage goals (sequenced relative to the architecture above)

| Goal | Definition of done | Maps to step |
|---|---|---|
| 1 — Every line gets a HuMo clip | One HuMo MP4 per `ledger.lines[]` entry, regardless of speaker_role | Steps 1-5 above |
| 2 — Per-clip mux + concat lossless | All audio passes through with `-c copy` only | Step 6 |
| 3 — Continuity layer (drift mitigation) | Hybrid blending across HuMo windows for >7s narrative beats | Post-Step 7 (v2.0-beta candidate) |

Goal 3 still gates "production unattended" runs but is now decoupled from "audio is correct" — once Steps 1-7 ship, the audio path is locked even if visual continuity needs more work.

#### Other items still relevant (deferred until Steps 1-7 ship)

- **Per-scene environment FLUX still + LTX/zoompan animated background.** v2.0-beta enhancement once the audio architecture is locked. Adds the bottom-layer animated background under the HuMo center pillarbox in middle (non-bookend) windows.
- **Procgen-CRT lighten layer over the whole frame.** v2.0-beta. Audio-reactive scanlines + flicker on TOP of the HuMo + LTX composite. Sells the "failing broadcast" identity. Was the old `signal_lost_<id>.mp4` proc gen layer.
- **Drifted-filename smoke for BUG-LOCAL-118.** Force an underscore-drifted .mp4 stem and run BatchHumoRender to verify the fallback chain fires before relying on it in a long soak.
- **Reconcile `16294df` ROADMAP-vs-git-log mismatch.** Git log says "BUG-LOCAL-112 news-history reset" (same as `61a85b3`); prior ROADMAP narrative had it as "Wire ScriptCritic" — likely a rebase/amend artifact.

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

**Decision deferred** until first clean 16 GB FULL run lands so we have real end-to-end data before designing the 8GB fallback. File post-decision in `docs/2026-04-28-8gb-strategy-decision.md`.

**Leaning option (Jeffrey 2026-04-28):** Single master JSON with bypassable video-stack groups. Shared audio chain → procgen, then multiple side-by-side render groups (FLUX-fp8 stills, HuMo 14B, LTX-Video 0.9, CogVideoX-2B, SDXL+AnimateDiff, etc.) — each group bypassable via Ctrl+B. Final VideoComposite takes whichever group is active. 8 GB users bypass HuMo+FLUX, enable a lightweight pair.

**Acceptance for 8GB path:**
- Full audio pipeline (LLM + Bark + AudioGen + MusicGen + SceneSequencer + EpisodeAssembler) — same as 16 GB.
- SignalLostVideo procgen base — same.
- **Visual layer must be MOTION VIDEO, not stills.** Jeffrey 2026-04-28: *"i just don't want stills if we can find a 8gb vid model"*.
- Final mp4 lands in `output/episodes_for_obs/<ep>/<ep>.mp4` same as 16 GB.

**Model research before designing the 8GB workflow:**
- Image model under 8 GB: SDXL-Turbo, SD 3 Medium, FLUX-schnell at lower precision, SD 1.5 (last resort).
- Video model under 8 GB: HuMo's smaller siblings if any, Wan 2.1 1.3B, LTX-Video 0.9 (~5 GB), CogVideoX-2B fp8, AnimateDiff with lightweight base. Lip-sync ideal but not required.
- Pairing: image + video should share tokenizer/conditioning if possible to avoid double prompt-engineering surface.

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
