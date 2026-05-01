# OTR v2.0-alpha — Targeted code review of tonight's Step 3-6 work

**Branch:** `v2.0-alpha`
**Hardware/platform:** RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA. No Flash Attention. 100% local, no cloud.
**Audio rule (C7, non-negotiable):** Audio output must remain byte-identical to v1.5 baseline. Cloud LLMs (you) are for INTERNAL QA only — never shipped output.

---

## Context: what's been built and why

Project is a ComfyUI radio-drama generator ("SIGNAL LOST"). End-to-end pipeline:
LLM script writer → LLM director → Bark/Kokoro/MusicGen/AudioGen audio → SceneSequencer (scene mix) → EpisodeAssembler (master mix with opening/closing themes) → SignalLostVideo (procgen video + master mix audio in one mp4) → FLUX still renders → HuMo lip-sync per dialogue line → VideoComposite (final mp4).

**Today's architectural lock (2026-04-30):** every audio second of every episode is a HuMo-rendered video clip. The radio is the visual "performer" for non-dialogue lines (announcer, music, SFX) — its FLUX-rendered sci-fi radio still feeds HuMo as the I2V reference, and HuMo's per-line audio drives lip-sync motion (speaker grille pulse, vacuum tube flicker). Glitches on non-face references are pre-approved aesthetic (broadcast distress = SIGNAL LOST signature).

Audio quality goal: zero re-encodes downstream of SignalLostVideo. The previous `humo_concat` path re-encoded audio to AAC 192k three times (gap fill + per-clip normalize + final concat), breaking C7 byte-identity. The new path muxes the master mix audio onto silent HuMo video with `-c:a copy` exactly once.

**Six commits tonight, in order:**

1. **`1e1b6be` Step 3** — `visual/batch_flux_render.py`: dynamic radio FLUX prompt builder. Reads `meta.gen_params_initial.genre_flavor + style_variant` (phase-0 ledger init) and synthesizes a per-episode radio aesthetic from 10 genre presets × 7 style mood layers + universal SIGNAL LOST suffix. Empty widget = dynamic mode (default). Belt-and-suspenders ledger stamp under both `radio_bookend_path` (top-level) and `meta.radio_bookend_path`.

2. **`088177c` Steps 4 + 5** — new `nodes/_otr_speaker_role.py` taxonomy module (character / announcer / music_open / music_close / music_inter / sfx + helpers); `nodes/batch_humo_render.py` branches I2V reference resolution by `speaker_role`. Radio roles use `ledger.radio_bookend_path` (existence-checked); character falls through the existing BUG-088 portrait chain. New `_resolve_radio_still_path(ledger)` reads top-level OR `meta` location, returns `None` on any ambiguity so radio-role lines fall through to portrait gracefully.

3. **`8441a57` Step 4b** — `nodes/scene_sequencer.py`: stamps `speaker_role` on existing dialogue lines (announcer if character_name=ANNOUNCER, else character) and mirrors every SFX cue from `ledger.sfx[]` into `ledger.lines[]` with `speaker_role=sfx`. Idempotent across resume reruns. `ledger.sfx[]` kept populated for back-compat consumers.

4. **`95aa39e` Step 4c** — `nodes/scene_sequencer.py` (EpisodeAssembler half): mirrors every populated `ledger.music[]` cue into `ledger.lines[]` with `speaker_role=music_open` / `music_close` / `music_inter`. Stamped in master_mix space (post BUG-LOCAL-106 shift). Idempotent.

5. **`9c55773` Step 6** — `nodes/video_composite.py`: new `audio_source="master_mix_per_clip_mux"` (now default). Three-pass pipeline:
   - Pillarbox each HuMo clip silent (`libx264 + -an`)
   - Concat-demux the silent pillarboxed clips with `-c copy`
   - Single mux: silent_combined video + procgen master mix audio with `-c:v copy -c:a copy -shortest`

   Net audio re-encodes downstream of SignalLostVideo: **0** (was 3 in `humo_concat`). One libx264 video re-encode per clip is unavoidable (HuMo emits 480x832 / 640x640 portraits; canvas is 1920x1080).

   Three-tier fallback chain: `master_mix_per_clip_mux` → `humo_concat` → `master_mix`. A single failure mode never breaks a run.

286 unit tests + Bug Bible regression all green. AST clean.

---

## Specific things I want your independent opinion on

You're each independently reviewing the architecture. I'll synthesize across the three of you and discount anything that looks hallucinated or factually wrong.

1. **Is the per-clip-mux audio path actually byte-perfect end-to-end?** I'm trusting that ffmpeg `-c:a copy` on the procgen mp4 → final mp4 doesn't introduce ANY audio drift. Concat-demuxer with `-c copy` is supposed to passthrough but may produce frame-boundary glitches at clip seams. Is there a failure mode I'm missing? Should I be concerned about AAC frame alignment when the silent_combined.mp4 is muxed with the procgen audio (a single audio stream covering the whole episode, not per-clip slices)?

2. **HuMo on a non-face reference.** Wall-to-wall coverage means HuMo gets the radio still + music audio for `music_open` / `music_close` lines, and the radio still + SFX audio for `sfx` lines. HuMo was trained on human faces + speech. Is there a known failure mode where Whisper's audio encoder produces empty/garbled features for music/SFX, leading HuMo to produce static frames or visual artifacts that aren't even glitchy in an interesting way? (Owner accepts "glitchy" as the SIGNAL LOST aesthetic — but is there a risk of "static" rather than "glitchy"?)

3. **Idempotency of the SFX/music mirroring.** Step 4b/4c append entries to `ledger.lines[]` based on existing entries in `ledger.sfx[]` / `ledger.music[]`. Idempotency check is `line_id in existing_ids`. Is there a race condition I'm missing? What if `ledger.lines[]` is mutated between SceneSequencer and EpisodeAssembler in a way that breaks the resume contract?

4. **Three-tier fallback chain hides bugs.** `master_mix_per_clip_mux` → `humo_concat` → `master_mix`. Each tier silently activates on the previous tier's failure, with only a `log.warning`. Should I make the fallback LOUDER (raise to a stamped ledger field, or report-line with `WARNING:` prefix) so a regression doesn't go unnoticed for episodes?

5. **Radio prompt aesthetic mappings.** I have 10 genre presets (sci-fi, noir, horror, post-apocalyptic, cosmic-horror, western, thriller, cyberpunk, fantasy, comedy) and 7 style mood layers (atmospheric, kinetic, minimalist, baroque, documentary, dreamlike, gritty). Are any of these likely to produce FLUX-pathological prompts (e.g. tokens that consistently misfire on the FLUX-1-dev model)? Anything obvious I should add or remove?

6. **What did I miss?** Look at the 6 commits as a coherent change set. Anything load-bearing that's NOT covered by tests? Any contract that's implicit but not enforced?

---

## Constraints to respect

- 16 GB VRAM ceiling (14.5 GB target). No weight streaming, no Flash Attention chasing, no quantization heroics.
- Don't recommend modifying VRAM management code.
- Don't recommend adding cloud dependencies.
- Don't recommend re-encoding audio anywhere in the per_clip_mux path.
- Owner explicitly accepts HuMo OOD glitches on radio renders — they're the aesthetic.
- Be candid; flag uncertainty rather than bluffing.

---

## Sanity-check anchors (so you don't hallucinate API surfaces)

- ffmpeg invocations are real subprocess.run calls with real argv lists.
- ComfyUI ledger is JSON on disk under `output/otr/audio/<episode_id>_ledger.json`.
- HuMo = `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` (Kijai), loaded via stock `UNETLoader`.
- FLUX = `flux-dev-fp8` via `CheckpointLoaderSimple`.
- Whisper for HuMo audio conditioning = `whisper_large_v3_fp16.safetensors`.
- All paths use Windows-style absolute strings in this codebase.

If you cite a function name or file path, I will grep for it. False citations get caught.
