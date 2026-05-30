# OTR -- Burned-In SDH Captions -- Go-Forward (lean)

**Branch:** `v2.0-alpha` | **Base:** `bfc761a` | **Burn point:** Node 58 `OTR_PostUpscaleProcgenBlend`

Open captions (SDH) burned into the final 1920x1080 deliverable, synced to dialogue,
color-blind-safe and constantly legible. Audio never touched (`-c:a copy`).

## Locked design

- **Method:** generate `.ass` (libass) from the ledger `lines[]` (`text`, `start_s`,
  `dur_s`, `speaker_role`, `char_id`); burn inside the blend node's existing `filter_complex`
  (same encode, no extra pass). ffmpeg build has `--enable-libass` (verified).
- **Style `sdh_standard` (default):** Arial 52 px, **white** dialogue, opaque black box
  (`BorderStyle=3`, **Outline=5** -- must be >0 or libass draws no box; BackColour ~75%),
  bottom-center, `MarginV=70`, max 2 lines.
- **Speaker label = color-blind-safe:** bold **white** `NAME:` (the label text + weight IS
  the cue). Color is a **subtle pastel outline only** (`\b1\bord1\3c<pastel>`), never the fill,
  never the differentiator. Dialogue + name fill stay white on the box at all times.
- **`otr_crt` style:** themed green variant, A/B option only -- NOT default.
- **SDH rules:** <=2 lines, <=37 ch/line, target <=17 CPS (cap 20), min 1.0 s, no overlap
  (later cue clamps earlier end); long lines split into multiple timed cues.
- **Sound/music cues:** sparse, bracketed (`[STATIC HISS]`, `♪ ... ♪`) -- only when they matter.

## Master vs delivery

- **Clean master: captions OFF** (default). **Accessible delivery: `sdh_standard` ON.**
- Toggle is opt-in; the clean render is always preserved.

## Sprints (review -> wired -> regress -> commit, one per commit)

- **P0 -- offline ASS builder** (`scripts/otr_captions.py`): build + lint + unit test. QA'd
  on Generator's Grasp (announcer + character + white-hazmat stress frame). **[in progress]**
- **P1 -- env-gated burn:** `OTR_BURN_CAPTIONS=1` appends `ass=` to the blend filter; zero
  widget drift; audio stays `-c:a copy`.
- **P2 -- placement QA:** MarginV 70/90/110 across busy-HUD / dark / red-alert / dialogue /
  SFX-only frames.
- **P3 -- audio regression:** captioned vs non-captioned final audio stream MD5 identical.
- **P4 -- promote to widgets:** `burn_captions` + `caption_style` on Node 58; update
  `OTR_WorkflowValidator` expected vector; validator POST drift=0; wire JSON defaults (OFF).

## Gates (every sprint)

Bug Bible + core + `test_audio_byte_identical.py` after each change; validator POST after any
workflow JSON edit; real ComfyUI smoke after any ffmpeg-arg change; log bugs to `BUG_LOG.md`.
