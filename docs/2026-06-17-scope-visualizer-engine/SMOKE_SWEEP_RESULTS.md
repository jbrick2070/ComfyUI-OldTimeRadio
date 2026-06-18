# Verified-model 120w smoke sweep -- results (Task 2)

Status as of 2026-06-18 (autonomous overnight run). **Task 1 (visualizer engine) is
SHIPPED + unit-validated + pushed (`236db0e`).** Task 2's full-episode GPU sweep is
**BLOCKED on a headless-audio env gap** -- documented below with the verified
enumeration and the unblock path.

## Verified sets (enumerated PROGRAMMATICALLY, not hardcoded)
- **VIDEO** `registry.validated_engine_names()` (8): `humo`, `humo_1.7B`,
  `humo_1.7B_169`, `humo_14B_169`, `ltx_av_music`, `ltx_av_talk`, `ltx_video`,
  `wan_ti2v`.
- **IMAGE** `OTR_ImageDirector` validated (2): `flux_gen1`, `z_image_turbo`.
- **`visualizer`**: registered, NOT yet in `VALIDATED_ENGINES` (correctly gated --
  awaits a green full-episode E2E, see below).

## What ran
| leg | result | notes |
|---|---|---|
| visualizer (all 3 roles, forced, 30w) | **BLOCKED at audio** | episode aborted in the AUDIO phase before the video phase -- `RuntimeError: IndexTTS2 Path B not installed` (the default char voice's isolated venv is absent on the headless box). The visualizer engine was never reached. Prompt "executed" in 184s but errored. |
| visualizer render path (unit, CPU) | **PASS** | 17 tests incl. a REAL ffmpeg render of a silent 16:9 mp4 of the expected frame count, has_audio=False, determinism. The engine's render_clip + scope_draw + ffmpeg encode are proven; only the full-episode (audio+mux+byte-identical) is unproven. |

## THE BLOCKER (affects EVERY full-episode sweep leg)
The headless ComfyUI install (`C:\Users\jeffr\ComfyUI-Installs\...`) does **not** have
the indextts2 Path-B sidecar venv installed, and indextts2 is the DEFAULT char voice.
`scripts/queue_smoke.py` renders the workflow as-saved (char_voice=indextts2) -> the
audio phase fails LOUD before any video engine runs. This blocks all 8 video legs +
the 2 image legs + the visualizer leg equally (they all run the full pipeline).
`OTR_SOAK_CHAR_VOICE=bark` is read ONLY by the soak harness (`scripts/_otr_*_soak.py`),
NOT by `queue_smoke.py` -- which is why the earlier wan_ti2v SINGLE-engine smoke
(`OTR_VideoRenderBatch mode=single`, no audio phase) succeeded but a full episode does not.

## UNBLOCK PATH (one of)
1. **Install indextts2 headless**: `scripts\_otr_indextts2_install.ps1` (isolated venv
   + weights) on the ComfyUI-Installs box, then run the sweep via `queue_smoke.py`.
2. **Run the sweep via the SOAK HARNESS** (`scripts/_otr_combo_soak.py` family), which
   forces `OTR_SOAK_CHAR_VOICE=bark` (in-process, dep-free) so the audio phase works
   headless -- the sanctioned path the prior all-LTX / combo soaks used. Parameterize
   it over the 8 video + 2 image validated sets + one `visualizer`-all-roles leg
   (force-map `*=visualizer`, `OTR_ENABLE_VISUALIZER=1`), 120w, random OS-entropy seed,
   reset before EACH (CLAUDE.md sec 4), capture engine/pass/time/VRAM/byte-identical.

## Visualizer promotion gate (NOT yet met)
Per the plan, add `"visualizer"` to `registry.VALIDATED_ENGINES` only after a GREEN
full-episode E2E through `OTR_VideoRenderBatch` + mux with `test_audio_byte_identical`
green. That needs the audio phase working (unblock path above). The engine's own render
path is proven (unit), so this is purely the episode-audio env gap, not a visualizer
defect.

## Box state
Reset clean (port 8000 free, GPU baseline) after the attempt.
