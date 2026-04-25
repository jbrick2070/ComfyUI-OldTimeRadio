# TEST workflow rewired to read from ledger (design doc)

**Date:** 2026-04-24 (sketched while Jeffrey was at yoga; needs review before
implementation)

## Jeffrey's brief

> "test needs to use canned audio, script and batch the images from the
>  ledger script and then yessss enter the video pipeline!"

The TEST workflow today (`workflows/otr_videoplan_TEST.json`) reads a baked
director_json from a widget. That works but tightly couples TEST to one
hand-crafted JSON snapshot. The new vision: TEST reads the **production
ledger** of a previous successful FULL run as its source of truth.

This doc captures the design without yet building it, so we can review
together before code lands.

## Goal

A TEST workflow run should be able to:

1. **Load** a `signal_lost_*_ledger.json` from a previous FULL run
2. **Re-render** all PASS3 images via FLUX (test FLUX changes / FLUX.2 swap
   without paying for ScriptWriter+Director+Bark+MusicGen+ffmpeg)
3. **Reuse** the canned per-line Bark WAVs (already written in the ledger's
   `lines[].bark_wav_path` field once L2 lands) for HuMo Stage 2
4. **Enter the video pipeline** — HuMo character animation per shot, with
   the WAV slice from the ledger's `lines[].start_s` + `lines[].dur_s`
   driving lip-sync

## Architecture

### New node: `OTR_LoadLedger`

```
INPUT:
  ledger_path : STRING  (file path to signal_lost_*_ledger.json)

OUTPUT (matches OTR_VideoPlan exactly so it's a drop-in swap):
  pass1_char_prompts_json    : STRING  (cast portraits)
  pass2_scene_prompts_json   : STRING  (scene environments)
  pass3_compose_prompts_json : STRING  (per-shot composites with character + scene)
  pass3_prompt_count         : INT
  debug_summary              : STRING

ALSO OUTPUT (new sockets, used downstream by HuMo Stage 2):
  cast_json                  : STRING  (full cast table from ledger)
  audio_timeline_json        : STRING  (lines[] with bark_wav_path + start_s + dur_s)
  final_audio_path           : STRING  (the WAV/MP4 path to slice for HuMo)
```

The pass1/2/3 outputs are constructed FROM the ledger's `cast` + `scenes` +
`shots` tables, in the exact JSON shape OTR_VideoPlan produces today.
BatchFluxRender consumes them unchanged.

### Updated TEST workflow

```
OTR_LoadLedger
   ├─→ pass3_compose_prompts_json ──→ OTR_ShotDurationCalculator ──→ OTR_BatchFluxRender ──→ SaveImage
   ├─→ audio_timeline_json ─────────┐
   └─→ final_audio_path ────────────┴─→ (future) OTR_HuMoBatch ──→ ffmpeg overlay ──→ MP4
```

The image branch is identical to today's TEST. The audio/HuMo branch is new
(Stage 2 work).

### What stays the same

- `OTR_BatchFluxRender` — unchanged; eats the same PASS3 JSON shape
- `OTR_ShotDurationCalculator` — unchanged
- `SaveImage` — unchanged (still writes `output/otr_videoplan_pass3_NNNNN.png`)
- `nodes/production_ledger.py` — unchanged

### What gets added later (after we discuss)

- `OTR_HuMoBatch` — takes FLUX renders + sliced audio + line text → HuMo motion clips
- `OTR_LedgerVideoConcat` — overlays HuMo clips onto a base track using ledger timings
- A new fixture: hand-saved ledger from tonight's `signal_lost_astrotech_*` run
  as the canonical TEST input

## Why I didn't ship `OTR_LoadLedger` tonight

1. The output JSON shape needs to byte-match `OTR_VideoPlan`'s shape, which
   requires reading `build_pass1_char_prompts`, `build_pass2_scene_prompts`,
   `build_shot_plan` and reproducing them. ~150 lines, but a single field
   typo means BatchFluxRender silently rerenders the wrong shots.
2. Selecting which ledger to load needs a UX decision: file path widget
   (typo-prone) vs picker vs "always use newest". Worth deciding together.
3. Stage 2 HuMo isn't here yet — building the audio/timeline socket pair
   without a consumer is yagni risk.
4. **TEST workflow JSON surgery** — replacing one node with another and
   rewiring is the kind of change that previously broke widgets_values
   alignment (BUG-LOCAL-059). Worth doing with you watching.

## What DID ship tonight

- **Ledger gets shots[] from VideoPlan PASS3** (hook in `nodes/otr_video_plan.py`).
  Means: when TEST eventually loads a FULL-run ledger, the shots[] table
  has detailed `visual_prompt` per shot ready to feed BatchFluxRender.
- This is the prep step that makes `OTR_LoadLedger`'s job trivial: it
  reads `shots[].visual_prompt` and emits PASS3 JSON.

## Suggested next session order

1. Review this doc together (15 min)
2. Decide: file path widget vs picker vs newest-on-disk for ledger selection
3. Pick a canonical fixture ledger to use as the test input (probably
   tonight's `signal_lost_astrotech_research_facility_humming_*`)
4. Build `OTR_LoadLedger` (~30 min once shape is decided)
5. Patch `otr_videoplan_TEST.json` to swap OTR_VideoPlan -> OTR_LoadLedger
6. Run TEST against the canned ledger, verify same FLUX output as the
   original FULL run produced
7. Commit + green light Stage 1 (FLUX.2-klein swap)
8. Then Stage 2 (HuMo) — at which point the audio_timeline_json + HuMo
   path becomes load-bearing
