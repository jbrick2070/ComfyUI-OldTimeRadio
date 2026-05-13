# Fixture dur_s audit — Sprint 8.1

**Date:** 2026-05-12
**Scope:** every `dur_s` numeric sample under `tests/`
**Bound under audit:** G7 SFX `dur_s` invariant `[0.5, 10.0]`
(consumer intersection of `OTR_BatchAudioGen` and
`OTR_BatchProceduralSFX`, tightened in Sprint 6.4 / commit
`6093182`).

## Why this audit

Sprint 6.4 narrowed G7's SFX `dur_s` window from `(0.25, 12.0)` to
`(0.5, 10.0)`. The old window was the loose union of the two
downstream consumers' clamps; the new window is the strict
intersection — every value that passes G7 now renders at face
value through every consumer with no silent clamping.

Risk: a test fixture that emits an SFX `dur_s` outside the new
window would have passed under the old bound and now fails. The
audit confirms no such drift exists in the fixture corpus.

## Method

Regex sweep over every `tests/*.py` file matching:

- `"dur_s": NUMBER`  (dict-literal form)
- `dur_s = NUMBER`   (assignment / kwarg form)

Skips comment lines. Classifies each hit by:

1. **WITHIN** `[0.5, 10.0]` — fixture passes G7 directly.
2. **INTENTIONAL OOB** — out-of-bounds value in a test whose name
   signals deliberate rejection (`old_lower_bound`,
   `old_upper_bound`, `out_of_bounds`, `_oob`, `raises`,
   `rejects_dur`, `below_min`, `above_max`).
3. **UNEXPECTED OOB** — out-of-bounds in a test whose name does
   not signal rejection. These are the audit action items.

The `UNEXPECTED OOB` bucket then gets a manual second pass to
confirm whether each hit is actually a non-SFX line (character,
music, video shot — none of which are governed by G7).

## Results

```
WITHIN:        61 hits
INTENTIONAL:   6 hits  (all in test_per_cue_sfx_dur.py)
UNEXPECTED:    7 hits  (manual second-pass below)
```

### Intentional out-of-bounds (6 hits, all in `test_per_cue_sfx_dur.py`)

| Line | dur_s | Test | Purpose |
|------|-------|------|---------|
| 73   | 0.25  | `test_g7_dur_s_at_old_lower_bound_now_raises` | Pin: old lower bound now FAILS G7 |
| 80   | 0.25  | (same)                                        | Same fixture, second assertion |
| 95   | 12.0  | `test_g7_dur_s_at_old_upper_bound_now_raises` | Pin: old upper bound now FAILS G7 |
| 100  | 12.0  | (same)                                        | Same fixture, second assertion |
| 280  | 60.0  | `test_procsfx_clamps_out_of_bounds`           | Far-OOB sample exercises consumer clamp |
| 280  | -5.0  | (same)                                        | Negative dur_s exercises consumer clamp |

All correctly classified by the heuristic. No action.

### Unexpected out-of-bounds — manual classification (7 hits)

| File | Line | dur_s | Test | Manual verdict |
|------|------|-------|------|----------------|
| `test_per_cue_sfx_dur.py` | 152 | 25.0 | `test_g7_non_sfx_lines_excluded` | **Non-SFX** — line is `speaker_role: "character"`. The fixture's whole point is to verify G7 ignores non-SFX lines. Inline comment says so: `# would violate if sfx; ok as character`. |
| `test_production_ledger.py` | 493 | 12.0 | `test_apply_sfx_and_music_timings` | **Music cue** — value lands on `apply_music_timings`, not `apply_sfx_timings`. G7 governs SFX only. |
| `test_render_flux_batch.py` | 52 | 12.0 | (module-level fixture) | **Video shot** — `shots[].dur_s = 12.0` is a multi-beat shot duration. G7 doesn't govern video-shot durations. |
| `test_render_flux_batch.py` | 405 | 12.0 | `test_composite_finder_ignores_other_speakers` | **Video shot** — same `shots[].dur_s` field. |
| `test_render_flux_batch.py` | 409 | 12.0 | (same) | **Video shot** — same. |
| `test_video_composite_per_clip_mux.py` | 174 | 12.0 | `test_canvas_dims_in_filter` | **Music line** — `speaker_role: "music_open"`. G7 SFX-only. |
| `test_video_composite_per_clip_mux.py` | 392 | 0.0 | `test_skips_lines_with_invalid_timing` | **Invalid-timing rejection** — fixture deliberately emits `dur_s=0.0` to verify the muxer skips lines with invalid timing. The test's name says so explicitly. |

**Manual verdict — all 7 unexpected hits are correctly intentional.**
None is an SFX `dur_s` outside the new G7 window. No fixture changes
needed.

## Heuristic gaps to fix later (S8 follow-ups, not blocking)

The `INTENTIONAL_OOB` name list could be expanded to catch:

- `excluded` (`test_g7_non_sfx_lines_excluded` would auto-classify)
- `non_sfx`
- `music_` (when the fixture is on a music cue, not an SFX line)
- `_shot` / `shots[` (video-shot dur_s)
- `invalid_timing` / `skips_lines_with_invalid_timing`

Adding those keywords would cut the manual-pass burden on future
audits to zero. Out of scope for S8.1; the current heuristic +
manual pass produced a clean verdict.

## Verdict

**G7 fixture compliance: PASS.**

- 0 SFX-line fixtures emit `dur_s` outside `[0.5, 10.0]`.
- 6 deliberately-OOB samples are inside boundary-rejection tests
  whose subject IS the G7 invariant.
- 7 manually-confirmed non-SFX OOB samples are correct (G7 is
  SFX-only).

No fixture rewrites required for Sprint 8.1.

## Reproduce

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  C:\Users\jeffr\AppData\Roaming\Claude\local-agent-mode-sessions\
6503cb4a-a0ec-4e89-b422-c889f09084d6\
cff774c2-bb97-4b73-b186-daa560353c46\
local_0bf5f6c9-c201-48ca-8b80-1c50de1d90a1\outputs\_s8_1_dur_audit.py
```

(One-shot script, lives in the session outputs dir; not committed
to repo. Re-run on demand.)
