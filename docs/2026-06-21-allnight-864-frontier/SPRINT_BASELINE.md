# Story-Engine v1 -- SPRINT_BASELINE (measurement harness + soak)

**Status (2026-06-21):** ALL CODE for F1-F8 + the Sprint-0 harness is SHIPPED + PUSHED to `v2.0-alpha`
(HEAD `d9b25a0` == origin; full suite 4717 pass / 33 skip; Bug Bible 16/7/3; node imports verified clean in
the venv). The remaining work is the GPU MEASUREMENT (baseline + after) and the operator-requested 500-word
SOAK. This file records the exact, reproducible commands so the measurement is apples-to-apples.

## The measurement harness (committed)
`scripts/story_quality_scan.py` reads the on-disk episode ledgers (`output/otr/episodes/<ep>/audio/*_ledger.json`)
and reports, per leg + aggregate:
- `length_ratio` = (character + announcer voiced words; EXCLUDE music) / `target_words`
- `length_pass_fired` (from `meta.length_pass_report`)
- `episode_valid` = freeze CRITICAL pass AND `meta.slot_drama_contracts_audit.ok`
- `outro_hedge_vs_resolved` = HEDGE_LIST phrase in the closing announcer line AND `is_resolved_ending_change(ending_change)`
- `narration_self_address_lines` = count flagged by the SHARED `_otr_line_hygiene.detect_narration_self_address`

Run it against any directory of ledgers:
```
$env:PYTHONUTF8=1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\story_quality_scan.py `
  --ledgers "C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\*\audio\*_ledger.json" `
  --target-words 864 --label after --md-out docs\2026-06-21-allnight-864-frontier\scan_after.md
```

## Acceptance targets (12-leg fixed smoke, target_words=864)
| Metric | Today (pre-v1) | Target |
|---|---|---|
| `length_ratio` mean | 0.70 | >= 0.85, `length_pass_fired` <= 2/12 |
| `episode_valid` | 24% | >= 11/12 |
| `outro_hedge_vs_resolved` | several | 0/12 |
| `narration_self_address_lines` | >0 | 0/12 |
| full suite / Bug Bible / `test_audio_byte_identical` | green | stays green |

## 12-leg / 864-word measurement (GPU/LLM -- pending an operator/coder GPU window)
The baseline ideally runs on the pre-v1 commit `504466e`; the after-run on HEAD. Both via the story-only
soak harness (no video render), pinning a DISTINCT `OTR_STYLE_SEED` per leg (per-leg seed -> arc-shape +
news variety; reproducible). Drive it with `scripts/_otr_overnight_story_soak.py` (story-only) against a
dedicated `:8011` headless server (NOT the Desktop on :8000). Reset the box first (selective CIM kill;
ports 8000/8011 clear; VRAM at ~1.5 GB baseline).

## The operator-requested 500-word SOAK (the finale)
Config the operator asked for: indextts2 char voice, LTX-audio bookends (announcer + music), flux2_klein
char-beat stills, max creativity, cheap OpenRouter frontier writer. Reset the box first (CLAUDE.md sec 4).
Boot a dedicated headless server, then run the combo soak detached to a log:
```
# 1) box already clean if nvidia-smi ~1.5 GB and :8000/:8011 have no listener
# 2) boot the soak server (UTF-8), then set the combo env and run _otr_combo_soak.py:
$env:OTR_SOAK_TARGET_WORDS = "500"
$env:OTR_SOAK_CHAR_VOICE    = "indextts2"
$env:OTR_COMBO_ANNOUNCER    = "ltx_av_talk"     # LTX audio in the open bookend
$env:OTR_COMBO_MUSIC        = "ltx_av_music"     # LTX audio in the close bookend
$env:OTR_COMBO_BEATS        = "flat_still"       # character beats are stills...
$env:OTR_COMBO_BEATS_IMG    = "flux2_klein"      # ...rendered with flux2_klein
$env:OTR_ENABLE_FLUX2_KLEIN = "1"
$env:OTR_COMBO_CREATIVITY   = "maximum_chaos"    # max creativity (confirm the live enum label)
# writer = cheap OpenRouter frontier: set the slot-a/slot-b OpenRouter model on the workflow
#          (e.g. a gemini-flash / deepseek ~latest alias) before submitting.
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\_otr_combo_soak.py
```
NOTE: a 500-word episode with LTX-audio bookends + flux2 stills + indextts2 is a long render (tens of
minutes per episode); "a few" episodes is a multi-hour background run. Confirm each asset exists at its
canonical `otr/episodes/<ep>/` path before declaring success, and read the server log for `obs_publish OK`.

## What this session did NOT do (handed off)
- The GPU 12-leg baseline/after numbers (needs an uninterrupted GPU window).
- The 500-word soak render (the command above is ready; left for an operator GPU window so an unmonitored
  multi-hour render does not tie up the box).
- F8 macro-prompt arc_shape context (deferred; the dramatic-state path already carries arc_shape + the meta
  stamp, so the F8 acceptance is met).
