# R3 HARDENED -- WIRING: integration, no-JSON, re-baseline + calibration

Folds the R3 panel must-fixes (see pass03_judgment.md). Two opens RESOLVED by grounding
`tests/_run_baseline.py` + `scene_sequencer.py`.

## 1. NO workflow-JSON / node / widget change (grounded)
0 matches for normalization terms in `otr_scifi_16gb_full.json`; `OTR_SEGMENT_*` are env knobs like the
existing `OTR_MASTER_MAKEUP_DB`. The 3 dialogue edits (:747/:753/:775) are INSIDE `OTR_EpisodeAssembler`
(already wired) -> live on land, nothing dormant. (CLAUDE.md S0 "unwired = dead" N/A; env is runtime.)

## 2. Byte-identical gate mechanics (grounded)
Fixtures `tests/fixtures/baseline_v1.5.wav` + `.sha256`. Structural tests are content-independent (stay
green). Byte-compare gated behind `OTR_REGRESSION_RUNTIME=1` + GPU; default peak = byte-identical.
`_run_baseline.py` drives a RUNNING ComfyUI at `127.0.0.1:8000` (POST `/prompt`); audio captured as FLAC
-> ffmpeg -> WAV (pcm_s16le/48k/mono) -> sha256. `FIXED_SEEDS={}` -> determinism comes from the SERVER's
`OTR_CAST_SEED`/`OTR_STYLE_SEED` env, not injected seeds.

## 3. Prove peak parity BEFORE landing (new)
- IN-SUITE: `tests/test_segment_loudnorm.py` asserts `_level_dialogue_clip(x)` in peak mode returns
  byte-for-byte what `_normalize_clip(x)` returns (the call-site swap is a provable no-op by default).
- ONCE before Chunk 1 lands: run the runtime gate with `OTR_REGRESSION_RUNTIME=1` + `OTR_SEGMENT_LOUDNORM`
  unset/peak on the server -> confirm it still matches `baseline_v1.5`.

## 4. Re-baseline procedure (only when flipping to rms; operator-gated GPU)
Env lives in the SERVER process (the nodes read `os.environ` server-side; the capture shell does NOT
propagate):
1. Calibrate the target (section 5).
2. RESTART ComfyUI with: `OTR_SEGMENT_LOUDNORM=rms`, `OTR_SEGMENT_TARGET_RMS_DBFS=<measured+4>`,
   `OTR_MASTER_MAKEUP_DB=0` (explicit, so a stale 4.0 can't contaminate), `OTR_CAST_SEED=42`,
   `OTR_STYLE_SEED=42`. Confirm a render preflight log line shows effective mode=rms / target / makeup=0.
3. Copy the current `baseline_v1.5.wav` aside, then `python tests/test_audio_byte_identical.py
   --capture-baseline` -> rewrites WAV + `.sha256`.
4. A/B old vs new (ffmpeg `loudnorm` read + listen): delta is ONLY the intended leveling, no clipping
   (peak <= ceiling), no pumping / noise-floor lift.
5. Validate across 2-3 episodes (operator RELEASE acceptance -- NOT a unit test).
- FIXTURE/CI RULE: while default stays peak, do NOT commit rms fixtures (the byte-compare would generate
  peak bytes vs an rms golden -> false fail). Add a skip-guard: byte-compare skips when
  `OTR_SEGMENT_LOUDNORM` not in {unset, peak}. Promoting rms to DEFAULT = a future separate commit that
  flips the code default AND commits the rms golden together.

## 5. Calibration procedure (commit `tools/measure_dialogue_rms.py`)
Measure dialogue-only clips, PRE-master, over ACTIVE windows (above a silence floor), aggregated per
clip -> mean RMS dBFS (float full-scale, mono). Set `OTR_SEGMENT_TARGET_RMS_DBFS = measured + 4.0` to
compensate for the dropped +4 dB master makeup, so final program loudness ~matches today. (Bark sits
~ -16..-18 dBFS, so expect a target ~ -12..-14 after the +4.)

## 6. Commit chunking + gates
- Chunk 1 (one green commit): new functions + 3 call-site edits + `_master_loudness` makeup tweak +
  `tests/test_segment_loudnorm.py` (incl. peak-parity + SFX-stays-peak) + the env-knob README note.
  Default peak -> full suite + Bug Bible green. Commit + push to `v2.0-alpha`.
- Operator-gated (NOT in the code chunk): calibrate -> restart server rms -> re-baseline -> multi-episode listen.

## 7. Resolved / no remaining build opens
`import os`+`import math` already module-level (lines 28-29). `_run_baseline` server-based + `FIXED_SEEDS={}`
confirmed. SFX (:726) calls `_normalize_clip` directly (never the router; `_normalize_clip` never reads
the env) -> peak by construction; covered by a test.
