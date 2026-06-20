# Audio Segment Normalization -- Implementation Sequence

Date: 2026-06-19
Branch: `v2.0-alpha`
Grounded against HEAD `9935161`.

Chunk-by-chunk order for Phase 1 (CPU code) and Phase 2 (GPU verify). Each chunk
lands in code at a cited `file:line`, regresses green, and commits + pushes per the
standing rule. **No Phase 1 chunk starts until BOTH the story-quality soak
(`local_5c212bd6`) AND the qwen/hidream promotion (`local_608386ee`) post
completion** (SPEC_JEFFREY.md sequencing).

Guiding rule: **the parity test goes green before the feature exists.** We build the
no-op gate first, prove byte-identity, and only then add behavior behind it.

---

## Phase 1 -- CPU-side code

### Chunk 0 -- Parity test + no-op gate scaffold (THE CANARY FIRST)

**Goal:** disabled path provably byte-identical, before any normalization math.

- Add `_apply_segment_normalization(all_segments, sample_rate, render_log,
  enabled=None)` near the existing helpers at `nodes/scene_sequencer.py:93-145`.
  When `enabled` is falsy (default: read `OTR_ENABLE_SEGMENT_NORM`), **return the
  input list and arrays unchanged** + a meta dict `{"enabled": False,
  "method": "off", ...}`. No ON branch yet (raise `NotImplementedError` internally
  is fine -- it is never reached when OFF).
- Wire the gated call in immediately before the concat at
  `nodes/scene_sequencer.py:827`:
  ```python
  all_segments, _segnorm_meta = _apply_segment_normalization(
      all_segments, sample_rate, render_log)
  if all_segments:
      combined = np.concatenate(all_segments)
  ```
- Add `tests/test_audio_segment_normalization.py` with tests 1a + 1b
  (TEST_PLAN.md §1): OFF is object-identity and bitwise-equal concat.
- **Regress:** full suite + Bug Bible + disabled-path identity tests green.
- **Commit + push:** `audio-segnorm C0: gated no-op stage + disabled-path parity test (byte-identical canary)`.

### Chunk 1 -- Measurement (first-target step 1)

**Goal:** measure each segment's loudness; still OFF by default, behavior inert.

- Add a pure measurement helper (RMS dBFS + peak dBFS per segment; LUFS only if a
  loudness lib imports -- ARCHITECTURE.md §5). Lives next to
  `_apply_segment_normalization` in `scene_sequencer.py`.
- Populate the meta `per_segment_db` vector even when OFF? No -- when OFF the stage
  must not iterate/allocate in a way that risks divergence; measurement runs only
  on the ON path and (read-only) for the report. Keep OFF a strict no-op.
- Add test 6 `test_segnorm_measures_each_segment` (driven via `enabled=True`).
- **Regress** (suite + Bug Bible + parity). **Commit + push:**
  `audio-segnorm C1: per-segment loudness measurement (rms/peak; lufs if available)`.

### Chunk 2 -- Per-segment normalization (first-target step 2)

**Goal:** the ON path applies peak-safe per-segment gain toward a common reference.

- Implement the ON branch: compute reference loudness (decided here per
  ARCHITECTURE.md §8 -- target below the 0.85 peak headroom so no clipping), apply
  per-segment gain, clamp/guard so no sample reaches full scale. Return **new**
  arrays (inputs never mutated -- test 8).
- Dialogue/announcer turns normalized; SFX/music left alone in v1 (ARCHITECTURE.md
  §8). The join itself is unchanged: still `np.concatenate(all_segments)` at
  `scene_sequencer.py:829` (first-target step 3 -- "join segments" is the existing
  concat, now fed normalized arrays).
- Add tests 4, 5, 8 (no clipping, reduced spread, inputs untouched).
- **Regress.** **Commit + push:**
  `audio-segnorm C2: peak-safe per-segment normalization behind OTR_ENABLE_SEGMENT_NORM`.

### Chunk 3 -- Metadata stamp (enabled/disabled, both states)

**Goal:** stamp `meta["segment_normalization"]` in the ledger, OFF and ON.

- SceneSequencer records its `_segnorm_meta` into the in-flight ledger; the
  EpisodeAssembler folds it under `meta["segment_normalization"]` in the existing
  schema-l3 write-back at `nodes/scene_sequencer.py:1206-1237` (next to
  `record_phase_ms` / `append_audio_gate`). Ledger-only; never enters the master
  WAV bytes (ARCHITECTURE.md §4).
- Add test 7 `test_segnorm_meta_stamp_present` (both states) + ledger round-trip.
- **Regress** -- including the disabled-path identity gate (the stamp must not
  perturb the master). **Commit + push:**
  `audio-segnorm C3: ledger metadata stamp (segment_normalization enabled/disabled)`.

### Chunk 4 -- Final episode loudness report (first-target step 4)

**Goal:** emit LUFS/RMS/peak for the finished episode.

- Add `episode_loudness_report(episode_waveform, sample_rate)` (read-only) and call
  it after the master is frozen at `nodes/scene_sequencer.py:1126-1187`; log it and
  fold it into the ledger meta. Never alters `episode_waveform`.
- Add test 9 `test_episode_loudness_report_shape`.
- **Regress.** **Commit + push:**
  `audio-segnorm C4: final episode loudness report (lufs/rms/peak)`.

### Chunk 5 -- Docs/wiring close-out for Phase 1

- Confirm no `INPUT_TYPES` / `widgets_values` / workflow-JSON change was needed
  (env-var gate, ARCHITECTURE.md §3). If a widget was added instead, re-validate the
  workflow JSON per CLAUDE.md §0 in this same chunk.
- Update `docs/GO_FORWARD_PLAN.md` + the otr-build-tracker with Phase 1 status.
- **Commit + push:** `audio-segnorm C5: phase-1 close-out + go-forward update`.

---

## Phase 2 -- GPU verify (single-episode smoke)

Only after BOTH blocking sessions free the 5080.

1. Reset the box per CLAUDE.md §4 (selective CIM kill by CommandLine, NOT blanket
   `Stop-Process`; confirm port 8000 empty + VRAM at desktop baseline).
2. **Disabled-path runtime gate:** run `test_audio_byte_identical.py` with
   `OTR_REGRESSION_RUNTIME=1`, seeds pinned, `OTR_ENABLE_SEGMENT_NORM` unset --
   master SHA must equal the frozen baseline. (Canary at full scale.)
3. **Enabled-path smoke:** boot the headless API on :8000, load the REAL
   `workflows/otr_scifi_16gb_full.json` (CLAUDE.md §0), render ONE episode with
   `OTR_ENABLE_SEGMENT_NORM=1`. Asset lands at its canonical
   `otr/episodes/<ep>/` path (CLAUDE.md §6); confirm with `Test-Path`.
4. Capture per-segment LUFS/RMS/peak before+after and the episode loudness report.
5. Clipping check: peak `> 0 dBFS` must be **zero** new clipping.
6. Bark check: confirm Bark-rendered segment bytes are unchanged (Bark renders
   first; normalization is the downstream post-process).
7. Runtime impact: wall-clock of the normalization stage vs a baseline OFF run.
8. Example comparison: render the SAME script twice (OFF, then ON) for the
   side-by-side deliverable.

---

## Phase 3 -- Deliverables

- `TEST_RESULTS.md` -- every test outcome with `file:line` + green/red.
- `LOUDNESS_MEASUREMENTS.md` -- before/after per segment + episode totals.
- `RUNTIME_IMPACT.md` -- wall-clock comparison, breakdown by stage.
- `EXAMPLE_EPISODE_COMPARISON.md` -- paste-ready paths to the two example mp4s +
  walkthrough of where the difference is audible.
- Final commit + push; GO_FORWARD_PLAN + tracker updated.

---

## Commit cadence summary

| Chunk | Lands at | Commit (push same session) |
|-------|----------|----------------------------|
| C0 | `scene_sequencer.py:~93`, `:827`; new test file | gated no-op + parity canary |
| C1 | `scene_sequencer.py:~93` | per-segment measurement |
| C2 | `scene_sequencer.py:~93` ON branch | peak-safe normalization |
| C3 | `scene_sequencer.py:1206-1237` | ledger metadata stamp |
| C4 | `scene_sequencer.py:1126-1187` | episode loudness report |
| C5 | docs/wiring | phase-1 close-out |

Every chunk: full suite + Bug Bible + `test_audio_byte_identical` UNPROMPTED, green
before commit. Push to `v2.0-alpha` only (prod/main gated).
