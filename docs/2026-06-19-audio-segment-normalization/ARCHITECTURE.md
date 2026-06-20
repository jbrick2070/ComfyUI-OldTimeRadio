# Audio Segment Normalization -- Architecture Note

Date: 2026-06-19
Branch: `v2.0-alpha`
Status: Phase 0 (doc-only; no code edits yet -- story-quality soak owns the audio spine)

This note is grounded in the real code at HEAD `9935161`. Every claim cites
`file:line`. It defines exactly where the new optional stage goes, what stays
untouched, how the default-OFF gate works, where the metadata stamp lands, and
how the disabled path is proven byte-identical.

---

## 1. The audio spine today (where segments are born, normalized, and joined)

The audio pipeline has **two distinct join surfaces**. Knowing which one is "the
segment concat/join" the spec targets is the whole design decision.

### 1a. SceneSequencer -- per-line segment assembly (THE TARGET SURFACE)

`nodes/scene_sequencer.py`, the `SceneSequencer` render loop:

- Each ledger line is rendered/resampled into a **per-segment float32 numpy
  array** and accumulated into the `all_segments` list:
  - dialogue (announcer / Kokoro): `scene_sequencer.py:745-748`
  - dialogue (pre-rendered TTS clip): `scene_sequencer.py:750-755`
  - dialogue (inline Bark fallback): `scene_sequencer.py:772-776`
  - SFX: `scene_sequencer.py:724-730`
  - each branch calls `_normalize_clip(...)` (peak-normalize to 0.85) -- see 1c.
- Segments are appended at `scene_sequencer.py:792` (`all_segments.append(segment_np)`).
- **THE JOIN:** `scene_sequencer.py:827-829`

  ```python
  # Concatenate all dialogue/SFX segments
  if all_segments:
      combined = np.concatenate(all_segments)
  ```

  This is the "before audio concat/join" surface named in the spec. `all_segments`
  is a clean list of independent per-line waveforms here -- the natural place to
  measure-and-normalize each segment **before** they lose their boundaries in the
  concat.

- `combined` then gets the environment/room-tone bed mixed under it
  (`scene_sequencer.py:833-849`) and is returned as the scene `AUDIO`.

### 1b. EpisodeAssembler -- theme bookending + master (NOT the target)

`EpisodeAssembler.assemble` at `scene_sequencer.py:1031`:

- Builds `segments = [opening_theme?, main_waveform, closing_theme?]`
  (`scene_sequencer.py:1056-1075`).
- Joins them with an **equal-power (sqrt) crossfade** at
  `scene_sequencer.py:1088-1111`.
- Applies the **final loudness master** (`_master_loudness`) at
  `scene_sequencer.py:1122`.
- Writes the **frozen master WAV** (16-bit PCM, stdlib `wave`) at
  `scene_sequencer.py:1126-1187`.

This stage is the *episode-level master*, not per-segment work. The closing/opening
themes here are whole-program bookends, not "character turns." We do **not** insert
per-segment normalization here -- and we do **not** touch `_master_loudness` (that
is the roundtable-gated "loudness-target policy"; see §6).

### 1c. The frozen helpers that already exist (do not modify)

- `_normalize_clip(clip_np, target_peak=0.85)` -- `scene_sequencer.py:93-106`.
  Peak-normalize each clip to 0.85. **Already part of the frozen baseline.** The
  new stage is *additional and gated*, layered on top; it must not change
  `_normalize_clip` or its call sites, or the disabled path diverges.
- `_master_loudness(...)` -- `scene_sequencer.py:109-145`. Makeup gain + tanh soft
  limiter + -1.0 dBFS true-peak ceiling. **Roundtable-gated. Untouched.**

### 1d. The terminal mux (the byte-identity contract downstream)

`nodes/otr_master_audio_mux.py` -- `OTR_MasterAudioMux` is the ONLY node that adds
audio (invariant V-1). It muxes the frozen master WAV onto the silent composite
with `-c:a copy` (NO re-encode, NO `-shortest`) and **asserts the output audio
decodes PCM-identical to the master** (`otr_master_audio_mux.py:176-185`). So
whatever the audio pipeline freezes as the master WAV is, byte-for-byte, the
episode's audio. The byte-identity gate therefore lives at the master-WAV write
(`scene_sequencer.py:1126-1187`): if the disabled path produces the same master
WAV bytes, the whole episode is byte-identical.

---

## 2. Where the new stage goes (precise)

**Insertion point:** `nodes/scene_sequencer.py`, immediately before the concat at
`scene_sequencer.py:827-829`, operating on the `all_segments` list.

Proposed shape (Phase 1; shown for design clarity, not committed here):

```python
# --- Audio units assembled: {len(all_segments)} ---  (existing line 825)

# Optional, default-OFF per-segment loudness normalization (gated).
# Identity when disabled -> all_segments is returned unchanged -> the
# np.concatenate below sees the exact same arrays -> byte-identical master.
all_segments, _segnorm_meta = _apply_segment_normalization(
    all_segments, sample_rate, render_log,
)

# Concatenate all dialogue/SFX segments  (existing line 827-829)
if all_segments:
    combined = np.concatenate(all_segments)
```

`_apply_segment_normalization` is a **new pure helper** added near
`_normalize_clip` / `_master_loudness` (around `scene_sequencer.py:93-145`). It:

1. **Measures** each segment's loudness (per first-implementation-target step 1).
2. If the gate is OFF: returns `(all_segments, {"enabled": False, ...})`
   **without copying or mutating any array** -- identity, byte-identical.
3. If ON: computes a per-segment gain toward a common reference loudness, applies
   it (peak-safe, clipping-guarded), and returns the new list + a meta dict with
   per-segment before/after numbers.

The `_segnorm_meta` dict is threaded up to `EpisodeAssembler` for the stamp (§4).
Because `SceneSequencer` returns only `AUDIO`, the meta is carried via the ledger
(the same channel `audio_gate_record` already uses -- §4), not a new node socket.
This avoids any `INPUT_TYPES` / `widgets_values` / workflow-JSON change (CLAUDE.md
§0 positional-widget risk) for the measurement path.

### Why SceneSequencer and not EpisodeAssembler

- "Normalize each generated **segment** independently so **character turns** and
  scene transitions arrive at the assembly stage with more consistent loudness"
  -- the segments are individual at `scene_sequencer.py:792` and lose their
  boundaries at `:829`. By the EpisodeAssembler the per-line structure is gone
  (it sees one `main_waveform`). Per-segment work must happen here.
- "...arrive **at the assembly stage**..." -- the assembly stage is the
  EpisodeAssembler; the normalization is upstream of it, exactly as worded.

---

## 3. The default-OFF gate

**Recommendation: env-var primary, no widget in v1.**

- Gate: `OTR_ENABLE_SEGMENT_NORM` -- unset / `"0"` / `""` = OFF (default);
  `"1"` = ON. Read via `os.environ.get` inside `_apply_segment_normalization`.
- Optional method/target knobs (also env, read only when ON; never consulted when
  OFF so they can never perturb the disabled path):
  - `OTR_SEGNORM_METHOD` -- `"rms"` (default v1) | `"peak"` | `"lufs"` (only if a
    LUFS measurement lib is available; see §5). Method selection here is an
    *implementation* knob for the per-segment stage, **not** the roundtable-gated
    "LUFS standard selection" policy for the master -- it only affects the gated,
    default-OFF path and never the frozen master.
  - `OTR_SEGNORM_TARGET_DB` -- reference level the segments are matched toward
    (default chosen in Phase 1 to sit *below* the existing 0.85 peak headroom so
    no clipping is introduced).

**Why env-var, not a widget (matches existing precedent):**

- `_master_loudness` already gates its makeup gain on the env var
  `OTR_MASTER_MAKEUP_DB` (`scene_sequencer.py:118, 131`) -- env-var loudness knobs
  are the established pattern in this exact file.
- A widget on `SceneSequencer` would force an `INPUT_TYPES` change **and** a
  positional `widgets_values` append in `workflows/otr_scifi_16gb_full.json`
  (CLAUDE.md §0). That is real drift surface for a feature that ships default-OFF.
  Env-var has **zero** workflow-JSON churn and makes the disabled path trivially
  the current path.
- The disabled path is then *provably* the current code: when the env var is
  unset, `_apply_segment_normalization` returns its input list object unchanged
  and `np.concatenate` runs on the identical arrays.

A widget can be added later (append-only at the end of `widgets_values`, re-validated
per CLAUDE.md §0) if the operator wants it surfaced in the graph -- but that is an
explicit follow-up, not part of this first target.

---

## 4. The metadata stamp (enabled/disabled)

**Home: the production ledger meta**, written in the EpisodeAssembler's existing
schema-l3 ledger write-back block (`scene_sequencer.py:1206-1237`), alongside
`record_phase_ms` and `audio_gate_record` / `append_audio_gate`.

Stamp shape (per the kickoff):

```python
meta["segment_normalization"] = {
    "enabled": bool,                 # OTR_ENABLE_SEGMENT_NORM truthy
    "method": str,                   # "off" when disabled; else rms/peak/lufs
    "target_db": float | None,
    "per_segment_lufs_db": [...],    # or rms_db / peak_db per the available metric
    "before_after_delta_db": [...],  # gain applied per segment (0.0 vector when OFF)
}
```

- When OFF, the stamp is still written with `enabled: False, method: "off"` and
  empty/zero vectors -- the spec requires a stamp "indicating normalization
  enabled/disabled" in *both* states.
- The stamp is **ledger-only metadata**; it does NOT enter the master WAV bytes.
  Confirmed safe: the master WAV is written at `scene_sequencer.py:1175-1179` from
  `episode_waveform` PCM only; the ledger write-back is a *separate* best-effort
  block at `:1210` that never feeds back into the waveform. So stamping does not
  threaten byte-identity.
- The per-segment numbers come from `SceneSequencer` (where segments exist); they
  reach the EpisodeAssembler via the in-flight ledger (`_otr_ledger.in_flight_ledger_path`,
  already used at `scene_sequencer.py:1145, 1217`). SceneSequencer records its
  `_segnorm_meta` into the ledger at its own write-back; EpisodeAssembler reads it
  back and folds it under `meta["segment_normalization"]` plus the episode-level
  loudness report (§5).

---

## 5. The final episode loudness report (first-target step 4)

After the master is built (`scene_sequencer.py:1116-1122`) and the WAV frozen
(`:1126-1187`), emit an episode-level loudness report: LUFS (if a lib is available),
RMS dBFS, and true-peak dBFS for the final `episode_waveform`.

- RMS and peak are computable from numpy/torch with zero new deps and are always
  emitted.
- LUFS requires a loudness lib (e.g. `pyloudnorm`). **Availability is checked, not
  assumed** -- the spec says "LUFS/RMS/peak metrics **if available**." If no LUFS
  lib is importable, the report carries RMS + peak and notes LUFS as unavailable.
  We do **not** add a new hard dependency in this task.
- The report is written to the deliverable
  `docs/2026-06-19-audio-segment-normalization/LOUDNESS_MEASUREMENTS.md` (Phase 3)
  and also logged + folded into the ledger meta stamp (§4). **Measuring/reporting
  loudness is read-only** -- it never changes the master, so it is safe even on the
  disabled path (and on the disabled path it documents that the numbers equal the
  frozen baseline).

---

## 6. What stays untouched (the frozen spine)

- `_master_loudness` -- `scene_sequencer.py:109-145`. The episode loudness-target
  policy (makeup gain, limiter, ceiling). **Roundtable-gated. Not touched.**
- `_normalize_clip` and its existing call sites
  (`scene_sequencer.py:93-106, 726, 747, 753, 775`). The current frozen per-clip
  peak normalization. The new stage layers on top when ON; when OFF it is a no-op,
  so these run exactly as today.
- The EpisodeAssembler crossfade join `scene_sequencer.py:1088-1111` and the master
  WAV write `:1126-1187` -- unchanged bytes on the disabled path.
- `OTR_MasterAudioMux` (`nodes/otr_master_audio_mux.py`) -- `-c:a copy`, no
  `-shortest`, PCM-identity assert. Not touched; it is the downstream proof.
- **Bark rendering pipeline** -- Bark clips are produced upstream (pre-rendered into
  `tts_clips` / `announcer_clips` / `sfx_clips`, or inline via
  `_generate_bark_for_line` at `scene_sequencer.py:773`). The new stage operates on
  the **already-rendered, already-resampled numpy arrays** in `all_segments`. It is
  a pure downstream post-process: Bark's output bytes are never read back into Bark,
  never re-rendered, never altered. No Bark code is touched.

---

## 7. How the disabled path is proven byte-identical

1. **Design-level:** when `OTR_ENABLE_SEGMENT_NORM` is unset,
   `_apply_segment_normalization` returns the **same list object** with the **same
   array objects** -- no copy, no cast, no arithmetic. `np.concatenate` at
   `scene_sequencer.py:829` therefore consumes byte-for-byte identical input, so
   `combined`, `episode_waveform`, the master WAV, and the muxed episode are all
   identical.
2. **Test-level:** the parity test is written and green **before** any feature code
   (TEST_PLAN.md §1). Two layers:
   - **Pure/offline (always runs):** call the SceneSequencer assembly (or the
     extracted `_apply_segment_normalization` directly) with the gate OFF on a fixed
     synthetic `all_segments`, assert the output arrays are `is`-identical / bitwise
     equal to the input -- no GPU, runs in CI.
   - **Full runtime gate (operator/GPU):** the existing
     `tests/test_audio_byte_identical.py` SHA-256 gate
     (`test_audio_byte_identical_to_baseline`, `scene_sequencer`-fed master) run
     with `OTR_REGRESSION_RUNTIME=1` and `OTR_ENABLE_SEGMENT_NORM` **unset** must
     match the captured baseline hash exactly.
3. **Canary discipline:** `test_audio_byte_identical` runs UNPROMPTED after every
   Phase 1 chunk. If it ever goes red on the disabled path -> `git reset --hard`
   uncommitted + ping (SPEC_JEFFREY.md stop conditions).

---

## 8. Open design decisions deferred to Phase 1 (logged, not guessed)

- **Reference loudness for matching (ON path):** mean vs median of per-segment
  loudness, vs a fixed `OTR_SEGNORM_TARGET_DB`. Chosen in Phase 1 to guarantee no
  clipping (target below the 0.85 peak headroom). This is a *per-segment* knob on
  the gated path -- explicitly NOT the roundtable-gated master loudness-target
  policy.
- **Metric for v1 (`rms` vs `lufs`):** RMS is dependency-free and ships in v1;
  LUFS only if a lib is already importable on the box (checked, not added).
- **Music/theme exclusion:** whether `music_*` passthrough lines (which SceneSequencer
  does not segment -- `scene_sequencer.py:688-689, 716-717`) and the SFX bed are in
  or out of scope for matching. Default v1: normalize dialogue/announcer turns only,
  leave SFX/music dynamics alone (matches "character turns and scene transitions").
- **Widget surfacing:** deferred; env-var only in v1 (§3).

None of these widen scope beyond per-segment normalization; each is recorded here so
the Phase 1 coder makes the call deliberately, not silently.
