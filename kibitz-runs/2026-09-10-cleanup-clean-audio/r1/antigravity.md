VERDICT: yes-with-fixes. The core removal of synthetic roomtone and coloration defaults is sound and well-bounded, but the test harness expansion (P1) requires explicit synchronization with its pytest launcher (`test_canonical_audio_check.py`) and correlation gating for silent test cases.

MUST-FIX BEFORE BUILD:
1. [P1] `test_canonical_audio_check.py` receipt assertion desync.
   - Defect: Extending `scripts/otr_canonical_audio_check.py` from 2 to 4 cases (opening/no-opening x chirp/silence) and removing `rng_seed` breaks `tests/test_canonical_audio_check.py` at line 30 (`assert len(receipt["cases"]) == 2`) and line 33 (`assert case["rng_seed"] == 20260910`).
   - Concrete fix: Explicitly update `tests/test_canonical_audio_check.py` in P1 to assert `len(receipt["cases"]) == 4`, drop the `rng_seed` assertion, and assert for silent cases that `case["measured_scene_offset_s"] is None` and `case["nonzero_samples"] == 0`.

2. [P1] Correlation failure on zero audio in `scripts/otr_canonical_audio_check.py`.
   - Defect: When `silence=True`, running `correlate(decoded, template[start:stop])` on all-zero float arrays results in an all-zero cross-correlation where `np.argmax` returns index 0, producing an invalid pseudo-lag of `-start` (-0.65s) and failing the assertion at line 220 (`require(lag > 0 if opening else lag == 0)`).
   - Concrete fix: In `scripts/otr_canonical_audio_check.py:check_case`, wrap the acoustic template correlation and measured-offset verification (lines 214-234) inside `if not silence:`. For `silence=True`, set `measured_offset = None`, verify `np.count_nonzero(decoded) == 0` and `torch.count_nonzero(enhanced["waveform"]) == 0`, and verify ledger line records exist with valid structure without checking acoustic lag.

SHOULD-FIX:
1. [C4.1 / P1] Ambiguous "C1 stores/prose" scope boundary in `nodes/scene_sequencer.py`.
   - Defect: The plan states "include C1 stores/prose" without enumerating which exact dead state variables in `SceneSequencer.sequence` are in scope alongside `env_timeline`.
   - Concrete fix: Explicitly bound the in-method dead store cleanup to deleting `current_character_name = None` (`nodes/scene_sequencer.py:988`) and `current_character_name = character_name` (`:1124`), plus updating the section banner at line 1126 and module docstring lines 1-24. Do not include unrelated Wave 4 dead code items in this chunk.

2. [C4.2] Step numbering and leftover `scipy.signal` reference in `nodes/audio_enhance.py:_apply_tape_emulation`.
   - Defect: Deleting the noise block (lines 276-286) removes the only use of `scipy.signal` (`butter, sosfilt`) in `_apply_tape_emulation`, leaving Step 1 (saturation) and Step 3 (wow/flutter).
   - Concrete fix: Remove the local `from scipy.signal import butter, sosfilt` import entirely, renumber "Step 3" to "Step 2: Wow and Flutter", and update docstrings to reflect that tape emulation provides deterministic saturation and pitch modulation without synthetic noise.

3. [P1] Variant sync verification following canonical workflow edits.
   - Defect: Editing node 4 widget values in `workflows/otr_canonical.json:755-760` changes the canonical graph SHA. If variant generators or invariant checks compare against canonical, variants could drift.
   - Concrete fix: [ASSUMPTION] Add `python scripts/build_variants.py --check` (or variant sync run) to the P1 qualification battery to ensure any variant banks derived from canonical remain consistent.

OPTIONAL / NICE-TO-HAVE:
1. [C4.3] Update `AudioEnhance.INPUT_TYPES` tooltips in `nodes/audio_enhance.py:333-348` to clarify that `spatial_width=0.0` outputs clean dual-mono stereo without mid-side processing, and `lpf_cutoff_hz=0.0` disables low-pass filtering.

CUT THESE:
1. [P1] Any requirement to qualify GPU/CUDA audio execution paths in this wave. Removing `_generate_room_tone` statically eliminates both CPU and CUDA noise generation lines, and CPU-only verification keeps execution fast, deterministic, and hermetic.

ASSUMPTIONS MARKED:
- [ASSUMPTION] (Item 3 in SHOULD-FIX): Assumed that `scripts/build_variants.py` or `tests/test_bank_variants.py` checks variant consistency against `workflows/otr_canonical.json`.
