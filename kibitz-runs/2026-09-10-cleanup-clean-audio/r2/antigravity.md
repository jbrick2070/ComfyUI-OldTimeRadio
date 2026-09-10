VERDICT: build-ready as-is? yes-with-fixes. The core edits in `nodes/scene_sequencer.py` and `nodes/audio_enhance.py` are surgically grounded and sound, but the test harness script and launcher require specific sequencing guards and schema updates to avoid test breakage and ledger state pollution.

MUST-FIX BEFORE BUILD:
1. [P1.39] `scripts/otr_canonical_audio_check.py:check_case` / `nodes/_otr_ledger.py:538` — Supplemental tape mode loop risks mutating in-flight ledger singleton.
   Defect: `AudioEnhance.enhance` (lines 489-516) inspects `_OTRL.in_flight_ledger_path()` and appends `post_audio_enhance` audio gates to whatever active ledger exists on disk. If the supplemental 4-mode test loop runs after `pl.new_ledger()` has bound an episode, it will mutate that episode's ledger and alter its state.
   Fix: Place the supplemental `AudioEnhance` tape-mode verification loop at the start of `main()` before any `check_case` calls `pl.new_ledger()`, or run it in a clean scope where `in_flight_ledger_path()` returns `None`.

2. [P1.37 / P1.40] `tests/test_canonical_audio_check.py:30-33` — Launcher assertion breakages against 3-case receipt and removed RNG seeds.
   Defect: `tests/test_canonical_audio_check.py:30` asserts `assert len(receipt["cases"]) == 2`, and line 33 asserts `assert case["rng_seed"] == 20260910`. When `main()` is extended to 3 cases (chirp opening, chirp no-opening, silence opening) and `rng_seed` is removed, the launcher test will fail immediately.
   Fix: In `tests/test_canonical_audio_check.py`, update case count check to `assert len(receipt["cases"]) == 3`, delete the `rng_seed` assertion, and add checks asserting `case["measured_scene_offset_s"] is None` and `case["nonzero_samples"] == {"scene": 0, "enhanced": 0, "master": 0}` for the silence case.

3. [C4.26] `tests/test_sequencer_ledger.py:66-70,83-85` — Missing fixture cleanup will cause `AttributeError` in test suite.
   Defect: `tests/test_sequencer_ledger.py` patches `nodes.scene_sequencer._generate_room_tone` in `patched_sequencer_env`. Once `_generate_room_tone` is deleted from `nodes/scene_sequencer.py`, `unittest.mock.patch` will raise `AttributeError` during test setup.
   Fix: Delete `_fake_room_tone` and the `patch("nodes.scene_sequencer._generate_room_tone", ...)` block from `tests/test_sequencer_ledger.py` in the exact same chunk.

SHOULD-FIX:
1. [P1.37] `scripts/otr_canonical_audio_check.py:153` — Distinct episode directory paths for all 3 cases.
   Defect: If `silence=True` uses the same episode identifier as the chirp case (`canonical_audio_opening`), both runs will target `output_root / "otr" / "episodes" / "canonical_audio_opening" / "audio"`, causing ledger overwrite and race conditions on artifact generation.
   Fix: Explicitly map episode IDs to `canonical_audio_opening`, `canonical_audio_no_opening`, and `canonical_audio_silence_opening`.

2. [C4.16] `nodes/audio_enhance.py:346-347` — Tooltip and docstring discrepancy for `lpf_cutoff_hz`.
   Defect: Lowering `lpf_cutoff_hz` default from `16000.0` to `0.0` and min from `8000.0` to `0.0` leaves the tooltip text at line 347 stating `16000=default`, which contradicts the clean opt-in design.
   Fix: Update tooltip at line 347 to `"Low-pass filter cutoff Hz - kills Bark chirp artifacts (0=off/default, 16000=subtle cutoff)"`.

OPTIONAL / NICE-TO-HAVE:
1. [C4.15] `nodes/audio_enhance.py:251` — Update docstring of `_apply_tape_emulation` from `"saturation, hiss, wow/flutter"` to `"saturation, wow/flutter"`.
2. [C4.18] `nodes/audio_enhance.py:121` — Fix docstring for `_stereo_decorrelate` from `"0.0 = mono"` to `"0.0 = unchanged width"`.

CUT THESE (over-engineering):
1. [P1.37] Redundant second silent case (e.g. silence no-opening): One silent opening assembly proves both silence preservation through `SceneSequencer`, `AudioEnhance`, and `EpisodeAssembler`, and tests the assembly/crossfade path. A second silent case provides no additional coverage.

[ASSUMPTION] Assumed `scripts/build_variants.py --check` remains a soft-skip on this branch since no workflow variants are currently committed.
