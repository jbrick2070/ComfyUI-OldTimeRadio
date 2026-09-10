VERDICT: yes-with-fixes -- C4 production edits match the live node3->4->7 path; the proof/receipt/JSON-edit steps as written will fail at the keyboard.

MUST-FIX BEFORE BUILD:
1. [P1 bit-eq] "supplied scene waveform repeated to two channels" is ambiguous vs the chirp factory. Sequencer `_level_dialogue_clip` (`nodes/scene_sequencer.py:1094,1101` -> `_loudness_normalize_clip:508`) changes amplitude before node 4. Comparing enhance output to factory chirps fails. Fix: clone `route.values` after `run(OTR_SceneSequencer)` (the `(1,1,N)` tensor from `sequence:1204-1205`), then `torch.equal(enhanced["waveform"], torch.cat([wf, wf], dim=1))` to match `_mono_to_stereo` (`audio_enhance.py:91-95`). Assert both rates are 48000 first; if not, fail the declared assumption (SceneSequencer hardcodes 48000 at `scene_sequencer.py:983`; node 4 widget[0] is 48000 at `otr_canonical.json:755`).
2. [P1 silence factory] Zero-asset rank/duration is unspecified. Empty/`(0,)` tensors break `_extract_clips_from_audio` and assembler `max(s.shape[1])` (`scene_sequencer.py:1611`). Fix: keep chirp shapes -- `audio(1,*)` voices and `audio(2,*)` opening cue -- fill with `float32` zeros `(1,1,N)`. `_trim_trailing_silence:289-304` will keep 100 samples/clip (200-sample scene); music cues are not trimmed (`_cue_from_batch:2312`). Do not assert 2.0 s scene length. Opening 2 s zeros + 200-sample scene: `xf = min(24000, opening, 200) = 200` (`1624`); still all zeros; skip lag.
3. [P1 receipt / launcher] `check_case` currently returns `rng_seed` (`scripts/otr_canonical_audio_check.py:246`); launcher asserts it and `len(cases)==2` (`tests/test_canonical_audio_check.py:30-33`). Unspecified mixed-case keys plus reused episode id `canonical_audio_opening` (`check_case:153`) makes silence overwrite the chirp-opening WAV so post-run `master_sha256` checks fail. Fix: drop `rng_seed` everywhere; three ids `canonical_audio_opening` / `canonical_audio_no_opening` / `canonical_audio_silence_opening`; shared keys `{episode, canonical_sha256, calls, supplied_boundaries, ledger, master, master_sha256}`; chirp: float `measured_scene_offset_s`; silence: `None` plus `nonzero_samples={scene,enhanced,master}` all 0. Launcher: `len==3`, distinct labels, silence null/zeros, hashes for all; no seed.
4. [C4.4 / P1 SHA inspect] `widgets_values` at `otr_canonical.json:754-761` and `"spatial audio"` at `:3495` are unique -- good. `json.load`/`dump` of this file also rewrites `extra.ds` (`3485-3490`) and will fail "only two hunks". Fix: two byte-string replacements; do not reserialize. Assert `len(graph["links"])==62` (links start `:2984`; 62 entries). `last_link_id` is 290 (`:5`); do not use it as a count.
5. [C4.2] Remove hiss keys and `params["hiss"]` / noise block together (`audio_enhance.py:258-260,276-286`). Keys-only or block-only is `KeyError` or leftover RNG.

SHOULD-FIX:
1. [P1 tape loop] "Inside the child check" plus three `check_case` calls runs 12 enhance() invocations. Put the four-mode loop once in `main()` after load_package. Assert `count_nonzero==0` and `shape[-1]` unchanged, not merely `isfinite` -- leftover hiss is finite.
2. [P1 WorkflowValidator] Saved node 63 is `["", true, true, "", "", ""]` (`otr_canonical.json:97-104`). Empty path falls back to canonical (`_otr_workflow_validator.py:73-74,98-104`); `validate_anyway=true` actually runs the contract (`481-535`). Call `validate(str(CANONICAL), True, True)` (or empty path). `validate_anyway=False` is a silent skip (`496-502`). JSON round-trip = `loads` identity, not byte-equal `dumps`.
3. [C4.1 / C1] Same file docstring still advertises "breath buffers, BEAT/PAUSE tags" and "continuous room tone bed" (`scene_sequencer.py:8-10`). C1 locals `928-932` / `current_character_name:988,1124` are unread in this file -- delete them, and fix that header with the 1126 heading. Keep the `if segment_np is not None` body except the `env_timeline.append` at 1137.
4. [P1 launcher] `timeout=120` (`test_canonical_audio_check.py:22`) now covers three assemblies twice. [ASSUMPTION] 2-case runs fit today; bump to 180.
5. [P1 silence decode] `decoded.mean(axis=1)` (`otr_canonical_audio_check.py:212`) can hide anti-phase. Count nonzero on interleaved int16 (writer is `* 32767` no dither at `scene_sequencer.py:1757`).

OPTIONAL / NICE-TO-HAVE:
- `__init__.py:173` display name still `" Spatial Audio Enhance"`; plan only edits `:8-9`.
- `tests/test_sequencer_ledger.py:3-5` mock blurb; `test_segment_loudnorm.py:88` name `test_room_tone_below_gate_unchanged` does not call `_generate_room_tone` -- leave the test.
- `audio_enhance.py:24` "spatially widened" is covered by C4.3 header/prose.

CUT THESE:
1. Byte-equal canonical round-trip -- formatting will never match; CanonicalAudioRoute already parses live JSON.
2. Migrating `scripts/_otr_canonical_api_prompt.json` (and sibling snapshots) -- plan correctly leaves them; soaks must load `workflows/otr_canonical.json`.
3. Cache purge / replay-manifest rewrite -- `_assemble_replay` at `scene_sequencer.py:1526-1528` copies frozen WAV SHA; old masters stay noisy by design.
4. Second silent assembly / silent 1.50 s lag -- xf on a 200-sample zero scene is 200 samples, not 1.50 s; skip correlation as specified.

Grounded route: node 3 `scene_audio` link 5 -> node 4 -> link 6 -> node 7 `scene_audio`; music 282/283 untouched; episode_audio fans out 15/272 (`otr_canonical.json:631-633,745-747,780-784,840-851,870-877`). Replay `sequence:907-910` returns before the deleted bed. `_master_loudness:380-381` returns silent input unchanged. No-op branches: resample equal-rate `audio_enhance.py:75-76`; Haas `delay_samples < 1` `:106-107`; LPF `:201,434`; bass `:167-168,429`; width `:450`. Hiss is the remaining additive RNG on explicit tape.

[ASSUMPTION] no out-of-tree importer of `_generate_room_tone` (repo: define `scene_sequencer.py:634`, call `:1193`, patch `test_sequencer_ledger.py:83` only).
UNCOVERED: foley mux / video/scopes appearance; GPU room-tone path (deleted statically; check sets `CUDA_VISIBLE_DEVICES=""`).
