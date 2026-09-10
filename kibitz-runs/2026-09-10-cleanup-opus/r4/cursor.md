VERDICT: yes-with-fixes -- C3/C4 runtime chains are coherent, but P0 still forbids the C4 canonical edit that P1 requires, so a builder can ship colored production audio while believing they followed the plan.

MUST-FIX BEFORE BUILD:
1. [P0 vs C4.4 / P1] P0 still says "Do not edit canonical JSON: there is no intended node/widget/link change. Record byte-identical canonical before/after." That is leftover C1-C3 text and contradicts C4.4, P1 ("Canonical SHA MUST change"), and P4. Confirmed: `workflows/otr_canonical.json:754-761` is `[48000, 0.3, 0.8, 0.15, 16000, "subtle"]`; `extra.info.description:3495` still says "spatial audio". Smallest fix: P0 reads "C2/C3: canonical bytes identical to the post-C4 SHA. C4+C1 alone may change node 4's six widget values and the one info.description phrase; stop on any other JSON hunk." Do not leave both instructions live.

SHOULD-FIX:
1. [C3 vs tests/test_llm_slot_sweep.py:62-76] Plan cites `slot_sweep_accounting.json` (file does not exist). Live floor is `_MIN_EXPECTED_CALL_SITES = 12` in `tests/test_llm_slot_sweep.py`; scanner is `tests/_s28_llm_slot_sweep.py` (`request_slot` is a counted Call; `OTR_LedgerFreezeCascade.py` is not exempt). C3 focused pytest row omits that module. Replace the phantom path; add `tests/test_llm_slot_sweep.py` to the C3 command row. verify: actual site count at implementation HEAD (plan's 43 is unverified here).
2. [P1 command table vs CLAUDE.md §0] C4 is the only canonical-JSON chunk. Focused C4+C1 args omit `tests/test_widget_value_alignment.py`, `tests/test_canonical_widget_input_parity.py`, and `tests/test_workflow_link_target_indexes.py`. C4 does not reindex slots (`otr_canonical.json:671-737` widgets stay), so this is not a link-break, but the four-way gate belongs on the chunk that touches the graph. Add those three plus the already-listed `build_variants.py --check`.
3. [C4.5 vs `__init__.py:173`] Prose at 8-9 changes; display mapping stays `" Spatial Audio Enhance"`. One explicit "leave the mapping string" clause so a builder does not rename the registration.

OPTIONAL / NICE-TO-HAVE:
- Pin helper defaults `_haas_delay(delay_ms=0.4)` / `_stereo_decorrelate(amount=0.15)` only in comments; public path already passes explicit kwargs (`audio_enhance.py:440-451`).
- After C4, next `otr_canonical_api_run.py` dump will refresh `scripts/_otr_canonical_api_prompt.json:84-88`; do not treat the stale dump as a second graph (already out of scope).

CUT THESE:
1. P0's "byte-identical canonical before/after" sentence once the exception in must-fix 1 exists -- it is a contradiction, not extra work.
2. Any second silent assembly / 1.50 s silent lag (already forbidden at P1). Keep one silence-opening case.
3. Migrating noncanonical prompt snapshots. Live API path loads `workflows/otr_canonical.json` (`scripts/otr_canonical_api_run.py:6,25`).
4. Section 5 Kibitz-call arithmetic. It is roster bookkeeping, not a builder contract.

VERIFY-AT-BUILD:
- Recapture full-suite failing node-ID set at implementation HEAD (P4; prior 54 IDs / 14166 collect is pre-C4).
- `len(graph["links"])==62`; do not use `last_link_id` 290 (`otr_canonical.json:5`). verify: exact length.
- Post-C4 two-process WAV identity with seeds removed. Fail = leftover RNG, not a hash compare to colored receipts.
- Silence: all-channel int16 zeros; `_trim_trailing_silence` (`scene_sequencer.py:289-304`) and `_loudness_normalize_clip` peak<1e-6 (`508-524`) and `_master_loudness` peak<1e-8 (`380-381`) already no-op on zeros -- assert actual shapes, not 2 s.
- Clean enhance: SceneSequencer output is mono `(1,1,N)` (`scene_sequencer.py:1204`); `torch.cat([scene_copy, scene_copy], dim=1)` is the right stereo check at 48000 Hz.
- Tape modes: zeros stay zeros after hiss deletion (`audio_enhance.py:257-286`); saturation/wow remain for explicit tape (operator-clean defaults, not a hiss leak).
- C3: `require_model` nonempty/strip only (`_otr_model_inputs.py:79-85`); replay/no-ledger return before it (`OTR_LedgerFreezeCascade.py:215-263`); `request_slot`/`make_generate_fn` unused after edit; `run_freeze_cascade` first positional still required (`_otr_freeze_cascade.py:714-716`); no `generate_fn(` in that module; cleanup only `del generate_fn` (`676-692`). Poison `importlib.import_module(package.__name__ + ".nodes._otr_model_loader")`, not stock `nodes`.
- Link 115 is `[115, 1, 4, 62, 4, "STRING"]` (`otr_canonical.json:3050-3056`); node 62 widgets `[true, true]` (`1174-1177`); bind only `widget`-keyed inputs.
- `freeze_unload_ok=True` means unload did not raise; empty cache is a no-op (`_otr_model_loader.py:1718-1729`).
- Bank cleanup literals: original `retired_no_content_policy`; scifi_news_pro `not_applicable_content_owned`. Verdict in `{frozen_clean, frozen_with_warns}`.
- Out of scope, do not "verify" by running: shot-lock admission of `unused-model-boundary` outside `OTR_TEST_MODE`; external log scrapers; private `_generate_room_tone` importers.

Grounded non-defects (do not reopen): C1 names are store-only (`scene_sequencer.py` grep: `breath_ms`/`current_character_name` have no reads). C4 roomtone caller is `sequence:1193` only. C3+C4 order: P0.5 harness, C4+C1 (canonical SHA changes), C2, C3 (canonical SHA frozen; child reuses `load_package()`). Public tape + asymmetric stereo probes before `pl.new_ledger` match `in_flight_ledger_path` gating (`audio_enhance.py:489-490`). Dual-mono chirp cannot detect width; keep the stereo default probe.

[ASSUMPTION] Astra fixture still yields a successful freeze on `original`/`scifi_news_pro` at implementation HEAD; the new child test is not already-passed evidence.
