VERDICT: yes. The plan has converged end-to-end; boundaries, exact file edits, test updates, and validation contracts are fully specified and code-grounded.

MUST-FIX BEFORE BUILD:
None — plan converged.

SHOULD-FIX:
1. [Section 4: P1] Namespace consistency for ledger probe in audio check: The plan specifies checking `in_flight_ledger_path() is None` before test cases in `scripts/otr_canonical_audio_check.py`. Ensure the implementor imports `_otr_ledger` via `importlib.import_module(package.__name__ + ".nodes._otr_ledger")` rather than top-level `nodes._otr_ledger` to maintain the exact package-qualified isolation mandated in P3 step 2.
2. [Section 4: P1] Explicit silent-case tensor comparison reference: When comparing clean AudioEnhance output against SceneSequencer output on the silent case, confirm the comparison uses the post-sequenced waveform (200 samples after `_trim_trailing_silence`) rather than the raw 48000-sample input factory tensor. (Correctly documented in Section 4 line 160, but implementor should avoid referencing the factory input directly).

OPTIONAL / NICE-TO-HAVE:
- [Section 1: C4 / Section 4: P1] Docstring in `nodes/audio_enhance.py:AudioEnhance` can explicitly note that `lpf_cutoff_hz=0.0` disables filtering entirely, matching the tooltip update.

CUT THESE:
None. All proposed removals (C1 write-only stores, C2 duplicate line-ID check in per-line invariants, C3 unused model acquisition in freeze cascade, C4 roomtone generator and tape hiss block) are well-bounded, justified, and confirmed to have no live downstream consumers.

VERIFY-AT-BUILD checklist:
1. [Section 3 / Section 4: P0] Recapture the exact 54 failing node IDs baseline at implementation HEAD before making runtime edits.
2. [Section 4: P1] Confirm canonical JSON SHA-256 changes only in P1 (node 4 widgets `[48000, 0.0, 0.0, 0.0, 0.0, "off"]` and description) and remains unchanged across P2 and P3.
3. [Section 4: P1] Confirm two independent process runs of the unseeded clean audio check produce byte-identical WAV hashes for chirp cases.
4. [Section 4: P1] Confirm `scripts/build_variants.py --check` passes cleanly (soft-skips when no variants are committed).
5. [Section 4: P3] Confirm `test_canonical_freeze_without_acquisition_in_fresh_process` fails fast on unedited C3 code (proving the poison acquisition sentinel works) and passes after C3.
6. [ASSUMPTION] Any unverified external log scrapers or third-party workflows relying on the old `"duplicated"` line-ID substring or noisy audio defaults will need updating to the G8 summary format and clean audio defaults.
