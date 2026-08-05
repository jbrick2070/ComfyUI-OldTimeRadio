VERDICT: build-ready as-is? yes-with-fixes. Stamping meta["episode_seed"] at line 3968 fixes the frozen voice/music defect, but Candidate (b) in Q1 must be explicitly rejected to prevent double-stamping with mismatching seeds, and missing episode_seed fallbacks require log warnings.

MUST-FIX BEFORE BUILD:
1. [Section: Q1 / Section 3] Reject Candidate (b) (lifting `CONTENT_OWNED` gate at `nodes/OTR_LedgerScriptWriter.py:6041`).
   Defect: Lifting the `CONTENT_OWNED` gate in `_run_writer_tail` at line 6041 causes legacy/inline script writer runs to execute `_resolve_cast_rng_seed()` a second time at line 6079. This generates a second OS-entropy seed distinct from `cast_contract.cast_seed` (generated upstream at line 3944), producing an internal schema mismatch where `meta["episode_seed"] != meta["cast_contract"]["cast_seed"]`. Voice selection (`nodes/cast_lock.py:502`) and music generation (`nodes/stable_audio_theme.py:265`) would execute under a seed disconnected from character casting.
   Fix: Implement Candidate (a) in `nodes/OTR_LedgerScriptWriter.py:3967-3968` by adding `if meta.get("episode_seed") is None: meta["episode_seed"] = int(cast_seed)` immediately after `cast_seed, cast_seed_source = _resolve_cast_rng_seed()` at line 3944. Do NOT modify the `CONTENT_OWNED` gate at line 6041 in `_run_writer_tail`.

2. [Section: Q2 / Section 4] Null / Missing Meta Fallback Warning in Voice Consumers.
   Defect: In `nodes/cast_lock.py:501-502` and `nodes/_otr_voice_node_common.py:426-427`, missing `meta["episode_seed"]` causes `coerce_int_seed(None)` in `nodes/_otr_voice_node_common.py:190` to silently return constant `5362114964413277558`. Legacy script files or unit test fixtures missing `episode_seed` will silently continue drawing identical announcer (`bf_emma`) and character voice selections without warning.
   Fix: In `nodes/cast_lock.py:502` and `nodes/_otr_voice_node_common.py:427`, log a warning when `meta.get("episode_seed")` is `None` before delegating to `coerce_int_seed(None)` so silent fallback to constant `5362114964413277558` is explicitly reported in execution logs.

SHOULD-FIX:
1. [Section: Q4 / Section 5] Explicit Seed Pinning in Evaluation Harnesses.
   Defect: Audition scripts and test harnesses (e.g. `scripts/bark_preset_audition.py`, `scripts/otr_ltx_mad.py`) calling `OTR_LedgerScriptWriter` without explicitly setting `OTR_CAST_SEED` env var or `meta["episode_seed"]` will now receive OS-entropy variation across runs, breaking comparative benchmark reproducibility [ASSUMPTION: harnesses expect deterministic script output unless configured otherwise].
   Fix: Ensure benchmark harnesses set `os.environ["OTR_CAST_SEED"]` or pass `meta={"episode_seed": <int>}` when deterministic multi-arm evaluation is required.

2. [Section: Q5 / Section 5] Music Bed Seed Domain Coercion Consistency.
   Defect: `nodes/stable_audio_theme.py:265` derives `music_rng_seed_v1` using `coerce_int_seed(meta.get("episode_seed"))`. If `episode_seed` is passed as a string representation of an integer vs a raw int, `coerce_int_seed` returns `int`, but un-coerced dict access in legacy paths could pass raw strings into `_seed_to_int64`.
   Fix: Ensure `nodes/stable_audio_theme.py:265` explicitly wraps `meta.get("episode_seed")` in `coerce_int_seed` before passing to `_seed_to_int64`.

OPTIONAL / NICE-TO-HAVE:
1. [Section: Section 4] Add a unit test assertion in `tests/test_fable2_tail_context.py` verifying `meta["episode_seed"] == meta["cast_contract"]["cast_seed"]` on all generated legacy ledgers to prevent future seed drift.

CUT THESE (over-engineering):
1. Adding a new `OTR_EPISODE_SEED` environment variable alongside `OTR_CAST_SEED`.
   Why safe to cut: `OTR_CAST_SEED` in `nodes/OTR_LedgerScriptWriter.py:1364` already overrides `_resolve_cast_rng_seed()`. Adding a second environment variable creates redundant configuration paths.
