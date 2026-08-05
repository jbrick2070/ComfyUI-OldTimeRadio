VERDICT: build-ready as-is? yes-with-fixes. The core seed stamp at nodes/OTR_LedgerScriptWriter.py:3968 fixes the frozen voice/music defect cleanly, but requires explicit guards against content-owned seed divergence, missing test suite env overrides, and credit receipt mismatch.

MUST-FIX BEFORE BUILD:
1. [Q1 / Sequencing & Seed Ownership] nodes/OTR_LedgerScriptWriter.py:3968 vs :6078
Defect: Minting/stamping meta["episode_seed"] MUST occur at nodes/OTR_LedgerScriptWriter.py:3968 (immediately after _resolve_cast_rng_seed() at :3944) and MUST NOT be moved to candidate (b) at :6041/:6078. If candidate (b) lifts the CONTENT_OWNED gate at :6041, line :6079 invokes _resolve_cast_rng_seed() a SECOND time for legacy/inline lanes, minting a separate OS-entropy integer. This causes meta["cast_contract"]["cast_seed"] (from :3944) and meta["episode_seed"] (from :6079) to hold divergent numbers on the same run, violating single-ownership and breaking credit receipt honesty.
Fix: Retain the stamp at nodes/OTR_LedgerScriptWriter.py:3968 guarded by `if meta.get("episode_seed") is None: meta["episode_seed"] = int(cast_seed)`, and keep line :6078 as a fallback reserved exclusively for CONTENT_OWNED lanes.

2. [Q2 / Interface Contracts & Replay Guard] nodes/cast_lock.py:341-353 & nodes/OTR_LedgerScriptWriter.py:6072-6077
Defect: Promoting cast_seed into meta.episode_seed is safe, but copying episode_seed into meta.cast_contract.cast_seed on lanes that did not run the writer's seeded cast picker will cause CastLock._assign_bark_voices (nodes/cast_lock.py:341) to execute replay_voice_assignment (nodes/_otr_casting.py:1058) without a valid num_characters_request, triggering replay failure ("num_characters must be 1-6, got 0").
Fix: Enforce strict separation between generic episode seed (meta["episode_seed"], read by voice/announcer/music nodes) and replay claim (meta["cast_contract"]["cast_seed"]). Never copy meta["episode_seed"] into meta["cast_contract"] on CONTENT_OWNED or custom lanes.

3. [Q3 / Test Invariants & C7 Audio Gate] tests/test_cast_randomization.py:147-162 & nodes/OTR_LedgerScriptWriter.py:3944
Defect: With meta.episode_seed unfrozen, any unit test, harness, or C7 audio regression check that executes OTR_LedgerScriptWriter without setting OTR_CAST_SEED env var or explicit meta["episode_seed"] will now receive OS-entropy, causing announcer selection, character voices, and music beds to vary unpredictably across test runs.
Fix: Ensure all regression test suites and C7 audio gate scripts (such as tests/test_cast_randomization.py:152) explicitly set `monkeypatch.setenv("OTR_CAST_SEED", "12345")` or pass fixed seeds in ledger fixtures.

4. [Q5 / Music Bed & Downstream Integration] nodes/stable_audio_theme.py:264-266 & nodes/_otr_audio_engines/eng_kokoro.py:93
Defect: stable_audio_theme.py:265 derives `music_seed_base = _seed_to_int64("music_rng_seed_v1", coerce_int_seed(meta.get("episode_seed")))`. Previously, every episode shared the same music seed base (`coerce_int_seed(None)` = 5362114964413277558). Unfreezing episode_seed means every episode will now generate a distinct music seed. If an engine or pipeline component assumes static music parameters, it will now vary.
Fix: Verify that stable_audio_theme.py:277 logs the computed engine_seed and that OTR_CAST_SEED properly freezes both vocal and music seeds simultaneously.

SHOULD-FIX:
1. [Q4 / Bake-off & A/B Harness Pinning] scripts/bark_preset_audition.py & scripts/otr_talking_radio_probe_eval.py
Defect: Model evaluation and bake-off harnesses comparing voice engines across arms will receive different voice/music seeds per arm if episode_seed varies dynamically.
Fix: In evaluation harnesses, set `os.environ["OTR_CAST_SEED"]` or explicitly populate `meta["episode_seed"]` to hold seed constant across comparative arms [ASSUMPTION: bake-off harnesses expect identical seed inputs].

2. [Credits Roll Receipt Alignment] nodes/otr_credits_roll.py:313-318
Defect: nodes/otr_credits_roll.py:313 checks `(meta.get("cast_contract") or {}).get("cast_seed", meta.get("episode_seed"))`. If cast_seed and episode_seed ever diverge, credits display cast_seed while CastLock and music read episode_seed.
Fix: Add a warning in nodes/otr_credits_roll.py:315 if both keys exist but hold unequal integer values.

OPTIONAL / NICE-TO-HAVE:
1. Add an explicit debug log in nodes/_otr_voice_node_common.py:427 when coerce_int_seed(meta.get("episode_seed")) falls back to the default constant for legacy/unseeded ledgers.

CUT THESE (over-engineering):
1. Candidate (b) in Q1 (lifting the CONTENT_OWNED gate at nodes/OTR_LedgerScriptWriter.py:6041 for all lanes): Safe to cut because calling _resolve_cast_rng_seed() in the writer tail duplicates seed generation for legacy/inline lanes, creating divergent seeds and breaking single-ownership.
2. Creating a separate top-level `music_seed` key in meta: Safe to cut because deriving `music_rng_seed_v1` from `episode_seed` via `_seed_to_int64` in nodes/stable_audio_theme.py:265 provides clean domain isolation under a single episode entropy root.
