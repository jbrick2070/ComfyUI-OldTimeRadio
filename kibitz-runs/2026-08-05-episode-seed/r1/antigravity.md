VERDICT: build-ready as-is? no.
The document correctly diagnoses `meta.episode_seed` un-stamped on legacy script lanes, but stalls on hand-wringing over non-existent key collisions and over-engineered re-pinning instead of placing one clean stamp site.

MUST-FIX BEFORE BUILD:
1. [## What I want the panel to answer - Q1] Dual-Seed Generation Risk.
   Lifting the `CONTENT_OWNED` gate in Option (b) (`nodes/OTR_LedgerScriptWriter.py:6041`) unconditionally would call `_resolve_cast_rng_seed()` a second time at `nodes/OTR_LedgerScriptWriter.py:6056`, minting a second OS-entropy seed for legacy lanes that already minted `cast_seed` at `nodes/OTR_LedgerScriptWriter.py:3944`. Option (a) (`:4087`) places the stamp inside `lock_cast`, missing any lane flow that bypasses cast locking.
   Fix: Stamp `meta["episode_seed"] = int(cast_seed)` immediately after `_resolve_cast_rng_seed()` runs at `nodes/OTR_LedgerScriptWriter.py:3944`. Leave the fallback check `if meta.get("episode_seed") is None:` at `nodes/OTR_LedgerScriptWriter.py:6055` so content-owned lanes (which skip line 3944) continue to stamp their seed at the tail without re-seeding legacy lanes.

2. [## Every consumer of the frozen value (grepped, non-test) / ## What I want the panel to answer - Q5] Consumer Seed Priority Inconsistency.
   `nodes/otr_credits_roll.py:313-314` prioritizes `meta.cast_contract.cast_seed` over `meta.episode_seed`, whereas `nodes/cast_lock.py:502`, `nodes/_otr_voice_node_common.py:427`, `nodes/stable_audio_theme.py:265`, and `nodes/_otr_audio_engines/eng_kokoro.py:93` all consume `meta.get("episode_seed")`. If `cast_seed` and `episode_seed` ever diverge, the credits roll will display a seed receipt that contradicts the voices and music bed actually rendered.
   Fix: Standardize `nodes/otr_credits_roll.py:313-314` to read `meta.get("episode_seed")` as primary, or enforce `meta["episode_seed"] == meta["cast_contract"]["cast_seed"]` at the writer boundary.

SHOULD-FIX:
1. [## What I want the panel to answer - Q3] Test Suite Fixture Audit.
   Unit test fixtures in `tests/test_cast_lock.py:56` and `tests/test_credits_roll_spec.py:61` pass explicit `episode_seed` values, but full script-writer integration tests (e.g. `tests/test_fable2_tail_context.py:270`) [ASSUMPTION] must be verified to ensure they do not assert fixed downstream voice choices or music bed outputs when `meta.episode_seed` varies.
   Fix: Verify test suite behavior under dynamic seeds, and explicitly pass `OTR_CAST_SEED` env var or explicit `episode_seed` in fixtures where byte-identical output is asserted.

OPTIONAL / NICE-TO-HAVE:
- [## The seed already exists, one key away] Add a log line at `nodes/OTR_LedgerScriptWriter.py:3948` explicitly logging `meta.episode_seed` alongside `cast_seed` for clear ledger traceability.

CUT THESE (scope / over-engineering):
1. [## What I want the panel to answer - Q2] Replay Claim Collision Anxiety.
   Safe to cut because it rests on a false premise. `nodes/cast_lock.py:340-341` explicitly inspects `meta.get("cast_contract", {}).get("cast_seed")` for replay claims, never `meta["episode_seed"]`. The two keys are isolated in code, so stamping `meta["episode_seed"] = int(cast_seed)` carries zero risk of triggering CastLock replay logic.

2. [## What I want the panel to answer - Q4] Redundant Seed Pinning Mechanism.
   Safe to cut as over-engineering. `_resolve_cast_rng_seed()` at `nodes/OTR_LedgerScriptWriter.py:3935-3938` already checks `os.environ.get("OTR_CAST_SEED")` to force a fixed seed for bake-off/AB testing and byte-identical C7 runs. Designing a new pinning mechanism or graph widget is redundant.
