VERDICT: yes-with-fixes. The core seed unpinning logic correctly restores voice and music variety, but line 3977 in `nodes/OTR_LedgerScriptWriter.py` introduces a crash-inducing `ValueError` when `episode_seed` is passed as an empty string.

MUST-FIX BEFORE BUILD:
1. [stamp-site-robustness] `nodes/OTR_LedgerScriptWriter.py:3975-3977`: `int(meta["episode_seed"])` causes an unhandled `ValueError` crash if `meta["episode_seed"]` is an empty string `""` or non-numeric string (e.g. from UI widget drift or uninitialized metadata).
   Fix: Import `coerce_int_seed` from `nodes._otr_voice_node_common` (or use `try/except (ValueError, TypeError)`) to safely evaluate `existing_seed = coerce_int_seed(meta.get("episode_seed"), default=None)`. If `existing_seed is None`, stamp `meta["episode_seed"] = int(cast_seed)`; otherwise compare `existing_seed != int(cast_seed)`.
2. [writer-tail-empty-seed] `nodes/OTR_LedgerScriptWriter.py:6098`: `if meta.get("episode_seed") is None:` on `CONTENT_OWNED` lanes fails to catch an empty string `""`, leaving `""` in `meta["episode_seed"]` which causes downstream voice/announcer engines to fall back to unseeded selection.
   Fix: Change line 6098 to `if not meta.get("episode_seed"):` or validate integer coercion.

SHOULD-FIX:
1. [cast-lock-warning-gate] `nodes/cast_lock.py:502`: `if meta.get("episode_seed") is None:` skips the absence warning when `meta["episode_seed"]` is `""`, while `coerce_int_seed("")` silently returns the default constant `5362114964413277558`.
   Fix: Change line 502 to `if meta.get("episode_seed") in (None, ""):`.
2. [credits-roll-provenance-check] `nodes/otr_credits_roll.py:313-315`: `seed = (meta.get("cast_contract") or {}).get("cast_seed", meta.get("episode_seed"))` prefers `cast_seed` over `episode_seed` without checking if both exist but diverge.
   Fix: Log a warning at line 315 if both `cast_seed` and `episode_seed` exist in `meta` and `int(cast_seed) != int(episode_seed)`.

OPTIONAL / NICE-TO-HAVE:
1. [logging-seed-source] `nodes/OTR_LedgerScriptWriter.py:3989-3993`: Add `episode_seed_source` ("minted_os_entropy" vs "caller_supplied") to the info log to simplify troubleshooting during production runs.

CUT THESE:
1. [unconditional-tail-re-mint] `nodes/OTR_LedgerScriptWriter.py:6084-6100`: Do NOT remove the `CONTENT_OWNED` gate to re-mint a seed in the tail for legacy/inline lanes. Re-minting in the tail would generate a second OS-entropy seed, causing `meta["cast_contract"]["cast_seed"]` and `meta["episode_seed"]` to diverge. Safe to cut because stamping at mint (`nodes/OTR_LedgerScriptWriter.py:3975`) fully handles legacy/inline lanes.

VERIFY-AT-BUILD:
1. Verify `nodes/OTR_LedgerScriptWriter.py:3975-3977` handles `meta={"episode_seed": ""}` and `meta={"episode_seed": None}` without raising `ValueError`.
2. Verify Kokoro announcer voice selection (`nodes/_otr_audio_engines/eng_kokoro.py:93`) and character voice draws (`nodes/cast_lock.py:515`) vary across 10+ unseeded episode runs.
3. Verify `stable_line_seed_v1` (`nodes/_otr_resolved_request.py:265`) and `music_rng_seed_v1` (`nodes/stable_audio_theme.py:265`) vary per episode when `episode_seed` is unpinned.
4. Verify explicit seed overrides (`meta={"episode_seed": 42}` or `OTR_CAST_SEED=42`) produce byte-identical determinism for C7 tests.

[ASSUMPTION] Assumed caller metadata passed into `OTR_LedgerScriptWriter` may contain empty strings `""` due to UI widget defaults or unpopulated JSON keys based on historic bug logs (BUG-LOCAL-253, BUG-LOCAL-742).
