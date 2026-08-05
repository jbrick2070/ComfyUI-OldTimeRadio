VERDICT: yes-with-fixes. Literal implementation has one build-breaking scope error, loses a new ledger contract, and specifies an invalid A/B control.

MUST-FIX BEFORE BUILD:

1. [Track 3 Step 4] `floor` is scoped incorrectly. The plan places `floor = _min_tier_pool()` inside `_ladder_pick` but uses it earlier in `stable_cast_seed` at `nodes/_otr_voice_bank.py:394-402`; literal implementation raises `NameError`. Compute `floor` once before line 394, then close over that value in `_ladder_pick` at `nodes/_otr_voice_bank.py:424-434`.

2. [Track 1 Step 4(iv-v); LEDGER COMPLETENESS] `cast_source_contract` cannot reach the durable ledger as specified. `_otr_casting.lock_cast` returns a private `cast_meta` at `nodes/_otr_casting.py:1770-1783`; `OTR_LedgerScriptWriter` copies selected fields only at `nodes/OTR_LedgerScriptWriter.py:4069-4078`. The plan neither copies `cast_source_contract` there nor passes the resolver’s evidence/tier/roster fields through the proposed `source_character_genders` argument. Pass one rich source contract into `lock_cast`, derive the pin map from it, return it in `cast_meta`, and explicitly persist `meta["cast_source_contract"] = cast_meta["cast_source_contract"]` beside line 4073. Stamp a normalized empty contract on invention lanes.

3. [Track 2 Steps 2-3-6-9] The control assertion is wrong. With only `OTR_PORTRAIT_REFERENCE=0`, Step 2’s identity seed remains enabled, so Step 3 assigns `portrait_anchor_mode="seed"`; it cannot be `""` as Step 9 requires. Make the reference-only A/B assert treatment=`reference_latent`, control=`seed`. A separate old-behavior arm may set both `OTR_PORTRAIT_REFERENCE=0` and `OTR_PORTRAIT_IDENTITY_SEED=0` and assert `""`.

4. [Track 2 Step 9; CONFIGURATION SEQUENCING] `scripts/otr_canonical_api_run.py` only loads/applies/submits the workflow (`scripts/otr_canonical_api_run.py:132-165`); it cannot change environment variables inside the resident ComfyUI process. Run each A/B arm under a fresh selective reset and server boot with its environment fixed before launch. Otherwise the control can reuse the treatment process environment and cached node outputs.

5. [Track 2 Steps 3+6; LEDGER COMPLETENESS] `portrait_anchor_mode` currently has two planned writers: Step 3 stamps seed mode and Step 6 later overwrites it. Cache hits start by cloning an old row at `nodes/otr_image_gen_dispatcher.py:1026-1041`, so conditional stamping can retain stale values. Compute one final mode before the cache branch—`reference_latent` if a reference resolved, otherwise the seed mode—and unconditionally stamp both `derived_from_portrait_hash` and `portrait_anchor_mode`, including explicit empty strings, on cache-hit and fresh-render scene/jump rows.

6. [Merged Track 1 Step 6 + Track 3 Step 7; Q8; LEDGER COMPLETENESS] The deterministic any-reference choice has no shared callable owner. It exists only inside `_resolve_clone_ref_path` at `nodes/_otr_voice_node_common.py:109-127`, while CastLock currently only reports and continues at `nodes/cast_lock.py:616-620`. Extract a shared quality-filtered selector returning the `VoiceBankEntry`; use it from both CastLock and `_resolve_clone_ref_path`. Initialize `voice_cast_fallback=""` for every non-announcer row before the policy branch, including `preserve_ledger`, and set `gender_unservable` only on the fallback. Rewrite Track 3’s anyref test to omit `voice_ref_id`; the direct anyref path remains reachable for legacy/preserve rows, while the merged auto-registry test covers the stamped-ID path.

SHOULD-FIX:

1. [Q1; Track 1 Step 4 -> Track 3 Step 4] The order does not invalidate the floor invariant. `_ladder_pick` constructs its pool from bank entries and used-reference state at `nodes/_otr_voice_bank.py:424-433`; demand distribution does not alter bank cardinality, and the gender-only terminal tier remains available. Treat the male-heavy collision rate as telemetry, not a build blocker. Add one pinned, male-heavy integration test exercising used IDs and the canonical reuse setting.

2. [Q5; Track 3 Step 1] Import `_SEEDED_ANNOUNCER_ENGINES` through CastLock’s existing function-local import at `nodes/cast_lock.py:493-496`. `_otr_voice_bank.py` does not import `cast_lock.py`, so no cycle exists, but retaining the lazy-import boundary matches the current startup design. The behaviorally duplicated seeded-engine condition is only `nodes/_otr_voice_bank.py:717` and `nodes/cast_lock.py:527`; the dropdown tuple at `nodes/cast_lock.py:47-48` is a different contract.

OPTIONAL / NICE-TO-HAVE:

1. [Q2] Trailing dataclass appends are safe. `EnsembleSlot` has five required positional fields followed by defaulted `age_band` at `nodes/_otr_casting.py:535-552`; only `tests/test_cast_llm_naming.py:141` and `:151` use five positional arguments. All other `EnsembleSlot` sites and every `CastSlot` construction at `nodes/_otr_casting.py:1123,1315` and `tests/test_otr_casting.py:412` use keywords. No scripts construction exists.

2. [Q3] The complete non-temporary caller set is `nodes/_otr_shakespeare_sources.py:487`, `tests/test_shakespeare_sources.py:177`, `nodes/_otr_public_domain_sources.py:518`, and `tests/test_public_domain_sources.py:129`. Keyword-only `text_path=None` preserves tests, provided the two production calls explicitly pass `text_path=text_path`.

3. [Q4] The sidecar join resolves correctly. The active manifest is selected at `nodes/story_packs/banks.json:146`; all fourteen `text_path` values point into `sources/` (`config/source_banks/shakespeare/curated_scenes.sample.json:25-324`), and matching stem-identical `.provenance.json` files exist there. The writer uses the same stem rule at `scripts/otr_fetch_public_domain.py:244-246`.

4. [Q6] Confirmed: a branch using `base` before its assignment at `nodes/otr_image_gen_dispatcher.py:159` raises `UnboundLocalError`. The caller at `:998` is outside the path-guard `try/except` ending at `:997`, so dispatch dies outright. Placement after the mode gate at `:160-161` is required.

5. [Q7] Resolving the reference immediately before seed/key construction is safe. `engine_id` is already final at `nodes/otr_image_gen_dispatcher.py:939`, prompt validation has completed by `:997`, and the cache-hit path exits at `:1047`; moving resolution later would make the anchor unavailable to the cache key.

CUT THESE (over-engineering):

1. [Q2] Cut constructor adapters or positional-call migrations. A trailing defaulted field preserves every current call.

2. [Q3] Cut compatibility wrappers for source-meta functions. The keyword-only default plus two explicit production call updates covers the complete caller set.

3. [Q5] Cut any cycle-breaking module or new registry for the announcer constant. Reusing the existing local `_otr_voice_bank` import is sufficient.
