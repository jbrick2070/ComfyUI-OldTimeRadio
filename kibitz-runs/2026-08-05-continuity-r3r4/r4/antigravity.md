VERDICT: yes-with-fixes
Plan is structurally sound, grounded, and near-locked; 3 minor field/test specification fixes required before code lock.

MUST-FIX BEFORE BUILD:
1. [Track 1 Step 6 / Track 3 Step 7] **Merged Fallback Test Shadowing**: Landing Track 1 Step 6 causes `CastLock` to stamp `voice_ref_id` on `VoiceCastingError` at `nodes/cast_lock.py:616-620`, which causes downstream `_resolve_clone_ref_path` at `nodes/_otr_voice_node_common.py:91-95` to hit the `vrid` lookup first. This shadows the `anyref` fallback path, making Track 3 Step 7's unit test exercise dead code if run via full `CastLock`.
   - **Fix**: Specify in the merged step that the `anyref` fallback test must explicitly invoke `_resolve_clone_ref_path(..., vrid=None)` directly, while the `CastLock` test asserts `voice_cast_fallback == 'gender_unservable'` on the stamped row.
2. [Track 1 Step 6] **Unowned Ledger Field on Normal Cast Path**: `voice_cast_fallback` is added on the `VoiceCastingError` fallback path in `nodes/cast_lock.py`, but is left uninitialized on normal cast paths. Under the operator rule that every ledger field must be explicitly owned on all paths, missing key on normal rows creates ambiguity.
   - **Fix**: Explicitly set `voice_cast_fallback: None` on normal cast row stamping in `CastLock` so every row in `meta['cast_voice_slots']` defines the key.
3. [Track 2 Step 2 & 6] **Unowned Portrait Hash Fields on Cache-Hit & Non-Character Paths**: Track 2 introduces `derived_from_portrait_hash` and `portrait_anchor_mode` in `nodes/otr_image_gen_dispatcher.py`, but does not specify their values on cache-hit paths (`:1026-1044`) or non-character still requests (`kind != 'scene_character'`).
   - **Fix**: Require `derived_from_portrait_hash: ""` (or `None`) and `portrait_anchor_mode: "none"` to be explicitly set on all cache-hit reconstructed records and non-character still rows.

SHOULD-FIX:
1. [Track 1 D01] **Correct RNG Tally Citation in Plan Text**: D01 rejection text claims `_plan_gender_distribution` consumes 0 draws at count 0/1, 2 at 2/3, 4 at 4/5. Real `random.shuffle` consumption at `nodes/_otr_casting.py:615` is 0 at count 0/1, 3 at count 2/3, 9 at count 4, 11 at count 5.
   - **Fix**: Correct the numbers in D01 documentation text. The conclusion (override-in-place avoids stream alteration) remains 100% valid and unchanged.
2. [Track 3 Step 4] **Male-Shifted Adaptation Demand Collision Monitoring**: Landing Track 1 Step 4 pins male-heavy Shakespeare characters, increasing male pool demand against `indextts2` (17 male vs 23 female refs in `config/voice_reference_bank.json`).
   - **Fix**: Add a run-time log check during adaptation test suites to monitor male-voice collision rates under `floor=2`.

OPTIONAL / NICE-TO-HAVE:
- Document in `PROD_BUG_LOG.md` the latent Bark replay divergence at seed 424242 (`nodes/_otr_casting.py:1313`) as planned.

CUT THESE:
1. **Track 3 Step 5 (Kokoro char_voice / lang_code / role leak)**: Safe to cut because `config/audio_engine_profiles.yaml:154` locks `allowed_voice_banks: [kokoro_builtin]` while canonical workflow node 80 ships `voice_bank='default'`. Kokoro char_voice is completely unreachable in production.
2. **Track 3 Step 8 (Timbre Synonyms)**: Safe to cut because `floor=2` at `nodes/_otr_voice_bank.py:432` reduces size-1 first-accepted tiers from 3 to 0 across the entire bank without needing vocabulary expansion.
3. **Track 3 Step 6 (Announcer Drift Guard as separate step)**: Safe to fold into Step 1's commit as 3 test assertions rather than carrying it as an independent pipeline step.
4. **Track 2 Step 8 (Flux2 Klein speculative wiring)**: Safe to hold until live A/B testing demonstrates Z-Image Turbo requires fallback.
5. **Track 1 Step 3 Ariel & Puck Hints**: Safe to cut; 2 of 42 hints with no source grounding that are better left to default rolls.

VERIFY-AT-BUILD CHECKLIST:
- [ ] `_plan_gender_distribution` RNG stream invariance with override-in-place verified over 400 seeds (`nodes/_otr_casting.py:615`, `nodes/_otr_casting.py:683`).
- [ ] Shakespeare adaptation ledgers gender correctness confirmed for MALVOLIO, MARIA, ROMEO, JULIET, MIRANDA, CELIA, ROSALIND, LEAR (`config/source_banks/shakespeare/sources/*.provenance.json`).
- [ ] Announcer rotation across Kokoro voices (`bm_george`, `bm_fable`, `bf_emma`, `bf_lily`) using `_seeded_preferred_announcer_voice_ref` (`nodes/_otr_voice_bank.py:702`).
- [ ] Scene character still seed derivation from portrait prompt hash verified (`nodes/otr_image_gen_dispatcher.py:160`).
- [ ] Reference image resolution precedes `request_cache_key` computation (`nodes/otr_image_gen_dispatcher.py:998`, `:1000`).
- [ ] Profile mapping `role_overrides.character_visual` loads correctly via `--profile` in `scripts/otr_canonical_api_run.py:156-158` (`config/profiles/widget_mapping.json:23-31`).

AUDIT OF THE 19 DISCARDED OBJECTIONS:
- **D01** (`nodes/_otr_casting.py:615`): Stated RNG draw counts (0,0,2,2,4,4) are factually wrong against current code (actual counts: 0,0,3,3,9,11), but core finding that altering `count` shifts RNG stream is factually true. Rejection disposes of objection. Override-in-place design stands. [SHOULD-FIX #1]
- **D02** (`scripts/otr_fetch_public_domain.py:324-325`, `config/source_banks/_corpus/`): Factually true. `_corpus/` contains 64 Gutenberg prose bodies and zero Folger plays; `body` is rebound to sliced scene at `:325`. Disposes of objection cleanly.
- **D03** (`config/source_banks/shakespeare/sources/midsummer__act3_scene2.provenance.json:40-66`): Factually true. All four characters have populated `roster_name` ('HERMIA','LYSANDER','DEMETRIUS','HELENA') and `description: ""`, `gender: "unknown"`. Disposes of objection cleanly.
- **D04** (`nodes/cast_lock.py:616-621`): Factually true. `VoiceCastingError` caught, logged, continued without stamp; override-in-place preserves existing unpinned rolls. Disposes of objection cleanly.
- **D05** (`nodes/_otr_casting.py:1270-1318`): Factually true. `source_bank_id` controls Lemmy exclusion with 0 RNG calls; `pick_first_last` at `:1313` consumes RNG. Action accepted, characterization corrected. Disposes of objection cleanly.
- **D06** (`midsummer__act3_scene2.provenance.json:33-37`): Factually true. `ROBIN` is absent_from_roster, `PRINCE` is unknown. Renaming yields unknown gender anyway. Disposes of objection cleanly.
- **D07** (`nodes/story_orchestrator.py:861-866`): Factually true. List at `:861` contains Bark presets (`v2/en_speaker_*`), not Kokoro. Disposes of objection cleanly.
- **D08** (`nodes/_otr_voice_bank.py:432`): Factually true. Measured 8/24, 5/24, 2/24 timbre-honouring combos; `floor=2` removes all size-1 first tiers. Disposes of objection cleanly.
- **D09** (`config/voice_reference_bank.json`, `scripts/otr_ingest_pd_voices.py:72-74`): Factually true. 2 same-human pairs have distinct `ref_path`s within each engine, requiring 12 rows. Disposes of objection cleanly.
- **D10** (`nodes/cast_lock.py:536-537`, `workflows/otr_canonical.json`): Factually true. `target_engine != announcer_engine` in canonical workflow node 80. Disposes of objection cleanly.
- **D11** (Structural invariant): Factually true. Structural invariant (no size-1 first tier) replaces brittle percentage thresholds. Disposes of objection cleanly.
- **D12** (Separability): Factually true. `floor=2` alone removes all size-1 tiers (3->0); floor-first is safe and synonyms are cut. Disposes of objection cleanly.
- **D13** (`config/profiles/widget_mapping.json:23-31`, `scripts/otr_canonical_api_run.py:156-158`): Factually true. Profile sets `role_overrides.character_visual`; canonical node 87 widget edit is unnecessary and wrong. Disposes of objection cleanly.
- **D14** (`nodes/_otr_image_engines/schemas.py:49,73,102`, `tests/test_image_platform_c1.py`): Factually true. `CanonicalImage` and `ImageRequest` constructed nowhere else in `nodes/`. Disposes of objection cleanly.
- **D15** (`nodes/_otr_image_engines/eng_google_image.py:144-151`): Factually true. Extending `_reject_reference_inputs` with `reference_image` key is 1 line. Disposes of objection cleanly.
- **D16** (`nodes.py:1885-1889`): Factually true. `width=0`/`height=768`/`crop='disabled'` preserves aspect ratio without cropping head. Disposes of objection cleanly.
- **D17** (File sizes): Factually true. Difference is decimal GB vs binary GiB confusion. Disposes of objection cleanly.
- **D18** (`nodes/_otr_image_engines/z_image_turbo.py:318-320`, `engine_registry_base.py:148-150`): Factually true. Singleton caching of `_classes` would lock out reference keys for subsequent mints. Disposes of objection cleanly.
- **D19** (`tests/test_multiclip_jump_stills.py:234-237`, `nodes/otr_image_gen_dispatcher.py:718-729`): Factually true. Jump cuts explicitly require distinct seeds across clips. Disposes of objection cleanly.

[ASSUMPTION] Assumed `voice_cast_fallback` should be explicitly `None` on normal cast rows for explicit schema cleanliness across all ledger records.
