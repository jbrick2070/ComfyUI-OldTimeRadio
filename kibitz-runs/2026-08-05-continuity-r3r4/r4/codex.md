VERDICT: no. The design largely converged, but unresolved ledger ownership, contradictory portrait-mode semantics, an incomplete Shakespeare live command, and an impossible 42/42 gender-supplement criterion still block the build.

MUST-FIX BEFORE BUILD:

1. [Track gender-voice Step 4(iv-v)] `cast_source_contract` cannot be produced as specified. `lock_cast` currently receives names and `source_bank_id`, and the plan adds only `source_character_genders`; it cannot stamp the promised per-name `evidence`, `tier`, and `roster_name` (`nodes/_otr_casting.py:1468-1481`). Pass one complete normalized contract into `lock_cast`, derive `gender_by_name` from it, and have `lock_cast` always emit exactly one stable shape—empty values on invention/unpinned paths.

2. [Track portrait Steps 2, 3, 6, 9] The ledger contract is contradictory. Step 2 says all non-identity paths return `(seed, '')`, but its test requires malformed/undefined `"fixed-mode base"` text. Step 3 assigns `portrait_anchor_mode='seed'`; Step 6 overwrites it with `'reference_latent'`; Step 9 incorrectly expects `''` in the reference-disabled control even though the seed feature remains enabled. Define one final writer: `reference_latent` when a reference was consumed, `seed` when only portrait-seed identity was used, otherwise `''`; fixed mode returns `(base, '')`. Stamp `derived_from_portrait_hash` only when the rendered/cache-hit row is actually bound to that portrait hash, avoiding false provenance on old cache hits. Current cache hits clone old rows at `nodes/otr_image_gen_dispatcher.py:1002-1047`, while fresh rows are built at `:1170-1197`.

3. [Track portrait Step 9] The named command does not select Shakespeare. `--profile otr_w45_still_flat` changes engine roles but does not set `source_bank`, `source_ref`, or target words (`config/profiles/otr_w45_still_flat.json:10-24,30-35`); canonical node 1 remains `scifi_news`. Add explicit `--source-bank shakespeare`, `--words`, and a fixed `OTR_LedgerScriptWriter.source_ref` through `--set`; the runner exposes these controls at `scripts/otr_canonical_api_run.py:172-190`. Pin a source/word budget known to yield one character on at least three beats, or fail preflight before spending the A/B.

4. [Track portrait Step 9] The A/B is not controlled. An environment change cannot reach an already-running ComfyUI process, and `OTRImageGenDispatcher` has no `IS_CHANGED` despite depending on the new environment flags (`nodes/otr_image_gen_dispatcher.py:1412-1461`). Reset and reboot the server separately for each arm, hold the source snapshot, prompts, seeds, portrait bytes, and model configuration equal, and assert those hashes match before comparing identity. Otherwise the A/B confounds the reference with a different script or portrait.

5. [Track gender-voice Step 3] “12 supplemented names” and “all 42 resolve” conflict with the cuts. The twelve include Ariel and Puck, but the plan also cuts their entries and supplies no gender/evidence values. The manifest really contains 42 hints, including Ariel, Puck, and Don Pedro (`config/source_banks/shakespeare/curated_scenes.sample.json:20-25,40-46,86-92,179-184`). Make the supplement ten entries and assert 40/42 resolved with Ariel/Puck explicitly unknown, or provide operator-approved evidence-backed values. Do not leave a 42/42 test no valid implementation can satisfy.

6. [Track gender-voice Step 6 / merged Track voice-variety Step 7] `voice_cast_fallback` is defined only on the exception path, violating the stated every-path ledger rule. Make `_stamp` write `voice_cast_fallback=''` normally and `'gender_unservable'` on the fallback (`nodes/cast_lock.py:603-624,686-698`). Preserve direct coverage of the legacy any-reference resolver with a row lacking `voice_ref_id`; after CastLock stamps the fallback, ordinary rendering takes the earlier `voice_ref_id` branch at `nodes/_otr_voice_node_common.py:91-109`.

7. [Ship order / verification] Track gender-voice and Track voice-variety have no concrete canonical live acceptance leg, despite referencing one. Add a post-chunk Shakespeare run through `workflows/otr_canonical.json` proving source genders, `cast_source_contract`, every non-announcer `voice_ref_id`, seeded Kokoro variation, `RESULT SUCCESS`, `obs_publish OK`, and final assets. Unit tests alone do not establish ledger completeness.

Discarded-objections audit [§2.283-308 / §927-930]:

D01 — REJECTION PARTIAL. The proposed reduced-count fix remains wrong because `_plan_gender_distribution` has one length-sensitive `rng.shuffle` (`nodes/_otr_casting.py:555-615`), but the rejected absolute RNG-call tally is unsupported and already contradicted. Replace it with a state-equality regression test; retain override-in-place.

D02 — REJECTION PARTIAL. The decisive facts stand: the vendor parses before rebinding to the sliced scene (`scripts/otr_fetch_public_domain.py:317-329`), and no Folger whole-play body is present. The exact “64 bodies + one sidecar” census is stale; `_vendor_report.json:2-80` and the current corpus contents disagree. Remove the numeric claim.

D03 — REJECTION SOUND. All four rows have populated `roster_name` but empty descriptions and unknown gender (`config/source_banks/shakespeare/sources/midsummer__act3_scene2.provenance.json:40-64`). A supplement is the available offline correction.

D04 — REJECTION SOUND [ASSUMPTION: implementation follows the stated post-shuffle override exactly]. The allocator’s only RNG mutation is the shuffle (`nodes/_otr_casting.py:610-615`); overriding selected results afterward preserves the unpinned roll. Verify byte-identical RNG state in the proposed sweep.

D05 — REJECTION SOUND / MOOT. `source_bank_id` controls the Lemmy exclusion/filter (`nodes/_otr_casting.py:1270-1303`), while source-name queue consumption versus random name selection is the RNG asymmetry (`:1308-1318`). The step is cut; no residue should enter merged chunk 3.

D06 — REJECTION SOUND. ROBIN has blank `roster_name` and `absent_from_roster` (`config/source_banks/shakespeare/sources/midsummer__act3_scene1.provenance.json:60-64`); PRINCE is likewise unknown (`config/source_banks/shakespeare/sources/much_ado__act2_scene3.provenance.json:41-45`). Renaming gives no gender result.

D07 — REJECTION SOUND. `story_orchestrator.py:859-866` contains Bark `v2/en_speaker_*` presets, not Kokoro IDs. Its mutation at `:591-603` is Bark health filtering.

D08 — REJECTION SOUND, subject to build enumeration. The last ladder tier is gender-only (`nodes/_otr_voice_bank.py:67-74`), so floor 2 preserves a terminal choice and floor 3 is a supported A/B. Verify the stated 8/5/2 census directly against the committed bank.

D09 — REJECTION SOUND. Collision identity currently covers ID, path, and provider only (`nodes/_otr_voice_bank.py:333-358`). The Smith and LJSpeech pairs use distinct paths within each engine, so all twelve mirror rows require the two shared `speaker_id` values.

D10 — REJECTION SOUND. Announcer reservation triggers only when target and announcer engines match and reuse is disabled (`nodes/cast_lock.py:523-538`). Canonical node 80 uses IndexTTS2 characters, Kokoro announcer, and reuse enabled.

D11 — REJECTION SOUND. First-accepted-tier cardinality follows directly from `_LADDER` and `_matches` (`nodes/_otr_voice_bank.py:321-330,431-440`); it does not need a seed simulation. Expose a helper or test-local evaluator so the assertion does not duplicate semantics invisibly.

D12 — REJECTION SOUND / MOOT. Synonyms are cut; floor 2 independently eliminates singleton accepted tiers. Remove the obsolete combined-commit discussion from the builder queue.

D13 — REJECTION SOUND. `role_overrides.character_visual` maps to `OTR_VideoDirector.character_video_model` (`config/profiles/widget_mapping.json:23-30`), the selected profile sets `still_flat` and Z-Image (`config/profiles/otr_w45_still_flat.json:10-16`), and the runner loads canonical before applying the profile (`scripts/otr_canonical_api_run.py:132-164`). No canonical widget edit is required. Step 9 still needs the command fix above.

D14 — REJECTION SOUND. `ImageRequest` and `CanonicalImage` are instantiated only in their schema module and schema tests; the cited extra-field test intentionally expects rejection (`tests/test_image_platform_c1.py:1126-1132`). Dispatcher rows are plain dictionaries (`nodes/otr_image_gen_dispatcher.py:1170-1197`).

D15 — REJECTION SOUND. Google currently guards `init_image` and plural `reference_images` only (`nodes/_otr_image_engines/eng_google_image.py:144-151`). Adding the singular key is smaller and keeps the request type truthful.

D16 — REJECTION PARTIAL. The repository adapter itself marks the installed Z-Image node signatures as verify-at-build (`nodes/_otr_image_engines/z_image_turbo.py:27-35`). The local ComfyUI source/object-info was unavailable during review. Verify width=0 aspect derivation, VAE crop behavior, and all required keys live before accepting the rejection.

D17 — REJECTION SOUND. The Klein adapter documents approximately 2.6 decimal GB (`nodes/_otr_image_engines/flux2_klein.py:8-13`); the installed GGUF is 2,604,311,104 bytes. The dispute is GB versus GiB, not plan drift.

D18 — REJECTION SOUND. Registry registration stores one instance (`nodes/_otr_shared/engine_registry_base.py:141-150`), and both adapters retain resolved classes (`nodes/_otr_image_engines/z_image_turbo.py:318-320`; `nodes/_otr_image_engines/flux2_klein.py:249-253`). A separate lazy reference-class cache is required.

D19 — REJECTION SOUND. Jump segments deliberately require distinct seeds (`tests/test_multiclip_jump_stills.py:225-237`) and must diverge from fixed bookends (`:540-562`). They should receive reference anchoring, not the character-wide seed.

SHOULD-FIX:

1. [Track portrait Step 7] Keep `_build_zimage_graph` pure. Stage the reference file once in `render_image`, pass the staged basename in params, then build the graph. `stage_into_comfy_input` performs filesystem mutation (`nodes/_otr_video_engines/wrapper_bridge.py:999-1012`); burying it in the graph constructor complicates snapshot tests and retries.

2. [Track gender-voice Summary] Replace the stale “94 ledgers / 188 rows / 23%” headline with the grounded 88 / 176 / 25%, or state “44 confirmed contradictions” without a volatile denominator.

3. [Cross-track ship order] Add an explicit post-gender bank check for the male-shifted Shakespeare demand. The invariant is bank-based, but the production path first excludes used entries and only then permits reuse (`nodes/_otr_voice_bank.py:443-450`); measure collision/reuse behavior for the actual two-character canonical shape.

4. [Document structure] Remove executable descriptions of steps already declared cut. Keeping full implementations beside “do not build” instructions creates two reasonable, incompatible readings.

OPTIONAL / NICE-TO-HAVE:

- Record the deterministic-composer slash/path-guard exposure as a separate candidate only after live production evidence; it is not part of this build.
- Keep Klein reference support as a documented fallback decision after the Z-Image A/B, not prebuilt code.

CUT THESE:

1. [Declared cuts] Remove Track gender-voice Step 5, Track voice-variety Steps 5, 6, and 8, and Track portrait Step 8 from the builder-facing plan. Preserve only one-line deferred receipts; their removal is already operator-approved.

2. [Track voice-variety Step 3] Move operator-recording ingestion to a separate optional feature change. It closes no diagnosed defect, is absent from the ship order, and expands the locked continuity build into external asset creation.

3. [Context bulk] Move the cross-track critic narrative and rejected-design histories to the judgment archive. The builder needs the final queue, invariants, tests, and live gates—not parallel obsolete plans.

VERIFY-AT-BUILD checklist:

- Confirm live `/object_info` signatures for `ImageScale`, `VAEEncode`, `ReferenceLatent`, loaders, latent node, and sampler nodes before submit.
- Confirm Z-Image’s installed checkpoint executes the reference graph; graph construction alone is insufficient.
- Run treatment and control on separate fresh server boots, with identical source snapshot, prompts, seeds, portrait hashes, model versions, and profile.
- Assert control modes exactly: scene rows `seed`, jump rows `''`; treatment rows consuming references `reference_latent`.
- Prove portrait decoded-pixel hash stability across two identical fresh boots; invalidate the A/B if hashes differ.
- Measure peak VRAM and per-still latency for both arms against the 16 GiB machine budget.
- Audit all four new ledger fields on fresh render, cache hit, missing portrait, reference-disabled, invention, adaptation, normal voice-cast, and gender-unservable paths.
- Run the canonical workflow validator, JSON round-trip/link-widget audit, full regression suite, and Bug Bible regression.
- Require `RESULT SUCCESS`, `obs_publish OK`, and canonical episode/OBS assets on disk for the gender/voice live leg and portrait treatment/control legs.
