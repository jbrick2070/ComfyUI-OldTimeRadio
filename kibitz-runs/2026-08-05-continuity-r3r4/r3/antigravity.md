VERDICT: build-ready with fixes. Critical sequencing, parameter contract mismatches, and dead-code seams in fallback paths require fixes before build.

MUST-FIX BEFORE BUILD:

1. [Track 1 Step 6 / Track 3 Step 7 / Chunk 3] Merged Fallback Stamping Orphans `_resolve_clone_ref_path` Fallback Path
   - Defect: When CastLock stamps `voice_ref_id` on `gender='other'` unservable rows (`nodes/cast_lock.py:616-620`), `cast.get("voice_ref_id")` becomes non-empty. When `_resolve_clone_ref_path` (`nodes/_otr_voice_node_common.py:91-127`) runs at render time, the `vrid` lookup at `:91-95` succeeds immediately, completely skipping lines `:96-126`. Track 3 Step 7's test for `_resolve_clone_ref_path` fallback will exercise dead code if run on a CastLock-processed cast dict.
   - Concrete Fix: In Chunk 3: (a) CastLock stamps `voice_ref_id` with the fallback ref and sets `voice_cast_fallback="gender_unservable"`; (b) `_resolve_clone_ref_path` keeps lines `:109-127` for un-stamped raw cast dicts; (c) Track 3 Step 7's test suite must explicitly assert BOTH CastLock's fallback stamping on cast dicts AND direct `_resolve_clone_ref_path` execution on dicts with `voice_ref_id=None`.

2. [Track 2 Step 7 / Step 8] `ImageScale` Parameter Contract Mismatch Crashes Render Graph
   - Defect: In `nodes/_otr_image_engines/z_image_turbo.py:230-270` and `nodes/_otr_image_engines/flux2_klein.py:159-206`, constructing `scale_ref` with partial arguments violates ComfyUI's `ImageScale.upscale` signature (`nodes.py:1883`), which strictly requires 5 arguments (`image`, `upscale_method`, `width`, `height`, `crop`). `wrapper_bridge.py:401-404` `run_graph` fails with `TypeError` -> `GraphExecutionError`, terminating the episode with `ImageRenderError NO FALLBACK` (`nodes/otr_image_gen_dispatcher.py:1143-1150`).
   - Concrete Fix: Specify all 5 required arguments in `scale_ref` graph nodes: `{image: W('load_ref',0), upscale_method: 'lanczos', width: 0, height: params['reference_height'], crop: 'disabled'}`.

3. [Track 2 Step 8] `flux2_klein` Candidate Binding Collision with `FluxKontextImageScale`
   - Defect: `nodes/_otr_image_engines/flux2_klein.py:140` uses candidate resolution (`wrapper_bridge.py:100-107`) which binds the first matching installed class name. Placing `FluxKontextImageScale` (`comfy_extras/nodes_flux.py:126`, taking image ONLY) in the `scale_ref` candidate tuple causes candidate resolution to bind it instead of `ImageScale`, raising `TypeError` when passed `upscale_method`/`width`/`height`/`crop`.
   - Concrete Fix: Omit `FluxKontextImageScale` from `_REF_CANDIDATES['scale_ref']`. Explicitly bind `('ImageScale',)` in `_REF_CANDIDATES` for `flux2_klein.py`.

4. [Track 2 Step 2 / Step 3] `UnboundLocalError` and Contract Bypass in `resolve_seed_and_mode`
   - Defect: Placing the `scene_character` seed branch before line 159 in `nodes/otr_image_gen_dispatcher.py:134-164` references `base` before its definition (`base = int(cfg.get("request_seed") or 0)` at `:159`), raising `UnboundLocalError`. Placing it before lines `:160-161` bypasses the `mode == "fixed"` contract. `dispatch_images` calls `resolve_object_seed` at `:998` outside any `try/except`, causing an uncaught crash.
   - Concrete Fix: Position the `scene_character` branch in `resolve_seed_and_mode` strictly AFTER lines `:160-161` (`if str(cfg.get("mode") or "request_hash") != "request_hash": return (base, "")`).

5. [Ledger Completeness] Undefined / Unowned Ledger Meta Keys Across Execution Paths
   - Defect: Violates the operator invariant that all new ledger fields must have exactly one owner and a defined value across ALL execution paths:
     - `cast_source_contract` (`nodes/_otr_casting.py:1780`): Missing/unowned on invention / non-adaptation lanes (`gender_by_name=None`).
     - `derived_from_portrait_hash` and `portrait_anchor_mode` (`nodes/otr_image_gen_dispatcher.py:1170-1194`): Missing on non-character image rows (`portrait`, `scene_open`, `radio_host_portrait`).
     - `voice_cast_fallback` (`nodes/cast_lock.py:616`): Missing on standard (non-fallback) cast rows.
   - Concrete Fix:
     - `cast_source_contract`: Always stamp in `lock_cast` (`None` when `source_character_genders` is `None`).
     - `derived_from_portrait_hash` & `portrait_anchor_mode`: Initialize to `""` on all image ledger rows in `dispatch_images`.
     - `voice_cast_fallback`: Initialize to `None` on all cast ledger rows in `CastLock`.


SHOULD-FIX:

1. [Track 1 Step 4 / Q1] Male-Shifted Adaptation Demand vs. Male-Light `indextts2` Bank
   - Defect: Track 1 pins adaptation cast rows to Shakespeare source genders (male-heavy: 30 male / 23 female in roster), while `indextts2` is male-light (17 male / 23 female). Landing Track 1 before Track 3 Step 4 (tier floor=2) increases male voice collisions on 2-character adaptation scenes.
   - Concrete Fix: Maintain strict ship ordering: deploy Track 3 Step 4 (tier floor=2) immediately following Track 1.

2. [Track 2 Step 4] Path Guard Exception Exposure in Prompt Prepend
   - Defect: In `nodes/otr_meta_brief_image_prompt.py:1339-1409`, prepending raw `appearance` text containing slashes (`/` or `\`) causes `path_guard_arm` (`nodes/otr_image_gen_dispatcher.py:219-272`) to trigger `arm='alternate_separator'`, resulting in a skipped still and NO image generated.
   - Concrete Fix: Sanitize `_app = _app.replace('\\', ' ').replace('/', ' or ')` prior to prepending in `_compose_char_scene_prompt`.

3. [Track 1 Step 1 / Step 4] Roster Object Mutability vs. Ledger Serialization
   - Defect: `load_roster_characters` in `nodes/_otr_roster_gender.py` must return plain, JSON-serializable `dict` objects. Returning `MappingProxyType` or frozen dataclasses causes `json.dumps` failures in `_copy_sidecar` (`nodes/_otr_source_payload.py:195-203`).
   - Concrete Fix: Strictly return `tuple[dict, ...]` of standard Python `dict`s from `load_roster_characters`.


OPTIONAL / NICE-TO-HAVE:

1. [Track 3 Step 8] Timbre Synonyms: Defer until after tier floor=2 lands.
2. [Track 3 Step 3] Personal Voice Registration: Limit strictly to the 3 approved operator recordings (`mr_jeffrey_uk.m4a`, `mr_jeffrey_uk_expressive.m4a`, `mr_jeffrey_usa.m4a`).


CUT THESE (over-engineering):

1. Track 3 Step 5 (Kokoro char_voice per-lang_code / roles / begin_episode): Unreachable in production. `config/audio_engine_profiles.yaml:154` pins `allowed_voice_banks: [kokoro_builtin]` while canonical workflow runs `voice_bank='default'`. Safe to cut completely.
2. Track 3 Step 8 (Timbre synonyms as mandatory): Over-engineering; tier floor=2 alone eliminates all size-1 tiers without introducing a secondary vocabulary mapping.
3. Track 3 Step 6 (Drift guard as separate step): Fold assertions directly into Step 1 test suite.
4. Track 2 Step 8 (Flux2 Klein reference wiring): Defer until Step 9 live A/B proves whether `z_image_turbo_nvfp4` requires a fallback.
5. Track 1 Step 5 (Bark replay parity): Downgrade to `PROD_BUG_LOG.md` entry (seed 424242) since node 80 runs `indextts2`.
6. Cast renames (`Puck` -> `Robin`, `Don Pedro` -> `Prince` in Track 1 Step 3): Zero gender payoff, broadcast-facing name changes. Cut.


EXPLICIT WIRING & CONVERGENCE ANSWERS (Q1 - Q8):

- Q1: Note. The tier floor invariant is bank-wide, but Shakespeare's male bias shifts demand against `indextts2` (17 male / 23 female). Landing Track 3 Step 4 immediately after Track 1 resolves collision pressure.
- Q2: Yes, safe. Trailing append of `source_owned: bool = False` after `age_band` in `EnsembleSlot` (`nodes/_otr_casting.py:536-552`) preserves positional calls like `tests/test_cast_llm_naming.py:141` which use 5 positional args matching the first 5 fields (`char_id`, `name`, `gender`, `timbre`, `role`).
- Q3: All callers green. `source_meta_from_scene` (`nodes/_otr_shakespeare_sources.py:428`) and `source_meta_from_unit` (`nodes/_otr_public_domain_sources.py:540`) are only called in those files and `tests/test_shakespeare_sources.py:177`, `tests/test_public_domain_sources.py:129`. Keyword-only `text_path: Path | None = None` preserves positional compatibility.
- Q4: Yes, resolves. `sidecar_path_for_text(text_path)` derives `text_path.parent / (text_path.stem + '.provenance.json')`. Verified `config/source_banks/shakespeare/sources/` contains all 14 `.provenance.json` sidecars alongside their `.txt` files.
- Q5: No import cycle. `nodes/_otr_voice_bank.py` does not import `cast_lock.py`. `nodes/cast_lock.py:527` is indeed the sole duplicate tuple `("google_tts", "chatterbox", "dia")`.
- Q6: Confirmed. Placing the branch before line `:159` in `nodes/otr_image_gen_dispatcher.py` causes `UnboundLocalError` on `base`. Placing before `:160` bypasses `mode == "fixed"`. `dispatch_images:998` calls `resolve_object_seed` outside any `try/except`, causing an uncaught crash.
- Q7: Safe. Resolving references at lines `:997-998` occurs after path guard checks and `engine_id` resolution (`:939`), immediately prior to `request_cache_key` at `:1000`.
- Q8: Yes, seam confirmed. Stamping `voice_ref_id` in CastLock causes `_resolve_clone_ref_path` (`nodes/_otr_voice_node_common.py:91`) to hit on `vrid`, skipping lines `:109-127`. Fixed in MUST-FIX #1.


DISCARDED OBJECTIONS AUDIT (D01 - D19):

- D01: Stated rejection valid in conclusion (RNG stream shifts on count change), though cited tally was imprecise. Keep design (leave `_plan_gender_distribution` untouched).
- D02: Stated rejection valid. `config/source_banks/_corpus/` contains 64 prose bodies + 1 sidecar, zero Folger plays. `body` is rebound to sliced scene text. Rejection stands.
- D03: Stated rejection valid. `midsummer__act3_scene2.provenance.json` has `roster_name` but empty descriptions for all 4 characters. Supplement file is cleaner. Rejection stands.
- D04: Stated rejection valid. Override-in-place leaves unpinned slots and roll stream untouched. Rejection stands.
- D05: Stated rejection valid. Track 1 Step 5 is downgraded/cut, D05 is moot. Rejection stands.
- D06: Stated rejection valid. Zero gender payoff, broadcast-facing name changes. Rejection stands.
- D07: Stated rejection valid. `story_orchestrator.py:861` is a Bark list, not Kokoro. Rejection stands.
- D08: Stated rejection valid. Floor=3 deletes timbre for 22/24 combos (only 2/24 honour timbre vs 5/24 at floor=2). Measured: 8/24, 5/24, 2/24. Rejection stands.
- D09: Stated rejection valid. 40 shared `ref_paths` across mirrors; only 2 genuine different-file pairs (12 rows). Rejection stands.
- D10: Stated rejection valid. Unreachable in production (announcer is Kokoro, chars are IndexTTS2) + re-bases Chatterbox/Dia. Rejection stands.
- D11: Stated rejection valid. Structural invariant (no size-1 first-accepted tier) is deterministic and measurable without simulation. Rejection stands.
- D12: Stated rejection valid. Asymmetry is real; synonyms now cut/optional. Rejection stands.
- D13: Stated rejection valid [CRITICAL]. `config/profiles/widget_mapping.json` maps `role_overrides.character_visual` to `[OTR_VideoDirector, character_video_model]`. `scripts/otr_canonical_api_run.py --profile` loads canonical JSON and applies profile. Editing node 87 widget is wrong. Rejection stands.
- D14: Stated rejection valid. `CanonicalImage`/`ImageRequest` are constructed nowhere outside `schemas.py` and one test. Rejection stands.
- D15: Stated rejection valid. Single path, `reference_image` matches existing guard. Rejection stands.
- D16: Stated rejection valid. `ImageScale` derives missing side when `width=0` (`nodes.py:1885-1889`), `VAE.encode` crops internally. Rejection stands.
- D17: Stated rejection valid. Reviewer error on disk math. Rejection stands.
- D18: Stated rejection valid [SHARP]. `_classes` is cached on registry singleton (`z_image_turbo.py:318-320`), so params-gated candidates resolve without ref keys on portrait and crash subsequent referenced mints. Rejection stands.
- D19: Stated rejection valid. Jump segments get anchor and reference, but NOT identical seed (`tests/test_multiclip_jump_stills.py:234-237` asserts 3 distinct seeds). Rejection stands.
