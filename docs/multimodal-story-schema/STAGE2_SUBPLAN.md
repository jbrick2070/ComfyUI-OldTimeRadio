# Multi-Modal Story Schema -- STAGE 2 HARDENED SUB-PLAN (v1, pre-kibitz)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`.
Status: DRAFT for kibitz r1 (codex + antigravity, Claude anchor+judge).
Precondition SHIPPED first (this session): the `_otr_outline.py` `except Exception ->
period_system_overlay = None` swallow around the outline resolver call is REMOVED
(Fable forward-note), pinned by `test_outline_resolver_call_not_swallowed`.

## 0. Scope (what Stage 2 is -- and is not)

Stage 2 = **story-path ROUTING + the new lane packs**, per BUILD_PLAN:

1. **Registries:** `nodes/story_packs/banks.json` + `nodes/story_packs/pipelines.json`
   (adapt `docs/multimodal-story-schema/schema-examples/{banks,pipelines}.json`).
2. **Resolution (Python):** fail-loud `source_bank` / `story_model` / `story_pipeline`
   resolution -- unknown id = hard error, no fallback. Replaces the router's
   transitional hard-bound `_SCIENCE_PACK_PATH` with bank -> pack-coordinate routing.
3. **Lane packs authored:** `public_domain_story`, `media_archive`,
   `simple_4_prompt_experimental` (adapt `schema-examples/story_packs/*` +
   `source_packets/*`). Same ledger contract; content adapted to REAL production seams.

NOT in Stage 2 (explicit):
- No visual_style anything (Stage 3). No asserts->JSON (Stage 4).
- The NEW lanes do not EXECUTE end-to-end yet -- their interpreters/fetchers and the
  `simple_4` pass runner are not built in this stage. Stage 2 makes the lanes
  ADDRESSABLE (registered, validated, routed, selectable data) without changing the
  science lane's bytes. Lane execution rides later stages/sub-chunks; a runnable=false
  bank selected at run time FAILS LOUD naming the bank (never silently runs sci-fi).
- Workflow-JSON widget (source_bank dropdown): DEFERRED to the end of Stage 2 as its
  own gated chunk (2C) -- it is the first user-visible surface and must land in
  `workflows/otr_scifi_16gb_full.json` in the same commit as its code (CLAUDE.md 0).

## 1. Chunking

**Chunk 2A -- registries + resolver (science lane byte-identical).**
- `nodes/story_packs/banks.json`: adapt the 4 example banks (science_news,
  media_archive, public_domain_story, custom_source_bank). Fields: source_bank_id,
  label, source_kind, interpreter, fetcher, default_story_model,
  default_story_pipeline, defaults{}, required_seams[], runnable, guide_ref.
  (default_visual_style: keep the FIELD, validated as an opaque string -- Stage 3
  resolves it. Rejecting it now would force a schema bump later.)
- `nodes/story_packs/pipelines.json`: the 2 example pipelines (legacy_many_pass
  DESCRIPTIVE-ONLY metadata; simple_4_prompt_experimental executable_in_lab=false
  in-repo -- there is no lab; rename the flag `executable` and set false until the
  pass runner exists).
- New module `nodes/_otr_story_routing.py` (stdlib-only, same posture as
  `_otr_story_pack.py`): load-once cached registry loaders + strict validators
  (dup-key reject, unknown-key reject, non-empty ids, cross-refs: every bank's
  default_story_model must resolve to an on-disk pack `story_packs/<bank>/<model>.json`
  whose header triple matches; every default_story_pipeline must exist in
  pipelines.json; every pipeline seam_ref must be in the pack-seam allowlist) +
  `resolve_story_pack(source_bank_id, story_model_id=None) -> StoryPack` +
  `get_bank(source_bank_id)` / `get_pipeline(story_pipeline_id)` -- unknown id =
  typed hard error (StoryRoutingError hierarchy mirroring StoryPackError).
- Router change: `_otr_creative_prompt_router.py` drops `_SCIENCE_PACK_PATH` and asks
  `resolve_story_pack("science_news")`. Byte-identical output pinned by the existing
  equivalence tests (they assert VALUE identity to `L._SYSTEM_PROMPT`).
- Tests move in the same commit: registry fail-loud matrix (unknown bank, unknown
  pipeline, missing pack file, cross-ref mismatch, dup key, malformed), router
  equivalence stays green, science pack still byte-identical, sanctioned-consumer
  guard extended to allow `_otr_story_routing.py`.

**Chunk 2B -- lane packs authored + validated.**
- `nodes/story_packs/public_domain_story/faithful_radio_adaptation.json`,
  `nodes/story_packs/media_archive/media_restoration_adventure.json`,
  `nodes/story_packs/experimental/simple_4_prompt_experimental.json`
  (one pack per lane in Stage 2 -- the other lab variants are follow-on content).
- **Seam-name law:** lane packs use the PRODUCTION seam names
  (`PRODUCTION_SEAM_ALLOWLIST`), NOT the lab's aspirational names. Mapping:
  lab `outline_system` content -> authored across `outline_macro/phase/beat_system`
  where it makes sense; `coda_system` maps 1:1. Lab seams with NO production consumer
  yet (pitch_room_system, story_select_system, dramatic_state_system, title_system,
  line_grounding) are NOT invented as allowlist entries in 2B -- they ride with the
  stage that builds their consumers. simple_4's `pass_1..pass_4` seams DO enter the
  allowlist in 2B (namespaced `pass_1_creative_story` etc.) because pipelines.json
  references them and the validator cross-checks seam_refs; they are inert until the
  pass runner exists.
- OPEN QUESTION for the panel: extend `PRODUCTION_SEAM_ALLOWLIST` per-pack (a
  pack-kind-scoped allowlist) vs one global list. Draft answer: ONE global allowlist,
  grown deliberately -- simplest to police, no dynamic key admission.
- Non-science lanes need lane-appropriate CONTENT for the seams the production writer
  actually consumes today (at minimum `line_composer_system`); adapt the lab prose.
  Every authored seam validates non-empty; `status` field marks lanes
  `ready_fixture` (not yet executable).
- Tests: each lane pack loads + validates; exact-key-set per pack; registry
  resolution reaches each lane; runnable=false / non-executable pipeline selection
  raises loud; science-lane byte-identity unchanged.

**Chunk 2C (GATED, last) -- the selector surface.**
- `source_bank` widget on the writer node, wired IN `workflows/otr_scifi_16gb_full.json`
  same commit; saved default = `science_news` (byte-identical episodes by default).
  Unknown widget value = hard error at episode start. Gate: kibitz on the wiring +
  validator + widget audit (BUG-LOCAL-097 positional rule: APPEND only).

## 2. Invariants (unchanged)

JSON owns content; Python owns validation/routing/execution; NO fallbacks; unknown
id = hard error; audio spine FROZEN (`test_audio_byte_identical` green); science lane
byte-identical through 2A/2B; suite + Bug Bible + B7 green per chunk; UTF-8 no BOM;
commit per green chunk (push per operator instruction this session); prod/main gated.

## 3. Acceptance

- 2A: registries load fail-loud; router routes science via the bank; zero episode change.
- 2B: 3 lane packs on disk, validated, addressable via resolve_story_pack; selecting
  a non-executable lane raises loud; science unchanged.
- 2C: dropdown present in the real JSON, default science_news, validator green.
