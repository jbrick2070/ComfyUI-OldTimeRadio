<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Build order is plausible, but L1/L2 cannot be wired as written until the outline prompt/output parser, line request call site, validator insertion point, and metadata/flag propagation are explicitly connected; L5a also conflicts with the current freeze cascade terminal-stop behavior.

MUST-FIX BEFORE BUILD:
1. [Build order step 1 / L5a / `_otr_freeze_cascade.py:593-605`, `:730-766`] Defect: “fix critic `too_many_edits -> arc="?"` abort” is underspecified and likely ordered wrong as a measurement prerequisite. Grounding says terminal reviewer verdicts `too_many_edits / needs_full_rerun` stop the cascade before the story critic runs. The story critic call is later at `_otr_freeze_cascade.py:730-766`, so no critic telemetry/arc verdict can be produced after `too_many_edits` unless the cascade sequencing changes. Concrete fix: decide and implement one explicit path:
   - either run advisory `run_story_critic` before terminal reviewer stop, using the restored/original ledger snapshot and stamping only `meta.story_critic_report`; or
   - do not claim critic metrics are available for terminal-review ledgers, and limit L5a to reviewer telemetry only.
   Add a regression test where `apply_doctor_edits(...) == -1` still yields the intended telemetry/critic behavior.

2. [L1/L2 scaffolding / `Beat` model `_otr_outline.py:84-135`] Defect: adding `conflict_object`, `conflict_type`, `beat_role`, and later `choice_summary`/presence markers to `Beat` is not enough. `Beat` is a Pydantic model with required existing fields and likely parses LLM outline JSON. If the outline LLM is expected to emit these fields once `OTR_STORY_QUALITY_L12` is ON, the outline prompt, parser/schema instructions, fallback/default population, and serialization tests must all be updated together. Otherwise the fields stay empty and composer rendering/validators receive nothing. Concrete fix: in the scaffold step, add optional Pydantic fields with defaults to `Beat`; in render step, update the outline prompt/output contract and/or post-process every voiced beat to fill `beat_role`, `conflict_object`, `conflict_type`, `choice_summary`, and required role markers before validation/composer mapping. Verify exact outline generation function and parser.

3. [L2 validator / R3 target 2 / `_otr_episode_budget.py:190-240`] Defect: the new `beat_role` validator depends on the final voiced-beat sequence, but the plan does not specify its insertion point relative to existing outline validators and any deterministic fallback beat factory. If inserted before role assignment/fallback, it will fail on empty defaults; if inserted after a validator that returns on first failure, role errors may be hidden; if it mutates after budget validation, it can violate word/phase constraints. Concrete fix: sequence explicitly:
   1. generate/parse outline;
   2. assign deterministic roles and conflict slots to voiced beats;
   3. run fallback beat factory for missing required role content;
   4. run existing budget/arc validators unchanged;
   5. run new role validator with the same first-failure contract.
   Or, if existing `validate_outline_against_budget` must remain the single entry point, add the role validator at the end after deterministic population. Verify exact validator list/function.

4. [L2 fallback beat factory] Defect: fallback beat factory “returning ALL required beat fields” can silently break `EpisodeBudget` constraints if it changes `target_words`, `speaker`, `arc_phase`, `sfx_cue`, or beat count after budget validation. Grounding shows `EpisodeBudget` consumers include per-phase words/beats and arc phases; `arc_phases` must not be overloaded. Concrete fix: fallback factory must be a same-beat replacement only: preserve `beat_id`, `speaker_role`, phase allocation, beat count, and target word range; only fill/replace narrative fields and new optional SQ fields. Run budget validators after fallback.

5. [LineRequest / composer wiring / `_otr_line_composer.py:581-700`] Defect: new `LineRequest` fields are named, but no call-site mapping is specified. Grounding says the caller maps `Beat` fields into `LineRequest`; adding dataclass fields alone will not populate them. Concrete fix: update the writer/orchestrator call site that constructs `LineRequest` to pass:
   - `beat_role=beat.beat_role`
   - `conflict_object=beat.conflict_object`
   - `conflict_type=beat.conflict_type`
   - `choice_summary=...` if composer/outro needs it
   - existing `allowed_people`/`allowed_things`
   Add a unit test that a populated `Beat` produces a `LineRequest` whose prompt contains the new block only when `OTR_STORY_QUALITY_L12` is ON. Verify exact call site.

6. [L1a allowed roster routing / `LineRequest` fields `_otr_line_composer.py:581-700`] Defect: the plan says “route via writer call site that already populates split fields,” but grounding only proves `LineRequest` has `allowed_people`/`allowed_things`; it does not prove they are populated today. If they remain empty, composer falls back to legacy combined `allowed_roster` block, defeating the L1a premise-anchor requirement. Concrete fix: verify and update the writer call site to build split sets from `allowed_roster`, filtering `"ANNOUNCER"` from render-only people/things while preserving `allowed_roster` for phantom-name gate. Add a test where `allowed_roster={"CHANDRA","EL NINO","US SPACE FORCE","ANNOUNCER"}` results in prompt-rendered split fields excluding `ANNOUNCER` but phantom gate still receives the union as intended.

7. [L2 personal_stake source / `LineRequest.all_voice_cards` `_otr_line_composer.py:581-700`] Defect: the plan depends on a structured character cost/fear field but grounding shows composer receives `all_voice_cards` only as rendered text, and no structured cost/fear field is shown. Parsing private stakes out of `all_voice_cards` would create an unstable format dependency. Concrete fix: implement the deterministic fallback table as the default for v0 unless a real structured source is verified. If a structured field exists, thread it from cast rows to beat metadata before composer; do not parse the rendered `all_voice_cards` string. verify: cast row schema/source for cost/fear.

8. [L1 crisis-noun repair] Defect: “only in GENERATED intent-like fields” conflicts with the concrete data model, which currently shows `Beat.intent`, `mood`, `sfx_cue`, `arc_phase`; no list of safe mutable fields is specified. A broad repair pass can corrupt prompt-visible entities or budget fields. Concrete fix: define an allowlist for substitution, initially only `Beat.intent` and any new SQ-specific summary fields; never mutate `speaker`, `beat_id`, `speaker_role`, `arc_phase`, `target_words`, `sfx_cue`, or roster/title/premise strings. Add a whole-token singular/plural regression test.

9. [L3/L4 flags / Build order step 4] Defect: L3 and L4 are both audio-affecting but their order and interfaces to TTS/freeze are not grounded. L4 says it operates on FINAL transcript line TEXT only; L3 strips composer output. If L4 runs after freeze/TTS, audio and ledger diverge; if it runs before phantom-name/quote validations, it may alter text after validators have passed. Concrete fix: specify insertion points:
   - L3 immediately after compose/polish returns text and before line persistence/compose_flags.
   - L4 after all line text is final but before freeze/TTS/hash/golden generation.
   Add tests proving speaker labels/identity fields are unchanged and sanitized text is what TTS receives. verify: exact transcript persistence/TTS handoff function.

10. [Configuration/flags] Defect: flags are named but propagation is incomplete. `OTR_STORY_QUALITY_L12` must gate both prompt rendering and deterministic outline mutations if “flag OFF = byte-identical” is mandatory. If selectors/fallbacks run with the flag off and are serialized into `Beat`/meta, JSON will drift even if prompts do not. Concrete fix: in scaffolding commit, new fields default empty and no generated ledger/meta changes occur unless the flag is enabled. Tests: no-drift JSON assert with all flags off; prompt byte-identical assert for `_build_user_prompt`; ledger lacks `meta.story_quality` unless the relevant telemetry flag/path is enabled.

11. [Telemetry / `_otr_ledger_scrub.py:981-1011`] Defect: acceptance says persist new counts under `meta.story_quality`, but current aggregation overwrites `meta["story_quality"]` with only `l1_rerolls`, `l7_splits`, `l7_split_failures` when `_sqv2_on`. Adding L1/L2 metrics elsewhere before scrub will be lost. Concrete fix: change aggregation to merge into existing `meta.story_quality` instead of replacing it, or centralize all story-quality metrics in scrub. Add a test with preexisting `meta.story_quality={"ungrounded_crisis_density": ...}` and compose flags; scrub must preserve and augment.

12. [Telemetry naming / `_otr_ledger_scrub.py:981-1011`] Defect: current telemetry uses `_sqv2_on`, while the new flags are `OTR_STORY_QUALITY_L12`, `OTR_COMPOSER_ACTION_STRIP`, and `OTR_TRANSCRIPT_SANITIZER`. The plan does not say whether `_sqv2_on` maps to the old `story_quality_v2_enabled`, the new L12 flag, or another env var. This can produce missing acceptance metrics even when L1/L2 is enabled. Concrete fix: define one flag contract: which env var controls scrub aggregation, how it is read, and whether L1/L2/L3/L4 counters are written independently. Prefer `meta.story_quality` aggregation enabled if any SQ feature flag is on, while preserving no-key behavior when all are off.

SHOULD-FIX:
1. [L2 announcer outro] Defect: “outro references the climax CHOICE” needs a data path. `choice_summary` is mentioned but not included in the initial “Where every new field lives” list for `Beat`/`LineRequest`. Concrete fix: add `choice_summary: str = ""` to `Beat` and/or `meta.story_quality`/beat meta, and pass it to the announcer closing-beat composer path or deterministic outro template. Test closing announcer beat contains one seeded template rendering when flag ON and is unchanged when OFF.

2. [L2 field-presence markers] Defect: “sensory-consequence + state-change checked by FIELD-PRESENCE” requires fields, but they are not named in the data model. Concrete fix: add explicit optional fields, e.g. `sensory_consequence: str = ""` and `state_change: str = ""`, or store under `beat.meta.story_quality`; then update validator/fallback/composer mapping consistently. Avoid prose regex.

3. [Meta/compose_flags compatibility / R3 target 5] Defect: the plan assumes unknown `meta`/`compose_flags` keys are ignored by freeze/TTS/serialization but does not prove it. Concrete fix: add a compatibility test with unknown `meta.story_quality` keys and new `compose_flags`; freeze/scrub/TTS should either preserve or ignore them without schema errors. verify: frozen ledger schema enforcement code.

4. [L1 domain selector] Defect: ordered keyword map over raw `premise`/`meta` can be nondeterministic if it iterates unordered dict keys or unnormalized Unicode/case. Concrete fix: normalize `premise` and selected `meta` text with casefold + Unicode normalization; inspect only an explicit ordered list of meta fields, not arbitrary dict iteration.

5. [L1 palette selector] Defect: “sha256(...)-> sorted-palette modulo” is underspecified for two independent selections. Using the same digest/modulo for object and type can create stable pair coupling and reduce variety. Concrete fix: domain-object digest and domain-type digest should include distinct labels, e.g. `...:object` and `...:type`.

6. [L3 compose flags] Defect: L3 strips action but plan says do not persist `internal_action`; however telemetry may still need to know strip counts. Concrete fix: add only a minimal `compose_flags` marker such as `action_strip:marked` / `action_strip:regex` if flag ON; no new text field.

7. [L4 quote balancing] Defect: “balance quotes conservatively” can still alter intended spoken text and affect audio. Concrete fix: limit to obvious unmatched leading/trailing straight/curly quote wrappers; do not normalize all quotes. Add apostrophe/measurement tests.

OPTIONAL / NICE-TO-HAVE:
- [Acceptance metric] Store raw numerator/denominator for `ungrounded_crisis_density`, not just the ratio, so soak comparisons are auditable.
- [L1 palette table] Put the UTF-8 palette table in a deterministic Python module or JSON loaded with explicit UTF-8 encoding and sorted validation at import.
- [Build order] Add a per-flag prompt snapshot test for `OTR_STORY_QUALITY_L12`, `OTR_COMPOSER_ACTION_STRIP`, and `OTR_TRANSCRIPT_SANITIZER`.

CUT THESE (over-engineering):
1. [L2 / structured personal_stake source] Cut structured cost/fear discovery from v0 unless a real field is verified. The deterministic fallback table satisfies the interface without creating a fragile cast-schema dependency.
2. [L4 / mojibake repair] Cut repair entirely for v0. The plan already says VERIFY-ONLY; do not add encoding transforms until a real ledger proves the artifact and insertion point.
3. [L5b / gemma-12b bake-off] Keep deferred. It has external model/runtime side effects and does not unblock L1/L2 wiring or telemetry correctness.