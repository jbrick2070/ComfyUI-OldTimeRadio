<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Core L1/L2 is not build-ready: required data shapes, insertion points, flags, fallback algorithms, and schema/validator changes are underspecified against the shown APIs.

MUST-FIX BEFORE BUILD:
1. [L1b / Build order / Open code-verify #1] `domain -> conflict-object/conflict-type palette` has no defined domain input or fallback. The plan says “VERIFY a category field exists in meta (else classify from the logline)” but classification method is unspecified and “classify from logline” risks adding a new model-dependent step. Concrete fix: define a deterministic function signature and source, e.g. `select_domain(meta: dict, premise: str) -> str` using an explicit ordered keyword map, with default `"general"`, and define the curated table contents and fallback objects/types for `"general"`.

2. [L1] “Every beat intent carries a structured `conflict_object` + `conflict_type` SLOT” is not tied to any shown data model. Grounding shows `_build_beat_user_prompt(...)` only receives `req, macro, phase_name, beat_speaker, beat_position, previous_beat_intent, next_beat_speaker, phase_summary`; no beat slot fields are present. Concrete fix: specify the exact model/container where these fields live: Outline beat dataclass/Pydantic field names, types, defaults, serialization behavior, and whether they ride `beat.meta` to satisfy fixed ledger schema. Verify actual `OutlineBeat` shape before coding.

3. [L1 / post-outline substitution] “post-outline check COUNTS ungrounded crisis nouns and substitutes the beat’s conflict noun” is not implementable as written. It does not define which text fields are scanned, how “ungrounded” is decided case/plural-wise, how many occurrences trigger per episode vs per beat, or how substitution preserves grammar. Concrete fix: define a pure function, e.g. `repair_crisis_nouns(outline, allowed_palette: set[str], conflict_by_beat: dict[str, str]) -> RepairReport`, list scanned fields (`intent`, `objective`, etc.), use word-boundary regex with singular/plural map, and replace only whole-token matched crisis terms in generated intent-like fields. Do not replace proper allowed entities.

4. [L1] The denylist contains words that may be legitimate in existing ledgers/news domains (“switch” appears in `signal_lost_the_scorchedearth_switch_20260623_060918`; “venting o2” appears in another ledger title). The plan says exclude only nouns “NOT in the brief palette,” but the grounded `meta.allowed_roster` does not contain ordinary title/premise nouns. Concrete fix: build the allowed palette from more than `allowed_roster`: include normalized title/premise/logline nouns or explicitly exempt title tokens. Otherwise the deterministic repair can corrupt legitimate episode-specific terminology.

5. [L2] `beat_role in {setup, pressure, personal_stake, irreversible_choice, consequence}` contradicts “exactly one `climax/irreversible_choice` as the last voiced beat.” `climax` is referenced but not in the enum. Concrete fix: choose one representation. Prefer `beat_role="irreversible_choice"` and optional `beat_function_label="climax"` if needed, or add `climax` to the enum and update all validators accordingly.

6. [L2 / arc_phases wiring] “map roles onto existing `arc_phases` positions” is underspecified and can break validator #5. Grounding shows monotonic validation uses `budget.arc_phases` and each voiced beat’s `arc_phase`; it does not know `beat_role`. Concrete fix: do not add roles to `EpisodeBudget.arc_phases` unless they are actual phase names. Add `beat_role` separately to beat metadata, and add a new validator after arc_phase monotonic validation that checks role ordering among voiced beats. If you truly modify `arc_phases`, specify the exact new `arc_phases`, `per_phase_words`, `per_phase_beats` lengths and update `_format_episode_budget_block`/`validate_outline_against_budget`.

7. [L2 / small budget behavior] “drop optional pressure beats first, never the climax” lacks an algorithm for budgets with fewer than required roles. The required contract needs at least setup + personal_stake + irreversible_choice, possibly consequence, but the plan does not define minimum voiced beats. Concrete fix: define role allocation for `n_voiced_beats`: e.g. `n=1 -> irreversible_choice only`, `n=2 -> personal_stake, irreversible_choice`, `n>=3 -> setup, personal_stake, pressures..., irreversible_choice`; decide whether `consequence` is possible if “irreversible_choice as last voiced beat” is mandatory.

8. [L2] “personal_stake beat injects a character-specific private cost/fear from the cast/character sheet” is not grounded in shown APIs. `_build_user_prompt` receives `all_voice_cards`/`character_voice_card` as rendered strings, not structured fears/costs. Concrete fix: verify character model fields. If no structured field exists, define a deterministic fallback table keyed by speaker role/name and premise domain, and store selected `personal_stake` text in beat meta.

9. [L2 / composer carry-through] “Carry the `beat_role` + `conflict_object` TAG into the composer” has no concrete `LineRequest` interface change. Grounding shows `_build_user_prompt(req: LineRequest)` accesses many fields but no `beat_role` or `conflict_object`. Concrete fix: add optional fields to `LineRequest`, e.g. `beat_role: str = ""`, `conflict_object: str = ""`, `conflict_type: str = ""`, and render them in an existing optional block such as DRAMATIC FRAME only when non-empty. Update all call sites/tests or provide defaults preserving byte-identical prompts when unset.

10. [L2 / deterministic fallback] “substitute a deterministic hand-authored fallback beat keyed to (phase, conflict_object)” is missing required output shape. The outline validator expects voiced beats to have at least `speaker_role`, `arc_phase`, `target_words`, and likely `beat_id`/speaker fields [ASSUMPTION: full beat model not shown]. Concrete fix: specify fallback factory signature and complete fields, e.g. `make_required_role_beat(role, arc_phase, speaker, target_words, conflict_object, conflict_type, ...) -> OutlineBeatPatch`, and state whether it patches only intent/mood or replaces the whole beat.

11. [L2 / announcer outro] “outro references the climax CHOICE as a semantic requirement” lacks a deterministic source for “choice.” `irreversible_choice` is a role, not necessarily a parsed choice. Concrete fix: require a `choice_summary` field/meta for the climax beat, filled by the same deterministic fallback/slot system, and pass it into announcer close generation/template. Define fallback if missing.

12. [Hard constraints #4 / L3 / L4 / Build order] Audio-affecting work requires exact flag/default/rebaseline procedure, but L3 only says “Name the flag + default + re-baseline procedure” and L1/L2 will also change dialogue. Concrete fix before build: define flags now. Example: `STORY_QUALITY_L12_ENABLED=false` default-off for L1/L2 prompt/beat changes; `COMPOSER_ACTION_DELIMITER_STRIP=false` default-off for L3; `TRANSCRIPT_SANITIZER_V1=false` default-off for L4. Define golden rebaseline command/path and operator approval condition. Without this, `test_audio_byte_identical` cannot be protected.

13. [L5a] “fix the critic `too_many_edits -> arc='?'` abort + telemetry under-count” is not implementable from the plan or grounding. No critic file, function, data schema, or aggregation shape is provided. Concrete fix: identify exact files/functions and current condition: e.g. where `too_many_edits` is set, where `arc="?"` aborts grading, where `meta.story_quality` is aggregated, and expected before/after telemetry keys. Verify against source.

14. [Acceptance metric] Primary metric “distinct conflict-object / conflict-type n-grams across episodes” cannot run until `conflict_object/conflict_type` are defined and persisted. It also does not define denominator, tokenizer, or pass/fail threshold. Concrete fix: define metric functions and thresholds before coding, e.g. `ungrounded_crisis_density = matches / total_voiced_words`, `distinct_conflict_types / episodes >= X`, and persist per-episode counts under `meta.story_quality` or a harness-only report.

SHOULD-FIX:
1. [L1a] “anchor beats on the REAL `allowed_roster` proper nouns” conflicts with composer grounding: `_build_user_prompt` uses `allowed_people` and `allowed_things`; comment says `allowed_roster` is still consumed downstream by `detect_phantom_names`, not prompt rendering. Concrete fix: specify conversion from `meta.allowed_roster` to `allowed_people`/`allowed_things` or use the already split fields if available. Verify writer call site.

2. [L1] Seed-keyed rotation is unspecified. Concrete fix: define stable key material and ordering, e.g. `sha256(f"{episode_seed}:{beat_index}:{domain}")`, sorted palette, modulo length. Avoid Python `hash()` because it is process-randomized unless hash seed is pinned.

3. [L2] “exactly one `personal_stake` BEFORE the first `irreversible_choice`” and “exactly one irreversible_choice as the last voiced beat” should be validator-backed, not prompt-backed. Concrete fix: add a deterministic validator over voiced beat order and fail/repair before compose, not after TTS.

4. [L2] The phrase “If a generated intent lacks the marker” is ambiguous. Marker could mean role tag, conflict object, sensory consequence, state-change, or private cost. Concrete fix: list markers per role and exact detection logic, preferably field-presence checks rather than regex over prose.

5. [L3] Bracket stripping can destroy legitimate spoken content like “[laughs]” intentionally voiced or bracketed acronyms/measurements. Concrete fix: only strip a line/action segment if it matches a conservative action pattern or only strip content after an explicit marker such as `ACTION:`. If using brackets, define whether nested/multiline brackets are handled.

6. [L4] “lowercase `announcer:` inside a character line” needs field-aware handling. Concrete fix: run sanitizer on final transcript line text only, never on speaker labels/ledger identity fields; define a regex anchored to beginning or quoted leakage, not any occurrence of “announcer:” in dialogue.

7. [Hard constraints #3] “unknown keys MUST be ignored by freeze/TTS/tests/serialization -- verify” is not a plan; it is a risk. Concrete fix: add an explicit compatibility test inserting unknown `meta`/`compose_flags` keys into a ledger and asserting freeze/TTS serialization ignores them byte-for-byte.

8. [Build order] “L1+L2 together” is correct conceptually but too large for safe implementation. Concrete fix: split into non-audio-affecting scaffolding first: data fields, deterministic selectors, validators, tests with flag off; then enable prompt/rendering under one flag.

9. [L5b] Bake-off design says “5 briefs x gemma-12b vs current default, scored” but does not define scorer after L5a. Concrete fix: specify deterministic metrics plus human/LLM critic if used, and make clear it is evaluation-only, not a generation gate.

OPTIONAL / NICE-TO-HAVE:
- Add a debug report per episode listing selected domain, conflict palette, per-beat role/object/type, and any deterministic crisis substitutions. This will make soak failures diagnosable.
- Add property tests for role allocation across beat budgets 1..N and phase counts.
- Keep the curated domain palette in a standalone UTF-8 data file with tests for duplicate/empty entries.

CUT THESE (over-engineering):
1. [L3] `internal_action` persistence in `meta`. Safe to cut initially because the only required behavior is stripping non-spoken action before freeze/TTS; persisting stripped action increases schema compatibility risk without improving audio/story output.

2. [L4] Mojibake repair. The plan already says VERIFY-ONLY and the issue may be a build artifact. Do not implement encoding repair until a real ledger/TTS path reproducer exists.

3. [L6] Best-of-N line selection. Safe to keep cut: it is expensive, structurally downstream of the beat problem, and operationally resembles the selection gate the constraints prohibit.