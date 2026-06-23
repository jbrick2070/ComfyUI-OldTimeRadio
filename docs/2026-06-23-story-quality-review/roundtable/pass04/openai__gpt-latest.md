<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes — the plan is close, but two build-blocking ambiguities remain: the SQ flag contract is inconsistent across L12 vs L3/L4/L5a, and L5a recommends both cap scaling and an advisory critic without pinning exact behavior/output.

MUST-FIX BEFORE BUILD:
1. [Data model / L5a / Acceptance metric / Build order] Defect: SQ flag contract is contradictory. “Flags” says aggregate `meta.story_quality` if ANY SQ flag is on, but only names `OTR_STORY_QUALITY_L12`, `OTR_COMPOSER_ACTION_STRIP`, and `OTR_TRANSCRIPT_SANITIZER`; L5a telemetry is build step 1 and prerequisite, but no dedicated flag is named for L5a. Grounding shows current scrub uses `_sqv2_on`, but the plan does not define what env var drives it or whether L5a telemetry is written when only L5a is enabled. This can yield incompatible implementations: one writes `meta.story_quality` under L12 only; another writes it during L5a; another treats L3/L4 as SQ flags.
   Concrete fix: define one exact telemetry gate, e.g.:
   - `OTR_STORY_QUALITY_TELEMETRY` gates `_otr_ledger_scrub.py` aggregation of `meta.story_quality`.
   - “ANY SQ flag” means `OTR_STORY_QUALITY_TELEMETRY OR OTR_STORY_QUALITY_L12 OR OTR_COMPOSER_ACTION_STRIP OR OTR_TRANSCRIPT_SANITIZER`.
   - Build step 1 enables only `OTR_STORY_QUALITY_TELEMETRY` for telemetry tests; all audio/text mutation flags stay off.
   Or alternatively state L5a telemetry is unconditional after build. Pick exactly one.

2. [L5a(i)] Defect: cap scaling remains under-specified. “raise/scale the cap ceiling (e.g. scale by word count…) OR add…” plus “Recommend BOTH” does not give a single required formula. A builder can choose different caps and acceptance behavior, producing different terminal verdict rates.
   Concrete fix: replace with one formula and test cases. Example:
   `compute_edit_cap(voiced_beats, total_words=None) = max(3, min(20, max(voiced_beats // 2, ceil(total_words / 180))))`, with fallback `total_words=None -> max(3, min(12, voiced_beats // 2))`.
   Add explicit expected values for 6, 18, and 19 voiced beats and one dense EP16-like word count. If signature change is too invasive, keep the signature and specify `min(12, max(3, ceil(voiced_beats * 0.6)))`.

3. [L5a(i)] Defect: advisory grade-only critic before terminal stop is not wired precisely enough and risks contradicting the existing cascade invariant. Grounding shows `run_story_critic` currently runs in `_otr_freeze_cascade.py` after reviewer non-terminal flow; `too_many_edits` rollback/return happens inside `_otr_ledger_reviewer.py`, before the cascade has a critic opportunity. The plan says “add an ADVISORY grade-only `run_story_critic` BEFORE the terminal stop” but does not specify whether this call lives inside reviewer or cascade, what data snapshot it receives, or where the report is stamped after rollback.
   Concrete fix: pin location and persistence:
   - In `_otr_ledger_reviewer.py`, before `led.data.clear(); led.data.update(original_snapshot)` on `edits_applied == -1`, optionally call the critic on `candidate`/pre-rollback snapshot if and only if the needed `generate_fn` is available there; verify: reviewer currently has access to the story critic dependencies.
   - If reviewer does not have those dependencies, do not add advisory critic before terminal; instead require downstream consumers tolerate missing `meta.story_critic_report`.
   - If implemented, stamp only `meta.story_critic_report` onto `meta_after` after rollback, and do not change verdict/reroll/line text.
   This must be stated as a build decision, not left as “OR/Recommend BOTH”.

4. [L3 / L5a(ii) / Grounding `_otr_ledger_scrub.py:981-1011`] Defect: compose flag names are inconsistent. Current scrub counts `action_split:` / `action_split_failed:`. L3 plan says add `compose_flags` marker `action_strip:regex`. If scrub’s “ANY SQ” telemetry is extended to L3, an implementor may either add new counters, rename old action_split counters, or leave action_strip uncounted.
   Concrete fix: explicitly state:
   - Keep existing `action_split:*` counters unchanged.
   - Add separate `action_strip_regex` counter for exact flag `action_strip:regex`, or state L3 action-strip marker is not aggregated into `meta.story_quality` v0.
   - Do not reuse `l7_splits` for `action_strip`; it is a different mechanism.

5. [Outline build sequence step 3 / Data model] Defect: fallback beat factory mutation scope conflicts with “flag OFF => nothing populated” but not with “preserve outline JSON” when flag ON. The plan says same-beat replacement preserves IDs/counts/ranges and fills narrative + new meta fields, but does not define which original fields can be overwritten as “narrative”. It earlier names L1 mutable fields as `beat.intent` + SQ meta summary fields only, but fallback could reasonably alter `mood`, `sfx_cue`, or `target_words` within range.
   Concrete fix: pin fallback mutation allowlist. Recommended: fallback may mutate only `beat.intent`, `beat.mood` if already invalid/missing, and `beat.meta` SQ keys; it must not mutate `target_words`, `sfx_cue`, `speaker`, `speaker_role`, `arc_phase`, `beat_id`, or beat order. If `mood` should not change, exclude it explicitly.

SHOULD-FIX:
1. [Data model / Outline build sequence] Defect: `beat.meta` is required by the plan, but the grounding excerpt for `Beat` does not show a `meta` field. The plan assumes free-form `beat.meta` exists. If absent or if Pydantic forbids extras, build fails.
   Concrete fix: add to verify-at-build and implementation: verify `Beat` currently has `meta` or permits extras. If not, add `meta: dict[str, Any] = Field(default_factory=dict)` only if no-drift flag-off tests confirm defaults do not alter frozen JSON; otherwise store SQ data in an existing free-form outline meta structure. [ASSUMPTION] The excerpt may be truncated before the end of `Beat`.

2. [Data model] Defect: “If a top-level field is unavoidable, it MUST be `Field(default="", exclude=True)`” is unsafe as written for Pydantic schema/prompt generation. Even excluded fields may appear in generated JSON schema or validation instructions depending on how the outline parser prompts the LLM. This could change prompts with flag off.
   Concrete fix: make meta-only mandatory for v0. Delete the top-level fallback or require a prompt/schema byte-identical test in addition to `model_dump()` no-drift before using it.

3. [Outline build sequence step 2] Defect: “casefold+NFC-normalize inputs, ordered meta-field inspection” does not define the exact object/type candidate lists or tie-breaking after hashing. Different builders could choose different palette object ordering or domain extraction order.
   Concrete fix: specify selectors as:
   - candidate objects = sorted unique normalized allowed palette nouns for selected domain, with original display spelling retained by first occurrence;
   - hash maps to `index = int(hash,16) % len(candidates)`;
   - if empty, use deterministic fallback table entry for domain.
   Do the same for `conflict_type`.

4. [L1 crisis-noun repair] Defect: “normalized title/premise/logline nouns” lacks a noun extractor definition. That can create incompatible allowed palettes.
   Concrete fix: define v0 extractor as a deterministic regex/token filter, not NLP. Example: NFC/casefold tokens matching `[A-Za-z][A-Za-z'-]{2,}`, excluding stopwords and excluding all-caps speakers; plural normalization only by trailing `s`/`es` rules already used by the repair.

5. [L4] Defect: “Mojibake = CUT for v0 (verify-only)” lacks the concrete verify step. Earlier text says verify-only but does not say fail/warn criteria.
   Concrete fix: add verify-at-build item: run sanitizer tests over known mojibake samples and assert sanitizer does not modify them; optionally emit warning metric only. Production behavior: no repair, no transcript mutation.

OPTIONAL / NICE-TO-HAVE:
- [L1a] Filtering `"ANNOUNCER"`/`"NARRATOR"` from render while phantom gate receives union is fine, but mark as optional v0 only if not needed for the no-drift or hallucination metric.
- [Acceptance metric] Add threshold targets after first soak; current metrics define measurement, not pass/fail quality gates.

CUT THESE:
1. [Data model] Cut the “If a top-level field is unavoidable…” escape hatch. Meta-only is already the R3-corrected placement and avoids Pydantic serialization/schema drift.
2. [L5a(i)] Cut “Recommend BOTH” unless the advisory critic location is pinned. Cap scaling plus missing-report tolerance is sufficient to unblock measurement without invasive reviewer/cascade dependency changes.
3. [L1a] Cut optional filtering of `ANNOUNCER`/`NARRATOR` from render for v0 if it requires extra prompt golden churn; keeping union behavior is safe because the phantom gate already expects the union and this is not core to crisis-noun repair.
4. [L4] Cut all mojibake implementation language; retain only a verify/no-mutation test.

VERIFY-AT-BUILD checklist:
1. [Residual verify-at-build] Confirm the beat class is `Beat` only; no separate `OutlineBeat` or alternate outline row model must be patched. Current grounding shows `Beat`, but verify full source.
2. [Residual verify-at-build / Data model] Confirm the outline-to-ledger serialization path: whether `Beat.model_dump()` reaches frozen ledger and whether defaults in any new field would serialize. Required no-drift assert: flag off outline JSON and frozen ledger are byte-identical to baseline.
3. [Residual verify-at-build / L1a] Confirm `allowed_people` and `allowed_things` are populated at the writer call site today. Grounding only shows dataclass fields, not call-site wiring.
4. [Data model] Verify `Beat` has a `meta` dict or supports free-form extra fields. If not, add a safe meta field and rerun prompt/schema/JSON no-drift tests. [ASSUMPTION] Grounding excerpt may omit later fields.
5. [L5a(i)] Verify whether `_otr_ledger_reviewer.py` has access to `generate_fn` / story critic dependencies before implementing any pre-terminal advisory critic. If not, implement only cap scaling plus missing-report tolerance.
6. [L5a(ii)] Verify scrub aggregation runs after cascade rollback/restore on the final persisted ledger rows. Add a test where `objective_literal_retry` appears before a rollback path and assert counts match saved rows, not discarded rows.
7. [L5a(ii)] Verify `_meta.setdefault("story_quality", {}).update(...)` preserves injected unknown `meta.story_quality` keys.
8. [Acceptance metric] Verify unknown `meta.story_quality` keys and new `compose_flags` are ignored by freeze/TTS/serialize/hash paths.
9. [L3] Verify `ACTION:` stripping runs after compose/polish returns text and before line persistence/`compose_flags`; assert only explicit `ACTION:` segments are stripped and speaker/text fields remain otherwise unchanged.
10. [L4] Verify transcript sanitizer runs after final text generation and before freeze/TTS/hash/golden; assert it never mutates speaker labels or identity fields.
11. [L4] Verify mojibake samples are not repaired/mutated in v0; warning/metric only if implemented.
12. [Flags] Verify all new flags default off and flag-off run produces byte-identical prompts, outline JSON, frozen ledger JSON, and no `meta.story_quality` key unless the finalized telemetry contract explicitly says otherwise.