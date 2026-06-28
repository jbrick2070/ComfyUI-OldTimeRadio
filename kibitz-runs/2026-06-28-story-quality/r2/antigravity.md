VERDICT: yes-with-fixes. Multiple critical LineRequest reconstruction gaps in _otr_reroll.py and potential runtime crashes in OTR_LedgerScriptWriter.py and compose_news_coda.

MUST-FIX BEFORE BUILD:
1. [Section 0 / S3] Missing `entity_policy` reconstruction in reroll
   - Defect: `_otr_reroll.py` (`build_reroll_line_request` L366-403) does not reconstruct the `entity_policy` field on `LineRequest` from `meta`. Because the `leak_floor_v2` validator gates on `req.entity_policy is not None` (`_otr_line_composer.py` L2572), all leak checks and strips (malformed quotes, participle extracts, roster name leaks, banned entities) are silently skipped on all targeted-reroll recompositions. [ASSUMPTION] We infer that `EntityPolicy` is required by the `leak_floor_v2` validator and can be reconstructed from the `leak_floor_v2` key in `meta`.
   - Fix: Import `EntityPolicy` dynamically inside `_otr_reroll.py`, parse the `leak_floor_v2` configuration from `meta.get("leak_floor_v2")`, reconstruct the `EntityPolicy` using the reconstructed `allowed_roster` and the banned list from `meta`, and pass it as the `entity_policy` keyword argument when instantiating `LineRequest` in `_otr_reroll.py:366`.

2. [Section 0 / S1] Missing `speaker_gender` reconstruction in reroll
   - Defect: `build_reroll_line_request` in `_otr_reroll.py` fails to reconstruct the `speaker_gender` field of `LineRequest`. During recomposition of dialogue lines, the prompt generator will lack the gender/pronouns context (which is used in first-pass compile to map titles and pronouns correctly), causing potential "Mister <female>" style regression issues on rerolled lines.
   - Fix: Look up the character's gender in `build_reroll_line_request` using `cast_row.get("gender")` from the matching character row in `cast_rows` and pass it as `speaker_gender` to `LineRequest`.

3. [Section S3] Missing import in `OTR_LedgerScriptWriter.py`
   - Defect: The proposed defect-count logic in `OTR_LedgerScriptWriter.py:4528` calls `is_truncated` to evaluate grammaticality, but the file does not import `is_truncated` from `nodes\_otr_line_hygiene.py`, which will cause a runtime `NameError` and fail the build.
   - Fix: Add `from ._otr_line_hygiene import is_truncated` to the imports of `OTR_LedgerScriptWriter.py`, or import it dynamically within the body-gate block.

4. [Section S2] Potential `ZeroDivisionError` in news coda premise-template selection
   - Defect: The proposed premise-template strategy extracts phrases from `premise` and rotates through them using `sha256(cast_seed) % len(valid_phrases)`. If the premise is empty or no phrase passes `validate_news_coda_bridge`, `len(valid_phrases)` will be 0, causing a `ZeroDivisionError` crash.
   - Fix: Add a safety check: if `len(valid_phrases) == 0`, fall back to the original `NEWS_CODA_POOL` logic.

5. [Section S2] System prompt example gating
   - Defect: Modifying the module-level string constant `_NEWS_CODA_SYSTEM` in `nodes\_otr_line_composer.py` to append 1-2 in-context examples would modify the prompt globally, which violates the strict flag-OFF byte-identity requirement when `story_quality_v2_enabled` is False.
   - Fix: Keep `_NEWS_CODA_SYSTEM` unchanged. Inside `compose_news_coda`, copy the system prompt to a local variable and dynamically append the in-context examples only when `story_quality_v2_enabled` is True.

6. [Section S3 / S1] Missing L1/L2 beat shaping variables in reroll
   - Defect: `build_reroll_line_request` fails to reconstruct `beat_role`, `conflict_object`, and `conflict_type` on `LineRequest` from `meta`. This means any first-pass quality checks relying on these fields (such as conflict object presence or role-based templates) will not execute or will produce empty/divergent output on targeted rerolls, violating determinism. [ASSUMPTION] We infer that the `story_quality` metadata in `meta` contains the parsed `by_beat` dictionary containing L1/L2 shaping fields.
   - Fix: Extract these fields in `_otr_reroll.py` from `meta.get("story_quality", {}).get("by_beat", {}).get(line_id)` and populate `beat_role`, `conflict_object`, and `conflict_type` in `LineRequest`.

SHOULD-FIX:
1. [Section 0 / S1] Missing `allowed_things` and `current_beat_block` in reroll
   - Defect: `LineRequest` fields `allowed_things` and `current_beat_block` are not reconstructed in `_otr_reroll.py:366`, meaning the composer's prompt rendering on targeted rerolls will differ from the first-pass compose, resulting in different line outputs even for identical seeds.
   - Fix: Reconstruct `allowed_things` by reading `meta.get("news", {}).get("key_terms")` and filtering out `meta.get("leak_floor_v2", {}).get("filtered_key_terms")`. Reconstruct `current_beat_block` by splitting `meta.get("outline_spine")` and selecting the line matching the target `line_id`. Pass both to the reconstructed `LineRequest`.

2. [Section S4] Cliche capitalization mismatch
   - Defect: Replacing a cliche (like `"over my dead body"`) with a lowercase string from the curated safe-replacement map (like `"not while I'm here"`) will produce a capitalization error if the cliche was at the start of the sentence (e.g. `"Over my dead body, Lemmy."` -> `"not while I'm here, Lemmy."`).
   - Fix: Implement a capitalization matcher in the cliche replacement logic to match the casing (e.g. title case or capitalized first letter) of the original matched phrase.

3. [Section S1] Empty anchors list infinite loop/reroll exhaustion
   - Defect: If an episode has no specificity anchors defined, the off-premise window-level detector will see zero anchors in the last N lines on every beat, causing it to continuously trigger rerolls and exhaust the reroll budget for the entire script.
   - Fix: Gate the off-premise window detector so it is skipped entirely if `len(anchors) == 0`.

4. [Section S5] Undefined "two principals"
   - Defect: The concept of "two principals" is not formally defined on characters or casts. The developer will have to invent an algorithm to select the two principal characters to run the measurement-only `register_overlap` counter. [ASSUMPTION] We infer that the wants in the dramatic state are prefixed by character names (e.g. 'NAME: wants') or that characters can be mapped via dialogue counts.
   - Fix: Explicitly define "the two principals" as the characters whose wants are captured in the dramatic state `character_a_wants` and `character_b_wants` (by parsing the name prefix if present) or default to the top two characters by dialogue line count.

OPTIONAL / NICE-TO-HAVE:
1. [Section S1 / 0] Persistence of `sfx_cue` on character lines: Add `sfx_cue` to the ledger line schema in `nodes\production_ledger.py` so it can be reconstructed during rerolls, preventing prompt mismatch for lines that contain a background sound environment cue.
2. [Section S3] Custom defect score tuning: When counting defects to choose the better draft, assign higher weight to `leak_floor` and `is_truncated` defects than to stylistic quality flags (e.g. `cliche` or `on_the_nose`).

CUT THESE (over-engineering):
- None. (The spec already pruned phantom name checks and register-divergence loops).
