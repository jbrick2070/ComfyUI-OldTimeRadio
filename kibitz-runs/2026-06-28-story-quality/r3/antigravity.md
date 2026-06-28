VERDICT: no. Several critical interface contract mismatches, undefined variables, and division by zero risks will break the build or runtime execution.

MUST-FIX BEFORE BUILD:
1. [G1.3] (Dynamic cap) First-Pass/Reroll Dynamic Cap Mismatch.
   Defect: The plan adds `words_per_beat_range` to `LineRequest` and reconstructs it for rerolls in [nodes/_otr_reroll.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_reroll.py#L366-403), but fails to pass it in the first-pass `LineRequest` constructor call in [nodes/OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L4235). This causes the first pass to use the default `(0,0)` (falling back to static cap `28` in `flag_one_breath`), while the reroll uses the reconstructed dynamic cap, causing a validation divergence between the first pass and reroll.
   Fix: In [nodes/OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L4235), pass `words_per_beat_range=episode_budget.words_per_beat_range` inside the `_build_line_request_for_beat` helper when instantiating `LineRequest`.

2. [S2] (coda floor) Missing `arc_shape` Parameter in `compose_news_coda`.
   Defect: `compose_news_coda` selects templates from an `arc_shape`-keyed pool, but `arc_shape` is not passed to `compose_news_coda` in [nodes/OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L4770) or defined in its signature. Without it, the function cannot access `meta["arc_shape"]` to query the pool.
   Fix: Add `arc_shape: str = ""` to `compose_news_coda`'s signature in [nodes/_otr_line_composer.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_line_composer.py#L3278), and pass it from the writer in [nodes/OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L4770) using `arc_shape=str(meta.get("arc_shape") or "")`.

3. [S3] (body-gate accept) Undefined `run_on` in Body-Gate Scoring Formula.
   Defect: In [nodes/OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L4528), when both drafts are imperfect, the plan specifies a scoring formula: `score = 3*hard_leak + 2*is_truncated + 2*run_on + 1*roster_caps`. However, `run_on` is never defined or imported in the writer, causing a `NameError` at execution.
   Fix: Define `run_on` for both the original draft and the reroll draft in [nodes/OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L4528) as `1 if _OTRHY.flag_one_breath(text, max_words=max_words_cap)[0] else 0` where `max_words_cap` is computed from `episode_budget.words_per_beat_range`. Also, import `is_truncated` as `_OTRHY.is_truncated`.

4. [S2] (coda floor) Curated Coda Fallback ZeroDivisionError.
   Defect: When choosing a template from the curated pool, the plan selects it by `sha256(cast_seed)`. If `arc_shape` is invalid, not in the pool, or has no valid templates, `len(valid_templates_for_arc_shape)` will be `0`. Performing modulo `h % len(...)` will trigger a `ZeroDivisionError`.
   Fix: Guard the template selection in [nodes/_otr_line_composer.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_line_composer.py#L3321): if the selected `arc_shape` is not in the curated pool or has zero valid templates passing `validate_news_coda_bridge`, immediately fall back to selecting from the legacy `NEWS_CODA_POOL` to avoid division by zero.

5. [S3] (body-gate accept) Mid-Line Roster-Caps (Vocative) Stripping implementation missing.
   Defect: The plan requires adding "mid-line position tests" to the roster-caps rule (stripping full names only when they match a cast member's full name in ALL-CAPS). However, `scrub_roster_vocative` in [nodes/_otr_line_hygiene.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_line_hygiene.py#L1231) only implements leading and trailing vocative regex checks, neglecting mid-line names.
   Fix: Add a regex replacement for mid-line vocatives (e.g., `re.sub(rf",\s*{esc}\s*,", ",", out)`) to properly handle and strip mid-line matched names in [nodes/_otr_line_hygiene.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_line_hygiene.py#L1257-1264) without leaving duplicate commas or bad spacing.

SHOULD-FIX:
1. [G1.1] (Hint) Backwards Compatibility of `_QUALITY_COLLAPSE_HINT`.
   Defect: Globally changing `_QUALITY_COLLAPSE_HINT` in [nodes/_otr_line_composer.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_line_composer.py#L2293) to the new 133-char string might break existing unit tests that expect the old hint when v2 is disabled.
   Fix: Define a new `_QUALITY_COLLAPSE_HINT_V2` constant for the v2-enabled path and select between the two inside `_quality_reroll_hint` by passing `req` (or `req.story_quality_v2_enabled`).

2. [S2] (coda floor) Coda System Prompt Modification for Retry Attempts.
   Defect: If `story_quality_v2_enabled` is True, attempts increase from 2 to 3. However, `_msgs` is a helper that receives a boolean `retry`, meaning it cannot differentiate between attempt 2 and attempt 3, which could lead to redundant user prompt suffixes.
   Fix: Modify `_msgs` helper inside `compose_news_coda` in [nodes/_otr_line_composer.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_line_composer.py#L3305) to accept the attempt index/number (`attempt_idx: int`), adding customized user prompts for attempt 2 and attempt 3.

3. [G1.3] (Dynamic cap) Robust Parsing of `words_per_beat_range`.
   Defect: `words_per_beat_range` is stored in `meta`. When serialized to JSON, it becomes a list, which could cause type errors if treated as a tuple directly in Python code.
   Fix: Safely extract and coerce `words_per_beat_range` in both [nodes/_otr_reroll.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_reroll.py#L366) and [scripts/story_quality_scan.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/story_quality_scan.py#L387) by checking if the metadata key is list-like and has at least two numeric elements.

4. [S5] (voices) Scan-based Register Overlap Redefinition logic missing.
   Defect: The plan states to define "two principals" as the characters in `character_a_wants` / `character_b_wants` or the top two by dialogue count, but does not specify the updates required to the register overlap logic in [scripts/story_quality_scan.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/story_quality_scan.py#L431).
   Fix: Implement the "two principals" search in [scripts/story_quality_scan.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/story_quality_scan.py#L431) (by parsing the names from the dramatic state keys or accumulating line counts per speaker) and compute `register_overlap_ratio` using only those two speech signatures.

OPTIONAL / NICE-TO-HAVE:
- Add docstrings explaining the dynamic cap logic in `flag_one_breath` and `r3_quality_metrics` to ensure future developers understand the metric sync.

CUT THESE (over-engineering):
- None. (The plan already cuts S6 and defers S1).

[ASSUMPTION]:
- It is assumed that `character_a_wants` and `character_b_wants` names can be reliably extracted as the first word of the respective fields (e.g. up to the first space or word "wants") in the metadata structure.
