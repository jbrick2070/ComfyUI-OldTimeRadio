VERDICT: yes-with-fixes. A1 core architecture has converged, but two concrete bugs in news source labeling and fenced JSON parsing in `finally` must be patched before build lock.

MUST-FIX BEFORE BUILD:
1. [_otr_writer_tail.py:1561-1569 / _build_news_payload:530-532]
   Defect: My Story news payload (`news_json`) and video HUD display falsely report `source: "RSS Auto-Fetch"`. In `nodes/_otr_writer_tail.py`, `headline_override` was updated to include `"my_story_fields"`, but `source_label` at line 1561 only checks `resolved["seed_source"] == "original_llm"`, passing `""` for `my_story_fields`. In `_build_news_payload` (line 531), when `source_label` is empty, `seed_source == "custom_premise"` evaluates to `False`, causing `source` to fall through to `"RSS Auto-Fetch"`. This violates `banks.json` (`source_material_label: "Listener story idea"`), `my_story.json`, and the user contract by claiming a listener's personal story was auto-fetched from RSS.
   Concrete fix: In `nodes/_otr_writer_tail.py:1561-1564`, set `source_label` for `my_story_fields` using `_bank_defaults.get("source_material_label")` or `"Listener story idea"`:
   ```python
   source_label=(
       "Original (LLM)"
       if resolved["seed_source"] == "original_llm"
       else (str(_bank_defaults.get("source_material_label") or "Listener story idea")
             if resolved["seed_source"] == "my_story_fields" else "")
   ),
   ```
   Alternatively, add `"my_story_fields": "User Story"` directly into `_build_news_payload` fallback at line 531.

2. [_otr_my_story.py:1121-1124]
   Defect: `finally` attempt counter parsing uses strict `json.loads` on raw model output, silently leaving `proposed_acts` and `proposed_characters` as `None` whenever an LLM emits markdown code fences. While `structured_call` uses tolerant `_otr_json.parse_first_json_object` to peel markdown code fences (` ```json ... ``` `), `nodes/_otr_my_story.py:1122` calls naive `json.loads(attempt["raw_output"])`. When a frontier model returns valid JSON enclosed in markdown fences, `json.loads` raises `json.JSONDecodeError` (a subclass of `ValueError`), which is caught by `except (ValueError, TypeError): continue`. The attempt is skipped and `story["counts"]["proposed_acts"]` / `story["counts"]["proposed_characters"]` remain `None` in the final saved ledger.
   Concrete fix: In `nodes/_otr_my_story.py:1121-1124`, use `_otr_json.parse_first_json_object(attempt["raw_output"])` instead of `json.loads(attempt["raw_output"])`.

SHOULD-FIX:
1. [_otr_my_story.py:1129]
   Defect: In-flight abort/cancellation exceptions can be overwritten if `led.save()` fails in `finally`. At line 1129, `_require_ledger_save(led, "the My Story attempt history")` is executed in `finally:`. If `run_my_story_episode` is interrupted (e.g., `KeyboardInterrupt`, `GenerationContextOverflowError`, or external worker abort) and disk persistence fails (e.g. disk full or read-only filesystem), `_require_ledger_save` raises a new `MyStoryError("ledger_save", ...)`, obliterating the original in-flight exception and root cause.
   Concrete fix: Guard the save in `finally` so that if `sys.exc_info()[0] is not None` and `led.save() is None`, it logs a warning instead of raising, preserving the active exception.

2. [tests/test_scifi_news_pro_tail_context.py:516]
   Defect: Missing assertion on `news_json` source provenance allowed the `"RSS Auto-Fetch"` leak to go undetected. In `test_my_story_cleanup_title_reaches_canon_file_wire_and_news`, `out[2]` (`news_json`) was checked for `headline == title`, but not for `source`.
   Concrete fix: Add `assert json.loads(out[2])[0]["source"] != "RSS Auto-Fetch"` to pin the provenance contract.

OPTIONAL / NICE-TO-HAVE:
1. [_otr_my_story.py:993-1000] In `fidelity_discrepancies`, deduplicate entries if the model listed the same character multiple times in `named_cast`.
2. [_otr_my_story.py:1057-1061] If a model puts the attribution sentence inside `frame.coda` rather than `intro` or `outro`, `coda` is omitted from `spoken_frame` check and attribution is re-appended to `outro`. Keeping it bounded to intro/outro aligns with prompt instructions, but checking `frame.coda` as well avoids potential duplicate attribution if a model writes it into the coda.

CUT THESE:
1. None. All dead/deprecated code identified in prior rounds (such as `MyStoryInputTooLongError`, redundant word-budget gating, and old title-override fallbacks) was already removed cleanly in this diff.

VERIFY-AT-BUILD checklist:
1. [VERIFY-AT-BUILD] Confirm on live model execution that local Qwen-4B treatment count mismatches (e.g., model proposing 2 acts for 1 act requested) successfully invoke `_full_artifact_repair` and repair cleanly on attempt 2 with the complete prior draft (>400 chars) present in the assistant message.
2. [VERIFY-AT-BUILD] Confirm that `meta.my_story.counts` in the persisted ledger contains non-null integers for `proposed_acts` and `proposed_characters` under real model runs that output markdown fences.
3. [VERIFY-AT-BUILD] Confirm that `news_used` JSON and video HUD display the source label as "Listener story idea" (or "User Story") and never "RSS Auto-Fetch".
4. [VERIFY-AT-BUILD] Confirm full-suite pass without any regressions in baseline comparisons against `c7cbf45b`/S4.
