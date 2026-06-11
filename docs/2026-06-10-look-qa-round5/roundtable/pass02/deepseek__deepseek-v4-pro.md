<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? yes-with-fixes. The plan describes concrete, low-risk code changes; however, multiple assumptions about data availability and preconditions must be confirmed before a build that would pass the acceptance gate.

MUST-FIX BEFORE BUILD:
1. [F1] The eng_ltx_video.py insertion needs a robust `_env_int` helper (parse, clamp, default) not described in the plan; a missing or malformed env variable must not crash. Ensure the same cap is wired into eng_wan_i2v if its ask path is shared (plan says “verify at build” — that verification must be done).
2. [F2] The per-beat composition depends on `beat_intent` and `arc_phase` fields present on every line. The plan asserts they are “confirmed present on lines.” If ANY line in a real episode lacks them, the prompt will silently drop the beat clause, potentially losing intended variety. MUST validate this against a live ledger before building.
3. [F4] The self-vocative repair requires determining the “scene/exchange” and counting “exactly ONE other character row.” The plan does not specify how the writer obtains that exchange structure from the ledger lines. If the data model does not support that reliably, the repair will fail silently or misattribute. MUST confirm the available data structures (scene_id, exchange grouping, etc.) before implementing.
4. [F5] Announcer char_id resolution: the plan states “resolve from the cast table by NAME match (”ANNOUNCER“)” but does not specify where in ShotLock this normalization occurs. The join must happen before `_portrait_index` is used; if the announcer’s char_id remains missing, the LOUD warning will fire but no portrait will be assigned, possibly degrading the render. Implement and verify the cast lookup location.
5. [F6] Tests are listed but not written; a build without them risks regressions on the new critical paths (cap, prompt diversity, person-anchor, self-vocative). At minimum, the cap test, prompt diversity sha8 test, and the writer re-attribution test must be in place before the operator eyeball.

SHOULD-FIX:
1. [F2] The beat_intent mapping table (e.g., revelation→“a moment of revelation”) is small and fixed; if new intents are added later, the table will silently skip them. Consider a looser fallback (e.g., “a beat of {intent}”) or a clear log when an unknown intent is encountered, so operator can detect gaps.
2. [F4] The ShotLock backstop warning (“warn when a locked talking-head beat’s text starts with its own speaker’s name”) may produce false positives for lines like “John, I think …” that are properly attributed. Add a simple heuristic to exclude common non-self-vocative patterns (e.g., if the next word is a verb like “said” or “thought”).
3. [F5] The `build_request_from_shot` LOUD warning for missing portrait index is useful, but it could fire on synthetic beats (e.g., opening music) that legitimately have no char_id. Gate the warning on role being `character_video` (talking-head) to avoid noise.

OPTIONAL / NICE-TO-HAVE:
- The plan’s acceptance gate includes extracting spot frames and doing a per-second YAVG check; these are manual eyeball steps, not code changes.