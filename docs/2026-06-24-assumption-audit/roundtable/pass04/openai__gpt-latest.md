<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes — substance converged, but BUILD 1/2/3 still have build-blocking ambiguities that would produce incompatible implementations.

MUST-FIX BEFORE BUILD:
1. [BUILD 2 / Sequencing] StoryContract seed and placement are ambiguous. The spec says build after news interpretation and before D.5, but `cast_seed` is not available until D.3 in the grounded writer flow. Implementors could use `0`, style RNG, or cast_seed. Concrete fix: build the StoryContract immediately after cast lock / `cast_seed` creation and before `OutlineRequest(...)`, using `cast_seed` as the `seed` argument and `script_brief or resolved["news_seed"]` as the premise/brief text.

2. [BUILD 2 / Scope + byte identity] The contract activation gate is not explicit. The top says byte-identical when contract absent / story-quality killed, but BUILD 2 reads like StoryContract is always built and stamped. Concrete fix: state the exact gate: “Only build/inject/stamp `StoryContract` when `_OTRCFG.style_grammar_enabled()` is true or the existing story-quality grammar lever is on; otherwise no `meta["story_contract"]`, no new prompt blocks, and all default fields remain empty.” If intended always-on, delete the byte-identical claim.

3. [BUILD 1 / Reroll semantics] Validator-fail reroll outcome is under-specified. The plan says one reroll, but not what to ship if reroll succeeds yet still fails validation. Concrete fix: specify deterministic selection, e.g. “If reroll returns and validates, ship reroll. If reroll returns but still fails, ship the original, stamp `body_gate_failed` and the original reasons, and do not append reroll text to `last_lines`.” Or explicitly choose “ship reroll” — but choose one.

4. [BUILD 1 / Reroll hint] “Hint built ONLY from offending tokens” conflicts with `missing_conflict_object`. If the line merely omits the conflict object, there may be no offending generic token, so the hint cannot repair the object miss. Concrete fix: split hints:
   - `ungrounded_crisis`: mention only offending generic tokens plus “replace them with the conflict object already named in THIS BEAT.”
   - `missing_conflict_object`: allow the grounded `sq_entry["conflict_object"]` in the hint, because it is deterministic/non-invented.
   Keep all hints derived only from validator facts and `sq_entry`.

5. [BUILD 1 / Helper contract] `validate_composed_grounding` rules are still ambiguous. “object match by HEAD NOUN / token overlap” has multiple valid interpretations. Concrete fix: define exact token rules:
   - use the same token regex as `_otr_story_quality_l12._TOKEN_RE` or expose a public tokenizer;
   - normalize by `casefold()`;
   - conflict-object match passes if the line contains either the conflict object head noun or at least one non-stopword token from `conflict_object`;
   - define head noun as the last non-stopword token after stripping possessive `’s/'s`.
   Also replace `{PRESSURE}` with the actual constant name `BEAT_ROLE_PRESSURE`.

6. [BUILD 3 / Role-keyed enrichment] The role-keyed tail map is not specified. “class-specific tail text” is not enough for a deterministic build or tests. Concrete fix: include the exact map text for `setup`, `pressure`, `personal_stake`, and every member of `CLIMAX_CLASS_ROLES`; explicitly omit `consequence`. Tests should assert exact fragments, especially that `revelation`, `reversal`, `confession`, and `quiet_acceptance` do not use the old personal-stake wording.

7. [BUILD 2 / K10] “DOMAIN_PALETTE scored matching” is under-specified. Current grounding shows ordered first-match selection, with `"trial"` appearing under medicine before law. Concrete fix: either keep first-match and only move/remove the law/medicine collision, or define the scorer exactly: tokenization, weights, tie-break order, fallback, and deterministic behavior. Add tests for “clinical trial” => medicine and “court trial” => law.

8. [BUILD 4 / Writer call] The plan adds `ending_tag` to `compose_announcer_outro`, but the grounded writer’s I.5 call currently passes only `ending_change` and `final_character_line`. Concrete fix: explicitly add `ending_tag=contract.ending_tag if contract else ""` at the I.5 call site, and ensure fallback placeholder for the last announcer beat also uses the non-resolving fallback policy when the tag is in the non-resolving set.

SHOULD-FIX:
1. [BUILD 1 / De-license prompt] “Scope composer 1162/1442 to compatible styles” is ambiguous. Define compatible styles/tags exactly, or state the default rule: remove/soften “mission control” and “Ground this line in the news facts” unless `contract.tags` or slug indicates procedural/newsroom/emergency. Otherwise two builders will make different prompt gates.

2. [BUILD 2 / StoryContract dataclass] `build_story_contract(premise, meta, seed)` uses “premise” but sequencing text says pass `script_brief if script_brief else resolved["news_seed"]`. Rename the parameter to `premise_text` or explicitly document that `script_brief/news_seed` is the first argument.

3. [BUILD 2 / F2 removal] Add a grep-level acceptance criterion: after implementation, `select_style(` may appear only inside `_otr_style_catalog.build_story_contract` and tests, not in `OTR_LedgerScriptWriter` after `generate_outline`.

4. [K7] Define the exact POSITION render after appending `ARC_PHASE_GUIDANCE`. Concrete fix: `POSITION: {req.position}\n  {ARC_PHASE_GUIDANCE[req.arc_phase]}` when both are present. This avoids prompt snapshot drift.

5. [BUILD 1 / Telemetry] Define whether `body_gate_failed` counts original validation failures, final shipped failures, or both. Concrete fix: use `body_gate_failed += 1` only when the shipped line still fails; use `body_gate_rerolls += 1` for attempted rerolls.

OPTIONAL / NICE-TO-HAVE:
- Add a small public helper in `_otr_story_quality_l12`, e.g. `offending_crisis_tokens(text, grounded)`, so BUILD 1 does not duplicate private tokenization logic.
- Add one prompt snapshot test for a style-grammar-on character beat showing `style_grammar`, `sound_world`, `story_engine`, `grounded_nouns`, and final-beat `ending_template`.

CUT THESE:
1. [BUILD 2 / K9] Cut “select_style best-fit wording” from this locked build unless there is a concrete prompt/function being edited. The selector is deterministic in the grounded catalog; wording tweaks are not load-bearing for the first production run.

2. [BUILD 2 / Consumer migration note] Cut any migration work beyond stamping `meta["story_contract"]`. The plan already says ADD-not-collapse and defer consumer migration; keep the build lean.

3. [BUILD 3 / consequence handling] Cut any enrichment or tests for `BEAT_ROLE_CONSEQUENCE` in this build. The plan correctly notes `assign_beat_roles` never assigns it under the current climax-last invariant.

VERIFY-AT-BUILD checklist:
1. [BUILD 2 / Pitch room] Verify `_otr_pitch_room.run_pitch_room` uses `dataclasses.replace` in a way that preserves new `OutlineRequest` fields: `style_grammar`, `sound_world`, `story_engine`, `ending_tag`.

2. [BUILD 2 / Style split] First run must log/dump both `resolved["style"]` and `contract.slug`; verify `resolved["style"]` still feeds `build_news_briefs`, cast, `meta["style"]`, and `meta["visual_plan"]["style"]`, while `meta["story_contract"]` is additive.

3. [BUILD 2 / No late select] Grep verify no `select_style(outline.premise, ...)` remains in writer F2 or anywhere after `generate_outline`.

4. [BUILD 2 / LineRequest sites] Verify every `LineRequest(...)` construction site and `_otr_reroll.build_reroll_line_request` populates/preserves new fields: `style_grammar`, `sound_world`, `story_engine`, `grounded_nouns`, and existing `ending_template`.

5. [BUILD 1 / Body gate placement] Verify the body gate runs after both `cleaned = _ex_text` and `cleaned = line_res.text`, but before `last_lines.append(...)`, `led.update_line_text(...)`, and `led.save()`.

6. [BUILD 1 / Reroll exception path] Force `compose_line` exception during guarded reroll; verify original text ships, `last_lines` receives original text only, and `grounding_reroll_failed` is stamped.

7. [BUILD 1 / Reroll invalid path] Force reroll to return invalid text; verify the chosen behavior from MUST-FIX #3 exactly.

8. [BUILD 1 / Palette] Verify `grounded_nouns` includes roster tokens, `resolved["news_seed"]`, `outline.premise`, and `premise_texts(meta)`; specifically test an object present only in `outline.premise`.

9. [BUILD 3 / Enrichment] Unit test each assigned climax-class role and confirm no non-irreversible climax gets personal-stake wording.

10. [BUILD 3 / Truncation] Unit test `_INTENT_MAX=200` behavior: original intent is truncated before tail append, and full tail survives.

11. [BUILD 4 / Non-resolving outro] For `ending_tag in {"unresolved_final_sound", "revelation", "quiet_acceptance"}`, verify prompt says do not resolve/state outcome and fallback never calls `_resolved_outro_fallback`.

12. [K7] Prompt snapshot verify `ARC_PHASE_GUIDANCE` appears in the POSITION block when `req.position` is truthy.

13. [BUILD 2 / K10] Verify domain selection tests cover “clinical trial” vs “court trial” and tie-breaks are deterministic.