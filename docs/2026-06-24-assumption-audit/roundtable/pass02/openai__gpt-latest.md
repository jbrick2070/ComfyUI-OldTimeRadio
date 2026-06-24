<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan identifies real seams, but the proposed fixes are under-specified and conflict with current call order, dataclass interfaces, and announcer/outro logic.

MUST-FIX BEFORE BUILD:
1. [K1] Body-output gate is placed too vaguely and will be ineffective if implemented “before freeze” as a late scrub. In `OTR_LedgerScriptWriter.run` the character branch appends `cleaned` to `last_lines` immediately after `compose_line`; a later ledger scrub means the bad line has already polluted subsequent prompts. Concrete fix: implement the gate inside the character branch immediately after `line_res = _OTRLC.compose_line(...)` and before `cleaned = line_res.text` is appended to `last_lines` / written to ledger. For `use_exchange=True`, gate `_ex_text` too or explicitly bypass exchange lines through the same validator.

2. [K1] “over threshold” and “conflict_object presence check” are undefined and not safely codable. `count_ungrounded_crisis(intent, grounded)` returns an integer but no threshold is specified; `conflict_object` is a multiword phrase like “the trial’s enrollment list”, so exact presence in dialogue will false-fail normal lines. Concrete fix: add a pure helper with explicit signature, e.g. `validate_composed_grounding(text: str, sq_entry: Mapping[str, Any], grounded: frozenset[str], *, max_ungrounded: int = 0, require_conflict_object_on_roles: frozenset[str] = CLIMAX_CLASS_ROLES | {BEAT_ROLE_PRESSURE}) -> tuple[bool, list[str]]`. For object matching, define normalized token overlap or require only the object head noun, not full-string equality.

3. [K1] The reroll path is underspecified for failure and flagging. `compose_line` can raise `LineCompositionFailedError`; the plan does not say whether to keep the first line, keep the failed reroll raw, or fail the episode. Concrete fix: one guarded reroll using existing `reroll_hint`; on reroll exception, keep the original text and append a compose flag like `grounding_reroll_failed:<reason>`. If reroll succeeds but still fails, ship rerolled or original deterministically and stamp `ungrounded_crisis:<n>` / `missing_conflict_object`.

4. [K1] “HARD banned-word hint built from `GENERIC_CRISIS_NOUNS`” is too broad and conflicts with `count_ungrounded_crisis`, which intentionally allows grounded generic nouns. Banning all terms would incorrectly ban premise-relevant words like “reactor” for energy stories. Concrete fix: build the reroll hint only from the offending tokens actually found in the composed text where `low in GENERIC_CRISIS_NOUNS and low not in grounded`.

5. [K1] “Mirror the existing stage-direction strip in `_otr_ledger_scrub`” is not grounded by the excerpts. The shown strip is inline in `OTR_LedgerScriptWriter.run` section I.6, not `_otr_ledger_scrub`. Concrete fix: either cite/verify `_otr_ledger_scrub` exists, or implement the grounding gate in the writer/composer path shown here, not in a nonexistent/unchecked scrub module. [ASSUMPTION] `_otr_ledger_scrub` may exist elsewhere; verify before referencing it.

6. [K2/K5] Selecting the catalog style “BEFORE `generate_outline`” conflicts with the current implementation, which selects `_style_slug` after outline generation using `outline.premise` in section F2. Concrete fix: select from `outline_req.script_brief or outline_req.news_seed` before constructing/calling `generate_outline`, store the selected slug/grammar/ending_tag in a concrete object, then pass that into `OutlineRequest` and later reuse the same object in F2. Do not reselect from `outline.premise` after the outline exists.

7. [K2/K5] The requested `LineRequest` fields `style_engine` / `sound_world` do not exist. `LineRequest` currently has `style_descriptor` and `ending_template` only. Concrete fix: add explicit fields to `LineRequest`, e.g. `style_grammar: str = ""`, `sound_world: str = ""`, `story_engine: str = ""`, and update `_build_user_prompt` to render them. Also update every construction site, reroll reconstruction path [VERIFY: `_otr_reroll.build_reroll_line_request`], and tests that instantiate `LineRequest`.

8. [K2/K5] `OutlineRequest` has no style grammar fields, and Path C prompts only render `req.style` in `_build_macro_user_prompt`; phase and beat prompts do not receive catalog grammar. Concrete fix: extend `OutlineRequest` with defaulted fields such as `style_grammar: str = ""`, `sound_world: str = ""`, `story_engine: str = ""`, `ending_tag: str = ""`. Render `style_grammar` in `_build_macro_user_prompt`, `_build_phase_user_prompt`, and `_build_beat_user_prompt`. Keep defaults empty for compatibility.

9. [K2/K5] “One `EpisodeStyle`/`StoryContract` threaded through outline/composer/announcer/critic/video/telemetry” is not an implementable interface. No class, module, serializer, ownership, or backward compatibility path is specified. Concrete fix: define a minimal frozen dataclass in `_otr_style_catalog`, e.g. `StoryContract(slug, label, sound_world, story_engine, ending_tag, ending_template, grammar)`, plus `build_story_contract(premise, meta, seed) -> StoryContract` and `model_dump()/as_meta_dict()` or plain dict conversion. First build should thread only writer -> outline -> composer -> meta. Leave critic/video for later.

10. [K3] Extending `_enrich_intent` to “all `CLIMAX_CLASS_ROLES` with CLASS-KEYED text” cannot be done by just widening the existing condition. `_enrich_intent` currently treats every non-`irreversible_choice` role as `personal_stake`. Concrete fix: replace the `if beat_role == irreversible_choice else personal_stake` branch with a mapping keyed by every role in `CLIMAX_CLASS_ROLES` plus setup/pressure/consequence. Add tests proving `revelation`, `reversal`, `confession`, etc. do not receive the personal-stake wording.

11. [K3] Intent enrichment is capped at `Beat.intent` max 200 chars via `new_intent[:_INTENT_MAX]`, so appending role text can silently truncate the load-bearing object/ending clause. Concrete fix: build the enrichment tail first, reserve space for it, and truncate the original intent, not the appended deterministic clause.

12. [Self-correction announcer close] “Do not resolve outcome” directly conflicts with `compose_announcer_outro`, which currently has F3 logic: when `is_resolved_ending_change(ending_change)` is true it instructs “State this outcome plainly” and recomposes if the line hedges. Concrete fix: define policy by `ending_tag`. For unresolved/revelation/quiet image styles, suppress `ending_change` or add a new `close_policy` parameter. For resolved-choice styles, keep F3. Add defaulted params to `compose_announcer_outro(..., ending_tag: str = "", style_slug: str = "", close_policy: str = "")` and make the F3 resolved-outcome branch conditional.

13. [Self-correction announcer close] Passing `ending_tag`/`style_slug` into `compose_announcer_outro` requires call-site and signature changes. Current writer call passes `ending_change` and `final_character_line` only. Concrete fix: after the pre-outline style contract is built, store `ending_tag`/`style_slug` in writer scope and pass them in section I.5; defaults must preserve existing tests/callers.

14. [K4] Early climax + denouement contradicts current invariants. `assign_beat_roles` forces last character beat to climax; `validate_beat_roles` raises unless climax is last; writer injects `ending_template` into `_climax_beat_id`; `compose_announcer_outro` reads the final character line as the resolution source. Concrete fix: do not attempt K4 in the same build. If kept, redesign role assignment to return both `climax_beat_id` and `post_climax_roles`, relax validator, update ending-template injection, and update outro logic to not assume final character line equals climax.

SHOULD-FIX:
1. [K7] The “`ARC_PHASE_GUIDANCE` is dead-coded” claim is only true for the line composer’s `elif req.arc_phase` branch when writer always sets `position`. It is not globally dead: `_otr_outline._phase_summary()` imports and uses `ARC_PHASE_GUIDANCE` in Stage 3 beat prompts. Concrete fix: do not delete `ARC_PHASE_GUIDANCE`. If the line composer needs phase-function prose, include `_phase_summary`/guidance inside `_position_for()` or add a separate `phase_focus` field.

2. [K8] `_PERSONAL_COST` only has `"general"`, so `fallback_content(role, domain, ...)` does not produce domain-specific personal stakes. Concrete fix: either add entries for every `DOMAIN_PALETTE` key or rename documentation/telemetry to stop claiming domain-specific personal cost.

3. [K9] `select_style` is not “best-fit” beyond emergency gating; it is a sha256 selection over a pool. Concrete fix: rename comments/docstrings to “deterministic style draw with emergency eligibility” or implement a scored selector over catalog tags/sound/story keywords.

4. [K10] `select_domain` is first-keyword-wins and `_DOMAIN_KEYWORDS` contains duplicate `"trial"` for medicine and law, with medicine winning. Concrete fix: replace first hit with scoring: count all keyword hits per domain, prefer highest score, deterministic tie-break by ordered domain list or seed.

5. [K11] `eng_ltx_av.py` style-driven render profile is not currently plumbed. The engine reads env constants for cfg/steps/sharp/negative and consumes only `plan["text_prompt"]`. Concrete fix: before changing the engine, identify the upstream request builder that creates `text_prompt` / motion request [VERIFY: video director/request shape]. Add style text to the prompt there first; only then add optional render-profile knobs if needed.

6. [K2] `render_style_grammar(slug)` currently returns four prose lines but no ending tag/template. If this becomes the canonical prompt injection, it should include `ending_tag` or the caller must separately render ending policy. Concrete fix: either extend grammar output with `Ending tag: ...` or keep `ending_template` as a separate final-beat-only field.

7. [K1] Telemetry for the new gate is unspecified. Concrete fix: stamp per-line flags and aggregate counts under `meta["story_quality"]`, e.g. `body_gate_ungrounded_crisis_count`, `body_gate_rerolls`, `body_gate_failed_after_reroll`, without overwriting existing `story_quality` keys.

8. [K1] The gate should run on shipped spoken text only. Current writer processes announcer, character, and non-voiced rows differently. Concrete fix: scope to `speaker_role == "character"` unless a separate announcer policy is defined; do not run crisis-noun gating on `[SFX:]` rows or music rows.

OPTIONAL / NICE-TO-HAVE:
- [K5] Add a ledger meta snapshot of the selected story contract `{slug, label, sound_world, story_engine, ending_tag}` for audits.
- [K2] Add a grep/test asserting `render_style_grammar` has at least one production caller.
- [K1] Add unit tests with the proven failure phrases: “red lever”, “fuel cells”, “mission control”, and a grounded exception where the premise legitimately contains “reactor”.

CUT THESE (over-engineering):
1. [K5] Cut “thread `StoryContract` through critic/video/telemetry” from the first build. It is broad and crosses modules not shown. Safe to cut because the proven failures are in outline/composer injection and shipped dialogue gating.

2. [K4] Cut early-climax + denouement from this build. It requires changing role invariants, ending-template targeting, and outro assumptions. Safe to cut because K1/K2 address the demonstrated NASA/lever/fuel-cell failures without structural climax-position redesign.

3. [Model-capability gate] Cut “prefer mistral / branch known-weak models” until a concrete model registry API and model capability table are specified. Safe to cut because K1’s deterministic output gate is the actual safety net and does not depend on model identity.

4. [K11] Cut style-driven cfg/steps/SHARP render profiles initially. Safe to cut because visual diversity can improve via prompt/style text first; sampler-profile changes are high-risk and engine-specific.