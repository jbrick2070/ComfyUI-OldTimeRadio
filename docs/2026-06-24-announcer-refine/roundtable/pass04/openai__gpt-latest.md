<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes — close, but build is still blocked by one real wiring gap in phase/beat style transport and several underspecified coda/contract details that can produce incompatible implementations.

MUST-FIX BEFORE BUILD:
1. [STEP C] `_build_phase_user_prompt` / `_build_beat_user_prompt` cannot receive `contract.story_engine` from their internal call sites unless it is carried through `OutlineRequest`. The plan only adds `style_grammar`, so `contract` is out of scope inside `_otr_outline.py`. Concrete fix: add `story_engine: str = ""` to `OutlineRequest`; at writer call site set `story_engine=(contract.story_engine if contract else "")`; inside outline generation pass `req.story_engine` to `_build_phase_user_prompt` and `_build_beat_user_prompt`.

2. [STEP F] Flag-on coda path can still accidentally include `script_brief` because the current `compose_announcer_outro` builds `user_parts` from `brief` first, and the new call still passes `script_brief=script_brief`. That reintroduces fictional-story contamination into the real-news coda. Concrete fix: in `compose_announcer_outro`, when `story_scaffold` is true, do not add `Tonight's story brief` to `user_parts` and do not use `brief` in fallback selection. Use only `news_close_brief`, optional `intro_text` if retained, `climax_character_line or final_character_line` for tone, and `ending_change` only as forbidden content.

3. [STEP F] `<fixed lead-in>` is still a placeholder, but validation, fallback, and final rendered text depend on its exact string. Two builders can produce different outputs and different validator behavior. Concrete fix: define one constant, e.g. `NEWS_CODA_LEAD_IN = "And in the real story,"`, and use that exact value at the outro call site, in `validate_news_coda_line(... lead_in=...)`, and in `fallback_news_coda_outro`.

4. [STEP B/C/G] `StoryContract` consumed fields are not fully specified in this locked document. STEP C consumes `contract.story_engine`; STEP G consumes `contract.ending_template`; STEP B only shows meta fields and says “see pass02 §0.” Concrete fix: restate the required dataclass fields here before build: `slug`, `label`, `grammar`, `story_engine`, `ending_tag`, `ending_template`. Ensure `build_story_contract(...)` populates all of them deterministically from `cast_seed`, `script_brief/news_seed`, and `meta`.

5. [STEP F] `fallback_news_coda_outro(coda_lead_in, "")` for empty `news_close_brief` is underspecified and may emit a dangling lead-in with no fact body. Concrete fix: define the exact deterministic empty-close fallback text and compose flag. Example requirement: if `story_scaffold and not close`, return `LineResult(text=fallback_news_coda_outro(coda_lead_in, ""), compose_flags=("news_coda_fallback", "news_coda_empty_close"))`, and `fallback_news_coda_outro` must guarantee a complete sentence even with empty body.

6. [STEP H] KILL-4 still depends on “pass02 §4” and “real constants l12:55-72” for the role-keyed map and truncation reserve formula. [ASSUMPTION] If pass02 §4 is not part of the build ticket handed to the coder, this is build-blocking. Concrete fix: paste the exact role map and reserve/truncation formula into STEP H/C4, including which `CLIMAX_CLASS_ROLES` members receive which fallback content and confirming consequence remains deferred.

SHOULD-FIX:
1. [STEP F] `validate_news_coda_line` says “strong content-token overlap with `ending_change`” but gives no threshold/tokenization. Concrete fix: define deterministic tokenization and threshold, e.g. lowercase alnum tokens length >= 4, stopword-filtered; reject if overlap with ending tokens is `>= 3` tokens or `>= 40%` of body content tokens, whichever is smaller. Otherwise validators will diverge.

2. [STEP D / BYTE-IDENTITY] STEP D code appears unconditional, while BYTE-IDENTITY says D skips off. It likely does not affect audio, but it violates the stated “none of the new branches run” invariant. Concrete fix: initialize `safe_open_brief = None`; only construct `SafeOpenBrief` under `if _style_grammar_on:`.

3. [STEP D] SafeOpenBrief cast source uses `led.data.get("cast")` immediately after `generate_outline`. Grounding does not prove `led.data["cast"]` is populated at that point. Concrete fix: either use the already-available cast structure used to build `OutlineRequest.character_cast`, or add a build-time assertion/test that `led.data["cast"]` is populated before STEP D. [ASSUMPTION]

4. [TELEMETRY] Telemetry key notation `meta.story_quality.{...}` is inconsistent with the dict usage elsewhere. Concrete fix: specify exact implementation: `sq = meta.setdefault("story_quality", {})`; then assign primitive keys `story_contract_slug`, `news_coda_emitted`, `news_coda_fallback`, `open_safe_fallback`.

5. [BYTE-IDENTITY] “VERIFY: any test comparing OutlineRequest asdict/repr -> update fixture or gate serialization” weakens the byte-identity claim. Concrete fix: prefer updating only internal test fixtures; do not alter any persisted ledger/meta serialization off-flag. Add an explicit off-flag ledger JSON golden check.

OPTIONAL / NICE-TO-HAVE:
- [STEP F] Under `story_scaffold`, consider omitting `intro_text` from the coda prompt as well. The coda’s job is the real fact body; intro echoing is legacy outro behavior and can increase fiction bleed.
- [STEP E/F] Add one deterministic unit test each for intro and coda fallbacks with deliberately spoiler-heavy `script_brief` to prove flag-on paths never read it for emitted text.

CUT THESE:
1. [VERIFY-AT-BUILD] `build_reroll_line_request` check is safe to cut from the first-build checklist because the per-line compact register is explicitly cut. Keep only a note: “re-check if line-level register is reintroduced.”

2. [STEP F] Cut any story-scaffold use of `script_brief` inside coda composition. It is not needed for the news coda, and removing it reduces spoiler/fiction contamination risk without losing the goal.

3. [BUILD CHUNKS] “commit AND push to v2.0-alpha per green chunk” is process, not build logic. Safe to move out of the implementation spec if the builder already follows release procedure.

VERIFY-AT-BUILD checklist:
- [STEP A / BYTE-IDENTITY] With `story_scaffold` off, verify open line, outro line, ledger meta, and audio outputs are byte-identical to baseline.
- [STEP C] Verify `OutlineRequest` has both `style_grammar` and `story_engine` defaults as empty strings, and that off-flag prompt rendering omits both.
- [STEP C] Verify macro prompt receives `contract.grammar`; phase and beat prompts receive `req.story_engine`; no `contract` reference exists inside `_otr_outline.py` call sites.
- [STEP D] Verify concrete `era` source for `SafeOpenBrief`; default `""` is acceptable if no period metadata exists.
- [STEP D] Verify cast data used for `SafeOpenBrief.cast` is populated at capture time, or switch to the earlier cast source. [ASSUMPTION]
- [STEP F] Verify ledger line `beat_id` exists on climax character lines before outro patching; test reversed lookup selects the last line for `_climax_beat_id`.
- [STEP F] Verify `news_close_brief` is distinct from `ending_change`; if empty, the loud fallback path triggers and never calls the LLM.
- [STEP F] Verify story-scaffold coda prompt does not include `script_brief` and does not enter `_resolved_outro_fallback`.
- [STEP F] Verify `validate_news_coda_line` runs on raw LLM body before prefixing and rejects bodies that already contain the lead-in.
- [BYTE-IDENTITY] Verify any `OutlineRequest.asdict/repr` snapshot fixtures are intentionally updated and no off-flag persisted outputs gain new fields.
- [TELEMETRY] Verify `meta["story_quality"]` exists only under flag and contains primitives only.
- [STEP H] Verify KILL-4 enrichment covers setup, pressure, personal_stake, and every non-deferred `CLIMAX_CLASS_ROLES` member; consequence remains deferred; truncation uses the new reserve/clamp path.