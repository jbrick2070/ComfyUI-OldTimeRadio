<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? **no**. The plan has concrete implementability gaps in data contracts, truncation logic, and API surface that will produce broken code if coded as described.

---

MUST-FIX BEFORE BUILD:

1. **[§1 / §8] `opening_status_quo` is undefined in the outline schema.**
   The plan requires `OutlineRequest` to carry `opening_status_quo` (a NEW outline field, §2 Job 1) and the outline to emit it (§8 step 5). The grounding shows `OutlineRequest` fields at `_otr_outline.py:283` — `opening_status_quo` does not exist there. The outline MACRO produces `premise`, `setting`, `time_of_day` but NOT `opening_status_quo`. The plan provides no schema change, no prompt instruction to the outline LLM to produce it, and no extraction logic. **Fix:** Add `opening_status_quo: str = ""` to `OutlineRequest`; add a prompt instruction in the outline macro user-prompt to emit "the situation at the START of the episode"; add extraction in the outline parser; add the field to the outline dataclass.

2. **[§1 / §8] `SafeOpenBrief` data shape is never defined.**
   The plan says "Build a `SafeOpenBrief` from outline + contract: era/`time_of_day`, `setting`, cast names+roles, `opening_status_quo`, contract tone." No struct, no constructor signature, no field types. `compose_announcer_intro` currently takes `script_brief: str` — the plan says "SEVER `script_brief` from `compose_announcer_intro` under `story_scaffold` (new structured params)" but never specifies the replacement signature. **Fix:** Define `SafeOpenBrief` as a frozen dataclass with fields: `era: str`, `time_of_day: str`, `setting: str`, `cast_names: list[str]`, `cast_roles: list[str]`, `opening_status_quo: str`, `tone: str`. Change `compose_announcer_intro` signature to accept `safe_open_brief: SafeOpenBrief | None = None` under the flag, with `script_brief` retained for the off-flag path.

3. **[§1] `build_story_contract` signature and return type are never specified.**
   The plan says "one frozen `StoryContract(slug,label,sound_world,story_engine,ending_tag,ending_template,grammar)` + `build_story_contract` in `_otr_style_catalog`, built ONCE after cast-lock (seed=`cast_seed`) + news interpretation, BEFORE `OutlineRequest`, from `script_brief or news_seed`." The `StoryContract` fields are listed but no constructor signature, no return type, no indication of how `grammar` differs from the other fields (it appears to be the rendered string from `render_style_grammar` but that function takes `slug` — circular). **Fix:** Define `StoryContract` as a frozen dataclass with explicit typed fields. Define `build_story_contract(cast_seed: int, script_brief: str, news_seed: str) -> StoryContract`. The `grammar` field should be the rendered prose string, built by calling `render_style_grammar(slug)` internally.

4. **[§1] `meta.story_contract` — where does `meta` live and what type is it?**
   The plan says "ADD `meta.story_contract`; do NOT overwrite `resolved["style"]`/`meta.style`/`visual_plan.style`." The grounding shows `meta` used in `select_style(premise, meta, cast_seed)` and `meta.dramatic_state.ending_change`, but never defines the `meta` object's class or where `story_contract` would be attached. Is this `OTR_LedgerScriptWriter`'s local `meta` dict? The `LedgerMeta` dataclass? **Fix:** Identify the exact `meta` object in scope at the point where `build_story_contract` is called (the `run()` scope at `_otr_ledger_script_writer.py`). Add `story_contract: StoryContract | None = None` to that object's type. If it's a dict, use a string key `"story_contract"`. Be explicit.

5. **[§1 / §8] `OutlineRequest` style fields — what exactly is added?**
   The plan says "Add style fields to `OutlineRequest` (rendered in `_build_macro/phase/beat_user_prompt`)." The grounding shows `OutlineRequest` already has a `style` field (`:303`, user-selected style string). The plan says "do NOT overwrite `resolved['style']`/`meta.style`." So what NEW fields carry `story_engine` and `ending_mode`? Are they `story_engine: str = ""` and `ending_mode: str = ""`? The plan never names them. **Fix:** Add `story_engine: str = ""` and `ending_mode: str = ""` to `OutlineRequest`. These are distinct from the existing `style` field. Specify that `_build_macro_user_prompt` (and phase/beat equivalents) render these fields into the prompt text under the flag.

6. **[§3] KILL-4 truncation order fix is underspecified and will break.**
   The plan says: "truncate the ORIGINAL intent to `_INTENT_MAX - len(enrichment)` FIRST, then append the enrichment (reserve the tail); if reserve is negative, truncate the enrichment slot." The grounding shows `_INTENT_MAX` is 200 (`:800`). The current code does `new_intent = new_intent.strip()[:_INTENT_MAX].strip()` AFTER enrichment. The fix requires: (a) compute `enrichment_text` BEFORE the truncation, (b) `reserve = _INTENT_MAX - len(enrichment_text)`, (c) if `reserve < 0`, truncate `enrichment_text` to `_INTENT_MAX`, (d) truncate original intent to `max(0, reserve)`, (e) append. But the plan says "if reserve is negative, truncate the enrichment slot" — truncate to WHAT length? `_INTENT_MAX`? Zero? **Fix:** Specify: if `reserve <= 0`, enrichment_text = `enrichment_text[:_INTENT_MAX]`, original_intent = `""`. If `reserve > 0`, original_intent = `original_intent[:reserve]`, then `final = (original_intent + " " + enrichment_text).strip()[:_INTENT_MAX]`. Provide the exact code block.

7. **[§2 Job 3] `_ANNOUNCER_OUTRO_SYSTEM` rewrite — the two-part close structure is described narratively but the actual prompt text is not specified.**
   The plan says "Rewrite `_ANNOUNCER_OUTRO_SYSTEM` to a two-part close: (a) the character-close reflection (keep the concrete-image discipline), then (b) a deliberate pivot to the REAL news as a plain fact." The implementor must write the actual system prompt string. The current prompt FORBIDS news-summary framing. The new prompt must explicitly ALLOW and STRUCTURE the pivot. Without the exact prompt text, the implementor will guess and likely produce a prompt that still fights the coda. **Fix:** Provide the exact replacement string for `_ANNOUNCER_OUTRO_SYSTEM` under the flag, or at minimum specify: (1) the concrete-image rule applies ONLY to part (a), (2) part (b) is explicitly permitted to state a fact/lesson, (3) the pivot between them is marked by the deterministic lead-in prefix.

8. **[§2 Job 3] The deterministic lead-in is injected as a PREFIX but the injection point is ambiguous.**
   The plan says "inject the lead-in as a PREFIX, not LLM discretion." Does this mean: (a) the lead-in is prepended to the LLM's generated text in post-processing, OR (b) the lead-in is included in the user prompt as a required opening phrase the LLM must output? If (a), the validator must check that the LLM didn't ALSO generate a lead-in. If (b), weak models may still blend. **Fix:** Specify: the lead-in is a POST-GENERATION prefix — the LLM generates only the news fact text; the composer prepends the lead-in. The validator checks that the LLM output does NOT contain any lead-in variant. The fallback uses the same prefix + `news_close_brief`.

9. **[§2 Job 1] The spoiler post-gate "token-overlaps the `ending_change` / `news_close_brief` outcome vocabulary" is vague.**
   "Token-overlaps" — what tokenization? Word-level? Subword? What overlap threshold triggers rejection? Exact substring match? Shared nouns? The plan says "reroll once; else the structured fallback" — but without a precise gate definition, the implementor cannot code it. **Fix:** Define the gate as: extract the set of content words (nouns, verbs, adjectives) from `ending_change` and `news_close_brief` (stoplist-filtered, lowercase). If the generated open shares ≥ N content words (suggest N=2 for a 15-30 word line), reject. Provide the stoplist or reference an existing one in the repo.

10. **[§2 Job 2] `climax_character_line` — the extraction logic is described but the handoff to `compose_announcer_outro` is not.**
    The plan says "pass the CLIMAX beat's line as `climax_character_line`." The grounding shows `compose_announcer_outro` already has a `final_character_line` parameter. The plan says to add `climax_character_line` — is this a NEW parameter, or does it REPLACE `final_character_line` under the flag? The plan says "Today climax==last => byte-identical" which implies the new param gets the same value as the old one today. **Fix:** Under the flag, add `climax_character_line: str = ""` to `compose_announcer_outro`. The call site passes the climax beat's line text. The function uses `climax_character_line` in the prompt instead of (or in addition to) `final_character_line`. Specify which.

---

SHOULD-FIX:

1. **[§1] `select_style` deletion — "FIRST verify no other caller" is a build-time instruction, not a spec.**
   The plan says delete the late `select_style(outline.premise, meta, cast_seed)` at `:3224` but "FIRST verify no other caller; if other callers exist, stop calling it from the outline path only." This pushes a discovery task to the implementor. **Fix:** The author should grep the repo for `select_style` callers before finalizing the plan. If there are other callers, specify how they migrate.

2. **[§1] `StoryContract` reuse in F2 — "Reuse in F2 (delete the late `select_style`...)" — what is F2?**
   The plan references "F2" without definition. From context it appears to be the outline path. **Fix:** Replace "F2" with the actual function/module name or remove the reference.

3. **[§2 Job 3] Coda length budget "18-45 words" — the existing validator `validate_announcer_line` uses `_ANNOUNCER_OUTRO_MAX_CHARS` (character count, not word count).**
   The plan specifies a word-count band but the existing validation is character-based. **Fix:** Either specify the character equivalent, or add a word-count check to the validator under the flag, or acknowledge that the existing char-based validator will be used with adjusted constants.

4. **[§5] OFF-flag golden-output tests — "NO changed fallback/outro text" but `fallback_announcer_intro` currently echoes `script_brief`.**
   If the plan SEVERS `script_brief` from `compose_announcer_intro` under the flag, the fallback must also change under the flag (it currently reads `script_brief`). The plan says "NO changed fallback" when off — correct — but does not specify the ON-flag fallback behavior. **Fix:** Specify: under the flag, `fallback_announcer_intro` takes `SafeOpenBrief` instead of `script_brief` and builds its text from those fields.

5. **[§6] Telemetry field `open_spoiler_rerolls` — the plan only allows ONE reroll.**
   The field name is plural ("rerolls") but the spec says "reroll once." **Fix:** Rename to `open_spoiler_reroll` (singular) or allow N rerolls and keep plural.

---

OPTIONAL / NICE-TO-HAVE:

- The `era` field for `SafeOpenBrief` — the plan says "era source = verify-at-build, likely meta/period." This is fine as a build-time discovery, but the plan should note that if `meta.period` doesn't exist, `era` will need a fallback (e.g., from the style catalog's period or a default).
- The lead-in variant set "3-5 recognizable variants" — the plan says "optionally a small CLOSED seed-keyed set." Make this deterministic: seed-keyed selection from a hardcoded tuple, so it's reproducible.

---

CUT THESE (over-engineering):

1. **[§1] "premise-specific conflict objects beyond the domain pool" is already DEFERRED — no cut needed.**

2. **[§2 Job 1] The "BELT (post-gate)" with reroll logic adds complexity for marginal gain given the input-starvation primary mechanism.**
   If the open prompt truly contains no outcome-bearing fields, the gate should never fire. The reroll + fallback chain is defensive depth that may never be exercised. Consider: skip the belt entirely, rely on input starvation alone, and add the gate only if re-soak shows leaks. Safe to cut the gate for the first build; add it if needed.

---

[ASSUMPTION] The `meta` object referenced for `meta.story_contract` is the same `meta` dict/object passed to `select_style(premise, meta, cast_seed)` at `:3224`. If it's a different `meta`, the plan must specify.

[ASSUMPTION] `_build_macro_user_prompt`, `_build_phase_user_prompt`, and `_build_beat_user_prompt` exist and accept an `OutlineRequest` or its fields. The grounding does not show these functions; verify they exist and their signatures before adding style fields.

[ASSUMPTION] The `opening_status_quo` field can be extracted from the outline LLM's output by adding it to the outline schema/prompt. If the outline is generated by a structured output parser, the schema must be updated. If it's free-text, a regex extraction is needed. The plan provides neither.