<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still has ordering/gating contradictions before `OutlineRequest`, an off-path break around `select_style`, and unresolved handoffs for phase/beat prompts, rerolls, and coda validation.

MUST-FIX BEFORE BUILD:
1. [§1/§7/§9, R3 WIRING FACTS] `_style_grammar_on` is currently computed at writer `:3216`, but §9 requires `build_story_contract` before `OutlineRequest` at `:3032`. That gate is needed before the contract/OutlineRequest branch exists. Concrete fix: compute `_style_grammar_on = _OTRCFG.style_grammar_enabled()` immediately after `_apply_story_scaffold_env(story_scaffold)` at run top (`:2402`) and before any contract/outline construction. Use that same variable for all new contract/announcer gates.

2. [§1/§7, R2 WIRING FACTS] “DELETE the late `select_style(...)` @ `:3224`” conflicts with §7 “OFF => no contract build, no select_style move effect.” If the late call is unconditionally deleted and contract build is disabled off-path, downstream style data has no source. Concrete fix: do not delete the current `:3224` behavior for `_style_grammar_on == False`; wrap it:
   - off: execute existing late `select_style(_premise_str, meta, cast_seed)` path byte-identically.
   - on: use pre-outline `StoryContract` slug/ending fields and skip the late draw.

3. [§0/§1, `_otr_outline.py` prompt handoff] The plan adds `OutlineRequest.story_engine` / `ending_mode` and renders them in `_build_macro_user_prompt`, but still leaves “VERIFY-AT-BUILD” for `_build_phase_user_prompt` / `_build_beat_user_prompt`, which take `macro`, not `OutlineRequest`. That is not build-ready wiring. Concrete fix: choose one explicit path before coding:
   - add explicit `story_engine` / `ending_mode` params to phase/beat prompt builders and update all call sites, or
   - attach the contract fields to the macro/combiner object before phase/beat prompting and render from there.
   Do not rely on `OutlineRequest` fields being visible in phase/beat prompts unless the handoff is implemented.

4. [§2/§9, R2 WIRING FACTS open call `:4465`] `SafeOpenBrief.opening_status_quo` depends on “FIRST character beat’s intent,” but the open is composed inside the line loop at the first announcer beat. If implementation derives it from generated ledger/current loop state, it will not exist yet. Concrete fix: precompute `safe_open_brief` once after `generate_outline(...)` and before entering the line loop by scanning the outline beats for the first character/setup beat intent; do not derive it from ledger lines.

5. [§2, R2/R3 WIRING FACTS] The open spoiler belt requires `forbidden_tokens` from `ending_change + news_close_brief`, but the only grounded `ending_change` retrieval is at outro time (`:4615`). The open call is earlier (`:4465`). Concrete fix: before the line loop, resolve `_open_ending_change` from the same `meta.dramatic_state.ending_change` source used for outro and resolve `_open_news_close_brief` from the brief source; build `forbidden_tokens` there. If either is unavailable, record the telemetry condition and skip only the belt, not the safe-open prompt.

6. [§1, R3 WIRING FACTS reroll rebuild] “LINE level: pass ONLY a compact register tag + existing `conflict_object`” introduces a new line-level field, but the spec does not define how rerolls preserve it. R3 states `build_reroll_line_request` otherwise loses new line-level fields unless stamped in meta and rebuilt there. Concrete fix: either cut the compact register tag from LineRequest, or define the exact field name/value and stamp it into line meta so `build_reroll_line_request` at writer `:3922` rebuilds it. [ASSUMPTION] Exact LineRequest fields are not shown; verify the concrete constructor and reroll payload.

7. [§3] `validate_news_coda_line(text, *, lead_in, ...)` conflicts with “deterministic lead-in is a POST-GENERATION PREFIX” and “body has NO lead-in variant.” If the validator is run after returning `f"{coda_lead_in} {body}"`, it will reject the required prefix. Concrete fix: validate the generated body before prefixing, or make the validator strip exactly one expected prefix before checking for duplicate lead-in variants. Return the prefixed line only after body validation.

8. [§3, compose_announcer_outro signature] The new `climax_character_line` parameter is added, but the existing composer context uses `final_character_line`. If the flagged coda path does not explicitly prefer `climax_character_line`, the decoupling has no effect. Concrete fix: in `compose_announcer_outro`, under `story_scaffold`, use `climax_character_line or final_character_line` for coda context; under flag off, preserve the current `final_character_line` behavior byte-identically.

9. [§3/VERIFY-AT-BUILD] The news coda requires `news_close_brief` to state the real fact, but the spec only says “verify news_close_brief never empty.” That is a runtime dependency, not a wiring plan. Concrete fix: add an explicit guarded branch before composing the flagged coda:
   - if `story_scaffold` and `news_close_brief` is empty, emit `news_coda_fallback`/loud telemetry and use a deterministic safe fallback that does not read `script_brief` or `ending_change`, or fail the run if a real coda is mandatory.
   - do not silently pass an empty close brief into the LLM path.

10. [§6 chunking/R3 REFINE loop] C1 placement must not be inside any pitch-room/non-refine block. R3 says `_refine_loop` re-invokes `run(... _refine_active=True ...)`, pitch_room is skipped under `_refine_active`, and the contract is still needed every pass for the outline. Concrete fix: place contract build after cast/brief availability and before `OutlineRequest`, outside any `if not _refine_active` pitch-room branch.

SHOULD-FIX:
1. [§2] The call site says pass `story_scaffold=True` under flag. To avoid a second gate drifting from the real kill switch, pass `story_scaffold=_style_grammar_on` to both announcer composers. Do not use the raw operator/widget value or a new env var.

2. [§3] `validate_news_coda_line` says “reuse the news key_terms if available,” but the proposed signature has no `key_terms` parameter. Concrete fix: either derive key terms solely from `news_close_brief` inside the validator, or add `key_terms: tuple[str, ...] = ()` to the signature and thread the existing outline/request key terms explicitly.

3. [§2] Importing `_TOKEN_RE` / `_content_tokens` / `_strip_possessive` from `_otr_story_quality_l12` into `_otr_line_composer.py` may introduce an import-cycle risk. verify: current import direction between `_otr_line_composer.py`, writer, and `_otr_story_quality_l12.py`. If cyclic, move token helpers to a small shared utility module.

4. [§8] Telemetry path `meta.story_quality.{...}` needs explicit initialization and JSON-safe mutation. Concrete fix: before increments/flags, ensure `meta["story_quality"]` exists as a dict, and write only primitive values.

5. [§0/§1] `StoryContract.grammar` is computed via `render_style_grammar(slug)`, but the plan only threads `story_engine`/`ending_mode` and stores only slug/label/ending_tag in meta. If `grammar` is not consumed, remove it or keep it local-only; do not let an unused field become a false dependency.

6. [§2] `SafeOpenBrief.cast` is `(name, role)` tuples. The prompt must say these are the only allowed proper names, otherwise the new intro system’s “invent none” constraint has no enforceable source. Concrete fix: render cast as a deterministic list in the safe-open user prompt.

7. [§3] The coda system says “Start immediately with the facts,” while the composer later prefixes a lead-in. That is fine only if the system prompt is clearly addressed to the body generation. Concrete fix: user/system prompt should say “Write only the fact body; do not include the lead-in.”

OPTIONAL / NICE-TO-HAVE:
- [§2] Add one test that asserts `creative_fn` messages for the flagged intro contain no `script_brief` substring.
- [§3] Add one test that validates duplicate lead-in rejection using both exact lead-in and a close variant.
- [§7] Add a snapshot guard for `OutlineRequest` repr/asdict only if existing fixtures actually compare it.

CUT THESE (over-engineering):
1. [§2] Cut the spoiler reroll belt from the first build if import/reroll wiring gets risky. Input starvation via `SafeOpenBrief` is the actual guarantee; the belt is explicitly described as deferrable. Keep deterministic fallback and add the belt in a follow-up.

2. [§3] Cut any seed-keyed lead-in set. The spec already says first build is one fixed lead-in; multiple variants add snapshot/test surface without improving wiring.

3. [§0/§1] Cut `StoryContract.grammar` unless a concrete prompt consumes it. `story_engine` and `ending_mode` are the fields being wired; computing dead grammar repeats the current zero-caller problem.