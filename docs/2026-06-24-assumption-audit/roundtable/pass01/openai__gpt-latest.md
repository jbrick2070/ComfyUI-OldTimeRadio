<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The stated story-grammar vision is not what the grounded code actually does: style is mostly selected after outline generation, body beats remain prompt-only, and the announcer close can still override/steal the climax.

MUST-FIX BEFORE BUILD:

1. [ASSUMPTION WE SHOULD KILL: “catalog style governs the episode body”] `_otr_style_catalog.py::module doc/render_style_grammar/select_style` + `OTR_LedgerScriptWriter.py::F2 Story-Quality LIFT` + `_otr_line_composer.py::_build_user_prompt` — The catalog claims `sound_world` / `story_engine` / `ending_mode` are injected into macro / beat / line prompts, but the shown writer selects `_style_slug` only after `generate_outline()` and only threads `beat_role`, `conflict_object`, `conflict_type`, and climax-only `ending_template` into `LineRequest`. `_otr_style_catalog.render_style_grammar()` is not shown being called anywhere. Concrete fix: select the catalog style before `OutlineRequest` construction, replace/augment `OutlineRequest.style` with `render_style_grammar(style_slug)`, and pass the same rendered grammar into `LineRequest.style_descriptor` for every body beat. Then delete or deprecate the unused “style grammar is injected” claim until true.

2. [ASSUMPTION WE SHOULD KILL: “ending template fixes style collapse”] `OTR_LedgerScriptWriter.py::F2 _ending_template/_climax_beat_id` + `_otr_line_composer.py::_build_user_prompt DRAMATIC FRAME` — Style grammar reaches only the climax-class beat via `ending_template`; body beats get no `sound_world` or `story_engine`. This directly matches the proven failure: style=`memory_erasure_clinic_session` can still become NASA procedural because body prompts only see normal news grounding and generic dramatic frame. Concrete fix: add per-beat style fields to `LineRequest` such as `style_sound_world`, `style_story_engine`, `style_forbidden_defaults`, and render them inside `THIS BEAT` for all character beats, not just the climax.

3. [ASSUMPTION WE SHOULD KILL: “last character climax means the announcer cannot steal the ending”] `_otr_outline.py::_assemble_outline announcer close intent` + `OTR_LedgerScriptWriter.py::I.5 compose_announcer_outro call` + `_otr_line_composer.py::compose_announcer_outro` — `_assemble_outline` tries to make the announcer close “Do NOT state, summarize, or resolve the outcome,” but the writer later ignores that beat intent and always runs the dedicated outro path. `compose_announcer_outro()` can explicitly instruct the close to “state the outcome plainly” when `ending_change` is resolved. These policies conflict. Concrete fix: define one close policy. If style grammar is on, pass `ending_tag` / `style_slug` / “do not resolve outcome” into `compose_announcer_outro()` and disable the F3 “state outcome plainly” branch for unresolved/revelation/reversal/quiet styles.

4. [ASSUMPTION WE SHOULD KILL: “climax taxonomy is load-bearing for all climax types”] `_otr_story_quality_l12.py::build_sq_data` — `assign_beat_roles()` can assign climax roles like `reversal`, `revelation`, `confession`, etc., but `build_sq_data()` only enriches fallback content when `beat_role in (PERSONAL_STAKE, IRREVERSIBLE_CHOICE)`. Non-irreversible climax classes do not get the deterministic on-stage intent enrichment. Concrete fix: change that condition to `beat_role == PERSONAL_STAKE or beat_role in CLIMAX_CLASS_ROLES`, with climax-specific enrichment text keyed by the actual climax class.

5. [ASSUMPTION WE SHOULD KILL: “crisis-noun grounding prevents console-standoff output”] `_otr_story_quality_l12.py::ground_crisis_nouns/build_sq_data` + `_otr_line_composer.py::_build_user_prompt` — Crisis noun grounding mutates only `beat.intent`; it does not validate or rewrite the generated line. The composer can still output “lever,” “console,” “vacuum,” “lockout,” etc. because the final text has no post-generation crisis-noun gate. Concrete fix: after `compose_line_draft()` and before ledger persistence, count ungrounded crisis nouns in the generated line using the same `GENERIC_CRISIS_NOUNS` + grounded palette; reroll once with a hard hint, then deterministically substitute or flag if still present.

6. [ASSUMPTION WE SHOULD KILL: “select_style picks the best-fit style”] `_otr_style_catalog.py::select_style/premise_wants_emergency` — `select_style()` uses only `premise_wants_emergency()` plus `sha256(cast_seed)` over the eligible pool. It is not a best-fit selector; most article content is ignored except disaster keywords. Concrete fix: either rename it to `seeded_style_draw()` and stop claiming fit, or implement actual content-to-style scoring using tags / keyword maps / news brief fields.

7. [ASSUMPTION WE SHOULD KILL: “style diversity survives writer resolution”] `OTR_LedgerScriptWriter.py::_resolve_inputs style path` + `OTR_LedgerScriptWriter.py::D.2 style picker` + `OTR_LedgerScriptWriter.py::F2 style-grammar select` — There are two separate style systems: the user/LLM resolved style used as `style_descriptor`, and the catalog `_style_slug` selected later for grammar. They can disagree, and the later catalog style does not replace the earlier `style_descriptor`. Concrete fix: merge them into one `EpisodeStyle` object with `source`, `slug`, `label`, `sound_world`, `story_engine`, `ending_tag`, and thread that single object through outline, composer, critic, video metadata, and ledger telemetry.

8. [ASSUMPTION WE SHOULD KILL: “body beat role text is enough for weak local models”] `_otr_line_composer.py::_build_user_prompt DRAMATIC FRAME` — Body role instructions are soft prose. There is no output contract requiring the line to mention/embody `conflict_object`, avoid generic crisis terms, or fulfill `beat_role`. Concrete fix: add a deterministic body-beat validator: for `pressure`, require either conflict object term or domain synonym; for `personal_stake`, require personal-cost signal; for climax, require no future-tense deferral; reroll once before freeze.

9. [ASSUMPTION WE SHOULD KILL: “news grounding always helps”] `_otr_line_composer.py::_build_user_prompt WRITE LINE tail` — The universal tail says “Ground this line in the news facts and this scene’s premise,” while style grammar is only a label or climax template. This biases the weak writer back toward the literal article premise, explaining why a memory-erasure style can become a NASA mission story. Concrete fix: change the grounding rule to “Use one concrete news anchor, but dramatize it through the selected style engine,” and include the style engine immediately above this line.

10. [ASSUMPTION WE SHOULD KILL: “domain palette gives real premise specificity”] `_otr_story_quality_l12.py::DOMAIN_PALETTE/_DOMAIN_KEYWORDS/select_domain` — Domain selection is first-keyword-wins, with broad collisions such as `"trial"` appearing in both medicine and law keyword lists; the first match wins. Also `_PERSONAL_COST` has only `"general"`, so domain-specific stakes do not exist. Concrete fix: split domain detection into scored keyword matching and add domain-specific `_PERSONAL_COST` entries before relying on it as a “personal stake” lever.

11. [ASSUMPTION WE SHOULD KILL: “the announcer open/close are harmless bookends”] `_otr_outline.py::_assemble_outline announcer open/close` + `OTR_LedgerScriptWriter.py::compose_announcer_intro/compose_announcer_outro path` — Announcer beats are hardcoded into the outline, get fixed target words/moods, and the final close is later regenerated from separate prompts. They are not passive; they can reshape the episode’s arc after the body is written. Concrete fix: treat announcer intro/outro as explicit story-contract consumers with the same `EpisodeStyle`, `ending_tag`, and “do not steal climax” rules as the body.

12. [ASSUMPTION WE SHOULD KILL: “video style follows story style”] `_otr_style_catalog.py::sound_world` + `eng_ltx_av.py::_LTX_DEFAULT_NEGATIVE/_sharp_enabled/_LTX_AV_* constants` — The style catalog has `sound_world`, but the LTX-AV engine has forced negative prompt, default SHARP mode, fixed steps/cfg, and no shown connection to `sound_world`. Visual/audio render diversity is therefore likely downstream-collapsed. [ASSUMPTION: no unseen video prompt bridge uses `render_style_grammar`; verify.] Concrete fix: thread `sound_world` into visual/audio prompt construction and record it in the render request; make SHARP/cfg/negative recipe an engine profile, not a universal default.

SHOULD-FIX:

1. [FALSE DISTINCTION / MERGE] `OTR_LedgerScriptWriter.py::_STYLE_CHOICES/_STYLE_PICKER_SEED_POOL` + `_otr_style_catalog.py::STYLE_CATALOG` — The old style picker and the new catalog are two competing style ontologies. Collapse to one catalog-backed style resolver. Keep user free-text as an override that maps to `custom_style` with explicit missing fields, not as a parallel system.

2. [DEFENDED INVARIANT AUDIT] `_otr_story_quality_l12.py::module comments “DETERMINISTIC, UPSTREAM”` — The deterministic upstream layer is defended as the “only effective fix,” but it only changes beat intents and prompt fields. It does not constrain generated text. Concrete fix: rename this layer to “planning hints” unless paired with output validators; add validator telemetry showing body-beat compliance.

3. [UPSTREAM VARIES / DOWNSTREAM COLLAPSES] `_otr_style_catalog.py::STYLE_CATALOG` + `_otr_line_composer.py::_build_user_prompt` — 100 styles collapse into one generic composer prompt plus a climax ending line. Concrete fix: require every style to provide a compact `body_rule` and `forbidden_defaults` string consumed by all character beats.

4. [DEAD / CARGO-CULT] `_otr_style_catalog.py::render_style_grammar` — No grounded caller is shown. Concrete fix: wire it into writer/outline/composer or delete it until the integration exists.

5. [CORRECTNESS CLAIM] `_otr_style_catalog.py::select_style docstring` — It says “best-fit style,” but the implementation is hash selection over emergency/non-emergency pool. Concrete fix: update wording or implement fit scoring.

6. [FORCED DEFAULT] `_otr_outline.py::_assemble_outline` — Announcer open intent, target_words=15, mood=`welcoming`; announcer close target_words=15, mood=`reflective`; music_inter target_words=5, mood=`transitional` are fixed. Concrete fix: derive announcer bookend moods from selected style and ending tag.

7. [FORCED DEFAULT] `eng_ltx_av.py::_sharp_enabled` — SHARP defaults on via `OTR_LTX_AV_SHARP` default `"1"`. Concrete fix: make it a render profile selected by workflow/meta, not a silent default, or stamp it as a deliberate style choice.

8. [SINGLE-PRIOR TRAP] `_otr_line_composer.py::_SYSTEM_PROMPT` — The prompt still uses broad radio-drama craft language. Weak local models can satisfy it with their training prior: urgent procedural conflict. Concrete fix: move from general craft prose to per-style hard constraints and post-generation checks.

9. [SINGLE-PRIOR TRAP] `_otr_line_composer.py::_build_user_prompt “Generic roles ... mission control are fine”` — This explicitly permits “mission control,” which is part of the observed collapse surface. Concrete fix: make generic-role allowance style/domain-aware; do not allow mission-control vocabulary outside compatible styles.

10. [ASSUMPTION SURFACE] `OTR_LedgerScriptWriter.py::run F2 style grammar` — Style grammar is wrapped in broad `except Exception` and silently disables itself on failure. Concrete fix: fail loud in development / soak mode, and stamp `style_grammar_error` in ledger meta when it degrades.

OPTIONAL / NICE-TO-HAVE:

- Add a per-episode “style adherence report” comparing selected style fields against final text: `sound_world_terms_used`, `story_engine_terms_used`, `ending_tag_satisfied`, `generic_crisis_nouns`.
- Add a small A/B harness for gemma vs mistral using the same outline and cast to isolate writer-dependent collapse.
- Add a “body beat diversity” metric: distinct conflict objects actually present in shipped dialogue, not just in `_sq_by_beat`.

CUT THESE (scope / over-engineering):

1. `_otr_style_catalog.py::STYLE_CATALOG sound_world/story_engine for all 100 styles` — Safe to cut to a smaller exercised subset until those fields are actually consumed. Right now the breadth creates the illusion of diversity while downstream ignores most of it.

2. `OTR_LedgerScriptWriter.py::best-of-N / refine / pitch_room gates` — [ASSUMPTION: default-off or env-gated per comments.] These are structural bloat relative to the proven failure. They multiply outline attempts but do not fix body-beat prompt collapse. Keep one simple path until the story contract is load-bearing.

3. `eng_ltx_av.py::forced SHARP default and fixed negative recipe` — Safe to cut from the story-diversity effort because it cannot fix narrative collapse and can mask visual diversity. Keep a baseline render profile; add style-driven profiles later.

4. `OTR_LedgerScriptWriter.py::dual remote slot/router complexity` — [ASSUMPTION: not required for local gemma/mistral failure reproduction.] Do not expand model-routing features while the narrative contract is unenforced. Freeze this surface during story-grammar repair.

5. `_otr_story_quality_l12.py::_PERSONAL_COST general-only table` — Cut the claim that it is domain personal-stake support until domain-specific entries exist. General fallback can remain, but it should not be treated as a diversity lever.

3 HIGHEST-LEVERAGE STRUCTURAL CHANGES:

1. Build one `EpisodeStoryContract` before outline generation.
   - Fields: selected style slug, sound_world, story_engine, ending_tag, forbidden crisis defaults, domain, conflict palette, required body-beat obligations.
   - Consumers: outline macro prompt, beat fleshout prompt, line composer, announcer close, critic/reroll, video prompt.
   - This kills the current split between resolved style, catalog style, L12 roles, and announcer policy.

2. Add deterministic body-output validators before ledger freeze.
   - Validate shipped text, not just beat intent.
   - Checks: no ungrounded `GENERIC_CRISIS_NOUNS`; line uses or implies `conflict_object`; beat_role satisfied; climax does not defer; announcer does not restate outcome when style forbids it.
   - Reroll once with a targeted hint, then deterministic substitute/flag. This directly addresses gemma ignoring instructions.

3. Move style grammar upstream and make body beats style-specific.
   - Select catalog style before `generate_outline()`.
   - Use `render_style_grammar()` in macro/phase/beat prompts.
   - Render `sound_world` and `story_engine` in every character `LineRequest`.
   - Stop treating `ending_template` as the primary style lever; it is only the final-beat landing instruction.