# ASSUMPTIONS WE SHOULD KILL -- FINAL (converged R1->R4, grounded)

4-round assumption-attack. Panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro;
Claude code-grounded judge+panelist. Total spend ~$2.29. CONVERGENCE: all three
models + the anchor landed the same two root seams in R1 and never dissented;
R4 verdicts all "yes-with-fixes / converged" (only build-precision left, no new
assumptions). Every claim below is CONFIRMED against the real files.

## THE RANKED KILL LIST

### KILL 1 (highest leverage) -- "the deterministic story layer governs the body."
It does not: it only edits `beat.intent`; the SHIPPED dialogue is ungated, and
the prompt actively LICENSES the failure.
- Evidence: `build_sq_data` mutates `beat.intent` only (`_otr_story_quality_l12`
  ~699-705); `count_ungrounded_crisis` only COUNTS (~444). Composer line 1162
  permits "mission control", line 1442 says "Ground this line in the news facts".
  Proven live: gemma -> "press this red lever and let the atmosphere out",
  "blowing the fuel cells".
- FIX: a deterministic body-output gate, IN-LOOP, after the common `cleaned`
  assignment (writer 4163 exchange / 4206 compose) and before `last_lines.append`
  (4222) -- covers the `use_exchange` bypass. New
  `validate_composed_grounding(text, sq_entry, grounded, max_ungrounded=0,
  require_conflict_object_on_roles=CLIMAX_CLASS_ROLES|{BEAT_ROLE_PRESSURE})`;
  object match by head-noun/token-overlap (reuse `_TOKEN_RE`, casefold, strip
  possessive). Add `grounded_nouns` to `LineRequest`, computed once from
  `premise_noun_palette(roster, news_seed, outline.premise, *premise_texts(meta))`
  + thread through `_otr_reroll.build_reroll_line_request`. One guarded reroll;
  split hints (ungrounded_crisis -> offending tokens only; missing_conflict_object
  -> the grounded `conflict_object`); ship-reroll-if-valid-else-original,
  deterministic; stamp `meta.story_quality.{body_gate_rerolls, body_gate_failed,
  body_gate_ungrounded_crisis}`. De-license composer 1162/1442 by style.

### KILL 2 -- "the catalog style governs the episode."
Only the ending_tag survives; the rich grammar is never injected.
- Evidence: `render_style_grammar` (`_otr_style_catalog` ~678) has ZERO callers;
  the writer threads only `ending_template` (climax) + role/conflict fields. The
  prompt "STYLE:" is `style_descriptor`, a DIFFERENT value than the catalog
  `_style_slug`. Proven live: `memory_erasure_clinic_session` -> NASA story.
- FIX: one frozen `StoryContract(slug,label,sound_world,story_engine,ending_tag,
  ending_template,grammar)` + `build_story_contract` in `_otr_style_catalog`,
  built once AFTER cast-lock (`cast_seed` as seed) + news interpretation, BEFORE
  `OutlineRequest`, from `script_brief or news_seed`. Reuse it in F2 (delete the
  late `select_style(outline.premise,...)`). Add style fields to `OutlineRequest`
  + render in `_build_macro/phase/beat_user_prompt`; add them to `LineRequest` +
  render for EVERY character beat (not just climax). ADD `meta.story_contract`;
  do NOT overwrite `resolved["style"]`/`meta.style`/`visual_plan.style` (they feed
  build_news_briefs + cast) -- defer the collapse.

### KILL 3 -- "the climax is always the last voiced beat." A forced mono-shape.
- Evidence: `assign_beat_roles` forces `i==n-1 -> climax` (~511); `validate_beat_
  roles` makes it law (~558). Every episode rises to a final-beat peak; no test
  covers the alternative (forbidden) -> undertested SPOF.
- FIX (DEFERRED -- cascading): let the ending taxonomy choose climax POSITION.
  Breaks the validator + `ending_template` target + the outro's "last char line =
  resolution" assumption; do it AFTER KILL 1/2 as its own build.

### KILL 4 -- "body beats need only soft role prose." They get nothing deterministic.
- Evidence: `build_sq_data` enrich gate `if beat_role in (PERSONAL_STAKE,
  IRREVERSIBLE_CHOICE)` (~694); `_enrich_intent` treats every other role as
  personal_stake. setup/pressure/consequence + all non-irreversible climax classes
  are starved.
- FIX: role-keyed enrichment map (setup/pressure/personal_stake + every
  CLIMAX_CLASS_ROLES member, class-specific text; CUT consequence -- unreachable
  under climax-last). Fix the 200-char truncation order (reserve the tail, truncate
  the original).

### KILL 5 -- "the announcer close is governed by my C5 gate." It is NOT (self-correction).
- Evidence: `compose_announcer_outro` runs post-loop + OVERWRITES the close
  (writer 4230) from `news_close_brief`, with an F3 branch that injects "State
  this outcome plainly" when `is_resolved_ending_change` (composer 2819/2785).
  The live non-outcome closes came from the pre-existing `flag_thesis_close`, not
  my gate.
- FIX: add `ending_tag` to `compose_announcer_outro`; for {unresolved_final_sound,
  revelation, quiet_acceptance} force `resolved=False` + "do not resolve", AND
  route the fallback to `fallback_announcer_outro` (never `_resolved_outro_
  fallback`). Pass `contract.ending_tag` at writer I.5. Retire the moot C5 gate.

### KILL 6 -- assorted dead / cargo-cult / forced-default (low-cost, do alongside)
- `render_style_grammar` dead (fixed by KILL 2). `ARC_PHASE_GUIDANCE` dead in the
  LINE COMPOSER only (`position` shadow; append it to the POSITION block ~1213) --
  NOT dead in the outline (`_phase_summary` 1233). `_PERSONAL_COST` general-only
  (the `domain` arg is dead) -- add domain rows or drop the claim. `select_style`
  is a sha256 DRAW, not "best-fit" (rename). `DOMAIN_PALETTE` first-keyword-wins
  with "trial" in BOTH medicine+law -> scored matching + tests ("clinical trial"
  vs "court trial"). `eng_ltx_av` forced cfg/steps/SHARP/negative -> a style-driven
  render profile (DEFERRED).

### DELETE-IT / SPOF findings
- `DEFAULT_LLM=mistral-nemo` -- remove and everything changes (the writer choice
  DECIDES story quality; gemma bad). The single most output-determining default
  with NO quality gate behind it -> KILL 1's body gate is the model-agnostic net;
  a model-capability gate is the deferred belt-and-suspenders.
- climax-last invariant + `BEAT_ROLE_IRREVERSIBLE_CHOICE` default = undertested
  SPOFs the whole byte-identity story rests on.

## THE 3 HIGHEST-LEVERAGE STRUCTURAL CHANGES
1. **Body-output gate (KILL 1).** Validate the shipped line, not the intent;
   in-loop, grounded_nouns incl. outline.premise, offending-token reroll. This is
   the change that stops gemma's console standoff -- the proven failure.
2. **StoryContract injected into the whole body (KILL 2).** Select once
   pre-outline; render the full style grammar into macro/phase/beat + every body
   LineRequest; ADD `meta.story_contract`, defer the two-system collapse.
3. **Unify the close + un-starve the body (KILL 5 + KILL 4).** Govern the
   announcer close by `ending_tag` (prompt AND fallback); role-keyed enrichment
   for every dramatic role. (KILL 3 climax-position is the next build after these.)

## VERIFY-AT-BUILD (from R4; the coder's acceptance checklist)
seed=cast_seed after cast-lock; activation only when story-quality flag on (else
byte-identical, no meta.story_contract); grep: no `select_style(` after
generate_outline; pitch_room `dataclasses.replace` preserves new OutlineRequest
fields; gate runs after BOTH `cleaned` sites; force reroll-exception + reroll-
invalid paths; palette test for an object only in outline.premise; per-class
enrichment tests (no personal-stake wording on revelation/reversal/confession/
quiet); 200-char truncation test; non-resolving-outro test; POSITION+guidance
snapshot; domain "clinical trial" vs "court trial".
