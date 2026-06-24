# ASSUMPTIONS TO KILL -- R1 synthesis (grounded; anchor + 3-model panel)

Every item below is CONFIRMED against the real files unless tagged
[VERIFY-AT-BUILD]. Convergence: all 3 panel models + the anchor independently
landed the same two root seams (style never injected; body never gated).

## TIER 1 -- the two seams that explain the proven failures

### K1. "The deterministic layer governs the body." FALSE -- it only touches
beat.intent; the shipped dialogue is ungated.
- Evidence: `_otr_story_quality_l12.build_sq_data` mutates `beat.intent` only
  (~699-705); `count_ungrounded_crisis` (~444-452) COUNTS, never strips/rerolls.
  The composer's anti-trope lines are soft prose (`_otr_line_composer.py`
  ~1288-1292) and the prompt actively LICENSES the failure surface: line 1162
  `Generic roles ("the tech","the lab","mission control") are fine`, line 1442
  `Ground this line in the news facts and this scene's premise`.
- Proven live: gemma wrote "press this red lever and let the atmosphere out",
  "blowing the fuel cells" with L12 ON.
- FIX (load-bearing): a **deterministic body-output gate before freeze** -- run
  `count_ungrounded_crisis` (+ the `conflict_object` presence check) on each
  COMPOSED character line; over threshold -> one reroll with a HARD banned-word
  hint built from `GENERIC_CRISIS_NOUNS`, then flag. Mirror the existing
  stage-direction strip in `_otr_ledger_scrub`. Also: scope the
  "mission control fine" + "news facts" prompt lines to compatible styles.

### K2. "The catalog style governs the episode." FALSE -- the rich grammar is
never injected; only the ending_tag survives.
- Evidence: `_otr_style_catalog.render_style_grammar` (~678-689) has **ZERO
  callers** (grep: defined once, referenced only in a comment). The writer
  threads only `ending_template` (climax beat) + `beat_role`/`conflict_*` into
  `LineRequest`; `sound_world`/`story_engine` never reach a prompt. The prompt's
  "STYLE:" line (composer 1131-1132) renders `style_descriptor`, a DIFFERENT
  value from the catalog `_style_slug` (two style systems -- see K5).
- Proven live: style=`memory_erasure_clinic_session` -> NASA story.
- FIX: select the catalog style BEFORE `generate_outline`; render
  `render_style_grammar(slug)` into the macro/beat prompts AND every character
  `LineRequest` (a `style_engine`/`sound_world` field), not just the climax.
  Demote `ending_template` to "final-beat landing instruction," not the lever.

## TIER 2 -- forced mono-shape + starved body

### K3. "Body beats need only soft role prose." FALSE -- body beats get no
deterministic dramatic content at all.
- Evidence: `build_sq_data` gates `fallback_content`+`_enrich_intent` behind
  `if beat_role in (PERSONAL_STAKE, IRREVERSIBLE_CHOICE)` (~694); setup/pressure/
  consequence AND every non-irreversible climax class (revelation/reversal/...)
  get nothing. Confirmed by GPT + Gemini independently.
- FIX: extend enrichment to `PERSONAL_STAKE` + all `CLIMAX_CLASS_ROLES` with
  CLASS-KEYED text (not the irreversible-choice string blindly), and give
  setup/pressure/consequence a minimal per-role concretization.

### K4. "The climax is always the last voiced beat." A forced mono-shape.
- Evidence: `assign_beat_roles` forces `i==n-1 -> climax` (~511-520);
  `validate_beat_roles` makes "climax-class beat must be LAST" law (~558-562).
  Every episode rises to a final-beat peak; no early-climax + denouement. No test
  can cover the alternative because it's forbidden -> undertested SPOF.
- FIX: let the ending taxonomy choose climax POSITION (some classes peak earlier,
  e.g. revelation), not only type. Lower priority than K1/K2 but structural.

### K5. "There is one episode style." FALSE -- there are two competing style
ontologies that can disagree.
- Evidence: the resolved widget/LLM `style` -> `style_descriptor` (rendered as
  "STYLE:") vs the catalog `_style_slug` from `select_style` (drives only the
  ending). The catalog pick never replaces `style_descriptor`. [VERIFY-AT-BUILD:
  confirm the two values in one live ledger meta.]
- FIX: one `EpisodeStyle`/`StoryContract` object {source, slug, label,
  sound_world, story_engine, ending_tag, domain, conflict_palette,
  forbidden_defaults} built once, threaded through outline/composer/announcer/
  critic/video/telemetry. (All 3 models' #1 structural change.)

## TIER 3 -- dead / cargo-cult / forced defaults

### K6. `render_style_grammar` is DEAD (K2) -- wire it or delete it.
### K7. `ARC_PHASE_GUIDANCE` is dead-coded -- `position` shadows it.
- Evidence: writer always sets `position=_position_for(beat)` (writer 3997);
  composer `if req.position:` ... `elif req.arc_phase:` (1210-1217) -> the
  guidance branch never runs. [VERIFY: does `_position_for` carry the
  dramatic-function directive, or just "phase, beat N of M"? If the latter, the
  model lost the "what setup MEANS" guidance -- R2 dig.]
### K8. `_PERSONAL_COST` is general-only -> the `domain` arg to `fallback_content`
is dead (~583-606). Either add real domain entries or drop the "premise-anchored
personal stake" claim.
### K9. `select_style` is a sha256 DRAW, not the "best-fit" the docstring claims
(~718-733; only `premise_wants_emergency` reads content). Rename or implement fit.
### K10. `DOMAIN_PALETTE` is first-keyword-wins with collisions ("trial" in BOTH
medicine and law `_DOMAIN_KEYWORDS`) -> mis-domained episodes. Scored matching.
### K11. Forced render recipe: `eng_ltx_av` cfg=3.0/steps=8/SHARP-on/fixed
negative (54-60,85), no `sound_world` hook -> visual diversity collapses too.
Make it a style-driven render profile.

## SELF-CORRECTION (the panel caught this)
- My C5 announcer-close INTENT gate in `_assemble_outline` is **largely moot**:
  `compose_announcer_outro` runs post-loop and OVERWRITES the close (writer 4230,
  4405-4409) from `news_close_brief`, ignoring the outline beat intent. The
  non-outcome closes observed live came from the pre-existing `flag_thesis_close`
  image-enforcement, NOT my gate. FIX: move the "do not resolve outcome" policy
  INTO `compose_announcer_outro` (pass `ending_tag`/`style_slug`), and stop
  relying on the outline-intent gate. [Reconciles with the live evidence that
  closes WERE non-outcome -- that was flag_thesis_close doing the work.]

## 3 HIGHEST-LEVERAGE STRUCTURAL CHANGES (R1)
1. **Deterministic body-output gate before freeze** (K1) -- validate the shipped
   LINE, not the intent: ungrounded-crisis-noun reroll + conflict-object
   presence. This is what would have stopped gemma's fuel-cell standoff.
2. **One `StoryContract` built pre-outline + style injected into every beat**
   (K2/K5) -- render the full style grammar into the body, not just the climax;
   collapse the two style systems.
3. **Model-capability gate** (panel DeepSeek/GPT) -- the prompt strategy assumes
   instruction-following gemma lacks; either prefer mistral for the creative
   slot by default or branch to a harder, simpler anti-standoff directive for
   known-weak models. (Pairs with K1; the gate is the safety net.)
