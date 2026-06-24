# Claude anchor review -- R1 (assumption attack, code-grounded)

VERDICT: the pipeline's diversity is real upstream and **collapses at two seams**
-- the writer (style ignored, prior wins) and the body beats (no machinery gate).
The "story-grammar" only governs the climax SHAPE + the announcer CLOSE; the
middle of every episode is left to the weak model's prior. Several "premise-
anchored" mechanisms are not actually premise-anchored. Below, every claim is
CONFIRMED against the files (line numbers approximate to current HEAD).

## Hunt 7 -- SINGLE-PRIOR TRAP (highest severity; proven live)
- **A7.1 The body beats have no machinery gate; we rely on instruction-following
  the weak model ignores.** `build_sq_data` mutates `beat.intent` ONLY
  (`_otr_story_quality_l12.py` ~699-705 `setattr(b,"intent",...)`); the COMPOSED
  dialogue line is never machinery-checked. `count_ungrounded_crisis` (~444-452)
  only COUNTS generic crisis nouns; nothing STRIPS or REROLLS them. The composer
  emits soft instructions ("Beat function: PRESSURE -- raise the stake ... not
  through a generic alarm or timer", `_otr_line_composer.py` ~1288-1292) that
  gemma overrides (live: "press this red lever and let the atmosphere out").
  CONFIRMED. **Fix: a deterministic body-beat machinery gate** -- run
  `count_ungrounded_crisis` on each COMPOSED character line; over threshold ->
  one reroll with the grounded `conflict_object` forced, mirroring the existing
  stage-direction strip in `_otr_ledger_scrub`. This is the load-bearing fix.
- **A7.2 Style sound_world/story_engine is injected but the prior wins.** The
  prompt carries the style grammar, but the writer anchors to the news premise
  (live: `memory_erasure_clinic_session` -> NASA story). The style's
  `story_engine`/`sound_world` are advisory text the model discards. CONFIRMED.
  Fix: either stop pretending the style drives content (use it only for the
  ending + sound design), or make the premise itself style-derived (pitch room),
  not a thin overlay on the news brief.

## Hunt 4 -- UPSTREAM-VARIES / DOWNSTREAM-COLLAPSES
- **A4.1 Style diversity dies at the writer** (same evidence as A7.2): 90 styles
  selected deterministically (`_otr_style_catalog.select_style` ~718-733) ->
  only the `ending_tag` survives to the output; sound_world/story_engine are
  dropped by the model. The 100-style catalog is mostly decorative.
- **A4.2 premise diversity is gated OFF by default.** The pitch room
  (`_otr_pitch_room`, `OTR_ENABLE_PITCH_ROOM`) is the only premise-divergence
  lever and defaults off, so every episode's premise = the single news brief.
  The "diverge" lever exists but is dark. CONFIRMED.
- **A4.3 crisis grounding dies between intent and line** (A7.1): conflict_object
  is chosen per beat but the composed line re-introduces generic machinery.

## Hunt 1 -- FORCED-DEFAULT
- **A1.1 Climax is ALWAYS the last voiced character beat.** `assign_beat_roles`
  (~511-520) forces `i==n-1 -> climax`; `validate_beat_roles` (~558-562) makes
  "climax-class beat must be LAST" an invariant. Every episode has the identical
  rising-to-final-beat shape; no mid-story climax + denouement is possible.
  CONFIRMED. What's lost: reversal/revelation structures that peak early then
  fall.
- **A1.2 `personal_cost` is NOT premise-anchored -- it is one of 3 generic
  strings for EVERY domain.** `_PERSONAL_COST` has only a `"general"` key
  (~583-589); `fallback_content` (~606) does `_PERSONAL_COST.get(domain) or
  _PERSONAL_COST["general"]` -> the `domain` arg is dead; identical pool every
  episode. CONFIRMED (also Hunt 5).
- **A1.3 Announcer open intent + all beat moods are hardcoded identical.**
  `_otr_outline._assemble_outline`: announcer-open intent "Open the episode and
  orient the listener." (~1591), moods "welcoming"/"reflective"/"transitional"
  fixed, announcer target_words=15 always. CONFIRMED.
- **A1.4 LTX render recipe is one fixed recipe per render.** `eng_ltx_av.py`:
  `_LTX_AV_CFG=3.0` (55), `_LTX_AV_STEPS=8` (54), one shared `_LTX_DEFAULT_
  NEGATIVE` (58-60), SHARP default-on (85). Every clip same recipe (env-
  overridable but never varied per-beat/mood). CONFIRMED.

## Hunt 5 -- DEAD / CARGO-CULT
- **A5.1 `fallback_content`'s `domain` arg is dead for personal_cost** (A1.2):
  only "general" exists. CONFIRMED.
- **A5.2 `conflict_type` is emitted-but-ignored.** Rendered in the DRAMATIC FRAME
  (`_otr_line_composer.py` ~1267-1271) but the composed dialogue does not honor
  it (gemma). Telemetry-real, output-inert. PARTIAL (it may nudge mistral).
- **A5.3 style `sound_world`/`story_engine` fields are near-dead for STORY**
  (A4.1) -- they survive only into the (optional) render prompt, not the script.
- VERIFY-AT-BUILD: is `render_style_grammar` (~678-689, renders sound_world/
  story_engine/ending_mode) still called now that select_style->ending_tag is the
  path? If unused -> dead. (Check callers before R2.)

## Hunt 3 -- DEFENDED-INVARIANT AUDIT
- **A3.1 "climax must be the LAST voiced beat"** -- defended as the dramatic-
  function contract (`validate_beat_roles` docstring ~536-540). SUSPECT: it is
  the structural mono-shape (A1.1). No test can cover a non-last climax because
  it's forbidden -> undertested assumption baked as law.
- **A3.2 "byte-identical when off"** -- defended everywhere as the safety virtue.
  SUSPECT: it is WHY good features ship dark and never get enabled -- the
  operator had to MANUALLY flip the grammar default this session after it sat off
  through 6 chunks. The invariant optimizes for safety over ever-shipping-value.
- **A3.3 V-12 "cold-import" -> DUPLICATED helpers.** `eng_ltx_av.py` duplicates
  `LTX_DISTILLED_SIGMAS` + sampler helpers from frozen `eng_ltx_video` (~88-95,
  comment defends the duplication). SUSPECT: two LTX recipes drift independently.

## Hunt 2 -- FALSE-DISTINCTION / MERGE
- **A2.1 `OTR_ENABLE_STYLE_GRAMMAR` vs `OTR_STORY_QUALITY_L12` are now one lever.**
  The writer runs the L12 build when `l12_enabled() OR style_grammar_on`
  (`OTR_LedgerScriptWriter.py` F2 ~3088+); grammar-on implies the L12 path. Two
  env flags, one behavior. Collapse to a single `story_quality` flag with a sub-
  mode. CONFIRMED.
- **A2.2 `ltx_av_talk` vs `ltx_av_music`** -- "two adapters over one shared core"
  (`eng_ltx_av.py` ~10-18). Genuinely different graphs (I2V+audio vs audio-only),
  so a DEFENSIBLE distinction, not a merge -- list but rank low.

## Hunt 6 -- "DELETE IT" TEST (5 most load-bearing)
- **A6.1 `DEFAULT_LLM` = mistral-nemo** (writer model default). Delete ->
  everything changes (proven: writer choice DECIDES story quality; gemma bad,
  mistral good). The single most output-determining default + it has no quality
  gate behind it. Undertested SPOF.
- **A6.2 climax-last invariant** -- delete the validator -> episodes could vary
  climax position; everything changes; zero tests for the alternative -> SPOF.
- **A6.3 `BEAT_ROLE_IRREVERSIBLE_CHOICE` as the assign_beat_roles default** (485)
  -- delete -> unkeyed calls break + the whole byte-identity story rests on it.
- **A6.4 `STYLE_GRAMMAR_DEFAULT=True`** (just flipped) -- delete -> reverts to the
  console standoff on the climax. Now load-bearing.
- **A6.5 `_LTX_AV_SHARP` default-on** -- delete -> base pass = blurry (comment
  ~72-80). Load-bearing for render quality.

## Ranked KILL list (anchor, pre-panel)
1. Kill "the body is governed by instruction-following" -> add a deterministic
   body-beat machinery gate (A7.1). HIGHEST leverage; explains the gemma failure.
2. Kill "style drives the script" -> make the premise style/pitch-derived OR
   demote style to ending+sound only (A7.2/A4.1/A4.2).
3. Kill "climax is always last" -> let the ending taxonomy choose climax POSITION,
   not just type (A1.1/A3.1).
4. Kill "personal_cost is premise-anchored" -> it is one generic pool; either
   make it real or delete the pretense (A1.2/A5.1).
5. Kill the two-flag story-quality split (A2.1).
