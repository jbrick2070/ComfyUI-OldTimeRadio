# OTR Story-Quality Synthesis (2026-06-27 overnight batch)

Role: grounded synthesis judge. Scope: DOCS-ONLY planning. No code edits, no
workflow-JSON edits, no GPU / :8000, no git push. This doc is the plan only.

Hard constraints honored by every fix below:
- Reuse the existing reroll loop (`_otr_line_composer.compose_line` in-line gate at
  ~L2364, and `_otr_reroll.run_targeted_reroll` at L497). No new reroll engine.
- NO ledger-schema change (new signals ride the freeform `meta` dict and the
  freeform `compose_flags` list, which already carry arbitrary strings).
- NO workflow-JSON change. All touches are pure-python nodes + the scan script.

## Corpus map (15 salvaged episodes)

Writer is `meta.gen_params_initial.creative_writing_model`. The frontier writer is
`openrouter:slot-a` (grok-4.3); it wrote exactly 4 -- one per creativity tier.

| Creativity (temp) | Frontier slot-a (grok-4.3) | Local (mistral-nemo / gemma-4-E2B) |
|---|---|---|
| safe & tight (0.6) | spindle_turns_again | marked_for_erasure (mistral), brass_hinge (gemma) |
| balanced (0.85) | links_ascent_lesson | dialing_shadows, shadows_of_the_past, heatwave_decryption, frostbite_facility (mistral); brass_button (gemma) |
| wild & rough (0.92) | bar_chip_ultimatum | shredded_hope (mistral), seal_of_the_compound (gemma) |
| maximum chaos (0.95) | marks_keep_climbing | power_play (mistral), compass_points_true (gemma) |

Scripts read in full for this judgment: dialing_shadows, marked_for_erasure,
compass_points_true, shadows_of_the_past, heatwave_decryption (local) and
bar_chip_ultimatum, links_ascent_lesson, marks_keep_climbing, spindle_turns_again
(frontier); dramatic_state read for power_play + frostbite_facility. >6 spanning
both ends, all four creativity tiers.

---

## 1. Anchor critique (my own, grounded; local vs frontier)

### Frontier (grok-4.3) -- ~6.5/10
The drama IS the science. Grounds in real, specific nouns and finds concrete
physical craft the local end never reaches:
- bar_chip b002: `Name's on the chip, Steiner. Let's see it before dawn.` -- speakable, a real object, a clock.
- bar_chip b003: `Your prints are all over it, and that species list will remember exactly whose they are.` -- threat via subtext, grounded in the species-listing seed.
- links b007: `My thumb keeps catching on the log's corner, but the door's seal just dropped.` -- a real body in a real room.
- spindle b002: `That record's already on the spindle. One more turn of the dial and the whole line hears the wreck.` -- speakable, concrete metaphor for broadcasting a death.

Frontier failure = anchor-stuffing under the body-gate. The middle act degrades
into proper-noun run-ons no one would speak:
- links b004: `Boys and girls the last of the Neil Gehrels Swift Observatory observing time vanishes right now as the Swift Boost Mission countdown to fiery re-entry starts on LINK.`
- marks b004: `The cooperative's vote demands the mitochondrial farm footage air now so UCLA psychologists can trace every mark of childhood adversity and mitochondrial function back to the segments you committed.`
- spindle b008: `Refusing your deadline means the disputed record of the rail death now includes this partial echo of the transgender performer's suicide thoughts and the discrimination behind harassment and suicide attempts.`

And the L12 cost-boilerplate leaks into spoken text -- links b009 ends
`...no matter who trusts either of us` (verbatim `_PERSONAL_COST` tail; see 3.6).

### Local (mistral / gemma) -- ~3.5/10, three distinct failure shapes
- Mistral @ balanced abandons the science for generic melodrama. dialing_shadows
  seed is tree-planting targets; the episode is a hospital missing-brother soap:
  b001 `Midnight in the hushed halls of Mercy General... eyes ablaze, questions the unyielding Stone Malone` (invented hospital, contradicts the seed); b003 `You're playing with fire, Watson...`. The seed only appears in the coda;
  `meta.post_assembly_key_terms.passed = false`, `repair_pass = deferred`.
- Mistral @ safe & tight stays on-topic but is flat, cliche, vocative-stuffed.
  marked_for_erasure names the addressee almost every line (`...Leviathan`,
  `...Travis`, `Now, Beatty!`) and runs `Over my dead body, Leviathan.`
- gemma @ max chaos stays on-topic but is incoherent at sentence level.
  compass b008 `I feel the tension in the brass, Alice` (no brass in a telescope
  story); b012 `Alice, we're playing with fire here... the funding goes up in smoke`;
  b011 `Lemmy, Euclid's eyes are on us. We can't afford to blink.` is verbatim the
  b005 `reviewer_note` example -- the critic's suggestion was copied into the script.

Across all local episodes the two characters are interchangeable (threat ->
counter-threat) despite distinct `speech_signature`s on the cast rows.

A spoken line can be PURE stage action on the local end -- heatwave b008
(speaker_role=character): `snaps off pen's tip, jams it into the decryption
machine's port, turning it into scrap metal` -- shipped with EMPTY compose_flags
(the stage-direction detector missed it); heatwave b011 fused a bare direction with
dialogue: `steps forward, revealing a keycard tucked in her pocket Perhaps you
should've considered...`.

---

## 2. Panel grounding ledger (each claim checked vs real code / real ledgers)

KEEP = grounded survivor. CORRECT = real defect, wrong attribution/quote.
DISCARD = misread or hallucination.

### Panel A (trope-soup panel)
- A1 "Empty melodrama / 'Not X, but Y'" -- PARTIAL KEEP. The pattern is real
  (compass b002 `It's not about the numbers anymore`; the dialing/marked cliche
  register). The specific gemma quotes (`breaking your faith`, `rusted through,
  DJINN VANCE`) could not be verified verbatim -- treat as illustrative, not cited.
  Fix kept at LOW priority (a "Not X but Y" ban risks false positives -- it is also
  a legit rhetorical device).
- A1b "Lean harder on leak-floor-v2 to demand concrete physical verbs" -- DISCARD.
  Misread. `leak-floor-v2` (`_otr_line_hygiene.verify_and_repair_line`) is four
  NARROW structural leak rules (participle-before-quote, roster vocative, malformed
  quote, banned entity). It is explicitly not a verb-whitelist / verb-demander.
- A2 "Character drift: Doug threatens Krit, then 'Why are you protecting me, Doug?'"
  -- DISCARD (specific example). heatwave_decryption's cast is AYESHA REEVES + KRIT
  HALLOWAY; there is no "Doug" and no such line. The general "voices are
  interchangeable" point survives under item 3.7, but the pronoun/identity-flip
  evidence is fabricated.
- A3 "Literalism disaster: character_a_wants = take sole credit for transgender
  people" -- KEEP + CORRECT. The quote is REAL and VERBATIM, but in
  frostbite_facility (local mistral, `dramatic_state_source: "fallback"`), not the
  episode Panel A labeled. Full record: `central_object: "transgender people"`,
  `dramatic_question: "Who will control what becomes of transgender people?"`,
  `character_a_wants: "take sole credit for transgender people"`,
  `character_b_wants: "make sure transgender people is shared openly and freely"`,
  `ending_change: "Control of transgender people passes to whoever is willing to
  pay the higher price."` This is the single most serious defect in the batch
  (quality AND dignity/safety). Root cause is now grounded -- see fix 3.1.
- A4 "StoryCritic rubber-stamps garbage (`validated: true`)" -- DISCARD the
  interpretation. `story_critic_status.validated` means the critic's JSON parsed,
  NOT that the story passed. The real, grounded finding (the critic is advisory,
  not blocking) survives via `_otr_render_plan._BLOCKING_ARC_VERDICTS` (only
  `mid_collapse`/`flat` block; `uneven` ships) and the `a2_ship_through` path. Kept
  under item 5.
- A-fix3 "Anti-clustering anchor-density check" -- KEEP (consensus with Panel B and
  my own body-gate finding). -> item 3.2.
- A-fix5 "Force mechanical syntax per character (A only questions, B only
  imperatives)" -- DOWNGRADE. Too rigid; would read robotic. The real gap is that
  `speech_signature` never differentiates the text -- reframed as item 3.7.
- A5 "Specificity anchors backfire by clumping" -- KEEP. -> item 3.2.
- A6 "speech_signatures not firing for local" -- KEEP. -> item 3.7.

### Panel B (educational-lens panel)
- B-lens "Coda is the teaching payoff; judge bridge/teach execution, do not kill the
  coda" -- KEEP. Correct framing; the coda items below fix execution, not presence.
- B1 "Anchor-stuffing gate (3+ anchors/line)" -- KEEP, GROUNDED (spindle b008-b010,
  links b004, marks b004). -> item 3.2.
- B2 "One-breath dialogue gate (~28+ words / over-clausal)" -- KEEP, GROUNDED
  (spindle b008 30w, marks b004 29w, links b004 28w). -> item 3.2.
- B3 "Broader stage-action leak gate" -- KEEP, GROUNDED. heatwave b008 (pure
  narration, no flag) and b011 (fused direction) prove `detect_stage_business_for_reroll`
  misses fully-narrated and post-quote-fused lines. -> item 3.3.
- B4 "Expanded cliche floor" -- KEEP, GROUNDED. `_CLICHE_RES` has only 6 phrases;
  the batch shipped `hangs in the balance` (dialing b002), `over my dead body`
  (marked), `playing with fire` (dialing/compass), `goes up in smoke` (compass).
  -> item 3.4.
- B5 "Coda execution counters (truncated / mojibake / generic_bridge / fallback)"
  -- KEEP, GROUNDED. `detect_mojibake` exists (verify-only) in `_otr_line_hygiene`;
  `validate_news_coda_bridge` (L3127) + `compose_news_coda` (L3149) exist;
  `news_coda_fallback` flag already emitted (shadows b005). -> item 3.5.
- B "central object measured-not-used; `inject_central_object_into_brief` returns
  the brief unchanged" -- KEEP, GROUNDED (verified in `_otr_specificity.py` L156-174).
  Important nuance: central_object is dead in the CODA but ALIVE and HARMFUL in the
  dramatic_state fallback (it is the `{t}` that became "transgender people"). -> 3.1.

---

## 3. Synthesized plan (ranked by impact-to-effort)

| # | Change | Impact | Effort | Touches |
|---|---|---|---|---|
| 3.1 | Guard the dramatic_state `{t}` ownership templates + central_object noun-class | Very high (quality + dignity/safety) | Low | `_otr_dramatic_state_llm.py`, `_otr_specificity.py` |
| 3.2 | Anchor-stuffing + one-breath reroll gate | High (every episode) | Med | `_otr_line_hygiene.py`, `_otr_line_composer.py` |
| 3.3 | Broaden stage-action leak detector | High (audio-correctness) | Low-Med | `_otr_line_hygiene.py` |
| 3.4 | Expand cliche floor + re-verify after the quality reroll | Med-High | Low | `_otr_line_hygiene.py`, `_otr_line_composer.py` |
| 3.5 | Coda execution counters (no coda removal) | Med | Low | `scripts/story_quality_scan.py`, `_otr_line_composer.py` |
| 3.6 | Stop appending L12 personal-cost boilerplate to every intent | Med | Low-Med | `_otr_story_quality_l12.py` |
| 3.7 | Make speech_signature actually differentiate the two voices | Med | Med | `_otr_casting.py`, `_otr_line_composer.py` |

### 3.1 Guard the dramatic_state ownership templates + the central_object (BUILD FIRST)
Defect (verbatim, frostbite_facility): `character_a_wants: "take sole credit for
transgender people"` / `character_b_wants: "make sure transgender people is shared
openly and freely"` / `ending_change: "Control of transgender people passes to
whoever is willing to pay the higher price."` This is broken English AND treats a
real human group as an ownable object -- it is unshippable on dignity grounds, and
it poisons every downstream line before the writer starts.
Root cause (grounded): `nodes/_otr_dramatic_state_llm.py` ~L196-199 holds a
template tuple `("take sole credit for {t}", "make sure {t} is shared openly and
freely", "Who will control what becomes of {t}?", "Control of {t} passes to whoever
is willing to pay the higher price.")`. `{t}` is filled with the central_object;
for this episode `dramatic_state_source == "fallback"` and
`central_object == "transgender people"` (from `_otr_specificity.derive_central_object`).
Fix (model-agnostic; a strong line never reaches this because its LLM dramatic_state
succeeds): (a) in `derive_central_object` (`_otr_specificity.py`), reject a topic
that is a person / group / identity noun (a small closed blocklist + a
"refers-to-people" guard) so `{t}` can never be a human collective; fall back to a
concrete object/system from the same key_terms. (b) In `_otr_dramatic_state_llm.py`,
gate the ownership/credit/"shared freely" templates so they only apply to a thing,
never a people-class `{t}`; otherwise pick a non-possessive template. (c) When
`dramatic_state_source == "fallback"`, route one reroll through the EXISTING reroll
path rather than shipping the templated wants verbatim.
Measure: new scan counters in `story_quality_scan.py` -- `dramatic_state_fallback`
(count of `dramatic_state_source == "fallback"`) and `ownable_people_object`
(central_object or `{t}` matches the people/identity blocklist). Target both -> 0.

### 3.2 Anchor-stuffing + one-breath reroll gate (highest-leverage QUALITY lever)
Defect: the body-gate satisfies grounding by cramming anchors/key-terms into one
breath. spindle b008 (30w): `...the disputed record of the rail death now includes
this partial echo of the transgender performer's suicide thoughts and the
discrimination behind harassment and suicide attempts.` marks b004, links b004 same
shape. This is the dominant frontier failure and the local "trope-soup" mirror.
Fix: add `flag_anchor_stuffing(text, anchors, key_terms)` and
`flag_one_breath(text)` to `_otr_line_hygiene.py` (pure, deterministic). Flag a
CHARACTER line when it contains >=3 distinct anchors/key-terms, OR exceeds ~28 words
/ ~3 independent clauses. Wire them into the EXISTING in-line gate in
`_otr_line_composer.compose_line` next to `flag_cliche`/`flag_on_the_nose`
(~L2364-2392), reusing the same single-reroll-with-hint pattern. Hint: "use ONE of
these details as a physical prop or action; do not list the others." Lifts the weak
end; the strong concrete lines (bar_chip b002, spindle b002) already pass both
flags. Exempt the announcer coda line (it is allowed to teach -- see open Q3).
Measure: new counters `anchor_stuffing_lines`, `one_breath_violation_lines` in
`r2_lever_metrics` (`story_quality_scan.py`). Target: local and frontier both near 0;
compare pre/post.

### 3.3 Broaden the stage-action leak detector
Defect: a fully-narrated line ships as dialogue. heatwave b008
`snaps off pen's tip, jams it into the decryption machine's port, turning it into
scrap metal` (EMPTY compose_flags -- not caught); b011 fused a leading direction
with the line. `detect_stage_business_for_reroll` (`_otr_line_hygiene.py` L430) +
`is_third_person_action_clause` (L384) only catch leading/after-quote/balanced cases
and cap the clause at 12 words, so a whole-line action narration and a
direction+dialogue fusion slip through.
Fix: extend `is_third_person_action_clause` / add a "line is >=N% third-person
narration verbs and carries no first/second-person dialogue" detector; route a hit
through the EXISTING reroll with the existing `_BARE_STAGE_HINT` (write only spoken
words). Keep it a reroll, not an aggressive scrub (Panel B caveat) so we never emit
an empty line.
Measure: new counter `stage_action_leak_lines` (character lines that are wholly or
mostly narration) in `story_quality_scan.py`, alongside the existing
`narration_self_address_lines` / `stage_business_lines` / `leading_stage_dir_lines`.

### 3.4 Expand the cliche floor + re-verify after the quality reroll
Defect 1: `_CLICHE_RES` (`_otr_line_hygiene.py` L628) holds only 6 phrases; the
batch shipped `hangs in the balance`, `over my dead body`, `goes up in smoke`, and
`we're/I'm playing with fire` (the regex only matches `you're`).
Defect 2 (mechanism): the gate at `_otr_line_composer` L2364 rerolls ONCE and ships
the result WITHOUT re-checking it (the recursive call passes
`_stage_dir_repair_attempted=True`, which skips the whole flag block). That is why
`You're playing with fire, Watson` shipped in dialing despite an exact regex match --
the weak model reproduced it and nothing looked again.
Fix: (a) widen `_CLICHE_RES` with the batch phrases above + pronoun variants
(`(you|we|i|they)['` + curly]?re playing with fire`, `not on my watch`,
`best left buried`, `running out of time`, `running out of options`,
`before it's too late`). (b) After the single reroll returns, re-run
`flag_cliche`/`flag_on_the_nose`/`flag_stage_business` on the result and keep
whichever of the two drafts has FEWER hits (deterministic, no extra LLM call). A
strong line passes the floor on the first pass.
Measure: existing `cliche_lines` / `on_the_nose_lines` in `r2_lever_metrics`; add
`cliche_shipped_after_reroll`. Target: floor reductions, `cliche_shipped_after_reroll` -> 0.

### 3.5 Coda execution counters (keep the coda; fix its execution)
Defect: codas truncate / show mojibake / fall back to a generic bridge. Panel B
cited `...violence, harassment, and hostile.` (truncated) and cp1252 mojibake in
proper nouns / `41.3` degrees; shadows b005 shipped `news_coda_fallback` +
`news_coda_bridge_invalid` (bare headline). The teaching content is correct -- the
execution is not.
Fix: this is primarily MEASUREMENT first (no behavior change to ship blind). Add
counters to `story_quality_scan.py`: `news_coda_truncated` (coda ends without
terminal punctuation / on a connective), `news_coda_mojibake` (reuse
`_otr_line_hygiene.detect_mojibake`), `news_coda_generic_bridge` (bridge in a known
generic set like "The real story:" / "The true account:"), `news_coda_fallback`
(already a compose_flag). Once quantified, harden `compose_news_coda` /
`validate_news_coda_bridge` (`_otr_line_composer.py` L3127/L3149) to reroll on
truncation and to vary the bridge. Encoding (mojibake) is a separate verify-only
build artifact -- report, do not silently mutate (matches current
`detect_mojibake` policy).
Measure: the four counters above.

### 3.6 Stop appending the L12 personal-cost boilerplate to every beat intent
Defect (mine; neither panel caught it): `_otr_story_quality_l12._PERSONAL_COST`
("the trust they will lose either way" / "what it costs them to be the one who
decides" / "the part of themselves they have to set down") + `_ENRICH_TAILS` are
glued onto nearly EVERY beat intent in `build_sq_data`, homogenizing every beat and
leaking into a spoken line (links b009 `...no matter who trusts either of us`). The
generic `{obj}` splice also corrupts intent grammar (marks b003 intent: `Jenna the
cooperative's vote the accusation by ordering...`).
Fix: route `personal_cost` as a WITHHELD subtext field the composer is told not to
say (the composer already consumes `beat_subtext`), not appended into intent text;
add the three exact cost phrases to the final-line banned-phrase check so they can
never reach dialogue; and validate the `{obj}` splice leaves the intent's verb
intact (skip the splice if it would produce a verb-less clause).
Measure: new counter `cost_tail_in_intent` (beats whose intent contains a
`_PERSONAL_COST` phrase) -- ~100% today -> target 0 in intent and 0 in dialogue.

### 3.7 Make speech_signature actually differentiate the two voices
Defect: cast rows carry distinct `speech_signature`s (dialing: "measured, precise,
weary" vs "measured, concise"; shadows: "wry and indirect" vs "skeptical, precise")
but both speakers read identically. The lever is computed, not landing.
Fix (lighter than Panel A's rigid syntax rule): thread each speaker's
`speech_signature` into the per-line composer prompt as a short, persistent style
directive (not a one-off), and add a deterministic register-divergence check between
the two principals' lines; on low divergence, reroll the offending line through the
existing loop with the speaker's signature as the hint.
Measure: existing `voice_distinct_ratio` (signature distinctness) plus a new
`register_overlap` counter (lexical/structural similarity between the two speakers'
line sets). Target: register_overlap drops on the local end toward the frontier
level.

---

## 4. Single highest-leverage change to build FIRST

Build 3.1 first. It is the best impact-to-effort in the batch: a small, surgical,
deterministic guard at `_otr_dramatic_state_llm.py` + `_otr_specificity.py` that
prevents the worst-case defect (a broken, undignified dramatic_state -- "control
transgender people" -- that corrupts an entire episode and is unshippable on safety
grounds). It is cheap and removes harm immediately.

3.2 (anchor-stuffing + one-breath) is the highest-leverage BROAD-QUALITY lever and
should land immediately after 3.1: it is the consensus of both panels and my own
critique, fires on every episode at both ends, and is measured by two new scan
counters. Sequence: 3.1 (safety, cheap) -> 3.2 (broad quality) -> 3.3/3.4 (leak +
cliche correctness) -> 3.5/3.6/3.7.

---

## 5. Shipped levers NOT firing or backfiring (grounded)

- dramatic_state fallback templates (`_otr_dramatic_state_llm.py`): BACKFIRING --
  template + people-noun `{t}` -> "take sole credit for transgender people"
  (frostbite). See 3.1.
- L12 enrichment (`_otr_story_quality_l12.py`): BACKFIRING -- cost-tails on every
  intent homogenize beats and leak to dialogue; `{obj}` splice corrupts intent
  grammar. See 3.6.
- Specificity anchors: FIRING then BACKFIRING -- they ground the prompt, then the
  model clumps them into one breath (spindle b008, links b004). See 3.2.
- central_object: HALF-DEAD -- `inject_central_object_into_brief`
  (`_otr_specificity.py` L156) returns the brief unchanged, so it never shapes the
  coda; yet it is alive and harmful in the dramatic_state fallback. See 3.1 / open Q6.
- cliche / on-the-nose floor: TOO NARROW and NOT RE-VERIFIED -- exact matches still
  ship (dialing `You're playing with fire`). See 3.4.
- Stage-direction cleanup: PARTIAL -- catches some, misses whole-line narration and
  fused directions (heatwave b008/b011). See 3.3.
- StoryCritic / ArcVerdict (`_otr_story_critic.py`): ADVISORY, NOT DECISIVE -- many
  episodes ship `arc_verdict == "uneven"` because `_otr_render_plan._BLOCKING_ARC_VERDICTS`
  (L186) blocks only `mid_collapse`/`flat`; the `validated: true` flag is JSON-parse
  success, not a quality pass (Panel A misread this). See open Q7.
- contrasting speech_signatures: NOT LANDING for local writers (interchangeable
  voices). See 3.7.
- Minor format leak: a stray leading `"` shipped on the frontier slot-a path
  (bar_chip b005, spindle b003); `sanitize_transcript_text._balance_wrapper_quotes`
  / leak-floor rule 3 should balance a single edge quote -- worth a one-line check
  but low priority.

---

## 6. Open questions for the operator

1. Dignity guard (3.1): confirm a hard blocklist of person / group / identity nouns
   that can never become a `central_object` or fill the ownership/credit/"shared
   freely" dramatic_state templates, plus a forced reroll when
   `dramatic_state_source == "fallback"`. Is the bar "never ship," i.e. fail the
   episode if the guard cannot produce a clean dramatic_state?
2. One-breath threshold (3.2): 28 words / 3 clauses -- right number? Hard reroll, or
   soft warn (the educational coda intentionally packs)?
3. Anchor cap (3.2): cap character lines at 2 anchors, but EXEMPT the announcer coda
   line so it can still teach the news fact? (Panel B's lens.)
4. Cliche list governance (3.4): keep hand-extending `_CLICHE_RES` per batch, or
   move the phrase list to a data file you edit without a code change?
5. Goal (3.2): lift LOCAL toward frontier parity only, or also trim FRONTIER
   overstuffing (the anchor/one-breath gate would do both)?
6. central_object (3.1 / 3.5): revive it (weave into the coda as a final image) or
   remove it entirely (today it only causes harm in the fallback and is dead in the
   coda)?
7. StoryCritic (item 5): promote `uneven` into `_BLOCKING_ARC_VERDICTS` given how
   many episodes ship uneven, or keep it advisory and rely on the per-line gates?

---

Sources (read this session): ledgers for signal_lost_{dialing_shadows,
marked_for_erasure, compass_points_true, shadows_of_the_past, heatwave_decryption,
power_play, frostbite_facility} (local) and signal_lost_{bar_chip_ultimatum,
links_ascent_lesson, marks_keep_climbing, spindle_turns_again} (frontier), all under
output/otr/episodes/<ep>/audio/. Code: nodes/_otr_line_hygiene.py,
_otr_line_composer.py, _otr_outline.py, _otr_dramatic_state.py,
_otr_dramatic_state_llm.py, _otr_specificity.py, _otr_casting.py,
_otr_story_quality_l12.py, _otr_story_critic.py, _otr_reroll.py, _otr_render_plan.py;
scripts/story_quality_scan.py.
