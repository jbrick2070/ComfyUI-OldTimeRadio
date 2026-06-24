# ASSUMPTIONS TO KILL -- R3 synthesis (wiring-exact, grounded build spec)

Convergence reached on substance; R3 nailed sequencing/integration. Build =
BUILD 1 (body gate) + BUILD 2 (StoryContract, ADD-not-collapse) + BUILD 3 (un-
starve body) + BUILD 4 (announcer policy). All new fields defaulted empty ->
byte-identical when the contract is absent / story-quality killed.

## BUILD 1 -- deterministic body-output gate (load-bearing)
- **Exact placement (grounded):** writer section I sets `cleaned` at two sites --
  `cleaned = _ex_text` (4163, use_exchange bypass) and `cleaned = line_res.text`
  (4206, normal compose) -- then the COMMON `last_lines.append(...,cleaned)`
  (4222). Put the gate AFTER `cleaned` is set, BEFORE 4222 + the ledger write, so
  it covers BOTH the exchange and compose paths.
- **Helper (new):** `_otr_story_quality_l12.validate_composed_grounding(text,
  sq_entry, grounded, *, max_ungrounded=0, require_conflict_object_on_roles=
  CLIMAX_CLASS_ROLES|{PRESSURE}) -> (ok, reasons)`. Crisis via
  `count_ungrounded_crisis`; object match by HEAD NOUN / token overlap.
- **Palette plumbing:** add `grounded_nouns: frozenset = field(default_factory=
  frozenset)` to `LineRequest`; compute once in writer section I via
  `premise_noun_palette(roster, news_seed, outline.premise, *premise_texts(meta))`
  (INCLUDE `outline.premise` -- R3 catch: it carries legit grounded objects).
  Thread into `_otr_reroll.build_reroll_line_request` too (else reroll loses it).
- **Reroll + failure cases (grounded):** failure modes differ. (a) original
  `compose_line` raises -> keep existing raise behavior. (b) original OK but
  validator fails -> ONE guarded reroll, hint built ONLY from the offending
  tokens (`low in GENERIC_CRISIS_NOUNS and low not in grounded`); on reroll
  exception keep the original + flag `grounding_reroll_failed`; ship
  deterministically; stamp `ungrounded_crisis:<n>`/`missing_conflict_object`.
- **Scope:** `speaker_role=="character"` only. **Telemetry:** in-place update of
  the existing `meta["story_quality"]` dict (do not replace): `body_gate_rerolls`,
  `body_gate_failed`, `body_gate_ungrounded_crisis`.
- **De-license the prompt:** scope composer 1162 ("mission control ... fine") +
  1442 ("Ground this line in the news facts") to compatible styles.

## BUILD 2 -- StoryContract: select pre-outline, inject into body, ADD-not-collapse
- **Dataclass (new, in `_otr_style_catalog`):** frozen `StoryContract(slug,
  label, sound_world, story_engine, ending_tag, ending_template, grammar)` over
  the EXISTING catalog keys (`ending_template=ending_template_for(slug)`,
  `grammar=render_style_grammar(slug)`) + `build_story_contract(premise, meta,
  seed)` + `as_meta_dict()`.
- **Sequencing (grounded):** `script_brief` is not set until `build_news_briefs`
  (~2658). Build the contract AFTER news interpretation, BEFORE D.5
  `OutlineRequest(...)`, input `script_brief if script_brief else
  resolved["news_seed"]`. Keep `resolved["style"]` for `build_news_briefs`
  (2666) + cast.
- **F2 (grounded):** the live F2 still calls `select_style(outline.premise,...)`.
  REPLACE that with the pre-built contract's fields; call `select_style` NOWHERE
  after `generate_outline` (else outline/body/outro grammar can disagree).
- **Outline injection:** add defaulted `style_grammar/sound_world/story_engine/
  ending_tag` to `OutlineRequest`; render `style_grammar` in `_build_macro_user_
  prompt` + `_build_phase_user_prompt` + `_build_beat_user_prompt` (today only
  macro sees `req.style`). Verify `_otr_pitch_room.run_pitch_room`'s
  `dataclasses.replace` PRESERVES these fields.
- **Body injection:** add `style_grammar/sound_world/story_engine` to
  `LineRequest`; render ONE canonical style-grammar block in `_build_user_prompt`
  for ALL character beats. Update every construction site + reroll rebuilder +
  the LineRequest tests. Keep `ending_template` as final-beat-only.
- **K5 -- ADD, do NOT collapse (grounded):** `resolved["style"]` feeds
  `build_news_briefs` (2666/2796/2924) + cast + `meta["style"]`/`visual_plan.
  style`. DO NOT overwrite those. Stamp `meta["story_contract"]=as_meta_dict()`
  alongside; migrate consumers later (deferred). [VERIFY first run: dump
  resolved["style"] vs contract.slug.]
- K9 (`select_style` "best-fit" wording) + K10 (DOMAIN_PALETTE scored matching,
  fix the "trial" medicine/law collision) -- do here, small.

## BUILD 3 -- un-starve the body beats
- Replace the `if beat_role in (PERSONAL_STAKE, IRREVERSIBLE_CHOICE)` gate (~694)
  + `_enrich_intent`'s "non-irreversible == personal_stake" branch with a
  ROLE-KEYED map covering setup/pressure/personal_stake + every assignable
  `CLIMAX_CLASS_ROLES` member, each with class-specific tail text. **CUT
  consequence** (R3: `assign_beat_roles` never assigns CONSEQUENCE under the
  climax-last validator -- unreachable until K4). Tests: revelation/reversal/
  confession/quiet do NOT receive personal-stake wording.
- Truncation fix: build the tail FIRST, reserve `len(sep+tail)` from `_INTENT_MAX`
  (200), truncate the ORIGINAL intent, then append the full tail.

## BUILD 4 -- announcer close policy (self-correction made real)
- Live: `compose_announcer_outro` (~2747) sets `resolved=is_resolved_ending_
  change(ending_change)` and, if resolved, injects "State this outcome plainly"
  (2819) -- opposite the grammar for unresolved/revelation/quiet. The live non-
  outcome closes came from `flag_thesis_close`, not my C5 gate.
- FIX: add `ending_tag: str=""` to `compose_announcer_outro`; for `ending_tag in
  {unresolved_final_sound, revelation, quiet_acceptance}` (the non-resolving set)
  force `resolved=False` + inject "do NOT resolve/state the outcome". Thread the
  same `ending_tag` into FALLBACK selection (R3 catch): use
  `fallback_announcer_outro`, NEVER `_resolved_outro_fallback`, for that set.
  Pass `contract.ending_tag` at the writer I.5 call. Retire the moot C5 outline-
  intent gate (keep no-op-safe).

## CORRECTION folded (K7)
- `ARC_PHASE_GUIDANCE` is dead ONLY in the line composer (`_position_for` fills
  `position` with "phase, beat N of M" minus the directive; `req.position` always
  truthy shadows the `elif req.arc_phase` branch). Fix: append
  `ARC_PHASE_GUIDANCE.get(req.arc_phase)` to the POSITION block (composer ~1213).
  It is NOT dead in the outline (`_phase_summary` 1233 uses it).

## DEFERRED (cascading-risk, not this build)
K4 climax POSITION (breaks validator + ending_template target + outro last-line
assumption); full two-style COLLAPSE (consumer migration); model-capability gate
(K1 net first); K11 render profiles; K8 _PERSONAL_COST domain rows.

## 3 HIGHEST-LEVERAGE STRUCTURAL CHANGES
1. **BUILD 1** -- in-loop body-output gate (covers the use_exchange bypass;
   grounded_nouns incl. outline.premise; offending-token reroll). The fix that
   stops gemma's fuel-cell standoff.
2. **BUILD 2** -- StoryContract selected pre-outline (after news), style grammar
   rendered into macro/phase/beat + every body LineRequest; ADD `meta.story_
   contract`, defer the collapse.
3. **BUILD 4 + BUILD 3** -- announcer close governed by `ending_tag` (prompt AND
   fallback), and the body beats un-starved with role-keyed enrichment.
