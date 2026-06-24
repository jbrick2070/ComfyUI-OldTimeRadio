# ASSUMPTIONS TO KILL -- R2 synthesis (implementation-hardened, grounded)

R1 found the seams; R2 made the fixes buildable and caught real ordering/
interface conflicts. Build scope = K1 + K2/K5 + K3 + the announcer-close fix.
DEFER K4, model-gate, render-profiles (cascading-risk; the body gate is the
model-agnostic safety net).

## BUILD 1 -- K1: deterministic body-output gate (the load-bearing fix)
Validate the SHIPPED character line, not the intent. Proven target: gemma's
"red lever / fuel cells / mission control".
- **Placement (corrected):** INSIDE the writer character branch, immediately
  after `line_res = compose_line(...)` and BEFORE `cleaned` is appended to
  `last_lines` / ledger (`OTR_LedgerScriptWriter.run` section I). A late
  `_otr_ledger_scrub` pass is too late -- the bad line already seeds the next
  beat's prompt. Gate `use_exchange` `_ex_text` too (or route it through the
  same validator).
- **New pure helper** (`_otr_story_quality_l12`):
  `validate_composed_grounding(text, sq_entry, grounded, *, max_ungrounded=0,
  require_conflict_object_on_roles=CLIMAX_CLASS_ROLES|{PRESSURE}) ->
  (ok, reasons)`. Crisis check reuses `count_ungrounded_crisis`; object match by
  HEAD NOUN / normalized token overlap (NOT full-string -- "the trial's
  enrollment list" would false-fail).
- **Reroll:** one guarded reroll via the existing reroll path, hint built ONLY
  from the offending tokens actually found (`low in GENERIC_CRISIS_NOUNS and low
  not in grounded`) -- never the whole set (would ban premise-legit "reactor").
  On `LineCompositionFailedError`, keep the original + stamp
  `grounding_reroll_failed:<reason>`; ship deterministically; stamp
  `ungrounded_crisis:<n>` / `missing_conflict_object`.
- **Data plumbing (R2 catch):** `LineRequest` has no grounded palette. Add
  `grounded_nouns: frozenset = frozenset()`; compute once in the writer via
  `premise_noun_palette(roster, news_seed, *premise_texts(meta))` and thread in.
- **Scope:** `speaker_role == "character"` only (never SFX/music/announcer rows).
- **Telemetry:** `meta.story_quality.{body_gate_rerolls, body_gate_failed,
  body_gate_ungrounded_crisis}` (setdefault/update, no overwrite).
- **Prompt de-licensing:** scope composer line 1162 ("mission control ... fine")
  and line 1442 ("Ground this line in the news facts") to compatible styles --
  they currently invite the exact failure surface.

## BUILD 2 -- K2/K5: one StoryContract, selected pre-outline, injected into the body
- **New frozen dataclass** in `_otr_style_catalog`: `StoryContract(slug, label,
  sound_world, story_engine, ending_tag, ending_template, grammar)` +
  `build_story_contract(premise, meta, seed) -> StoryContract` + `as_meta_dict()`.
- **Select PRE-outline (R2 catch):** move `select_style` to writer section D.2,
  feeding `script_brief or news_seed` (NOT `outline.premise`, which doesn't exist
  yet). Reuse the SAME contract in F2 -- do not reselect from `outline.premise`.
- **Inject into the OUTLINE:** add defaulted `style_grammar/sound_world/
  story_engine/ending_tag` to `OutlineRequest`; render `style_grammar` in
  `_build_macro_user_prompt` + `_build_phase_user_prompt` + `_build_beat_user_
  prompt` (today only macro sees `req.style`).
- **Inject into every BODY beat:** add `style_grammar/sound_world/story_engine`
  to `LineRequest`; render in `_build_user_prompt` THIS-BEAT for ALL character
  beats (not just the climax). Update EVERY `LineRequest` construction site + the
  reroll rebuilder (`_otr_reroll.build_reroll_line_request`) + tests.
- `render_style_grammar` (dead today, K6) becomes the canonical injector; keep
  `ending_template` as the final-beat landing instruction only.
- Collapse the two style systems (K5): the resolved widget `style_descriptor`
  and the catalog `_style_slug` become one `StoryContract.slug`. [VERIFY-AT-BUILD:
  dump both from one live ledger first to confirm they diverge.]

## BUILD 3 -- K3: stop starving the body beats
- `build_sq_data` enrich gate is `if beat_role in (PERSONAL_STAKE,
  IRREVERSIBLE_CHOICE)` (~694); `_enrich_intent` treats every non-irreversible
  role as personal_stake. Replace with a ROLE-KEYED map covering every
  `CLIMAX_CLASS_ROLES` member + setup/pressure/consequence (class-specific text;
  revelation/reversal/confession must NOT get personal-stake wording). Add tests.
- Truncation (R2 catch): `_enrich_intent` appends then `[:200]` -- build the tail
  FIRST, reserve its length, truncate the ORIGINAL intent, never the clause.

## BUILD 4 -- announcer close (self-correction, made real)
- The live non-outcome closes came from `flag_thesis_close`, not my C5 gate.
  `compose_announcer_outro` (composer ~2747) has an F3 branch: when
  `is_resolved_ending_change(ending_change)` it emits "State this outcome plainly"
  (line 2819) -- directly opposite the grammar's intent for unresolved/revelation/
  quiet styles.
- FIX: add `ending_tag: str = ""` to `compose_announcer_outro`; for
  `unresolved_final_sound` (and revelation/quiet image classes) force
  `resolved=False` + append "do not resolve the outcome", overriding F3. Pass
  `ending_tag` from the StoryContract at the writer's section I.5 call site.
- Retire the now-moot C5 outline-intent gate (or keep it only as a no-op-safe
  default; the real lever is here).

## DEFERRED (cascading-risk; not this build)
- **K4 climax POSITION** -- forcing climax-last is wrong, but moving it breaks
  `validate_beat_roles`, the `ending_template` injection target, AND the outro's
  "last character line = resolution" assumption (composer reversed-search; Gemini
  + GPT). If/when done: return `(climax_beat_id, post_climax_roles)`, relax the
  validator, look up `_climax_beat_id` for `final_character_line` instead of the
  last char line.
- **Model-capability gate** -- prefer mistral / branch weak models. The K1 body
  gate is the model-agnostic safety net and ships first; revisit a model registry
  flag after.
- **K11 style-driven render profile** -- thread `sound_world`/`style_slug` into
  the render request builder first; sampler-profile knobs later (high engine
  risk).
- **K8 _PERSONAL_COST** -- general-only is harmless fallback tech-debt; either add
  domain rows or drop the "domain-specific personal stake" claim. Low priority.
- **K9 select_style "best-fit" wording**, **K10 DOMAIN_PALETTE scored matching**
  -- small, do alongside Build 2.

## CORRECTIONS folded from R2
- K7 refined: `ARC_PHASE_GUIDANCE` is NOT globally dead -- `_otr_outline.
  _phase_summary` (1233) uses it in the OUTLINE beat prompts. It is dead only in
  the LINE COMPOSER (shadowed by `position`, which `_position_for` fills with
  "phase, beat N of M" minus the directive). Fix: append the guidance to the
  POSITION block unconditionally (composer ~1213).

## 3 HIGHEST-LEVERAGE (unchanged, now buildable)
1. K1 body-output gate (in-loop, grounded_nouns plumbed, offending-token reroll).
2. K2/K5 StoryContract selected pre-outline + style injected into every body beat.
3. Announcer-close policy unified in `compose_announcer_outro` by `ending_tag`
   (Build 4) -- the model-capability gate is deferred behind the K1 net.
