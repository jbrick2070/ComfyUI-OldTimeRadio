# CODE MAP -- exact anchors for the announcer/KILL-2 build (companion to pass04_plan.md)

Goal: the coder edits by ANCHOR, never by reasoning. Every entry gives the file,
the line number AS OF HEAD `b717980d`, and a unique GREP ANCHOR string (grep that
if the line drifted), plus the existing pattern to MIRROR.

Paths are repo-relative to `...\ComfyUI-OldTimeRadio\nodes\`.
Rule: grep the anchor; if the line number disagrees, trust the anchor.

---

## 0. REUSE INVENTORY (import these; do NOT reinvent)

| symbol | file:line | grep anchor |
|---|---|---|
| `render_style_grammar(slug)` | _otr_style_catalog.py:678 | `def render_style_grammar(` |
| `select_style(premise,meta,cast_seed)` | _otr_style_catalog.py:718 | `def select_style(` |
| `ending_template_for(slug)` | _otr_style_catalog.py:612 | `def ending_template_for(` |
| `get_style(slug)` -> dict (label/sound_world/story_engine/ending_mode/ending_tag) | _otr_style_catalog.py:654 | `def get_style(` |
| `premise_wants_emergency(premise,meta)` | _otr_style_catalog.py:706 | `def premise_wants_emergency(` |
| `CLIMAX_CLASS_ROLES` (frozenset) | _otr_story_quality_l12.py:72 | `CLIMAX_CLASS_ROLES = frozenset(` |
| role consts SETUP/PRESSURE/PERSONAL_STAKE/CONSEQUENCE | _otr_story_quality_l12.py:55-58 | `BEAT_ROLE_SETUP = "setup"` |
| `select_domain(meta,premise)` | _otr_story_quality_l12.py:375 | `def select_domain(` |
| `assign_conflict_slot(domain,beat_index,seed)` | _otr_story_quality_l12.py:401 | `def assign_conflict_slot(` |
| `_TOKEN_RE` | _otr_story_quality_l12.py:418 | `_TOKEN_RE = re.compile(` |
| `premise_noun_palette(roster,*texts)` | _otr_story_quality_l12.py:421 | `def premise_noun_palette(` |
| `count_ungrounded_crisis(intent,grounded)` | _otr_story_quality_l12.py:457 | `def count_ungrounded_crisis(` |
| `_strip_possessive(tok)` | _otr_story_quality_l12.py:497 | `def _strip_possessive(` |
| `_content_tokens(text)` -> frozenset | _otr_story_quality_l12.py:509 | `def _content_tokens(` |
| `fallback_content(role,domain,seed,idx)` | _otr_story_quality_l12.py:699 | `def fallback_content(` |
| `premise_texts(meta)` | _otr_story_quality_l12.py:810 | `def premise_texts(` |
| `_enrich_intent(...)` | _otr_story_quality_l12.py:821 | `def _enrich_intent(` |
| `clean_one_line(text,max_chars)` | _otr_line_composer.py:2558 | `def clean_one_line(` |
| `strip_line_formatting(raw)` | _otr_line_composer.py:177 | `def strip_line_formatting(` |
| `validate_announcer_line(text,*,min_chars,max_chars)` | _otr_line_composer.py:2581 | `def validate_announcer_line(` |
| announcer char bands (24/300/28/340) | _otr_line_composer.py:2508-2511 | `_ANNOUNCER_INTRO_MIN_CHARS =` |
| `LineRequest` (dataclass) | _otr_line_composer.py:589 | `class LineRequest` |
| `LineResult` (dataclass) | _otr_line_composer.py:777 | `class LineResult` |
| `build_reroll_line_request(...)` | _otr_reroll.py:259 | `def build_reroll_line_request(` |
| ledger lines carry `beat_id` (+ shot_id, boundary) | _otr_ledger.py:96 | `lines[].beat_id` |

NOTE: the writer module is `OTR_LedgerScriptWriter.py`; it imports the composer as
`_OTRLC`, style catalog as `_OTRSTYLE`, l12 as `_OTRSQL12`, config as `_OTRCFG`,
outline as `_OTRO`, ledger as `_OTRL`. Mirror those aliases.

---

## 1. NEW SYMBOLS TO CREATE (where to put each)

| new symbol | put in | next to (grep anchor) |
|---|---|---|
| `class StoryContract` (frozen) | _otr_style_catalog.py | after `def render_style_grammar(` :678 |
| `build_story_contract(cast_seed,script_brief,news_seed,meta)` | _otr_style_catalog.py | after `def select_style(` :718 |
| `class SafeOpenBrief` (frozen) | _otr_line_composer.py | before `class LineRequest` :589 |
| `NEWS_CODA_LEAD_IN = "The real story:"` | _otr_line_composer.py | by the bands `_ANNOUNCER_INTRO_MIN_CHARS =` :2508 |
| `fallback_safe_open(safe_open_brief)` | _otr_line_composer.py | after `def fallback_announcer_intro(` :2614 |
| `fallback_news_coda_outro(coda_lead_in,news_close_brief)` | _otr_line_composer.py | after `def fallback_announcer_outro(` :2635 |
| `validate_news_coda_line(...)` | _otr_line_composer.py | after `def validate_announcer_line(` :2581 |
| KILL-4 role->content map + class text | _otr_story_quality_l12.py | by `def fallback_content(` :699 / `def _enrich_intent(` :821 |

Field lists + signatures: pass04_plan.md "DATA CONTRACTS".

---

## 2. CHUNK C1 -- StoryContract + flag hoist + OutlineRequest + select_style swap

All in `OTR_LedgerScriptWriter.py` unless noted.

- **Flag hoist.** grep `def _apply_story_scaffold_env(` (:1551) defines it; its CALL
  is `_scaffold = _apply_story_scaffold_env(story_scaffold)` (:2402). Immediately
  AFTER that call, add `_style_grammar_on = _OTRCFG.style_grammar_enabled()`. Then
  DELETE the late recompute -- grep `_style_grammar_on = _OTRCFG.style_grammar_enabled()`
  at :3216 (the existing one) and remove it (now hoisted).
- **Build contract.** Anchor: the line `cast_seed = int(` (:2878) is where cast_seed
  binds; `script_brief = briefs.script_brief` (:2785) binds the brief. Insert the
  `build_story_contract(...)` block (pass04 STEP B) BEFORE grep `outline_req = _OTRO.OutlineRequest(`
  (:3032), OUTSIDE any `if not _refine_active`. `resolved.get("news_seed","")` is
  the established local (already used at grep `resolved.get("news_seed"` :3261).
- **OutlineRequest fields.** grep `class OutlineRequest` (_otr_outline.py:283); add
  `style_grammar: str = ""` and `story_engine: str = ""` after grep
  `script_brief: str = ""` (:322). At the call site grep `outline_req = _OTRO.OutlineRequest(`
  (:3032) add `style_grammar=(contract.grammar if contract else "")`,
  `story_engine=(contract.story_engine if contract else "")`.
- **Macro prompt render.** grep `def _build_macro_user_prompt(req: OutlineRequest)`
  (_otr_outline.py:1133); render `req.style_grammar` when non-empty (this is the
  ONLY place sound_world appears -- structure/mood, never dialogue).
- **Phase/beat threading.** grep `def _build_phase_user_prompt(` (:1187) +
  `def _build_beat_user_prompt(` (:1236); add a `story_engine: str = ""` PARAM to
  each (they take `macro`, NOT the request -> request fields do NOT pass through);
  pass `contract.story_engine` from their call sites inside `generate_outline`.
  OMIT sound_world here.
- **select_style swap.** grep `select_style(_premise_str, meta, cast_seed)` (:3224):
  under flag set `_style_slug = contract.slug` / `_ending_tag = contract.ending_tag`
  / `_ending_template = contract.ending_template` (mirror the existing assignments
  just below it at grep `_ending_template = _OTRSTYLE.ending_template_for(` :3230);
  keep the late `select_style` call for the OFF path (byte-identical).
- **meta dict.** `meta.setdefault("story_quality", {})` then `meta["story_contract"]
  = {...}` (pass04 STEP B). Mirror the existing meta-stamp style at grep
  `meta["story_quality_l12_enabled"] = True` (:3253).

## 3. CHUNK C2 -- safe open

- **Capture BEFORE mutation.** grep `outline = _OTRO.generate_outline(` (:3158 path)
  -- the capture block (pass04 STEP D) goes AFTER `outline` is bound and BEFORE grep
  `_sq_by_beat = _OTRSQL12.build_sq_data(` (:3245). Guard under `if _style_grammar_on:`.
  Cast from the LOCKED cast already passed as `OutlineRequest.character_cast` (grep
  `character_cast=` near :3037), NOT `led.data["cast"]`. `era = meta.get("period","")`.
- **Intro signature.** grep `def compose_announcer_intro(` (_otr_line_composer.py:2709);
  add the keyword-only defaults `story_scaffold=False, safe_open_brief=None`. Flag-ON
  branch builds from the safe fields; rewrite `_ANNOUNCER_INTRO_SYSTEM` (grep
  `_ANNOUNCER_INTRO_SYSTEM = """` :2519) FLAG-GATED ("use only names in this cast
  list"). Mirror the existing ok/fallback return shape in the same function.
- **Intro fallback.** `fallback_safe_open` (new) -- NEVER reads script_brief; mirror
  `def fallback_announcer_intro(` (:2614) template style.
- **Call site.** grep `line_res = _OTRLC.compose_announcer_intro(` (:4465); under flag
  pass `script_brief=("" if _style_grammar_on else script_brief)`,
  `story_scaffold=_style_grammar_on`, `safe_open_brief=safe_open_brief`.

## 4. CHUNK C3 -- news coda (the outro)
> **UPDATED 2026-06-24: build the coda per `coda-segue/roundtable/pass03_plan.md`
> (`compose_news_coda` -- NEW function in `_otr_line_composer.py` near
> `def compose_announcer_outro(` :2778; dynamic bridge from `outline.premise` +
> `intro_text`, append `news_close_brief`, sha256(cast_seed) rotating-pool fallback,
> coda-specific `validate_news_coda_bridge`). The call-site branch is at the outro
> call (grep `outro_res = _OTRLC.compose_announcer_outro(` :4626): `if _style_grammar_on
> and nc_brief -> compose_news_coda(...)` else the UNCHANGED `compose_announcer_outro`.
> The fixed-lead-in items + the climax-line decoupling below are SUPERSEDED -- skip them.**

- **Outro signature.** grep `def compose_announcer_outro(` (:2778); add keyword-only
  `story_scaffold=False, climax_character_line="", coda_lead_in=""`. Flag-ON NEWS-CODA
  path: do NOT append the `brief` ("Tonight's story brief") to user_parts; SUPPRESS
  the resolved-fiction block -- grep `State this outcome plainly` (:2854) wrap it in
  `if not story_scaffold:`; rewrite `_ANNOUNCER_OUTRO_SYSTEM` (grep
  `_ANNOUNCER_OUTRO_SYSTEM = """` :2536) FLAG-GATED ("write only the fact body; no
  intro phrase"). VALIDATE the raw body (new `validate_news_coda_line`) THEN return
  `f"{coda_lead_in} {body}"`.
- **Early-out + empty-close fallback.** grep `if not brief and not close:` (:2832) --
  under flag route to `fallback_news_coda_outro(coda_lead_in, close)`, NEVER grep
  `_resolved_outro_fallback(` (:2653). Add the empty-close LOUD guard.
- **Validator + fallback (new).** `validate_news_coda_line` near `def validate_announcer_line(`
  (:2581); `fallback_news_coda_outro` near `def fallback_announcer_outro(` (:2635).
  Derive key terms inside the validator from `news_close_brief` (reuse `_content_tokens`
  / `_TOKEN_RE` from l12 if importing is clean; else duplicate the tiny tokenizer).
- **Climax-line decoupling + call site.** grep `_outro_final_char_line = ""` (:4618)
  -- the existing loop (:4619-4623) takes the LAST character line. ADD a climax lookup
  (pass04 STEP F) using `_climax_beat_id` (grep `_climax_beat_id = str(_bid)` :3271,
  in scope) matching ledger `beat_id`. grep `outro_res = _OTRLC.compose_announcer_outro(`
  (:4626); add `story_scaffold=_style_grammar_on`, `climax_character_line=_climax_line`,
  `coda_lead_in=(NEWS_CODA_LEAD_IN if _style_grammar_on else "")`, and pass
  `script_brief=("" if _style_grammar_on else script_brief)`.

## 5. CHUNK C4 -- KILL 4 (all in _otr_story_quality_l12.py)

- **Role map.** grep `if beat_role in (BEAT_ROLE_PERSONAL_STAKE, BEAT_ROLE_IRREVERSIBLE_CHOICE):`
  (:795) -- replace the 2-role gate with a role->content map covering SETUP / PRESSURE
  / PERSONAL_STAKE + every `CLIMAX_CLASS_ROLES` member (class-specific text). Reuse
  `fallback_content` (:699) + `_enrich_intent` (:821). CONSEQUENCE is omitted (not a stub).
- **Truncation clamp.** grep `new_intent = new_intent.strip()[:_INTENT_MAX].strip()`
  (:800) -- replace with the reserve/clamp (pass04 STEP H; `max(0, ...)` required).

## 6. TESTS / FIXTURES (update per chunk; green before commit)
- Existing patterns to mirror: `tests/test_body_output_gate.py` (KILL 1),
  `tests/test_story_grammar_wiring.py` (default-OFF byte-identity + flag-ON render).
- New: off-flag golden (open line / outro line / ledger meta JSON); intro-never-reads-
  script_brief assert; coda lead-in present + no double lead-in; coda no-ending_change-
  overlap; KILL-4 truncation `len(original)+len(enrichment)>200`; OutlineRequest
  asdict/repr snapshot fixtures gain `style_grammar`/`story_engine` default-"" keys.
- Run: `$env:PYTHONUTF8=1; pytest -q -p no:cacheprovider` (Windows venv); Bug Bible
  via the survival-guide repo relative `tests\bug_bible_regression.py`;
  `test_audio_byte_identical` (OTR_REGRESSION_RUNTIME=1) green or recapture-justified.

## 7. THINGS THAT MUST NOT MOVE (byte-identity)
OFF (`story_scaffold`=off -> `_style_grammar_on=False`): every block above is skipped;
intro+outro both still receive the real `script_brief`; no `meta.story_contract`; the
late `select_style` runs; new OutlineRequest fields stay "". The audio spine is FROZEN
(`test_audio_byte_identical`, mux-LAST). Commit AND push per green chunk to v2.0-alpha.
