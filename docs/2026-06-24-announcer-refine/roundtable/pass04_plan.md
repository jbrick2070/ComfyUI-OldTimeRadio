# ANNOUNCER REDESIGN + NEWS CODA + KILL-2 -- FINAL CONVERGED BUILD TICKET (pass04)

Converged across R1 (creative arc) -> R2 (implementability) -> R3 (wiring) -> R4
(residual). Panel each round: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro; Claude
code-grounded judge+panelist. R4 found NO new architecture -- only spec-precision
fixes (now folded). This document is SELF-CONTAINED: a coder builds from this
alone. Every claim is grounded to the real files this session.

OPERATOR THESIS: the show TEACHES. Drama delivers; the NEWS is the payload,
explicit at the very end -- framed deliberately, never stealing the character
climax. EVERYTHING behind `story_scaffold` (the existing writer widget); when off,
byte-identical. NO workflow-JSON change (the widget already exists).

---

## SCOPE (first build) -- two deliberate cuts (panel-justified, deferrable)
- **No per-line register tag.** The contract reaches the body via OUTLINE
  injection + the deterministic `conflict_object` (KILL 1). A line-level tonal
  register is deferred; if added later it MUST thread via `meta["story_contract"]`
  and be rebuilt in `build_reroll_line_request` (:3922), or rerolls drop it.
- **No open spoiler belt.** Input starvation (severing `script_brief`) is the
  deterministic no-spoiler guarantee. The token-overlap reroll belt is deferred
  (it added an import cycle + an `ending_change`-availability question). Add it
  WRITER-side (pass a precomputed frozenset) only if a re-soak shows leaks.

This first build proves STRUCTURAL style + grounding + the teaching frame, not
in-dialogue tonal style (the deferred model-ceiling question).

---

## DATA CONTRACTS (inline; do not cross-reference)

```python
# nodes/_otr_style_catalog.py  -- get_style(slug) dict already has label,
# sound_world, story_engine, ending_mode, ending_tag; ending_template_for(slug)
# and render_style_grammar(slug) already exist.
@dataclass(frozen=True)
class StoryContract:
    slug: str
    label: str
    sound_world: str
    story_engine: str
    ending_mode: str
    ending_tag: str
    ending_template: str
    grammar: str            # == render_style_grammar(slug); CONSUMED at the macro
                            # prompt -> finally gives render_style_grammar a caller
                            # (the literal KILL-2 "zero callers" fix)

def build_story_contract(cast_seed, script_brief: str, news_seed: str, meta) -> StoryContract:
    text = (script_brief or news_seed or "")
    slug = select_style(text, meta, cast_seed)   # single caller today (:3224);
                                                 # pick is sha256(cast_seed)-keyed,
                                                 # only the emergency-pool input
                                                 # shifts premise->text (flag-gated)
    s = get_style(slug) or {}
    return StoryContract(
        slug=slug, label=s.get("label",""), sound_world=s.get("sound_world",""),
        story_engine=s.get("story_engine",""), ending_mode=s.get("ending_mode",""),
        ending_tag=s.get("ending_tag",""), ending_template=ending_template_for(slug),
        grammar=render_style_grammar(slug))

# in the line composer module (or a small shared util)
@dataclass(frozen=True)
class SafeOpenBrief:
    setting: str
    time_of_day: str
    opening_status_quo: str
    cast: tuple[str, ...]   # allowed proper names (the LOCKED cast)
    era: str = ""
```

`OutlineRequest` (frozen, `_otr_outline.py:283`) gets TWO new defaulted fields,
DISTINCT from the existing user `style`:
`style_grammar: str = ""` and `story_engine: str = ""`.

`compose_announcer_intro` (backward-compatible; current sig `(*, creative_fn,
script_brief, creative_repo_id=None)`):
```python
def compose_announcer_intro(*, creative_fn, script_brief, creative_repo_id=None,
        story_scaffold=False, safe_open_brief=None):
```

`compose_announcer_outro` (extends current sig):
```python
def compose_announcer_outro(*, creative_fn, script_brief, news_close_brief,
        intro_text, creative_repo_id=None, ending_change="", final_character_line="",
        story_scaffold=False, climax_character_line="", coda_lead_in=""):
```

Constant: `NEWS_CODA_LEAD_IN = "The real story:"`  (operator's lean; he may swap to
an in-voice variant -- "From tonight's headlines:" / "The true account:" -- this
ONE constant is the single source used at the call site, the validator, and the
fallback). Final wording = operator's creative call.

---

## STEP A -- hoist the flag gate (R3/R4)
After `_apply_story_scaffold_env(story_scaffold)` (:2402), compute
`_style_grammar_on = _OTRCFG.style_grammar_enabled()` at run() top (today it is
computed late at :3216 -- move it up). ALL new branches gate on this ONE variable;
pass `story_scaffold=_style_grammar_on` to both composers (never the raw widget,
never a new env). OFF => kill-switch => every new branch below is skipped =>
byte-identical.

## STEP B -- build the contract pre-outline (every pass, incl. refine)
After cast-lock (`cast_seed` @ :2878) + briefs (`script_brief` @ :2785), BEFORE
`OutlineRequest` (:3032), OUTSIDE any `if not _refine_active` guard:
```python
contract = None
if _style_grammar_on:
    contract = _OTRSTYLE.build_story_contract(
        cast_seed, script_brief, str(resolved.get("news_seed","") or ""), meta)
    meta.setdefault("story_quality", {})
    meta["story_contract"] = {"slug": contract.slug, "label": contract.label,
                              "ending_tag": contract.ending_tag}
```
(`resolved` is the established run() local -- the existing KILL-1 code already uses
`resolved.get("news_seed","")` at :3261. `meta` is a dict.)

## STEP C -- OutlineRequest carries the style; thread to phase/beat (R3/R4)
At the `OutlineRequest(...)` call (:3032) add, under flag:
`style_grammar=(contract.grammar if contract else "")`,
`story_engine=(contract.story_engine if contract else "")`.
- `_build_macro_user_prompt` (:1133) renders `req.style_grammar` only when
  non-empty (this is where the full grammar block -- incl `sound_world` -- lives;
  it is the ONLY place sound_world appears; it shapes structure/mood, never a
  dialogue line).
- `macro` is the LLM OUTPUT object -> request fields do NOT pass through it. Add an
  explicit `story_engine: str = ""` param to `_build_phase_user_prompt` (:1187) and
  `_build_beat_user_prompt` (:1236); pass `req.story_engine` from the outline call
  sites. OMIT `sound_world` from phase/beat (keep audio vocab out of the
  beat-intent->line path).
- VERIFY the macro STRUCTURED-OUTPUT parse is unaffected by the added prompt text
  (it is extra instruction, not a schema key).

## STEP D -- capture the SAFE OPEN brief AFTER the outline, BEFORE KILL-4 mutates (R3/R4)
build_sq_data (:3245) mutates `beat.intent` IN PLACE (l12:803) and KILL-4 enriches
SETUP -> capture the setup intent BEFORE that. Place AFTER `outline =
generate_outline(...)` (:3158) and BEFORE build_sq_data (:3245), under the flag:
```python
safe_open_brief = None
if _style_grammar_on:
    _open_status_quo = ""
    for _b in outline.beats:
        if str(getattr(_b, "speaker_role", "")) == "character":
            _open_status_quo = _OTRLC.clean_one_line(str(getattr(_b,"intent","") or ""), 200)
            break
    safe_open_brief = SafeOpenBrief(
        setting=str(getattr(outline, "setting", "") or ""),
        time_of_day=str(getattr(outline, "time_of_day", "") or ""),
        opening_status_quo=_open_status_quo,
        cast=tuple(character_cast),          # the LOCKED cast already passed to
                                             # OutlineRequest.character_cast
        era=str(meta.get("period", "") or ""))
```

## STEP E -- OPEN composed in the line loop (input starvation)
At the intro call site (:4465), under flag, build from the safe brief and pass NO
script_brief content:
```python
line_res = _OTRLC.compose_announcer_intro(
    creative_fn=creative_generate_fn,
    script_brief=("" if _style_grammar_on else script_brief),
    creative_repo_id=resolved["creative_writing_model"],
    story_scaffold=_style_grammar_on, safe_open_brief=safe_open_brief)
```
`compose_announcer_intro` flag-ON: a flag-gated rewritten `_ANNOUNCER_INTRO_SYSTEM`
builds the prompt from `safe_open_brief.setting`, `.time_of_day`,
`.opening_status_quo`, `.era`, and renders `.cast` as the ONLY allowed proper names
("Use only proper names in this cast list; invent none" -- replaces the current
"names that appear in the brief" since the brief is no longer passed). Cold-open
shape: S1 orients (era/time/place/cast/status-quo), S2 intrigue, NO
outcome/twist/climax terms. Fallback = `fallback_safe_open(safe_open_brief)`
(deterministic; NEVER reads `script_brief`). Telemetry `open_safe_fallback`.

## STEP F -- CODA (UPDATED 2026-06-24 by the coda-segue roundtable)
> **SUPERSEDED -> build the coda from `coda-segue/roundtable/pass03_plan.md`
> (`compose_news_coda`): a DYNAMIC LLM bridge (from `outline.premise` + the safe
> `intro_text`, NEVER the outcome) + a deterministically-APPENDED `news_close_brief`,
> with a sha256(cast_seed) rotating-pool fallback. This REPLACES the fixed
> `NEWS_CODA_LEAD_IN` + `validate_news_coda_line` below, and DROPS the climax-line
> decoupling (the coda never touches the fictional climax -> "protect the climax"
> holds by construction; `compose_announcer_outro` stays UNTOUCHED = off-path
> byte-identical). The text below is the PRIOR fixed-lead-in design, kept for trace.**

## STEP F (PRIOR design -- superseded) -- CODA = the repurposed outro (post-loop)
Climax line lookup (byte-identical now; exactly one climax-class beat today, == the
last char beat). Take the LAST ledger line for that beat; fall back to the existing
last-character scan when `_climax_beat_id == ""`:
```python
_climax_line = ""
if _style_grammar_on and _climax_beat_id:
    for _ln in reversed(led.data.get("lines") or []):
        if str(_ln.get("beat_id") or "") == _climax_beat_id \
           and str(_ln.get("speaker_role") or "") == "character":
            _climax_line = str(_ln.get("text") or "").strip(); break
outro_res = _OTRLC.compose_announcer_outro(
    creative_fn=creative_generate_fn,
    script_brief=("" if _style_grammar_on else script_brief),   # drop fiction brief under flag
    news_close_brief=nc_brief, intro_text=intro_text,
    creative_repo_id=resolved["creative_writing_model"],
    ending_change=_outro_ending_change, final_character_line=_outro_final_char_line,
    story_scaffold=_style_grammar_on, climax_character_line=_climax_line,
    coda_lead_in=(NEWS_CODA_LEAD_IN if _style_grammar_on else ""))
```
INVARIANT NOTE: `_climax_beat_id` is the FIRST `CLIMAX_CLASS_ROLES` beat (:3271);
today there is exactly ONE such beat (== last char beat). If KILL 3 later allows
multiple, change the lookup to the LAST climax-class beat_id.

`compose_announcer_outro` flag-ON NEWS-CODA path:
- Do NOT add `script_brief` ("Tonight's story brief") to `user_parts`; do NOT use
  `brief` in fallback selection (R4 GPT#2 -- no fiction bleed into the real-news coda).
- flag-gated rewritten `_ANNOUNCER_OUTRO_SYSTEM`: state the REAL fact from
  `news_close_brief` plainly; "Write ONLY the fact body; do NOT write any
  introductory phrase" (the lead-in is prepended later -> prevents a double lead-in);
  the concrete-image rule is relaxed for the coda.
- Use `climax_character_line or final_character_line` for tone only.
- SUPPRESS the resolved-fiction branch under `if story_scaffold` (skip the "State
  this outcome plainly" block, :2854); inject `ending_change` as FORBIDDEN content
  ("do NOT restate this fictional outcome").
- EARLY-OUT fallback: when `story_scaffold`, the `if not brief and not close:`
  branch routes to `fallback_news_coda_outro(coda_lead_in, close)`, NEVER
  `_resolved_outro_fallback` (which leaks the fictional ending).
- EMPTY `news_close_brief` guard: if `story_scaffold and not close`, log a LOUD
  warning + return `LineResult(fallback_news_coda_outro(coda_lead_in, ""),
  compose_flags=("news_coda_fallback","news_coda_empty_close"))`; never pass an
  empty close into the LLM path. `fallback_news_coda_outro` must return a complete
  sentence even with an empty body.
- VALIDATION ORDER: run `validate_news_coda_line` on the RAW LLM body (assert NO
  lead-in variant present) BEFORE prefixing; THEN return `f"{NEWS_CODA_LEAD_IN}
  {body}"`. Same prefix on the fallback. Telemetry `news_coda_emitted`.

`validate_news_coda_line(text, *, lead_in, news_close_brief, ending_change,
min_words=18, max_words=45) -> (ok, cleaned)` (NO key_terms param -- derive
internally):
- word band [min_words, max_words]; no leading bracket; body contains NO lead-in
  variant; >=1 content term from `news_close_brief` present; reject if the body
  shares >=3 content tokens (lowercase alnum, len>=4, stopword-filtered) with
  `ending_change` (don't restate the fiction).

## STEP G -- select_style: REPLACE the source under flag, do NOT delete (R3/R4)
At :3216-:3230 (the existing style block), under flag set the locals from the
PRE-outline contract instead of re-drawing:
```python
if _style_grammar_on and contract is not None:
    _style_slug      = contract.slug
    _ending_tag      = contract.ending_tag
    _ending_template = contract.ending_template
```
Delete only the late `select_style(_premise_str, meta, cast_seed)` CALL for the
flag-ON path; keep the existing late path verbatim when OFF (so `_style_slug` etc.
are still populated off-flag -- byte-identical).

## STEP H -- KILL 4 (inline; do not cross-reference)
Role-keyed enrichment map keyed by the real constants (l12:55-72):
`BEAT_ROLE_SETUP`, `BEAT_ROLE_PRESSURE`, `BEAT_ROLE_PERSONAL_STAKE`, and EACH
member of `CLIMAX_CLASS_ROLES` (revelation / reversal / unresolved_final_sound /
reconciliation / bittersweet_parting / ironic_twist / quiet_acceptance /
confession), each with class-specific fallback content. `BEAT_ROLE_CONSEQUENCE` is
NOT in the map (cleanly omitted -- unreachable under climax-last today; revisit
with KILL 3; not a stub). Replace the post-enrichment truncation at l12:800
(`new_intent = new_intent.strip()[:_INTENT_MAX].strip()`) with a reserve+clamp:
```python
sep = " "
reserve = _INTENT_MAX - len(sep) - len(enrichment)
if reserve <= 0:
    new_intent = enrichment[:_INTENT_MAX]
else:
    new_intent = (original_intent[:max(0, reserve)].strip() + sep + enrichment.strip())[:_INTENT_MAX]
```
(`max(0, ...)` is required -- `s[:-5]` does NOT clamp to zero.) Test
`len(original)+len(enrichment) > 200`.

## STEP I -- KILL 3 DEFERRED (principle: climax position is spine-driven; remove
the FORCE, don't mandate a move). Only the outro climax-decoupling (STEP F) is
pulled forward and is byte-identical today.

---

## BUILD CHUNKS (separate commits; bisectable)
- C1 = STEP A+B+C+G (flag hoist + contract + OutlineRequest fields + prompt
  threading + select_style source-swap).
- C2 = STEP D+E (safe open + intro rewrite + fallback_safe_open).
- C3 = STEP F (coda) -- BUILD FROM `coda-segue/roundtable/pass03_plan.md`:
  `compose_news_coda` (dynamic bridge + appended `news_close_brief` + rotating-pool
  fallback). REPLACES the fixed lead-in; DROP the climax-line decoupling;
  `compose_announcer_outro` UNTOUCHED.
- C4 = STEP H (KILL 4 role map + truncation clamp).
Each: full suite + Bug Bible green vs the 5 pre-existing 267a53e workflow-pin
fails; run()-level OFF-flag golden tests (open line, outro line, ledger meta JSON);
`test_audio_byte_identical` green; commit AND push to v2.0-alpha per green chunk.
After C1-C4: LIVE re-soak (gemma + mistral) via the `story_scaffold` on/off toggle.

## BYTE-IDENTITY (explicit boundary)
OFF (`story_scaffold`=off -> `_style_grammar_on=False`): no contract build, no
safe_open_brief, no select_style source-swap (late path runs), no new prompt text,
no `meta.story_contract`, intro+outro run their CURRENT code verbatim (both still
receive the real `script_brief` off-flag). New `OutlineRequest` fields default ""
and render only under flag. VERIFY: any test comparing `OutlineRequest`
asdict/repr gains two default-"" keys -> update that fixture (do NOT change any
persisted ledger/meta serialization off-flag); add an off-flag ledger JSON golden.

## TELEMETRY (under flag only; primitives)
`sq = meta.setdefault("story_quality", {})`; then
`sq["story_contract_slug"]`, `sq["news_coda_emitted"]`, `sq["news_coda_fallback"]`,
`sq["open_safe_fallback"]`.

## INVARIANTS (a fix that breaks one is rejected)
Behind `story_scaffold` -> byte-identical off; audio spine FROZEN
(`test_audio_byte_identical`, mux-LAST); suite + Bug Bible per chunk; commit+push
per green chunk to v2.0-alpha; 100% local; determinism (cast_seed-keyed); LOUD
fallbacks; UTF-8 no BOM; SFW; prod/main + tags GATED.

## VERIFY-AT-BUILD CHECKLIST (each has a home above)
1. `era` source `meta.get("period","")`; if always empty, `era=""` is acceptable
   (the prompt leans on time_of_day). [STEP D]
2. `character_cast` (the locked cast) is in scope at STEP D (it is -- it is passed
   to OutlineRequest at :3037). Use it for `SafeOpenBrief.cast`, NOT
   `led.data["cast"]` (timing uncertain). [STEP D]
3. ledger line `beat_id` on the climax row at outro time (doc'd ledger:96); assert
   `_climax_line` non-empty when `_style_grammar_on`. [STEP F]
4. `news_close_brief` distinct from `ending_change` + never empty (guarded). [STEP F]
5. macro STRUCTURED-OUTPUT parse unaffected by the grammar-block injection. [STEP C]
6. `_build_phase/beat_user_prompt` new `story_engine` param compiles + all call
   sites updated. [STEP C]
7. both `_ANNOUNCER_*_SYSTEM` rewrites emit new text ONLY under flag. [STEP A/E/F]
8. `OutlineRequest` asdict/repr snapshot fixtures updated for the two new keys.
9. `build_reroll_line_request` remains the ONLY LineRequest rebuild site (moot now
   -- per-line register cut -- re-check if reintroduced).
10. run()-level OFF-flag golden + `test_audio_byte_identical` green. [BUILD CHUNKS]

## OPERATOR DECISION (not a build blocker)
`NEWS_CODA_LEAD_IN` wording: "The real story:" (operator's lean, reads modern) vs.
in-voice ("From tonight's headlines:", "The true account:") that protect the OTR
fiction. The constant is the single source; the coder wires whichever the operator
picks.
