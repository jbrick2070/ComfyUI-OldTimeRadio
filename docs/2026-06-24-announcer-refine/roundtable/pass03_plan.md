# ANNOUNCER REDESIGN + NEWS CODA + KILL-2 -- WIRING-HARDENED (pass03, post-R3)

R3 (wiring/sequencing) caught a real in-place-mutation sequence bug + several
call-site handoffs, and justified two scope cuts that remove the riskiest wiring.
Organized along the run() sequence so a coder follows it linearly. All grounded.

OPERATOR THESIS unchanged. All behind `story_scaffold`; byte-identical off.

---

## SCOPE CUTS adopted in R3 (reduce wiring risk; both panel-flagged deferrable)
- **CUT the per-line "compact register tag" from the first build.** It required a
  new LineRequest field + composer render + a meta stamp + a `build_reroll_line_
  request` rebuild (R3 reroll-loss catch). The contract already reaches the body
  via OUTLINE injection + the deterministic `conflict_object` (KILL 1). Add a
  line-level register ONLY if re-soak shows tonally flat dialogue -- and if so it
  MUST thread via `meta["story_contract"]` and be rebuilt in
  `build_reroll_line_request` (:3922) per Gemini.
- **DEFER the open spoiler BELT (forbidden_tokens reroll).** Input starvation is
  the deterministic guarantee; the belt added an import-cycle risk
  (`_otr_story_quality_l12` helpers into `_otr_line_composer`) + an
  `ending_change`-availability question at open time. First build = starvation
  only; add the belt (token extraction WRITER-side, passed as a precomputed
  frozenset so the composer needs no l12 import) only if re-soak shows leaks.

---

## STEP A -- flag gate, hoisted (R3 MUST: GPT#1)
Compute `_style_grammar_on = _OTRCFG.style_grammar_enabled()` immediately after
`_apply_story_scaffold_env(story_scaffold)` at run() top (:2402) -- BEFORE any
contract/outline construction (today it is computed late at :3216). ALL new gates
(contract, intro, outro, KILL-4 roles) read this ONE variable. Pass
`story_scaffold=_style_grammar_on` to both announcer composers (not the raw widget,
not a new env). OFF => kill-switch => none of the new branches run => byte-identical.

## STEP B -- build the contract pre-outline (every pass, incl. refine)
After cast-lock (`cast_seed` @ :2878) + briefs (`script_brief` @ :2785), BEFORE
`OutlineRequest` (:3032), OUTSIDE any `if not _refine_active` guard (it is needed
each refine pass; deterministic from cast_seed):
```python
contract = None
if _style_grammar_on:
    contract = _OTRSTYLE.build_story_contract(cast_seed, script_brief,
                   str(resolved.get("news_seed","") or ""), meta)
    meta["story_contract"] = {"slug": contract.slug, "label": contract.label,
                              "ending_tag": contract.ending_tag}
```
StoryContract / build_story_contract dataclass + signature: see pass02 §0. Drop
the dead `grammar` field UNLESS consumed (below it IS consumed at the macro prompt,
so KEEP it = `render_style_grammar(slug)`; this finally gives that function a
caller -- the literal KILL-2 fix).

## STEP C -- OutlineRequest carries the style (R3 MUST: DeepSeek#1/#2, Gemini#4)
Add ONE field to `OutlineRequest` (frozen, :283): `style_grammar: str = ""`
(distinct from the user `style`). At the call site (:3032) inject it under flag:
`style_grammar = (contract.grammar if contract else "")`. Render it in
`_build_macro_user_prompt` (:1133) only when non-empty.
- PHASE/BEAT threading (R3 MUST: GPT#3/DeepSeek#3/Gemini#4 -- decide now, not
  verify): `macro` is the LLM OUTPUT object; request fields do NOT pass through it.
  Add an explicit `story_engine: str = ""` param to `_build_phase_user_prompt`
  (:1187) + `_build_beat_user_prompt` (:1236) and pass `contract.story_engine`
  from the call sites. OMIT `sound_world` from phase/beat (it is audio vocabulary
  -> keep it out of the beat-intent->line path; the full grammar block, incl
  sound_world, lives ONLY in the macro prompt, which sets structure/mood not
  dialogue).

## STEP D -- capture the SAFE OPEN brief BEFORE KILL-4 mutates intents (R3 MUST: Gemini#1/GPT#4/DeepSeek#5)
build_sq_data (:3245) mutates `beat.intent` IN PLACE (`setattr(b,"intent",...)`,
l12:803), and KILL-4 now enriches the SETUP beat too -> reading the setup intent
later poisons `opening_status_quo` with enrichment text. So capture immediately
after `generate_outline` (:3158) and BEFORE build_sq_data (:3245):
```python
_open_status_quo = ""
for _b in outline.beats:
    if str(getattr(_b,"speaker_role","")) == "character":
        _open_status_quo = clean_one_line(str(getattr(_b,"intent","") or ""), 200)
        break
safe_open_brief = SafeOpenBrief(setting=outline.setting,
    time_of_day=outline.time_of_day, opening_status_quo=_open_status_quo,
    cast=tuple((r["name"], r.get("role","")) for r in (led.data.get("cast") or [])
               if r.get("name")),
    era=<meta/period or "">, tone=(contract.label if contract else ""))
```
Precompute ONCE here (not in the line loop, not from ledger lines).

## STEP E -- OPEN composed in the line loop (input starvation)
At the intro call site (:4465), under flag, pass the safe brief and DO NOT pass
`script_brief` content:
```python
line_res = _OTRLC.compose_announcer_intro(creative_fn=creative_generate_fn,
    script_brief=script_brief, creative_repo_id=resolved["creative_writing_model"],
    story_scaffold=_style_grammar_on, safe_open_brief=safe_open_brief)
```
`compose_announcer_intro` (signature pass02 §2): flag-ON builds the prompt from the
safe fields via a flag-gated rewritten `_ANNOUNCER_INTRO_SYSTEM`; the cast list is
rendered as the only allowed proper names (enforces "invent none"). Fallback =
`fallback_safe_open(safe_open_brief)` -- NEVER reads script_brief. (Belt deferred.)

## STEP F -- CODA = the repurposed outro (post-loop) (R3 MUST: GPT/DeepSeek/Gemini)
Climax line lookup (byte-identical now; climax==last): take the LAST ledger line
whose `beat_id == _climax_beat_id` (Gemini SHOULD#1: a beat -> multiple lines;
match the existing reversed() behavior); fall back to the existing last-character
scan when `_climax_beat_id == ""`.
```python
_climax_line = ""
if _style_grammar_on and _climax_beat_id:
    for _ln in reversed(led.data.get("lines") or []):
        if str(_ln.get("beat_id") or "") == _climax_beat_id \
           and str(_ln.get("speaker_role") or "") == "character":
            _climax_line = str(_ln.get("text") or "").strip(); break
outro_res = _OTRLC.compose_announcer_outro(creative_fn=..., script_brief=script_brief,
    news_close_brief=nc_brief, intro_text=intro_text,
    creative_repo_id=resolved["creative_writing_model"],
    ending_change=_outro_ending_change, final_character_line=_outro_final_char_line,
    story_scaffold=_style_grammar_on, climax_character_line=_climax_line,
    coda_lead_in=<fixed lead-in> if _style_grammar_on else "")
```
`compose_announcer_outro` (signature pass02 §3), flag-ON NEWS-CODA path:
- flag-gated rewritten `_ANNOUNCER_OUTRO_SYSTEM`: state the real fact from
  `news_close_brief` plainly; "Write ONLY the fact body; do NOT include any
  introductory phrase" (prevents the double-lead-in stutter); concrete-image rule
  relaxed for the coda.
- Use `climax_character_line or final_character_line` for tone (R3 GPT#8).
- SUPPRESS the resolved-fiction branch under an explicit `if story_scaffold` gate
  (R3 DeepSeek#8); inject `ending_change` as FORBIDDEN "do NOT restate this
  fictional outcome" content (DeepSeek#9).
- EARLY-OUT fallback (R3 Gemini#3 leak): when `story_scaffold`, the
  `if not brief and not close:` branch must route to `fallback_news_coda_outro`,
  NEVER `_resolved_outro_fallback` (which leaks the fictional ending).
- EMPTY news_close_brief guard (R3 GPT#9): if `story_scaffold` and `close` empty,
  emit `news_coda_fallback` LOUD + use `fallback_news_coda_outro(coda_lead_in, "")`;
  never pass an empty close into the LLM path silently.
- VALIDATION ORDER (R3 GPT#7/Gemini#2): run `validate_news_coda_line` on the RAW
  LLM body (asserts NO lead-in variant present) BEFORE prefixing; THEN return
  `f"{coda_lead_in} {body}"`. Same prefix on the fallback.
- `validate_news_coda_line(text, *, lead_in, news_close_brief, ending_change,
  key_terms=(), min_words=18, max_words=45)`: word band; no leading bracket; no
  lead-in variant in body; >=1 content term from `news_close_brief`; no strong
  content-token overlap with `ending_change`. Derive key terms from
  `news_close_brief` inside the validator if `key_terms` not threaded (R3 GPT#2).

## STEP G -- select_style: REPLACE the source, do NOT delete (R3 MUST: Gemini#5/GPT#2)
Do NOT blindly delete :3224 (it feeds `_style_slug`/`_ending_tag`/`_ending_template`
downstream). Under flag, set `_style_slug = contract.slug` (+ `_ending_tag =
contract.ending_tag`, `_ending_template = contract.ending_template`) so downstream
state is preserved from the PRE-outline contract; OFF, keep the existing late
`select_style(_premise_str, meta, cast_seed)` path byte-identically.

## STEP H -- KILL 4 (unchanged from pass02 §4)
Role-keyed map (real constants l12:55-72) for setup/pressure/personal_stake +
every CLIMAX_CLASS_ROLES member; DEFER consequence (don't delete). Truncation
clamp (the `max(0,...)` slice fix + reserve formula). Wire it INTO the actual line
(replace `new_intent = new_intent.strip()[:_INTENT_MAX].strip()` at :800).

---

## BUILD CHUNKS (separate commits; R2 GPT#7)
C1 = STEP A+B+C+G (contract + flag hoist + OutlineRequest field + prompt threading
+ select_style source-swap). C2 = STEP D+E (safe open). C3 = STEP F (coda). C4 =
STEP H (KILL 4). Each: full suite + Bug Bible green vs the 5 pre-existing 267a53e
fails; run()-level OFF-flag golden tests (open line, outro line, ledger meta);
commit AND push to v2.0-alpha per green chunk.

## BYTE-IDENTITY
OFF => STEP A sets `_style_grammar_on=False` => B/C/D/E/F news-path/G-swap/H all
skip => intro+outro run current code verbatim. New OutlineRequest `style_grammar`
defaults "" + renders only under flag. VERIFY: any test comparing OutlineRequest
asdict/repr (one new default-"" key) -> update fixture or gate serialization.

## TELEMETRY (under flag only)
`meta.story_quality.{story_contract_slug, news_coda_emitted, news_coda_fallback,
open_safe_fallback}`. Ensure `meta["story_quality"]` exists as a dict; primitives
only. (open_spoiler_* deferred with the belt.)

## INVARIANTS (a fix that breaks one is rejected)
Behind `story_scaffold` -> byte-identical off; audio spine FROZEN
(`test_audio_byte_identical`, mux-LAST); suite + Bug Bible per chunk; commit+push
per green chunk; 100% local; determinism (cast_seed); LOUD fallbacks; UTF-8 no
BOM; SFW; prod/main + tags GATED.

## VERIFY-AT-BUILD (carried, shrunk)
- `era` source for SafeOpenBrief (meta/period; "" ok).
- ledger line `beat_id` on the climax row at outro time (doc'd ledger:96).
- `news_close_brief` distinct from `ending_change`; never empty (guarded in F).
- OutlineRequest asdict/repr snapshot fixtures.
- `build_reroll_line_request` is the ONLY LineRequest reroll-rebuild site (moot
  for the first build now that the per-line register is cut; re-check if added).
