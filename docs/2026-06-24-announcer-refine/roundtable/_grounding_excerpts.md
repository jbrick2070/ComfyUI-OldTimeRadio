# GROUNDING EXCERPTS -- verbatim from the real files (2026-06-24)

These are exact copies of the current code. Verify every claim against THESE.
Do not invent APIs or contents you cannot see here.

## _otr_outline.py:1587-1596  -- the hardcoded announcer OPEN beat
```python
    beats.append(Beat(
        beat_id=_next_bid(),
        speaker="ANNOUNCER",
        speaker_role="announcer",
        intent="Open the episode and orient the listener.",
        target_words=15,
        mood="welcoming",
        sfx_cue=None,
        arc_phase=arc_phases[0],
    ))
```
NOTE: the OPEN's spoken content is NOT built from this `intent`. It is composed
by `compose_announcer_intro`, which reads `script_brief` (below). So editing this
intent does not change the open.

## _otr_line_composer.py:2519-2555  -- announcer system prompts (CURRENT)
```python
_ANNOUNCER_INTRO_SYSTEM = """\
You are the radio announcer for SIGNAL LOST, an old-time radio drama.
Write exactly ONE spoken opening line that frames tonight's story.

OUTPUT - strict:
- Only the words the announcer says out loud.
- One line. No line breaks.
- No speaker name, no colon, no quotation marks.
- No stage directions, no brackets, no sound cues.
- One or two sentences, roughly 12 to 30 words.

VOICE:
- A period radio host: warm, measured, a little mysterious.
- Orient the listener -- hint at the story, do not summarize it.
- Use only proper names that appear in the brief. Invent none.
"""

_ANNOUNCER_OUTRO_SYSTEM = """\
You are the radio announcer for SIGNAL LOST, an old-time radio drama.
Write exactly ONE spoken closing line that ends tonight's broadcast.

OUTPUT - strict:
- Only the words the announcer says out loud.
- One line. No line breaks.
- No speaker name, no colon, no quotation marks.
- No stage directions, no brackets, no sound cues.
- One or two sentences, roughly 14 to 34 words.

VOICE:
- A period radio host: warm, measured, reflective.
- Land the journalistic note from the closing brief.
- Lightly echo the opening line's tone; do not repeat its words.
- Use only proper names that appear in the briefs. Invent none.
- CLOSE ON A CONCRETE FINAL IMAGE: show what physically changed -- a person,
  an object, a place. Do NOT state a moral, lesson, or news-summary ("the
  lesson is", "reminding us", "tonight's revelation", "this shows").
"""
```
NOTE the tension for JOB 3 (news coda): the outro prompt CURRENTLY forbids a
news-summary / lesson framing and demands a concrete final image. An explicit
"here's the real story" teaching coda contradicts this voice as written.

## _otr_line_composer.py:2709-2775  -- compose_announcer_intro (CURRENT; only script_brief)
```python
def compose_announcer_intro(
    *,
    creative_fn,
    script_brief: str,
    creative_repo_id: str | None = None,
) -> LineResult:
    brief = clean_one_line(script_brief or "", max_chars=0)
    if not brief:
        return LineResult(
            text=fallback_announcer_intro(""),
            compose_flags=("announcer_intro_fallback",),
        )
    messages = [
        {"role": "system", "content": _ANNOUNCER_INTRO_SYSTEM},
        {"role": "user", "content": (
            f"Tonight's story brief:\n{brief}\n\n"
            f"Write the announcer's opening line now."
        )},
    ]
    raw = _announcer_generate(creative_fn, messages)
    cleaned = strip_line_formatting(raw or "")
    ok, validated = validate_announcer_line(
        cleaned, min_chars=_ANNOUNCER_INTRO_MIN_CHARS, max_chars=_ANNOUNCER_INTRO_MAX_CHARS)
    if ok:
        return LineResult(text=validated, compose_flags=("announcer_intro",))
    return LineResult(text=fallback_announcer_intro(brief),
                      compose_flags=("announcer_intro_fallback",))
```

## _otr_line_composer.py:2614-2632  -- fallback_announcer_intro (built from script_brief)
```python
def fallback_announcer_intro(script_brief: str) -> str:
    brief = clean_one_line(script_brief or "", max_chars=200)
    if brief:
        if brief[-1] not in ".!?":
            brief += "."
        return (f"Good evening. This is SIGNAL LOST. Tonight: {brief} Stay with us.")
    return ("Good evening. This is SIGNAL LOST. Tonight, a signal breaks "
            "through the static. Stay with us.")
```
NOTE: even the deterministic fallback echoes `script_brief` verbatim -- if the
brief contains the outcome, the fallback spoils too.

## _otr_line_composer.py:2778-2867  -- compose_announcer_outro (CURRENT; carries news_close_brief)
```python
def compose_announcer_outro(
    *, creative_fn, script_brief, news_close_brief, intro_text,
    creative_repo_id=None, ending_change="", final_character_line="",
) -> LineResult:
    resolved = is_resolved_ending_change(ending_change)
    ending = clean_one_line(ending_change or "", max_chars=240)
    final_line = clean_one_line(final_character_line or "", max_chars=240)
    brief = clean_one_line(script_brief or "", max_chars=0)
    close = clean_one_line(news_close_brief or "", max_chars=0)
    intro = clean_one_line(intro_text or "", max_chars=0)
    if not brief and not close:
        fb = (_resolved_outro_fallback(ending_change, close)
              if resolved else fallback_announcer_outro(close))
        return LineResult(text=fb, compose_flags=("announcer_outro_fallback",))
    user_parts = []
    if brief: user_parts.append(f"Tonight's story brief:\n{brief}")
    if close: user_parts.append(f"Closing brief (the journalistic note to land):\n{close}")
    if intro: user_parts.append(f"The announcer's opening line was:\n{intro}")
    if final_line: user_parts.append(f"The final character line was:\n{final_line}")
    if resolved and ending:
        user_parts.append("The dramatic question RESOLVED. The outcome: "
            f"{ending}\nState this outcome plainly in the close. Do NOT "
            "hedge -- do not say it 'remains to be seen' or 'time will tell'.")
    user_parts.append("Write the announcer's closing line now.")
    system_content = _ANNOUNCER_OUTRO_SYSTEM
    if resolved and ending:
        system_content = (_ANNOUNCER_OUTRO_SYSTEM
            + "\n- The story resolved tonight: state the outcome; never "
              "hedge or defer it to the future.")
    # ... LLM call + validate + fallback ...
```
NOTE: `news_close_brief` is the REAL journalistic note (the news coda payload).
It is ALREADY threaded here. `ending_change` is the FICTIONAL dramatic outcome
(different thing). The "State this outcome plainly" branch is about the FICTIONAL
ending, not the news.

## _otr_style_catalog.py:678-689  -- render_style_grammar (the KILL-2 dead block)
```python
def render_style_grammar(slug: str) -> str:
    s = get_style(slug)
    if not s:
        return ""
    return (
        f"Style: {s['label']}.\n"
        f"Sound world: {s['sound_world']}.\n"
        f"Story engine: {s['story_engine']}.\n"
        f"Ending mode: {s['ending_mode']}."
    )
```
pass04 (converged, grounded): this has ZERO callers -> the rich style grammar
never reaches the prompts; only `ending_tag` survives. KILL 2 wires a
StoryContract carrying these fields into macro/phase/beat + every body line.

## _otr_style_catalog.py:718-733  -- select_style (sha256 DRAW, not best-fit)
```python
def select_style(premise, meta, cast_seed) -> str:
    import hashlib
    emergency = premise_wants_emergency(premise, meta)
    pool = sorted(all_slugs() if emergency else non_emergency_slugs())
    if not pool:
        pool = sorted(all_slugs())
    h = int(hashlib.sha256(f"{cast_seed}:style:{int(emergency)}".encode("utf-8")).hexdigest(), 16)
    return pool[h % len(pool)]
```

## _otr_story_quality_l12.py:794-806  -- the KILL-4 enrich gate (only 2 roles enriched)
```python
        beat_role = entry["beat_role"]
        if beat_role in (BEAT_ROLE_PERSONAL_STAKE, BEAT_ROLE_IRREVERSIBLE_CHOICE):
            fc = fallback_content(beat_role, domain, seed, idx)
            entry.update(fc)
            new_intent = _enrich_intent(new_intent, beat_role, slot, fc)
        new_intent = new_intent.strip()[:_INTENT_MAX].strip()
        if new_intent and new_intent != intent:
            try:
                setattr(b, "intent", new_intent)
            except Exception:
                pass
        sq[bid] = entry
```
NOTE: setup / pressure / consequence / non-irreversible climax classes get NO
deterministic enrichment. KILL 4 = role-keyed map for all dramatic roles.
The `[:_INTENT_MAX]` (200) truncation happens AFTER enrichment -> can chop the
appended tail; KILL 4 fixes truncation order.
```

## R2 WIRING FACTS -- verified in OTR_LedgerScriptWriter.py.run() (single run() scope)
- `cast_seed = int(...)` at :2878; passed onward at :2908. `script_brief =
  briefs.script_brief` at :2785 (or "" at :2854). BOTH exist BEFORE
  `outline_req = OutlineRequest(...)` at :3032 and `generate_outline(...)` at
  :3101/:3158. => building a StoryContract pre-outline (after cast-lock, before
  OutlineRequest) is feasible from `script_brief or news_seed`. RESOLVES the
  cast-lock-timing question.
- TODAY the style is selected LATE, post-outline: `select_style(_premise_str,
  meta, cast_seed)` at :3224 (reads `outline.premise`). KILL 2 moves this
  pre-outline; this :3224 call is the one to delete/move.
- `_climax_beat_id` is computed at :3266-3272 (first beat whose role is in
  `CLIMAX_CLASS_ROLES`; comment: "exactly one, the last voiced character beat").
  In scope for the whole run(). Today climax == last char beat (forced).
- OPEN call site :4465-4471 -- `compose_announcer_intro(creative_fn=...,
  script_brief=script_brief, creative_repo_id=...)`. The open beat is
  `first_announcer_id`.
- OUTRO call site :4615-4634 -- `_outro_final_char_line` is built by iterating
  `reversed(led.data["lines"])` and taking the LAST `speaker_role=="character"`
  line (:4619-4623); passed as `final_character_line`. `_outro_ending_change`
  from `meta.dramatic_state.ending_change`. The outro beat is `last_announcer_id`.
- STRUCTURE: there is exactly ONE trailing announcer beat (`last_announcer_id`).
  Its in-loop value is a placeholder `fallback_announcer_outro(nc_brief)` (:4483),
  OVERWRITTEN post-loop by `compose_announcer_outro` (:4638 patch_line_text).
  => the NEWS CODA = the repurposed outro line; NO new beat is needed.

## _otr_outline.py:283-337 + macro schema -- OutlineRequest + outline fields
- `class OutlineRequest` (frozen dataclass) at :283. Fields: `news_seed`,
  `style` (user-selected style string, :303), `target_words`, `character_cast`,
  `script_brief=""`, `key_terms=()`. Adding NEW style-grammar fields with `=""`
  defaults preserves byte-identity. NOTE the existing `style` field is the
  USER style, distinct from the StoryContract -- do not conflate.
- The outline MACRO already produces `premise` (:211/:1045), `setting`
  (:212/:1046), `time_of_day` (:213/:1047). => the OPEN's structured inputs
  setting + time_of_day ALREADY EXIST; only `opening_status_quo` is NEW. (era
  source = verify-at-build, likely meta/period.)

## R3 WIRING FACTS -- flag plumb / refine loop / reroll rebuild (verified)
- SINGLE flag plumb: `_apply_story_scaffold_env(story_scaffold)` (writer :1551)
  runs ONCE at run() top (:2402) and resolves the auto/on/off widget ->
  `OTR_ENABLE_STYLE_GRAMMAR`. The in-run gate is `_style_grammar_on =
  _OTRCFG.style_grammar_enabled()` (:3216). "off" = kill-switch (byte-identical);
  "auto" restores the import-time baseline (no cross-prompt leak). => all new
  announcer/contract gates MUST hang on `_style_grammar_on` (NOT a new env).
- REFINE loop: `_refine_loop` (:2137) re-invokes `self.run(_refine_active=True,
  ..., _refine_forced_cast_seed=seed)` up to N times sharing ONE cast_seed
  (:2190). `_apply_story_scaffold_env` re-applies each pass (:2402). pitch_room is
  SKIPPED under `_refine_active` (:3053) -- the contract build must NOT be (it is
  deterministic from cast_seed and the outline needs it every pass).
- REROLL rebuild: `build_reroll_line_request` (writer :3922; meta stamp :4080,
  comment "build_reroll_line_request otherwise loses all of it. META ONLY") is how
  KILL 1 threaded `grounded_nouns`. Any NEW line-level field (the compact register)
  must be stamped on meta + rebuilt there, or a rerolled body line loses it.
  `conflict_object` is already handled by KILL 1.
