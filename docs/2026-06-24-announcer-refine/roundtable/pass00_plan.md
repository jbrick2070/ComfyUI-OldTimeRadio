# ANNOUNCER REDESIGN + NEWS CODA + KILL-2 INJECTION -- DESIGN TO HARDEN (pass00)

Refine-before-code roundtable. We are hardening the CREATIVE + BUILD design for
three coupled changes before a coder window implements them, all behind the
existing `story_scaffold` flag (byte-identical when off):

1. KILL 2 -- StoryContract: make the catalog style actually shape the body.
2. Announcer redesign -- a teaching frame: scene-setting OPEN (no spoilers) +
   character CLOSE (drama in their voices) + NEWS CODA (the real fact, every
   episode, lightly labeled).
3. KILL 4 -- un-starve the body (role-keyed enrichment).

OPERATOR THESIS (honor it): the show TEACHES. The drama is the delivery; the
NEWS is the payload, explicit at the very end. We do NOT suppress the real-world
outcome -- we frame it deliberately as a coda, AFTER the characters land the
fictional climax, so the news teaches without stealing the drama.

This doc is what the panel critiques. The four pointed asks are in section 5.

---

## 0. GROUNDED SEAM INDEX (verified against the real files this session)

All CONFIRMED by reading the actual code (not the plan prose):

- Open beat intent is HARDCODED: `_otr_outline.py:1591`
  `intent="Open the episode and orient the listener."` (target_words=15,
  mood="welcoming", speaker_role="announcer").
- The OPEN is composed by `compose_announcer_intro` (`_otr_line_composer.py:2709`)
  -- and it takes **only `script_brief`** (plus `creative_fn`, `creative_repo_id`).
  It does NOT currently receive time / place / cast / opening-situation. It is an
  LLM pass (`_announcer_generate`) with a deterministic `fallback_announcer_intro`.
  IMPORTANT: the hardcoded `beat.intent` at 1591 is NOT the lever for the open's
  content -- the composer reads `script_brief`, not `beat.intent`. The real lever
  is `compose_announcer_intro`'s inputs + system prompt + its writer call site.
- The CLOSE is composed by `compose_announcer_outro` (`_otr_line_composer.py:2778`),
  driven by `script_brief` + `news_close_brief` + `intro_text`, with optional
  `ending_change` / `final_character_line`. Fallbacks: `fallback_announcer_outro`
  (`:2635`) and `_resolved_outro_fallback` (`:2653`). The F3 "State this outcome
  plainly" branch is at `:2854-2867` (gated by `is_resolved_ending_change`,
  `_otr_dramatic_state.py:90`).
- `news_close_brief` is real + wired: authored in `news_interpreter.py`
  (Field `:167`, `<=250` chars), threaded to the outro in
  `OTR_LedgerScriptWriter.py:3945-3949` + `:4583-4629`. => the NEWS CODA is
  LARGELY WIRED already. (This is the KILL-5 reframe: the news close exists; the
  work is FRAMING + protecting the character climax, not building a news pull.)
- `render_style_grammar` (`_otr_style_catalog.py:678`) exists; pass04 (converged,
  grounded) found it has ZERO callers -> the rich style grammar never reaches the
  prompts; only `ending_tag` survives. (re-confirm zero-callers at build.)
- `select_style` (`_otr_style_catalog.py:718`) is a sha256 DRAW keyed by
  cast_seed, not a best-fit match.
- KILL 4 enrich gate: `build_sq_data` enriches only
  `beat_role in (PERSONAL_STAKE, IRREVERSIBLE_CHOICE)` (`_otr_story_quality_l12.py
  :700`); setup / pressure / consequence / non-irreversible climax classes are
  starved.

---

## 1. KILL 2 -- ONE StoryContract, selected pre-outline, injected into the body

From the converged assumption-audit `pass04_plan.md` (grounded):

- Build one frozen `StoryContract(slug, label, sound_world, story_engine,
  ending_tag, ending_template, grammar)` + `build_story_contract` in
  `_otr_style_catalog`. Build ONCE after cast-lock (seed = `cast_seed`) + news
  interpretation, BEFORE `OutlineRequest`, from `script_brief or news_seed`.
- Reuse it in F2 (delete the late `select_style(outline.premise, ...)`).
- Add style fields to `OutlineRequest`; render in
  `_build_macro/phase/beat_user_prompt`. Add the same fields to `LineRequest`;
  render for EVERY character beat (not just the climax).
- ADD `meta.story_contract`. Do NOT overwrite `resolved["style"]` /
  `meta.style` / `visual_plan.style` (they feed `build_news_briefs` + cast) --
  defer the two-system collapse.
- Make conflict objects premise-specific instead of the generic domain pool.

ACCEPTANCE: read N episodes -- the chosen style's register actually shows in the
content; two different styles on the same news produce visibly different stories.

OPEN RISK (this is ask #3): does injecting sound_world / story_engine into every
body-beat prompt actually MOVE a weak local writer (mistral-nemo / gemma)? KILL 1
already proved that instruction-following alone does NOT hold for the body -- we
needed a deterministic output gate. So: is KILL 2 a real lever, or does it need
its own deterministic enforcement like KILL 1?

---

## 2. ANNOUNCER REDESIGN -- three jobs

### JOB 1 -- THE OPEN: set the scene, do not give it away
Deterministically orient from outline + StoryContract: TIME (era + time_of_day),
PLACE (setting), WHO (cast by name + roles), WHERE-THEY-ARE-NOW (opening
situation / status quo). Orient + intrigue, then withhold.

HARD CONSTRAINT -- no spoilers: the open must NOT reveal the climax, outcome, or
twist. (Twilight-Zone cold-open register.)

BUILD: feed `compose_announcer_intro` the time/place/cast/opening-situation (new
structured inputs) from the outline + contract, with an explicit no-spoiler
constraint in the system prompt.

ACCEPTANCE: the open names setting + era + characters, states the opening
situation, and contains NO climax/outcome/twist words.

OPEN RISK (ask #2): the open is currently a free LLM pass over `script_brief` --
and `script_brief` can itself contain the outcome. A prompt instruction ("don't
spoil") is the SAME single-prior trap KILL 1 disproved. How do we enforce
no-spoiler DETERMINISTICALLY (structured open built from withheld-outcome fields
+ a post-gate that rejects/strips outcome tokens, reroll-once-else-fallback),
not just an instruction?

### JOB 2 -- THE CHARACTER CLOSE: the drama lands in their voices
The last voiced CHARACTER beat carries the dramatic climax (governed by
ending_tag / climax class). The announcer must NOT restate the fictional
resolution as if it were the climax. (KILL 3 governs climax POSITION; deferred.)

### JOB 3 -- THE NEWS CODA: the teaching beat, at the very end
After the drama, the announcer delivers the REAL news/fact -- the educational
payload -- every episode, lightly labeled, recognizable (so listeners learn the
format: drama, then the real fact). Source the REAL fact (news_close_brief), NOT
the fictional outcome.

GROUNDING WIN: the close already pulls the real news via `news_close_brief ->
compose_announcer_outro`. So the coda is largely WIRED; the work is (a) frame it
as a deliberate "here's the real story" coda, and (b) ensure it reads as the
real-world fact AFTER the character climax, not as the drama's resolution.

This REPLACES the old KILL-5 "force resolved=False / suppress" plan: keep the
news, frame it, protect the character climax.

OPEN RISK (ask #1): the news-coda lead-in phrasing. Operator leans a light fixed
tag ("The real story:" / "What actually happened:"). Pressure-test teachability
(consistency = the audience learns the format) vs. heavy-handedness (a fixed tag
every episode could feel mechanical / break the OTR fiction). Fixed vs.
varied-but-recognizable?

---

## 3. KILL 4 -- un-starve the body
Role-keyed enrichment map for setup / pressure / personal_stake + every
CLIMAX_CLASS_ROLES member (class-specific text; CUT consequence -- unreachable
under climax-last). Fix the 200-char truncation order (reserve the tail, truncate
the original).

---

## 4. KILL 3 -- climax POSITION = spine-driven (DEFERRED, but settle the principle)
The current `assign_beat_roles` forces `i==n-1 -> climax` (`l12 ~511`) and
`validate_beat_roles` makes it law (`~558`). Operator (ask #4): a last-beat crux
is FINE when the spine calls for it -- many stories peak last. The fix is to
remove the FORCE (let the spine/ending class decide position), NOT to mandate a
move. Confirm: relax the validator to allow either; keep last-beat valid + common.

---

## 5. THE FOUR ASKS FOR THE PANEL (attack these)

1. **NEWS-CODA lead-in phrasing.** Light fixed tag ("The real story:") vs. a
   varied-but-recognizable sign-off. Optimize for TEACHABILITY without
   heavy-handedness. Recommend a concrete approach.

2. **THE OPEN -- deterministic no-spoiler.** Confirm the 1-2 sentence cold-open
   structure, and specify how to enforce "no spoiler" DETERMINISTICALLY (not a
   prompt instruction): what structured inputs, what post-gate, what
   reroll/fallback. (Ground: the open is an LLM pass over `script_brief`, which
   can contain the outcome.)

3. **KILL-2 injection -- is it a real lever or the single-prior trap again?**
   ATTACK HARDEST. Does rendering sound_world / story_engine into every body beat
   actually move mistral/gemma, or does KILL 2 ALSO need a deterministic gate
   like KILL 1 (we already proved instruction-following alone fails for the
   body)? If a gate is needed, what does it check, and is "style register"
   even gate-able deterministically?

4. **KILL 3 climax position.** Confirm SPINE-DRIVEN (remove the force, don't
   mandate a move). Any hidden coupling that makes "allow either position" break
   the outro's "last char line = resolution" assumption or the ending-template
   target?

---

## 6. STANDING DISCIPLINE / INVARIANTS (a "fix" that breaks one is rejected)
- Behind the `story_scaffold` flag -> byte-identical when off (new fields default
  empty; no `meta.story_contract` when off).
- NO workflow-JSON change required for KILL 2/4/announcer (env/flag-gated) unless
  a new widget is genuinely needed -- if so it goes IN `otr_scifi_16gb_full.json`
  in the same change (positional widgets_values; append-only).
- The audio spine is FROZEN: `test_audio_byte_identical` stays green; mux-LAST.
- Full suite + Bug Bible per chunk (5 pre-existing 267a53e workflow-pin fails are
  not ours). Commit AND push per green chunk to v2.0-alpha.
- 100% local for the pipeline; determinism (seed-keyed); LOUD fallbacks; UTF-8
  no BOM; SFW. prod/main + tags GATED.
- The 3-test "is it baked in" check per feature: (1) it's CALLED not just
  defined; (2) a shipped output shows its telemetry fired; (3) delete-it test
  reverts behavior.

## 7. BUILD ORDER (proposed; the panel may reorder with rationale)
KILL 2 (StoryContract) -> announcer redesign + KILL 4 (alongside, so the open
draws the same style) -> re-soak -> KILL 3 (its own later build). Hand to a coder
window after convergence.
