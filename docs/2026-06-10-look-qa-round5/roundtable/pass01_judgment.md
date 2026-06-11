# Pass 01 judgment -- look-QA round 5 (3-model panel, ~$0.03)

Panel: gpt-5.5-20260423 (no), gemini-3.1-pro-preview-20260219 (yes-with-fixes),
deepseek-v4-pro-20260423 (no). Grounded against render_driver.py, eng_ltx_video.py,
eng_humo.py, otr_shot_lock.py, _otr_story_brief_helpers.py, the legacy plan file,
otr_silent_composite.py (post-hoc), and tonight's LIVE ledger probes.

## ACCEPTED (grounded CONFIRMED)

1. **No beat->scene mapping exists** (GPT-2, Gem-1, DS-2) -- STRONGER than the panel
   knew: probe shows `meta.visual_plan.scenes == []` and `led.scenes == []` in
   tonight's ledger. The legacy PASS-2 scene derivation died with CW-1 and was NOT
   restored by c51526b. R5-2 reshaped: deterministic per-beat composition from the
   FROZEN line's own signals (`beat_intent`, `arc_phase`, role) -- both fields
   confirmed present on lines (probe) -- no LLM, no scene table.
2. **Frame cap placement** (Gem-2 exact): `min(length, OTR_LTX_MAX_FRAMES=121)`
   between the `max()` (eng_ltx_video.py L279-280) and the 8n+1 snap (L281).
   Default 121 (GPT S-2); LOUD log `requested->capped->snapped` (GPT S-1/S-3);
   gate asserts final graph length <=121 and %8==1.
3. **M4 person anchor** (GPT-4, Gem-4, DS-6): the deterministic builder
   (`otr_shot_lock.py` L486-494) prepends a cast-anchored visible-subject clause;
   `_prompt_is_consistent` (L339) extended to REQUIRE person-anchor framing for
   CHARACTER_BEARING_ROLES (object-only prompts rejected -> anchored deterministic
   template fallback); the batch-LLM instruction gains the named-character-visible
   requirement (Gem optional). Prompt-text level only (GPT cut-3 honored).
4. **Writer-side self-vocative re-attribution binds the re-render** -- each accept30
   run writes a FRESH episode (DS-3's frozen-episode premise is a MISREAD of the
   harness; nothing re-renders a frozen episode). Deterministic rule pre-freeze:
   text starts with the SPEAKER's own name as a vocative AND the scene has exactly
   one other character -> re-attribute to the interlocutor, LOUD log; ambiguous ->
   LOUD + keep. No LLM repair (GPT cut-2, DS-7 agree). ShotLock backstop warning
   (Gem S-1).
5. **Announcer id join** (Gem-3 + Gem S-2): resolve the announcer's char_id from
   the cast table by NAME match (probe: cast rows have NO role field), never
   hardcode c01; LOUD warning in `build_request_from_shot` when a talking-head
   shot's char_id misses the portrait index.
6. **Finisher trim hazard** (GPT S-6): bright-radio clause goes FIRST among the
   clauses; test asserts the capped string keeps the bright tokens.
7. **Precise b000 gate** (GPT S-8): stddev of per-second YAVG over the b000 window
   of the SILENT composite > 2.0 (tonight's mud: ~0.2) + the cap log line present.
8. **Per-shot prompt manifest line** (GPT optional): one INFO line per text-engine
   shot (`source=m4|env|brief`, chars, sha8) + an LTX prompt-DIVERSITY gate
   (b000/b001/b005 sha8 not all equal) so "code ran but look identical" is caught
   pre-eyeball.

## REJECTED (with reason)

- **Composite can't fill (DS-1 hard blocker, GPT-3 assumption)** -- MISREAD:
  `otr_silent_composite.py` already normalizes every clip to its window with
  `tpad=stop_mode=clone` ("truncates a long source, holds the last frame for a
  short one", L264-279, L305). A capped 121f clip in a 238f window hold-fills with
  EXISTING code. Plan cites it; no composite change. The Ken-Burns-zoom nicety is
  dropped (static hold of a real scene beats mud; zoom = new code, new risk).
- **Split long beats into chained LTX clips** (DS-4, GPT cut-1, Gem cut-1): cut --
  render cost + new seams; hold-fill suffices for the open.
- **Video-only char_id override on shots** (GPT-1/5): unnecessary for a
  fresh-episode acceptance gate (see ACCEPTED-4); revisit only if a same-episode
  re-render mode ever exists. The "don't mutate frozen line rows" warning is
  honored everywhere regardless.
- **Weaken the acceptance gate** (DS-3a): not needed once the writer-side fix
  binds fresh episodes.
- **scenes[0] fallback** (Gem-1's concrete suggestion): impossible -- scenes is
  EMPTY tonight; superseded by ACCEPTED-1.

## VERIFY-AT-BUILD

- HuMo text-dominance: will the anchored M4 prompt actually re-center the face?
  Empirical; the re-render is the test (frame-grab b002-equivalent mid-beat).
- `beat_intent` / `arc_phase` population on ALL lines in fresh episodes: guard
  with safe fallbacks (absent -> role-based clause only).
- Cap interaction with `_LTX_MIN_FRAMES` when OTR_LTX_MAX_FRAMES is set < MIN
  (clamp defensively, LOUD on invalid env -- GPT S-2).

## Convergence

Material new items: YES (cap placement, M4 contract precision, scenes-empty pivot,
fresh-episode attribution scope). -> pass02 fans out the UPDATED plan.
