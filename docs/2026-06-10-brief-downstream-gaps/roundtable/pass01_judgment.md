# Pass 01 judgment (Claude, grounded against the code)

Panel: gpt-5.5 (resolved 20260423), gemini-3.1-pro-preview (20260219, truncated
after item 4 -- reasoning ate the budget), deepseek-v4-pro (20260423).
Spend: ~$0.03 metered + unpriced GPT/Gemini (est. < $1.50 total).

## CONFIRMED (folded into pass01_plan)
- **Disposition contract** (GPT#1, Gem#1): helpers emit
  `[story_brief:<consumer_id>]` and the docstring mandates ONCE per run.
  -> logging moves OUT of the finisher; each node logs once
  (`ltx_scene_open` in run_real_episode, `shotlock_m4` in lock,
  `flux_portrait` in generate). Acceptance greps `[story_brief:`.
- **Era-tail fallback** (GPT#2, DS#2): `get_story_brief_lighting` returns ""
  fail-soft -> the finisher needs `ERA_TAIL_DEFAULT = "timeless cinematic
  aesthetic"`. CONFIRMED in helpers + legacy.
- **Era-tail v2 precedence** (Gem#2): legacy `_resolve_era_tail` read
  `atmosphere_line` -> `visual_palette` (top 3) -> v1 lighting. CONFIRMED at
  legacy lines 167-203. The finisher ports this precedence, not just lighting.
- **LTX char budget** (GPT#3, Gem#3, DS#3): helper docstring pins LTX motion
  budget 220-240 chars (brief fragment 90). CONFIRMED -> `max_chars` parameter,
  word-boundary trim, style tail omitted for LTX when over budget.
- **Hash-before-finish defect** (GPT#6, DS#1): `otr_shot_lock.py:510`
  `prompt_hash = _content_hash(text_prompt)`; image prompts hash at acceptance.
  CONFIRMED -> finish BEFORE hash at both sites, after the guards.
- **Scene-open clauses must survive** (GPT#4): keep the LTX motion/render
  clauses (vintage radio set, slow drift, no on-screen text) around the brief
  core; brief replaces only the CORE.
- **Env override semantics** (GPT#5): `OTR_LTX_RADIO_PROMPT` verbatim (no
  finishing) but the node still disposition-logs; fix the warning text that
  currently claims "composed from the episode brief" on the override path.
- **Lipsync base bypass** (GPT#7): `_provide_lipsync_base` overwrites with a
  hardcoded prompt. CONFIRMED -> prefer the already-finished request prompt;
  `OTR_LSYNC_BASE_PROMPT` stays verbatim when set.
- **Generic fallback for non-announcer text-engine shots** (GPT#8): CONFIRMED
  (`build_request` default "a 1940s radio studio..."). The brief-core fallback
  extends to ANY no-creative shot on ltx_video/wan_i2v, all roles.
- **M4 LLM instruction note** (GPT-SF#3, DS-SF#3): one sentence "do not include
  film-stock or lighting terms; they are appended later" -- JSON schema
  unchanged (beat_id/expression/motion/camera).
- **G8 resolution** (GPT#9, DS-SF#4): `nodes/musicgen_theme.py` NO LONGER
  EXISTS; the live music lane is `_otr_music_prompt.py` via the brief-reader
  protocol (v2 music_mood_terms -> v1). CONFIRMED -> fix the stale helper
  docstring; comment-deprecate `get_story_brief_music_mood`; no deletion.

## REJECTED / MISREAD
- **"meta.visual_plan is orphaned"** (GPT#10, DS-SF#5): MISREAD. Live
  consumers: `_otr_radio_editor.py` (freeze-gate corpus, lines 541-555, 2019),
  `_otr_casting.py:372`, `video_engine.py:1307` (HUD genre). No fix.
- **Import-path claim** (GPT-SF#7): render_driver lives at
  `nodes/_otr_video_engines/`, not `nodes/video/`; correct relative import is
  `.._otr_story_brief_helpers` -- the substance (two different relative depths)
  stands, the cited path was wrong.

## CUT (3-model consensus)
- Style-preset-aware tails (legacy moved away from preset-derived visuals;
  single STYLE_TAIL_DEFAULT constant).
- The dedupe step in the finisher (fragile, unnecessary).

## Still-open / verify-at-build
- Tail duplication when the writer LLM ignores the "no film terms" note:
  acceptable (harmless repetition; dedupe deliberately cut).
- Gemini items past #4 unknown (truncated); its visible items all overlapped
  GPT/DeepSeek.
