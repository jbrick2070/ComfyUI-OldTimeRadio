# Brief-to-Downstream Gap Audit + Fix Plan (2026-06-10)

## Why this audit exists

Operator look-QA on post-refactor episodes found a PATTERN: rich upstream
data the writer stamps (the Meta brief, character descriptions) silently
stopped reaching downstream prompt composition after the video-platform
refactor (CW-1 teardown commit `e74a3ce` deleted `nodes/otr_video_plan.py`,
the legacy brain that consumed the brief). Episodes from 2026-06-01 looked
scenic and period-correct; post-refactor episodes render generic. The
operator's directive: find ALL such gaps end-to-end BEFORE the next test
render, instead of discovering them one eyeball at a time.

Already found and FIXED today (commits 3f55ef9..dae597a, unpushed):
- cast `character_description` never read by portrait/M4 appearance chains
  -> whole cast shared ONE portrait;
- the writer-LLM seam called a `make_generate_fn` signature that never
  existed -> EVERY live run silently used bare template prompts since the
  refactor (this is the single biggest reason the new path looks generic);
- a hardcoded "radio station studio" prompt override replaced scene prompts
  on every announcer/music beat (today's regression, already reworked to a
  brief-composed prompt);
- portrait prompts could depict no person (microphone-no-face live catch)
  -> person guard;
- stage directions + self-vocatives leaking into TTS/captions -> pre-freeze
  scrubs;
- LTX/Wan inheriting the HuMo portrait canvas -> full-frame landscape.

## The newly confirmed gaps (grounded, not yet fixed)

**G1 (HIGH) - `get_story_brief_ltx()` orphaned.** `nodes/_otr_story_brief_helpers.py`
exposes a purpose-built SHORT brief string for LTX prompts
(`get_story_brief_ltx(meta, max_chars=90)`). `git grep` shows ZERO callers
outside the helpers module itself. The legacy video plan used the brief as
the scene-prompt core; the new path never does. Today's hotfix in
`render_driver.build_request_from_shot` composes a scene prompt from raw
`story_brief_terms.setting` tokens -- it should lead with
`get_story_brief_ltx()` (the curated short prompt) instead.

**G2 (HIGH) - era tail orphaned.** Legacy composed every visual prompt as
`<subject> + era_tail + ", " + style_tail` where
`era_tail = get_story_brief_lighting(meta)` (brief-derived lighting/period
prose; fallback "timeless cinematic aesthetic"). ZERO callers now. No new-path
prompt (portraits, M4 character beats, scene opens) carries the brief's
lighting/period prose.

**G3 (HIGH) - style tail deleted.** Legacy `_DEFAULT_STYLE_TAIL =
"cinematic, 35mm film look, subtle film grain, volumetric lighting"` (plus
style-preset variants) died with `otr_video_plan.py`. Nothing appends a film
aesthetic tail to any prompt now.

**G4 (MED) - the disposition audit died.** `log_story_brief_disposition(meta,
consumer_id, log)` was the LOUD per-consumer line proving the brief reached
each prompt site. ZERO callers -> all of the above failed SILENTLY. Restoring
the audit line at every consumer is what prevents the next regression of this
family.

**G5 (LOW) - stale promise in the writer.** `OTR_LedgerScriptWriter.py`
(~line 3988) still documents "the downstream FLUX composer
(compose_shot_prompt) appends era_tail + style_tail at render time".
`compose_shot_prompt` no longer exists. Doc rot that misleads maintainers.

**G6 (MED) - M4 character-beat template has no tails.** ShotLock's
deterministic per-beat template is `{appearance}, {setting}, {beat_text}` and
the LLM refinement instruction does not request period/film styling. Character
HuMo beats render without the brief's aesthetic.

**G7 (MED) - portrait prompts have no tails.** `otr_meta_brief_image_prompt`
composes `{appearance}, {setting} setting, {STYLE_ANCHOR}`; STYLE_ANCHOR is a
generic in-character-portrait anchor, not the brief's era/style prose.

**G8 (CHECK) - music mood.** `get_story_brief_music_mood()` has zero callers;
the audio lane uses `nodes/_otr_music_prompt.py` (its own Meta-brief protocol,
consumed by eng_musicgen + stable_audio_theme). HYPOTHESIS: audio is fine via
its own reader and `get_story_brief_music_mood` is a dead duplicate -- verify,
then either wire or delete-with-note.

## Fix design (one coherent change, not seven patches)

**F1. One shared prompt-finishing seam.** Add `finish_visual_prompt(meta,
prompt, consumer_id, log)` to `nodes/_otr_story_brief_helpers.py` (zero-dep,
already the brief authority): appends `era_tail` (get_story_brief_lighting)
+ `style_tail` (the legacy default, style-preset aware if cheap), dedupes
fragments already present, and calls `log_story_brief_disposition` so every
consumer logs the brief disposition LOUDLY. All three composition sites call
it; no drifting copies.

**F2. Scene opens lead with the brief.** In
`render_driver.build_request_from_shot`, the announcer/music LTX prompt
becomes: `get_story_brief_ltx(meta)` (when non-empty) as the core, else the
current setting-composed fallback; then F1 finishing. `OTR_LTX_RADIO_PROMPT`
stays an explicit operator override (skips composition, still F1-finished?
-- NO: an explicit override is used verbatim; document that).

**F3. M4 + portraits get finished.** ShotLock's template/LLM output and
`derive_image_prompts` accepted prompts (template AND llm, after the person
guard) run through F1. The LLM instructions mention the era/style tail will
be appended so the model does not duplicate it.

**F4. Doc rot.** Fix the stale writer comment (G5) to name the REAL seam.

**F5. G8 verification.** Confirm `_otr_music_prompt` reads the brief on the
live path (one grep + one log line in the next render); if
`get_story_brief_music_mood` is a dead duplicate, mark it deprecated in a
comment (do not delete in this sprint).

**Acceptance for the fix sprint:** suite + Bug Bible green; a single 30w
production render whose logs show (a) `[story_brief] consumer=...` disposition
lines for scene-open, M4, and portrait sites, (b) scene-open prompt containing
brief prose + tails, (c) portrait prompts ending with the tails; operator
eyeball confirms the look. Frozen audio untouched; byte-identical gate green.

## Invariants the fixes must not break

- V-1 frozen master audio; mux-LAST; `test_audio_byte_identical` green.
- Fail-closed philosophy: a missing/empty brief NEVER blocks a render --
  tails degrade to defaults, never raise.
- The operator's explicit env overrides always win.
- Person guard + consistency gate ordering in `derive_image_prompts` stays
  (guards run BEFORE finishing; finishing must not re-trigger guards).
- No new widgets; no workflow JSON surgery; converter-valid.
- UTF-8 no BOM; SFW; suite 3804+ green per chunk.

## Question for the panel

You are reviewing this audit for COMPLETENESS and CORRECTNESS OF THE FIX
DESIGN. Specifically:
1. What OTHER writer-stamped surfaces (meta fields, ledger sections) does the
   grounding code show being produced but never consumed post-refactor?
2. Holes in the fix design: double-append risks, tail duplication with the
   LLM instruction, env-override semantics, fail-open vs fail-closed.
3. Is the one-shared-seam approach (F1) right, or does any consumer need a
   different finishing policy (e.g., LTX 90-char cap vs FLUX long prompts)?
4. Anything in the grounding that contradicts the gap claims above.
