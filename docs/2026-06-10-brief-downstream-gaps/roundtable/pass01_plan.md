# Brief-to-Downstream Gap Fix Plan -- v1 (post pass-01 roundtable, grounded)

The gap matrix (G1-G8) stands as in pass00 with one amendment: meta.visual_plan
is NOT orphaned (radio editor / casting / HUD genre consume it). The fix design
below supersedes pass00's F1-F5.

## F1. The prompt finisher (in `nodes/_otr_story_brief_helpers.py`)

```
ERA_TAIL_DEFAULT   = "timeless cinematic aesthetic"
STYLE_TAIL_DEFAULT = "cinematic, 35mm film look, subtle film grain, volumetric lighting"

def get_era_tail(meta) -> str
    # legacy _resolve_era_tail precedence, ported:
    # atmosphere_line -> visual_palette (top 3, joined) -> v1 lighting
    # (get_story_brief_lighting) -> ERA_TAIL_DEFAULT. Never raises, never "".

def finish_visual_prompt(meta, prompt, *, max_chars=None, style_tail=True) -> str
    # prompt + ", " + era_tail [+ ", " + STYLE_TAIL_DEFAULT if style_tail]
    # max_chars: word-boundary trim of the FINISHED string; a trailing
    # "no on-screen text" clause, when present in `prompt`, is preserved
    # (re-appended after the trim). No dedupe. No logging. Pure.
```

- NO logging inside the finisher: `log_story_brief_disposition` keeps its
  once-per-run contract -- each consuming NODE calls it once with its id:
  `ltx_scene_open` (run_real_episode), `shotlock_m4` (OTRShotLock.lock),
  `flux_portrait` (OTRMetaBriefImagePromptGen.generate).
- No style presets; no dedupe (3-model consensus cut).

## F2. Scene prompts in `render_driver.build_request_from_shot`

Order of precedence for a text-engine shot (`ltx_video` / `wan_i2v`):
1. The writer's M4 `creative.text_prompt` (finished at ShotLock, F3) -- wins.
2. `OTR_LTX_RADIO_PROMPT` env (announcer/music roles only) -- VERBATIM, no
   finishing; the log line must say "operator override" (today it wrongly says
   "composed from the episode brief").
3. Brief-composed: core = `get_story_brief_ltx(meta)` (else the
   setting-composed fallback), + the LTX scene clauses ("a vintage radio set
   glowing in the scene" for announcer/music roles, "moody dusk light",
   "slow cinematic camera drift", "no on-screen text"), finished via
   `finish_visual_prompt(meta, p, max_chars=240, style_tail=False)`.
   This branch covers ALL roles with no creative prompt (scene_broll,
   background_abstract included), not only announcer/music -- kills the
   "a 1940s radio studio" generic default for text engines.
- Keep the `group_id` -> role fallback parsing.
- Lipsync base (`_provide_lipsync_base`): prefer the request's existing
  (finished) `text_prompt` when non-default; `OTR_LSYNC_BASE_PROMPT` verbatim.

## F3. M4 + portraits

- `otr_shot_lock.py`: finish `text_prompt` BEFORE line ~510
  `prompt_hash = _content_hash(text_prompt)` (style_tail=True, no cap --
  HuMo/FLUX take long prompts). The batch-LLM instruction gains: "Do not
  include film-stock or lighting terms; they are appended later." JSON schema
  unchanged.
- `otr_meta_brief_image_prompt.derive_image_prompts`: finish AFTER the
  consistency + person guards, BEFORE `out[cid]` hash stamping
  (style_tail=True, no cap). Both LLM instructions gain the same note.

## F4. Doc rot

- `OTR_LedgerScriptWriter.py` ~3988: replace the stale "compose_shot_prompt
  appends era_tail + style_tail" comment with the real seam
  (`finish_visual_prompt` at ShotLock/image-prompt/render-driver).
- `_otr_story_brief_helpers.get_story_brief_music_mood` docstring: the claimed
  consumer `nodes/musicgen_theme.py` no longer exists; the live music lane is
  `_otr_music_prompt.py` via the brief-reader protocol. Comment-deprecate; keep.

## F5. Tests (CPU, with the suite)

1. `finish_visual_prompt`: era fallback when brief empty; v2 precedence
   (atmosphere_line wins); max_chars word-boundary trim preserving
   "no on-screen text"; style_tail toggle.
2. ShotLock: `prompt_hash` matches the FINISHED text_prompt.
3. Image prompts: stamped hash matches the finished prompt; person-guard
   fallback also gets finished; guards never re-run post-finish.
4. render_driver: env override verbatim + logged as override; brief-composed
   path under 240 chars; scene_broll-on-ltx no longer gets the generic default.
5. Disposition: one `[story_brief:<id>]` line per node run (caplog).

## Acceptance (after fixes, ONE 30w production render)

Logs show `[story_brief:ltx_scene_open]`, `[story_brief:shotlock_m4]`,
`[story_brief:flux_portrait]`; the scene-open prompt carries brief prose +
era tail within budget; portrait/M4 prompts end with the tails; suite + Bug
Bible green; audio byte-identical; operator eyeball gates the look.

## Invariants (unchanged from pass00)

Frozen audio / mux-LAST; fail-soft tails (never raise, never block); explicit
env overrides win verbatim; guards before finishing before hash; no new
widgets; UTF-8 no BOM; SFW.
