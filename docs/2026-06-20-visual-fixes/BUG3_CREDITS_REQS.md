# BUG 3 -- model/engine detail onto the on-screen rolling credits (reqs captured 2026-06-20)

Operator ask (last night): the full forensic detail that the EPISODE TREATMENT
already records should ALSO scroll on the on-screen rolling credits -- so a viewer
sees how the episode was made without opening the `_treatment.txt` sidecar.

## Source of truth for "what should appear"
The agreed forensic spec is what `_write_story_treatment` already emits
(`nodes/video_engine.py`, commits `e8f3094` "full forensic credits sheet" +
`f03db0a` "resolved OpenRouter concrete model + cost"). Full block list:

1. **Header** -- Title, Style, Produced timestamp.
2. **WRITER / LLM CONFIG** -- Creative slot (A) model, Technical slot (B) model,
   Slot routing (A<->B transitions), Calls by slot, Creativity, Temperature + top_p,
   Optimization profile, Seed source, Target words + Actual (char / announcer split),
   Characters count.
3. **Resolved (OpenRouter)** -- each `~latest` alias -> the CONCRETE model it
   resolved to server-side, with call count + USD cost, + total cost.
4. **NEWS SEED** -- the headline(s) the premise came from.
5. **STORY SPINE** -- Premise (script_brief), Key terms, Casting brief, Sign-off;
   Dramatic Question, A wants, B wants, Ending change, Costly-choice beat.
6. **CAST & VOICES** -- name -> [voice_engine] preset + description.
7. **RENDER ENGINES** -- Video per role (engine xN), Image per role (engine xN),
   engine histogram, video revision.
8. **SCENE ARC / FULL SCRIPT** -- flat dialogue + sfx list (per-line voice preset).
9. **PRODUCTION** -- Duration, Resolution @ fps, File, Size, VRAM peak.

## What the on-screen credits ALREADY show
`_parse_hud_data()` -> `_TelemetryHUDRenderer` (`nodes/video_engine.py` ~L1068,
drawn as the post-roll Telemetry HUD). Its returned dict today carries ONLY:
`title, style, produced, duration_s, resolution, news_seeds, cast[char/preset/desc],
scenes[transcript items], telemetry{peak, speed, model}`.

So on-screen the viewer already gets: title, style, news seed, cast & voices,
the transcript, and basic telemetry (peak VRAM / speed / one model id).

## THE GAP -- what BUG 3 must ADD to the credits (from the SAME meta the treatment reads)
- **WRITER / LLM CONFIG** block: creative_writing_model + technical_model (from
  `meta.gen_params_initial` / `meta`), slot_transitions, creativity, temperature/top_p,
  target vs actual words.
- **Resolved OpenRouter** block: `resolved_models_snapshot()` -> concrete model + cost.
- **STORY SPINE** block: `meta.news` (script_brief / key_terms / casting_brief /
  sign-off) + `meta.dramatic_state` (question / A wants / B wants / ending / costly beat).
- **RENDER ENGINES** block: video per role (`meta.render_engines.by_role`) + image per
  role (from `led.images.images` role+engine_id) + histogram.

## Wiring / constraints
- Enrich the `_parse_hud_data()` return dict with the blocks above + teach
  `_TelemetryHUDRenderer` to draw them (extra HUD panels/columns; keep the green CRT look).
- CONTENT-ONLY -- the data already lives in `meta`/`led`; this mirrors the treatment
  work, so it needs **no node/widget/wiring change -> NO workflow-JSON edit** (unless a
  new widget is genuinely added, in which case it goes IN otr_scifi_16gb_full.json
  same-commit per CLAUDE.md S0).
- Keep credits readable: paginate/scroll the new blocks; don't let them shove the
  transcript off-screen. Audio byte-identical; suite + Bug Bible green.
