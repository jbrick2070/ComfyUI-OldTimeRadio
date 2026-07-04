# R1 synthesis + judgment (Cowork Claude, sole judge) -- Sprint B remainder

## Panel convergence (codex + antigravity, both grounded by me)

Both agents: sequencing OK (ideo_word before B6), but ideo_word is NOT build-ready
until registry identity + role-vs-kind routing + composer plumbing are pinned.
Strong convergence -- no conflicts to resolve.

## Grounded verdicts on panel claims

- CONFIRMED (codex#2/antigravity#2, load-bearing): the image dispatcher resolves
  engine PURELY BY ROLE -> slot (`otr_image_gen_dispatcher.py:151-173`
  resolve_engine_for_role; announcer_visual/music_visual/other_beats). There is
  NO per-kind engine selection. Selecting `ideo_word` for a slot routes EVERY
  object in that role (portraits, scene stills, cards) through it -- a hijack.
- CONFIRMED (antigravity#2): the prompt composer builds objects with
  `compose_still_prompt(meta, kind=tgt["kind"], role=tgt["role"], ...)`
  (`otr_meta_brief_image_prompt.py:1286-1308`) and NEVER references image_models
  -- it cannot know `ideo_word` is selected, so it cannot emit kind=lyric_card.
- CONFIRMED (codex#1/antigravity#1): registry keys by engine NAME
  (engine_registry_base). Two names = two dropdown entries; pick ONE public
  `ideo_word` that branches internally.
- CONFIRMED (codex#4/antigravity#3): episode title = `meta["episode_title"]`
  (writer stamps it; video title resolution reads it first). title_mood reads it,
  fail LOUD if empty.
- CONFIRMED (codex SHOULD#1/antigravity SHOULD#2): image-model widgets live on
  `OTR_VideoDirector` (node 87, `otr_video_director.py:221-229`), combo built
  dynamically from `_ireg.all_engine_names()`; `OTR_ImageDirector` has NO
  image-model widgets. My plan mis-named it. Registering ideo_word auto-appears;
  defaults stay flux_gen1 (no JSON change) -- but ADD an assertion the id is in
  the combo.
- CONFIRMED (both): conformance `_engine_by_node_key()` last-wins overwrite; fix
  to node_key -> list, assert ALL engines.
- ACCEPTED CUT (both): drop cloud_media_cache wiring from v1 -- lyric cards are
  per-beat unique (near-zero cross-episode hit rate); the dispatcher's in-run
  content-hash already prevents same-run regen. Global billing cache can follow.

## HARDENED PLAN (forward)

DECISION: build ideo_word before B6 (panel + anchor agree; B6 gates PARKED 3D).

ideo_word build (revised):
1. ONE registered image engine `ideo_word` (name "ideo_word", node_key
   cloud_ideogram_v4), EMPTY default_roles. Internally branches lyric_text vs
   title_mood by the object's `kind`/`role` in the request (it forwards
   request["prompt"] to Ideogram either way -- the MODE divergence is in the
   PROMPT, built upstream). CAPABILITIES row + guarded __init__ import same change.
2. Composer plumbing (the real work): pass the resolved `image_models` into the
   object-building path in `OTRMetaBriefImagePromptGen.generate()`; when the
   role's selected engine == "ideo_word", emit `kind=lyric_card` for that role's
   BEAT-CARD objects. Add the `kind=lyric_card` branch in compose_still_prompt:
   - lyric_text (character/announcer roles): prompt = deterministic first-clause/
     8-word excerpt of the beat line (pure helper + tests: quotes/ellipsis/
     em-dash/stage-direction/empty), quoted-text shape, BYPASSES NO_TEXT_CLAUSE.
   - title_mood (music_visual): wordless old-radio card from meta["episode_title"]
     + Meta brief/era tail (finish_visual_prompt), KEEPS NO_TEXT_CLAUSE. Empty
     title -> raise (fail LOUD).
   Portraits/scene stills in that role are UNAFFECTED only if the routing is
   scoped to the beat-card objects -- VERIFY the object set per role so ideo_word
   does not hijack portraits (open r2/r3 item: is the per-role object set ONLY
   cards, or mixed? if mixed, scope lyric_card to the beat objects and leave the
   others on the role's engine, OR document that ideo_word is a whole-role
   typographic treatment by design).
3. Cost: reuse the shipped speed->price map (_ideogram_est_usd, TURBO default).
4. Conformance: node_key -> list; assert all engines per node_key.
5. Workflow JSON: no new widget (dynamic combo); ADD a test asserting "ideo_word"
   in the OTR_VideoDirector image combo; defaults stay flux_gen1.
6. Family invariant: lyric_text per-beat unique, NEVER pooled.
7. CUT cloud_media_cache from v1.

## Residual open item for the next round (r2/r3)

The load-bearing unresolved question: within a role routed to ideo_word, is the
per-role object set ONLY beat cards, or does it also contain portraits/scene
stills that would be wrongly typographic'd? This determines whether lyric_card
routing is per-object (scoped) or per-role (whole treatment). Ground the object
set that generate() produces per role before coding the composer branch.
