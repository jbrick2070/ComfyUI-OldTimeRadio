# R1 anchor review (Cowork Claude, code-grounded) -- arc / creative coherence

VERDICT: PROCEED. Build `ideo_word` before B6. The plan's arc is coherent and the
sequencing is defensible from the real code. No blocking arc-level defect.

## Sequencing (grounded)

- CONFIRMED: B6 gates the 3D flag; 3D lanes are PARKED (GO_FORWARD_PLAN.md
  section 8 lists "3D GPU lanes until S-3D-0 + operator green light"). A gate for
  a feature that cannot run is lower urgency than the operator-priority stills
  words lane. -> ideo_word first.
- CONFIRMED: registry keys are the engine NAME, not node_key
  (`_otr_image_engines/registry.py` binds `register = _IMAGE_REGISTRY.register`;
  EngineRegistry.register keys on `eng.name`). So `ideo` (name "ideo") and
  `ideo_word` (a distinct name) both pointing at node_key `cloud_ideogram_v4` do
  NOT collide as registry entries. Safe.
- CONFIRMED: `_engine_by_node_key()` in tests/test_cloud_partner_conformance.py
  maps node_key -> ONE engine (dict, last write wins). Once two engines share
  cloud_ideogram_v4 only one is conformance-checked. MUST-FIX: iterate ALL
  engines per node_key. (Already flagged in the plan #6; keep it a MUST-FIX.)

## Composer integration (grounded)

- CONFIRMED: compose_still_prompt + NO_TEXT_CLAUSE + finish_visual_prompt all live
  in nodes/otr_meta_brief_image_prompt.py; the no-text clause is appended
  per-kind (e.g. ~:745-746 scene, ~:831-832 plate). A new `kind=lyric_card` path
  that OMITS the clause (lyric_text) and one that KEEPS it (title_mood) is the
  right seam. CONFIRMED the mechanism exists; the exact insertion function is a
  verify-at-build (r2).
- CONFIRMED: episode title is stamped at ledger.meta.episode_title
  (OTR_LedgerScriptWriter :1385 "episode_title": ...). title_mood pulls from
  there. A missing title must fail LOUD (no silent generic card) -- directive.

## Arc-level MUST-FIX (fold into the plan)

1. Keep the conformance-map fix (per-node_key iterate ALL engines) as a MUST-FIX,
   same commit as ideo_word.
2. The prompt for lyric_text/title_mood must be built in the COMPOSER (upstream),
   not the adapter -- the adapter only receives request["prompt"]. So the mode
   selection needs the composer to know the selected engine (ideo_word) AND the
   role. Verify the composer has (engine_id, role, beat text) at the point it
   builds the prompt (r2 wiring question).

## SHOULD-FIX

- Two engine NAMES for the two modes (e.g. "ideo_word" for lyric_text video roles
  + a title_mood variant) vs one engine that branches on role: the doc prefers
  two small classes only IF dispatch gets conditional. Since the mode is
  role-derived and the PROMPT is built upstream, ONE engine "ideo_word" whose
  render path is identical (it just forwards request["prompt"]) may suffice --
  the mode divergence lives in the composer, not the adapter. Resolve in r2.

## Open (verify-at-build, not blocking)

- cloud_media_cache RequestCacheKey fit for an image request (r3).
- Which composer function/object-path adds kind -- exact call site (r2).
