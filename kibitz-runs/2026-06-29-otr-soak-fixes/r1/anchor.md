# r1 ANCHOR REVIEW (Claude, code-grounded) -- arc / creative coherence

VERDICT: the sprint set is COMPLETE and correctly scoped; the arc is sound. Three arc-level
must-fixes below before it is wire-ready. Grounded against the real Windows files.

## MUST-FIX (arc)

1. NO-FALLBACKS rip-out is a LARGER, test-heavy refactor than S-E reads -- sequence it explicitly.
   CONFIRMED: fallback machinery is pervasive -- `nodes/_otr_shared/fallback.py` (the dep-free
   resolver), `eng_humo.py` `fallback_engine = "humo_1.7B"`, `render_driver.py`,
   `nodes/_otr_shared/retry_taxonomy.py`, plus >=6 tests that ASSERT fallback behavior
   (`test_video_fallback_chain_additive.py`, `test_video_retry_taxonomy.py`, `test_video_humo.py`,
   `test_ltx_audio_in_engine.py`, ...). Ripping the chains out flips every one of those tests from
   "asserts it falls back" to "asserts it hard-fails LOUD". S-E must own that test conversion in the
   same chunk (the C3 ripple), or the suite goes red. Add it to S-E acceptance.

2. DISAMBIGUATE retry vs fallback -- they are different code paths (`retry_taxonomy.py`). The operator
   wants NO cross-engine FALLBACK (a selected engine must not silently become a different engine).
   Same-engine RETRY (transient error -> try the SAME engine again) is a separate question. The plan
   must state which dies: kill cross-engine fallback; decide retry explicitly (recommend: keep a
   bounded same-engine retry, kill the cross-engine degrade). Otherwise "rip out all fallbacks" is
   ambiguous and a coder could gut the retry taxonomy too.

3. RECIPE-STAMP is narrower than written -- the IMAGE side is ALREADY durable.
   CONFIRMED: `otr_video_render_batch.py:31` -- "The per-role IMAGE engine is already durable in
   ledger['images']"; `otr_image_gen_dispatcher.py:625` writes `ledger["images"]`. So S-E's stamp work
   = the per-beat VIDEO `delivered_engine` + `recipe/quant`, NOT the image engine (already there). The
   real gap is (a) the VIDEO delivered-engine/recipe, and (b) durability through
   `production_ledger._merge_with_disk` (CONFIRMED exists, production_ledger.py:1192) -- the dispatcher
   comment at otr_image_gen_dispatcher.py:385 already notes `ledger['images']` is "dropped before the
   credits read", i.e. the drop is real. Re-scope S-E + S-A forensic accordingly.

## SHOULD-FIX (arc)

- S-F-first vs S-A-first tension (already an open question): if the baked fixture's reference episode
  itself shows the S-A underrun, the fixture encodes a defect as "baseline". Resolve: bake the fixture
  from a SHORT-beat episode (no underrun) OR land S-A first so the baseline is clean. Lean toward
  S-F-first but pick a clean reference episode.
- BUG-411 + the radio-still bookend (S-E) BOTH touch `otr_meta_brief_image_prompt.py` -- co-schedule so
  two windows don't edit the same prompt file (the workflow-JSON corruption-by-two-editors risk).
- `visualizer_rainbow` is correctly deferred (own roundtable). Keep it OUT of the correctness arc.

## GROUNDED CLAIM LABELS
- CONFIRMED: eng_humo cfg defaults deliberate (not a regression); 49-frame cap (eng_humo.py:61);
  fallback machinery pervasive; `_merge_with_disk` exists; image engine already durable in ledger;
  retire targets exist (AbstractFamily / StillMotionFamily / StationCardFamily in cheap_families.py;
  VisualizerEngine in eng_visualizer.py); EpisodeAssembler boundary exists (S-F seam real).
- VERIFY-AT-BUILD: exact composite HOLD-path location in `otr_silent_composite.py`; the precise
  `_merge_with_disk` lines that drop top-level keys; whether ltx_audio_in's 15.9 GB is loader-resident
  or decode-peak (S-B observability answers it).
