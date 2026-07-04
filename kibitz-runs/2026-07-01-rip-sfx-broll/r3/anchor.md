# r3 anchor review (Claude driver, code-grounded) -- BUILD_PLAN.md wiring pass

VERDICT: BUILD-READY with 3 MUST-FIX wiring corrections folded below. Every claim here
was read from the real Windows files this session (paths + line numbers cited).

## MUST-FIX (anchor)

1. CONFIRMED -- shot "strategy" stamp after pooling rip (otr_shot_lock.py:782).
   Today: `"strategy": {"mode": budget.get("clip_mode", "unique_per_beat")}`. Budget loses
   clip_mode, so the plan must stamp the constant `{"mode": "unique_per_beat"}` (schema-stable
   for downstream readers). Grounded consumer scan: render_driver reads shot.get("still_pool_key")
   at :1015/:1046/:1130 as `still_index.get(pool_key or _bid)` -- with the stamp gone it falls
   to the per-beat _bid naturally, but the dead read should be DELETED (root cause), not left
   as an inert `or` chain.

2. CONFIRMED -- ENGINE_FAMILY needs NO role edits (render_driver.py:70-87 is an
   engine_id->family map; no scene_broll/background_abstract rows exist). The contract line
   "render_driver ENGINE_FAMILY/_PROFILES" resolves to: _PROFILES only (:92-101, drop the
   scene_broll still_motion + background_abstract still_flat legs -> 4 legs).

3. CONFIRMED -- _otr_legacy_to_stage1_adapter.py:527 is DOCSTRING-only (grep: single hit).
   No body read of beat.sfx_cue. Comment-only touch, not a code path.

## CONFIRMED wiring facts the panel should not re-litigate

- Node 87 widgets_values grounded at 19 values, order exactly as BUILD_PLAN 2.1
  (mid-list drop idx 6,7 shifts fps..custom_models down 2; tail drop 17,18). All four
  removed widgets' converted-input entries have link:null -- links[] untouched.
- Node 3 (OTR_SceneSequencer) widgets_values = ["[]",0,999,"","bark",0,0];
  sfx_offset_ms is TAIL; sfx_audio_clips + sfx_offset_ms inputs both link:null.
- resolve_speaker_role + stamp_default_role have ZERO production callers (repo grep;
  tests only) -- converting them to raise cannot break a production path.
- The music_*/sfx `[SFX: {text}]` branch in assemble_script_text_from_ledger
  (production_ledger.py:1338-1339) is dead-by-construction for music rows post-S1
  (init stamps text="" without sfx_cue; empty text rows `continue` at :1322-1324).
  Making it a skip is behavior-identical for every ledger the new writer can produce.
- OTR_LedgerScriptWriter NON_VOICED branch: `token = "[SFX: ]"` would be appended to
  script_text_parts (:4855) for music rows, but slot-0 authority is
  assemble_script_text_from_ledger post-loop -- suppressing empty tokens is safe.
- wan_ti2v roles=ROLES (eng_wan_ti2v.py:103) auto-shrinks with the enum; the other five
  engine role-tuples need the explicit edits in BUILD_PLAN 5.
- widget_mapping.json is the ONLY config carrying scene_broll_visual /
  background_abstract_visual; 16gb_full/8gb_lite/cpu_floor set neither (grep-grounded).
- ALLOWED_SPEAKER_ROLES drop (ledger_freeze:91-98) makes the existing per-line invariant
  (:307-315) the LOUD old-ledger gate; G7 (:658-660 call, :667-750) + SFX_DUR_* (:63-64,
  :682-683) delete cleanly -- only test_per_cue_sfx_dur.py pins them (file deleted).

## SHOULD-FIX / verify-at-build (downgraded honestly)

- UNVERIFIABLE (not yet run): which of the ~20 video-test files hard-code 5-role fixtures.
  The suite loop resolves this; fix fixtures to the 3-role model, never re-add fallbacks.
- VERIFY-AT-BUILD: registry.default_engine_for_role over the shrunk role set -- every
  surviving role must resolve a non-empty default (announcer/music -> ltx_av_music,
  character -> humo per current default_roles declarations; StillMotionFamily
  default_roles=() removes only the dead scene_broll default).
- VERIFY-AT-BUILD: scripts/otr_video_soak.py + otr_coverage_sweep.py +
  run_otr_30word_smoke.py role enumerations (FIVE_ROLES rename fallout).
- VERIFY-AT-BUILD: test_capability_profiles / test_workflow_apply may pin
  widget_mapping.json keys or _VIDEO_DIRECTOR_WIDGETS membership.
