# r3 ANCHOR REVIEW (Claude, code-grounded) -- wiring / integration / sequencing

VERDICT: wire-ready once the node-87 widget migration is specified exactly. Grounded against the live
`otr_scifi_16gb_full.json` (parsed node 87/90/91/92).

## CONFIRMED WIRING FACTS (parsed from the real JSON)
- node 87 `OTR_VideoDirector` widgets_values (widget index -> input, after the `gate_in` LINK input):
  wv0 announcer_video=`visualizer`, wv1 music_video=`visualizer`, wv2 other_beats_video=`visualizer`,
  wv3 announcer_image=`flux_gen1`, wv4 music_image=`flux_gen1`, wv5 other_beats_image=`flux_gen1`,
  wv6 other_beats_clip_mode, wv7 other_beats_n=4, wv8 fps=25, wv9 canvas_w=832, wv10 canvas_h=480,
  wv11 seed_mode, wv12 request_seed=42, **wv13 allow_auto_fallback=False**, wv14 episode_duration_target,
  wv15 custom_models_json, wv16 character_video=`humo_14B_169`, wv17 scene_broll, wv18 background_abstract.
- node 90 `OTR_ShotLock` owns `audio_done`; node 91 `OTR_ImageGenDispatcher` = script_json/
  image_policy_json/image_prompts_json/gate_in/episode_id; node 92 `OTR_VideoRenderBatch` =
  patched_ledger_json/master_audio_path/image_done/... (confirms the r2 node-IO correction exactly).

## MUST-FIX (wiring)
1. allow_auto_fallback removal = a PRECISE positional rewrite. Deleting input/widget 13 drops wv13 and
   shifts wv14..wv18 (episode_duration_target, custom_models_json, character_video_model, scene_broll,
   background_abstract) up one -> silent value corruption unless node-87 `widgets_values` is rewritten in
   the SAME commit + `OTR_WorkflowValidator` + widget audit. RECOMMEND deprecate-in-place (keep the slot,
   force-ignore the value, relabel "(deprecated)") to AVOID the shift entirely; only delete if a clean
   re-baseline is wanted. Decide explicitly.
2. Engine retirement is LOW JSON risk (good news): the current saved node-87 values are
   visualizer/flux_gen1/humo_14B_169 + defaults -- NONE is `still_motion`/`station_card`/`abstract`, so
   unregistering won't orphan a saved widget value. The dropdown OPTION lists are registry-driven and
   regenerate on unregister. ACTION: just verify no capability_profile / role-default still NAMES a
   retired engine before removing it; no node-87 value rewrite needed for the retirements.
3. S-F injection wires to the CONFIRMED sockets: feed node 90 `audio_done`, node 91 `script_json`/
   `image_prompts_json`/`image_policy_json`, node 92 `patched_ledger_json`/`master_audio_path` from the
   baked fixture, and SEVER the upstream writer(node 1)+audio producers so they are not dependencies.
   Keep the `gate_in` chain intact (it sequences 87->90->91). Do this in the API-built /prompt graph --
   NO production-JSON node addition.

## SHOULD-FIX
- recipe-stamp rides node 92's `meta.render_engines` payload (`otr_video_render_batch.py`) -- no JSON
  change; just extend the payload + the engine-return plumbing (S-B/S-E prereq).
- Only ONE widgets_values rewrite is implicated (the allow_auto_fallback decision). Labels are
  display-only (no new widget). Keep the JSON migration surgical: one node, one validated round-trip.

## GROUNDED LABELS
- CONFIRMED: the entire node 87/90/91/92 wiring above (parsed live).
- VERIFY-AT-BUILD: whether any `capability_profiles`/role default references a retired engine; the exact
  upstream link ids feeding node 90/92 to sever for S-F.
