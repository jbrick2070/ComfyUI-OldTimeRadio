# CLEANBREAK + COMBO-SOAK -- r3-CONVERGED PLAN (Claude-synthesized, grounded)

Panel r3 (Claude Code + Codex + Antigravity) all grounded vs the real workflow links. r3's decisive
correction: the engine selection is baked into the ledger by ShotLock (node 90), UPSTREAM of node 91 --
so an "image-gen-start" boundary at node 91 CANNOT vary engines. The correct baked boundary is
UPSTREAM OF THE DIRECTORS: bake the STORY + AUDIO, keep the director->shotlock->imagegen->video->
composite->upscale->mux chain LIVE, and vary the engine picks per combo.

## GROUNDED WORKFLOW GRAPH (from otr_scifi_16gb_full.json links)
62 LedgerFreezeCascade --script_json--> 12, 89, 90
7  EpisodeAssembler    --episode_audio--> 12, 94 ; --output_path(master)--> 92, 85 ; --audio_done--> 90,91,85
87 VideoDirector --video_policy--> 88, 90
88 ImageDirector --image_policy--> 89, 91
89 MetaBriefImagePromptGen --image_prompts--> 91
90 ShotLock --patched_ledger(w/ per-beat engine_id)--> 91 ; --episode_id--> 91
91 ImageGenDispatcher --patched_ledger--> 92 ; --image_done--> 92
92 VideoRenderBatch --clip_manifest--> 84, 94
84 SilentComposite <--base_video(procgen) 12 ; --silent_video--> 86
86 CaptionBurn --> 93
94 SceneAwareScopes <--episode_audio 7 ; clip_manifest 92 --> 93
93 PostUpscaleProcgenBlend <--source 86, procgen 12, scopes 94 --upscaled--> 85
85 MasterAudioMux <--silent 93, master 7, audio_done 7  (OUTPUT -> obs final)

## THE MODIFIED "COMBO SOAK" WORKFLOW (what the operator asked for)
Take `otr_scifi_16gb_full.json` and REPLACE the upstream story+audio generators with BAKED inputs,
keep everything from the directors down LIVE:
- BAKE (from a CLEAN reference run): node-62 `script_json` (the FROZEN story ledger) as a literal +
  the node-7 audio artifacts (master wav + episode_audio) loaded from disk. This removes node 1
  (writer) + the whole TTS/music audio front-end -> NO gemma, NO TTS per leg (fast, deterministic,
  ONE story for every combo = apples-to-apples).
- KEEP LIVE: 87 VideoDirector, 88 ImageDirector, 89 MetaBrief, 90 ShotLock, 91 ImageGenDispatcher,
  92 VideoRenderBatch, 84 SilentComposite, 86 CaptionBurn, 94 SceneAwareScopes, 93
  PostUpscaleProcgenBlend (the UPSCALE), 85 MasterAudioMux (-> obs).
- PER-COMBO KNOB: patch node-87 (VideoDirector) video-engine dropdowns + node-88 (ImageDirector)
  image-engine selection per leg; the whole director->shotlock->prompts->image->video re-runs on the
  SAME baked story+audio, so engine swaps are honored atomically (r3 fix: do NOT patch only node-91 --
  ShotLock/prompts must re-run from the new policy). CLEAR `ledger["images"]["images"]` +
  `["cache_index"]` (otr_image_gen_dispatcher.py:379-381,625-631) so stills re-mint per leg.
- AUDIO BYTE-IDENTICAL: node-85 muxes the SAME baked master every leg -> identical by construction;
  assert the baked master hash once.
- UPSCALE -> OBS: node-93 PostUpscaleProcgenBlend is the upscale/blend; node-85 publishes the final to
  `output\otr\obs`. Both already in the chain.

## CAPTURE SEAMS NEEDED (for the bake)
- node-92 already writes `state/node_episode_input.json` (ledger + master path) -- S-F.
- ADD: node-7/node-62 artifact capture (the frozen script_json + master wav + episode_audio paths) and
  a dispatcher forensic `node_image_report.json` (image_done + made/reused + image histogram +
  meta.image_engines.by_role) so the combo soak can PROVE the image engine ran (node-91 is not an
  OUTPUT_NODE; `outputs.keys()` can't see it -- Codex r3).

## ACCEPTANCE (per combo leg)
Executed graph carries only the LIVE node set (writer + audio-gen ABSENT); image leg: dispatcher
report shows the selected engine + made>0/reused==0 + stills on disk; video leg: node-92 histogram +
the E5 recipe receipt name the engine; NO silent floor (fallbacks gone -> LOUD named hard-fail);
baked master hash unchanged; NVML <= 14.5 GB; deterministic 2-pass (same histogram + clip metadata) on
>=1 leg. Matrix = additive ~15-20 (the 15 video + 5 image legs the runner already enumerates).

## CLEAN-BREAK RIP-OUT (Part A, r2-final, unchanged) -- ships independently of the soak
KEEP FamilyInputGap/_assert_family_inputs_satisfiable (production guard) + engine_family +
classify_failure + RenderError. REMOVE make_fallback_of/FLOOR_NAMES/UNIVERSAL_FLOOR/SYNTH_FALLBACKS +
OomSignal/force_oom + the whole soak + fallback.py + every fallback_engine attr + retry_taxonomy
fallback bits; migrate/delete the grounded test + script + profile + fixture references in-commit;
retire abstract/station_card/still_motion AFTER reassigning their scene_broll/background_abstract/
announcer_visual defaults to still_pan/still_flat (+ fix 8gb_lite/cpu_floor profiles). allow_auto_fallback
clean-delete + JSON widget-vector rebaseline. ENGINE_FAMILY dict pruned with the retirement.

## CONVERGENCE
r1->r2->r3 surfaced + resolved every structural, coding, and wiring defect (prune-boundary corrected to
the director boundary; FamilyInputGap kept; defaults reassigned; capture seams specified; the test/
profile/script migration enumerated). r4 (residual-defect convergence) is a confirm pass -- no new
must-fix is expected; the plan is BUILD-READY. Build order: Part A rip-out chunks C1-C4 (operator look
when awake), then the combo-soak modified workflow + runner (C5) on the baked reference.
