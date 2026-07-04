# Claude anchor review -- r3 (wiring / integration) -- GROUNDED from the workflow links

VERDICT: the image_gen replay boundary is now fully specifiable. It is a 7-NODE baked subgraph (NOT
S-F's single-node prune). Exact graph grounded from `workflows/otr_scifi_16gb_full.json` links.

## EXACT REPLAY GRAPH (image_gen boundary)
LIVE nodes (kept in the pruned prompt): 91, 92, 84, 86, 93, 94, 85.
Live internal links (left intact): 91.patched_ledger->92 ; 91.image_done->92 ;
92.clip_manifest->84 AND ->94 ; 84.silent_video->86 ; 86.video->93 ; 94.scopes->93 ;
93.final_mp4->85.silent_video.

BAKED external literals (the cut edges -- bake each as a constant so its producer node is excluded):
- 91.script_json (<-90 ShotLock.out0), 91.image_policy_json (<-88 ImageDirector.out0),
  91.image_prompts_json (<-89 MetaBrief.out0), 91.gate_in (<-7 EpisodeAssembler.out3 audio_done),
  91.episode_id (<-90 ShotLock.out4)  -- from the NEW dispatcher capture.
- 92.master_audio_path (<-7.out1)  -- node-92 capture.
- 84.base_video_path (<-12 SignalLostVideo.out0)  -- the procgen base mp4 (bake the artifact).
- 93.procgen_mp4_path (<-12.out0)  -- SAME procgen base mp4.
- 94.audio (<-7.out0 episode_audio)  -- bake the artifact.
- 85.master_audio_path (<-7.out1) + 85.audio_done (<-7.out3)  -- bake the master + an opaque gate token.

## WIRING MUST-FIX
W1. THE BUNDLE BAKES 3 NEW ARTIFACTS beyond the stills/ledger/master: the node-12 PROCGEN BASE mp4,
    the node-7 EPISODE_AUDIO wav (for node-94 scopes), and the node-7 MASTER (for 85 + 92). The
    dispatcher capture (state/node_image_input.json) supplies 91's 5 inputs; a node-7/node-12 artifact
    capture supplies the rest. All Test-Path+hash preflighted (the S-F discipline).
W2. AUDIO BYTE-IDENTICAL IS FREE: bake node-7's master into 85.master_audio_path -> every leg muxes the
    SAME master file -> byte-identical by construction (assert the baked master hash once at bake;
    do NOT re-hash per leg -- r2 cut).
W3. EXECUTED-NODE ASSERTION = exactly {91,92,84,86,93,94,85} (+63 if the validator runs). Writer +
    audio + director nodes (1,7,12,62,87,88,89,90) ABSENT. Node-91 is NOT an OUTPUT_NODE but RUNS as
    92's dependency; verify it via the dispatcher report / image_done / stills-on-disk /
    meta.image_engines.by_role -- NOT history outputs.keys().
W4. PER-LEG ENGINE OVERRIDE (the swap knob): the IMAGE engine is selected inside the baked
    image_policy_json (88's output) -> patch that baked literal per image leg. The VIDEO engine rides
    the per-beat engine_id in 91's patched_ledger output -> either patch node-92's `engine` widget or
    the baked ledger's per-beat engine_id per video leg. CLEAR `ledger["images"]` + the dispatcher
    image cache so 91 re-mints (r2's clear step; the dispatcher cache key/storage is the one remaining
    grounding item -- read otr_image_gen_dispatcher.py before coding C5).
W5. node-12 procgen base is CONSTANT across legs (only the image+video engine varies) -> baking ONE
    procgen mp4 is correct + keeps legs apples-to-apples. The composite blends the per-beat clips over
    it exactly as production does.

## SEQUENCING
C1->C2->C3->C4 (clean-break + retirement) ship first and independently. C5 (combo soak) depends on the
NEW captures (dispatcher + node-7/12 artifacts), so the capture-seam commit lands with C5's offline
parts; the LIVE combo run is in the GPU batch after a clean reference episode writes the captures.

## REMAINING GROUNDING (before C5 code)
The dispatcher's image cache key + storage (dict on the ledger? a disk index? otr_image_gen_dispatcher.py
~478-517) -> the exact clear mechanism for W4.
