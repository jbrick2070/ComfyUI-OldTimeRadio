VERDICT: build-ready as-is? no. The plan cannot yet execute its same-ledger experiment protocol through the canonical workflow, and it lacks arm-scoped selection, durable request identity, and cross-beat continuity.

MUST-FIX BEFORE BUILD:

1. [Sections 7 and 9] The prescribed A/A-plus-arm comparison is not executable through the current canonical graph. Node 1 always re-executes, creates a new ledger, and rerolls cast state; the canonical runner exposes no accepted-ledger replay path, and workflows/otr_canonical.json contains no replay/load node (nodes/OTR_LedgerScriptWriter.py:2788-2793, 3200-3204, 3656-3673; scripts/otr_canonical_api_run.py:7-12, 78-121). Add a canonical replay mode that injects a copied accepted ledger after authorship, preserves its audio/still evidence, runs the real ShotLock-to-publish path, and writes a distinct result directory. Pinning writer widgets is insufficient.

2. [Section 7 prerequisite] Changing animatediff15_v3_haunted_video globally to required_inputs=("text_prompt", "init_image") is not an experiment arm; it changes every profile using the shipping engine and reverses the 8 GB provisioning exemption. still_plan is a static engine declaration read before rendering, while the provisioner classifies all AnimateDiff engines as no-still (nodes/_otr_video_engines/eng_ghost_signal.py:334-351; nodes/otr_meta_brief_image_prompt.py:781-785, 843-848; scripts/otr_provision.py:1538-1551). Create arm-specific, nonshipping haunted engine variants selectable through the existing OTR_VideoDirector/profile path, each with its own static still_plan and recipe receipt. Fold only the winner into the shipping engine.

3. [Sections 2, 7/E3, 7/E9, and 9] render_request_hash is currently only a comparison-seed basis: it hashes brief, cast, beat, and character, then directly derives the sampler seed. It excludes the prompt, still hash, adapter strength, denoise, and graph recipe (nodes/otr_shot_lock.py:1315-1317, 1475-1478; nodes/_otr_video_engines/render_driver.py:3748-3771). E9 therefore cannot use it as evidence of the request actually rendered. Preserve it under an honest comparison_seed_hash name, then add a full render_request_hash covering final prompts, source-still content hash, recipe id, adapter strength, denoise, and every enabled ADE arm parameter. OTR_VideoRenderBatch must be the sole writer of the actual-render trace; ShotLock may own only the planned trace (nodes/otr_shot_lock.py:3060-3072; nodes/production_ledger.py:712-731, 1592-1598).

4. [Sections 3, 6, and 7] The vision invokes shot grouping and promises an alternative to “eleven unrelated hallucinations,” but every proposed still remains per-beat and independently rendered. ContextRef stabilizes windows inside one clip, not adjacent beats; the current still target and shot planners explicitly assign one still and one render request per beat (nodes/otr_meta_brief_image_prompt.py:1282-1318; nodes/otr_shot_lock.py:2685-2717). Add one continuity arm that gives all beats in a shot or scene a shared environment anchor while retaining character-specific foreground identity. Define whether that anchor is a shared scene plate, previous terminal frame, or shot-keyed still seed.

5. [Sections 7 and 9] The operator is told which render is experimental, so the visual verdict is vulnerable to expectation bias. The repo’s own experiment rule requires an unlabeled comparison and accepts “no meaningful difference” (docs/PRODUCTION_SPRINT_LESSONS.md:739-761). Randomize the three published candidates, keep the mapping in a separate receipt, obtain the verdict before reveal, and score style fidelity, face continuity, setting continuity, and motion-to-speech fit separately before asking “overall better.”

SHOULD-FIX:

1. [Section 7/E4] The audio mechanism is incomplete. audio_motion_profiles stores one aggregate eight-field record per beat, not a frame-length curve; ADE_ValueScheduling outputs FLOATS, while the loader socket requires MULTIVAL (nodes/_otr_audio_motion.py:14-24, 82-92, 132-170; ComfyUI-AnimateDiff-Evolved/animatediff/nodes_scheduling.py:82-99; ComfyUI-AnimateDiff-Evolved/animatediff/nodes_multival.py:88-100). Specify a source-frame-aligned envelope and the required ADE_MultivalDynamicFloats conversion.

2. [Sections 7/E0 and 7/E11] Freezing one global adapter strength before style work conflicts with the later claim that non-photographic styles may need different strengths. The strength is presently one process-wide value (nodes/_otr_video_engines/eng_ghost_signal_official.py:103-116, 158-174). Treat E0 as baseline calibration only; do not declare a final global value until at least the house style, anime, and engraving have been checked.

3. [Sections 7/E7, 7/E8, and 7/E11] [ASSUMPTION] New ControlNet, CameraCtrl, and checkpoint branches have no stated model-ownership or release contract. The current engine explicitly tracks and detaches every base, LoRA, and ADE patcher before decode (nodes/_otr_video_engines/eng_ghost_signal.py:850-887, 979-1047). Require equivalent ownership and post-release VRAM evidence before any heavy branch enters the 14.5 GB gate.

OPTIONAL / NICE-TO-HAVE:

Add a contact sheet with prompt, still, first/middle/last frames, audio-envelope trace, and blinded candidate id. This would make failures attributable without replacing the operator’s visual verdict.

CUT THESE (scope / over-engineering):

1. [Section 7/E13] Cut style-aware exclusion from this campaign. It avoids styles the lane fails to obey instead of fixing the stated defect.

2. [Section 7/E14] Cut it as a separate arm. It duplicates E9 exactly.

3. [Sections 7/E8 and 7/E11] Cut CameraCtrl and per-style checkpoint/LoRA expansion from the first build. Both enlarge model inventory and recipe surface before the simpler still-fed and full-style-language hypotheses are resolved.

4. [Section 7/E6] Defer FreeInit. It targets generic flicker, not the campaign’s primary ledger, still, style, or cross-beat continuity failures.
