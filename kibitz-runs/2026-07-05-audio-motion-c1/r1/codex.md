VERDICT: no. The plan is still a fork, and Option A’s stated graph arc contradicts the real workflow while both options leave durable ledger stamping undefined.

MUST-FIX BEFORE BUILD:
1. [Option A] The proposed arc “ShotLock -> [96] -> VideoRenderBatch” bypasses the real producer immediately upstream of node 92: workflow link 256 is 90->91, and link 260 is 91->92, so node 92 consumes OTR_ImageGenDispatcher.patched_ledger_json, not OTR_ShotLock directly (workflows/otr_scifi_16gb_full.json; nodes 90/91/92). Concrete fix: define Option A as 91 -> OTR_AudioMotionProfile -> 92, preserving node 91’s patched ledger and image_done gate, or explicitly justify a pre-image-ledger profile and add a separate link without replacing node 91’s ledger path.

2. [What C1 says / Option A / Option B] “ledger stamp” is the stated deliverable, but neither option states the durable ledger write authority. The shipped core only mutates an in-memory dict and says the caller owns durable save (nodes/_otr_audio_motion.py:229-239). OTR_VideoRenderBatch currently parses wire JSON, calls run_real_episode, and separately stamps only meta.render_engines via production_ledger.stamp_durable (nodes/otr_video_render_batch.py:67-84, 231-307). Concrete fix: specify exactly whether audio_motion_profiles are written through production_ledger.stamp_durable, save_ledger_safe on the active ledger path, or only the wire JSON; C1 is not complete unless the stamp survives into the production ledger on disk.

3. [Option B] The claim “reuses the existing slice” is conceptually incomplete. The slice is created inside build_request_from_shot while building each render request, but run_episode only returns ledger/clips/trace/vram_peak, not the slice paths (nodes/_otr_video_engines/render_driver.py:1334-1404, 1979-2085). Concrete fix: either make the request/audio_ref path observable in the returned trace/result before profiling, or admit Option B will call a resolver/slicer again and design the cache/key behavior around that.

4. [What C1 says / Recommendation] The plan never chooses the row universe for the profile. The core accepts arbitrary rows with id/beat_id/line_id plus timing (nodes/_otr_audio_motion.py:183-193), while the render path iterates ledger.video.shots (nodes/_otr_video_engines/render_driver.py:2024-2057). Concrete fix: declare whether C1 stamps one profile per video shot, per line, per beat, or per rendered clip, and define the id/timing fallback for rows with missing start_s/dur_s.

SHOULD-FIX:
1. [Recommendation] The default recommends Option B because “C2 has no consumer yet,” but C1’s own stated goal includes “schema field, producer node, IS_CHANGED/cache key” and “producer wiring.” That is a goal/method mismatch. Concrete fix: either rewrite C1 as “existing-node opportunistic stamp” or keep “producer node” and make Option A the plan.

2. [Option A] “ALWAYS runs” needs a workflow-level condition, not a claim. OTR_VideoRenderBatch is OUTPUT_NODE=True (nodes/otr_video_render_batch.py:91-100), but a new producer node would run only if it remains on a live dependency path in the submitted workflow. Concrete fix: state the exact dependency path and validation proving node 96 is reachable from an output in workflows/otr_scifi_16gb_full.json. [ASSUMPTION] This matters most for procgen-only modes if node 92 is skipped or returns no clips.

3. [Option B] “zero JSON change” is true only if the existing node’s interface is unchanged. If the implementation adds any input/widget/mode to node 92, the project’s workflow-source rule requires updating workflows/otr_scifi_16gb_full.json in the same change. Concrete fix: explicitly constrain Option B to no INPUT_TYPES change, or move it into the JSON-edit path.

OPTIONAL / NICE-TO-HAVE:
- Add a short “future C2 contract” note: how consumers should treat missing/failed rows, version amp-1, and ok/reason rows from nodes/_otr_audio_motion.py:38, 183-228.

CUT THESE (scope / over-engineering):
1. [Option A] Cut custom IS_CHANGED for the first build unless a concrete stale-cache failure is demonstrated. The ledger JSON string should already include audio.master_audio_sha256 if the upstream ledger is correct, and _slice_master_audio already keys slices on master hash/timing/version (nodes/_otr_video_engines/render_driver.py:241-273, 1384-1388). Keep this as a follow-up cache hardening item.

2. [Recommendation] Cut “operator graph eyeball” as a build-plan dependency. Keep validator + JSON round-trip + link/widget audit as acceptance gates; an eyeball gate is process, not architecture.