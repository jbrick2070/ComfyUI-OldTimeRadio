VERDICT: no. D2′ has no implementable data path to the sampler inputs, and replay state cannot reach the proposed node-7/node-4 branches.

MUST-FIX BEFORE BUILD:

1. [D2′] Node 92 does not create or receive the request objects it supposedly records. It calls run_real_episode(ledger, ...) once; render_driver constructs requests internally and returns only ledger, clips, its existing reduced trace, peak VRAM, and audio-motion rows (nodes/otr_video_render_batch.py:563-564; nodes/_otr_video_engines/render_driver.py:4764-4792, 4870-4873). Multi-segment requests are also created inside render_beat_coverage (render_driver.py:4192-4196). Concrete fix: edit render_driver despite D2′, emit one normalized ACTUAL receipt for every rendered segment, and return those receipts to node 92.

2. [§2.1, D2′, §5.4] actual_request_sha has no defined schema or authoritative construction point. For LTX 2.5, _build_render_request contains only init_image, text_prompt, fps, frame count, and seed, while sampler, CFG, anchor, quantization, and other effective settings are supplied later from recipe constants and graph construction (nodes/_otr_video_engines/eng_ltx25.py:1348-1393, 1720-1739). Concrete fix: define a versioned canonical receipt at the adapter’s final graph/inference boundary: common envelope plus engine-specific sampler_inputs; explicit nullable fields; full positive/negative text; seed; sampler/scheduler/steps/CFG/denoise; frame/canvas values; adapter and resolved strength; context/injection values; still content hash; model digests; recipe and implementation version. Hash deterministic canonical JSON with full SHA-256. Node 92 must persist this returned receipt, not reconstruct it.

3. [D3′] EpisodeAssembler and AudioEnhance cannot inspect meta.replay_from. Node 7 accepts AUDIO, title, themes, cue manifest, and video policy only; node 4 accepts only AUDIO and widgets (nodes/scene_sequencer.py:1151-1232; nodes/audio_enhance.py:294-348). Concrete fix: append a forceInput replay descriptor/script_json input to node 7 and wire it in workflows/otr_canonical.json. Cut the node-4 replay branch: let node 3 emit a tiny valid CPU AUDIO placeholder, allow node 4 to process it, and have node 7 ignore it while copying the frozen master byte-for-byte. If node 4 must bypass, it also needs an appended, wired replay input.

4. [D3′] “Populate led.data from the bundle” is not a valid replay-clone transaction. new_ledger binds the new workspace, but wholesale assignment can restore the source episode_id, absolute paths, terminal output pointers, and a publication receipt belonging to the source episode. The code already has component-aware path rebasing and receipt rebasing that this plan bypasses (nodes/production_ledger.py:371-441, 473-488, 736-772, 1508-1524); the publisher rejects a receipt for the wrong episode (nodes/otr_master_audio_mux.py:808-848). Concrete fix: add one validated clone/import operation that deep-copies the source ledger, sets the new root episode_id, preserves source identity separately, rebases every episode-local path after assets are materialized, rebases or re-evaluates publication eligibility, clears source terminal/obs paths, resets run-volatile telemetry, and then saves atomically.

5. [D3′] “Pass-through stub on 80” and “CastLock forced preserve_ledger” specify different control flows. The current CastLock path increments cast_lock_revision and reassigns Bark voices even before its policy branch (nodes/cast_lock.py:331-365), so merely forcing preserve_ledger does not preserve an unchanged ledger. Concrete fix: define a replay return before the freeze gate, revision increment, voice assignment, and model resolution. Return the original ledger, original revision, an explicit replay report, and a non-empty done token.

6. [D2′, D3′] A cloned source ledger already contains its prior meta.render_trace, but the plan neither clears it nor assigns run identity. Replacing versus appending is therefore ambiguous and can mix original and replay ACTUAL rows. stamp_durable performs a shallow meta update, not an append transaction (nodes/production_ledger.py:527-558). Concrete fix: clear current-run render_trace during clone, build a complete fresh ordered trace locally, and stamp it once only after every segment succeeds. Give rows render_run_id, shot_id, segment_index, and explicit completion status; preserve the source trace only in the frozen bundle.

7. [D3′, §7 open item] Bundle images will not automatically produce cache hits. Dispatcher lookup follows cache_index to an inherited row and selects pool_path before path; a missing referenced file falls through to regeneration (nodes/otr_image_gen_dispatcher.py:1513-1598). It also increments image_revision and stamps a new images section (otr_image_gen_dispatcher.py:1160, 2015-2070). Concrete fix: choose one replay contract now: validate bundled image bytes, materialize them into the new canonical episode stills directory, rebase every row and cache reference, and make node 91 verify/re-stamp those exact bytes without invoking gen_fn. Peer-engine re-minting is outside an A/A replay.

8. [D4′] The manifest format is insufficient for safe deterministic import. Existence, size, and digest checks do not define schema version, authoritative ledger/master relative paths, duplicate handling, containment, symlink/reparse behavior, or completion atomicity. Concrete fix: use normalized relative paths only; reject absolute paths, .. traversal, escaping links/reparse points, and case-folded duplicates; record schema_version, source_episode_id, source commit, sizes, and SHA-256; build in a temporary sibling directory and rename only after all verification succeeds. Replay must consume only manifested files.

9. [D6′] The live acceptance omits mandatory proof for the production two-stage LTX 2.5 path. The local profile requires canonical publication plus loader/node evidence that the latent upsampler ran per shot and VAE decode used the upscaled canvas; adapter self-reporting is insufficient (.kibitz/comfyui.local.md:42-50). Concrete fix: add those log receipts, verify the live obs path, and enforce the 2.6x cost stop in the replay acceptance harness.

SHOULD-FIX:

1. [D1′] Stamp video_revision in meta alongside the producer-owned video section. stamp_durable copies only the supplied sections/meta fields, while ShotLock’s wire-only meta cannot update the singleton implicitly (nodes/production_ledger.py:527-558). Use sections={"video": section} and meta_updates={"video_revision": revision} in one call.

2. [§2.1, D2′] Define whether peak VRAM and wall_seconds are per physical segment or aggregate beat values. The current driver retains a single episode maximum and one trace row after render_beat_coverage (nodes/_otr_video_engines/render_driver.py:4789-4858). Instrument each actual engine invocation with monotonic elapsed time and segment-local peak; derive beat/episode aggregates separately.

3. [§2.1] Avoid re-hashing multi-gigabyte model files per clip. [ASSUMPTION] Repeated full-file hashing would scale with model size times clip count. Stream each distinct resolved model once per run, store a model_artifacts digest table, and reference its digest from each trace row.

4. [D3′] Node 7 must use a filesystem byte copy, verify the copied SHA against the manifest before emitting audio_done, then load that copied file only for its AUDIO return. Do not decode and re-encode the frozen master. Branch before current cue/audio processing begins (nodes/scene_sequencer.py:1227-1237).

5. [D6′] Add negative-path tests: traversal and digest mismatch; missing/zero-byte assets; copy failure; no CUDA/LLM/TTS/image-engine calls; multi-segment trace ordering; recomputation of actual_request_sha; publication receipt rebasing; source-bundle immutability; and failure to publish if durable trace stamping fails.

6. [D3′] Verify whether freeze_timestamp is allowed to identify two different episode workspaces. The merge code treats matching freeze timestamps as the same durable run (nodes/production_ledger.py:1585-1587). verify: every consumer of freeze_timestamp before preserving it unchanged across source and replay IDs; otherwise separate content_freeze_id from the new execution identity.

OPTIONAL / NICE-TO-HAVE:

Add a standalone receipt verifier that recomputes manifest hashes, actual_request_sha, request seeds, trace cardinality, and original-versus-replay equality without loading ComfyUI.

CUT THESE (over-engineering):

1. [D3′] Cut the AudioEnhance replay branch. A tiny typed placeholder from node 3 is sufficient because the properly wired node-7 replay branch ignores enhanced_audio.

2. [D3, §7 open item] Cut all peer-engine still re-minting from item 0. It invalidates the A/A input identity and belongs to a later candidate arm.

3. [D3] Keep per-line WAV copying cut. Node 92 slices conditioning audio from the master already (nodes/otr_video_render_batch.py:420-426); copying unused line assets adds failure surface.

4. [D1] Do not add a top-level render_trace or TOP_PRESERVE entry for it. meta.render_trace plus one atomic stamp is sufficient because meta already has per-key preservation semantics (nodes/production_ledger.py:1600-1618).
