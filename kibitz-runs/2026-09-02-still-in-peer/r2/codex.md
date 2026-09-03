VERDICT: build-ready as-is? no. The plan contains schema-invalid declarations, mutually exclusive plate-reuse mechanics, an unwired replay override, and insufficient data/receipt contracts for the claimed A/A proof.

MUST-FIX BEFORE BUILD:

1. [D1 / §4.2-3] `status: lab` is invalid in both proposed locations. Profiles accept only `shipping` or `draft` (`nodes/_otr_shared/capability_profiles.py:62-64,85-90`), while CAPABILITIES rejects every key outside `_DECL_KEYS`, which does not include `status` (`nodes/_otr_shared/capability_profiles.py:443-455,468-480`). Set the profile to `"status": "draft"` and omit `status` from the CAPABILITIES row; keep “lab” in the engine/profile identifiers and display name.

2. [D5 / D7] Fresh sampling and replay reuse cannot be implemented together as written. D5 forbids `VAEEncode` and retains the seven-class map, but D7 requires loading and encoding the copied PNG. The inherited map contains neither an image loader nor `VAEEncode` (`nodes/_otr_video_engines/eng_ghost_signal.py:148-160`). Specify two explicit branches: fresh = encode plate prompt, sample latent, repeat latent, decode/save PNG; reused = load PNG as `[1,H,W,C]` float32, run a verified `VAEEncode`, repeat its LATENT. Add the required candidate and branch tests. Verify: live `/object_info` names and input signatures for the installed `VAEEncode` and any proposed image loader.

3. [D3 / D7 / §8] The engine cannot construct `<episode>/stills/...` from the proposed request. `VideoRequest` has no episode or plate-path field (`nodes/_otr_video_engines/schemas.py:136-169`), `render_clip` receives only `(request, prepared)` (`nodes/_otr_video_engines/eng_ghost_signal.py:781-797`), and the canonical stills helper requires a valid episode id (`nodes/_otr_paths.py:312-323`). Add an optional non-causal `plate_path` field to `VideoRequest`; populate it in `build_request_from_shot`, which already receives the ledger (`nodes/_otr_video_engines/render_driver.py:2085-2093`). Do not read the production-ledger singleton from the engine.

4. [D11 / §4] No transport exists for `--replay-engine`. The CLI only patches `replay_from` (`scripts/otr_canonical_api_run.py:92-128,351-410`), the writer’s last optional input and run argument are `replay_from` (`nodes/OTR_LedgerScriptWriter.py:2764-2778,2908-2927`), and the canonical workflow ends with that widget (`workflows/otr_canonical.json:438-453`). Add `--replay-engine`, append a trailing `replay_engine_override` writer input and run argument, patch it in `_apply_writer_shortcuts`, stamp it into replay metadata, and update `workflows/otr_canonical.json` in the same change. Then run every canonical workflow/widget/link/object-info gate required by `.kibitz/comfyui.local.md:25-35`.

5. [D11] Restamping only `shot.engine_id` makes the replay internally inconsistent. The render boundary requires each shot to match `video.roles_effective` and its execution group (`nodes/_otr_video_engines/render_driver.py:5157-5159,5203-5243`), while ShotLock currently reuses the entire planned video section unchanged and returns immediately (`nodes/otr_shot_lock.py:2993-3015`). For this experiment, make the override whole-plan and restrict it to registered Ghost siblings with equal family, roles, `prompt_profile`, and `frame_contract`; atomically update `roles_effective`, every affected shot’s engine/family, and every affected execution group. Revalidate coverage plans; a stamped contract is rejected after an engine swap (`nodes/_otr_video_engines/render_driver.py:5355-5387`). Arbitrary cross-family overrides require replanning and are outside this frozen-replay design.

6. [D5 / §8] `init = samples.repeat(...)` has no valid data shape. `run_graph(..., terminal=...)` returns the terminal output tuple (`nodes/_otr_video_engines/wrapper_bridge.py:460-470`), and the current engine treats `sampled[0]` as the LATENT dictionary (`nodes/_otr_video_engines/eng_ghost_signal.py:917-924`). Define the operation as: validate `plate_latent["samples"]` has batch 1; shallow-copy the LATENT dict; repeat only its `"samples"` tensor to `[U,C,H,W]`; preserve every other LATENT key; construct the repeated `batch_index`; pass `(init_latent,)` through `external_results`. Verify: the exact `batch_index` transformation in the live installed ComfyUI implementation, because the cited `ComfyUI/nodes.py:1314-1330` was not available in the inspected runtime tree.

7. [D8 / D9] `plate_sha256` is not currently durable or verifiable. `CanonicalClip.qc` exists (`nodes/_otr_video_engines/schemas.py:213-244`), but `_clip_from_raw` drops it (`nodes/_otr_video_engines/eng_ghost_signal.py:1123-1146`), `build_actual_receipt` projects neither `qc` nor a plate hash (`nodes/_otr_video_engines/render_driver.py:4115-4146`), and `otr_verify_replay.py` compares only seeds and `actual_request_sha` (`scripts/otr_verify_replay.py:94-109`). Propagate `qc["plate_sha256"]`, then add `plate_sha256`, `plate_name`, and `plate_source` (`minted`/`reused`) as non-causal receipt fields. Extend the verifier to require non-empty equal plate hashes across peer A/A rows and verify the named file’s bytes against the receipt.

8. [§4.1 / D8] The proposed “session_identity moves with the plate sha” test is wrong and cannot be implemented cleanly: `session_identity(self)` has no request and describes model handles, while `shot_cache_identity(request)` owns per-shot state (`nodes/_otr_video_engines/eng_ghost_signal.py:456-501`). BeatSession also requires session identity to remain stable within a multi-segment beat (`nodes/_otr_video_engines/beat_session.py:270-289`). Remove the session-identity override/test. Put the plate-input identity and resolved denoise only in `shot_cache_identity`.

9. [D8 / D9] The denoise override has no defined name, range, or invalid-value behavior. Copying `lora_strength` would silently replace malformed input with the default (`nodes/_otr_video_engines/eng_ghost_signal_official.py:158-174`), and `run_graph` calls node functions directly rather than applying ComfyUI widget validation (`nodes/_otr_video_engines/wrapper_bridge.py:575-592`). Define one environment name, reject non-numeric/non-finite/out-of-range values outside `[0,1]`, resolve it once per request, and use the same resolved value for sampling, cache identity, and receipt.

10. [D7 / D8] “sha8 of plate INPUTS” is not a data contract. Define one canonical identity object containing checkpoint artifact digest, positive and negative plate prompts, seed, steps, CFG, sampler, scheduler, canvas, and adapter strength. Hash sorted canonical JSON, retain the full SHA-256 in the receipt, sanitize `shot_id` before using it as a filename because the schema accepts an unrestricted string (`nodes/_otr_video_engines/schemas.py:395-405`), and write via temporary sibling plus atomic replace. A reused file must be non-empty, decodable, 512×288, and match its recorded digest; otherwise fail loudly.

SHOULD-FIX:

1. [§4.1 / D3 / D8] Reconcile the stale build list with the revised decisions. `_plate_prompt(request, vstyle, ledger_world)` cannot live in the engine because `render_clip` has no style or ledger arguments (`nodes/_otr_video_engines/eng_ghost_signal.py:781-797`); composition belongs in `render_driver`, where `_vstyle` is already resolved (`nodes/_otr_video_engines/render_driver.py:2118-2123,2878-2882`). Likewise choose either `CanonicalClip.qc` or a new schema field, and either seven candidates or a replay encoder path—not both descriptions.

2. [D5 / §4.4] Replace “exactly N” with branch-specific numbers. [ASSUMPTION] Following D5 literally, the fresh path executes 11 render-time node instances: three text encodes, two plate-sampling nodes, four motion-sampling nodes, and two decodes. The replay count changes once the PNG-encoding branch is defined. Pin both branches; the existing parent test’s eight-instance count covers only the current graph (`tests/test_ghost_signal_haunted.py:165-170`).

3. [D9] Require positive execution evidence, not only a resulting hash. `run_graph` already supports `audit_node_ids` and structured execution records (`nodes/_otr_video_engines/wrapper_bridge.py:509-514,594-605`). A fresh leg must prove the plate KSampler and plate decode executed; replay legs must instead stamp `plate_source=reused`. This distinguishes a real branch from a copied or fabricated receipt.

4. [D9] Name the measurement implementation and make failures terminal. `scripts/otr_ltx_mad.py:1-34` already computes mean inter-frame difference; reuse its `mad_of` function rather than creating a second formula. Add the triptych generator, output paths, row-to-shot matching, and a numerical definition of “inside the null band.” Its current CLI catches per-file errors and continues, which is unsuitable for acceptance.

5. [D3 / §4.4 / §8] Resolve the nine-versus-ten style fixture ambiguity. Parameterize over the registered JSON styles via `list_style_ids()` (`nodes/_otr_visual_styles.py:516-564`) and add the embedded `visual_storybased` case handled separately at `nodes/_otr_visual_styles.py:586-603`. Assert both the 69-token protected-head target and the one-window ceiling; those are distinct constants (`nodes/_otr_video_engines/ghost_signal_author.py:124-129`).

6. [D5 / D9] Specify cleanup ownership on every exception path. The inherited path explicitly releases ADE/LoRA/base patchers before decode (`nodes/_otr_video_engines/eng_ghost_signal.py:925-932,1080-1086`). The new plate conditioning, sampled latent, repeated latent, and decoded plate image need `finally`-based reference clearing and reclaim points so a failed PNG write or second sampler does not retain beat-sized CUDA tensors.

7. [§4.1] Keep all new PIL/numpy/torch-dependent work lazy. The package promises cold-import isolation (`nodes/_otr_video_engines/__init__.py:9-13`); extend the cold-import test to the new guarded module.

OPTIONAL / NICE-TO-HAVE:

- [D7] Store the full plate-input SHA-256 in metadata while using a 12- or 16-character filename suffix for readability; eight characters is unnecessary truncation for a persistent cache.
- [D8] Add a small pure `plate_identity(request, resolved_denoise)` helper shared by filename selection, `shot_cache_identity`, and `sampler_inputs_for` to prevent those three representations drifting.

CUT THESE (over-engineering):

1. [§4.1] Cut the `session_identity` override and its plate-SHA test. The inherited implementation already includes `self.name`, recipe, and artifact receipts (`nodes/_otr_video_engines/eng_ghost_signal.py:456-478`).

2. [§2 / §4.1] Cut a `model_artifacts` override. The inherited method reads the subclass’s checkpoint, motion-module, and adapter attributes already (`nodes/_otr_video_engines/eng_ghost_signal.py:433-441`).

3. [§4.1 / D8] Cut the proposed top-level `CanonicalClip.plate_sha256` schema extension. Existing `qc` is the correct clip-local container (`nodes/_otr_video_engines/schemas.py:239-244`); only the durable render receipt needs a named non-causal projection.

4. [D11] Cut generic per-role and cross-family replay-engine syntax. A single whole-plan override constrained to compatible Ghost siblings is sufficient for this campaign and avoids re-authoring prompts or rebuilding coverage plans.

5. [D9] Cut a second motion-energy implementation. Reuse the existing MAD calculation in `scripts/otr_ltx_mad.py`; add only fail-loud campaign orchestration and the missing triptych output.

Skill used: kibitz, for a repository-grounded adversarial coding-plan review.
