VERDICT: yes-with-fixes -- the peer can sit beside haunted, but four contracts as written will not connect (STAGE 3b latent wire, plate_path dir vs file, D11 rewrite seat, evidence-manifest regenerate).

MUST-FIX BEFORE BUILD:

1. [§10 Engine render_clip / D5] STAGE 3b latent wire is unspecified and the default copy is a collision. Parent sample_graph still contains NODE_LATENT EmptyLatentImage with `latent_image: Wire(NODE_LATENT)` (`eng_ghost_signal.py:849-867`). `run_graph` raises if an id is in both `graph` and `external_results` (`wrapper_bridge.py:548-551`). If NODE_LATENT is left in and the repeated latent is passed under a different id, STAGE 3b still samples empty noise and the plate is dropped. Concrete fix: copy the parent sample_graph, DELETE the `latent` node, set `NODE_SAMPLER.inputs["latent_image"]` to `Wire("<external_id>", 0)`, pass `(repeated_latent,)` as that external, set `denoise` to the resolved env value. Do not call `super().render_clip`.

2. [§10 Request and driver / r2 #3 vs §10 Engine] `plate_path` contract contradicts itself. r2 judgment: directory, filled by `build_request_from_shot`. §10 engine: "PNG to `request.plate_path` via temp sibling + `os.replace`" (a file). `_otr_paths.otr_stills_dir` returns `<episodes>/<id>/stills` and does not mkdir (`_otr_paths.py:27-32, 312-323`). `os.replace` onto a missing parent or onto a directory fails. Concrete fix: `plate_path` is the directory `<otr_stills_dir(episode_id)>/ghost_plates`; engine `os.makedirs(plate_path, exist_ok=True)`, write to `os.path.join(plate_path, "<shot_id>_<sha16>.png")` via temp sibling + `os.replace`. Driver never names the file (it cannot see checkpoint identity).

3. [§10 D11 / `otr_shot_lock.py:2993-3015`] Replay reuse currently stamps the imported `video` section unchanged and returns. `assert_frozen_route` then requires `shot.engine_id == video.roles_effective[role]` (`render_driver.py:5209-5236`). A lab-profile widget swap cannot move a replay: frozen route wins (`build_request_from_shot` `render_driver.py:2104-2112`). Coverage rewrite is a no-op for this pair (`coverage_contract_receipt` returns None when `max_frames=0` equals the declared contract, `frame_contract.py:506-509`) but `roles_effective` / `execution_groups[*].engine_id` / every shot `engine_id`+`family` must still move together. Concrete fix: inside the replay early-return, BEFORE `_stamp_durable`: if `meta.replay_engine_override` is set, validate Ghost-sibling (family, `roles`, `prompt_profile`, `frame_contract` value-equal), rewrite those four surfaces atomically, refuse named on any leftover mismatch. Probe shipping baseline uses `--derive-engine animatediff15_v3_haunted_video`, not a profile change.

4. [§10 Engine render_clip / `wrapper_bridge.py:517-523`] Two audited `run_graph` calls cannot share `execution_records`. A non-empty list at graph start is a named `GraphExecutionError`. Concrete fix: pass a fresh `[]` into STAGE 3a (plate sampler) and another into STAGE 4 plate decode; concatenate into `qc["graph_exec"]` after both succeed.

5. [§10 Engine render_clip cleanup vs STAGE 4] Plate PNG decode needs the batch-1 latent from STAGE 3a, not the U-repeated tensor STAGE 3b consumes. Clearing `plate_latent` after 3a (the `finally` list) leaves STAGE 4 nothing to decode, or decodes U frames. `reclaim_idle_models` after 3a is safe (it detaches loaded patchers, `wrapper_bridge.py:404-457`; parent already does this post-encode while `base_model` stays as a handle). Concrete fix: keep a Python reference to the batch-1 latent dict until the plate VAEDecode returns; clear it in the same `finally` as the decoded plate. STAGE 3b gets a separate repeated copy.

6. [§10 Registration / G4] `scripts/build_video_evidence_manifest.py` regenerate will not run. Generator emits `manifest_version: 1` (`:270`); live JSON is version 7 (`docs/evidence/video_evidence_manifest.json:3`) and the generator refuses to overwrite a newer file (`:379-387`). G4 reads the JSON (`test_lane_preflight_matrix.py:451-452, 796-807`), not the generator. Haunted's sentence exists only in the JSON (`:27`), not in the generator's `admission_unenforced` dict. Concrete fix: append a >=6-word `admission_unenforced` sentence for `animatediff15_v3_stillin_lab_video` to the JSON in the same commit as `@register`. Do not run the generator.

7. [§10 Registration / `_otr_video_engines/__init__.py:318-378`] Guarded import of `eng_ghost_signal_stillin_lab` must sit after `eng_ghost_signal_official` (it subclasses `GhostSignalV3HauntedEngine`) and before the roster audit at the bottom. An import after the audit, or a CAPABILITIES row without a successful `@register`, is a silent dropdown miss plus a roster-audit error.

SHOULD-FIX:

1. [D1 menu label] There is no display-name seam. Live combo is `_label_for(id)` = internal id + aspect suffix (`otr_video_director.py:116-137`). Profile JSON stores internal ids (see `otr_ghost_signal_v3_haunted.json:11-24`). `build_variants` calls `exact_menu_option_for`. Do not try to stamp "AnimateDiff v3 still-in lab (512x288)" into widgets.

2. [G3.7 / `test_lane_preflight_matrix.py:421-433, 1069-1072`] Overriding `render_clip` moves the G3.7 grep onto the NEW module. That file must contain `negative_prompt` (use `plan = self._build_render_request(request)` and encode `plan["negative_prompt"]`). A child that only calls helpers in the parent module fails G3.7 RED.

3. [§10 Request and driver] Add `plate_prompt: str = ""` and `plate_path: str = ""` to `VideoRequest` (`schemas.py:136-169`, `extra="forbid"`). Fill both in the Ghost branch after `_eng_id` is the post-rewrite id and `_vstyle` is resolved (`render_driver.py:2118-2122, 2878-2918`), only when `getattr(eng, "wants_plate_prompt", False)`. `_prune_strict_text_only_request` does not strip unknown top-level keys (`:1888-1951`) -- it will not delete `plate_prompt`. `build_actual_receipt` must copy `clip["qc"]` plate fields into the non-causal receipt block (`:4136-4142`); `_stamp_render_trace` copies receipt dicts wholesale (`otr_video_render_batch.py:329-334`), so a missing projection never reaches the verifier.

4. [§10 D9 / `otr_verify_replay.py:104-109`] The two-replay A/A compares `actual_request_sha`. Denoise-grid legs (0.35/0.50/0.80) MUST NOT be those two args. Cross-engine (peer vs haunted) is seed-only (`:99-103`) and must not expect equal shas. Plate-hash equality only when both rows carry `plate_sha256`.

5. [§10 plate_identity] Do not sha256 the checkpoint file per shot. `session_identity` already uses `(size, mtime_ns)` (`eng_ghost_signal.py:444-454, 464-469`); `build_actual_receipt` already content-hashes artifacts once per process (`render_driver.py:4045-4062`). Use `(size, mtime_ns)` or the cached digest. A 2 GB hash inside `shot_cache_identity` is a hidden stall on beat 1.

6. [§10 Engine STAGE 2] Third encode needs a new node id (not `NODE_POSITIVE`). Shared negative_cond for plate sampler and video sampler keeps the 11-instance pin (3 CLIPTextEncode + plate EmptyLatentImage + plate KSampler + context + lora + ADE + video KSampler + 2 VAEDecode). Prepare's CheckpointLoaderSimple is outside that count (`test_ghost_signal_haunted.py:165-170` is 8 render-time tags). Pin the test the same way.

7. [Q6] Ghost `frame_contract.max_frames=0` (`eng_ghost_signal.py:386-393`) so `render_beat_coverage` will not split. `torch.repeat` with no 64 cap is the right U>64 answer (`RepeatLatentBatch` amount is capped at 64 in stock ComfyUI -- verify: `ComfyUI/nodes.py` RepeatLatentBatch; this tree has no copy). No render_driver EmptyLatentImage assumption to patch.

OPTIONAL / NICE-TO-HAVE:

- Node 91 already ignores extra files (`verify_replay_images` only checks ledger `images[]` rows, `otr_image_gen_dispatcher.py:1036-1053`). Freeze `stills/**` rglob will copy `ghost_plates/` (`otr_freeze_replay_bundle.py:146-153`); harmless because r2 remints. No node 91 change.
- `test_ghost_signal_peers.py` LANES is only haunted + unregistered v3 (`:38-40`). Floors inherit; no edit required for green.
- Lab profile `launch.env` may set `OTR_STILLIN_LAB_DENOISE` for the 0.65 source; grid legs should set the env per invocation, not bake 0.65 into the JSON.

CUT THESE:

1. Custom director menu label (D1) -- `_label_for` already produces a unique combo option from the id; a second spelling breaks `exact_menu_option_for` bijection.
2. Node 91 "tolerate unlisted plate" work -- already true; extra PNGs are not validated.
3. Running `build_video_evidence_manifest.py` -- it cannot regenerate the v7 JSON; edit the JSON `admission_unenforced` map the way haunted was added.
4. PNG-reuse / VAEEncode / RepeatLatentBatch node candidates -- already cut in r2; putting them back reopens the seven-class map. First build remints.
5. `session_identity` override -- BeatSession compares identity mid-beat with no request (`beat_session.py:280-290`; `eng_ghost_signal.py:456-462`). Denoise/plate belong in `shot_cache_identity` only.
6. New `CanonicalClip` keys -- `qc` is already a declared dict (`schemas.py:244`). Nested plate fields go there.
7. Per-role / cross-family D11 -- frozen route and coverage boundary have no partial-override shape; whole-plan Ghost-sibling only.

[ASSUMPTION] Stock `RepeatLatentBatch` batch_index offset rule was not readable from this tree's ComfyUI checkout (`C:\Users\jeffr\Documents\ComfyUI\nodes.py` is not present). Copy the live node's body at build; do not invent a second rule.

[ASSUMPTION] `wants_plate_prompt` is a new duck-typed attribute; nothing in the registry schema consumes it. That is acceptable if the only reader is `build_request_from_shot`.
