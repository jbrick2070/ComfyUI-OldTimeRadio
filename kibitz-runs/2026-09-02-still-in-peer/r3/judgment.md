# r3 judgment -- still-in lab peer (2026-09-02)

Roster, r3: Cursor (`agent -p`, cursor-grok-4.6-high, `cursor.md`, 9.7 KB, file-grounded).
Driver: Claude (Cowork). Verdict "yes-with-fixes"; every claim checked against the real
Windows files before disposition. Seven must-fixes, all CONFIRMED, all taken.

## Must-fixes

1. STAGE 3b latent wire -- CONFIRMED. `run_graph` raises when an external id collides with a
   graph node (`wrapper_bridge.py:546-551`) and the parent's `sample_graph` carries
   `NODE_LATENT` (EmptyLatentImage) wired to `latent_image` (`eng_ghost_signal.py:849-867`).
   TAKEN: the peer builds STAGE 3b from a copy of the parent's graph with the `latent` node
   DELETED, `NODE_SAMPLER.inputs["latent_image"] = Wire("plate_init", 0)`, the repeated latent
   passed as `external_results["plate_init"]`, `denoise` = the resolved value. No
   `super().render_clip`.
2. `plate_path` dir-vs-file contradiction -- CONFIRMED (`_otr_paths.otr_stills_dir` returns
   `<episodes>/<id>/stills` and does not create it, `:312-323`). TAKEN: `plate_path` is the
   DIRECTORY `<otr_stills_dir(episode_id)>/ghost_plates`; the engine `makedirs(exist_ok=True)`
   and names the file `<sanitised shot_id>_<sha16>.png` itself (the driver cannot see the
   checkpoint identity).
3. Replay reuse and the frozen route -- CONFIRMED. `assert_frozen_route` requires
   `shot.engine_id == video.roles_effective[role]` (`render_driver.py:5209-5236`); the frozen
   route wins in `build_request_from_shot` (`:2104-2112`), so a profile/widget swap cannot move
   a replay; `coverage_contract_receipt` is `None` when the effective contract equals the
   declared one (`frame_contract.py:506-509`), so the coverage rewrite is a no-op for this
   sibling pair but the four surfaces still move together. TAKEN: the rewrite sits INSIDE
   ShotLock's replay early-return, BEFORE `_stamp_durable`, applied only when
   `meta.replay_engine_override` is set and validated as a Ghost sibling (family, roles,
   `prompt_profile`, `frame_contract` value-equal); it rewrites `roles_effective`, every shot's
   `engine_id` + `family`, every `execution_groups[*].engine_id`, and re-derives the coverage
   contract, refusing named on any leftover mismatch. The shipping baseline is
   `--derive-engine animatediff15_v3_haunted_video`, never a profile change.
4. Two audited `run_graph` calls cannot share `execution_records` -- CONFIRMED (a non-empty list
   at graph start is a named error, `wrapper_bridge.py:517-523`). TAKEN: a fresh `[]` per call;
   concatenate into `qc["graph_exec"]` after both succeed.
5. The plate decode needs the batch-1 latent, not the repeated one -- TAKEN: keep the batch-1
   LATENT dict referenced until the plate `VAEDecode` returns; STAGE 3b gets its own repeated
   copy; both cleared in the same `finally` as the decoded plate. `reclaim_idle_models` after
   3a is safe (`wrapper_bridge.py:404-457`, the parent does it post-encode).
6. The evidence-manifest generator cannot regenerate the live JSON -- CONFIRMED (generator
   emits `manifest_version: 1`, `:270`; the live file is version 7; `:376-387` refuses to
   overwrite a newer file; the haunted sentence exists only in the JSON, `:27`). TAKEN: append
   the peer's >= 6-word `admission_unenforced` sentence to the JSON in the same commit as
   `@register`; do not run the generator. (A generator that cannot reproduce its own artifact is
   a defect noted for the plan, outside this item.)
7. Guarded import order -- CONFIRMED (`eng_ghost_signal_official` imports at `__init__.py:333`;
   the roster audit runs at `:359`). TAKEN: the peer's guarded import sits between them.

## Should-fixes -- all TAKEN
S1 no custom menu label: `_label_for` derives the option from the id plus derived suffixes
(`otr_video_director.py:116-137`); a second spelling would break the bijection. The D1 label
is CUT.
S2 G3.7 greps the module that DEFINES `render_clip` (`_defining_module_source`,
`test_lane_preflight_matrix.py:421-433, 1069-1072`): the peer's module must itself read
`negative_prompt` (it encodes `plan["negative_prompt"]` from `_build_render_request`).
S3 `_prune_strict_text_only_request` strips only audio / base-clip / image asset keys
(`render_driver.py:1888-1951`) so `plate_prompt` / `plate_path` survive; `_stamp_render_trace`
copies receipt dicts wholesale (`otr_video_render_batch.py:329-334`), so the receipt
projection in `build_actual_receipt` is the ONLY path for the plate fields.
S4 verifier pair semantics: the two-replay A/A (equal `actual_request_sha`) is ONLY the two
same-denoise peer replays; the denoise-grid legs and the cross-engine baseline compare seeds
only; plate-hash equality only when both rows carry it. Written into the probe runner.
S5 no per-shot checkpoint SHA: `plate_identity` uses the checkpoint's `(size, mtime_ns)`
receipt (as `session_identity` does, `eng_ghost_signal.py:444-454`); the content digest is
already on the receipt once per process.
S6 a new node id for the plate text encode; the shared negative cond keeps the 11 render-time
instances (3 encodes, plate EmptyLatentImage, plate KSampler, context, lora, ADE loader,
video KSampler, 2 decodes); prepare's loader is outside the count, as in the parent's pin.
S7 `frame_contract.max_frames = 0` (`eng_ghost_signal.py:386-393`): no split, no
render_driver assumption to patch; `torch.repeat` is the U > 64 answer.

## Optional -- CONFIRMED, folded
Node 91 iterates listed `images[]` rows only (`otr_image_gen_dispatcher.py:1036-1053`), so an
extra `ghost_plates/` PNG is never validated: the anchor's verify-at-build item is closed.
`test_ghost_signal_peers.py` LANES are haunted + the unregistered v3 only; floors inherit.
The lab profile does not bake a denoise; grid legs set `OTR_STILLIN_LAB_DENOISE` per invocation.

## Cuts -- all TAKEN
Custom label; node-91 work; the generator run; reuse / VAEEncode / RepeatLatentBatch
candidates; `session_identity` override; new `CanonicalClip` keys; per-role / cross-family
override.
