# r2 judgment -- still-in lab peer (2026-09-02)

Roster, r2: Codex (`codex exec`, gpt-5.6-sol, `codex.md`, 11.6 KB, file-grounded). Driver:
Claude (Cowork). Every claim checked against the real Windows files before disposition.

## Must-fixes

1. `status: lab` invalid -- CONFIRMED. `_STATUSES = ("shipping", "draft")`
   (`capability_profiles.py:64`); CAPABILITIES rows are validated against `_DECL_KEYS` and an
   unknown key raises `ProfileError` (`:443-455`). TAKEN: profile `"status": "draft"`; no
   `status` key in the CAPABILITIES row; "lab" lives in the id and the label only.
2. Fresh sampling and PNG reuse cannot both hold with a seven-class map -- CONFIRMED
   (`GHOST_NODE_CANDIDATES`, `eng_ghost_signal.py:148-160`, has no loader and no `VAEEncode`).
   TAKEN by SIMPLIFYING, not by adding a branch: the first build RE-MINTS the plate on every
   render (deterministic from the plate identity) and records `plate_sha256` non-causally; the
   verifier requires the peer's A/A rows to carry EQUAL plate hashes. If SD1.5 turns out not
   bit-stable on this box, PNG reuse becomes the next knob with its own encoder branch and
   candidate names. Seven classes, one topology, one instance count.
3. The engine cannot build `<episode>/stills/...` from the request -- CONFIRMED (`VideoRequest`
   fields `schemas.py:136-169`; `render_clip(request, prepared)`; `build_request_from_shot`
   receives the ledger, `render_driver.py:2085-2093`). TAKEN: optional NON-causal
   `VideoRequest.plate_path: str = ""` (the directory), populated in `build_request_from_shot`
   for engines declaring `wants_plate_prompt`; the engine writes the PNG there and never reads
   the ledger singleton.
4. No transport for `--replay-engine` -- CONFIRMED (`otr_canonical_api_run.py:92-128`; the
   writer's last input is `replay_from`; the canonical ends with that widget). DISPOSITION: the
   override travels IN THE BUNDLE, not on a new widget. `scripts/otr_freeze_replay_bundle.py
   --derive-engine <id> <bundle>` writes a sibling bundle `<bundle>__engine_<id>` whose manifest
   carries `engine_override` (same files, same hashes, a new manifest, immutable like any
   bundle); `import_replay_bundle` reads it and stamps `meta.replay_engine_override`; ShotLock's
   reuse branch applies it. No canonical edit, no widget, no whitelist change; the replay's
   `replay_from` path names the override in every receipt. Codex's widget route is the
   fallback if the panel finds a reason the manifest cannot carry it.
5. Re-stamping only `shot.engine_id` is inconsistent -- CONFIRMED (`video.roles_effective` read at
   the render boundary `render_driver.py:5158`; the coverage contract is re-derived per engine
   and a mismatch is a NAMED RenderError `:5355-5362`; ShotLock's reuse returns the section
   unchanged). TAKEN: whole-plan override only, restricted to registered Ghost siblings with
   equal family, roles, `prompt_profile` and `frame_contract`; the reuse branch updates
   `roles_effective`, every shot's `engine_id` / family, every execution group, and re-derives
   each shot's coverage contract through the same function the boundary uses, refusing (named)
   if any differs. Per-role and cross-family overrides are CUT.
6. `samples.repeat` shape -- CONFIRMED. `run_graph(..., terminal=)` returns the terminal node's
   tuple (`wrapper_bridge.py:460-470`); the lane treats `sampled[0]` as the LATENT dict
   (`eng_ghost_signal.py:917-924`). The live `RepeatLatentBatch` (`ComfyUI/nodes.py:1317-1330`)
   copies the dict, repeats `samples` along batch, repeats `noise_mask` when present, and
   extends `batch_index` by offset blocks. TAKEN as the contract: assert batch 1, shallow-copy,
   repeat `samples` to `[U, C, H, W]`, replicate `batch_index` with the same offset rule when
   present, pass `(init_latent,)` through `external_results`.
7. `plate_sha256` not durable -- CONFIRMED (`_clip_from_raw` returns a fixed dict without `qc`;
   `build_actual_receipt`'s non-causal block is `clip_path / frame_count / vram_peak_mb /
   wall_s / status`, `render_driver.py:4137-4142`; the verifier compares seeds and
   `actual_request_sha` only). TAKEN: the clip carries `qc["plate_sha256"]`, the receipt gains
   non-causal `plate_sha256`, `plate_name`, `plate_source` ("minted" only, in this build), and
   `otr_verify_replay.py` requires non-empty equal plate hashes across the peer's A/A rows and
   verifies the named file's bytes when the episode dir is at hand.
8. `session_identity` has no request and BeatSession requires it stable within a beat --
   CONFIRMED (`eng_ghost_signal.py:456-478`; `beat_session.py:280-290` raises
   `SessionIdentityDrift`). TAKEN: no `session_identity` override; plate identity + resolved
   denoise go only into `shot_cache_identity`.
9. Denoise env contract -- TAKEN: `OTR_STILLIN_LAB_DENOISE`, read once per request inside
   `assert_usable`, must parse as a finite float in [0, 1] or raise a NAMED `EngineUnusable`
   (never a silent default -- the shipping `lora_strength` pattern is not copied here because
   a silently-defaulted denoise would print a receipt that lies); the resolved value feeds the
   sampler, `shot_cache_identity` and the receipt.
10. Plate identity object -- TAKEN: canonical sorted JSON of {checkpoint digest, plate positive,
    plate negative, seed, steps, cfg, sampler, scheduler, canvas, plate adapter strength}
    -> SHA-256 (full in the receipt; 16 chars in the filename); `shot_id` sanitised for the
    filename; write via temp sibling + `os.replace`.

## Should-fixes -- all TAKEN
S1 build list reconciled (composition in `render_driver`; `qc` only; seven candidates only).
S2 one instance count for the fresh path: 11 (3 text encodes, plate EmptyLatentImage + plate
KSampler, context + lora + ADE loader + sampler, beat decode + plate decode) -- pinned by the
peer's test; the parent's 8 stays.
S3 positive execution evidence: the plate KSampler and plate decode are `audit_node_ids`
(`wrapper_bridge.py:509-514, 594-605` CONFIRMED); their records are stamped on the clip's `qc`.
S4 motion energy: reuse `scripts/otr_ltx_mad.py::mad_of` (`:26`, CONFIRMED); the campaign
runner (new, `scripts/otr_stillin_probe_report.py`) is fail-loud, matches rows to shots by
`shot_id`, writes the triptych cards, and defines the null band as the interval between the
two A/A nulls' per-beat MAD widened by 10% of its width.
S5 fixtures: parameterise over `list_style_ids()` plus one embedded `visual_storybased` case;
assert both the 69-token target and the 77-token window.
S6 cleanup: `finally`-based clearing of plate cond, plate latent, repeated latent and decoded
plate; reclaim points after STAGE 3a and after the plate decode.
S7 lazy imports; extend `test_cold_import_no_heavy_libs` to the new module.

## Cuts -- all TAKEN
`session_identity` override; `model_artifacts` override (inherited reads the subclass attrs);
`CanonicalClip` field; per-role / cross-family override; a second MAD implementation.
