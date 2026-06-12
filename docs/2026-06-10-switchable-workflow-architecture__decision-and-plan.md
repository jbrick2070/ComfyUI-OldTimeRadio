# Switchable Workflow Architecture -- Decision + Draft Coding Plan (FINAL, roundtable-hardened)
*2026-06-10. Judge + panelist: claude-fable-5 (final say). Panel: gpt-5.5, gemini-3.1-pro, grok-4.3, deepseek-v4-pro.*
*3 grounded passes, converged. Campaign artifacts: `docs/2026-06-10-switchable-workflow-architecture/roundtable/` (pass00-03 plans, raw reviews, judgment logs). Source problem statement: `2026-06-10-switchable-workflow-architecture__problem-statement.md`.*
*STATUS: PLANNING ONLY. No code has been written. The sprint plan below is a DRAFT awaiting operator go.*

> **EXECUTION ORDER (operator-directed 2026-06-10):** this doc is a SPEC / decision record, NOT a separate plan to track. Its sprints are sequenced inside the 3D plan -- S0-S2 are GATE B (BEFORE the 3D sprints; the drift kill), S3-S6 are the closing distribution phase (AFTER the 3D work). See `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` section 0. One forward plan, not three.

---

## 1. The decision

**ONE master switchable graph, GENERATED per-tier snapshots, a thin committed PROFILE layer.** The operator instinct (one architecture with switches) survived three adversarial passes intact; no panelist challenged it. What the panel reshaped -- heavily -- is the mechanics, captured below.

- **Author/maintain exactly one master:** `workflows/otr_scifi_16gb_full.json` keeps its name and role. Single source of truth for graph STRUCTURE. The master is the dev/reference artifact, stays UNSTAMPED, and remains runnable.
- **Distribute generated snapshots:** `workflows/generated/otr_16gb_full.gen.json`, `otr_8gb_lite.gen.json`, `otr_cpu_floor.gen.json`. `.gen.json` = never hand-edit (banner also written into `extra.info`). End users click-Run a snapshot; nobody rewires nodes.
- **Profiles are committed data:** `config/profiles/<id>.json` -- capability POLICY, not creative presets. They carry OVERRIDES only; registry defaults supply the base.
- **The drift bug dies structurally:** ONE applier consumed by the generator, the headless scripts, and CI -- the hand-coded patch lists inside soak/smoke scripts (the actual mechanism of the captions/credits/LTX-open bug) are deleted.

## 2. Verified ground truth this plan stands on
- The drift was never "two graphs": headless loads the SAME production JSON and patches widgets by name at submit; the second source of truth was the per-script patch list. (Verified in `scripts/otr_api.py` + `queue_smoke.py`.)
- Switch grain already exists: Director per-role dropdowns (`OTR_VideoDirector` node 87 / `OTR_ImageDirector` node 88), ~17 `OTR_ENABLE_*` flags, registries with fallback chains for audio/video/image, `OTR_FORCE_ENGINE_MAP` (video-only, `render_driver.py:607+`), sage gate via `comfy.model_management.sage_attention_enabled()` (`motion_common.py:63-77`).
- Engine selectors live OUTSIDE the Directors too (verified in the JSON): node 81 `OTR_BatchCharacterVoices` ("indextts2"), 82 `OTR_AnnouncerVoice` ("kokoro"), 83 `OTR_StableAudioTheme` ("stable_audio_3"), 92 `OTR_VideoRenderBatch` ("humo"), 3 `OTR_SceneSequencer` ("bark").
- Node 63 `OTR_WorkflowValidator` ships `["", true, true]`; empty path falls back to the canonical 16gb file, and non-empty relative paths resolve against process CWD (`_load_workflow`, line ~69; `IS_CHANGED` same defect).
- `VRAM_CEILING_MB = 14500` is hardcoded (`wrapper_bridge.py:37`).
- Zero `torch.backends.mps` references; ~35 scattered cuda/platform checks; no central device router. ffmpeg is already Mac-safe (libx264 everywhere, mux `-c:v copy`, nvenc behind `_check_nvenc()`).

## 3. The profile system
```
config/profiles/<id>.json
{ "id": "8gb_lite", "display_name": "8 GB NVIDIA (lite)", "platform": "any|win|mac",
  "device_backend": "cuda|cpu",              // mps deliberately absent in v1
  "vram_budget_mb": 7300,                     // -> runtime ceiling
  "toolchains": ["cu128"] | [],
  "allow_sidecars": false,                    // false => sidecar-requiring engines excluded
  "max_model_class": "light",                 // caps engine vram_class
  "role_overrides": { },                      // single engine id per role widget; chains stay registry-owned
  "slot_overrides": { },                      // audio/video slot widgets (types of nodes 81/82/83/92/3)
  "features": { "burn_captions": true, "procgen_credits": true, "ltx_radio_open": false },
  "seed_policy": { "request_seed": 42, "seed_mode": "request_hash",   // applies to EVERY widget named request_seed
                    "cast_seed_env": null, "style_seed_env": null },
  "launch": { "sage_attention": false, "extra_args": [] } }
```
- **Derived enable-set, never hand-listed.** Registries gain DEP-FREE per-engine declarations in the registry table modules (NOT in adapter modules): `vram_class`/`vram_estimate_mb`, `required_toolchain`, `requires_sidecar`, minimal `model_requirements`. `enabled(P)` = engines whose declarations fit P. New engine ships its own metadata; zero profile edits.
- **Profile validator:** S0 = shape + unknown-key rejection; S1 adds capability cross-checks (every override must be in `enabled(P)`; per-engine fit ONLY -- no static co-residency rejection, residency is `wrapper_bridge`'s runtime invariant; regression test: a profile with two heavy roles still yields an enable-set).
- **Availability formulas (explicit):**
  - normal: `registry reality  ∩  OTR_ENABLE_* env gates  ∩  enabled(P)`
  - forced (`OTR_FORCE_ENGINE_MAP`, VIDEO-scoped in v1): `registry reality  ∩  env gates` -- explicitly NOT `enabled(P)`; forcing outside the profile logs LOUD. Tests: parse-error + outside-profile.
  - selection: FORCE map > saved widgets (= profile-applied) > resolver fallback chains.
  - One LOUD line at queue start: resolved profile, source, every override in effect (shared availability object with reason codes, reused by validator/wizard/log).

## 4. How a profile reaches the graph
- **Two operations, strictly split:**
  - `apply_profile(workflow, profile)` -- pure semantic widget patching. Never stamps, never touches paths.
  - `emit_snapshot(applied, artifact_path)` -- generator-only; writes node-63 artifact path + stamp.
- **One shared module** (promote the `scripts/otr_api.py` patch-by-NAME machinery into the package). Offline schemas via direct `NODE_CLASS_MAPPINGS` import through an **INPUT_TYPES -> /object_info adapter** that replicates `_serialized_slot_names` semantics (forceInput consumes no slot; seed companions) and is tested against that function. The OTR package root is imported FIRST so registries self-populate (otherwise Director COMBOs are empty and patching fails). Live `/object_info` cross-check stays in the soak lane; CI runs offline.
- **Stamp = three new optional STRING widgets on `OTR_WorkflowValidator`** (`profile_id`, `master_hash` = sha256 of master content at emit, `generated_by`). Same commit: INPUT_TYPES change + master `widgets_values` padded to the new length + widget-vector tests updated. No node `properties` (not executable in API prompts). Master: stamps empty. The three stamp widgets are the ONLY new optional fields allowed on node 63.
- **Startup assertion** lives in `OTR_WorkflowValidator.validate` (FUNCTION is `validate`, not `execute`), ACTIVE whenever `profile_id` is non-empty, and `validate_anyway` can NEVER skip it (it only skips the contract check; CI rejects snapshots shipping `validate_anyway=false`). It resolves the stamp against the committed profile + DETECTED reality; mismatch -> raise (prompt aborts) with a reason->suggestion table (no cuda -> cpu_floor; VRAM<10GB -> 8gb_lite; mac -> cpu_floor).
- **Runtime export (every execution, not "if unset"):** the stamped validator sets `OTR_VRAM_CEILING_MB`, `OTR_ACTIVE_PROFILE`, `OTR_SNAPSHOT_HASH` (sha256 of the validated file) into `os.environ`. A long-running server persists env across prompts, so stale values are overwritten each run; if an operator-set value conflicts, warn LOUD and the SMALLER ceiling wins. `wrapper_bridge` reads the env at dispatch time (env > 14500 fallback) -- no graph introspection. Ledger restamp + registries read the env; headless sets the same vars directly.
- **Ordering guarantee:** wire node 63's `validation_report` into the EXISTING `gate_in` (`forceInput=True`) on `OTR_VideoDirector` (node 87) -- zero schema change. S0 verifies whether the image lane (88 -> 91 `OTR_ImageGenDispatcher`) is downstream of that gate; if not, ONE forceInput-only gate input is added to the image dispatcher type. Gates are never widget-backed. (The UI `order` field guarantees nothing server-side -- the API prompt does not carry it.) Accepted, documented gap: independent AUDIO nodes may execute before an assertion failure aborts the prompt -- cheap + LOUD (validator docstring note + queue_smoke LOUD log when a profile is present).

## 5. Widget coverage -- closed by test, not by list
Coverage test enumerates from the checked-in MAPPING (S0 deliverable: a JSON mapping profile key -> `(node_type, widget_name)` read from real INPUT_TYPES, consumed by both the applier and tests; unique-match assertion; raw node ids banned):
- Direction 1: every profile field has a graph target.
- Direction 2: every managed widget is patched ONLY via the applier.
- Engine-widget detection: COMBO CHOICES intersecting registry ids (not saved values); plus feature BOOLEANs and every `request_seed` widget.
- Initial classification -- MANAGED: Director dropdowns (`OTR_VideoDirector`/`OTR_ImageDirector` types), `OTR_BatchCharacterVoices`, `OTR_AnnouncerVoice`, `OTR_StableAudioTheme`, `OTR_VideoRenderBatch` engine default, `OTR_SceneSequencer` voice default, feature widgets, node-63 path+stamp (emit-only). EXEMPT (creative, headless-whitelisted): `OTR_LedgerScriptWriter` model slots (`openrouter_slot_*`/`comfy_slot_*`, admissible via the existing `_is_openrouter_admissible`/`_is_comfy_admissible` paths, which the CI COMBO check and the cold-load test both reuse). SEED-POLICY: `request_seed` (never patch a widget named `seed` -- companion-slot trap).

## 6. CI gates
- **Identity (S2):** `to_api_prompt(master) == to_api_prompt(apply_profile(master, profile_16gb))` -- dict equality on the offline adapter; both sides have empty stamps (no ignore needed). BOOTSTRAP: the 16gb profile is EXTRACTED from the master's current values, so day-one green; later divergence fails CI unless master + profile change together.
- **Parity (S3, all tiers):** `to_api_prompt(snapshot(P)) == to_api_prompt(apply_profile(master, P))`, deleting node-63's `workflow_json_path` + stamp fields from BOTH dicts by node-type lookup.
- **Regenerate-diff (S3):** regenerate every tier from master -> byte-identical to committed `.gen.json`, else fail.
- **Contract-on-every-artifact (S3):** widget-vector validation for each `.gen.json`; saved COMBO values must exist in the target schema (dynamic slots via admissibility paths).
- **Cold-load per tier (S3, ALL tiers):** each `.gen.json` loads in its target env; saved values present in that env's roster.
- **Punch-list acceptance demo (S2):** captions + procgen credits + LTX radio open render identically from UI-load and headless-submit on the 16gb profile -- the original bug, closed and proven.

## 7. VRAM safety
`wrapper_bridge`: `OTR_VRAM_CEILING_MB` env (read at dispatch) else 14500. Sources: stamped validator export (UI), launcher/scripts (headless). **8gb_lite ship gate:** full-episode soak on the 5080 under the simulated 8GB ceiling with peak-VRAM assertion. A tier we never rendered does not ship.

## 8. Headless conversion (the drift kill)
`queue_smoke.py` + soak runners take `--profile <id>`, call `apply_profile`, DELETE hard-coded engine/feature patch lists, and print the resolved profile LOUD. Whitelist (exact): `target_words, num_characters, act_count, request_seed` + seed-policy fields, prompt/title text fields, `openrouter_slot_*`/`comfy_slot_*` (via a whitelisted helper). Enforcement, both stateless: `patch_creative()` validates names against the whitelist; a regression test asserts no direct `patch_widget_by_name` on managed names in scripts.

## 9. v1 tier set (honest matrix)
`16gb_full`, `8gb_lite`, `cpu_floor`. **Mac ships AS cpu_floor, labeled exactly that.** MPS is a PARKED follow-on sprint (central `device_routing.py` + per-adapter plumbing + probes). No "Mac-MPS" name in any v1 artifact. `cpu_floor` profile is committed in S0 but NON-SHIPPING until the S1 cold-import and S3 cold-load gates pass.

## 10. Determinism (scoped honestly)
Ship gate per tier (at tier release, not per-commit): given an identical script/ledger fixture + pinned seeds -- the gate harness sets `OTR_CAST_SEED`/`OTR_STYLE_SEED`; production default keeps OS entropy per the platform invariant -- the render pipeline produces normalized-identical outputs: audio byte-identical (existing machinery), video by stream hash with container metadata stripped, ledger fields compared. Writer/LLM stage is OUT of contract (remote sampling is not seedable). Ledger records `profile_id` + `snapshot_hash` from env.

## 11. Decision B -- setup wizard (after the core)
CLI, one-shot, OUTSIDE the render path: detect -> propose tier -> confirm -> record choice -> optional model download -> emit launcher (`OTR_VRAM_CEILING_MB`, optional sage args). Manifest GENERATED from registry `model_requirements` of the SELECTED + fallback-chain engines (optional "all compatible" mode); `huggingface_hub` natively honors `HF_HOME`; destination honors `extra_model_paths.yaml`; size-confirm before download; resumable + skip-existing.

## 12. CUT from this build
Gradio/Streamlit; `h264_videotoolbox` (separate perf ticket); MPS routing (parked); `character_3d` as a v1 switch (the profile design must not block it later -- nothing more; the 3D image-routing must-fixes remain pre-reading for whoever wires it); full cartesian testing; byte-identical master self-check (replaced by api-prompt dict equality); static co-residency rejection; per-commit determinism CI; duplicate stamp homes; detailed downloader metadata in v1 registry declarations.

## 13. Draft sprint plan (DRAFT -- awaiting operator go; coder-window work, not this window)
- **S0 -- Profile foundation (data/docs only, no graph change):** schema + `config/profiles/` (3 tiers; 16gb EXTRACTED from master) + shape validator + precedence spec + stamp format + checked-in widget MAPPING JSON read from real INPUT_TYPES (resolves all name assumptions: 92/81/82/83/3, request_seed instances) + verify image-lane gate topology (87 vs 91).
- **S1 -- Registry metadata + early gates:** dep-free capability declarations; derived enable-set + cross-validation (+ two-heavy-roles regression); dynamic ceiling in wrapper_bridge (env-at-dispatch > 14500); **BLOCKING cold-import gate** with the lazy-import refactor strategy and the defined fallback (an un-import-safe adapter registers behind try/except and is absent from that env's roster -- the architecture survives any single stubborn adapter).
- **S2 -- One applier + headless conversion (closes the drift bug):** promote otr_api machinery + offline schema adapter (tested vs `_serialized_slot_names`); `apply_profile`; coverage test; `_load_workflow` + `IS_CHANGED` repo-root fix (MUST merge before S3 emits anything); node-63 stamp widgets + master padding + widget-vector test updates (same commit); assertion + env export in `validate`; gate wired via node-87 `gate_in` (+ image dispatcher if S0 says so); queue_smoke/soaks on `--profile` (lists deleted, whitelist enforced); IDENTITY gate green; punch-list acceptance demo.
- **S3 -- Generator + CI gates:** `emit_snapshot`; `workflows/generated/*.gen.json` (stamped, `extra.info` banner); regenerate-diff; PARITY gate (all tiers); contract-on-every-artifact; cold-LOAD per tier; optional profile-application report artifact.
- **S4 -- Tier reality:** 8gb_lite overrides defined; simulated-8GB full-episode soak + peak assertion; cpu_floor smoke; per-tier determinism double-run (ship gate).
- **S5 -- Wizard + models:** CLI detect/confirm/record; generated manifests; downloads; launcher emission.
- **S6 -- README/newbie refresh:** fold with the already-tracked README sprint; document tiers, profiles, "never edit .gen.json", Mac=cpu_floor.
- Every sprint: full suite + Bug Bible regression + audio byte-identical + platform invariants (mux-LAST, single-heavy residency, fail-closed LOUD, validator contract).

## 14. Verify-at-build register (what only code can answer)
1. Exact widget names on node types of 92/81/82/83/3 + all `request_seed` instances (S0 mapping doc).
2. Image-lane reachability from node-87 `gate_in` (S0; else one forceInput gate on the dispatcher).
3. Cold-import reality across adapters (S1 blocking gate; fallback defined).
4. Saved-COMBO load behavior in lite envs (S3 cold-load).
5. Lever-1 VRAM-reclaim numbers under the dynamic ceiling (existing open probe; informs S4 floors).

## 15. Campaign record
Three passes, 4-model panel (GPT-5.5, Gemini 3.1 Pro, Grok 4.3, DeepSeek V4 Pro) + claude-fable-5 as independent panelist (review written before reading the panel) and sole judge. Architecture direction unchallenged in all three passes; pass01 reshaped seams (stamp, coverage, VRAM, applier), pass02 fixed the synthesis mechanics (split apply/emit, env-primary ceiling, node-63 path bug), pass03 refined implementation precision (gate wiring, env persistence, coverage algorithm, determinism scope) -- diminishing returns, converged. Total panel spend: **$0.78**. Raw reviews + per-pass judgment logs: `docs/2026-06-10-switchable-workflow-architecture/roundtable/`.
