# Switchable Workflow Architecture -- synthesized plan v3 (pass02, judge: claude-fable-5)
*2026-06-10. Status: PLANNING ONLY -- no code in this phase. Supersedes pass01_plan.md.*

## Decision A (settled, unchallenged 2 passes): one master graph, generated tier artifacts
- ONE master graph: `workflows/otr_scifi_16gb_full.json` keeps its name and role; single source of truth for STRUCTURE. The master is the DEV/reference artifact and stays UNSTAMPED; end users run snapshots.
- GENERATED per-tier snapshots in `workflows/generated/`: `otr_16gb_full.gen.json`, `otr_8gb_lite.gen.json`, `otr_cpu_floor.gen.json`. `.gen.json` = never hand-edit.
- Two split operations (NOT one):
  - `apply_profile(workflow, profile)` -- pure semantic widget patching. Never stamps, never touches paths.
  - `emit_snapshot(applied, artifact_path)` -- writes node-63 artifact path + the profile STAMP. Generator-only.
- **Identity gate (CI):** `to_api_prompt(master) == to_api_prompt(apply_profile(master, profile_16gb))` as dict-equality (UI state excluded by conversion; no byte comparison). BOOTSTRAP: the initial 16gb profile is EXTRACTED from the master's current saved values, so the gate is green on day one and any later master/profile divergence fails CI unless both change together.
- **Parity gate (CI, once generator exists):** `to_api_prompt(snapshot(P)) == to_api_prompt(apply_profile(master, P))` for every P, IGNORING exactly: node-63 `workflow_json_path` + the three stamp fields. Seeds are profile-fixed so they match by construction.

## The profile system
Committed at `config/profiles/<id>.json`; schema-validated; unknown keys rejected. Profiles are capability POLICY, not creative presets, and carry OVERRIDES only -- the registries' `default_engine_for_role` supplies the base.
```
{ "id": "8gb_lite", "display_name": "8 GB NVIDIA (lite)", "platform": "any|win|mac",
  "device_backend": "cuda|cpu",              // mps deliberately absent in v1
  "vram_budget_mb": 7300,                     // -> runtime ceiling (see VRAM)
  "toolchains": ["cu128"] | [],
  "allow_sidecars": false,                    // false => sidecar-requiring engines excluded from enable-set
  "max_model_class": "light",                 // caps engine vram_class in the derived enable-set
  "role_overrides": { },                      // optional; per-role chains only where deviating from registry defaults
  "slot_overrides": { },                      // optional; audio/video slot widgets (nodes 81/82/83/92, node 3 voice default)
  "features": { "burn_captions": true, "procgen_credits": true, "ltx_radio_open": false },
  "seed_policy": { "request_seed": 42, "seed_mode": "request_hash",
                    "cast_seed_env": null, "style_seed_env": null },
  "launch": { "sage_attention": false, "extra_args": [] } }
```
- **Derived enable-set, never hand-listed:** registries gain DEP-FREE per-engine capability declarations (`vram_class`/`vram_estimate_mb`, `required_toolchain`, `requires_sidecar`, `model_requirements` = HF repo/path/revision/size) living in the registry table modules, NOT in adapter modules that import heavy deps. enabled(P) = engines whose declarations fit P's budgets/toolchains/sidecars/class-cap.
- **Profile validator:** shape + unknown-key rejection (S0); capability cross-checks (S1): every role/slot override must be in enabled(P); per-engine fit only -- NO static co-residency rejection (residency is wrapper_bridge's runtime lifecycle invariant; the master legitimately saves several heavy choices across roles).

## How a profile reaches the graph (corrected mechanics)
- Profile data flows at APPLY time into ordinary widgets via ONE shared module (the `scripts/otr_api.py` patch-by-NAME machinery PROMOTED into the package). Schemas for offline use come from direct `NODE_CLASS_MAPPINGS` import through an explicit **INPUT_TYPES -> /object_info shape adapter** (tuples->lists), byte-compatible with the live schema path; the live `/object_info` cross-check remains in the soak lane.
- **The STAMP is three new optional STRING widgets on `OTR_WorkflowValidator`** (`profile_id`, `master_hash`, `generated_by`; defaults ""), added to INPUT_TYPES with the widget-vector contract tests updated in the same change. No node `properties` are used (not executable in API prompts). Master: stamp widgets empty; snapshots: filled by `emit_snapshot`.
- **Startup assertion (named home):** `OTR_WorkflowValidator.execute` -- ACTIVE ONLY when `profile_id` is non-empty. Resolves the stamp against `config/profiles/<id>.json` + DETECTED reality (VRAM, platform, toolchains): mismatch -> raise (prompt aborts) with a one-key downshift suggestion ("detected 8GB; use otr_8gb_lite.gen.json"). It also EXPORTS the resolved `vram_budget_mb` into `os.environ["OTR_VRAM_CEILING_MB"]` (if not already set) before video dispatch.
- **Ordering guarantee:** wire node 63's existing `validation_report` STRING output into a NEW optional input on node 92 (`OTR_VideoRenderBatch`) in the master -- a one-time structural edit that makes the assertion + env export topologically precede heavy video work and survives API-prompt conversion. Accepted gap: independent audio nodes may run before an assertion failure aborts the prompt -- LOUD and cheap; documented.
- **Precedence (selection):** `OTR_FORCE_ENGINE_MAP` > saved widgets (= profile-applied) > resolver fallback chains. FORCE is a dev bypass of the PROFILE enable-set but NEVER of registry reality (cannot force an engine whose models/toolchain are absent); forcing outside the profile logs a LOUD warning. Tests: parse-error case + outside-profile case.
- **Precedence (availability):** registry reality INTERSECT `OTR_ENABLE_*` env (dark-ship gates, behavior unchanged) INTERSECT enabled(P). One LOUD line at queue start: resolved profile, source, every override in effect.

## Widget coverage (closed by test, not by list)
A schema-driven COVERAGE TEST enumerates every COMBO/STRING widget (via NODE_CLASS_MAPPINGS) whose saved value is a registered engine id; each must be either PROFILE-MANAGED -- keyed `(node_type, widget_name)` with a unique-match assertion (no raw node ids) -- or EXEMPTED with a written reason. Initial classification from the real JSON:
- MANAGED: Director per-role dropdowns (nodes 19/88 types), `OTR_BatchCharacterVoices` (81, "indextts2"), `OTR_AnnouncerVoice` (82, "kokoro"), `OTR_StableAudioTheme` (83, "stable_audio_3"), `OTR_VideoRenderBatch` engine default (92, "humo"), `OTR_SceneSequencer` voice default (3, "bark"), feature widgets (captions etc.), node-63 path+stamp (via emit_snapshot only).
- EXEMPT (creative/user choice, headless-whitelisted): `OTR_LedgerScriptWriter` (1) writer model slots (`openrouter_slot_*`/`comfy_slot_*` -- their dynamic values stay admissible via the existing `_is_openrouter_admissible`/`_is_comfy_admissible` paths, which the CI COMBO check reuses).
- SEED-POLICY: `request_seed` (exactly that name; never patch a widget named `seed` -- companion-slot trap).
- Exact widget NAMES for 92/81/82/83/3 are read from INPUT_TYPES in S0 and committed as a mapping doc (profile key -> node_type.widget_name).

## Validator + node 63 fixes (verified defects)
- `_load_workflow` resolves non-empty relative paths against process CWD (`p = Path(path)`, line ~69): FIX to resolve non-absolute against `_REPO_ROOT`. Generator then writes repo-relative artifact paths portably.
- CI runs the widget-vector contract on EVERY `.gen.json`; build fails on any saved COMBO value absent from the target schema (dynamic slot values via the admissibility paths above).

## VRAM safety (simplified, profile-driven)
`wrapper_bridge` reads `OTR_VRAM_CEILING_MB` env override ELSE 14500. NO graph introspection in wrapper_bridge. Sources of the env: headless/launcher set it from the profile; the UI path gets it from the stamped validator's export (above). The 8gb_lite SHIP GATE is a full-episode soak on the 5080 under the simulated 8GB ceiling with a peak-VRAM assertion.

## Headless = same applier (the drift kill)
- `queue_smoke.py` + soak runners take `--profile <id>`, call `apply_profile`, and DELETE hard-coded engine/feature patch lists -- migrated in the SAME sprint, including their remaining creative patches onto the whitelist.
- WHITELIST (exact): `target_words, num_characters, act_count, request_seed` + seed-policy fields, prompt/title text fields, `openrouter_slot_*`/`comfy_slot_*`.
- Enforcement, both stateless: `patch_creative()` pure function validates names against the whitelist; one regression test asserts soak scripts contain no direct `patch_widget_by_name` on profile-managed names.
- **Acceptance demo (closes the original punch-list bug):** captions + procgen credits + LTX radio open render identically from UI-load (snapshot) and headless-submit (master+apply) on the 16gb profile.

## v1 tier set (honest matrix)
`16gb_full`, `8gb_lite`, `cpu_floor`. Mac SHIPS AS `cpu_floor`, labeled exactly that. MPS is a PARKED follow-on sprint (centralized `device_routing.py` + per-adapter plumbing + probes; ~35 scattered cuda/platform checks exist today, zero mps references). No "Mac-MPS" name in any v1 artifact. ffmpeg already Mac-safe (libx264 everywhere; mux `-c:v copy`; nvenc behind `_check_nvenc`). `cpu_floor` artifact generation is BLOCKED on the cold-import gate (below).

## Determinism
Contract: `(profile_id, seed) -> normalized-identical outputs` per tier: audio byte-identical (existing machinery), video compared by stream hash with container metadata stripped, ledger fields compared. Ledger records `profile_id + master_hash + snapshot_hash`. This is a TIER-SHIP gate (run at tier release), not per-commit CI. Creative RNGs keep OS entropy by default (platform invariant); `OTR_CAST_SEED`/`OTR_STYLE_SEED` policy lives in the profile.

## Decision B (after the core, slimmed)
CLI wizard, one-shot setup tool OUTSIDE the render path: detect -> propose tier -> confirm -> record choice -> optional model download -> emit launcher (sets `OTR_VRAM_CEILING_MB`, optional sage args). Per-profile model manifest is GENERATED from registry `model_requirements` of enabled(P); `huggingface_hub` natively honors `HF_HOME` (no custom parsing); destination honors `extra_model_paths.yaml`; size-confirm before download; resumable + skip-existing.

## CUT from this build (explicit)
Gradio/Streamlit; `h264_videotoolbox` (separate perf ticket); MPS routing (parked); `character_3d` as a v1 switch (design must not block it later, nothing more); full cartesian testing; byte-identical master self-check (replaced by api-prompt dict equality); static co-residency rejection; per-commit determinism CI; duplicate stamp homes.

## Draft sprint plan (DRAFT -- operator has not green-lit coding)
- **S0 Profile foundation (no graph change):** schema + `config/profiles/` (3 tiers; 16gb EXTRACTED from master) + shape-only validator + precedence spec + stamp field format + the widget-name MAPPING DOC read from real INPUT_TYPES (closes the 92/81/82/83/3 name assumptions).
- **S1 Registry metadata + early gates:** dep-free capability declarations; derived enable-set; capability cross-validation; dynamic ceiling in wrapper_bridge (`env > 14500`); **BLOCKING cold-import gate** (all node packages import without CUDA/sidecar deps -- the generator depends on it; cpu_floor depends on it).
- **S2 One applier + headless conversion:** promote otr_api machinery + INPUT_TYPES->object_info adapter; `apply_profile`; coverage test; `_load_workflow` repo-root fix; node-63 stamp widgets + widget-vector test update; assertion + env export in validator execute; node-92 gate input wired in master; queue_smoke/soaks on `--profile` (hard-coded lists deleted, whitelist enforced); IDENTITY gate green; punch-list acceptance demo. *This sprint closes the original drift bug.*
- **S3 Generator + CI gates:** `emit_snapshot`; `workflows/generated/*.gen.json`; regenerate-diff CI; PARITY gate; validator-on-every-artifact; per-tier cold-LOAD test (saved COMBO values vs target env roster).
- **S4 Tier reality:** 8gb_lite overrides defined; full-episode simulated-8GB soak + peak assertion; cpu_floor smoke; per-tier determinism double-run (ship gate).
- **S5 Wizard + models:** CLI detect/confirm/record; generated manifests; downloads; launcher emission.
- **S6 README/newbie refresh:** fold with the already-tracked README sprint; document tiers, profiles, "never edit .gen.json", Mac=cpu_floor.
- Every sprint: full suite + Bug Bible regression + audio byte-identical + invariants (mux-LAST, single-heavy residency, fail-closed LOUD, validator contract).

## Verify-at-build notes
- Exact widget names on nodes 92/81/82/83/3 (S0 mapping doc resolves; flagged as assumptions by two panelists).
- Cold-import reality across all adapters (S1 blocking gate exists precisely because this is unverified).
- Saved-COMBO load behavior in lite envs (S3 cold-load test).
- Lever-1 VRAM-reclaim numbers under dynamic ceiling (existing open probe; informs S4).
