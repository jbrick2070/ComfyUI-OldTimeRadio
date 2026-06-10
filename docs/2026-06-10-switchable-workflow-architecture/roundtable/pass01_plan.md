# Switchable Workflow Architecture -- synthesized plan (pass01, judge: claude-fable-5)
*2026-06-10. Status: PLANNING ONLY -- no code in this phase. Supersedes the working recommendation in the problem statement.*

## Decision A (settled): one master graph, generated tier artifacts
- **Author/maintain exactly ONE master graph**: `workflows/otr_scifi_16gb_full.json` keeps its name and role (it is already the only production JSON; renaming breaks scripts and muscle memory). It is the single source of truth for STRUCTURE.
- **Distribute GENERATED per-tier snapshots** under `workflows/generated/` (`otr_16gb_full.gen.json`, `otr_8gb_lite.gen.json`, `otr_cpu_floor.gen.json`). `.gen.json` suffix = "never hand-edit" visible in the filename. Users click-Run a snapshot; they never rewire.
- **Self-check loop:** CI asserts `apply(master, 16gb_full) == master` byte-identical. The master must always be dialed to its own reference profile; any master edit that diverges fails CI unless the 16gb profile changes in the same commit. The master cannot drift from its own profile.

## The profile system (the new core)
**A profile is data, not a graph.** Committed at `config/profiles/<id>.json`, schema-validated, unknown keys rejected.

Schema (minimal, capability-based -- NOT hand-listed engine flags):
```
{ "id": "8gb_lite", "display_name": "...", "platform": "any|win|mac",
  "device_backend": "cuda|cpu",            // mps deliberately absent in v1
  "vram_budget_mb": 7300,                   // drives the runtime ceiling, see VRAM below
  "toolchains": ["cu128"]|[],               // what may run; sidecar availability
  "allow_sidecars": true|false,
  "max_model_class": "heavy|medium|light",
  "role_defaults": { "announcer_video": ["humo_1p7b", "still_kenburns"], ... },   // per-role chains
  "slot_defaults": { "char_voice": "indextts2", "announcer_voice": "kokoro",
                      "theme_music": "stable_audio_3", "render_batch_engine": "humo" },
  "features": { "burn_captions": true, "procgen_credits": true, "ltx_radio_open": true },
  "seed_policy": { "request_seed": 42, "seed_mode": "request_hash",
                    "cast_seed_env": null, "style_seed_env": null },
  "launch": { "sage_attention": false, "extra_args": [] } }
```
- **The enable-set is DERIVED, never listed.** The engine registries gain per-engine capability metadata (`vram_class`/`vram_estimate_mb`, `required_toolchain`, `model_requirements` = HF repo/path/revision/size). Profile validator computes: enabled = engines whose requirements fit the profile budgets/toolchains. A new engine ships its own metadata; every profile picks it up with zero profile edits. The registry stays the single place engines describe themselves.
- **Profile validator hard-rejects impossible profiles** (engine in a role_default that needs cu128 when toolchains=[]; vram_class heavy on 8gb; two heavies co-resident; unknown keys).

## How a profile reaches the graph: APPLY-time, not render-time plumbing
Profiles flow through ONE applier into ORDINARY widgets + env. No new data wiring through the graph.
- **`apply(workflow_or_api_prompt, profile)`** lives in ONE shared module (promote the patch-by-NAME machinery from `scripts/otr_api.py` into the package so scripts, generator, and tests import one implementation). Widget schemas come from direct `NODE_CLASS_MAPPINGS` import (the validator already does this in tests) -- NO live ComfyUI server required for the build step; the live `/object_info` cross-check stays available in the soak lane.
- **Profile-managed widget map covers ALL engine-bearing widgets, not just Directors** (verified in the real JSON): Director per-role dropdowns, node 81 `OTR_BatchCharacterVoices` ("indextts2"), node 82 `OTR_AnnouncerVoice` ("kokoo"->"kokoro"), node 83 `OTR_StableAudioTheme` ("stable_audio_3"), node 92 `OTR_VideoRenderBatch` engine default ("humo" -- verify exact widget name at build), node 63 `OTR_WorkflowValidator` artifact path, feature widgets (captions burn etc.), `request_seed` (exactly that name -- it deliberately avoids the `control_after_generate` companion; never patch a widget named `seed`).
- **The graph carries a profile STAMP, not profile data.** The generator writes `profile_id` + `master_hash` + `generated_by` into the snapshot (node-63 widget + node properties). At load/first-queue, a startup assertion resolves the stamp against `config/profiles/<id>.json` and against DETECTED reality (VRAM, platform, toolchains): mismatch = HARD STOP with a one-key downshift suggestion ("detected 8GB; use 8gb_lite"). Per-shot fallback stays for transient failures; the assertion catches systematically wrong profiles. Different failure classes, different mechanisms.
- **Precedence (selection):** `OTR_FORCE_ENGINE_MAP` (dev override, unchanged) > saved Director/slot widgets (= profile-applied values) > resolver fallback chains.
  **Precedence (availability):** registry reality (model present, toolchain present) INTERSECT `OTR_ENABLE_*` env (dark-ship/override gates, default behavior unchanged) INTERSECT profile-derived enable-set. One LOUD log line at queue start prints the resolved profile, source, and every override in effect.

## Validator + node 63 (confirmed defect, must fix)
Node 63 ships `widgets_values: ["", true, ...]`; empty path falls back to the canonical 16gb JSON -- a generated snapshot would self-validate the WRONG file. Fix: generator sets node 63's path to the artifact's own repo-relative path (portable resolution), and CI runs the widget-vector contract on EVERY `.gen.json`, not just the master. The build fails if any saved COMBO value is absent from the target schema.

## VRAM safety becomes profile-driven
`VRAM_CEILING_MB = 14500` is a hardcoded constant in `wrapper_bridge.py:37`; an 8gb tier with a 14.5GB allowance OOMs at the CUDA level. Fix: `wrapper_bridge` reads `OTR_VRAM_CEILING_MB` env override, else resolves the in-graph profile stamp -> committed profile `vram_budget_mb`, else falls back to 14500. The 8gb-lite gate is a FULL-EPISODE soak on the 5080 under the simulated 8GB ceiling with peak-VRAM assertion -- a tier we never rendered is marketing, not a tier.

## Headless = same applier, restricted patch surface (the actual drift kill)
- `queue_smoke.py` / soak runners take `--profile <id>`, call the SAME `apply()`, and DELETE their hard-coded engine/feature patch lists.
- Ad-hoc patching after `apply()` is restricted to a WHITELIST of creative request inputs (target_words, seeds, prompt fields). The applier API enforces it: patching a profile-managed widget outside `apply()` raises. A regression test greps/imports soak scripts to prove no banned patches.
- **Acceptance demo that closes the original punch-list bug:** captions + procgen credits + LTX radio open render identically from UI-load (snapshot) and headless-submit (master+apply) on the 16gb profile; plus dict-equality parity: `to_api_prompt(snapshot(P)) == to_api_prompt(apply(master,P))` for every P (modulo seed/run-id).

## v1 tier set (honest matrix)
`16gb_full`, `8gb_lite`, `cpu_floor`. Mac SHIPS AS `cpu_floor` and is labeled exactly that (zero `torch.backends.mps` references exist; ~35 scattered cuda/platform checks; there is no device-routing module to hang MPS on). **MPS is a PARKED follow-on sprint**: centralized `device_routing.py` + per-adapter plumbing + its own probe gates. No "Mac-MPS" name appears anywhere in v1 artifacts. ffmpeg is already Mac-safe (libx264 everywhere, mux `-c:v copy`, nvenc gated behind `_check_nvenc`).

## Determinism
Contract becomes `(profile_id, seed) -> identical bytes` per tier. Ledger records `profile_id + master_hash + snapshot_hash`. Profile carries the seed policy (`request_seed`, `seed_mode`, optional `OTR_CAST_SEED`/`OTR_STYLE_SEED` env policy -- creative RNGs keep OS entropy by default per the platform invariant). Per-tier double-run compare is a shipped gate.

## Decision B (after the core, unchanged in spirit, slimmed)
CLI wizard (one-shot setup tool, OUTSIDE the render path): detect hardware -> propose tier -> confirm -> record profile choice -> optional model download -> emit launcher (sets `OTR_VRAM_CEILING_MB`, optional sage launch args). Model downloads: the per-profile manifest is GENERATED from registry `model_requirements` of the enabled set (never hand-written); `huggingface_hub` natively honors `HF_HOME` (no custom parsing); destination honors `extra_model_paths.yaml`; size-confirm before download; resumable + skip-existing.

## CUT from this build (explicit)
- Gradio/Streamlit UI (CLI wizard only).
- `h264_videotoolbox` fast path (separate perf ticket; libx264 is the Mac baseline).
- MPS routing (parked sprint, above).
- `character_3d` as a v1 capability switch (3D is platform-deferred; the profile design must not block it later, nothing more).
- Full cartesian switch testing (fixed tier gates only).

## Draft sprint plan (DRAFT -- operator has not green-lit coding)
- **S0 Profile foundation:** schema + `config/profiles/` (3 tiers) + profile validator + precedence spec + stamp format. Pure data/validation; no graph change.
- **S1 Registry capability metadata:** `vram_class`/`vram_estimate_mb`, `required_toolchain`, `model_requirements` per engine; derived enable-set; profile cross-validation; dynamic `OTR_VRAM_CEILING_MB` in `wrapper_bridge` (env > stamp > 14500).
- **S2 One applier, headless converted:** promote `otr_api.py` patch machinery to a shared module; profile-managed widget map (Directors + 81/82/83 + 92 + 63 + features + request_seed); `queue_smoke`/soaks consume `--profile`, hard-coded patch lists deleted; creative-input whitelist enforced; parity test + punch-list acceptance demo (this sprint closes the original drift bug).
- **S3 Generator + CI gates:** emit `workflows/generated/*.gen.json` (stamped, node-63 path fixed); regenerate-diff CI gate incl. `apply(master,16gb)==master`; validator on every artifact; cold-load per target env; cold-import test without CUDA deps.
- **S4 Tier reality:** 8gb_lite role/slot floor defined; full-episode soak under simulated 8GB ceiling with peak assertion; cpu_floor smoke; per-tier determinism double-runs.
- **S5 Wizard + models:** CLI detect/confirm/record; generated per-profile manifest; downloads (HF_HOME/extra_model_paths honored, confirm-before-dump); launcher emission.
- **S6 README/newbie refresh:** fold with the already-tracked README S6; document tiers, profiles, "never edit .gen.json".
- Every sprint: full suite green + Bug Bible regression + audio byte-identical invariant + no invariant regressions (mux-LAST, single-heavy residency, fail-closed LOUD).

## Verify-at-build notes (UNVERIFIABLE from code today)
- Node 92's engine-default widget exact name/semantics (GPT flagged its own assumption).
- Whether every adapter module cold-imports cleanly without CUDA sidecars present (needed for cpu_floor; test in S3).
- Load behavior of saved COMBO values not present in a lite env's registry roster (document + test in S3; generator already constrained to floor-guaranteed values).
- Lever-1 VRAM-reclaim numbers under the dynamic ceiling (existing open probe; informs S4 floor choices).
