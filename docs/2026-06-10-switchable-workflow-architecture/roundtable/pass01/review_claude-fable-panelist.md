<!-- panelist: claude-fable-5 (judge also serves as independent panelist; written BEFORE reading the OpenRouter panel reviews) -->
# Panelist Review -- Claude (independent position, pass01)

Verdict: the master-graph + generated-snapshots + profile-layer shape is RIGHT. But the plan as
written has one central unresolved seam and several must-fixes before it is build-ready.

## MUST-FIX 1 -- The profile is half widgets, half env, and only widgets serialize
The plan treats "set the switches and export" as one operation. Mechanically the switches are TWO
disjoint stores: Director widget values (live in `widgets_values`, DO serialize into a snapshot) and
`OTR_ENABLE_*` / `OTR_FORCE_ENGINE_MAP` env flags (do NOT exist in any JSON). A downloaded
`otr_8gb_lite.json` therefore does NOT fully determine behavior -- the env half is whatever the
user's shell happens to have. The "zero-friction click-Run snapshot" claim fails exactly there.
Fix: make the graph SELF-DESCRIBING. Add a single `OTR_Profile` node (or a profile widget on an
existing root node) whose value is the profile id + inline budget overrides; the registry/resolver
read profile FIRST, env flags become explicit OVERRIDES with fixed precedence:
`explicit env > in-graph profile > auto-detect`, every layer logged LOUD at queue time.
Then a snapshot carries its whole config, the validator can check it, and headless inherits it for
free. Without this, snapshots are half-configured and the drift bug returns through the env door.

## MUST-FIX 2 -- Startup profile-vs-reality assertion (fallback must not mask a wrong profile)
The resolver's per-shot fail-closed fallback is designed for transient/edge failures. If a user runs
the 16gb profile on an 8GB box, per-shot fallback will "work" -- every shot LOUDLY downgrades to the
floor engine and the user concludes OTR renders garbage. At graph load / first queue, assert
profile claims vs detected reality (VRAM, platform, toolchains present); on mismatch HARD-FAIL with
a one-key downshift suggestion ("detected 8GB; switch to 8gb_lite?"). Fallback handles shots;
the assertion handles systematically wrong profiles. These are different failure classes.

## MUST-FIX 3 -- Capability-based profile schema, not named bags of flags
If `profiles.json` is N named tiers x M hand-listed engine flags, every new engine edits every
profile and the file rots. Profiles should declare BUDGETS/capabilities only:
`{vram_budget_mb, platform, toolchains: [cu128,...], allow_sidecars, max_model_class}`.
Each engine already declares roles in the registry; extend engine metadata with
`vram_class` + `required_toolchain`. The ENABLE-SET IS DERIVED (registry filters engines by profile
budgets) -- new engines ship their own metadata and every profile picks them up automatically.
This reuses the existing registry grain instead of building a parallel config system.

## MUST-FIX 4 -- One applier function with three consumers, or the drift bug survives
The drift mechanism (verified) is hand-coded widget patch-lists inside headless scripts vs saved
defaults in the JSON. The redesign only kills it if there is EXACTLY ONE apply path:
`profile.apply(workflow_json | api_prompt)` in one module, consumed by (a) the snapshot generator,
(b) the headless submit path (queue_smoke/soak stop hard-coding patches), (c) the startup assertion.
Acceptance test that IS the drift-killer: for each profile P, `generated_snapshot(P) -> api_prompt`
and `master + apply(P) -> api_prompt` are IDENTICAL dicts (modulo seed/run-id). Plus the demo that
closes the original bug: captions + procgen credits + LTX radio open render identically from
UI-load and headless-submit on the 16gb profile.

## MUST-FIX 5 -- CI regeneration gate + snapshot stamping, or snapshots are the new drift
Users WILL hand-edit downloaded snapshots; the repo's committed copies must not rot. (a) CI job:
regenerate every tier from master -> byte-identical to committed snapshots, else fail. (b) The
generator stamps each snapshot with `generated_by + master_hash`; the loader/validator warns LOUD on
hash mismatch ("hand-edited or stale; regenerate"). (c) `OTR_WorkflowValidator` (today pointed at
the single canonical JSON) runs its widget-vector contract on EVERY shipped snapshot in CI.

## MUST-FIX 6 -- Define the 8GB floor as a rendered reality, not an exclusion list
"8GB excludes 14B" says what it is NOT. The profile must pin per-role default chains for the tier
(e.g. announcer=humo_1p7b@480, other-beats=ltx|still_kenburns, music=procgen) AND the tier gate must
be a real soak: drive `VRAM_CEILING_MB` from the profile and render a full episode under a simulated
8GB ceiling on the 5080 before the tier ships. A tier we never rendered is marketing, not a tier.

## SCOPE CALL -- MPS is OUT of v1; say it in the plan
Zero `torch.backends.mps` references exist; there is no centralized device-choice module to hang it
on; the cu128 sidecars are definitionally non-Mac. Honest v1 matrix: `16gb-full`, `8gb-lite`,
`cpu-floor` (and "Mac" ships AS cpu-floor, labeled). MPS routing is its own later sprint with probe
gates (device plumbing through every adapter). Otherwise the coder window burns a week on Metal in
the middle of an architecture sprint.

## SMALLER ITEMS
- Determinism key becomes `(profile_id, seed)`, not seed alone; ledger records profile_id +
  master_hash + snapshot_hash. Cheap; kills a class of "my render changed" reports.
- UX vocabulary: <=4 named tiers + `auto` (resolves to one of them) + `custom` (any manual deviation,
  stamped in the ledger as such for support triage). Director dropdowns remain the power-user layer;
  profiles just SET them.
- Wizard (Decision B) is a one-shot setup tool OUTSIDE the render path: detect -> propose -> confirm
  -> write profile + emit launcher + offer per-tier manifest download (resumable, checksum,
  honors HF_HOME/extra_model_paths.yaml, confirm-before-dump). CLI first is right; Gradio later.
- Generator input is the MASTER; tier files live in `workflows/generated/` (or carry a `.gen.json`
  suffix) so "never hand-edit" is visible in the filename.

## Draft sprint skeleton (for the coder window, after convergence)
- S0: profile schema + loader + `OTR_Profile` in-graph carrier + precedence + startup assertion.
- S1: registry capability metadata (`vram_class`, `required_toolchain`) + derived enable-set.
- S2: single applier module + convert headless scripts to consume it (delete hard-coded patch
  lists) + snapshot==apply parity test + punch-list acceptance demo.
- S3: snapshot generator + CI regen-diff gate + stamping + validator-over-all-snapshots.
- S4: 8gb-lite floor definition + ceiling-simulated full-episode soak; cpu-floor smoke.
- S5: setup wizard CLI (detect/confirm/write/manifest download/launcher).
- S6: README/newbie refresh (already tracked as the OpenRouter-plan S6; fold).
- Parked: MPS lane (own sprint + probes), Gradio UI, h264_videotoolbox polish.
Each sprint: full suite green + Bug Bible regression + audio byte-identical invariant intact.
