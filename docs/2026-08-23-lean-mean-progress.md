# LEAN-MEAN CLEANUP -- where it actually stands, 2026-08-23

Read `docs/LEAN_MEAN_CLEANUP.md` for the campaign itself. This file is the
honest progress line: what landed, what did not, and what needs the operator
rather than another window.

## DONE AND PUSHED -- orders 1 and 2 are COMPLETE

Seven commits, every one lockstep-verified (HEAD == origin, no 0-byte, no BOM,
AST parse on every touched `.py`, pack still registers 34/34 nodes).

| commit | what |
|---|---|
| `140b748d` | **Order 1, truth and prevention.** Three false README claims; two bare excepts; the credits double-parse; six dead symbols; the AST shadow-guard. |
| `b11a4269` | The preflight gate that could never pass nine profiles. |
| `8f371575` | QA follow-up: replaced a tautological test of my own; disarmed a vocabulary trap. |
| `cb73fd2e` | **Order 2** -- `_otr_probe.py`, `_otr_shared/sidecar.py`. |
| `14074ff2` | **Order 2** -- the cast prototype stack (728 lines + 3 test files + `_build_cast_rows`). |
| `8a5dbff4` | **Order 2 completes** -- `otr_shot_duration_calculator.py`, `scripts/normalize_workflow_widgets.py`. |
| `92b9edea` | Records the operator's consolidation ruling (below). |

Suite went 12251 -> 12270 -> 12271 -> 12210 -> 12183, each green with EXIT=0.
The drops are deleted test files, counted and named in their commits; no test
was lost to a failure.

**Two things found by accident that were worth more than the deletions:**

1. **The README promised users a safety net that does not exist.** Three places
   said a missing or OOMing engine falls back to a "guaranteed CRT floor" and
   never aborts. `render_driver.py:55-58` says that machinery was RIPPED and
   engine failure is a LOUD RenderError. The pack is on the Comfy Registry, so
   this was shipping to strangers.
2. **A failed VRAM eviction logged identically to a successful one**
   (`_otr_model_loader.py`). On a 16 GB card that is the usual cause of an OOM
   several stages later, and it was undiagnosable by construction.

## CANCELLED BY THE OPERATOR -- order 7

> *"Yeah every video lane is independent"* / *"Dont consolidate"* (2026-08-23)

Engine-to-engine duplication here is an ARCHITECTURAL CHOICE, not debt. Both
matrix rows and the order-7 line in `LEAN_MEAN_CLEANUP.md` are struck.

The `_role_of` consolidation had already been BUILT when the ruling landed --
six byte-identical copies to one import, full suite green, cold-import invariant
measured intact -- and was reverted unpushed. The viz-helper half was analysed
and never built; for the record, `_ref_path` and `_canvas_dims` are one
implementation across all four `viz_*` engines and `_build_render_request`
differs only by a docstring. **They are consolidatable and they are still not
going to be consolidated.** A duplicate-looking helper is a finding to report.

## NOT DONE, AND WHY -- orders 3 through 6

**These were deliberately not attempted overnight.** The operator's gate is
*"as long as episodes still render"*, and these are the orders that could break
exactly that.

* **Order 3 -- move unknown-engine rejection to the post-freeze VideoDirector
  boundary.** A live fail-closed guard (`otr_image_director.py:110-127`) is
  hidden inside the dormant 3D family that order 4 removes, so this must land
  first. It is a real behavioural change to a guard, not a deletion.
* **Order 4 -- retire the dormant 3D / dark engine family.** MEASURED before
  deciding not to do it: **156 references across 39 files** (nodes 50, tests 91,
  scripts 15) for `eng_character_3d` / `eng_still_parallax` / `eng_triposr` and
  the `character_3d` family vocabulary. The plan also requires rebasing the
  synthetic character-3D OOM / no-fallback proof in `render_driver.py` and
  `scripts/otr_video_soak.py` onto a live heavy engine BEFORE those branches are
  cut -- which is a GPU exercise with design content in it, not a sweep.
* **Order 5 -- retire ProjectStateLoader / SaveToEpisodeWorkspace / VideoProbe.**
  These are REGISTERED PUBLIC node IDs. Retiring them needs validator tombstones
  and a decision about `OTR_SaveToEpisodeWorkspace`, which has a known live
  consumer at `C:\Users\jeffr\Documents\ComfyUI\_otr_full_api.json:261`.
  **VRAMGuardian is explicitly an operator call** (retire+tombstone / debug-gate
  / rewrite onto targeted levers) -- the plan forbids silently keeping OR
  ripping it.
* **Order 6 -- the five-node `visual/` POC tree.** Blocked on moving the Hugging
  Face token startup behaviour first; that is the one live dependency that makes
  a blind tree deletion unsafe.

Orders 8-12 sit behind these and include the workflow-atomic Writer widget
schema epoch (order 10), which is a separate workflow epoch by design.

## WHAT NEEDS THE OPERATOR, not another window

1. **`OTR_SaveToEpisodeWorkspace`** -- migrate or explicitly retire the known
   saved artifact at `_otr_full_api.json:261`?
2. **`OTR_VRAMGuardian`** -- retire and tombstone, debug-gate, or rewrite onto
   targeted levers?
3. **`OTR_VideoProbe`** -- accept the loss of the public diagnostic, or build a
   replacement first?
4. **The historically alpha-public dark engine IDs** -- do they need named
   `RETIRED_ENGINE_IDS` compatibility, or is generic unknown-engine rejection
   enough? The plan says do not assume either answer.

## STILL OPEN FROM EARLIER, unchanged

* `docs/2026-08-22-variant-drift-DEFERRED.md` -- `build_variants.py --check`
  reports 54 variants / **2 failures**, both on `otr_ghost_signal_v3`, the SHIP
  CANDIDATE. Operator said `defer`. **Do not run `build_variants.py` without
  `--profiles`**: a blanket regeneration would revert the hand-edited
  ship-candidate settings.
* `docs/2026-08-22-negative-channel-declaration/driver_anchor.md` (untracked;
  `docs/2026-*/` is gitignored by convention) -- the item-D analysis, parked.
