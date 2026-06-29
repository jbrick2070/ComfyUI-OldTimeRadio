# Delete the opt-in flag -- HARDENED PLAN (kibitz r1 synthesis)

Panel: Codex (gpt-5.5/high) + Antigravity (gemini) + Claude anchor/judge, all
grounded against the real tree. Both panelists converged; findings below are
CONFIRMED against the code.

## The key reframe (r1 lead finding)
`requires_flag` is NOT only an opt-in "promotion". For the DARK SCAFFOLDS
(`triposr`, and `triposg_talk`/`hunyuan3d_talk`/`trellis_talk` in
`eng_character_3d.py`) it is a genuine "not implemented yet" guard -- their
`render_clip`/`prepare`/`canonicalize` raise `NotImplementedError`. Blanket-
deleting the flag would make "select it -> crash", NOT "select it -> renders".
=> The deletion applies ONLY to RENDER-READY (validated) engines; dark scaffolds
must NOT have their gate removed (leave gated, or unregister -- DECISION 1).

## Confirmed gate inventory (per engine -- do not blanket)
- Default-ON (never gated; no change): `visualizer`, `ltx_video`, `ltx_av`,
  `flux_gen1` (non-empty `default_roles`).
- Default-OFF + flag-gated, RENDER-READY (REMOVE the flag gate -> just render,
  files-on-disk only): `humo`, `wan_i2v`, `wan_ti2v`, `still_parallax`,
  `flux2_klein`, `z_image_turbo`. (These are the operator's smoke-list opt-ins.)
- Default-OFF + flag-gated, DARK scaffold (NOT render-ready; do NOT un-gate):
  `triposr`, `triposg_talk`, `hunyuan3d_talk`, `trellis_talk`, (verify
  `mesh_stage`). Already excluded from the dropdown by
  `test_tested_only_dropdown_gate.py`.
- Image adapters `flux2_klein`/`z_image_turbo` do NOT re-check the flag in their
  adapter `assert_usable` (only the registry gate + model-path check) -- so for
  them, removing the BASE registry gate is sufficient.

## The hardened change
0. Remove the interim option-B entirely: `apply_selection_enable_set` /
   `_restore_enable_set` / the `run_real_episode` try/finally, AND DELETE
   `tests/test_video_selection_enable_set.py` (obsolete, not revert). Revert
   commit 1c73aec's behaviour.
1. Base `engine_registry_base.py` `assert_usable`: remove the `GATED_BY_FLAG`
   block (L222-228) so a registered, role-fitting, RENDER-READY engine is usable.
   KEEP the `GATED_BY_FLAG` enum MEMBER (dead) -- `test_protocol_parity`
   (`test_video_platform_aseam.py`) + `test_image_protocol_parity`
   (`test_image_platform_c1.py`) assert the shared enum equals audio's frozen
   copy which retains it.
2. Render-ready adapters (`humo`, `wan_i2v`, `wan_ti2v`, `still_parallax`):
   delete the `requires_flag` check from `assert_usable`; KEEP the ckpt/dep-on-
   disk + genuine capability checks (wan_ti2v 2.2-VAE, etc.).
3. Dark scaffolds: leave the gate (DECISION 1a) OR unregister them (DECISION 1b).
4. `requires_flag` field: keep on `EngineCore` as OPTIONAL (still used by dark
   scaffolds if 1a, and by the dep-pilot manifest); set to `None` on every
   render-ready engine row. (Full field deletion only if DECISION 1b + no manifest
   use -- defer; it is cleanup, not behaviour.)
5. Harness (do NOT infer from CAPABILITIES -- too weak; both panelists):
   - `otr_video_gpu_smoke.py`: remove the `flag`/`flag_set` ready-assertion
     (L168-170, L209) so a no-flag engine is not reported NOT READY.
   - `otr_coverage_sweep.py`: delete the `OTR_ENABLE_WAN_*` `acceptance_preflight`
     check (L121-132) so acceptance does not exit 2.
   - `otr_video_dep_pilot.py` / `otr_image_dep_pilot.py`: KEEP the static
     `OPT_IN_ENGINES`/probe manifest (module/class/forward metadata the CAPABILITIES
     table does NOT carry); just delete the dead `flag` keys.
6. Curation = the dropdown (`validated_engine_names()`) stays the only "is this
   model good+tested" gate. DECISION 2: the force-map / raw-JSON / lsync-base /
   custom paths can still name a NON-validated engine -- accept as a documented
   DEV escape hatch, or add a production "validated-only unless dev mode" guard.
7. Tests by contract family (not "~20"): registry usability, adapter disk gates,
   dropdown curation, force-map/lsync routing, dep-pilot, gpu-smoke, coverage
   acceptance, protocol parity. Update each to the no-flag contract.

## Invariants preserved
Files-on-disk still enforced (LOUD MISSING_MODEL); no-fallback render still raises
on real failure; AUDIO spine frozen + untouched (own registry + own enum copy);
no workflow-JSON/widget change (V-11); determinism + 14.5GB + LOUD unchanged.

## Two decisions for the operator
1. DARK scaffolds (triposr + the 3 talkers; NOT in your smoke list): (a) leave
   registered-but-gated as parked future lanes [minimal], or (b) unregister now
   [cleanest; matches "remove untested before release"; they return when built].
2. DEV backdoor (force-map/raw-JSON can name a non-validated engine): (a) accept
   as a documented developer escape hatch, or (b) add a production guard that
   refuses non-validated engines outside an explicit dev mode.

## Sequencing (suite green per chunk)
C1 revert option-B + delete its test. C2 base gate + render-ready adapters +
their tests. C3 harness (gpu-smoke/coverage/dep-pilot) + their tests. C4 dark-
scaffold decision + curation guard (per DECISIONS). Commit per green chunk.
