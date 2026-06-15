# Coder-Window Kickoff — Cloud Engines (Flux Pro, LTX-2, Kling Avatar)

Paste this into a fresh CODER window. It is the build baton for adding three
opt-in Comfy-Cloud engines to OTR. A planner window produced the specs; your job
is the code.

## Read first (in this order)
1. `docs/2026-06-14-comfy-cloud-image-video/S0_SPIKE_AND_SPRINT.md` — the seam,
   the spike, and the sprint order. **START HERE.**
2. `docs/2026-06-14-comfy-cloud-image-video/WIRING_PLAN_cloud_engines.md` — the
   per-engine wiring (roles, required_inputs, lease, cost guard, outputs).
3. The two judgment logs in `roundtable_wiring/` + `roundtable_s0/` — why each
   decision is what it is (don't re-litigate settled items).

## Scope (operator-locked)
Three engines behind ONE flag `OTR_ENABLE_CLOUD`, default-OFF:
- `cloud_flux_pro` (image; all image roles).
- `cloud_ltx2` (video; `required_inputs=("text_prompt",)` → all 5 video roles,
  uses init_image opportunistically).
- `cloud_kling_avatar` (video; audio-driven; announcer + character roles only).

## HARD STOP — do S0 first, do not write adapters until it's green
Run the S0 spike (S0_SPIKE_AND_SPRINT.md §3): a standalone import/signature probe
AND an in-graph `/prompt` + `threading.Thread` proof that `comfy_api_nodes.util.
client` `sync_op`/`poll_op` work **headless with an explicit `OTR_COMFY_API_KEY`**
(the hidden auth tokens are `None` under `/prompt` — that's the whole risk).
Gate the spike behind `OTR_RUN_LIVE_CLOUD_SPIKE=1`. Write `S0_RESULTS.md` with
the pinned signatures/auth-arg/async-wrapper/output-type — **redact all secrets,
URLs, task ids, and local paths**. If headless auth can't be made to work, STOP
and raise with the operator.

## Then build in order (S0_SPIKE_AND_SPRINT.md §5)
S1 platform glue (no engine) → S2 `cloud_flux_pro` → S3 `cloud_ltx2` →
S4 `cloud_kling_avatar` → S5 polish + live smoke.

## Non-negotiable invariants (from the specs + CLAUDE.md)
- **Lease:** image — restructure `dispatch_images` to skip `_lease.acquire` +
  the post-gen NVML probe when `getattr(engine,"is_network",False)`. Video —
  override `MotionEngineBase.prepare()` to return `lease=None` (no acquire/load)
  when `is_network`; `teardown()` is already safe with a null lease.
- **Cost guard:** NOT in `assert_usable`. `reserve_cloud_cost` per object/clip,
  reserve→commit-on-success→**release-on-any-failure**; price-table estimate is
  the gate; episode ceiling `OTR_CLOUD_CREDIT_CEILING` in `ledger["billing"]`.
- **assert_usable:** flag on + key non-empty + `find_spec("comfy_api_nodes")`.
  Fail-closed, NO silent local substitution (BUG-LOCAL-405).
- **Outputs:** image → uint8 `(H,W,3)` numpy or `.png` path (no torch tensor);
  video → atomic-written silent `.mp4` (`has_audio=False`).
- **Cold-import (V-12):** the HTTP/comfy_api_nodes imports are LAZY inside
  `render_*` only; guarded `__init__.py` import logs LOUD on failure.
- **Workflow JSON:** no edit needed for v1 — engines auto-appear in the COMBO;
  default-OFF keeps saved selections local. If you must change a saved value,
  `widgets_values` is positional/append-only (BUG-LOCAL-097) → re-run
  `OTR_WorkflowValidator` + the link/widget audit.
- After EVERY change: run the suite + Bug Bible (CLAUDE.md §1); commit+push per
  green chunk to `v2.0-alpha`; verify HEAD==origin, no BOM, AST parse.

## Tests to land (S1)
cold-import; role_compat (kling=announcer+character, ltx=all 5 incl.
background_abstract, flux=all image roles); lease-skip (test FAILS if
`_lease.acquire` runs for a network engine; cloud video `prepare()` returns
`lease=None`); guarded-import-logs-LOUD + rows present with flag off; cost
reserve→commit→release-on-failure; image return shape/dtype assert.

## Open operator decisions (don't block S0)
Kling on music beats = no (talking-face only); episode ceiling default ~$5
(skip-to-floor); `commercial_clean=False` until ToS confirmed.
