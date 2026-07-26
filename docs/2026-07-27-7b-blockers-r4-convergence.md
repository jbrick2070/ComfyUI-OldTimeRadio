# r4 CONVERGENCE -- 7b blockers to 7d

Base **`7f4644a1`** on `v2.0-alpha`. Suite 6951/27/1. Bible 17. Canonical
`9872624A`. C1 is LANDED (see 1.1). r4's job: confirm no NEW must-fix, and
close the residual defects below.

## 1. LANDED SINCE r3

**1.1 -- C1 is DONE (`7f4644a1`).** codex r3 correctly observed the descriptor
now exists; it landed mid-arc. Node 87 carries `max_render_frames` as its last
input descriptor, `widgets_values` untouched, byte-exact edit, repo-wide parity
guard plus a name/position pin, mutation-proven both ways (14 vs 15 counts
confirmed), link validator 0 violations. **Note the link validator passes in
BOTH states -- it could never have caught this**, which is why the guard exists.
Remove "add descriptor" from the remaining build sequence.

## 2. SETTLED ACROSS r1-r3 (do not reopen)

* Option A cut; B demoted to an optimisation. `render_driver.py:2952-2958`
  already makes the divergence terminal on the MULTI-segment path by comparing
  OUTPUT to plan.
* B2: a SEPARATE frame-policy snapshot, never inside `routing_env_snapshot`
  (whose fingerprint feeds a terminal route-drift check). It needs BOTH
  consumers: `IS_CHANGED` (interactive cache) AND a ledger stamp re-validated
  at the render boundary, because headless runs never consult `IS_CHANGED`
  (`render_driver.py:3324-3326`).
* B3: typed taxonomy, `UnresolvableEngineError` vs `InvalidEngineConfigError`,
  at three sites (`frame_contract.py:243`, `otr_shot_lock.py:1154`,
  `render_driver.py:3435`).
* B4 narrowed to estimate/absent/zero producers. `ltx_8gb` returns an exact
  encoded count -- **proven live this window: asked 25, got 25.**
* Hard 7c gate before 7d. `ltx_8gb` is the 7d lane (`161 + (9-1) = 169`).
* The 169-frame beat comes from the opening-music seam
  (`otr_shot_lock.py:514-541`), not from `render.frame_budget` -- that widget is
  inert in episode mode (`otr_video_render_batch.py:189-192`).

## 3. r3 CORRECTIONS ADOPTED

**3.1 -- key the policy registry by ENGINE ID, not family, and CUT the
universal precedence rule.** Seedance, Cloud Wan, Vidu, Pixverse and Kling
Avatar have different ranges, quantizers and duration inputs inside ONE adapter
module (`eng_cloud_video.py:632-639, 686-718, 758-810, 847-879, 955-980`), so a
family-level table cannot represent them. Each entry declares aliases, parser,
precedence, hard bounds, quantizer, request projection, and exact-count
capability. **agy's r2 unified `env -> profile -> literal` rule is REJECTED:**
the adapters differ historically on purpose (LTX-8GB admits values to 16384
against a static 161; WAN clamps 17..177), and flattening them would silently
change behaviour on lanes nobody is testing.

**3.2 -- coverage must be REFUSED BEFORE a paid call for provider-chosen
lanes.** Google Omni advertises an arithmetic frame contract but its outbound
request carries no target-frame or duration control
(`eng_google_omni_video.py:180-194, 358-378, 422-464`; same for the SFX lane,
`eng_google_vid_sfx.py:498-525`). Strict decoded-count equality would therefore
fail only AFTER an external call -- i.e. it would cost money to discover.
Add an explicit control mode per engine: `exact` / `menu` / `provider_chosen`,
and reject exact coverage at ShotLock/preflight for `provider_chosen`.

**3.3 -- ONE receipt builder, both paths.** `render_beat_coverage` returns
early for the single-segment case (`render_driver.py:2861-2874`) and only the
multi-segment tail assembles a result (`:2972-2979`), so a receipt wired into
the tail misses every beat production actually runs. `render_beat_coverage`
OWNS receipt construction for both paths; `build_clip_manifest` and
`OTR_VideoRenderBatch` only propagate and persist. This is the same
"one predicate, not two copies" rule the rest of the build follows.

**3.4 -- instrumentation BEFORE the receipt.** C4 as ordered would populate
loader counts that do not exist yet (`beat_session.py:104-109`). Either move
instrumentation ahead of receipt population, or split C4 into schema/carrier
first and strict population after 7c. **Missing counters must FAIL
qualification, never default to zero.**

**3.5 -- the 7c internal sequence, corrected.** `run_graph` cannot accept
preloaded results today (`wrapper_bridge.py:301-377`), so "remove the loaders"
is not a first step. Order: (a) add preloaded-result/handle support to
`run_graph`; (b) execute loader nodes once in `prepare`; (c) make segment
graphs consume those handles; (d) release/offload once in `teardown`; (e) only
then delete the embedded loaders and the ping-pong; (f) finally enforce exact
loader counts with a real two-segment test.

**3.6 -- projection compares AFTER every mutation, not after `request_builder`.**
The chain-terminal image is injected between `request_builder` and
`render_shot` (`render_driver.py:2905-2938`), so the comparison belongs
immediately before `render_shot`, after reference pruning and chain injection.
This also disposes of agy r3's false-positive concern: `has_init_image` must be
evaluated at the same point on both sides, not at ShotLock for one and
post-injection for the other.

**3.7 -- registry coverage for EVERY registered motion engine**, not the six
named families. LTX-AV has its own `OTR_LTX_AV_MAX_FRAMES` and an 8n+1 contract
with a render-time cap (`eng_ltx_av.py:92-94, 985-989, 1264-1278`). Tests fail
when a registered engine lacks a policy entry.

## 4. REJECTED, WITH REASONS

* **agy r3: raise `ltx_8gb`'s ceiling to 169 so the opening beat stays
  single-segment.** REJECTED, and it inverts the objective. 169 is chosen
  PRECISELY BECAUSE it splits `[161, 9]` on a 161-cap engine -- that split IS
  the 7d proof. Raising the ceiling would delete the thing being qualified.
* **agy r2: unified precedence rule.** Rejected, see 3.1.
* **agy r2: `frame_budget: 169` profile variant.** Rejected -- the widget is
  inert in episode mode.
* **agy r2: `FrameContract` needs `to_dict` for frozensets/enums.** MISREAD --
  it holds a tuple and a str (`frame_contract.py:129-135`), both JSON-safe.
* **agy r3: projection will false-positive on chained segments.** Real trap,
  wrong remedy -- resolved by 3.6 (compare at one point) rather than by
  redefining the field.
* **Veo `reference_images` projection.** Unreachable: the canonical builder
  emits only `init_image` and `audio_ref` (`render_driver.py:2214-2216`).

## 5. RESIDUAL -- what r4 must confirm or kill

**5.1 -- canvas/fps never reach the final request (codex r3 MUST-FIX 1).**
`OTR_VideoRenderBatch` calls `run_real_episode` without canvas/fps
(`otr_video_render_batch.py:372-373`), `build_request` hard-codes 25 fps
(`render_driver.py:221-259, 2240-2289`), and env defaults can displace the
profile's 512x288 (`otr_8gb_ltx.json:63-71`). **Is this a 7d blocker or a
pre-existing condition 7d merely exposes?** If the canonical canvas is already
threaded some other way, say where; if not, it needs a slice before 7d, because
a 169-frame ask at the wrong canvas is not the qualification anyone wants.

**5.2 -- does anything else consume `coverage_plan` that a receipt must not
break?** Name every reader before C4 changes the shape.

**5.3 -- C5's artifact (still unresolved after three rounds).** codex proposes
a deterministic OFFLINE mutation-proven test over registered engines: an ask
within the effective contract yields one render request, no multi-segment
BeatSession path, one receipt, zero forbidden extension/replan/truncation
counters. Confirm that is sufficient, and that the live 169 proof stays in 7d.

**5.4 -- is there any remaining path where a plan is stamped and NOT
re-validated at the render boundary?** Both branches of
`resolve_final_shot_engines` call `assert_coverage_plans`; confirm no third
entry point (the HTTP single/soak routes) bypasses it.

## 6. BUILD ORDER (post-r3)

C1 DONE. Then: C2 (frame-count boundary + the `if got` fail-open, scoped per
2 and 3.2) -> C3 (per-ENGINE policy registry + typed taxonomy) -> C4a (receipt
schema/carrier, one builder per 3.3) -> C5 (offline single-segment proof) ->
C6 (stamped-vs-live + projection per 3.6) -> **7c** in the 3.5 order ->
**7d** live, local-only, no paid provider path invoked.

## 7. GROUND RULES

Cite `file:line`. This arc has refuted three driver claims and two panel claims,
all verified. Point at the code. THE LAW holds. UTF-8 no BOM, ASCII, SFW,
$0 external spend.
