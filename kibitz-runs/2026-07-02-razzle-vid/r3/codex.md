VERDICT: no. The plan has the right target, but the build order still lets cloud rows be registered, smoked, and even selected before the real request shape, workflow selection, and session/budget plumbing can carry them end-to-end.

MUST-FIX BEFORE BUILD:
1. [PHASE 1 / init_image request-shape fix] The worded-card spike is ordered before the init-image contract fix, but the current adapter cannot consume real render-driver requests: `render_driver.build_request()` writes `asset_refs["init_image"]` at `nodes/_otr_video_engines/render_driver.py:255`, while `_CloudVideoBase._init_image_input()` reads top-level `init_image` at `nodes/_otr_video_engines/eng_cloud_video.py:197`. Concrete fix: move the `asset_refs` resolver to the first Phase 1 step, before any live smoke/spike, and test `render_driver.build_request()` -> `_partner_inputs()` directly.

2. [PHASE 1 / Workflow JSON] "no new widgets expected" misses the real activation wiring. The production workflow currently saves `OTR_VideoDirector` video widgets as `["viz_green","viz_green","viz_green", ...]`, not any cloud row, in `workflows/otr_scifi_16gb_full.json:1`. Code registration alone will leave the razzle dormant. Concrete fix: for the spike, change the real workflow JSON `OTR_VideoDirector.widgets_values` for the target bookend role to the selected cloud engine in the same change, then run `OTR_WorkflowValidator` plus the link/widget audit.

3. [PHASE 1 / Cost enforcement] Per-episode cap cannot be implemented by only replacing `OTR_CLOUD_VIDEO_EST_USD`. Current adapters pass a scalar estimate from `_est_usd()` into `invoke_partner_node()` at `nodes/_otr_video_engines/eng_cloud_video.py:136`, while `CloudMediaSession` is keyed by prompt id and has only optional `episode_id` metadata at `nodes/_otr_shared/cloud_media_backend.py:247`. `teardown_session()` exists but has no caller in the searched source (`nodes/_otr_shared/cloud_media_backend.py:445`). Concrete fix: thread episode id/session lifecycle from the render batch or terminal mux, reserve a `CostQuote` per row before invoke, and tear down the cloud session at episode completion so caps and leak logs are scoped correctly.

4. [PHASE 1 / Duration fit] Exact-fit must happen inside the canonicalization path before the manifest is built, not as later productizing. Current cloud canonicalize computes `frame_count` from provider duration at `nodes/_otr_video_engines/eng_cloud_video.py:163`, while the timing authority is `timing.target_frame_count` in `nodes/_otr_video_engines/schemas.py:97`. Concrete fix: pass `target_frame_count` into `canonicalize_video`, trim/loop there, and assert the returned clip frame count equals the shot target before `build_clip_manifest()` sees it.

5. [ACCEPTANCE BAR / Phase 1 spike procedure] The proposed frame-extraction helper samples every ~0.5s, but the acceptance contract is "words stay READABLE for the WHOLE render -- every frame." Sampling cannot prove that contract. Concrete fix: extract every frame for the spike artifact set, or explicitly make the helper produce both all-frames and sampled contact sheets.

SHOULD-FIX:
1. [PHASE 0 / audit output] The plan says `--audit-i2v` emits JSON, but the existing pinner only supports default YAML pinning and `--check` in `scripts/otr_pin_partner_nodes.py:206`. Concrete fix: define the exact report path/name and add a test that invokes `--audit-i2v` without mutating `partner_nodes.yaml`.

2. [PHASE 1 / adapter conformance] New rows must update three places together, not just "new Engine subclass + register": package import in `nodes/_otr_video_engines/__init__.py:163`, `CAPABILITIES` in `nodes/_otr_video_engines/registry.py`, and tests like `tests/test_cloud_video_adapters.py:20`. Concrete fix: make the plan’s row-add checklist explicitly include all three, plus the static conformance test.

3. [PHASE 1 / optional prompt] The plan notes Kling should omit optional `prompt` when empty; current code always sends `"prompt": ""` at `nodes/_otr_video_engines/eng_cloud_video.py:229`. Concrete fix: filter empty optional kwargs generically before invoking partner nodes, or implement it per adapter.

4. [External system integration] There is timeout/interrupt handling, but no retry/backoff around provider transient failures in `invoke_partner_node()`; `_map_exception()` classifies retryable transport at `nodes/_otr_shared/cloud_media_invoke.py:527` but the call fails immediately. Concrete fix: add a small bounded retry policy for transport/5xx before billing or releasing the reservation.

OPTIONAL / NICE-TO-HAVE:
- [PHASE 0] Include import-path and class-name in the audit failure reasons so a later pin decision is traceable without reopening the live core.
- [PHASE 1] Record the final prompt string after era-tail and avoid-term expansion in the spike trace, because that is the cache/provenance boundary.

CUT THESE (over-engineering):
1. None. The expensive pieces called out here are contract closures, not optional architecture.