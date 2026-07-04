VERDICT: yes-with-fixes. The concept has converged, but four plan-text defects still leave incompatible build choices.

MUST-FIX BEFORE BUILD:
1. [PHASE 0 / PHASE 1 build order #1] Phase 0 still includes “ONE gated live smoke leg” for surviving rows, while Phase 1 says “NOTHING is smoked before” the asset_refs fix lands (`kibitz-runs/2026-07-02-razzle-vid/r4/input.md:66-70`, `:90-96`). Concrete fix: make Phase 0 audit/report-only; move all live smoke legs after Phase 1 step 1, or add an explicit smoke precondition: `render_driver.build_request()` output must feed `_partner_inputs()` successfully first. Grounding: prod requests put init image at `asset_refs["init_image"]` (`nodes/_otr_video_engines/render_driver.py:249-257`), while cloud base reads top-level `init_image` today (`nodes/_otr_video_engines/eng_cloud_video.py:197-202`).

2. [PHASE 1 build order #2] “provider_job_id / content_sha256 / actual_duration_s may need schema homes” is not build-ready (`input.md:97-102`). Current `CanonicalClip` forbids extras and has no such fields (`nodes/_otr_video_engines/schemas.py:217-248`), but cloud canonicalize returns those extra keys (`nodes/_otr_video_engines/eng_cloud_video.py:181-184`). Concrete fix: choose one rule now: add these fields to `CanonicalClip`, or place them under existing `qc["provider"]`; require `CanonicalClip.model_validate(clip)` in the test.

3. [Problem / goal + PHASE 1 build order #3] “per-episode cap knob mandatory” lacks an exact public knob and scope (`input.md:31`, `:103-108`, `:151`). Existing backend has `OTR_CLOUD_MEDIA_BUDGET_USD` as a prompt/session ceiling and `episode_id` only as metadata (`nodes/_otr_shared/cloud_media_backend.py:17-25`, `:249-255`). Concrete fix: define v1 as either “reuse `OTR_CLOUD_MEDIA_BUDGET_USD` as the episode cap; set `session.episode_id`; call `teardown_session()` at node completion” or add a named workflow/widget knob in `workflows/otr_scifi_16gb_full.json` in the same change.

4. [Rough size vs Adapter reality] The final size section reintroduces “new reactivity class” after the plan explicitly says “NO new reactivity class” (`input.md:136-142`, `:173-177`). Concrete fix: replace “new reactivity class” with “new prompt-forwarding `mute_only`-pattern cloud adapter subclass/row.”

SHOULD-FIX:
1. [Workflow JSON activation] Name the exact selector to change: `OTR_VideoDirector.widgets_values[1]` / `music_video_model`, currently `viz_green`, for the music_open/music_close spike. The saved workflow has `announcer_video_model`, `music_video_model`, `other_beats_video_model` in that order and all three are `viz_green` (`workflows/otr_scifi_16gb_full.json`, node 87).

2. [PHASE 1 cost plumbing] Specify retry count/backoff and reservation scope. Current bridge reserves before invoke and bills/releases inside `invoke_partner_node()` (`nodes/_otr_shared/cloud_media_invoke.py:561-593`), so “retry before billing” needs one exact implementation path.

3. [PHASE 0 audit schema] “documented max duration/resolution” may not be code-checkable from `INPUT_TYPES`. Mark these nullable with a `source` field, and only fail filters on observable schema facts.

OPTIONAL / NICE-TO-HAVE:
Frame contact sheets are useful for review speed, but the acceptance gate is all-frame extraction plus manual legibility verdict.

CUT THESE:
1. Cut contact sheets as a required deliverable. Safe because every-frame extraction is the actual proof artifact.
2. Cut any coded candidate ranking. The plan already says manual cheapest passing row with seed support first; keep it manual.

VERIFY-AT-BUILD checklist:
1. `--audit-i2v` imports live ComfyUI api-node modules, tolerates per-module import failures, and writes the defined JSON report without mutating `partner_nodes.yaml`.
2. Every selected row’s emitted kwargs exist in pinned required/optional inputs, including the exact prompt kwarg name.
3. `render_driver.build_request()` output feeds the chosen cloud adapter’s `_partner_inputs()` without top-level `init_image` assumptions.
4. Cloud canonicalize output validates against `CanonicalClip` after metadata homes are resolved.
5. Final clip frame count equals `timing.target_frame_count` before `build_clip_manifest()`.
6. Dispatch eligibility is still based on `required_inputs`, not `reactivity`.
7. `teardown_session()` is called at episode completion and leaves no open reservations.
8. Workflow JSON selector value is changed in `workflows/otr_scifi_16gb_full.json` and passes validator, JSON round-trip, widget-count, input-name, and link integrity audits.
9. Spike artifact exists under the canonical episode path after render, not only in cloud cache/temp.
10. Manual spike trace records attempt, seed, final expanded prompt, row, and every-frame legibility verdict.