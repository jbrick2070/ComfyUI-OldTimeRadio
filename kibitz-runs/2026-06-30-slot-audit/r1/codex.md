VERDICT: no. The plan’s central “all engines in all slots” story contradicts the capability model already in code, and the proposed soak still misses two of the three Other-Beats roles.

MUST-FIX BEFORE BUILD:
1. [THE AUDIT TO PRODUCE] Defect: “ELIGIBLE? ... should be TRUE for all” is false. `background_abstract` supplies only `text_prompt`, while engines like `humo*` require `audio_ref + init_image`, `visualizer` requires `audio_ref`, and `wan_i2v` / `still_parallax` require `init_image`; tests already assert these exclusions. See `nodes/_otr_shared/role_compat.py:55-72`, `nodes/_otr_shared/role_compat.py:107-131`, `tests/test_capability_routing.py:37-60`. Concrete fix: define the audit matrix as engine x five roles with expected cells derived from `required_inputs <= ROLE_AVAILABLE_INPUTS`, not “all TRUE.”

2. [THE 3 USER SLOTS / SPRINT FIX 2] Defect: the plan says “all 3 slots,” but routing is actually five render roles behind three UI surfaces; the legacy Other-Beats slot covers `character_video`, `scene_broll`, and `background_abstract`. The current coverage runner maps `other_beats_visual` only to `character_video`, so it cannot prove scene/background behavior. See `nodes/_otr_shared/role_slots.py:43-67`, `scripts/_otr_cov_runner.py:50-52`, `scripts/_otr_cov_runner.py:95-101`. Concrete fix: make the audit/render plan role-based over all five roles, then summarize back to the three user-facing slots.

3. [SPRINT FIX 1] Defect: “kill the drift” only names `engines_for_role`, but the shared base registry still gates both `engines_for_role` and `assert_usable` on `roles`. Leaving `assert_usable` as-is preserves a second incompatible source of truth for any video consumer that uses it later. See `nodes/_otr_shared/engine_registry_base.py:169-177`, `nodes/_otr_shared/engine_registry_base.py:193-216`. Concrete fix: either override the video registry’s role checks with `role_compat`, or explicitly retire video `assert_usable` from role eligibility and add a guard test that no video path uses the stale roles gate.

4. [SPRINT FIX 3 / SPRINT FIX 4] Defect: “renders content” is a goal, but the sprint tests only assert eligibility/director acceptance/soak fill; they do not define a black/frozen-content oracle. Current render code explicitly falls to the dark floor when `_still_index` misses, and the coverage runner records verdict/histogram/audio, not luma or motion. See `nodes/_otr_video_engines/render_driver.py:414-442`, `nodes/_otr_video_engines/render_driver.py:928-958`, `scripts/_otr_cov_runner.py:129-163`. Concrete fix: add an acceptance layer that checks required still keys exist before render and/or samples output clips for non-floor luma plus frame variance.

5. [CONFIRMED DEFECT 2] Defect: the plan names the still-carrier break but not the end-to-end contract that must hold. The render driver looks up `still_pool_key or beat_id` in `ledger["images"]["images"]` scene rows; the dispatcher writes rows with `kind`, `beat_id`, and `path`. See `docs/2026-06-30-black-clips/BLACK_CLIPS_DIAGNOSIS.md`, `nodes/_otr_video_engines/render_driver.py:414-442`, `nodes/otr_image_gen_dispatcher.py:424-625`. Concrete fix: state the invariant explicitly: every still-consuming shot must have a matching `scene_*` ledger row keyed by its render lookup before `OTR_VideoRenderBatch` runs.

SHOULD-FIX:
1. [OPEN QUESTIONS 2] Defect: legitimate incompatibility is framed as an open question even though the code and tests already define the rule. Concrete fix: move capability exceptions into the main plan now, and leave only unverified engine declarations for r2.

2. [THE AUDIT TO PRODUCE D] Defect: requiring a manual list of every `roles` / `default_roles` gap risks preserving the deprecated whitelist as documentation truth. Concrete fix: generate a drift report from registry introspection, and make `default_roles` the only remaining semantic field unless `roles` is renamed as non-gating metadata.

3. [OPEN QUESTIONS 3] Defect: aspect is left vague even though the director already derives aspect labels and per-role still aspects. See `nodes/otr_video_director.py:29-41`, `nodes/otr_video_director.py:105-117`, `nodes/otr_video_director.py:343-383`. Concrete fix: define acceptance as “selected aspect is propagated and labeled; pillarbox/crop quality is visual QA,” unless a specific aspect bug is found.

OPTIONAL / NICE-TO-HAVE:
- Add a generated CSV/JSON matrix artifact so the sprint plan does not rely on hand-maintained tables.
- Include the current workflow node-87 saved values as context, but do not make workflow edits unless widget/schema changes are actually required.

CUT THESE (scope / over-engineering):
1. [THE AUDIT TO PRODUCE D] Cut manual enumeration of every stale `roles` gap as a build deliverable. Safe to cut because an automated drift test/report is stricter and avoids turning deprecated metadata into a new maintenance surface.

2. [INVARIANTS] Cut “audio byte-identical” from this plan’s narrative scope except as an existing regression check. Safe to cut because this sprint is video eligibility/render-content routing; audio identity is not the conceptual defect.

3. [OPEN QUESTIONS 3] Cut broad aspect redesign from this sprint. Safe to cut because the immediate failures are eligibility drift and missing still handoff, while aspect propagation already exists in `OTR_VideoDirector`.