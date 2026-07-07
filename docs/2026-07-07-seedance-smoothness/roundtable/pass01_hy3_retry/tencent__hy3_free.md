<!-- requested_model: tencent/hy3:free | resolved_model: tencent/hy3-20260706:free -->

VERDICT: no. Plan lacks defined integration point for Seedance-only conditioner and leaves canonicalization root-cause unaddressed while carrying speculative audio scope.

MUST-FIX BEFORE BUILD:
1. [S4.1] Defect: Proposes Seedance-specific prompt conditioner but grounding shows `build_request_from_shot()` in `render_driver.py` finalizes `req["text_prompt"]` with no Seedance layer; no code location specified. Fix: add explicit post-`build_request_from_shot` hook (e.g. `_seedance_condition_prompt()` called when engine family is `audio_conditioned_video`) that appends stabilizer clause and softens verbs; cite exact function to modify.
2. [S3.5 / S4.2] Defect: Root cause #5 states canonicalization conform step may cause cadence issues, but S4.2 only adds trace warnings, not a fix, diverging from S0 smoothness goal. Fix: extend S4.5 harness to compare pre/post canonicalization frame intervals or assign owner to patch conform logic; do not ship only warnings.
3. [S0 / S4.1] Defect: Hard rule "no portrait pillarbox for wide bookend/b-roll" ignored; stabilizer clause "preserve reference image composition" may conflict with `OTR_CLOUD_SEEDANCE_RATIO` (grounding: env knob exists). Fix: condition must not alter `ratio` and must assert wide shots stay wide; verify in `eng_cloud_video.py`.

SHOULD-FIX:
1. [S4.3] Defect: A/B assumes seed yields deterministic comparable clips; video gen often non-deterministic. Fix: state [ASSUMPTION] or run multiple seeds. Verify: provider seed behavior in partner API.
2. [S4.5] Defect: Acceptance is "operator eyeball" with no recorded metric. Fix: add minimal motion-jitter score column to clip manifest (verify: manifest schema in code).
3. [S4.1] Defect: Only sci-fi radio verbs exemplified; other style packs not covered. Fix: specify generic verb-map applied when engine is Seedance, not just `sci_fi_radio.json`.
4. [S2] Defect: Log tail shows render latency (~2m37s/shot), not smoothness; narrative conflates. Fix: rephrase S2 to state latency only confirms input contract, unrelated to motion smoothness.

OPTIONAL / NICE-TO-HAVE:
- S4.2 trace warnings useful but low priority.
- S6 panel questions can be answered inside build doc.

CUT THESE:
1. [S4.4] Audio preconditioning (fade/peak-norm/low-pass/transient softening) – S3.2 is unconfirmed hypothesis; building deterministic per-beat derivative pipeline now is scope creep. Safe to cut until A/B in S4.5 proves audio-reactive jitter. [ASSUMPTION] audio cause.
2. [S5] "Do not touch cloud_kling_lipsync; that lane was nuked" – irrelevant to Seedance lane; cut from non-goals.
3. [S4.3] Optional C variant (stabilized + Seedance 2.0) – defer until A/B of prompt alone (A vs B) completes; reduces initial harness surface.