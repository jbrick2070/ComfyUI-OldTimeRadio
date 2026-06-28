# Promote HuMo-14B -- r1-hardened coding plan (Codex + Claude; grounded)

r1 panel: Codex (gpt-5.5/high, read the repo) + Claude anchor. Codex VERDICT "no" -- the plan
forced 14B into the shared other-beats slot while leaving the routing "open". All MUST-FIX
grounded + CONFIRMED. Antigravity offline (stale auth -- separate issue).

## GATING DECISION (operator) -- resolve BEFORE coding
HuMo-14B is `audio_driven_face` (REQUIRES `audio_ref` + `init_image`, eng_humo.py:81-89). The
profile slot `role_overrides.other_beats_visual` -> `OTR_VideoDirector.other_beats_video_model`
drives ALL other-beats: `character_video` + `scene_broll` + `background_abstract`
(otr_shot_lock.py:709-714); `scene_broll` has no `audio_ref` and `background_abstract` is
text-only (role_compat.py:55-72) -> humo_14B_169 there FAILS role_compat. AND the SAVED
workflow currently uses `visualizer (16:9)` for other-beats (the 2026-06-23 HuMo-free UI-save)
-> HuMo is NOT in the live default at all. So "promote the 14B" needs a ROUTE:
  R1. **14B on talking-FACE beats only** (the role HuMo serves + what the bakeoff tested).
      The single other_beats widget can't cleanly host it unless the director does PER-BEAT
      role_compat fallback for the face-less beats (verify the director already filters per
      beat; if not, this needs a per-sub-role selection = an architecture change). RECOMMENDED.
  R2. **14B for ALL other-beats** -- only valid if every other-beat carries audio+face; else
      define explicit scene_broll/background behavior (fallback engine) -- messy, not advised.
This is a product+architecture call; the coder plan is NOT build-ready until the operator
picks R1 vs R2 and the per-beat-fallback mechanics are confirmed in the director code.

## MUST-FIX (grounded; folded)
1. Resolve the route above; if R1, confirm OTR_VideoDirector applies role_compat PER BEAT so a
   face-less beat does not hard-fail when other_beats_video_model=humo_14B_169 (no-fallbacks
   rule raises LOUD otherwise).
2. node 92 (`OTR_VideoRenderBatch.engine`) is parity/smoke ONLY -- episode mode renders from the
   ShotLock ledger (otr_video_render_batch.py:127-134). Live acceptance must prove the ShotLock
   LEDGER ROWS / render manifest used `humo_14B_169`, not merely that node 92 changed.
3. Beat-length cap must be ENFORCED, not prose: HuMo clamps to `_HUMO_MAX_FRAMES=177`
   (eng_humo.py:54, used at :341) but the tested-safe envelope is <=49-81f. Pick the
   enforcement point (render-driver clamp / ShotLock budget split / hold-frame extend) + add
   acceptance that a max beat cannot exceed the cap.
4. Two-stage evict: RESOLVE the contradiction -- DROP it from the promotion (Step A showed only
   ~217 MB: 15996 single vs 15779 two-stage; not worth an eng_humo change). The safety story =
   the EXISTING post-decode `reclaim_idle_models` + single-resident AS-3 lease + the beat cap.
   Rebase acceptance on NO-evict 14B runs.

## SHOULD-FIX (grounded)
- Name the exact 5 failing tests (attach log), don't assert. The drift invariant: after the
  routing change, profile and SAVED workflow must MATCH EXACTLY (test_capability_profiles.py
  :176-205, test_workflow_apply.py:111-117) -- UPDATE to new truth, do NOT weaken to pass.
- Preflight HuMo enable+install (OTR_ENABLE_HUMO + ckpt present, eng_humo.py:13-14,150-156)
  before live acceptance.

## CUT
- "DECIDE whether to port pre-sampler evict" as a mid-build task (resolved: dropped).
- Raw "re-pin fixtures" language -> one explicit invariant update after routing is chosen.

## Build order (once the operator picks the route)
profile+workflow edit (SAME change, via _otr_workflow_apply.py) -> update the profile/workflow
match fixtures to new truth -> OTR_WorkflowValidator + JSON round-trip + link/widget audit ->
suite + Bug Bible + B7 -> HuMo preflight -> live episode render, prove ledger used humo_14B_169
+ OBS publish + no OOM at a representative AND a max-cap beat -> operator eyeball -> commit.
