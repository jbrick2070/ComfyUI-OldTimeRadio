# Claude anchor -- r1 (promote-14b coding plan), grounded before the panel

VERDICT: yes-with-fixes. The plan is sound but the load-bearing item is the OPEN ROUTING
QUESTION (where the 14B actually goes), not the mechanical profile flip -- which the grounded
code already pins down.

## Grounded (CONFIRMED)
- widget_mapping.json maps role_overrides.other_beats_visual -> [OTR_VideoDirector,
  other_beats_video_model] and slot_overrides.video_render_engine -> [OTR_VideoRenderBatch,
  engine]; applier nodes/_otr_workflow_apply.py. CONFIRMED.
- 16gb_full.json pins both at "humo_1.7B" (not _169). CONFIRMED.
- Saved workflow: other_beats_video_model="visualizer (16:9)", OTR_VideoRenderBatch.engine
  ="humo_1.7B" -> profile<->workflow DRIFT (the 5 pre-existing fails). CONFIRMED.
- Episode mode ignores node 92; renders from the ShotLock ledger role routing. CONFIRMED.
- humo_14B_169 is the wide 14B fp8 + lightx2v distill (6 steps/cfg1.0/shift8), audio_driven_face;
  needs init_image + audio_ref -> NOT role-valid for face-less scene b-roll. CONFIRMED.

## MUST-FIX
1. RESOLVE the routing question BEFORE coding: HuMo is audio_driven_face (needs face+audio).
   "other_beats_visual" can include face-less scene beats -> humo_14B_169 there would fail
   role_compat at those beats. So the 14B belongs on the AUDIO-DRIVEN-FACE beats (where the
   1.7B was the pick), not blanket other-beats. The plan must name the exact role slot(s) +
   reconcile the visualizer drift, not "guess".
2. The thin-headroom invariant violation must be EXPLICIT + bounded (single-resident lease +
   beat-length cap), with a LOUD acceptance note -- do not silently break the <=14.5 ceiling.
3. Re-baseline the capability-profile + workflow-pin fixtures in the SAME change (the 5 drift
   fails must go green, not be masked).

## SHOULD-FIX
- Decide pre-sampler two-stage evict in eng_humo: Step A says +~217 MB only -> likely SKIP
  (don't touch eng_humo); rely on the existing post-decode reclaim + single-resident.
- Live acceptance render (episode -> OBS) at a representative AND a max-safe beat before close.

## Convergence target
Mechanics are well-grounded; the round should converge on the routing decision + the explicit
thin-headroom acceptance, then it is build-ready for a coder window.
