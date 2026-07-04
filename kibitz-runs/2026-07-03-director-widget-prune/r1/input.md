# QA: which OTR_VideoDirector widgets can + SHOULD be removed (value, not tidiness)

Operator challenge: don't remove widgets just to "make code happy" -- each removal must
make the PROJECT better (honest UI / less dead code) without breaking behavior. Decide
PER WIDGET: (a) can it be removed safely, (b) does removal improve the project or is it
cosmetic/risky. Ground every claim against the real code, cite file:line.

## Anchor findings (Claude -- verify / correct)
- `episode_duration_target`: grep finds it ONLY inside `nodes/otr_video_director.py`
  (declare + param + policy emit). NO consumer reads `policy["episode_duration_target"]`.
  Episode length is word->script->audio driven (frozen audio). => a control that is
  SILENTLY IGNORED. Anchor verdict: REMOVE = real value (UI stops lying). Already removed
  locally (uncommitted); confirm nothing reads it anywhere (scripts, ShotLock,
  render_driver, compute_clip_budget, schemas).
- `canvas_w` / `canvas_h`: the policy `canvas` dict carries {w,h,fps}. ShotLock reads ONLY
  `canvas["fps"]` (otr_shot_lock.py ~:871). render_driver OVERRIDES the request canvas per
  engine family (render_driver.py ~:1378-1428); cloud_delivery_wh ignores w/h for the
  target. BUT build_request seeds its default canvas from the policy, so w/h are the
  FALLBACK canvas for any engine WITHOUT a per-family override. Anchor verdict: NOT cleanly
  dead -- removal needs a hardcoded fallback; weaker value, small risk. Confirm: is there
  any engine/path that actually renders at the policy canvas w/h (no override)? If none,
  removal is safe with a constant default; if any, keep.
- `character_video_model`: ACTIVE. role_slots.ROLE_TO_VIDEO_SLOT[character_video] =
  "character_video_model"; engine_id_for_role uses it, else falls back to
  other_beats_video_model. In the shipped JSON it = humo_14B_169 and DRIVES character
  beats. Anchor verdict: KEEP -- removing it breaks the character lane.
- `other_beats_video_model`: the LEGACY single-slot fallback; post rip-sfx-broll it serves
  ONLY character_video as the inherit-fallback (role_slots LEGACY_OTHER_BEATS_SLOT). Anchor
  verdict: this is the REAL redundancy -- but collapsing it into character_video_model is a
  role-resolution refactor (migration semantics + tests), not a widget delete. Assess
  whether the consolidation is worth it or if the fallback still earns its keep (old graphs
  / lighter tiers that leave character on the sentinel).

## Questions for codex (decide each, grounded)
1. `episode_duration_target`: confirm ZERO readers anywhere (code/scripts/tests/schema).
   Is removing it purely a UI-honesty win with no behavior change? Any planned-feature seam
   worth keeping?
2. `canvas_w`/`canvas_h`: does ANY render path actually use the policy canvas w/h (an engine
   with no per-family override, or the composite/still init)? If yes, name it (removal
   unsafe). If no, is the value of removal worth hardcoding a fallback default?
3. `character_video_model`: confirm it is ACTIVE and must stay (removal breaks character
   video). 
4. `other_beats_video_model`: is it still load-bearing (any path/old-graph that relies on
   the inherit-fallback), or is it safe to collapse into character_video_model? Is the
   consolidation a net project improvement or churn?
5. Overall: rank these by REAL project value of removal (honest UI / dead-code) vs risk /
   churn. Flag any I have wrong.

Invariants: node/JSON edited in the SAME change; positional widgets (BUG-LOCAL-097);
validator must pass (widget-count + no rogue input socket -- last removal tripped exactly
that); audio spine untouched; suite + Bug Bible + B7 green; no back-compat shims.
