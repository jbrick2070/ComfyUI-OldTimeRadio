# arch round robin -- convergence judgment (3 passes)

Panel: gpt-5.5 + gemini-3.1-pro + deepseek-v4-pro. 3 passes, ~$0.54. Topic: the v2
SCENE-AWARE reactive-scopes architecture (the proper fix for the landscape-gutter open
decision). CONVERGED on the shape; rounds 2-3 hardened the wiring.

## VERDICT: architecture CONVERGED.
Round 1 set the shape (3/3 agreed). Rounds 2-3 found WIRING precision, shrinking each
round; deepseek-r3 = near-pass ("CUT: None, minimal, addresses the goal"); gpt-r3 (the
strictest) still flags precision + verify-at-build, all folded. No panel challenged the
SHAPE after r1.

## The converged architecture (grounded, folded into §4D)
- The floor CANNOT be scene-aware (grounded: `render_video` gets no manifest/engine plan;
  `meta.visual_plan` = characters/style only; aspect resolves later). So split ONLY the
  SCOPES into an ADDITIVE late node; the floor (CRT + title) is NOT relocated.
- 4 parts: (1) circular GREEN-ONLY scope helpers (a rewrite -- the existing
  `_waveform_mirror`/`_freq_bars_wide` are rectangular + non-green); (2) NEW
  `OTR_SceneAwareScopes` -> `scopes_only.mp4` (black + green, no master decode); (3)
  EXTEND `OTR_PostUpscaleProcgenBlend` with a 3rd input + a gbrp-throughout double-blend
  (`screen` then **`lighten`** to avoid brightness compounding); (4) `draw_scopes` flag on
  the floor.
- Beat map: REUSE `plan_timeline_segments(...)` -> integer frame ranges + `total` (no
  source-video probe). Eligibility by `source`: clip+portrait -> gutters; clip+landscape
  -> suppress; head/inter gap -> draw (keep the gaps alive); TAIL/credits -> suppress.
- Aspect: `ffprobe` each clip segment's `path` (h>w=portrait); CUT the engine registry.
- Audio: the EXISTING `sr//fps` `_analyze_audio` (exact at 25fps -> frame-identical, no
  spine touch); `total` from the planner.

## GROUNDED RESOLUTIONS (the grounding step earning its keep)
- **Manifest source CONFIRMED:** `OTRVideoRenderBatch.RETURN_NAMES =
  ("render_report_json","clip_manifest_json")` -- the producer outputs it, so
  `OTRVideoRenderBatch -> OTR_SceneAwareScopes` is a real wire (NOT `OTR_SilentComposite`,
  which only returns `silent_video_path/report`).
- **Audio:** `sr//fps` is exact at 25fps (48000//25=1920) -> keep it; drop the
  round-boundary idea (gemini grounded this).
- **Frame count:** from `plan_timeline_segments`' returned `total` -> no `source_mp4_path`
  input needed (gemini).
- **Two screens compound brightness** -> 2nd blend = `lighten` (deepseek).

## Round-3 precision folded
fps HARD-LOCK 25 for this pipeline (the floor defaults 24; force 25 OR relax the
frame-assert -- the procgen is framesync'd by the existing blend; assert scopes==source
only); tail = the gap segment(s) at `cursor >= last-beat-end` (compute from beat extents,
NOT "last clip"); absent audio -> synthesize zero arrays (don't call `_analyze_audio`);
center-gap geometry `cx=out_w//2, r=int(min(out_w*0.16, out_h*0.30))`; register the node
in `NODE_CLASS_MAPPINGS`; thread `draw_scopes` render_video -> `_CRTRenderer` -> render().

## v1/v2 sharing (the one nuance, gpt-r3)
The new circular green scope HELPERS are shared: §4C-v1 calls them from the floor
(beat-agnostic), v2 from the late node (scene-aware). `draw_scopes=False` turns the floor
scopes off for v2. LOWER-RISK option (gpt cut): leave the floor's existing drawing alone +
build the new helpers in the NODE only -- i.e. skip §4C-v1's floor-scope placement and go
straight to v2. Operator's call at build time.

## VERIFY-AT-BUILD (the build checklist)
clip `path` is a pre-composite asset with real dims (else h>w misclassifies); the
`lighten` 2nd-blend stays visible over the floor CRT; the node's full INPUT_TYPES + the
`NODE_CLASS_MAPPINGS` registration; empty-manifest -> fail early (`total<=0`); golden-frame
tests (portrait / landscape / head gap / credits tail).
