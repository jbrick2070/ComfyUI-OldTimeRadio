# Coder kickoff -- optional green audio-reactive bars overlay (any video engine)

START AFTER the visualizer engine (CODER_KICKOFF.md Task 1) is committed + pushed
green. Small, additive change to the EXISTING overlay -- not a new engine.

## Goal

Let the operator optionally show green, audio-reactive bars over the FINAL video
REGARDLESS of which video engine rendered it (HuMo / LTX / Wan / etc.). This is the
"old-school bottom-of-screen" look -- a minimal accent over real video, distinct
from the full-frame `visualizer` engine.

## Where it lives (do NOT touch the engines)

This is a COMPOSITE-STAGE feature: `OTR_SceneAwareScopes` -> `OTR_PostUpscaleProcgenBlend`
already run AFTER the render and read the master audio, so they are independent of
the per-role engine dropdowns by construction. Today `plan_scope_frames` draws
only in portrait gutters / gap centres and SUPPRESSES landscape clips
(`mode = None`). That is why normal 16:9 episodes show nothing.

## The change (append-only)

1. **One new widget on `OTR_SceneAwareScopes`**: `landscape_bars` with values
   `off` (DEFAULT) | `bottom`. `off` MUST be byte-identical to today (regression
   test). Append at the END of `widgets_values` (BUG-LOCAL-097 positional rule);
   add the matching optional entry in `INPUT_TYPES`.
2. **`plan_scope_frames` landscape branch**: when `landscape_bars == "bottom"`,
   the `src == "clip"` + landscape case returns a new mode
   `("bars", x, y, w, h)` (a bottom strip geometry) INSTEAD of `None`. Portrait
   gutters, gap centres, and the tail-credits suppression stay exactly as-is.
3. **Draw**: in the `("bars", ...)` branch of the frame painter, draw a green
   frequency-bar strip (and/or mirrored waveform) using the SHARED
   `_otr_shared/scope_draw.py` freq-bar routine extracted for the engine (DRY --
   same look). GREEN-ONLY (CRT_GREEN/DIM/DARK); black background.
4. **Nothing else changes**: the blend (`OTR_PostUpscaleProcgenBlend`,
   lighten/screen of green-on-black) composites the bars over the video with no
   dimming -- no blend edit needed. The BUG-LOCAL-406 master-length padding, the
   silent `-an` encode, determinism, and the green-only palette are ALREADY in
   `OTR_SceneAwareScopes` and are inherited unchanged.
5. **Wiring (CLAUDE.md sec 0)**: add the `landscape_bars` widget value into
   `workflows/otr_scifi_16gb_full.json` (the node's `widgets_values`) in the SAME
   commit; re-validate (OTR_WorkflowValidator + JSON round-trip + widget audit).

## Caption layering (HARD)

Captions MUST render ABOVE the bars -- the bars can never occlude SDH text.
Verify the composite ORDER in `workflows/otr_scifi_16gb_full.json`:

- If the caption burn (the SDH burn-in node, ~Node 58) already runs AFTER
  `OTR_PostUpscaleProcgenBlend`, captions are already on top -- good; just confirm
  the link order and leave it.
- If caption burn runs BEFORE the blend, the bars would cover the subtitles. Fix
  by ordering so the caption burn is the LAST composite step (captions painted over
  the bars), OR keep the `bottom` bars strip ABOVE the caption safe-area (the lower
  ~12-15% reserved for subtitles) so they never overlap.
- Preferred: captions are the TOP layer unconditionally (burn after the bars
  blend); additionally keep the bars out of the caption safe-area as defense in
  depth. Add a test asserting the caption layer composites after the bars layer.

## Independence

The bars are controlled ONLY by the `landscape_bars` widget. Engine selection and
the bars toggle do not interact: bars work over any engine's output, and the
`visualizer` engine works with bars off. Default OFF -> zero change unless enabled.

## Tests (full suite + Bug Bible after every change)

- `landscape_bars="off"` -> plan + output byte-identical to pre-change (lock it).
- `landscape_bars="bottom"` -> landscape clip frames get the bars mode; portrait
  gutters / gap centres / tail suppression UNCHANGED.
- bars frames are GREEN-ONLY; scopes track still spans the master length (BUG-406);
  `test_audio_byte_identical` green.
- widget-count vs live INPUT_TYPES audit passes; JSON round-trips; UTF-8 no BOM.

Commit + push per green chunk to v2.0-alpha; verify HEAD==origin, AST parse.
