# pass02 judgment -- round-2 adversarial QA (sprint readiness)

Panel: gemini-3.1-pro, gpt-5.5, deepseek-v4-pro. ~$0.14. Strong convergence on
wiring bugs (not creative divergence) -- the design is settling.

## ACCEPTED MUST-FIX (all grounded vs real code, folded into pass02_plan)
- **Precompute per-frame state in __init__; keep render(fi) pure** (Gemini; GPT/DS on
  sequential-safety). The real render() is pure today; per-frame `self._*history`
  would couple output to execution order. Precompute arrays, read `[fi]`.
- **No np.roll on the frame array** (3/3) -- it wraps the center band into the
  portrait. Use draw-coordinate OFFSETS on gutter/title elements only.
- **Explicit timing interface** (GPT, DS) -- `__init__(... timing=dict of frame
  numbers)`, parsed from `led` in render_video, clamped, missing=disable-not-crash.
- **EMA read-only per element** (GPT, DS) -- the disabled v1.5.1 code multiplied the
  whole frame and dimmed text; the new signal/loss must only feed element effects, and
  the vignette choke must floor + exempt text.
- **Stable-hash local RNG** (3/3) -- `np.random.default_rng(blake2s(title|fi|salt))`,
  not Python `hash()` (process-random) and not global np.random.
- **Center-column sanctity = a computed band** (GPT) from the 480x832->1920 portrait
  scale (~center x in [636,1284]), clip scope geometry against it.

## GROUNDED -- a feared blocker CLEARED (the grounding step working)
- **Title-card reprisal in gap fills** (GPT, DS feared a release blocker). CONFIRMED
  SAFE by the code: `OTR_SilentComposite._floor_aligned` slices the floor
  TIMELINE-ALIGNED and the blend is FRAME-ALIGNED, so the open's title-card frames land
  only at the head/open, never in mid-roll gaps. Downgraded to a 2-beat verify smoke;
  no cross-file change (preserves "one file").

## ACCEPTED SHOULD-FIX
- EMA reset at fi==0 (not 0.5 half-locked); cap scope width 1-2px (code is 4px@1920);
  textbbox overflow guard on long titles; minimal fake-bold offset set; separate
  hierarchy layer floors; deterministic FFT idle-sweep phase from fi/fps; decide
  title-card state before section 1; outro renders INSIDE total_frames (mux preserved);
  halation via thicker dim-draw (no Pillow bloom); concrete scope x/r + a
  `_precompute_graticules()` method.

## ACCEPTED CUTS (lean v1; panel consensus)
- Telemetry micro-text (2/3) -> CUT (green mush + clashes with the real HUD).
- FFT peak-hold ghost + noise-floor shadow ring -> CUT (comet-tails suffice).
- Oscilloscope free-running trigger seam -> defer.
- Halation -> optional polish.
- Outro bookend -> conditional on the music_close window resolving.

## OPEN (committed v1 defaults so the plan stays buildable)
- Landscape gutters: v1 COMMITS to dim clamped-edge scopes on the beat-agnostic floor;
  per-beat gating is the eyeball-gated follow-up (needs cross-file clip-manifest
  plumbing). Operator can flip to suppression-on-landscape if preferred.
