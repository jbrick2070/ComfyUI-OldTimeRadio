# SIGNAL LOST procgen CRT upgrade -- synthesized design + sprint plan (pass01)

Synthesized by Claude from the blind round-1 panel (Gemini 3.1-pro, GPT-5.5,
DeepSeek-v4-pro), grounded against the real `_CRTRenderer` code. One file:
`nodes/video_engine.py` (`_CRTRenderer` + the `render_video` ledger plumbing).
No new node / widget / model / dependency; Pillow-only; green-only blend untouched;
audio spine frozen; 24fps procgen clock; deterministic per seed.

## The unifying concept (all 3 panels converged)
ONE signal-strength envelope is the master conductor. Re-enable the dormant
`self._brightness_ema` (render() section 8b, disabled v1.5.1) as a DUAL EMA: a slow
ambient `signal` (alpha ~0.05) for brightness, a fast `trig` (alpha ~0.3) for
glitch/lock triggers; `loss = 1 - signal`. Every dynamic element reads off it:
strong signal = crisp, locked, bright; weak signal (the silent inter-beat gaps) =
the receiver loses lock -- grid drops FIRST, ring spokes shorten + dim, static
crawls at the edges, the ident flickers LAST. The picture literally loses the
signal in the gaps and reacquires on the next cue. This is the grammar that ties
the title card (#1, "tune in") and the gutter scopes (#2, "hold the signal") into
one receiver.

## #2 -- Two gutter SCOPES (replaces center ring + particles + bottom waveform/bars)
Matched circular form, ASYMMETRIC data + failure modes (asymmetry is what stops twin
rings reading as wallpaper -- 3/3 panels). Both share radius + vertical center.
- **LEFT = FFT spectrum scope** (`_draw_fft_scope`): the 32 radial spokes (current
  section-2 data) + per-spoke phosphor-persistence comet-tails (a `self._fft_history`
  ring buffer, ~6 frames, tips fading to dim) + a peak-hold ghost ring (decaying max
  per bin) + a faint noise-floor shadow ring whose radius grows with `loss`. Idle
  (silence) -> collapses to a slow rotating radar sweep.
- **RIGHT = circular oscilloscope** (`_draw_scope`): the `wave` samples traced around
  the circumference + a bright SWEEP DOT with a decaying phosphor trail (the electron
  beam; `self._sweep_history`) + a trigger seam that free-runs/drifts under weak
  signal. Idle -> flattens to a jittering baseline circle. Absorbs the old section-5
  mirrored waveform (no separate bottom zone).
- **Graticules**: precompute a static tick-mark/crosshair overlay per ring in
  `__init__` (the `_scanlines` precompute pattern) and `alpha_composite` it. Makes the
  pair read as real instruments, near-zero per-frame cost.
- Retire section 5 (waveform) + section 6 (freq bars); thin section-3 particles to a
  faint orbit around each ring with brightness ROLES (primary/secondary/ghost), not
  hue. Line weight 1-2px; "thickness" comes from brightness/bloom, never geometry.
- **Center-column sanctity**: no bright reactive geometry in the portrait band (~the
  middle half); only the dim grid + scanlines + vignette may cross it.

## #1 -- Title card on the b000 music intro (decode -> reveal -> POP -> dock)
A windowed treatment invoked when `fi` is inside the b000 (music_open) window.
- **A. Carrier-lock**: "SIGNAL LOST" decodes from a seeded scramble / wrong-glyph
  snow into the solid terminal slab; a carrier-lock meter of broken phosphor blocks
  crawls to a solid line as the music swell arrives (driven by `signal`/window
  progress).
- **B. Episode-title reveal (the HERO, big + bold)**: the ACTUAL episode title at
  2-3x `f_title`, FAKE-BOLD via overstrike (draw 3-5x at 1px offsets -- `_load_font`
  loads only regular monospace, so there is no real bold to load), revealed as
  decoded fragments (wrong glyphs resolve over 6-12 frames per cluster) with a block
  cursor, STEPPING ON INTEGER FRAMES (no float-`t` mush).
- **C. Lock POP**: a 1-2 frame "chromatic tear" (an `np.roll` horizontal shear of the
  procgen frame) + a brightness bloom -- a BRIGHTNESS event, never a hue flash (the
  green-only blend kills color).
- **D. Dock (raster collapse)**: the hero title compresses into thin phosphor traces
  and snaps into the EXISTING section-1 ident + subtitle coordinates; normal section-1
  is suppressed while the card is active, then becomes the docked state. The intro
  rings "tune in" (incomplete arcs -> full circles) synced to the lock, tying #1<->#2.
- Timed in the 24fps procgen clock off the b000 start/end + first-dialogue frame from
  `led` (passed into `_CRTRenderer`). **Outro bookend**: same logic on music_close ->
  carrier drop -> hand to the existing `_TelemetryHUDRenderer` post-roll.

## Cross-cutting envelope behaviors (cheap, all off the one envelope)
- **Audio-choked vignette**: scale `self._vignette` intensity by `signal` (the tunnel
  closes when weak). Scalar math on the existing array.
- **Signal-loss sync-drift**: on low `signal`, `np.roll` the procgen chrome
  horizontally a few px (CRT losing horizontal hold). GROUNDED LIMIT: the green-only
  screen overlay can only shift the PROCGEN's own green chrome -- it CANNOT move,
  darken, or warp the portrait underneath. Clamp the roll so chrome never rolls into
  the center column.
- **Halation bloom**: behind bright green elements draw a larger `CRT_DIM` duplicate
  (cheap CRT glow, no Gaussian), only when `signal` > ~0.7.
- **Hierarchy clamp**: one multiplier used in sections 1/2/3/4/8 so weak signal drops
  the grid first and the ident last.

## SPRINT PLAN (one file; each sprint independently testable)
- **S1 -- envelope + plumbing + determinism.** Re-enable + dualize the EMA; expose
  `signal/loss/trig`. Plumb the b000 window + first-dialogue frame + music_close
  window from `led` into `_CRTRenderer`. SEED the RNG from (`fi`, title hash) and fix
  the section-8 noise (currently UNSEEDED `np.random.randint` -> non-deterministic,
  violates the determinism invariant).
- **S2 -- gutter scopes.** `_draw_fft_scope` (left) + `_draw_scope` (right) +
  precomputed graticules; retire sections 5/6, thin section-3 particles; enforce
  center-column sanctity + the landscape clamp.
- **S3 -- title card.** The b000-window helper (decode -> bold reveal -> POP -> dock)
  + intro-ring tune-in; suppress/restore normal section-1.
- **S4 -- envelope behaviors.** Vignette choke, gutter-clamped sync-drift, halation,
  hierarchy clamp.
- **S5 -- outro bookend + regression.** music_close handoff to the HUD post-roll;
  determinism + audio-byte-identical + no-new-widget regression.

## OPEN DECISIONS (operator's call; defaults chosen so the plan is sprint-ready)
1. **Landscape-beat gutters.** The procgen floor is rendered BEAT-AGNOSTIC --
   `_CRTRenderer` does NOT know which beats are portrait vs landscape (that is decided
   later in `OTR_SilentComposite`). So per-beat gutter awareness needs new plumbing
   (the clip-manifest timeline into the renderer). DEFAULT v1: fixed gutter rings
   clamped to the outer ~8-10%, no bright spokes crossing inward -- on landscape beats
   they ride the edges. Revisit per-beat awareness only if the eyeball rejects it.
2. **Telemetry micro-text** (SYNC / GAIN / dB labels on the graticules). DEFAULT:
   include faint static labels (reads as receiver diagnostics). Cut if it feels HUD-y.
3. **Noise inversion** (weak-carrier static crawl vs the current loud=sparkle). DEFAULT:
   keep loud-sparkle for S1-S5, add weak-signal edge-static as part of S4; do not
   invert the existing rule, ADD the edge-static so both read.

## VERIFY-AT-BUILD (do not assert these as done)
- The exact `led` field carrying the b000 music-open start/end + the first-dialogue
  frame (UNVERIFIABLE from the code read; `led` IS available in `render_video`).
- The floor is sliced timeline-aligned for gaps/credits in `OTR_SilentComposite`;
  ensure the b000 title-card frames map to the REAL episode open, not a reused mid-roll
  gap slice (a sequencing subtlety -- the title card must not reappear in inter-beat
  gap fills).
- `np.roll` chromatic-tear / sync-drift must be bounded so it never exposes a black
  edge or rolls chrome over the portrait.
