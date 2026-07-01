# Claude anchor review -- r1 (mandala engine, high-level arc / creative coherence)

Grounded vs `eng_viz_rainbow.py`, `mandala_proto.py`, `scope_draw.py`, `registry.py`, and the live
pycairo install (1.29.0 / cairo 1.18.4, verified rendering 300 frames on the box).

## VERDICT: SOUND + PROVEN prototype; 3 arc-level decisions to lock before it's an engine.

## CONFIRMED
- pycairo works on the box; the ImageSurface(ARGB32) -> numpy RGB path fed encode_silent_mp4 cleanly
  (300 frames). CONFIRMED.
- The audio-reactive source is real + already shared: analyze_audio_np (RMS + 32-bin FFT) + dual_ema; the
  proto derives bass/mid/treble bands + a spectral centroid + an RMS-delta onset. CONFIRMED.
- ffmpeg 8.0.1 = NO drawvg -> pycairo is the only crisp-vector path. CONFIRMED.

## MUST-FIX (arc)
1. **Engine shape = keep the zero-dep floor.** viz_mxc_cpu (PIL, no dep) already ships + is green. The
   mandala needs pycairo (a C-lib pip dep that CAN fail to install on some boxes). To honor "runs on ANY
   box", ship the mandala as a SEPARATE engine (`viz_mxc_mandala`) and KEEP viz_mxc_cpu as the zero-dep
   accessible floor -- do NOT replace viz_mxc_cpu's painter and make pycairo a hard dep for the whole viz
   lane. assert_usable on the mandala must FAIL LOUD "pip install pycairo" (no silent fallback), so a box
   without pycairo simply can't select it but still has viz_mxc_cpu.
2. **Add the OTR CRT glue (the proto is missing it).** The mandala currently has NO scanlines / vignette /
   film grain -- so it does NOT yet read "1940s receiver" like viz_mxc_cpu + the other engines. Reuse
   scope_draw.build_scanlines + build_vignette + seeded grain over the cairo output. Without this the arc
   is "cool mandala", not "OTR mystique".

## SHOULD-FIX
1. **cairo->numpy correctness (verify-at-build).** ARGB32 is BGRA + PREMULTIPLIED alpha. The proto is safe
   ONLY because it paints an OPAQUE bg first (final alpha=1 -> premultiplied == straight). Make that
   invariant explicit (always paint opaque bg; assert surface opaque) so a future alpha element doesn't
   silently darken colors. Also FORMAT_ARGB32 stride can exceed w*4 -- read via surface.get_stride(), not
   assumed w*4 (the proto's np.ndarray buffer assumes contiguous w*4; at 1472 it happens to align, but
   assert stride == w*4 or handle it).
2. **Static-layer cache.** The bg radial gradient + scanline/vignette overlays are frame-invariant -- build
   once, reuse per frame (the per-frame paint is the cost driver at 1472x832).
3. **Determinism (V-7).** The proto is deterministic (t=fi/fps, no RNG). The moment grain is added, seed it
   from request_seed via scope_draw._rng. No wall-clock.

## UNVERIFIABLE (verify-at-build)
- Per-frame cairo cost at 1472x832 -> whole-beat time vs the soak budget (should be far under any GPU
  engine; measure). The proto rendered 12s (300 frames) fast, but time it formally.
- Whether the reactivity reads well on speech-only beats (the operator's real content) vs music.

## SCOPE (agree with the plan)
- The other concepts (voice entities / portals / lissajous) are SEPARATE future engines -- do NOT fold as
  modes. One grammar per engine.
