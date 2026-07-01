# Cosmic Radio Mandala -> production engine -- PLAN for kibitz (2026-06-30)

Turn the working pycairo prototype (`docs/2026-06-30-viz-rainbow/mandala_proto.py`, rendered +
operator-approved direction) into a real OTR video engine. Operator constraints: runs on ANY GPU/CPU
(pycairo = CPU vector, cross-platform); OTR radio mystique (tuning eye / dial / signal-spectrum, muted
iridescence, CRT glue); 100% local/offline; SFW; UTF-8 no BOM. Build AFTER the current session's other
queued work per the operator's ordering.

## GROUNDING (what exists)
- `viz_mxc_cpu` SHIPPED this session (eng_viz_rainbow.py) = numpy/PIL scope painter, required_inputs=(),
  audio-optional, accepts_still=False, silent-clip contract, uses scope_draw.analyze_audio_np +
  dual_ema + encode_silent_mp4. Full suite green + pushed (b01d2363). The operator was UNDER-whelmed by
  its PIL look -> the mandala (cairo) is the upgrade.
- The prototype PROVES: pycairo 1.29.0 / cairo 1.18.4 installs + runs on the Windows box; a cairo
  ImageSurface(ARGB32) -> BGRA memory -> numpy RGB conversion feeds encode_silent_mp4; reactivity from
  the SAME analyze_audio_np (32-bin FFT -> bass/mid/treble bands + spectral centroid) + dual_ema signal +
  an RMS-delta onset; 300 frames rendered clean.
- ffmpeg on this box is 8.0.1 (NO drawvg/VGS -- ruled out; pycairo is the path).

## DECISIONS TO HARDEN
1. **Engine shape:** upgrade `viz_mxc_cpu` to the cairo mandala painter (making pycairo a dep for it), OR
   ship a SEPARATE `viz_mxc_mandala` engine and keep the PIL one as a zero-dep fallback floor. Lean:
   SEPARATE engine (viz_mxc_mandala) so the zero-dep PIL floor survives for users who can't/won't add
   pycairo -- but confirm against the registry/CAPABILITIES + the operator's "one grammar" rule.
2. **pycairo dependency:** cross-platform CPU pip dep. Add to requirements.video.txt + the install docs +
   the S5 wizard model/dep note. V-12 cold-import: `import cairo` LAZILY inside render_clip only (never
   module scope). assert_usable must FAIL LOUD with a clear "pip install pycairo" message if absent (no
   silent fallback -- the no-fallback rule).
3. **Reactivity mapping (from the proto, to refine):** bass(FFT 0-5)->ring radius+stroke; mids(5-16)->
   spoke count+rotation; treble(16-32)->filigree flicker/detail; onset(RMS delta)->symmetry-flip +
   signal-lock flash; spectral centroid->global hue drift. Audio-OPTIONAL: silence -> slow idle mandala.
4. **Determinism (V-7):** seed all stochastic elements (grain, jitter, any random spawn) from a
   seed-keyed RNG so same request_seed -> byte-identical frames. The proto's onset is deterministic;
   confirm no wall-clock/random leaks.
5. **Performance:** cairo paint at 1472x832 x N frames on CPU -- MEASURE per-frame cost + the whole-beat
   time vs the PIL painter + the 14B/LTX engines; confirm it's not a soak regression (it should be far
   cheaper than any GPU engine). Cache static layers (bg gradient, scanline/vignette overlay) across
   frames.
6. **CRT glue:** reuse scope_draw.build_scanlines + build_vignette + film grain over the cairo output for
   the period look (the proto doesn't yet -- add for OTR consistency with the other engines).
7. **Contract:** silent h264/yuv420p/bt709; has_audio=False; frame_count EXACT; family="abstract";
   the wiring the r1+r2 kibitz already nailed for viz_mxc_cpu (ambient-audio gate, ENGINE_FAMILY,
   content_oracle._FAMILY_FALLBACK, CAPABILITIES row, auto-label) applies to the mandala engine too.

## FUTURE MODES (NOT this build -- one grammar per engine)
The other operator concepts -- Spectral Voice Entities (bezier creatures), Quantum Static Portals,
Lissajous Dream Engine -- are SEPARATE future engines/looks, each its own grammar. Do NOT fold them into
the mandala as modes now.

## TESTS
Mirror tests/test_video_viz_rainbow.py: registration + CAPABILITIES consistency; required_inputs=() fits
all 5 roles; accepts_still=False; cold-import clean (cairo NOT at module scope); render-contract with
encode_silent_mp4 monkeypatched (audio-present + audio-absent) + a pycairo-missing assert_usable path;
frame-count exact; paint determinism; the ambient-audio + family-map regressions. Suite + Bug Bible + B7
green; push per chunk.

## OPEN QUESTIONS FOR THE PANEL
1. Separate `viz_mxc_mandala` engine vs upgrading viz_mxc_cpu in place (dep + fallback-floor implications)?
2. pycairo dep on the 100%-local stack: install/packaging path; is a pip dep acceptable vs a zero-dep
   numpy/PIL push?
3. Performance ceiling of cairo at 1472x832 per beat -- any risk vs the soak budget? Static-layer caching?
4. Determinism + cold-import gotchas specific to pycairo (the C cairo lib).
5. Exact cairo->numpy conversion correctness (ARGB32 BGRA byte order, premultiplied alpha) for the
   silent-clip contract.
