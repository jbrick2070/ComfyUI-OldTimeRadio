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

## DECISIONS -- LOCKED at r1 (grounded, see kibitz-runs/2026-06-30-mandala/r1/r1_judgment.md)
1. **Engine shape: SEPARATE `viz_mxc_mandala`.** Keep viz_mxc_cpu as the zero-dep selectable alternate.
   The "upgrade viz_mxc_cpu in place" branch is CUT.
2. **pycairo isolation.** NOT added to requirements.video.txt (pip pycairo needs system libcairo -> would
   break clean Linux/mac installs of OTHER engines). Lazy `import cairo` INSIDE render_clip only (V-12
   cold-import clean). assert_usable import-probes cairo and FAILS LOUD "pip install pycairo" if absent.
   pycairo does NOT go in the CAPABILITIES table (capability_profiles.py is fail-closed, rejects unknown
   keys at L266) -- the row uses required_toolchain=None like viz_mxc_cpu.
3. **Dual safety: fail-loud eligibility + loud fallback terminus.** assert_usable fails loud (above) AND
   the engine declares `fallback_engine="viz_mxc_cpu"`. render_driver HONORS the chain (L158) and
   restamps the ledger LOUDLY -- so a hard render failure degrades to the zero-dep PIL visualizer,
   logged, never silent. (Grounded: the "fallbacks are disabled" raise only fires for engines declaring
   fallback_engine=None.)
4. **Grammar (one sentence):** centered tuning-eye mandala FIRST; radio-dial rings/spokes SECOND; CRT
   scanlines/vignette/grain as POST only. Muted iridescence. NON-GOALS: no creatures, no portals, no
   lissajous, no mode-switch widget.
5. **cairo->numpy correctness:** read `surface.get_stride()` (never assume w*4); paint an OPAQUE bg first
   (premultiplied-alpha == straight); assert surface opaque. `ctx.save()/restore()` around the paint body
   (no state leak).
6. **Reactivity (from proto):** bass(0-5)->ring radius+stroke + outer band inner-radius; mids(5-16)->spoke
   count+rotation + band rotation; treble(16-32)->filigree flicker; onset(RMS delta)->symmetry-flip +
   signal-lock flash; spectral centroid->global hue drift; vol->tuning-eye pulse. Audio-OPTIONAL: silence
   -> slow idle mandala. Denser look (2026-06-30 operator): 48 solid spectrum wedges + 9 bolder rings.
7. **Determinism (V-7):** CRT grain seed-keyed via scope_draw `rng_key`; no wall-clock/global random.
8. **CRT glue:** reuse scope_draw scanlines/vignette/grain. DROP static-layer caching (overlays already
   build once/run; cairo radial gradient is cheap; scanlines/vignette sit ABOVE reactive layers).
9. **Contract:** silent h264/yuv420p/bt709; has_audio=False; frame_count EXACT; family="abstract";
   new CAPABILITIES row + the same map set viz_mxc_cpu touched (ambient-audio gate, ENGINE_FAMILY,
   content_oracle._FAMILY_FALLBACK, auto-label).

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

## OPEN FOR r2 (coding plan)
1. **Perf budget (numeric):** set max ms/frame + max s per 25-frame beat; benchmark viz_mxc_mandala vs
   viz_mxc_cpu. Confirm no soak regression (must be far cheaper than any GPU engine).
2. **CRT glue path:** PIL-roundtrip (proven, simple, reuses scope_draw as-is) vs native-cairo
   OPERATOR_MULTIPLY (faster, more code). Pick by the measured budget.
3. **16:9 bounding:** cap the tuning-eye + rings so they aren't clipped on 1472x832; let the outer band
   ring bleed intentionally (denser look). Lock the radius multipliers.
4. **Tests:** mirror test_video_viz_rainbow.py PLUS a visual-acceptance smoke (nonblack ratio,
   frame-to-frame delta, deterministic hash) to catch a dull/static mandala.

## OPEN FOR r3 (wiring)
- Register in the video dropdown enum + ENGINE_FAMILY / content_oracle / ambient-audio maps (same set
  viz_mxc_cpu touched). Decide whether any saved widget in otr_scifi_16gb_full.json defaults to it
  (likely no -- opt-in selectable), but the dropdown MUST list it. Re-validate JSON + link/widget audit.
