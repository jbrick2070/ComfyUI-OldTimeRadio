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
3. **Fail LOUD, NO fallback (CORRECTED at r2).** `fallback_engine=None` (mirror viz_mxc_cpu). GROUNDED:
   the production `render_shot` (render_driver.py:1531-1558) has the operator directive "NO FALLBACKS
   (2026-06-16, 'this is art, not a space shuttle')" -- `fallback_of` is IGNORED, a hard failure RAISES
   loud, no swap. So there is no graceful degrade. assert_usable import-probes cairo and fails loud
   "pip install pycairo" at eligibility; a cairo render crash raises loud like any engine. (My r1
   "fallback to viz_mxc_cpu" lock was wrong -- that chain helper is dead for render_shot.)
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
8. **CRT glue = PIL roundtrip, DECIDED** (cut native-cairo for v1). scope_draw grain is NOT public
   (private _rng, embedded in paint fns) -> add a PUBLIC `apply_crt_post_rgb(rgb, ..., rng_key)` helper in
   scope_draw (scanlines + vignette + seed-keyed grain over an RGB array), called by the mandala engine.
   DROP static-layer caching (overlays build once/run; cairo gradient cheap; scanlines sit ABOVE reactive).
9. **Registration is concrete:** eng_viz_mandala.py `@register`; import row in __init__.py; CAPABILITIES
   row; ENGINE_FAMILY + content_oracle._FAMILY_FALLBACK + _uses_ambient_master_audio maps (the viz_mxc_cpu
   set). Radius cap `outer_max = min(cx,cy,w-cx,h-cy)-margin`; only the outer band bleeds.
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

## WIRING -- LOCKED at r3 (grounded, r3_judgment.md; STRONG convergence)
- **Opt-in SELECTABLE engine, NOT a saved-widget default.** Registering makes it selectable in the
  registry-driven dropdown (nodes/otr_video_director.py) -- satisfies reachability (CLAUDE.md 0). Node 87
  keeps its current saved widgets; the operator MAY later set the music/title-bookend widget to
  viz_mxc_mandala (their radio-bookend vision) -- OFFER it, don't auto-set. Run validator + JSON
  round-trip + link/widget audit regardless.
- `__init__.py:~139` -- `from . import eng_viz_mandala as _eng_viz_mandala` (in the guarded block).
- `render_driver.py` -- `ENGINE_FAMILY["viz_mxc_mandala"]="abstract"` (~L64); add "viz_mxc_mandala" to the
  `_uses_ambient_master_audio` tuple (~L760).
- `content_oracle.py` -- `_FAMILY_FALLBACK["viz_mxc_mandala"]="abstract"` (~L42).
- `scripts/otr_video_soak.py:~56` -- add BOTH `"viz_mxc_cpu"` (latent gap) + `"viz_mxc_mandala"` to
  ENGINE_FAMILY. NO `_PROFILES` soak leg (soak coverage = the dedicated render-contract test).
- `registry.CAPABILITIES["viz_mxc_mandala"]` = FULL _DECL_KEYS dict (fail-closed, capability_profiles.py
  L260-273): `{"vram_class":"cpu","vram_estimate_mb":0,"required_toolchain":None,"requires_sidecar":
  False,"cpu_ok":True,"model_requirements":[]}`.
- **assert_usable** probes BOTH `import cairo` AND `_sd.find_ffmpeg(OTR_FFMPEG)` -- separate loud messages.
- **surface->rgb:** `bgra = np.ndarray((h,w,4), np.uint8, buffer=surface.get_data(),
  strides=(stride,4,1))`; `rgb = np.ascontiguousarray(bgra[:, :, [2,1,0]])`; assert `(h,w,3)` uint8
  (encode_silent_mp4 writes rgb24). stride from `surface.get_stride()`.
- **NEW helper:** implement + export `apply_crt_post_rgb(rgb, scan, vig, rng_key)` in scope_draw.py
  (deterministic, HxWx3 uint8, no in-place mutate) -- the CRT-glue path.
- Radius cap to avoid 16:9 clip: outer CORE rings <= ~0.33*min(w,h); only the outer spectrum band bleeds.
- Tests: `importorskip("cairo")` gates ONLY paint/determinism/visual-smoke; registration/capability/wiring
  /cold-import stay cairo-free; the missing-cairo assert_usable test stays UNSKIPPED via monkeypatched
  import machinery.
