# r3 JUDGMENT -- mandala engine (wiring)

Panel: Codex (no -- workflow-unwired + 2 render-contract gaps), Antigravity (yes-with-fixes -- schema
keys + BGRA->RGB + missing helper step). Claude anchor: wiring mirrors viz_mxc_cpu; 2 real decisions.
All grounded. STRONG CONVERGENCE -- no arc changes, all coding-detail locks.

## LOCKED (r3 wiring)
1. **Opt-in SELECTABLE engine (not a saved-widget default).** Grounded: node 87 OTR_VideoDirector saves
   "visualizer"/"humo_14B_169" (workflows/otr_scifi_16gb_full.json). Registering viz_mxc_mandala makes it
   selectable in the registry-driven dropdown (nodes/otr_video_director.py) -- that satisfies reachability
   (CLAUDE.md 0); it does NOT auto-use it. DROP "production engine" language. The operator MAY later set
   the music/title-bookend widget to viz_mxc_mandala (their stated "radio bookend" vision) -- OFFER that
   as a taste call, do NOT auto-set. Run OTR_WorkflowValidator + JSON round-trip + link/widget audit
   regardless.
2. **assert_usable probes BOTH cairo AND ffmpeg** (Codex #2): `import cairo` + `_sd.find_ffmpeg(
   os.environ.get("OTR_FFMPEG","ffmpeg"))` (eng_viz_rainbow already preflights ffmpeg at L81-86), with
   SEPARATE loud messages (missing pycairo vs missing ffmpeg).
3. **surface->rgb handoff** (Codex #3 + Antigravity #2, unanimous): encode_silent_mp4 writes rgb24 HxWx3.
   So: `bgra = np.ndarray((h,w,4), np.uint8, buffer=surface.get_data(), strides=(stride,4,1))` then
   `rgb = np.ascontiguousarray(bgra[:, :, [2,1,0]])`; assert `rgb.shape == (h,w,3)` and dtype uint8.
   (Stride from get_stride() handles non-w*4 alignment; the copy makes it owned+contiguous.)
4. **CAPABILITIES row = FULL _DECL_KEYS dict** (Antigravity #1, capability_profiles.py L260-273 is
   fail-closed on missing keys):
   `{"vram_class":"cpu","vram_estimate_mb":0,"required_toolchain":None,"requires_sidecar":False,
   "cpu_ok":True,"model_requirements":[]}`.
5. **Add `apply_crt_post_rgb(rgb, scan, vig, rng_key)` to scope_draw.py** as an EXPLICIT build step
   (Antigravity #3 -- it was missing from the wiring list; without it the engine AttributeErrors).
   Deterministic (rng_key), returns HxWx3 uint8, does not mutate input.
6. **otr_video_soak.py ENGINE_FAMILY: add BOTH `viz_mxc_cpu` + `viz_mxc_mandala`** (Antigravity SHOULD #1
   -- viz_mxc_cpu is a confirmed latent gap in the soak copy; close the drift). Do NOT add a soak
   `_PROFILES` leg (Codex SHOULD #2): the engine is opt-in CPU + cheap; soak coverage = the dedicated
   render-contract test, not a soak rotation leg. State this explicitly.

## TESTS (r3)
- `importorskip("cairo")` gates ONLY the real-paint tests (determinism, visual-smoke); registration /
  capability-row / wiring-map / cold-import tests stay cairo-FREE (engine import must be cairo-free, V-12).
- The missing-cairo `assert_usable` test stays UNSKIPPED via monkeypatched import machinery (Codex SHOULD
  #1) so cairo-less CI still proves the loud message.
- Optional: apply_crt_post_rgb contract test (same seed==, diff seed!=, HxWx3 uint8, no in-place mutate).

## VERDICT: wiring converged. Carry to r4 only a "no new must-fix" confirmation.
