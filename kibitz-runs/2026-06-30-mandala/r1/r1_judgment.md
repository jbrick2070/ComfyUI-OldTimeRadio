# r1 JUDGMENT -- mandala engine (arc / creative + dependency shape)

Panel: Codex (VERDICT no -- shape/dep/wiring unresolved), Antigravity (yes-with-fixes -- isolate pycairo).
Claude anchor: sound+proven, 3 arc decisions. All grounded vs the real Windows files.

## GROUNDED CONFLICT RESOLUTION
- **fallback vs fail-loud (Codex #2 vs Antigravity #2).** GROUNDED in render_driver.py: the driver DOES
  walk `fallback_engine` chains (line 158) and restamps the ledger LOUDLY (docstring L5-9); the
  "fallbacks are disabled" raise (L1557) fires ONLY for engines declaring `fallback_engine=None`
  (e.g. ltx_audio_in, L58). => They compose, they don't conflict. DECISION: viz_mxc_mandala declares
  BOTH -- `assert_usable` fails LOUD at ELIGIBILITY if pycairo is absent (Codex), AND
  `fallback_engine="viz_mxc_cpu"` as a LOUD render-time terminus (Antigravity). The zero-dep PIL
  visualizer is the perfect terminus; the swap is logged + ledger-restamped, never silent.
- **pycairo in the capability table (Codex #4).** GROUNDED: capability_profiles.py is fail-closed and
  REJECTS unknown declaration keys (L266); allowed keys are required_toolchain/model_requirements/etc
  (L244-247). => pycairo handling stays in requirements/docs/assert_usable ONLY; the CAPABILITIES row
  uses required_toolchain=None (mirror viz_mxc_cpu). No new schema field.

## LOCKED FOR THE BUILD (r1 convergence)
1. SEPARATE engine `viz_mxc_mandala`; KEEP viz_mxc_cpu as the zero-dep selectable alternate. CUT the
   "upgrade viz_mxc_cpu in place" branch entirely (Codex CUT #1).
2. pycairo = NOT added to requirements.video.txt (pip pycairo needs system libcairo -> breaks clean
   Linux/mac installs of OTHER engines; Antigravity #1). Lazy `import cairo` INSIDE render_clip only
   (V-12 cold-import clean). assert_usable: import-probe -> FAIL LOUD "pip install pycairo" if absent.
   Plus fallback_engine="viz_mxc_cpu".
3. Grammar (one sentence, Codex #6): centered tuning-eye mandala FIRST; radio-dial rings/spokes SECOND;
   CRT scanlines/vignette/grain as POST treatment only. Muted iridescence. Non-goals: no creatures, no
   portals, no lissajous, no mode-switch widget.
4. cairo->numpy: read `surface.get_stride()` (do NOT assume w*4; Antigravity #3 + anchor); paint an
   OPAQUE bg first so premultiplied-alpha == straight; assert surface opaque.
5. `ctx.save()/ctx.restore()` around the paint body (or fresh Context per frame) -- no state leak
   (Antigravity #3-SHOULD).
6. CRT grain seed-keyed via scope_draw `rng_key` (V-7 determinism; Codex SHOULD #4).
7. New CAPABILITIES row `viz_mxc_mandala` (cpu_ok, required_toolchain None, model_requirements []) or
   test_capability_profiles breaks (Antigravity #4-SHOULD).
8. DROP static-layer caching (Antigravity CUT + anchor concession): overlays already build once/run,
   the cairo radial gradient is cheap, and scanlines/vignette must sit ABOVE reactive layers.

## CARRIED TO r2 (coding plan)
- Perf: set a NUMERIC budget (max ms/frame, max s per 25-frame beat) + benchmark mandala vs
  viz_mxc_cpu (Codex SHOULD #1). Decide CRT glue: PIL-roundtrip (proven, simple) vs native-cairo
  OPERATOR_MULTIPLY (Antigravity SHOULD #1, faster) -- pick by the measured budget.
- 16:9 bounding: cap outer radius so the tuning-eye + rings aren't clipped; the outer spectrum BAND
  ring may intentionally bleed to the edges (operator asked for denser/thicker 2026-06-30). Confirm the
  denser v2 (48 wedges + 9 rings) reads well before locking radii.
- Tests: mirror test_video_viz_rainbow.py PLUS a visual-acceptance smoke (nonblack ratio, frame-to-frame
  delta, deterministic hash) so a dull/static mandala is caught (Codex SHOULD #2).

## CARRIED TO r3 (wiring)
- Reachability (Codex #5, CLAUDE.md 0 dead-code): register in the video dropdown enum (registry-driven)
  + the ENGINE_FAMILY/content_oracle/_uses_ambient_master_audio maps (same set viz_mxc_cpu touched).
  Decide whether any saved widget in otr_scifi_16gb_full.json defaults to it (likely no -- opt-in
  selectable), but the dropdown MUST list it. Re-validate JSON + link/widget audit.
