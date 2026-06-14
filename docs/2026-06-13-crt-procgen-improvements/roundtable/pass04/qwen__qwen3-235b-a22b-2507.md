<!-- requested_model: qwen/qwen3-235b-a22b-2507 | resolved_model: qwen/qwen3-235b-a22b-07-25 -->

- VERDICT: build-ready as-is? **no**. Signal/EMA state mismanaged; timing resolution unsafe; title card state machine conflicts with section-1 suppression.

- MUST-FIX BEFORE BUILD:
1. [S1, Concept] `self._brightness_ema` is reused for two EMAs (`signal` and `trig`) but `_CRTRenderer.__init__` only initializes one scalar. Precomputing dual EMAs requires two arrays or a structured buffer — not a single float. **Fix:** In `__init__`, allocate `self.signal = np.zeros(total)` and `self.trig = np.zeros(total)`, then fill in `__init__` using vectorized EMA from `volume` with respective alphas (0.05, 0.3); do not reuse `_brightness_ema` state.
2. [Timing extractor] `start_s` derived from ledger may be `None`, but `round(start_s*fps)` will crash. The fallback logic is described but not safely ordered: clamp/round must not precede null check. **Fix:** In timing extractor, first test `if start_s is None: derive from volume`; only then compute `start_f = max(0, min(total, round(start_s * fps)))`.
3. [S3, #1] Title card state machine disables section-1 draw during active window, but section-1 ident is *also* the dock target. Docking interpolates into section-1 coords — which are not drawn if suppressed. **Fix:** Do not skip section-1 draw; instead, draw title card *over* section-1 during `[music_open_start_f, dock_frames)` and suppress only the *original* ident/subtitle/text elements *within* section-1, leaving the layout space intact for docking.
4. [Determinism] `seed = int.from_bytes(hashlib.blake2s(...).digest()[:8], "big")` uses undefined `salt`. Document defines `salt` nowhere. **Fix:** Define `salt` in signature or pass explicitly; e.g., add `seed_salt=""` to `_CRTRenderer.__init__` and use it in the hash.

- SHOULD-FIX:
5. [Geometry] `left_cx~=323`, `right_cx~=1596` assumes fixed W=1920; code uses `w//2` for center. **Fix:** Use `gutter_width = (w - center_band_width) // 2` and compute `left_cx = gutter_width // 2`, `right_cx = w - gutter_width // 2` to respect actual layout.
6. [OPEN] Coordinate offset drift must be applied *only* to scope/title draw calls, not to section-4 grid or any center element. The spec says this, but grounding code shows no such clipping. **Fix:** In `render`, apply offset inside `_draw_fft_scope` and `_draw_scope` only; guard with comment: `[APPLY OFFSET ONLY IN GUTTERS]`.
7. [S2] `_precompute_graticules()` is specified but not implemented. **Fix:** Add method to `_CRTRenderer` that returns RGBA image of ticks/crosshair for each gutter, drawn once in `__init__`.

- OPTIONAL / NICE-TO-HAVE:
- Clarify whether `trig` EMA uses `volume` or `abs(volume - mean)` for glitch detection; current spec implies raw `volume`.
- Add `clamp(amp <= r * 0.35)` explicitly in `_draw_fft_scope` and `_draw_scope` to prevent center incursion.

- CUT THESE (over-engineering):
1. [v1 CUTS] Telemetry micro-text labels — already cut per consensus; do not reintroduce.
2. [v1 CUTS] FFT peak-hold ghost ring — comet-tails suffice; adding would increase CPU cost.
3. [v1 CUTS] Halation — explicitly rejected due to 2x draw pass cost.

[ASSUMPTION] The `led["lines"]` structure is available in `render_video` but not passed to `_CRTRenderer`; timing must be resolved upstream and passed as `timing` dict to constructor.
[ASSUMPTION] `dock_frames` refers to a fixed tail duration (e.g., 10 frames), not a global timeline slot.
[ASSUMPTION] `volume`, `freqs`, `waves` are precomputed for all frames and indexed by `fi`; no streaming constraints.
[ASSUMPTION] `textbbox` overflow case means title too wide for hero card — handled by scaling, not truncation.
verify: (1) actual `start_s` field presence in real ledger; (2) determinism checksum method; (3) long-title overflow behavior.