<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. The plan contradicts the grounded code in multiple places (signature, render path, timing access, EMA, randomness) and leaves unhandled failure modes in the ledger and geometry.

MUST-FIX BEFORE BUILD:
1. [Concrete specs / Signature] Plan states new `_CRTRenderer(w,h,title,volume,freqs,waves,fps,timing=None)` and changes render to `render(self,fi)`; grounded code (Excerpt 1 + Excerpt 3 L1556) has `__init__(self,w,h,title)` and `render(self,fi,total,fps,vol,freq,wave)`. Fix: keep the existing signatures exactly; move array storage and timing resolution into the existing `render_video` caller only.
2. [Concrete specs / Timing extractor] Plan claims `_CRTRenderer` can resolve `led["lines"]` with `speaker_role`/`start_s`; grounded code (Excerpt 2) shows `led` is only in `render_video` scope and `_CRTRenderer` never receives it. Fix: perform the `music_open`/`music_close`/`first_dialogue_f` extraction once in `render_video` and pass only the resulting frame windows (or None) to the renderer.
3. [Concrete specs / Determinism] Plan requires `import hashlib` + `np.random.default_rng(seed)` + replace section-8 `np.random.randint`; grounded code (Excerpt 1 section 8) still uses bare `np.random.randint`. Fix: add the seeded RNG inside the existing noise block only; do not change any other call sites.
4. [Concept + S1] Plan re-enables `_brightness_ema` as dual slow/fast EMA precomputed in `__init__` from full arrays; grounded code (Excerpt 1) stores a single scalar EMA updated inside `render`. Fix: keep the scalar and compute `signal`/`trig`/`loss` on the fly from the per-frame `vol` argument; never store full arrays inside the renderer.

SHOULD-FIX:
1. [Concrete specs / Geometry] Plan hard-codes gutter centers `left_cx~=323`, `right_cx~=1596`, `r~=235` and `amp <= r*0.35`; grounded code has only center `_ring_*` values. Fix: compute gutter centers from `w` at runtime (`w//4`, `3*w//4`) and clamp `r` to `min(235, (w//4)-20)` so the spec works at both 1920 and any other width passed to the constructor.
2. [Concrete specs / Text exemption] Plan requires title/ident drawn after vignette choke; grounded code (Excerpt 1 sections 1 and 8) draws text before vignette and noise. Fix: move the section-1 title block (and the new card) to a post-vignette composite pass.
3. [S3 + Concrete specs / Intervals] Plan says "half-open `start <= fi < end`" and "disable if None or `end <= start`"; no handling shown for the case where `music_open_end_f` falls after `total_frames`. Fix: add explicit `end = min(end, total)` clamp before the window test.
4. [S2] Plan retires sections 5/6 calls and thins section 3; grounded code still contains the calls and the particle hue logic. Fix: delete the two helper calls and replace the hue ternary with a single brightness scale using `CRT_GREEN`.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line `if timing is None: return` guard in the window logic so a missing ledger never crashes a render.
- Emit the four build-time smokes as a short pytest that runs against a 10-frame synthetic volume array.

CUT THESE (over-engineering):
1. Dual EMA precompute of entire arrays in `__init__` (S1) -- safe to cut because per-frame `vol` already exists and the plan already forbids storing full arrays.
2. Formal `textbbox` wrap/scale step for long titles (section B) -- safe to cut because the spec already says "measure... before effects" and the existing font loader plus one `textbbox` call is enough; no new layout engine required.
3. Separate `_precompute_graticules` mirroring `_scanlines` (S2) -- safe to cut because a static 4-line crosshair can be drawn once per gutter at the same cost as the existing scanline precompute.