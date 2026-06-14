<!-- requested_model: mistralai/mistral-large-2512 | resolved_model: mistralai/mistral-large-2512 -->

VERDICT: no

The plan is conceptually strong but contains critical correctness and sequencing defects that would break determinism, timing, and the green-only blend invariant. Without fixes, the build will fail or produce incorrect output.

---

### MUST-FIX BEFORE BUILD

1. **[S1 / Signature]**
   Defect: The plan changes `_CRTRenderer.__init__` to precompute `signal/loss/trig` from "the full arrays", but the grounding code shows the arrays (`vol`, `freq`, `wave`) are only available at `render_video` time (L1556), not at `__init__` time.
   Fix: Precompute the EMAs in `render_video` after `_analyze_audio` and pass them as new arrays (`signal`, `trig`) to `_CRTRenderer.__init__`. Update the signature to `_CRTRenderer(w, h, title, signal, trig, ...)` and store them as `self.signal`, `self.trig`.

2. **[S1 / Determinism]**
   Defect: The plan seeds the RNG with `f"{title}|{fi}|{salt}"`, but the grounding code shows the title is passed to `__init__` and the frame index `fi` is passed to `render(fi)`. The RNG must be seeded per-frame inside `render(fi)` using the stored `self.title` and the current `fi`.
   Fix: Move the seeding logic into `render(fi)` and use `self.title` and `fi` to seed the RNG. Do not seed in `__init__`.

3. **[S1 / EMA Initialization]**
   Defect: The plan states `signal[0]=trig[0]=volume[0]`, but the grounding code shows `volume` is a list of floats, not a numpy array. The EMA precompute must handle the first frame explicitly.
   Fix: In the EMA precompute loop, set `signal[0] = trig[0] = volume[0]` and then run the EMA update for `i >= 1`.

4. **[S2 / Gutter Geometry]**
   Defect: The plan specifies `left_cx~=323`, `right_cx~=1596`, `r~=235` and clamps `amp <= r*0.35`, but the grounding code shows the ring geometry is hardcoded to the center (`self._ring_cx = w // 2`). The gutter scopes must be drawn in their own coordinate systems, not the center ring’s.
   Fix: Add new geometry fields in `__init__`: `self.left_cx`, `self.left_cy`, `self.left_r`, `self.right_cx`, `self.right_cy`, `self.right_r` and compute them from the gutter rects. Use these for the scope draws.

5. **[S2 / Center-Band Clip]**
   Defect: The plan says "draw each scope onto a transparent layer sized to its gutter rect and `alpha_composite` it", but the grounding code shows the scopes are drawn directly onto the base image. The layer bounds must clip the scopes to the gutter.
   Fix: In `render(fi)`, create two transparent layers (`Image.new("RGBA", (gutter_width, h))`), draw the scopes onto them, and `alpha_composite` them onto the base image.

6. **[S3 / Title Card Timing]**
   Defect: The plan uses `music_open_start_f` and `music_open_end_f` but the grounding code shows the timing is resolved from the ledger in `render_video`, not inside `_CRTRenderer`. The renderer must receive the resolved intervals as frame indices, not seconds.
   Fix: In `render_video`, resolve the intervals to frame indices (`start_f = round(start_s * fps)`, `end_f = round((start_s + dur_s) * fps)`) and pass them to `_CRTRenderer.__init__` as `music_open_start_f`, `music_open_end_f`, `music_close_start_f`, `music_close_end_f`. Use these in `render(fi)` to gate the title card.

7. **[S3 / Text Exemption]**
   Defect: The plan says "section-1 ident + the title card draw AFTER the section-8 vignette/choke multiply", but the grounding code draws the title bar (section 1) before the vignette (section 8). The text exemption must be implemented as a post-vignette pass.
   Fix: Split the title bar and title card draws into a separate `Image` layer, apply the vignette to the base image, then `alpha_composite` the text layer on top.

8. **[S4 / Coordinate Offset]**
   Defect: The plan says "a horizontal coordinate OFFSET applied to the gutter-scope + title DRAW coords only", but the grounding code shows no offset mechanism. The offset must be computed per-frame and clamped to keep the scopes inside their gutters.
   Fix: In `render(fi)`, compute the offset from the `trig` EMA (e.g., `offset = int((1 - trig[fi]) * 20)`), clamp it to `±(gutter_width - (r + amp))`, and apply it to the scope and title card draw coordinates.

9. **[S5 / Outro Regression]**
   Defect: The plan says "render only for `fi < total_frames`; leave `_hud_frames` append unchanged", but the grounding code shows the HUD frames are appended after `total_frames`. The outro must not render into the HUD frames.
   Fix: In `render(fi)`, gate the outro logic with `if fi >= total_frames: return img` before any outro-specific draws.

---

### SHOULD-FIX

1. **[S1 / EMA Alpha Values]**
   Defect: The plan specifies `alpha ~0.05` (slow) and `alpha ~0.3` (fast), but the grounding code shows the disabled EMA uses `alpha = 0.08`. The values must be consistent with the signal/loss hierarchy.
   Fix: Use `alpha_signal = 0.05` and `alpha_trig = 0.3` in the EMA precompute.

2. **[S2 / Scope Line Widths]**
   Defect: The plan says "cap all scope line widths to 1-2px", but the grounding code shows the center ring uses `width=max(2, self.w // 400)` (4px at 1920). The scopes must use 1-2px.
   Fix: Hardcode the scope line widths to 1 or 2 in the scope draw helpers.

3. **[S3 / Title Card Wrap]**
   Defect: The plan says "wrap/scale long titles to a max bbox before effects", but the grounding code shows no wrap/scale logic. The title card must handle overflow.
   Fix: In the title card draw, measure the title with `ImageDraw.textbbox`, and if it overflows, reduce the font size or wrap the text to fit the gutter width.

4. **[S4 / Brightness Hierarchy]**
   Defect: The plan says "per-element brightness hierarchy", but the grounding code shows brightness scaling is ad-hoc per section. The hierarchy must be explicit.
   Fix: Define a brightness scale per element (e.g., `grid_brightness = signal[fi] * 0.7`, `ident_brightness = signal[fi] * 0.9`) and apply it consistently.

5. **[S5 / Determinism Checksum]**
   Defect: The plan says "a determinism checksum over RGB frames", but the grounding code shows no checksum logic. The checksum must be implemented in `render_video`.
   Fix: After rendering all frames, compute a checksum of the RGB data (e.g., `hashlib.blake2s(frame.tobytes()).hexdigest()`) and log it.

---

### OPTIONAL / NICE-TO-HAVE

- Add a debug mode that renders the gutter rects and scope bounding boxes for visual verification.
- Precompute the graticules as RGBA layers in `__init__` to reduce per-frame cost.
- Add a fallback for missing fonts (the grounding code already has a fallback to `ImageFont.load_default()`).

---

### CUT THESE (over-engineering)

1. **[Dual EMA Precompute in `__init__`]**
   Why: The EMA arrays can be precomputed in `render_video` and passed to `__init__`, avoiding the need to pass the full `volume` array to the renderer. The plan’s precompute in `__init__` is unnecessary complexity.

2. **[Seeded RNG for Section 8 Noise]**
   Why: The grounding code shows the noise in section 8 is already seeded by `np.random.randint`, which is deterministic per frame. Replacing it with a seeded RNG is redundant.

3. **[Title Card Fake-Bold Overstrike]**
   Why: The grounding code shows the title is drawn with a monospace font, and fake-bold may not be necessary for readability. The overstrike offsets can be removed if the font is sufficiently bold.

4. **[Title Card Block Cursor]**
   Why: The block cursor is a minor aesthetic detail that adds complexity to the reveal logic. It can be cut without losing the core effect.

5. **[Scope Comet-Tails Lookback]**
   Why: The lookback adds visual polish but requires storing 6 previous frames of data. A simpler decay (e.g., `brightness = freq[i] * (1 - lookback_factor)`) can achieve a similar effect with less code.