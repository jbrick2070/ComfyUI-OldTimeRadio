<!-- requested_model: mistralai/mistral-large-2512 | resolved_model: mistralai/mistral-large-2512 -->

VERDICT: no
The plan is conceptually sound but contains critical correctness and sequencing defects that would break determinism, timing, and the green-only blend invariant. It also over-engineers the EMA precompute and risks center-band incursion.

---

MUST-FIX BEFORE BUILD

1. [Concept / Signature]
   The plan assumes `_CRTRenderer` can be instantiated once with `title` and then called per-frame with `vol/freq/wave` arrays. The grounding code shows the constructor takes only `(w, h, title)` and the `render()` method takes `(fi, total, fps, vol, freq, wave)` as individual frame values.
   Fix: Change the signature to `_CRTRenderer(w, h, title, volume, freqs, waves, fps, timing=None)` and store the full arrays in `__init__` so `render(fi)` can slice them. This is the only way to guarantee determinism (the EMA must see the full envelope) and to avoid passing the same arrays repeatedly.

2. [S1 / Determinism]
   The plan says `signal[0]=trig[0]=volume[0]` but the grounding code shows `volume` is a list of floats, not a numpy array. The EMA precompute in `__init__` will fail if `volume` is not indexable.
   Fix: Convert `volume` to a numpy array in `__init__` so the EMA can be precomputed: `self.volume = np.array(volume, dtype=np.float32)`.

3. [S1 / EMA Precompute]
   The plan precomputes `signal` and `trig` in `__init__` but the grounding code shows the EMA update is done per-frame in `render()`. This is a sequencing defect: the EMA must be precomputed in `__init__` from the full `volume` array, not updated per-frame.
   Fix: Move the EMA precompute to `__init__`:
   ```python
   self.signal = np.zeros_like(self.volume)
   self.trig = np.zeros_like(self.volume)
   self.signal[0] = self.trig[0] = self.volume[0]
   for i in range(1, len(self.volume)):
       self.signal[i] = 0.95 * self.signal[i-1] + 0.05 * self.volume[i]
       self.trig[i] = 0.7 * self.trig[i-1] + 0.3 * self.volume[i]
   ```

4. [S3 / Title Card Timing]
   The plan says the title card window is `[music_open_start_f, music_open_end_f + dock_frames)` but the grounding code shows the ledger may have `start_s=None`. The plan does not specify how to derive `dock_frames` or what to do if `start_s` is missing.
   Fix: Add a fallback in `__init__`:
   ```python
   if timing and timing.get("music_open_start_s") is not None:
       self.music_open_start_f = round(timing["music_open_start_s"] * fps)
       self.music_open_end_f = round((timing["music_open_start_s"] + timing["music_open_dur_s"]) * fps)
       self.dock_frames = 30  # 1.25s at 24fps
   else:
       # Fallback: music from frame 0 to first dialogue onset
       first_dialogue_f = next((i for i, v in enumerate(self.volume) if v > 0.1), total_frames)
       self.music_open_start_f = 0
       self.music_open_end_f = min(first_dialogue_f, 96)  # cap at 4s
       self.dock_frames = 0
   ```

5. [S2 / Scope Geometry]
   The plan says "clamp the circular-scope amplitude `amp <= r*0.35`" but the grounding code shows the ring is drawn with `r = base_r + int(vol * base_r * 0.3)`. This would allow the ring to overflow the gutter if `vol` is high.
   Fix: In `_draw_fft_scope` and `_draw_scope`, clamp the amplitude:
   ```python
   amp = min(int(vol * r * 0.35), int(r * 0.35))
   ```

6. [S2 / Center-Band Clip]
   The plan says "draw each scope onto a transparent layer sized to its gutter rect and `alpha_composite` it" but the grounding code shows the scopes are drawn directly onto the base image. This risks center-band incursion.
   Fix: In `render()`, before drawing the scopes, create a transparent layer for each gutter:
   ```python
   left_scope_layer = Image.new("RGBA", (647, h), (0, 0, 0, 0))
   right_scope_layer = Image.new("RGBA", (647, h), (0, 0, 0, 0))
   ```
   Draw the scopes onto these layers, then `alpha_composite` them onto the base image.

7. [S4 / Text Exemption]
   The plan says "section-1 ident + the title card draw AFTER the section-8 vignette/choke multiply" but the grounding code shows the title bar is drawn in section 1, before the vignette. This would dim the text.
   Fix: Move the title bar draw to after the vignette multiply, or exempt it from the vignette by drawing it on a separate layer and compositing it last.

8. [S5 / Outro]
   The plan says "leave `_hud_frames` append unchanged" but the grounding code shows `_hud_frames` is appended after `total_frames`. The outro must not interfere with the HUD.
   Fix: In `render()`, add:
   ```python
   if fi >= total_frames:
       return _render_hud(fi - total_frames)
   ```

---

SHOULD-FIX

1. [S1 / EMA Alpha]
   The plan uses `alpha ~0.05` for `signal` and `alpha ~0.3` for `trig` but does not specify how these values are derived or if they are configurable. The grounding code shows the EMA alpha was hardcoded to 0.08.
   Fix: Make the alphas configurable in `__init__`:
   ```python
   def __init__(self, ..., signal_alpha=0.05, trig_alpha=0.3):
       self.signal_alpha = signal_alpha
       self.trig_alpha = trig_alpha
   ```

2. [S3 / Title Card Reveal]
   The plan says "decoded-fragment reveal stepping on INTEGER frames" but does not specify how the reveal is seeded or how long it takes. This could cause desync if the reveal duration is not fixed.
   Fix: Add a fixed reveal duration (e.g., 24 frames) and seed the reveal order with the episode title.

3. [S2 / Scope Trails]
   The plan says "bounded lookback over `freqs[fi-6:fi+1]`" but the grounding code shows `freqs` is a list of numpy arrays. Slicing a list of arrays is inefficient.
   Fix: Convert `freqs` to a numpy array in `__init__` so lookback is vectorized.

4. [S4 / Coordinate Offset]
   The plan says "a horizontal coordinate OFFSET applied to the gutter-scope + title DRAW coords only" but does not specify how the offset is bounded to prevent black edges.
   Fix: In `render()`, compute the offset as:
   ```python
   offset = int(10 * math.sin(fi * 0.1))  # example
   offset = max(-323, min(offset, 323))   # clamp to gutter width
   ```

---

OPTIONAL / NICE-TO-HAVE

- Add a debug mode that draws the gutter boundaries and scope layers for visual verification.
- Precompute the graticules in `__init__` to avoid per-frame draw calls.

---

CUT THESE (over-engineering)

1. [Concept / Dual EMA]
   The plan uses two EMAs (`signal` and `trig`) but the grounding code shows only one EMA was used. The second EMA adds complexity without a clear benefit.
   Why safe to cut: The "signal loss in the gaps" effect can be achieved with a single EMA and a threshold. The dual EMA is not necessary for the aesthetic.

2. [S3 / Title Card POP]
   The plan says "a 1-2 frame brightness bloom + a small horizontal coordinate tear". The tear is over-engineering: a brightness bloom alone is sufficient for the "lock" effect.
   Why safe to cut: The tear adds complexity and risks center-band incursion. The bloom is enough to convey the effect.

3. [S2 / Scope Graticules]
   The plan precomputes graticules as RGBA layers. The grounding code shows the graticules are drawn per-frame with `ImageDraw.line`. Precomputing them is unnecessary.
   Why safe to cut: The graticules are simple lines; precomputing them saves negligible time and adds complexity.

---

[ASSUMPTION] The ledger `timing` dict is passed to `_CRTRenderer` in the new signature. Verify the ledger structure contains `music_open_start_s` and `music_open_dur_s`.
[ASSUMPTION] The `render_video` caller will be updated to pass the `volume`, `freqs`, `waves`, and `timing` arrays to `_CRTRenderer`. Verify this change.