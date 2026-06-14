<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The timing interface is underspecified (unknown `led` field mapping), the render signature change and dock transition logic are not concretely defined, and the EMA initial value is missing — all would require on-the-fly decisions during implementation, risking build breaks or inconsistent behavior.

MUST-FIX BEFORE BUILD:
1. **[Timing plumbing / VERIFY-AT-BUILD]** The plan references `music_open_start_f`, `music_open_end_f`, etc. but never says how to extract them from `led`. Without a real schema, the `render_video` parsing code cannot be written. Fix: document the exact `led` structure (e.g. `led` is a dict with key `"beats"`, each beat has `id` and `start_frame`/`end_frame`; `music_open` is the beat with `id == "b000"`; `first_dialogue_f` is the start frame of the first dialogue beat, etc.). The statement "A missing/None field DISABLES that effect" is fine, but the mapping must be concrete.

2. **[render signature / `render_video` loop]** The plan says `render(fi)` becomes a pure reader, but the grounding excerpt shows `render(fi, total_frames, fps, volume[fi], freqs[fi], waves[fi])` called from `render_video`. If the constructor now receives the arrays, the render signature must be clearly stated as `render(self, fi)` and the caller in `render_video` must be updated. Fix: add an explicit note that `_CRTRenderer.render` changes to a single `fi` argument, and the `render_video` closure becomes `def _render_crt(fi): return renderer.render(fi)`. Also confirm that `total_frames` and `fps` remain accessible (they can be stored in `self` from constructor).

3. **[EMA initialization]** The plan says “reset EMAs at fi==0 (do NOT start half-locked at 0.5)” but does not give `signal[0]`. Fix: declare the initial EMA value (e.g., `signal[0] = vol[0]`, `trig[0] = vol[0]` or 0). This is necessary for the precomputation pass to produce deterministic arrays.

4. **[Dock transition logic]** The title card is drawn only when `fi in [music_open_start_f, music_open_end_f]`, but the plan describes a dock (raster collapse) that interpolates the hero bbox into the ident coordinates. This interpolation must happen *after* the window, or it’s a sudden jump. Fix: specify a transition period (e.g., `fi in [music_open_start_f, music_open_end_f + N]`) and how the card animates down to the corner. Without this, the title card will abruptly vanish.

5. **[Vignette exemption implementation]** “Exempts the title/ident text from any dimming” is stated, but no mechanism given. The current code multiplies the entire frame array by `_vignette` after drawing all elements. Fix: clarify whether text will be drawn *after* the vignette multiplication, or if a per-pixel mask will be used. This decision affects layer order and must be in the plan.

SHOULD-FIX:
1. **[Coordinate-offset bound]** The plan says “bound the offset so no black edge appears” but provides no formula. To avoid runtime guesses, define the maximum offset as the minimum distance from the drawn element’s bounding box to the frame edge, clamped symmetrically. This can be a TODO in code, but the plan should note it.

2. **[Center-band clipping method]** “Clip ALL bright scope geometry against it” is given, but the clipping algorithm (simple rejection vs. line clipping) is not specified. While implementation can fill this in, the plan should state that elements entirely outside the gutter band are dropped, and those crossing the boundary are truncated using a standard line-clipping routine (e.g., Cohen–Sutherland) to prevent partial draw.

3. **[fft_tips history structure]** The phrase “per-bin recent positions for comet-tails” is ambiguous. Specify the number of history points per bin (e.g., 4–6) and that they are stored as (angle, radius) pairs, consumed to draw a fading tail. This keeps memory predictable.

OPTIONAL / NICE-TO-HAVE:
- None.

CUT THESE (over-engineering):
- **Hierarchy clamp (“layer floors” S4).** The goal of dropping the grid before scopes can be approximated by simply scaling the existing grid alpha with signal strength, without a formal three-layer composite. The current plan already includes per-element brightness control via signal/loss, making a separate layer-floor system heavy for v1. Safe to cut; the “ident last” rule already preserves readability.

Mark [ASSUMPTION] where inferring: The field-name mapping assumption is the primary gap; the dock transition assumption that it extends beyond the window is also inferred. The plan must decide.