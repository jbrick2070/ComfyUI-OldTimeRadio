<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan is not build-ready: S2 explicitly violates the grounded gutter-only constraint, and the core renderer interface/timing contract is internally inconsistent.

MUST-FIX BEFORE BUILD:
1. [S2 / Landscape commitment] The plan says `_CRTRenderer` is beat-agnostic and “scopes draw on every frame,” treating the landscape constraint as “void for the floor layer.” This violates grounded hard constraint #2: gutter scopes are only safe on portrait beats because landscape b-roll has no gutters. Concrete fix: before implementing S2, parse portrait-vs-landscape/beat intervals in `render_video` from the existing ledger/manifest data available there [ASSUMPTION: ledger has enough beat/media type data], pass those frame intervals into `_CRTRenderer`, and suppress the gutter scopes on full-frame landscape intervals. If that data is not available in `nodes/video_engine.py`, cut S2 scopes from v1 rather than drawing them on every frame.

2. [ARCHITECTURE FIX / TIMING PLUMBING / S1] The constructor contract contradicts itself. ARCHITECTURE says pass full `volume/freqs/waves` arrays into `_CRTRenderer.__init__`; TIMING says only extend to `_CRTRenderer.__init__(w, h, title, timing=None)`. Grounded code currently instantiates `_CRTRenderer(W, H, episode_title)` and calls `renderer.render(fi, total_frames, fps, volume[fi], freqs[fi], waves[fi])`. Concrete fix: define one explicit signature, e.g. `_CRTRenderer(w, h, title, volume, freqs, waves, fps, timing=None, portrait_intervals=None)`, store `self.total = len(volume)`, precompute arrays there, and update Excerpt 3 plumbing to instantiate with those arrays. Then either reduce `render()` to `render(fi)` or keep the old signature only as a compatibility wrapper that ignores mutable state.

3. [TIMING PLUMBING / S3 / S5] Timing interval semantics are ambiguous and currently unsafe. The plan says clamp fields to `[0, total_frames)` but also invokes title card when `fi in [music_open_start_f, music_open_end_f]`, which reads as inclusive. That makes `end_f` ambiguous and prevents clean representation of an interval ending at `total_frames`. Concrete fix: use half-open intervals everywhere: `start <= fi < end`; clamp starts to `[0, total_frames]`, ends to `[0, total_frames]`; disable if `start is None`, `end is None`, or `end <= start`.

4. [TIMING PLUMBING / VERIFY-AT-BUILD] The plan leaves exact `led` field names for `b000`, opening music bounds, and first-dialogue unresolved. That is not sprint-ready because S3/S5 cannot be coded reliably from the spec. Concrete fix: before build, document the exact extractor against the real `led` shape, including fallback behavior for missing/renamed fields. If `first_dialogue_f` is not actually used by the title-card state machine, remove it from the required timing contract.

5. [DETERMINISM] The proposed deterministic RNG uses `hashlib.blake2s`, but grounded imports in `nodes/video_engine.py` do not include `hashlib`. Concrete fix: add `import hashlib` and replace every current `np.random.randint` use in `_CRTRenderer.render()` with the local seeded generator. Grounded code Excerpt 1 currently has `np.random.randint(...)` in section 8, so missing even one call keeps nondeterminism.

6. [24fps / render_video plumbing] The plan repeatedly says 24fps is fixed, but grounded code computes `total_frames = ceil(duration * fps)` and passes the runtime `fps` through to render. Concrete fix: either enforce `fps == 24` in `render_video` before analysis/encoding, or explicitly state and implement that this procgen path accepts the existing `fps` parameter. Do not leave “24fps” as an assumption while using a variable clock.

7. [S3 / gap-fill claim] The plan asserts that `_floor_aligned` makes title-card frames land only at the head/open and not in mid-roll gap fills, but that behavior is not shown in the grounding excerpts. If false, the b000 title card can reappear wherever the procgen floor is reused from frame 0. Concrete fix: verify this against the real `OTR_SilentComposite` source before build, or add an explicit smoke/regression that renders a mid-roll gap and confirms no b000/title frames are sampled there.

SHOULD-FIX:
1. [ARCHITECTURE FIX / EMA] Define exact EMA initial conditions. “EMAs reset at `fi==0`” and “do NOT start half-locked at 0.5” is not enough. Concrete fix: specify `signal[0] = volume[0]` and `trig[0] = volume[0]` or `0.0`, then recurse from frame 1. Also specify behavior for empty audio / `total_frames == 0`.

2. [S2 / center-column sanctity] “Clip all bright scope geometry” is underspecified for Pillow. `ImageDraw` has no general clipping region for lines/arcs. Concrete fix: draw each gutter scope onto a transparent layer/mask sized to the allowed left/right gutter rectangles, alpha-composite that layer, and never draw scope primitives directly onto the base image.

3. [NO np.roll / S4] “Bound the coordinate offset so no black edge appears / no center incursion” needs a formula. Concrete fix: compute the allowed x-range per drawable bbox after applying offset, clamp `dx` so the bbox remains inside its gutter rectangle, and apply no offset to center grid/background.

4. [EMA IS READ-ONLY / S4] The current renderer applies `arr *= self._vignette[:, :, np.newaxis]` to the whole frame, including title/timestamp text. The plan says the new choke must exempt title/ident text, but does not specify layer ordering. Concrete fix: render background/grid/scopes through vignette/choke first, then draw title/ident/timestamp after that, or maintain a text mask and restore/draw text last. Otherwise S4 can recreate the v1.5.1 readability failure.

5. [S3] Title wrapping/scale-down needs hard bounds. Concrete fix: define max hero bbox width/height, minimum font size, max lines, and fallback truncation/ellipsis behavior for very long titles. Test against `ImageDraw.textbbox` with the actual loaded font and with the default bitmap fallback.

6. [S1 / determinism regression] “byte-identical re-render” should not depend on MP4 container bytes unless ffmpeg metadata is known deterministic. Concrete fix: checksum generated RGB frames or raw frame stream for visual determinism; separately verify the muxed audio stream is unchanged.

7. [S5] Outro rendering “inside existing `total_frames`” needs explicit precedence with `_hud_frames`. Concrete fix: define that `_render_crt(fi)` handles outro only for `fi < total_frames`; `_render_hud(hi)` remains unchanged and starts after `total_frames`. Add a test where close music overlaps the final few CRT frames and HUD append still starts at the original boundary.

8. [S2] Retiring sections 5 and 6 is stated but not mapped to code. Concrete fix: remove or bypass the calls to `_waveform_mirror(...)` and `_freq_bars_wide(...)` in grounded render sections 5/6. Leaving helpers defined is fine, but the render path must not still draw them.

OPTIONAL / NICE-TO-HAVE:
- Add a visual debug mode or temporary frame dump showing protected center band and gutter masks for one frame, then remove/disable before release.
- Keep a small deterministic seed test that renders the same `fi` out of order and compares pixels.
- Add a long-title fixture and a silent-audio fixture.

CUT THESE (over-engineering):
1. [ARCHITECTURE FIX] Full precompute of `self.fft_tips[fi]` and `self.sweep_idx[fi]` for every frame is heavier than necessary. Safe cut: precompute only `signal/loss/trig`; compute comet tails and sweep trails in `render(fi)` by bounded lookback over `freqs[max(0, fi-N):fi+1]` and `waves[fi]`. This remains pure and deterministic.

2. [TIMING PLUMBING] `first_dialogue_f` should be cut unless S3 actually uses it to cap/suppress the title card. The open interval alone is sufficient for the stated title-card trigger.

3. [S4] Halation should remain cut for v1 unless readability/perf is proven. It adds draw passes and layer ordering risk without being required for the title card or gutter scopes.

4. [S5] Outro bookend can be cut from the first build. It adds another timing window and interaction with HUD append, while the core upgrade goal is the b000 title card plus gutter scopes.