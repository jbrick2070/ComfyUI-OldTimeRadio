<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan has unresolved timing semantics, contradictory draw/order constraints, and at least one concrete slicing bug that will break early-frame trails.

MUST-FIX BEFORE BUILD:
1. [Concrete specs / Timing extractor] “music-open role”, “dialogue line”, and “last music line” are not concretely defined. The plan says `speaker_role` exists, but gives no accepted values or fallback priority, and the grounding only says the ledger carries per-beat/line timing and that opening music is `b000`. Concrete fix: define exact role/id predicates in the spec, e.g. priority order `beat_id == "b000"` or `speaker_role in {...}` for open, `speaker_role in {...}` for dialogue, and explicit close-role detection. Missing/unknown roles must disable the card, not guess.

2. [Concrete specs / Timing extractor] The fallback is underspecified and partially circular: “derive the intro window from the `volume` envelope … to the first dialogue onset” but `first_dialogue_f` also depends on `start_s`, which may be unavailable. “capped” is also not defined. Concrete fix: specify a pure-volume fallback with numeric threshold, minimum/maximum duration, and cap, e.g. `start=0`, `end=min(total, round(fps * 6.0))` unless a valid dialogue `start_s` exists.

3. [#2 / LEFT `_draw_fft_scope`; RIGHT `_draw_scope`; Concrete specs / Trails are pure] The specified slices `freqs[fi-6:fi+1]` and lookback over `waves` are wrong for early frames: for `fi < 6`, Python negative starts can produce an empty or tail-based slice depending on list length. Concrete fix: use `lo = max(0, fi - N)` and slice `freqs[lo:fi+1]` / `waves[lo:fi+1]`.

4. [Concrete specs / No `np.roll` on the frame] The clamp rule contradicts the title-card requirement. It says drift/tear offsets apply to “gutter-scope + title DRAW coords only” and are “clamped so each bbox stays inside its gutter”; the title card is not a gutter element and cannot be clamped inside a gutter. Concrete fix: define separate clamp regions: scopes clamp to their local gutter rects; title-card tear clamps to the title/card bbox or protected center/title region, never to the gutter.

5. [#1 / Concrete specs / Text exemption] The plan conflicts on draw ordering. [#1] says if active, “skip the normal ident/subtitle/timestamp and draw the card” before section 1 draws; [Text exemption] says section-1 ident and title card draw after the section-8 vignette/choke multiply. Concrete fix: separate “state decision before section 1” from “actual text draw pass after vignette/choke.” Define one ordering, e.g. background/scopes/grid/noise base -> vignette/choke -> exempt text/title-card pass -> scanlines/noise if desired.

6. [Concrete specs / Text exemption] Only “section-1 ident + the title card” are exempted, but [#1] says active card suppresses ident/subtitle/timestamp, and current code also has timestamp and bottom HUD text. If the goal is “choke can never dim text,” the exemption set is incomplete. Concrete fix: explicitly list which text is post-vignette: ident, episode subtitle, timestamp, title-card text, and whether bottom bar/frame counter remain dimmable.

7. [S2 / #2] The sprint says “retire sections 5/6” and “thin section 3,” but the old center frequency ring is current render section 2 and must also be removed/disabled. [#2] says “replace center ring,” but S2 omits the concrete removal step. Concrete fix: explicitly delete/disable current section 2 center ring drawing when adding gutter scopes, otherwise the center-band remains polluted.

8. [Concrete specs / Determinism] The seed formula is `f"{title}|{fi}|{salt}"`; it does not include any render seed, despite the document claiming “deterministic per seed.” Grounding excerpts show `_CRTRenderer` currently receives only `(w, h, title)` and the proposed signature still has no seed. Concrete fix: either change the claim to “deterministic per title/frame/effect” or pass an existing render seed into `_CRTRenderer` if one exists. [ASSUMPTION] Verify whether `render_video` has an existing seed variable outside the excerpt.

9. [Concrete specs / Determinism] `signal[0]=trig[0]=volume[0]` will crash if `volume` is empty. Existing `_analyze_audio` normally returns `total_frames` entries, but zero-duration or malformed audio would make this unsafe. Concrete fix: if `len(volume) == 0`, set `self.total = 0`, create empty arrays, and make `render(fi)` raise/return a safe blank frame only if called invalidly.

10. [#1 / Outro bookend; Concrete specs / Outro] “same logic on `music_close_*` if it resolves” is not sufficiently specified and can misfire on the opening music if it is also the last music line. Concrete fix: require close music to be distinct from open and after first dialogue, or require an explicit close role/id; otherwise disable outro.

SHOULD-FIX:
1. [Concrete specs / Geometry] The geometry uses hardcoded 1920-derived gutter numbers (`[647,1273]`, `left_cx~=323`, `r~=235`) while the real code accepts arbitrary `resolution.split("x")`. Concrete fix: compute center band from `h * 480 / 832`, centered in `w`, then derive gutters/radius from that. If 1920x1080 is the only supported procgen resolution, state that and assert it.

2. [Concrete specs / Timing extractor] “Resolve it the way `otr_caption_burn` does” is a hidden dependency outside the one-file scope and not grounded in the excerpts. Concrete fix: either inline the needed ledger-resolution logic in `nodes/video_engine.py` or mark the exact helper/API to reuse. [ASSUMPTION] Verify that `otr_caption_burn` logic is callable from this file without cross-file changes.

3. [#1 / B. HERO title] “wrap/scale long titles to a max bbox” lacks bounds: max bbox, minimum font size, line count, and cursor placement are unspecified. Concrete fix: define max width/height as frame ratios and a minimum font size; if still overflowing, ellipsize.

4. [S4 / Envelope behaviors] “vignette choke (bounded/floored/text-exempt)” gives no numeric floor or formula. Concrete fix: specify the multiplier range, e.g. grid/background floor versus text exemption, so the coder does not recreate the v1.5.1 unreadable-text bug.

5. [#2 / Graticules] `_precompute_graticules()` must be parameterized per gutter/radius, not a single full-frame overlay, otherwise alpha-compositing into local gutter layers becomes ambiguous. Concrete fix: precompute left/right local RGBA graticules or one local graticule reused with local center coordinates.

6. [Concrete specs / Signature] `self.total = len(volume)` silently trusts `freqs` and `waves` to be the same length. Concrete fix: clamp `self.total = min(len(volume), len(freqs), len(waves))` or validate and fail early with a clear log warning. Current `_analyze_audio` returns equal lengths, but this hardens future callers.

7. [#1 / C. lock POP] “1-2 frame brightness bloom” needs to say whether bloom applies before or after green-only/no-hue constraints and whether it affects exempt text. Concrete fix: make it a green-channel/intensity multiplier on only title-card primitives, not the whole frame.

8. [S3] “the 2-beat gap-fill smoke” is listed as an implementation item but is not specified elsewhere. Concrete fix: define what is drawn, when it appears, and how it is bounded; or move it to test/verify only.

9. [Concrete specs / Center-band clip] The spec says to draw scopes on transparent gutter layers, but drift/tear offsets also need local-coordinate clipping. Concrete fix: define all scope helper coordinates as local to the gutter layer, then composite at `(gutter_x0, 0)`.

OPTIONAL / NICE-TO-HAVE:
1. Add a small internal debug flag or log line for resolved `music_open`, `first_dialogue`, and `music_close` frame intervals. Keep it non-UI and off by default.
2. Add a unit-ish smoke that renders frames `0..8` to catch the early-frame trail slicing bug.
3. Add one checksum smoke with noise enabled, since current section 8 uses nondeterministic global `np.random.randint`.

CUT THESE (over-engineering):
1. [S5 / #1 Outro bookend] Cut the conditional outro bookend from v1. It depends on unresolved close-music ledger semantics and is not necessary for the stated core upgrade: opening title card plus gutter scopes. Safe to cut because the open card and scopes deliver the feature without touching post-roll/HUD timing.

2. [S4 / coordinate-offset sync-drift] Cut generalized “sync-drift” in v1 and keep only the title-card POP tear plus scope-local jitter. General drift introduces clamp/center-incursion risk and conflicts with the no-`np.roll` rule. Safe to cut because the scopes and title reveal still function without global drift.

3. [S3 / 2-beat gap-fill smoke] Cut unless it is specified. It is currently not wired to a concrete interval or visual primitive and adds another timing dependency. Safe to cut because the title-card open window already covers the b000 intro objective.

4. [#2 / faint orbit particles] Cut section-3 particles entirely instead of thinning them. The new dual scopes replace the center visualizer role, and particles are currently hue-shifting center content. Safe to cut because removing them better satisfies “center stays clean.”