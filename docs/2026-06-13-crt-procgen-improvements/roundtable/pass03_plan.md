## 4C. CRT PROCGEN UPGRADE -- title card + gutter scopes (roundtable-hardened, SPRINT-READY)

> Roundtable-hardened 2026-06-13: 3 panels x 3 passes (gemini-3.1-pro + gpt-5.5 +
> deepseek-v4-pro; Claude judge/grounder; ~$0.40; raw reviews + judgments in
> `docs/2026-06-13-crt-procgen-improvements/roundtable/`). ONE file:
> `nodes/video_engine.py` (`_CRTRenderer` + the `render_video` plumbing). NO new
> node / widget / model / dependency; Pillow-only on CPU; the green-only `screen`
> blend is untouched; the audio spine is frozen (byte-identical); deterministic per
> seed. This window plans; the CODER window builds it.

### Concept -- the signal-strength envelope is the conductor
Re-enable the dormant `_brightness_ema` (disabled v1.5.1) as a DUAL EMA precomputed in
`__init__`: slow `signal` (alpha ~0.05, ambient brightness) + fast `trig` (alpha ~0.3,
lock/glitch triggers); `loss = 1 - signal`. Every dynamic element READS these
(brightness, drift, lock, hierarchy) -- they are NEVER multiplied into the whole frame
(that was the v1.5.1 "dimmed text unreadable" bug). Strong signal = crisp + locked;
weak signal (the silent inter-beat gaps) = the receiver loses lock: the grid dims
first, the scope spokes shorten, faint edge-static creeps, the ident flickers LAST.
The picture loses the signal in the gaps and reacquires on the next cue -- one conceit
that ties the title card (#1, "tune in") to the gutter scopes (#2, "hold the signal").

### #1 -- big-bold EPISODE-TITLE card on the b000 music intro
Decode -> reveal -> POP -> dock. Active window `[music_open_start_f, music_open_end_f +
dock_frames)`; decide the card state BEFORE section 1 draws (if active, skip the normal
ident/subtitle/timestamp and draw the card; after, the docked state IS the normal
section-1 draw).
- A. carrier-lock: "SIGNAL LOST" decodes from a seeded scramble into the solid terminal
  slab; a broken-phosphor-block carrier meter crawls to solid on the swell (`signal`).
- B. HERO title (big + bold): the actual episode title at 2-3x `f_title`, fake-bold by
  OVERSTRIKE with offsets `{(0,0),(1,0),(0,1),(1,1)}` (no real bold font is loaded),
  decoded-fragment reveal stepping on INTEGER frames + a block cursor. Measure with
  `ImageDraw.textbbox`, wrap/scale long titles to a max bbox before effects.
- C. lock POP: a 1-2 frame brightness bloom + a small horizontal coordinate "tear"
  (NOT a hue flash -- green-only blend; NOT `np.roll` -- see specs).
- D. dock (raster collapse): in the tail frames, interpolate the hero bbox down into the
  section-1 ident + subtitle coords. The intro scopes "tune in" (arcs -> full circles)
  synced to the lock.
- Outro bookend (conditional, S5): same logic on `music_close_*` if it resolves.

### #2 -- two asymmetric gutter SCOPES (replace center ring + particles + bottom waveform/bars)
Matched circular form, ASYMMETRIC data + failure (asymmetry is what stops twin rings
reading as wallpaper).
- LEFT `_draw_fft_scope`: 32 radial FFT spokes + per-spoke phosphor comet-tails
  (bounded lookback over `freqs[fi-6:fi+1]`); idle (low signal) -> slow rotating radar
  sweep, phase from `fi/fps` (deterministic).
- RIGHT `_draw_scope`: the `wave` samples traced around the circumference + a bright
  electron SWEEP DOT with a short decaying trail (lookback over `waves`); idle ->
  jittering baseline circle. Absorbs the old bottom waveform.
- Graticules: `_precompute_graticules()` (mirrors the `_scanlines` precompute) ->
  static tick/crosshair RGBA, alpha_composited (near-zero per-frame cost).
- Retire sections 5 (`_waveform_mirror`) + 6 (`_freq_bars_wide`) CALLS; thin section-3
  particles to a faint orbit with brightness ROLES (not hue). Cap all scope line widths
  to 1-2px (the current code is 4px at 1920).

### Sprint plan (one file; each sprint independently testable)
- **S1 -- foundation.** Refactor `_CRTRenderer` to precompute `signal/loss/trig` in
  `__init__` from the full arrays; resolve the timing dict; `import hashlib` + a local
  seeded RNG; EMA read-only discipline. Regression: determinism + audio untouched.
- **S2 -- gutter scopes.** Left/right scope helpers (bounded-lookback trails) +
  graticules + masked gutter-rect layers; retire sections 5/6, thin section 3.
- **S3 -- title card.** The b000-window state machine (decode->bold->POP->dock) +
  intro-scope tune-in + section-1 suppression; the 2-beat gap-fill smoke.
- **S4 -- envelope behaviors.** Vignette choke (bounded/floored/text-exempt), the
  coordinate-offset sync-drift, per-element brightness hierarchy.
- **S5 -- outro + regression.** Conditional `music_close` bookend (inside `total_frames`)
  + full regression (determinism checksum on RGB frames, audio-byte-identical, no new
  widget, CPU/VRAM unaffected).

### Concrete specs (the wiring -- baked from the QA rounds, do not re-derive)
- **Signature:** `_CRTRenderer(w, h, title, volume, freqs, waves, fps, timing=None)`;
  store `self.total = len(volume)`, `self.fps`; reduce to `render(self, fi)`; update the
  `render_video` caller (current L1556 `renderer = _CRTRenderer(W,H,episode_title)` ->
  pass the arrays + timing; L1559 closure -> `renderer.render(fi)`).
- **Timing extractor:** the SceneSequencer stamps `led["lines"]` with `speaker_role` +
  `start_s` + `dur_s` (persisted to the DISK ledger; resolve it the way
  `otr_caption_burn` does -- the wire ledger may carry `start_s=None`). `music_open` =
  the first line whose `speaker_role` is a music-open role; window =
  `round(start_s*fps) .. round((start_s+dur_s)*fps)`; `first_dialogue_f` = first
  dialogue line's `start_s*fps`; `music_close` = last music line. FALLBACK if `start_s`
  is unavailable: derive the intro window from the `volume` envelope (music from frame 0
  to the first dialogue onset), capped. Missing fields DISABLE that effect (no crash).
- **Intervals:** half-open `start <= fi < end`; clamp to `[0, total)`; disable if None
  or `end <= start`.
- **Determinism:** `import hashlib`; per effect `seed = int.from_bytes(
  hashlib.blake2s(f"{title}|{fi}|{salt}".encode()).digest()[:8], "big")`;
  `rng = np.random.default_rng(seed)`; replace the section-8 `np.random.randint` with
  `rng.integers`. `signal[0]=trig[0]=volume[0]`.
- **Geometry (from the real portrait scale):** the 480x832 portrait -> ~626px wide at
  1920, centered -> protected center band x in ~[647, 1273]; gutters [0,647] /
  [1273,1920]. Ring centered in each gutter: `left_cx~=323`, `right_cx~=1596`,
  `cy=h//2`, `r~=235`. **Clamp the circular-scope amplitude `amp <= r*0.35`** so
  `r+amp <= gutter_half_width (~323)` -- never crosses the center band, never overflows
  the frame edge.
- **No `np.roll` on the frame:** it wraps the center into the portrait. Drift + tear =
  a horizontal coordinate OFFSET applied to the gutter-scope + title DRAW coords only,
  clamped so each bbox stays inside its gutter (no black edge, no center incursion);
  center grid/background untouched.
- **Center-band clip:** draw each scope onto a transparent layer sized to its gutter
  rect and `alpha_composite` it (the layer bounds clip it); never draw scope primitives
  onto the base image.
- **Text exemption:** section-1 ident + the title card draw AFTER the section-8
  vignette/choke multiply (or on a post-vignette pass), so the choke can never dim text.
- **Trails are pure:** computed in `render(fi)` by bounded lookback over the input
  arrays (N~6), NOT mutable per-frame state.
- **Outro:** render only for `fi < total_frames`; leave `_hud_frames` append unchanged.

### v1 CUTS (panel consensus -- add later at the eyeball, not now)
Telemetry micro-text labels (illegible green mush downscaled + clashes with the real
`_TelemetryHUDRenderer`); the FFT peak-hold ghost ring + noise-floor shadow ring
(comet-tails already give persistence); the oscilloscope free-running trigger seam;
halation (a 2x per-frame draw pass threatens the 24fps gen speed); a formal hierarchy
layer-floor system (use per-element brightness scaling -- grid scales down faster than
the ident).

### OPEN (operator's call) + VERIFY-AT-BUILD
- **Landscape-beat gutters.** Per-beat gating is INFEASIBLE at procgen render time: the
  floor is rendered before the clips exist and before the clip manifest, so
  `_CRTRenderer` cannot know which beats will be landscape. v1 COMMITS to dim,
  gutter-clamped scopes that read as faint edge telemetry on landscape b-roll. The
  eyeball gate decides whether landscape needs a different treatment (a later option
  would push the per-beat clip-type timeline into the renderer -- cross-file, not v1).
- **Verify smokes (build-time):** (1) the exact disk-ledger `start_s`/`dur_s` field on a
  real episode; (2) a 2-beat gap-fill smoke proving the title card stays at the open
  (the timeline-aligned floor slice + frame-aligned blend already imply this); (3) a
  determinism checksum over RGB frames (not mp4 bytes); (4) a long-title `textbbox`
  overflow case; (5) the coordinate-offset bound (no black edge / no center incursion).
