# SIGNAL LOST procgen CRT upgrade -- sprint-ready coding plan (pass02, round-2 hardened)

One file: `nodes/video_engine.py` (`_CRTRenderer` + `render_video` plumbing). No new
node/widget/model/dep; Pillow-only; green-only blend untouched; audio spine frozen;
24fps; deterministic. Round-2 panel (gemini-3.1-pro, gpt-5.5, deepseek-v4-pro) fixes
folded; each grounded vs the real code.

## ARCHITECTURE FIX (the headline round-2 catch) -- keep render(fi) PURE
Today `render(fi, total, fps, vol, freq, wave)` is pure. The new effects need
per-frame STATE (EMA, comet-tails, sweep trail). Updating state inside `render()`
would make output depend on sequential frame execution -> breaks determinism /
out-of-order / resume. FIX (Gemini, confirmed): pass the FULL `volume/freqs/waves`
arrays into `_CRTRenderer.__init__` and PRECOMPUTE everything there into arrays
indexed by `fi`:
- `self.signal[fi]` (slow EMA, alpha~0.05), `self.trig[fi]` (fast EMA, alpha~0.3),
  `self.loss[fi] = 1 - signal[fi]`.
- `self.fft_tips[fi]` history (per-bin recent positions for comet-tails),
  `self.sweep_idx[fi]` (oscilloscope dot position + short trail).
`render(fi)` then only READS `[fi]` -- it stays a pure function of `fi`.

## TIMING PLUMBING (explicit interface -- round-2 must-fix)
Extend `_CRTRenderer.__init__(w, h, title, timing=None)` where `timing` is frame
numbers (24fps), parsed in `render_video` from `led` (already available there):
`{"music_open_start_f", "music_open_end_f", "first_dialogue_f",
"music_close_start_f", "music_close_end_f"}`. Clamp all to `[0, total_frames)`.
**A missing/None field DISABLES that effect (title card / outro) -- never crashes.**
VERIFY-AT-BUILD: exact `led` field names for the b000 (music_open) interval +
first-dialogue frame (`_parse_hud_data` already consumes `led`, so it is reachable).

## DETERMINISM (round-2 must-fix)
The current section-8 noise uses UNSEEDED `np.random.randint` -> non-deterministic.
FIX: a LOCAL generator with a STABLE seed (NOT Python `hash()`, which is
process-randomized): `seed = int.from_bytes(hashlib.blake2s(f"{title}|{fi}|{salt}"
.encode()).digest()[:8], "big"); rng = np.random.default_rng(seed)`; replace
`np.random.randint` with `rng.integers`. Per-effect `salt` so effects do not
correlate. EMAs reset at `fi==0` (do NOT start half-locked at 0.5).

## NO np.roll ON THE FRAME (round-2 must-fix, 3/3)
`np.roll` wraps pixels -> shifts the center grid/portrait band, violating
center-column sanctity. FIX (Gemini): implement sync-drift + the lock "chromatic
tear" as a horizontal COORDINATE OFFSET applied to the gutter-scope + title-card
DRAW coordinates only; leave the center grid + background untouched. Bound the offset
so no black edge appears.

## EMA IS READ-ONLY PER ELEMENT (round-2 must-fix) -- do NOT repeat the v1.5.1 bug
The disabled v1.5.1 code did `arr *= vignette*ema` over the WHOLE frame and dimmed
the CRT text unreadable. The new `signal/loss/trig` are READ by element-specific
effects ONLY (ring brightness, drift, halation, hierarchy clamp, vignette choke) --
NEVER multiplied into the whole frame array. The vignette choke keeps the immutable
base `self._vignette` and applies a BOUNDED loss-multiplier with a readable FLOOR,
and EXEMPTS the title/ident text from any dimming (ident flickers LAST).

## #2 -- two gutter scopes (S2)
Geometry (concrete): `left_cx = int(w*0.15)`, `right_cx = int(w*0.85)`,
`cy = int(h*0.5)`, `r = int(min(w*0.13, h*0.30))`; spoke/trace width capped 1-2px
(current code is 4px at 1920 -> cap it). `_precompute_graticules()` (mirrors
`_scanlines`): static tick/crosshair RGBA, alpha_composited.
- LEFT `_draw_fft_scope`: 32 spokes + per-spoke phosphor comet-tails (from
  `self.fft_tips[fi]`); idle (low signal) -> slow rotating radar sweep, phase from
  `fi/fps` only (deterministic). [CUT for v1: peak-hold ghost ring + noise-floor
  shadow ring -- comet-tails already give persistence; add later if wanted.]
- RIGHT `_draw_scope`: `wave` traced around the circumference + a bright sweep dot
  with a short decaying trail (`self.sweep_idx[fi]`); idle -> jittering baseline
  circle. [CUT for v1: free-running trigger seam -- defer until the basic scope is
  stable.]
- Retire section 5 (waveform) + section 6 (freq bars); thin section-3 particles to a
  faint orbit with brightness ROLES (not hue). 
- **Center-column sanctity = a checkable rule**: protected band computed from the real
  portrait scale (480x832 -> ~647px gutters at 1920, i.e. center ~x in [636,1284]);
  clip ALL bright scope geometry against it (not "middle half" / "outer 8-10%").
- **Landscape commitment (v1):** `_CRTRenderer` is beat-agnostic, so the scopes draw
  on every frame. v1 COMMITS to drawing them DIM + clamped to the outer band so on
  landscape b-roll they read as faint edge telemetry (constraint treated as void for
  the floor layer). Per-beat gating (pass the landscape intervals from the clip
  manifest into the renderer) is the eyeball-gated FOLLOW-UP, not v1 (it needs
  cross-file plumbing that breaks "one file").

## #1 -- title card on the b000 window (S3)
Invoked only when `fi in [music_open_start_f, music_open_end_f]`. Decide title-card
state BEFORE section 1 draws; if active, SKIP the normal ident/subtitle/timestamp and
draw the card; after the window, the docked state becomes the normal section-1 draw.
- A. carrier-lock: "SIGNAL LOST" decodes from seeded scramble -> solid slab; broken-
  block carrier meter crawls to solid on the swell (`signal`/window progress).
- B. HERO episode title: 2-3x `f_title`, fake-bold by OVERSTRIKE with a minimal fixed
  offset set `{(0,0),(1,0),(0,1),(1,1)}` (no real bold font exists), decoded-fragment
  reveal stepping on INTEGER frames + block cursor. Measure with `ImageDraw.textbbox`
  and wrap/scale-down long titles before effects (no overflow).
- C. lock POP: 1-2 frame brightness bloom + the coordinate-offset "tear" (NOT np.roll,
  NOT hue).
- D. dock (raster collapse): interpolate the hero bbox into the section-1 ident +
  subtitle coords.
- Intro rings "tune in" (arcs -> full circles) synced to the lock.
GROUNDED (gap-fill reprisal cleared): `OTR_SilentComposite` slices the floor
TIMELINE-ALIGNED (`_floor_aligned`) and the blend is FRAME-ALIGNED, so the title-card
frames (the open) land only at the head/open, NOT in mid-roll gap fills. VERIFY with a
2-beat smoke; no cross-file change needed.

## Cross-cutting (S4) + outro (S5)
- vignette choke (bounded, floored, text-exempt -- see above); gutter-clamped
  coordinate-offset sync-drift; halation via a THICKER-WIDTH `CRT_DIM` pass drawn
  BEFORE the bright element (Pillow has no bloom; no scaling) -- optional polish,
  only at `signal>0.7`; hierarchy clamp = SEPARATE layer floors (grid floor < scopes <
  ident/title) so grid drops first, ident last.
- Outro bookend (S5): only if `music_close_*` resolves; render INSIDE the existing
  `total_frames` (do NOT add/remove video frames -- preserves the mux + `_hud_frames`
  append), then hand to the existing `_TelemetryHUDRenderer`. Skip cleanly if the
  window is unavailable.

## SPRINTS (each independently testable)
- **S1** precompute-in-__init__ refactor (signal/loss/trig + histories) + timing
  interface + stable-hash local RNG + EMA-read-only discipline. Regression: byte-
  identical re-render (determinism), audio untouched.
- **S2** the two gutter scopes + graticules + center-band clip + landscape dim-clamp;
  retire sections 5/6, thin section 3.
- **S3** the title card (decode->bold->POP->dock) + intro-ring tune-in + section-1
  suppression; 2-beat smoke for the gap-fill check.
- **S4** vignette choke + sync-drift (coord-offset) + hierarchy floors [+ halation].
- **S5** outro bookend (conditional) + full regression (determinism, audio-byte-
  identical, no new widget, VRAM unaffected -- pure-PIL/CPU).

## CUT for v1 (panel consensus -- add later if the eyeball wants them)
Telemetry micro-text labels (green mush downscaled + clashes with the real HUD);
FFT peak-hold ghost ring + noise-floor shadow ring; oscilloscope free-running trigger
seam; halation if readability/perf is uncertain.

## VERIFY-AT-BUILD
Exact `led` fields for b000 + first-dialogue; the 2-beat gap-fill smoke; the
coordinate-offset bound (no black edge / no center incursion); textbbox overflow on a
long title; determinism checksum on a fixed audio/title/timing render.
