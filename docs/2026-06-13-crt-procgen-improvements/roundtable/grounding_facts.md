# GROUNDING FACTS -- SIGNAL LOST procgen CRT layer (honor these; do not propose anything that breaks an invariant)

## The show / aesthetic
- SIGNAL LOST is a procedurally generated, eerie/dread old-time-radio drama with a
  GREEN-PHOSPHOR CRT / radar-room look: phosphor green on deep navy-black, scanlines,
  vignette, volume-gated "snow"/noise, monospace terminal type. Think Outer Limits
  cold-open, EBS/CONELRAD test cards, WarGames/Pip-Boy phosphor -- NOT modern streaming.
- Palette (RGB): CRT_BG (8,8,16); CRT_GREEN (0,255,65); CRT_AMBER (255,176,0);
  CRT_CYAN (0,200,200); CRT_DIM (0,100,28); CRT_DARK (0,50,14); CRT_WHITE (180,200,180).

## How the procgen is drawn + composited (the pipeline)
- The drawer is `_CRTRenderer.render(fi, total, fps, vol, freq, wave)` in
  nodes/video_engine.py (see grounding_crt_code.py). It paints the WHOLE frame at
  1920x1080 @ **24fps** -- the "radio floor."
- Audio is ALREADY analyzed per frame and passed in: `vol` (normalized 0-1 RMS
  envelope), `freq` (32-bin FFT, normalized 0-1), `wave` (downsampled waveform
  samples, ~200/ frame). A dormant EMA (`self._brightness_ema`, alpha 0.08) exists in
  `__init__` but was DISABLED in v1.5.1 (render() section 8b).
- The v2 ledger (`led`) is available in `render_video` (NOT inside `_CRTRenderer`
  today). It carries per-beat/line timing; the opening-music beat is `b000`
  (music_open). The episode title is resolved in `render_video` and passed to
  `_CRTRenderer(w, h, title)`.
- Downstream the procgen mp4 plays two roles: (a) the floor/gap-fill in
  `OTR_SilentComposite` (scaled to the 1472x832 @ **25fps** timeline), and (b) a
  GREEN-ONLY `screen` overlay over the RTXUpscaled portrait in
  `OTR_PostUpscaleProcgenBlend`.

## HARD CONSTRAINTS (an idea that violates one is out of scope)
1. **GREEN-ONLY BLEND.** Production blend zeroes the R and B channels and uses
   `screen`. ONLY the GREEN channel of anything drawn survives onto the final
   picture. A colored element collapses to a green of its G value -- i.e. it reads as
   a BRIGHTNESS, not a hue. Any "flash"/accent must be a brightness event, not a color.
2. **GUTTERS ONLY ON PORTRAIT BEATS.** HuMo character beats are 480x832 pillarboxed
   into the frame -> wide black side gutters (~647px each side at 1920). LTX/Wan
   b-roll beats are landscape FULL-FRAME -> NO gutters. The center column is the
   portrait/subject and must stay readable.
3. **NO audio-spine touch.** The master audio is frozen/byte-identical; mux is a
   separate downstream node. Visual-only changes.
4. **ONE FILE, NO NEW SURFACE.** All work lives in `_CRTRenderer` + the
   `render_video` ledger-window plumbing. No new ComfyUI node, no new workflow
   widget, no new model. Pillow-only drawing, 100% local/offline, no new deps.
5. **Deterministic** per-seed within a render; **24fps** procgen clock (not 25).

## The two ideas being hardened (detail in pass00_plan.md)
- #1 Big-bold EPISODE-TITLE card during the b000 music intro: SIGNAL LOST carrier-
  locks, the actual title reveals BIG, then DOCKS into the existing corner ident.
- #2 Move the center frequency RING off the portrait into TWO asymmetric gutter
  rings: LEFT = FFT spectrum spoke-ring, RIGHT = circular oscilloscope (waveform
  around the circumference), absorbing the old bottom waveform. Center stays clean.
- Unifier: drive ring brightness / title lock / "signal loss in the gaps" off the
  (re-enabled) RMS signal-strength EMA.
