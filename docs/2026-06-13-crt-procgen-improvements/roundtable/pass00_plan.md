## 4C. CREATIVE BACKLOG -- Procgen Visual Layer (pre-roundtable)

> **STATUS: design-direction-only, `pre-roundtable`.** Two procgen-layer creative
> ideas from the operator (2026-06-13). NO code work starts yet -- Jeffrey wants to
> round-robin these designs across the LLM panel (batch the two) BEFORE
> implementation. Specified here so the panelists have grounded context. **Neither
> touches the audio spine** (frozen, byte-identical) -- both are VISUAL procgen-layer
> changes only. **GROUNDED 2026-06-13:** Claude analyzed the real draw code
> (`_CRTRenderer.render()` in `nodes/video_engine.py`); the architectural-surface
> notes + the design block below are now code-anchored, superseding the earlier
> `OTR_PostUpscaleProcgenBlend` guess.

**Idea #1 -- Procgen episode title card on the first music cue.**
- TRIGGER: the moment the episode's intro/opening music first starts (the first
  music event), not the first dialogue.
- RENDER: the show title **"SIGNAL LOST"** + the episode title (e.g. *"Mapping
  Desperation"*) in big bold PROCGEN letters during the intro-music window -- a
  movie-credits / title-sequence vibe, the title landing with the music swell.
- CONSTRAINT: procgen-rendered (vector-style, integrated with the existing CRT/HUD
  aesthetic), NOT a baked raster.
- ARCHITECTURAL SURFACE (GROUNDED 2026-06-13 vs the real code): the draw surface is
  `nodes/video_engine.py` class `_CRTRenderer.render()` -- the procgen frame drawer
  inside `OTR_SignalLostVideo` (= `SignalLostVideoRenderer`). NOT
  `OTR_PostUpscaleProcgenBlend`, which is only the downstream ffmpeg green-only
  `screen` blend that lays procgen over the upscaled portrait. The persistent
  "=== SIGNAL LOST ===" ident + `"{title}"` subtitle ALREADY draw every frame
  (render() section 1, top-left), so Idea #1 is a windowed BIG treatment that then
  DOCKS into that existing ident -- not a separate overlay that fades. The
  first-music-cue window is derivable from `led` (render_video() already parses the
  v2 ledger; b000 = music_open) and passed into `_CRTRenderer`; the per-frame
  `volume[fi]` envelope (already computed by `_analyze_audio`) gives the swell timing.
  No new model dependency.

**Idea #2 -- Move audio-reactive visuals to the side gutters; keep the portrait clean.**
- TODAY: the green HUD ring + waveform overlay the CENTRAL portrait, partially
  obscuring the character.
- CHANGE: relocate the audio-reactive elements to the LEFT/RIGHT side gutters (the
  negative space outside the central portrait area), so the portrait composition
  lands clean and cinematic while the reactive layer still pulses with audio --
  FRAMING the action instead of overlapping it.
- DIRECTION (operator-confirmed 2026-06-13): KEEP the ring motif (it is core to the
  CRT radar-scope aesthetic) but move it OFF the portrait into the side gutters -- two
  rings, ONE PER SIDE. Final form is Claude's aesthetic call; see LOCKED DESIGN below.
- ARCHITECTURAL SURFACE (GROUNDED 2026-06-13 vs the real code): LAYOUT-ONLY change in
  `_CRTRenderer.render()` (`nodes/video_engine.py`), NOT the blend node. What sits on
  the portrait today: the circular frequency RING (section 2, `cx=w/2, cy=0.42h,
  r=min(w,h)/5`, pulses with `vol`) dead-centre over the face; its 12 orbiting
  PARTICLES (section 3); the mirrored WAVEFORM (section 5, `y=0.72h`, full width) and
  the FREQ BARS (section 6, `y=0.86h`). The helpers `_waveform_mirror(...x,y,w,h...)`
  and `_freq_bars_wide(...x,y,w,h...)` are already position-parametrized, so relocating
  is geometry + a vertical transpose; the RMS/FFT/wave data source is UNCHANGED.
  GUTTER REALITY: HuMo character beats are 480x832 portrait pillarboxed into 1472x832
  -> ~496px black gutter EACH SIDE (the blend already fills the pillarbox bars), so
  gutters are real there; LTX/Wan b-roll beats are landscape full-frame (NO gutters).
  Guaranteed empty gutter real-estate exists only on portrait beats -- see the OPEN
  DECISION below.

**Claude design analysis (2026-06-13 -- GROUNDED against the real `_CRTRenderer` draw code; my best-judgment calls). The creative bullets below are now CONFIRMED by the code; the GROUNDED DELTAS block after them states what the code changes + the decisions I'd lock.**

- **Order: both worth doing; #2 FIRST, #1 second.** #2 fixes a composition error in ~100% of
  runtime (chrome sitting on the face, the emotional subject) -- every-frame upside, and the
  reactives get MORE legible once they stop fighting the portrait. #1 is a 4-8s delight moment that
  lands far better on an already-clean stage. They are synergistic (the cleared gutters become part
  of the title choreography).
- **The bigger hit (push this hardest): make the chrome SIGNAL-DRIVEN, not signal-themed.** Derive a
  "signal-strength" envelope from the audio and let it drive everything: strong signal
  (dialogue/music) = portrait stable, rails bright + locked, idents steady; weak signal (the silent
  gaps, scene seams) = rails decay inward, a faint scanline roll drifts the portrait, the call-sign
  flickers -- the picture LITERALLY loses the signal in the gaps and reacquires on the next cue. This
  turns the reactive layer from decoration into a narrative device, gives transitions a built-in
  grammar (signal drop = the cut), and makes #1 ("tunes in") and #2 ("hold the signal") two faces of
  ONE conceit. Cheap: same envelope feeds all of it. If only one thing goes to the panel as the
  headline, it is this.
- **#1 details.** Trap to avoid: a clean centered modern fade-in (reads as generic streaming, fights
  the CRT soul) -- the title must feel DECODED/tuned-in, not "presented." SEQUENCE, don't stack:
  `SIGNAL LOST` carrier-locks first (de-noise from green snow into a solid slab of the existing
  terminal face, one-frame chromatic tear on lock = the station ID), THEN the episode title
  teletype-reveals char-by-char with a cursor (the incoming transmission = the program). Timing OFF
  the audio (anchor entrance to the music-cue start, exit to first-dialogue-minus-a-beat), not a fixed
  clock. PERSIST BY DOCKING, not fading: the card is the BIRTH of the two persistent corner idents --
  `SIGNAL LOST` shrinks to a corner call-sign/channel-bug, the episode title settles into the corner
  terminal slot it already occupies. Palette stays green + one brief amber/white "signal acquired"
  flash on lock. References: *The Outer Limits* cold open ("we control the horizontal/vertical") is
  the spiritual touchstone; also *Twilight Zone* restraint, EBS/CONELRAD "please stand by" test cards,
  Pip-Boy/WarGames phosphor for the type; borrow *Stranger Things*' letters-lock MECHANIC but NOT its
  red-serif look.
- **#2 form factor (REVISED 2026-06-13 -- rings, NOT rails).** The operator wants to keep the ring, and
  I agree: it is truer to the radar-scope CRT aesthetic than abstract rails (my earlier pick), so I am
  superseding the rails. My call: TWO rings, one per gutter, same circular FORM but ASYMMETRIC DATA --
  LEFT = the existing FFT spoke-ring (the spectrum scope), RIGHT = a circular OSCILLOSCOPE (the waveform
  traced around the circumference), which consolidates the old bottom mirrored-waveform into it. Asymmetric
  data is what stops twin gutter rings reading as mirrored wallpaper. Reject: identical mirrored rings
  (wallpaper), a 4-ring stacked "scope rack" (too busy for a dread show -- held only as a denser variant),
  vertical EQ bars (Winamp cliche), VU needle gauges (too cozy-studio). Detail in LOCKED DESIGN.
- **Risks / cliche watch.** Green-on-green luma muddle (enforce hierarchy: portrait brightest,
  subtitles high-contrast, chrome dim -- watch subtitles hardest); glitch fatigue (reserve heavy
  glitch for MOMENTS -- title lock, signal-loss gaps -- keep steady-state calm); symmetric mirrored
  gutters read as wallpaper.
- **Bonus swing: an OUTRO bookend.** On the closing music, `SIGNAL LOST` reasserts and the picture
  drops to static/black (carrier drop, "we now return you to..."). Same toolkit as #1, bookends the
  episode, and literally dramatizes the show's name.
- **One-sentence evolution each, before code.** #1: reframe from "a title card" to "the birth of the
  persistent idents" -- sequence carrier-lock -> teletype, drive timing off the music-cue + first-
  dialogue stamps, then DOCK both into their permanent corner positions instead of fading. #2: reframe
  from "move the reactives to the gutters" to "two asymmetric vertical signal-rails (waveform L /
  spectrum R) whose brightness tracks an audio signal-strength envelope," consolidating the bottom
  waveform into them so the clean portrait is framed, not crowded.

**LOCKED DESIGN (2026-06-13 -- operator-confirmed direction + Claude's aesthetic calls, grounded vs the real `_CRTRenderer` code):**

- **The "signal-strength envelope" already exists -- WIRE it, don't build it.** `_analyze_audio()`
  returns a normalized per-frame `volume[fi]` (RMS) + 32-bin `freq[fi]`, and `_CRTRenderer` already
  carries a dormant EMA (`self._brightness_ema`, alpha 0.08) that v1.5.1 left disabled (render()
  section 8b). Re-enable that EMA as the MASTER signal-strength driver (it already smooths transients)
  and feed it to rail brightness + the carrier-lock/dropout behaviour. The headline conceit is near-free.
- **#2 LOCKED (rings, not rails) -- do it FIRST.** Move the centre ring (section 2) OFF the portrait into
  the side gutters as a MATCHED PAIR of CRT scopes: LEFT = the existing 32-spoke FFT ring (the spectrum
  scope), RIGHT = a circular OSCILLOSCOPE (the `wave` samples traced around the circumference, radius =
  base +- amplitude) which CONSOLIDATES the old bottom mirrored-waveform into it (no third reactive zone).
  Same circular FORM (keeps the current look), ASYMMETRIC DATA (left spiky/spectrum, right smooth/waveform)
  so the pair reads as a real two-instrument console, not mirrored wallpaper. Size each ring to its gutter
  (radius ~= min(gutter_w, h) * 0.3, vertically centred ~0.5h); the 12 particles (section 3) thin to a
  faint orbit around each ring (keeps life, no centre clutter). The centre column (the portrait region)
  stays free of bright chrome -- only the dim grid + scanlines + vignette. Ring brightness + lock ride the
  signal-strength EMA above.
- **#1 LOCKED -- the big bold EPISODE TITLE is the hero.** The small persistent ident is already render()
  section 1; #1 adds a WINDOWED title card over the b000 music-open window: the "SIGNAL LOST" show-ident
  carrier-locks first (de-noise from the vol-gated snow already in render() section 8, ~line 321), THEN
  the ACTUAL EPISODE TITLE reveals BIG + BOLD, centre stage, landing on the music swell as a brightness
  "signal-acquired" POP (NOT a hue flash -- the green-only blend kills colour). Then it DOCKS: the card
  shrinks into the EXISTING top-left ident + episode-title slot as the drama opens. Timed in the procgen
  24fps clock off the b000 start/end + first-dialogue frame from `led` (already parsed in render_video).
  Outro bookend = the same logic on music-close -> hand to the existing `_TelemetryHUDRenderer` post-roll.
- **OPEN DECISION (the one genuinely-open item): landscape-beat gutters.** Gutters are guaranteed empty
  only on pillarboxed portrait beats; LTX/Wan landscape beats fill 1472 wide. (A) accept the thin/dim
  rails riding the landscape edges on b-roll beats (cheapest), or (B) the compositor
  letterboxes/portrait-shrinks landscape beats so gutters always exist (cleaner, but touches composite
  geometry + the canvas/aspect question). My lean = (A) for v1, revisit (B) only if it reads badly at the
  eyeball. THIS is the item worth the panel's time; the rest is build.
- **Scope reality:** all of this is ONE file -- `nodes/video_engine.py` (`_CRTRenderer` + the
  `render_video` ledger-window plumbing). No audio-spine touch, no new model, no new node, no new widget.
  The blend stays green-only `screen`, so every rail/title reads as green phosphor automatically.

**Next step (operator-triggered):** the design is LOCKED (operator direction + my aesthetic calls). The
only open question = landscape-beat gutters (above). On your GO I package a `_CRTRenderer` implementation
ticket for the coder window -- ONE file, no audio-spine / model / node / widget change, the green-only
blend untouched. Still no code until you say go.

---
