<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

NEW IDEAS:

1. Title-card “carrier acquisition meter” under the episode title: a horizontal lock bar made of broken phosphor blocks that crawls from noisy fragments to a solid line exactly as the music swell arrives; fits because it reads like broadcast sync/Outer Limits calibration, not a modern progress bar; DRAW approach: new title-window helper in `_CRTRenderer.render()` after section 4 grid and before section 7 bottom bar, using `vol` / re-enabled `_brightness_ema` plus b000 window timing from `render_video` ledger plumbing; cost cheap.

2. Fake-bold monospace title by overstriking, not by new fonts: draw `SIGNAL LOST` and the episode title 3-5 times with 1px offsets, slight vertical scanline gaps, and occasional missing character strokes during lock; fits because it feels like phosphor burn-in / terminal overprint rather than clean streaming typography; DRAW approach: section 1/title-card helper using existing `_load_font()` at larger sizes, draw text multiple offsets in green-channel-bright fills only; cost cheap.

3. Docking “raster collapse” from hero card into the existing top-left ident: the big title compresses into two thin horizontal phosphor traces, then snaps into the current section 1 call-sign and subtitle positions; fits because it makes the persistent HUD feel born from the broadcast signal instead of faded UI; DRAW approach: title-window helper computes interpolation from center title bbox to section 1 text coordinates, draws shrinking guide lines/ghost text, then hands back to normal section 1; cost medium.

4. Gutter rings get oscilloscope graticules: faint crosshair, 10-degree tick marks, and tiny terminal labels like `SYNC`, `GAIN`, `KC/S`, `TRACE` around each ring; fits because it turns the rings into believable CRT instruments instead of decorative circles; DRAW approach: section 2 replacement helper for left FFT ring and right circular waveform ring, using gutter center/radius geometry and `CRT_DARK`/dim green brightness scaled by EMA; cost cheap.

5. Left FFT ring uses “phosphor persistence ghosts”: each spoke draws a bright current tip plus 1-2 shorter dim afterimages rotated a few degrees backward; fits because old CRT traces smear and decay, and it avoids Winamp-bar flatness; DRAW approach: section 2 FFT gutter ring, derive ghost length from current `freq[i]` and time `t`; no new audio analysis; cost cheap.

6. Right circular oscilloscope gets a trigger seam: one brighter moving dot / notch shows where the waveform trace begins, and in weak-signal gaps the seam drifts/free-runs instead of staying locked; fits because it is a real oscilloscope behavior and dramatizes “losing lock”; DRAW approach: new circular waveform helper replacing section 5 mirrored waveform, using `wave`, `t`, and signal EMA for phase stability; cost medium.

7. “Weak signal inward crawl” at gutter edges only: during low EMA, faint vertical noise combs creep inward from the frame sides but never cross into the portrait-safe center column; fits because it feels like RF interference eating the carrier while preserving subject readability; DRAW approach: section 8 CRT post/noise or a new pre-post helper after section 6, driven by `1.0 - _brightness_ema`; use deterministic per-frame hash noise, not unseeded random; cost medium.

8. Title reveal as decoded fragments, not a simple typewriter: episode-title characters appear first as wrong glyphs / `#`, `/`, `0`, then resolve into the correct title over 6-12 frames per cluster; fits WarGames/terminal decoding while staying monospace and green-only; DRAW approach: title-card helper, deterministic glyph substitution from `fi`, character index, and title string; uses existing fonts; cost cheap.

9. Signal-strength “chrome hierarchy clamp”: as EMA rises, rings brighten and sharpen; as EMA falls, rings dim, particles thin, grid drops first, ident flickers last; fits because the whole screen behaves like one receiver instead of unrelated overlays; DRAW approach: central multiplier function used in section 1 title bar, section 2 rings, section 3 particles, section 4 grid, section 8 noise; cost cheap.

10. Gutters use asymmetric idle states: left FFT ring collapses to a small rotating radar sweep in silence, right oscilloscope flattens to a jittering horizontal/circular baseline; fits because each instrument fails differently, making the pair feel functional rather than mirrored wallpaper; DRAW approach: section 2 left ring and right waveform helper, branch on low `_brightness_ema`; cost cheap.

11. “Station calibration slugs” during the intro only: tiny monospace readouts in gutters pulse with values like `CARRIER 073%`, `LOCK`, `DRIFT`, `NOISE FLOOR`; fits EBS/CONELRAD/test-equipment language without becoming dialogue captions; DRAW approach: title-window section plus gutter-ring helper, values derived from `vol`, EMA, `freq` averages; [ASSUMPTION] safe if non-story telemetry text is acceptable; cost cheap.

12. Replace colorful particle cycling with brightness-only phosphor hierarchy: particles currently choose green/cyan/amber, but the final blend collapses that to different green intensities; make that intentional by assigning “primary/secondary/ghost” brightness roles instead of hue roles; fits the green-only CRT rule and avoids accidental color-thinking; DRAW approach: section 3 particles, use only green-channel brightness values; cost cheap.

13. Deterministic snow as signal static, not loudness sparkle: current section 8 noise appears when `vol > 0.3`; invert/reshape it so low signal gets crawling static, strong signal clears, with a brief bright pop on acquisition; fits “signal lost/reacquired” better than loud passages getting noisy; DRAW approach: section 8 post-processing, replace `np.random.randint` with deterministic frame-seeded noise from `fi`/title and intensity from `1 - EMA` plus title-lock spike; cost medium.

14. Landscape-beat fallback: when no gutters exist, rings become ultra-dim edge instruments clipped to the outer 8-10% of frame, with no bright spokes crossing the subject; fits as broadcast edge telemetry while respecting full-frame b-roll; DRAW approach: geometry branch in section 2 based on available gutter width/window info from `render_video`; verify: whether renderer can know portrait-vs-landscape beat timing from `led`; cost medium.

15. Intro rings “tune in” before the title lands: during b000, left/right rings start as incomplete arcs and lock into full circles on the title pop; fits the receiver-calibration motif and ties #1 to #2; DRAW approach: title-window helper coordinates with section 2 ring draw, using music-window progress and EMA; cost cheap.

ELEVATIONS to the 3 existing pieces:

1. Big-bold EPISODE-TITLE card:
- Smallest high-impact change: make the episode title fake-bold, overstruck, and center-dominant while `SIGNAL LOST` behaves like the carrier label above it.
- Draw `SIGNAL LOST` first as a smaller locked slab, then the actual episode title at 2-3x `f_title` size using multi-pass offset text.
- Add a one-frame or two-frame brightness bloom on lock, not a color flash.
- Dock by interpolating the big title bbox into the existing section 1 subtitle coordinates instead of fading out.
- Suppress or dim normal section 1 ident while the hero card is active, then let the docked state become the normal section 1 draw.

2. Two asymmetric gutter rings:
- Smallest high-impact change: add graticules and distinct failure behavior.
- LEFT FFT ring: hard radial spokes, tick marks, tiny spectrum labels, phosphor ghost tips.
- RIGHT oscilloscope ring: continuous wavering trace around circumference, trigger seam/dot, flattening baseline in silence.
- Keep both rings same radius/vertical center so they feel like a matched console, but never mirror their data or motion.
- Remove the old full-width section 5 waveform during portrait beats once the right circular oscilloscope exists.

3. Signal-strength-driven chrome unifier:
- Smallest high-impact change: create one `signal = EMA(vol)` value and one `loss = 1 - signal` value, then use them everywhere.
- Strong signal: ring outlines crisp, title stable, grid faint but steady, particles sparse and locked.
- Weak signal: grid drops first, ring spokes shorten, oscilloscope free-runs, static creeps at edges, ident flickers only lightly.
- Do not use the envelope to brighten everything equally; use it to enforce hierarchy and behavior.
- Change section 8 noise from “loud = more noise” to “weak carrier = more static,” with brief acquisition pop moments.

AESTHETIC RISKS / TRAPS:

- Treating amber/cyan/white as actual colors. The production blend keeps only green, so every accent must be designed as brightness.
- Clean centered fade-ins. They read like streaming titles, not a decoded broadcast.
- Mirrored gutter rings. Symmetry will look like wallpaper unless left/right data and failure modes differ.
- Vertical EQ bars in gutters. That reads Winamp/modern visualizer, not old radio-room CRT.
- Too much glitch all the time. Heavy breakup should be reserved for title lock, cue seams, and weak-signal gaps.
- Chrome brighter than faces or subtitles. The portrait and captions must win; gutters are framing instruments.
- Drawing reactive elements across landscape b-roll as if gutters still exist. No gutters are guaranteed there.
- Cozy studio metaphors: VU needles, warm mixer lights, polished meters. Keep it cold, technical, haunted.
- Random nondeterministic snow. Any noise should be deterministic per frame/seed/title path, not uncontrolled global randomness.
- Over-labeling with cute UI text. Telemetry should feel like receiver diagnostics, not a game HUD.
- Too many rings or stacked scopes. The locked two-ring idea is strong because it is restrained.
- Attempting portrait displacement/glitch if the available surface cannot actually move the composited portrait. Screen overlay can add brightness; it cannot darken or warp underlying footage by itself.

MUST-KEEP PRINCIPLES:

- Green phosphor only: all design contrast is brightness, density, line weight, flicker, and persistence.
- Center subject remains readable; gutter chrome frames the drama and does not sit on faces.
- Two-ring grammar stays asymmetric: left = FFT spectrum scope, right = circular oscilloscope.
- The title card is a signal event: acquire, decode, pop, dock.
- The signal-strength EMA is the master behavior driver, not a decorative pulse.
- Use existing `vol`, `freq`, `wave`, `fi`, `fps`, title, and ledger timing only.
- One-file Pillow implementation in `_CRTRenderer.render()` plus `render_video` window plumbing.
- 24fps deterministic procgen clock.
- Steady-state calm, momentary breakdowns.
- Old broadcast/test-equipment/terminal language over modern UI polish.