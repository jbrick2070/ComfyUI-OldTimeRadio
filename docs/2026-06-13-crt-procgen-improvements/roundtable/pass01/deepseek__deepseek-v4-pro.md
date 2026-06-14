<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

NEW IDEAS (ranked by impact):

1. **Radar-sweep dot on the right oscilloscope ring** — A glowing green dot traces the circumference of the RIGHT (waveform) ring, leaving a rapidly decaying phosphor trail behind it. Sweep speed tied to audio‑phase / beat detection (e.g. a zero‑crossing derivative of the wave array). Fits the radar‑scope CRT aesthetic perfectly: it’s the visual cousin of the FFT spokes on the left ring. DRAW: inside the right‑ring drawing section (new helper `_draw_circular_oscilloscope`), each frame compute a trailing index from `wave` zero‑crossings and draw a bright dot at the corresponding angle, plus a few faint fading dots at previous angles stored in a small ring buffer (`self._sweep_history`). Cost: cheap (O(1) dots per frame, no heavy computation).

2. **Horizontal sync‑drift on signal loss** — When the signal‑strength EMA drops below a threshold (e.g. <0.2), the entire frame (or only the gutter rings) shifts a random 2–4 pixels sideways each frame, with occasional 1‑frame violent jitter. This directly dramatises “losing the signal” as a CRT losing horizontal hold. DRAW: in the final compositing step (before scanlines), apply `np.roll` to the image array along the horizontal axis by a small random int drawn from a distribution shaped by `(1 - ema)`. Cost: cheap (array roll + negligible RNG).

3. **Phosphor‑persistence trails on the FFT ring bars** — Each spectral bar’s tip leaves a faint, decaying “comet‑tail” of its previous positions. This emulates real P1 phosphor decay and adds liquid motion to the left ring. DRAW: maintain a per‑bar ring buffer (length ~6 frames) of previous bar‑end coordinates in `self._fft_bar_history`. Each frame, draw the trailing points as tiny dots with decreasing opacity (mapped to green brightness). Cost: medium (needs state arrays + a loop of dot draws, but still only ~32×6 points, trivial for Pillow).

4. **Graticule markings on both rings** — Add faint monospace tick‑marks, crosshairs, and “MHz” / “dB” labels at the perimeter of each scope, rendered as thin green lines and tiny terminal text. Gives the gutters the functional‑instrument feel of a real radar room, stops them reading as wall‑paper. DRAW: precompute a static graticule image for each ring (radius, tick angles) once in `__init__` and `alpha_composite` it onto the main canvas in the ring sections. Cost: cheap (pre‑computed overlay).

5. **Noise‑floor shadow ring on the FFT scope** — A dim, thin outer circle whose radius shrinks/grows as the inverse of signal strength, showing the background “hiss floor.” DRAW: in the left ring code, draw an extra faint circle (outline only) with radius = base_ring_r * (1.0 + (1.0 - ema) * 0.2). Fits the radar‑scope motif and gives the EMA another visual anchor. Cost: cheap (one `ellipse` call per frame).

---

ELEVATIONS to the 3 existing pieces:

- **#1 (big bold EPISODE TITLE card):** Right at the lock moment (when the carrier‑lock de‑noise finishes), trigger a 2‑frame “chromatic tear” – the entire frame shears horizontally by a random amount (e.g. ±8 px), then snaps stable. This is a single‑frame distortion that sells the “signal acquired” pop without any hue dependency. Draw: in the title‑card rendering code (the windowed block), on the lock frame, apply `np.roll` with a random shift to the whole image. Cost: near‑zero, one extra roll per lock.

- **#2 (two asymmetric gutter rings):** Add a “peak‑hold” ghost circle on the left FFT ring: show the maximum bar length reached over the past ~1.5 seconds as a thin, brighter outline just beyond the current bars. This is a classic spectrum‑analyzer feature and rewards the eye with a subtle physics‑like dynamic. Draw: maintain `self._fft_peak` as an exponentially decaying max of each bin; when drawing a bar, also draw a dot at the peak position with `fill=(0, 255, 65)` but slightly lower alpha (achieved by drawing smaller dots). Cost: cheap (one array update per frame, negligible draw).

- **#3 (signal‑strength envelope / unifier):** Drive a subtle “bloom” on the brightest green elements (the ring outlines, the title text, the radar dot) by drawing a slightly larger, 30%-dimmed duplicate just behind them, only when EMA > 0.7. This creates the CRT “halation” effect without any Gaussian blur (which is heavy). Draw: at the points where bright green is drawn, overlay a second, larger shape with `CRT_DIM`. Cost: cheap (a handful of extra lines/ellipses per frame, only on loud beats).

---

AESTHETIC RISKS / TRAPS:

- **Too many independent reactive layers** — if the original freq bars (section 6) or mirrored waveform (section 5) are merely moved instead of fully absorbed into the rings, the gutters become a chaotic light‑show that overwhelms the portrait. The rings must be the ONLY reactive zones; everything else must be removed or dimmed to near‑invisible.
- **Over‑glitching** — constant jitter, noise, tearing will fatigue the viewer and cheapen the moments of real narrative signal loss. Reserve heavy glitch for the title lock and extreme‑silence transitions; keep the steady‑state calm.
- **Modern‑streaming title transitions** — any soft fade, cross‑dissolve, or clean lettering without CRT artifacts (noise, scanlines, wobble) will read as generic. Every title frame must carry a tiny amount of condition‑driven noise or horizontal instability.
- **Mirrored symmetry** — if both gutter rings use identical data (e.g. both spectrum), the pair reads as wallpaper, not instruments. The design must maintain the asymmetry: left = FFT spokes, right = waveform/circular scope.

---

MUST‑KEEP PRINCIPLES:

- **Green‑only hierarchy** — the portrait must remain the brightest element; all chrome must sit at ≤70% of its maximum green channel, and subtitles must be high‑contrast.
- **Procedural, zero‑asset** — everything is drawn from math; no pre‑built images, no external fonts, no model calls.
- **The rings must read as CRT scopes** — circular, with oscilloscope/radar‑scope grammar (spokes, sweep, graticule, phosphor persistence), never as modern EQ bars or glowing donuts.
- **The signal‑strength EMA is the emotional conductor** — every dynamic element’s brightness, stability, or presence should be mapped to it. The picture should literally “lose the signal” in the gaps and reacquire it.
- **Gutter‑only reactives** — once title card docks, no bright moving element may sit over the portrait area; only the dim grid, scanlines, and the corner idents.