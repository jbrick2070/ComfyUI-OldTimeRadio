<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The precompute architecture and coordinate-offset fixes are excellent, but missing imports, draw-order conflicts, and unscaled geometry will break the build and violate hard constraints.

MUST-FIX BEFORE BUILD:
1. [DETERMINISM] **Missing import**: The plan relies on `hashlib.blake2s` for the local RNG seed, but `hashlib` is not imported in `nodes/video_engine.py` (Excerpt 1, lines 24-34). 
   *Fix*: Add `import hashlib` to the top of the file.
2. [S4] **Draw order prevents text exemption**: The plan states the vignette choke "EXEMPTS the title/ident text from any dimming". However, in the current pipeline (Excerpt 1), Section 1 (text) draws to `img` *before* Section 8 converts it to an array and multiplies by `_vignette`. 
   *Fix*: Move the text drawing routines (Section 1 and the new S3 Title Card) to execute *after* the Section 8 vignette/choke array multiplication is converted back to a PIL Image, or apply the vignette only to a background/scopes layer before compositing the text.
3. [S2] **Oscilloscope wave amplitude violates center-column sanctity**: The right scope is at `cx=1632` with `r=249`. The center protected band ends at `x=1284`. The maximum safe outward trace is `1632 - 1284 - 249 = 99px`. The existing waveform code (Excerpt 1, line 349) uses `amp = wave[i] * h * 0.45` (up to 486px). If traced around the circle, it will massively invade the center portrait.
   *Fix*: Explicitly scale down the wave amplitude for the circular scope to a strict maximum (e.g., `amp = wave[i] * r * 0.35`) so `r + amp` never exceeds the 348px distance to the center band.
4. [ARCHITECTURE FIX] **Broken `render_video` caller signatures**: The plan requires passing the full audio arrays to `__init__` and making `render(fi)` pure, but fails to update the caller in Excerpt 3.
   *Fix*: In `render_video`, update line 1556 to `renderer = _CRTRenderer(W, H, episode_title, volume, freqs, waves, timing_dict)` and update line 1559 to `return renderer.render(fi)`.

SHOULD-FIX:
1. [S3] **Textbbox wrapping**: When measuring `ImageDraw.textbbox` for the hero title, ensure the calculation accounts for multi-line text if the title is long and needs wrapping. Pillow's `textbbox` does not auto-wrap; you must split the string into lines manually before measuring/drawing to prevent horizontal overflow.
2. [S1] **Precompute allocation**: When precomputing `sweep_idx` and `fft_tips` in `__init__` across `total_frames`, pre-allocate them with `np.zeros` rather than appending in a Python loop to ensure initialization is instantaneous.

OPTIONAL / NICE-TO-HAVE:
- [ASSUMPTION] The plan assumes `timing` dict fields like `music_open_start_f` are guaranteed to be integers if present. Add a quick `int()` cast when parsing them from `led` to prevent `TypeError` during frame index comparisons.

CUT THESE (over-engineering):
1. [S4] **Halation via THICKER-WIDTH CRT_DIM pass**: Drawing complex FFT and oscilloscope geometry twice per frame in pure Python/Pillow will significantly increase CPU render time. Cut this to protect the 24fps generation speed; the persistence from comet-tails and sweep trails already provides enough visual thickness.