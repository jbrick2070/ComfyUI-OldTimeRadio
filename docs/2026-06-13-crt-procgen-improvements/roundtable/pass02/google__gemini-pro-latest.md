<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The aesthetic logic is sound, but introducing frame-to-frame state into the currently stateless render loop will break determinism, and array-rolling will violate the center-column constraint.

MUST-FIX BEFORE BUILD:
1. [S1 / S2] Statefulness / Hidden Dependency. The plan introduces `self._brightness_ema`, `self._fft_history`, and `self._sweep_history` updated per-frame inside `render(fi)`. The existing `render(fi)` is a pure function (ignoring the unseeded random bug). Updating state inside `render()` makes the output dependent on the exact sequential execution of frames, breaking if frames are skipped, resumed, or rendered out of order.
   - Fix: Pass the full `volume`, `freqs`, and `waves` arrays into `_CRTRenderer.__init__`. Precompute the `signal`, `trig`, `fft_history`, and `sweep_history` arrays for all frames upfront in `__init__`. Inside `render(fi)`, strictly read from these precomputed arrays using `[fi]`.
2. [Cross-cutting envelope behaviors / S4] `np.roll` Wrapping & Center Sanctity. The plan proposes using `np.roll` on the procgen frame for "sync-drift" and "chromatic tear". `np.roll` wraps pixels (right edge appears on the left). Rolling the whole image array will also shift the center grid/scanlines into the protected portrait column, violating Constraint 2 (Center-column sanctity).
   - Fix: Do not use `np.roll` on the pixel array. Implement sync-drift and tearing by applying a horizontal coordinate offset (`drift_x`) directly to the Pillow drawing coordinates of the gutter scopes and title card. Leave the center grid and background coordinates untouched.

SHOULD-FIX:
1. [Cross-cutting envelope behaviors] Halation Bloom Implementation. "Draw a larger CRT_DIM duplicate" is computationally expensive if it implies scaling the rendered image, and Pillow does not have a native "bloom" or "glow" filter.
   - Fix: Implement this purely in geometry by drawing the same lines/arcs with a thicker `width` (e.g., `width + 4`), or text with multiple 1px offsets, in `CRT_DIM` *before* drawing the core `CRT_GREEN` elements on top.
2. [S1] RNG Seeding. The plan correctly identifies the need to seed the section-8 noise. However, calling `np.random.seed()` inside a frame render loop alters global state and can cause race conditions if other processes are running.
   - Fix: Use a local generator: `rng = np.random.default_rng(hash(self.title) + fi)` to generate the noise array.

CUT THESE (over-engineering):
1. [OPEN DECISIONS #2] Telemetry micro-text (SYNC / GAIN / dB labels). Safe to cut. The procgen layer is blended via green-only screen and downscaled. Tiny static text will turn into illegible green mush, adding unnecessary Pillow text-drawing overhead and conflicting with the actual `_TelemetryHUDRenderer` HUD.
2. [#2] Peak-hold ghost ring (decaying max per bin). Safe to cut. Requires calculating and storing another precomputed history array. The per-spoke phosphor-persistence comet-tails already provide the necessary visual persistence and decay.

[ASSUMPTION] The plan assumes `OTR_SilentComposite` can handle the procgen layer gracefully on landscape beats. Since `_CRTRenderer` is beat-agnostic, the gutter scopes *will* draw over the edges of full-frame landscape video. I am assuming downstream compositing will mask or accept this, as the renderer has no beat-awareness to prevent it.