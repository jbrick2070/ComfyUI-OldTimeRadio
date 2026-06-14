# pass01 judgment -- grounding the blind creative round

Panel: gemini-3.1-pro, gpt-5.5, deepseek-v4-pro (blind). ~$0.10. All 3 converged.

## ACCEPTED (folded into pass01_plan; grounded vs _CRTRenderer)
- **Signal-strength EMA as master conductor** (3/3). CONFIRMED: `self._brightness_ema`
  exists (alpha 0.08) and was disabled v1.5.1. Dualized (slow/fast) per Gemini.
- **Sweep dot + phosphor-persistence trails** on the scopes (3/3). CONFIRMED feasible
  (ring-buffer state on `self`, dot draws per frame). Left FFT comet-tails + right
  oscilloscope electron-beam sweep.
- **Peak-hold ghost + noise-floor shadow ring** on the FFT scope (DeepSeek). Cheap.
- **Graticules precomputed in __init__ + alpha_composite** (GPT, DeepSeek). CONFIRMED:
  mirrors the existing `_scanlines` precompute pattern.
- **Title: decode -> bold reveal -> POP -> dock** (3/3). Fake-bold via OVERSTRIKE
  (GPT) CONFIRMED necessary: `_load_font` loads only regular monospace (consola/cour),
  no bold variant. Lock POP = brightness/chromatic-tear, not hue.
- **Asymmetric idle/failure states** (GPT): left -> radar sweep, right -> flat baseline.
- **Audio-choked vignette** (Gemini). CONFIRMED: `self._vignette` is multiplied in
  section 8; scaling by the envelope is scalar math.
- **Halation bloom via dimmed larger duplicate** (Gemini, DeepSeek). Cheap, no blur.
- **Particles -> brightness roles not hue** (GPT). CONFIRMED: section-3 cycles
  green/cyan/amber that the green-only blend collapses to luma anyway.
- **SEED the noise** (3/3 determinism). CONFIRMED BUG: section-8 noise uses
  `np.random.randint` with NO seed -> non-deterministic; violates the determinism
  invariant. Fix in S1.

## GROUNDED CAVEATS (accepted with a correction)
- **Sync-drift / V-hold roll** (Gemini, DeepSeek): a green-only SCREEN overlay can
  only ADD green brightness; it CANNOT move/darken/warp the portrait (GPT's catch,
  CONFIRMED). So the roll moves the procgen CHROME, not the face. Must be gutter-clamped
  so chrome never crosses the center column or exposes a black edge.
- **Per-beat landscape awareness** (GPT idea 14, "verify"): DOWNGRADED. The procgen
  floor is beat-agnostic -- `_CRTRenderer` cannot know portrait-vs-landscape per beat
  without new plumbing (the clip manifest timeline). v1 = fixed clamped edge rings.

## DEFERRED TO OPERATOR (defaults set so the plan stays sprint-ready)
- Telemetry micro-text (SYNC/GAIN/dB) -- GPT flagged [ASSUMPTION] re: non-story text.
- Noise inversion (weak=static vs loud=sparkle) -- a behavior change; default ADD
  edge-static rather than invert.

## REJECTED / OUT OF SCOPE
- Any hue-based accent (amber/cyan/white as color) -- dies in the green-only blend.
- Vertical EQ bars / VU needles / 4-ring scope racks -- cliche / too busy (3/3).
- Effects implying portrait displacement/warp -- the overlay cannot move footage.

## OPEN / VERIFY-AT-BUILD
- Exact `led` field for b000 start/end + first-dialogue frame.
- The title card must not reappear in `OTR_SilentComposite` inter-beat gap slices
  (the floor is sliced timeline-aligned) -- map it to the real episode open.
