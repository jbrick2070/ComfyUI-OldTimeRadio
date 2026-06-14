# pass03 judgment -- round-3 final QA (convergence)

Panel: gemini-3.1-pro, gpt-5.5, deepseek-v4-pro. ~$0.16 (cumulative ~$0.40). The
DESIGN converged; round 3 found only implementation-PRECISION must-fixes (no new
design direction) -> stopping at the operator-specified 2 QA rounds. All folded into
the final §4C.

## ACCEPTED MUST-FIX (precision; grounded)
- **Unify the constructor signature** (3/3): pass02 contradicted itself (full arrays
  vs `timing=None`). Final: `_CRTRenderer(w,h,title,volume,freqs,waves,fps,timing=None)`,
  `self.total=len(volume)`, `render(self, fi)`; update render_video L1556/L1559.
- **Half-open intervals** (GPT): `start <= fi < end`; disable if None or end<=start.
- **import hashlib** (GPT, Gemini): not imported today; the blake2s seed needs it.
- **24fps vs variable fps** (GPT): the code uses the runtime `fps` (default 24); the
  title clock must use the ACTUAL fps, not hardcode 24.
- **Geometry from the real portrait scale + amplitude clamp** (Gemini, concrete): the
  480x832->1920 portrait gives gutters [0,~647] / [~1273,1920]; ring centered in each
  (left_cx~323, right_cx~1596), r ~235; the circular-scope amplitude MUST be clamped
  (amp <= r*0.35) so r+amp never crosses into the center band OR past the frame edge.
- **Text exemption via draw order** (Gemini, DS): section-1 text draws BEFORE the
  section-8 vignette multiply today; draw the ident + title card AFTER the vignette
  (or on a post-vignette pass) so the choke never dims text -- the exact v1.5.1 bug.
- **Dock needs a tail window** (DS): the card animates down AFTER the music window;
  active window = [music_open_start, music_open_end + dock_frames).
- **EMA init** (GPT, DS): `signal[0]=trig[0]=volume[0]`, recurse from 1; handle
  total_frames==0.
- **led timing extractor RESOLVED** (3/3 flagged): per production_ledger.py the
  SceneSequencer stamps `led["lines"]` with `speaker_role` + `start_s` + `dur_s`
  (persisted to the DISK ledger; the wire ledger may have start_s=None, so resolve the
  TIMED ledger from disk the way otr_caption_burn does). music_open = first line whose
  speaker_role is a music-open role; window = [start_s, start_s+dur_s]*fps;
  first_dialogue_f = first dialogue line's start_s*fps. Fallback: derive the intro
  window from the `volume` envelope if start_s is unavailable.

## ARCHITECTURE simplification (round-3 cut, accepted)
- Do NOT precompute comet-tail / sweep history arrays; compute them in `render(fi)`
  by BOUNDED LOOKBACK over `freqs[max(0,fi-N):fi+1]` / `waves[fi]` (N~6). Reads past
  INPUT (not mutable state) -> render(fi) stays pure + deterministic + cheaper.
- Center-band clip = draw each scope on a transparent GUTTER-RECT layer +
  alpha_composite (the layer bounds clip it); no Cohen-Sutherland, no scope primitives
  on the base image.

## ACCEPTED CUTS for v1 (panel consensus)
Telemetry micro-text; FFT peak-hold ghost ring + noise-floor shadow ring; oscilloscope
free-running trigger seam; halation (2x draw cost threatens 24fps gen speed); the
FORMAL hierarchy layer-floor system (use per-element brightness scaling -- grid scales
down faster than the ident).

## GROUNDED -- feared blocker still CLEARED
Title-card gap-fill reprisal: `OTR_SilentComposite._floor_aligned` slices the floor
TIMELINE-ALIGNED + the blend is FRAME-ALIGNED, so the open's title frames land only at
the head/open. Kept as a 2-beat verify smoke; no cross-file change.

## CONVERGENCE
3 rounds, 9 grounded reviews, ~$0.40. Design stable since pass01; pass02/03 hardened
the wiring. Remaining items are explicit VERIFY-AT-BUILD smokes, not open design.
