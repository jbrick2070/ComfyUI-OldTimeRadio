# ideo_word_vid -- Animated Word-Video for Dialogue Beats (high-level, r1)

**STATUS: DRAFT (pre-panel)**
Candidate plan -- ideas window 2026-07-02. Companion to
`2026-07-02-ideogram-lyric-stills.md` (the STILL card lane, CODE-READY/S1-gated). This doc is
the HIGH-LEVEL exploration of the animated version: not still cards, but MOVING word-video --
kinetic typography -- that takes the character/announcer dialogue and animates the words while
retaining the visuality (period look, styling, mood) of the spoken line.

## Problem / goal

The still-card lane freezes a beat's words into one designed frame. The operator wants the words
to MOVE: an `ideo_word_vid` engine where the dialogue text itself is the animation -- words
appearing/sliding/burning in sync with the beat, lyric-video style, period-styled (1940s title
cards, radio-serial typography), one clip per character/announcer beat. The visual identity of
the dialogue (who is speaking, era, mood from the Meta brief) shapes the look; the words shape
the motion.

## Candidate approaches (to be ranked by the panel -- high level only)

- **A. Still -> cloud i2v animation.** Mint the Ideogram card, animate with a cloud i2v pass.
  KNOWN BLOCKED today (grounded in the companion doc): no plain Kling i2v row; cloud_wan_i2v
  forwards no text prompt + has a request-shape bug; seedance dark until V3 pin expansion.
  Revisits automatically when a promptable i2v row lands.
- **B. Multi-card sequence (pseudo-animation from stills).** Chunk the line into 2-4 Ideogram
  cards (word-progression: phrase 1 -> phrase 2 -> full line), cut them rapid on intra-beat
  timing (audio duration split), with the existing still-motion pass per card. No new model
  capability needed -- pure sequencing on the CODE-READY stills lane. Cost = N x $0.043.
- **C. Procgen kinetic typography (local, zero cloud cost).** The repo already burns styled
  text into video (SDH captions at Node 58; procgen rolling credits). A procgen word-animation
  engine (per-word timed reveal, slides, scale pulses, period fonts/palettes from the style
  inventory) driven by the line text + beat audio duration -- possibly word-timed if per-line
  TTS timing exists. Fully deterministic, seed-keyed, free. (Operator's cloud-only ruling was
  for the IDEOGRAM lane; a procgen typography engine is a different engine -- needs a ruling.)
- **D. Hybrid: Ideogram background + procgen word overlay.** Ideogram mints a wordless
  period-styled BACKGROUND plate (title_mood mode already does wordless); procgen animates the
  actual words over it. Gets Ideogram's look + deterministic legible motion + exact text
  fidelity (no model-rendered-text risk at all). Cost = 1 card per beat + free overlay.
- **E. Cloud t2v with in-video text.** Ask a text-capable t2v provider to render animated
  typography directly. No pinned t2v row supports reliable in-video text today; text fidelity
  in video models is far weaker than Ideogram stills. Likely reject; panel to confirm.

## Working preference (to pressure-test)

D (hybrid) as primary -- it is the only approach where the words are ALWAYS legible and exactly
the script text, the look is still Ideogram's, and the motion is controllable and seed-keyed.
B as the zero-new-code fallback; A parked on the i2v blockers; C as D without the Ideogram
plate (cheapest); E rejected pending panel.

## What it touches (high level)

- D: title_mood-style background mint (stills lane, S1-gated) + a NEW procgen word-overlay
  video engine (ffmpeg/PIL text animation, same family as captions/credits code) + word/phrase
  timing rule from beat audio duration.
- Registration dark/fail-closed, EMPTY defaults, selectable for character_video /
  announcer_visual (and music_visual with no words = plain animated plate).
- Workflow JSON: no new widgets expected (V-11); verify selector exposure at build.

## Risks / open questions (high level)

- Word-timing without forced alignment: per-word sync to TTS audio needs timing data --
  is per-line duration enough (evenly-spaced reveal), or does the repo have phoneme/word
  timestamps anywhere (SDH caption timing?)? Verify-at-build.
- Legibility at 1472x832 landscape + downstream encodes; minimum on-screen ms per word.
- Style drift between the Ideogram plate and the overlay font/palette -- palette must be
  extracted from or dictated to the plate (style inventory keys both).
- Continuity: kinetic-type cutaways mid-dialogue vs full-beat replacement -- v1 = full-beat
  replacement (same slot the still card occupies).
- SFW/text: same upstream gates as the stills lane.

## Rough size (if D)

Medium: background mint reuses the stills lane; the word-overlay engine is the real work
(~1-2 sessions: text layout/animation renderer + timing rule + tests). After S1 + the
still-card lane.
