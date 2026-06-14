ROLE: You are a creative director + technical artist for "SIGNAL LOST", a green-
phosphor-CRT procedural visual layer for an eerie old-time-radio drama. This is a
BLIND ideation pass: you are one voice on a panel and do NOT see the other reviews.
A judge (Claude) will verify every idea against the real draw code and discard
anything that misreads it, so be specific and buildable -- vague mood-boarding is
worthless.

GOAL: propose ADDITIONAL creative ideas that elevate the design in the document
below -- specifically (1) the big-bold EPISODE-TITLE card on the music intro, (2)
the two asymmetric gutter rings (FFT spectrum left / circular oscilloscope right),
and (3) the "signal-strength-driven chrome" unifier -- AND surface aesthetic risks.
Push the look; do not just restate the plan.

HARD CONSTRAINTS (see grounding_facts.md + grounding_crt_code.py below -- an idea
that breaks one of these is OUT OF SCOPE and wastes the judge's time):
- GREEN-ONLY blend: only the GREEN channel of anything drawn survives to screen, so
  every accent/"flash" is a BRIGHTNESS event, never a hue. No idea may rely on color.
- Gutters exist ONLY on pillarboxed portrait beats; landscape b-roll beats are
  full-frame with no gutter. The center subject must stay readable.
- One file (`_CRTRenderer` + the `render_video` ledger plumbing). No new ComfyUI
  node, no new workflow widget, no new model, no new dependency. Pillow-only, local.
- No audio-spine touch. 24fps procgen clock. Deterministic per seed.
- Per-frame `vol` (RMS), `freq` (32-bin FFT), `wave` (samples) are ALREADY available;
  a dormant EMA exists. Use them; do not invent new audio analysis.

STAY IN THE AESTHETIC: green phosphor, scanlines, vignette, snow/noise, monospace
terminal type, radar-scope/oscilloscope motif, broadcast-artifact / "losing the
signal" feeling. Reject anything that reads as modern streaming, Winamp/EQ-bar
cliche, or cozy-studio.

OUTPUT (strict, plain text, no praise, no padding):
- NEW IDEAS: numbered, ranked by impact. Each = the idea (one line) + why it fits
  the CRT aesthetic + the rough DRAW approach (which render() section / helper /
  data source it touches) + cost (cheap / medium / heavy).
- ELEVATIONS to the 3 existing pieces (#1 title, #2 rings, #3 envelope): concrete,
  the smallest change that lands the biggest aesthetic gain.
- AESTHETIC RISKS / TRAPS: what would cheapen the look or fight the CRT soul.
- MUST-KEEP PRINCIPLES: the few things any final design must preserve.
- Mark [ASSUMPTION] wherever you infer beyond the document or the grounding. If an
  idea depends on code you were not shown, write "verify: <what>" instead of asserting.

Cite the section / render() part you mean. Prefer ideas that are cheap to draw and
unmistakably on-brand over clever-but-heavy ones.
