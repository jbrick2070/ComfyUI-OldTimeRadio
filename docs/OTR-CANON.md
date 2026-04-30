# OTR — SIGNAL LOST Canon

Living document. Auto-updated post-render by the `_canon_update`
helper in `nodes/script_critic.py` (when implemented) or by manual
edits between sessions. Everything here is hard-canon: the writer
includes this in its system prompt at every run.

**Last updated:** 2026-04-29 (initial seed)

---

## Tonal canon

SIGNAL LOST is a 1940s-style sci-fi radio anthology. Each episode is
seeded from a real science news article (see `meta.news_seed` in the
ledger). The dramatic register is closer to *Suspense* (1940-1962)
and *X Minus One* (1955-1958) than to modern audio fiction:

- Voice work assumes a single mic per actor in a small studio.
- Foley is procedural and minimal — door, footsteps, scene-end
  sting. No constant ambient bed.
- Scenes are short (60-180 sec) and resolve on a concrete object,
  not a moral.
- Cliffhangers are concrete (a name not spoken, a door left open),
  not abstract (a character "wondering").

## Period rules

- Year-of-setting unless overridden by the news article: **1947**.
- No vocabulary post-1947 unless quoted from the news article. If
  the article uses modern terminology, the announcer translates it
  to a period-appropriate periphrasis on first use, then the
  characters use it freely.
- No technology beyond what existed in 1947 unless the article's
  central premise requires it (e.g. a story about quantum biology
  can include the modern science as the inciting event but the
  characters discuss it through 1947-era frames).
- Characters do not say "okay", "guys", "no problem", "you got
  this", "for sure". Period replacements: "very well", "men" /
  "fellows" / "the team", "no trouble", "I'm with you", "indeed".

## Recurring motifs

- **Radio static** as a transition only. Never as an ambient bed.
- **The Announcer** opens and closes every episode.
- **Names** are 1940s-American unless the article setting demands
  otherwise. Avoid sci-fi-coded names ("Zara", "Kade", "Vox").
- **Sound design** uses one motif per episode beyond the radio
  bookend — picked by the LLMDirector, not pre-specified here.

## Used premises (auto-updated)

This list grows after each successful render. The writer is
prompted to avoid repeating any premise within the rolling window
of the last 25 episodes.

- _no entries yet — first canonical episode pending_

## Used twists (auto-updated)

- _no entries yet_

## Used motifs (auto-updated)

- _no entries yet_

---

## How this file is read

`nodes/story_orchestrator.py::LLMScriptWriter` — at script-write
time, the writer reads this file and inlines the **Tonal canon**,
**Period rules**, and **Recurring motifs** sections into the
system prompt verbatim. The "Used premises", "Used twists", and
"Used motifs" sections are inlined as a "do not repeat" list.

If this file is missing, the writer falls back to the inline
defaults in `SCRIPT_SYSTEM_PROMPT` and logs a warning. Never a
hard fail — canon is an enhancement, not a blocker.

## How this file is written

`nodes/script_critic.py::LLMScriptCritic` — when an episode passes
the critic gate, the helper `_canon_update(ledger, canon_path)`
extracts the picked premise/twist/motif from the script and
appends to this file in the appropriate section, capped at the
last 25 entries per section so the writer's context budget stays
bounded.
