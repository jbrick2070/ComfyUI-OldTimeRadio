# r1 judgment -- Prompt v3, draw the crux (2026-09-02)

Roster, r1: Fable 5.1 cold read (creative, no sight of the anchor) + Antigravity (`agy`,
Gemini 3.8 Flash (High), `antigravity.md`, 9.5 KB). Driver: Claude. Every claim checked at
the real files before disposition. The two reviews disagree head-on about the character, and
the arithmetic decides most of the rest.

## The measurement that settles three claims at once

Estimating tokens at four characters each, on the real packs and the real brief:

| pack | compact cue | FULL positive_tail | brief (crux) | cue+crux | tail+crux |
|---|---|---|---|---|---|
| recur_frac | 9 | 58 | 28 | 37 | 86 |
| video_art | 8 | 38 | 28 | 36 | 66 |
| storybook_engraving | 6 | 27 | 28 | 34 | 55 |
| anime | 4 | 14 | 28 | 32 | 42 |

The gate is 69 at author time and 77 at the render boundary. So Antigravity M1 is CONFIRMED
in its arithmetic: the pack's FULL tail plus the brief verbatim is 66-86 tokens on the two
styles the operator cares about, leaving nothing for the beat or the motion, and
`assert_shell_fits` would refuse at boot. But its fix (go back to `compact_style_cue`) is the
defect the campaign exists to remove. **DISPOSITION: neither.** The style keeps a compact FRONT
anchor (the two-word cue, 4-9 tokens) and gains a 3-5 word TAIL lifted from the pack's own
language (Fable's shape), which is 8-12 tokens, not 27-58; and the CRUX is a compact SUBJECT
KERNEL (<= 15 tokens: `key_objects[0]` + the setting's first term), not the brief sentence
verbatim. Budget: style 12-20, crux 15, beat subject 12, world motion 12-18, punctuation --
inside 69 on every pack. The brief sentence is what the kernel is DERIVED from, never what is
sent.

## Must-fixes

1. **M1 token collapse** -- CONFIRMED (table above). TAKEN as the kernel + tail rule; the
   anchor's "the pack's authored language" and Fable's "3-5 word tail" both become: front cue
   compact, tail 3-5 words, crux a kernel. Recorded as a boot-time test over all nine packs.
2. **M2 "no people" / "no faces" in the positive** -- CONFIRMED at the file, and it is a
   standing law, not a preference: `ghost_signal_prompt.py:109-112` -- "There is no `no people`
   here and there never will be ... a positive clause that attends to an absent human is a
   request for the model to think about one." The anchor's D6 floor violated it. TAKEN: the
   positive law is ZERO words (Fable's answer to Q4 anyway); exclusions live in
   `compose_ghost_negative`, which already carries the lane hygiene head. "unbroken shot" is
   the only affirmative clause held in reserve.
3. **M3 total character erasure** -- CONFIRMED as a real risk and it is the one place
   Antigravity reads the operator better than the anchor did. His rule 1 says "characters
   moving through that world -- small, in it, never a coat in close-up", and his own rewrite
   for Sarah's beat is *"characters moving through a stagnant mass of water at a reservoir"*.
   The anchor's D2 ("the character disappears") over-corrected. TAKEN, with Fable's mechanism:
   a `hand` vantage draws one hand or a turned back on the story's thing, and a `world` vantage
   may carry distant plural shapes ("two small shapes on the gantry above the water") -- never
   a face, never clothing, never a named person. `_HUMAN_WORDS` (`:346-355`) already bans
   figure/man/woman/silhouette/pronouns while permitting hand, arm, shoulder, back; v3 applies
   it to every vantage. So the person is never the SUBJECT and never a costume, but the world
   is not emptied of people.
4. **M4 dialogue nouns are conversational** -- CONFIRMED in principle ("What did you find?"
   yields nothing) and it is why Fable's `key_objects` reading is the better source. TAKEN,
   and it REPLACES the anchor's D4: the beat subject is chosen from `meta.key_objects`
   (measured present: handheld brass communicator, telemetry screens, data logs, hydrographic
   charts), SELECTED by whichever the beat's stripped nouns are nearest, falling back to the
   scheduler's rotation through `key_objects` so a dialogue-only beat still varies. No
   open-vocabulary scraper, which also closes Antigravity S2 and Q3.
5. **M5 replay-time prompt mutation corrupts provenance** -- CONFIRMED: `render_request_hash`
   and `request_sha256` are the admission and cache keys, and the replay branch is contracted
   immutable. TAKEN: the v2-vs-v3 comparison is a FRESH run pinned to the frozen seeds, not a
   mutated replay -- the engine-override bundle already proves the shape (a derived bundle with
   its own manifest), so v3 gets the sibling: a derived bundle that re-authors prompts and
   stamps `prompt_version=ghost_signal_v3` with new hashes, while `seed_bundle.request_seed`
   is carried from the frozen plan. Q5 answered; the anchor's D9 is rewritten.

## Should-fixes

* **S1 creative suppression** -- the sharpest disagreement between the two reviewers.
  Antigravity wants the LLM to synthesize the whole visual sentence with Python as validator;
  Fable keeps Python owning crux, vantage and tail and gives the model one 8-12 word motion
  clause. TAKEN: Fable's split, with one concession to S1 -- the model may name the beat's
  SUBJECT from `key_objects` in its clause (so it can write "the hydrographic charts sliding
  off the desk into the water" rather than being handed the object), which is where the
  creativity the operator asked for actually lives. Python still owns the crux kernel, the
  vantage, the light and the tail, because those are what keep 29 beats one episode.
* **S3 the finalizer never trims** -- CONFIRMED verbatim at `:1306-1308`. TAKEN: all drop-order
  logic lives at COMPOSITION time; `finalize_ghost_prompt_v2` stays a refusal gate. The anchor's
  D7 said "drop order under the window" without naming the seat; it is the composer's.
* **S4 setting stutter** -- CONFIRMED (the brief already contains "research station"). TAKEN:
  the setting term is appended only when the crux kernel does not already contain a location
  word.
* **S2** is closed by M4's disposition (no open vocabulary).

## Optional, both taken

Per-slot token counts on `observability.prompt_slots`; and the prompt-length-vs-motion-module
question is exactly what the v2/v3 same-seed pair measures, so it costs nothing extra.

## Where this leaves the plan

D1 (shape) becomes five slots with a compact front cue and a short pack tail. D2 becomes
"the person is never the subject and never a costume; hands, backs and distant shapes are the
grammar". D4 becomes `key_objects`-driven. D6 becomes zero positive law words. D7's drop order
moves to composition. D9 becomes a fresh seed-pinned run, not a mutated replay. D3, D5, D8 and
D10 stand. r2 (Codex) plans the code against that.
