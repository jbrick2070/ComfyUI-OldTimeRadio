# QA target: verbatim passage selection for the play lanes

Review `nodes/_otr_passage_selector.py` and `tests/test_passage_selector.py`
(both new, uncommitted at review time) against the real repo. This is a CODE QA
pass, not a plan review: find defects, missed constraints, and wrong assumptions.

## What it does and why

Operator ruling: for the play lanes an episode is "very strict -- based on word
count and random choice it hones in on a specific part of a play to get real
specific dialogue, no paraphrasing." So an episode is a contiguous WINDOW of
consecutive speeches, carried verbatim, chosen to fit the word budget, the cast
ceiling and the beat topology. Nothing is summarized, so nothing can drift.

Context: both fidelity banks were found running on placeholder text (the Wells
fixture contained no Wells; all 14 Folger fixtures were ~100-word collages).
Authentic text is now vendored under `config/source_banks/*/sources/` with
provenance sidecars. A delivered episode had narrated an *As You Like It* scene
as if it were *Romeo and Juliet*, because the writer never saw the source.

## Design decisions to attack

1. **Form, not author.** The module is deliberately generic over play-form
   sources (Wilde, Ibsen, Chekhov would parse identically) and deliberately
   REFUSES prose -- `parse_speeches` returns empty, `select_passage` raises.
   Is that refusal airtight, or can prose with an accidental all-caps line slip
   through and yield a nonsense "passage"?
2. **Beats bound the passage, not words.** Measured from
   `_otr_episode_budget` (`auto_act_count` + `ACT_COUNT_CONFIG`): 30-120 target
   words buy exactly THREE voiced beats, 150-200 six, 300-1200 fourteen, and the
   hard ceiling is 19 beats x 80 words. `max_speeches` is a required argument.
   VERIFY that mapping is right, and that one-speech-per-voiced-beat is the
   correct model of how a row is spoken -- if a beat can carry more than one
   speech, or a long speech must SPLIT across beats, the constraint is wrong and
   I want to know now.
3. **Seeded choice.** Same seed, same passage, so a render is replayable.
   Is `sha256(seed)` mod window-count an acceptable selector, and is the seed
   plumbing going to collide with the existing cast-seed / style-seed receipts?
4. **Fail loud.** No window that fits => `PassageError`, never a relaxed
   tolerance or a trimmed source. Check every raise site for a path that could
   still degrade silently.
5. **Speech parsing.** Two Folger layouts, neither with a colon: verse puts the
   name alone on its line, prose writes `TOBY  Come thy ways` inline; either may
   carry `, [as Ganymede]`. Stage directions in brackets are dropped from spoken
   text. Indented lines continue the current speech.
   Attack this parser: `ALL`, `FIRST WITCH`, `ANTIPHOLUS OF EPHESUS`, a speech
   whose first line is itself bracketed, a scene heading, an all-caps line of
   dialogue, elision apostrophes (`I' th' name of truth`), and the em-dash and
   curly-quote characters Folger actually uses.

## Specific questions

- **Does the tolerance window make sense?** +/-25% around target, min 2
  speakers. At 300 words that admits 225-375. Too loose for a word budget the
  operator sets deliberately?
- **Is O(n^2) window enumeration acceptable?** The largest vendored scene is 144
  speeches; the outer loop breaks early on words/cast/speeches. Measure rather
  than assume.
- **Dramatic quality.** Selection is currently mechanically valid but
  dramatically naive -- a window can begin mid-argument and end mid-thought,
  while every OTR episode is required to have a start, a middle and an end.
  Propose DETERMINISTIC criteria (not an LLM pass) that would improve this:
  prefer windows starting after an entrance or a question, ending on a couplet,
  an exit or a resolution, avoiding starting mid-verse-line.
- **What breaks when this is WIRED?** It is not wired yet, by choice. The known
  blockers: the count-match invariant at `OTR_LedgerScriptWriter.py:4061-4067`
  hard-raises when locked != requested cast; `custom_premise`
  (`OTR_LedgerScriptWriter.py:1738-1760`) bypasses the authenticated fetcher for
  these banks; `run_post_script_spine` -> `strip_line_formatting`
  (`nodes/_otr_ledger_scrub.py:191-230`) can rewrite spoken text after
  composition, which would mutate verbatim source. Name anything else that would
  overwrite or reject source-owned rows.
- **Prose lane.** The operator is weighing two variants for prose sources -- a
  faithful/verbatim treatment versus a paraphrased dramatization ("public domain
  drama vs public domain paraphrase"). Given prose has no speech prefixes, say
  what the faithful variant should concretely BE (narrator/reader role speaking
  Wells' own sentences?) and whether it shares any of this module.

## Constraints

100% local/offline; 16 GB VRAM ceiling; no new ComfyUI node, socket or widget
(`widgets_values` is positional -- inserting shifts every saved value in
`workflows/otr_canonical.json`). Ledger fields need exactly ONE owner each.
