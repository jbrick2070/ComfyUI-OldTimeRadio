# `public_domain_plays`: a new verbatim bank for non-Shakespeare plays

Operator rulings (2026-08-03 evening), which govern:

1. **`shakespeare` stays as its own bank and is VERBATIM. That is settled.**
2. **`public_domain` stays as PROSE** -- "it's the LLM's job to try to do book prose
   but not perfect". Adaptation, not verbatim.
3. **`public_domain_plays` is NEW and VERBATIM** -- already-dialogue public-domain
   plays (Wilde, Ibsen, Chekhov, Sheridan).
4. **The word count is a REQUEST, not a gate.** "No refusals -- we do want to see it
   does its best to make text verbatim, but don't kill it if it doesn't work."
   Already implemented: `select_passage` falls back to the closest performable
   window instead of raising, and only raises when a source has no two-speaker
   exchange at all.
5. **No hazard / under-construction flags.** Real code, or nothing.

## What exists already (committed, `a82460ec`)

`nodes/_otr_passage_selector.py` -- picks a contiguous window of consecutive
speeches, verbatim, sized to the word request, the cast ceiling and the beat
topology (`ceil(words/80)` beats per speech; 30-120 target words buy 3 voiced
beats, 150-200 six, 300-1200 fourteen). 26 tests, proven on 14 vendored Folger
scenes. It parses the FOLGER layout only.

## The finding that shapes this plan

I ran the existing parser against a real Gutenberg play (Wilde, *The Importance of
Being Earnest*, ebook 844). It parsed 886 speeches and found the real cast --
ALGERNON, LANE, JACK, LADY BRACKNELL, GWENDOLEN, MISS PRISM, CECILY, CHASUBLE,
MERRIMAN -- but two things are wrong:

- Speaker names carry a **trailing period**: Gutenberg writes `ALGERNON.` where
  Folger writes `ALGERNON`.
- It **invented five speakers out of structural headings**: `THE PERSONS IN THE
  PLAY`, `THE SCENES OF THE PLAY`, `FIRST ACT`, `SECOND ACT`, `THIRD ACT`.

That second one is the Codex QA warning made real: *text alone cannot reliably
distinguish a character label from an uppercase heading*. So format must be
**DECLARED by the manifest, never sniffed from the text**.

## Plan

**P1 -- Declared source formats.** Each source row carries
`source_format: "folger" | "gutenberg_play"`. The selector takes the format and
dispatches to an adapter; no auto-detection, ever. Unknown format = loud failure at
manifest validation, not at render.

**P2 -- A `gutenberg_play` adapter.** Differs from `folger` in exactly two ways:
strip the trailing period from a speech prefix, and reject structural headings.
Heading rejection must not be a blocklist of five strings -- Gutenberg play
headings vary. Candidate rules to pressure-test: a heading is followed by a blank
line and then another heading-shaped line; a heading never recurs (a real speaker
speaks more than once); a heading appears before the first `[stage direction]`.
**CORRECTION -- my own preferred rule was wrong, and measuring it showed why.**
I proposed "require a speaker to appear at least twice". Measured on the Wilde text
it separates cleanly (real characters 17-218 turns, every false label exactly 1),
but it is DANGEROUS: a label that is rejected does not vanish, its line is appended
to the PREVIOUS speaker. So a legitimate one-line walk-on -- a servant announcing a
visitor -- would have their line silently spoken by whoever spoke before them. That
is the Titania-sings-Bottom's-song defect again, just from the other direction.

**Use format SYNTAX instead of frequency.** In the Gutenberg play layout a speech
prefix ENDS WITH A PERIOD (`JACK.`, `LANE.`, `LADY BRACKNELL.`) and a structural
heading does not (`FIRST ACT`, `TABLEAU`, `THE PERSONS IN THE PLAY`, `THE SCENES OF
THE PLAY`). Requiring the trailing period keeps all nine real characters, drops
every observed heading, and cannot merge a walk-on's line into someone else's
speech. The remaining all-caps false positives in my test (`DAMAGE.`, `LIMITED TO
WARRANTIES...PURPOSE.`) are Project Gutenberg licence boilerplate, which
`strip_gutenberg_boilerplate` already removes before parsing -- they only appeared
because the probe read the raw file.

Panel: pressure-test the period rule against a second and third vendored play
before it ships. Frequency may still be useful as a WARNING (log a one-turn
speaker) but must never silently drop a prefix.

**P3 -- Vendor the first plays.** `otr_fetch_public_domain.py` already fetches by
Gutenberg id and writes a provenance sidecar (URL, timestamp, SHA-256 of the
LF-normalized body, parsed speakers). It needs an act/scene slicer for the
`gutenberg_play` layout, matching the existing `--folger-play --act --scene`.

**P4 -- The bank row.** New entry in `nodes/story_packs/banks.json` plus a manifest
under `config/source_banks/public_domain_plays/`. Registry validation requires a
runnable bank to have a real execution lane, so the pipeline/pack wiring has to be
real, not a stub.

**P5 -- Share the lane with `shakespeare`.** Both are verbatim play lanes; they
differ only in source format and manifest. They should share the executor, the
wrapper templates and the ownership table
(`docs/2026-08-03-fidelity-pass-ownership.md`), not fork it.

## Questions for the panel

- Is "a speaker must appear at least twice" safe enough for heading rejection, or
  does it silently drop legitimate one-line characters? Test against real vendored
  plays, not reasoning.
- Should `shakespeare` and `public_domain_plays` be one bank with two formats, or
  two banks sharing one executor? The operator has ruled TWO BANKS; confirm nothing
  in the registry or dropdown makes that expensive.
- `public_domain_plays` has no act/scene manifest yet. Is a curated scene list
  (like Folger's 14) right, or should the bank point at whole plays and let the
  passage selector find the window? The latter is much less authoring work and the
  selector already handles it -- but it loses the human-curated "good scene" signal.
- What breaks first when this lane is wired, given the four overwrite paths already
  recorded in the ownership table?

## Constraints

100% local/offline; render path never touches the network (crawl once, vendor).
16 GB VRAM ceiling. No new ComfyUI node, socket or widget -- `widgets_values` is
positional. Every ledger field needs exactly one owner. Word target is a request,
never a gate.
