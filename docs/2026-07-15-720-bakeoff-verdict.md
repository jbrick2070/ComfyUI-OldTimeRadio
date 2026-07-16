# 720-Word Bake-off -- Meta-Analysis & Verdict

**Date:** 2026-07-15
**Scope:** 10 source banks x 3 word-count rungs (320/420/720), one pinned model pair
(creative `aion-labs/aion-3.0-mini` via OpenRouter, ctx 131,072 / technical local
`Mistral-Nemo-Instruct-2407`, ctx 8,192) held constant across every leg.
**Judged rung:** 720 -- the only rung that ran on fully uniform code and prompts
across all ten banks (`docs/2026-07-14-bakeoff-observations.md` OBS-1b/1c). 420 and
320 have no blind read; reasoned about below from structure only, with an added
confound noted at 420.
**Hard rule carried through:** word count is a lane property, never a quality gate.
A leg is valid on render success; length/ratio figures are descriptive only.

---

## 1. Best banks overall (story quality)

Ranked on DATA1, the 720 blind read -- the only quality signal that exists, and the
only rung proven to run on identical code for all ten banks.

1. **scifi_fable2** -- complete moral-dilemma radio play, best dialogue, only lane that fills the length with sustained tension.
2. **public_domain_story** -- faithful Time Machine adaptation, best source integration, clean reversal ending.
3. **scifi_codex** -- coherent publish-or-stay-silent ethics story, three distinct characters, honest disclaimer.
4. **shakespeare** -- lively, mostly faithful Comedy of Errors III.i; undercut by an invented "signed deed" device and no ending.
5. **media_archive** -- solid archivist cover-up standoff; ending a touch muddy.
6. **scifi_sonnet** -- best single speech in the set, but a themed monologue in a bureaucratic frame, not a drama.
7. **original_radio** -- strong noir mood, muddy/opaque plot.
8. **science_news** -- ambitious "ghost climate-scientists rewrite a play" conceit that collapses into incoherence.
9. **original_codex56sol** -- slight, low-stakes vent-grille puzzle; little story present.
10. **scifi_gemini** -- a ~45-word fragment; not enough story to judge.

**Top.** scifi_fable2's win isn't a flat quality edge -- DATA2 shows *why*. At 320/420
it looks like every other short-form lane: 7-11 beats, 3-7 voiced lines, 97-115
words/line -- essentially one long speech. At 720 it jumps to 68 beats and 64 voiced
lines at 12.3 words/line ("The Far Shore Relay," prompt `059ac956`) -- it structurally
converts into a real multi-character radio play only once it has room to. That is a
length-*dependent* transformation, not a static advantage, which matters for
Section 2: do not assume the 720 crown carries down to shorter rungs.

**Bottom.** scifi_gemini isn't just weak, it isn't really producing episodes. Its
asset is the smallest bank at every single tier (51.8 / 53.9 / 32.7 MB) and it
undershoots its target worst every time, cratering to a 45-word "River Code" at 720
(prompt `be813fb7`). That reads as a generation/reliability problem more than a
writing problem -- see Section 4, there's a specific named suspect on record.

## 2. Best bank per tier

**Archetypes** (from DATA2 structure, corroborating DATA1):
- *Fixed-template adaptation/conversation* (~18 beats at every tier, length grows via
  longer lines, not more beats): media_archive, original_radio, public_domain_story,
  science_news, shakespeare.
- *Compact multi-character conversation* (9-12 beats, beat count grows a little at
  720): scifi_codex.
- *Speech/monologue* (few long lines): scifi_sonnet at every tier; scifi_fable2 at
  320/420 only.
- *Thin/likely broken*: original_codex56sol, scifi_gemini.
- scifi_fable2 is the one lane that changes archetype by tier -- monologue at
  320/420, full ensemble drama at 720.

**720 (known).** scifi_fable2, by a clear margin -- see Section 1.

**420 (inferred, and confounded).** Per OBS-1b, the 420 rung is explicitly flagged
by the project's own records as **not a clean bank-vs-bank comparison**: schema caps
(`role_in_conflict` 120->180, `StructureReviewV4.rationale` 240->400) were raised
mid-sweep. Seven banks (media_archive, original_codex56sol, original_radio,
science_news, scifi_gemini, scifi_sonnet, shakespeare) ran their 420 leg at the
original caps; three (public_domain_story, scifi_codex, scifi_fable2) were re-legged
after the raise. Treat any 420 number touching those three as not directly
comparable to the other seven. With that caveat: **public_domain_story** is the
safer pick -- it's the most structurally consistent bank at every tier (near-1.0
ratio at both 320 and 420, ~18 beats everywhere) and its 720 strength (faithfully
pacing a fixed source text) isn't a length-dependent trick, so it likely holds.
scifi_codex is a reasonable second guess, though its words-per-line swings a lot
across tiers (19.4 -> 61.4 -> 17.4), suggesting its pacing isn't stable across
lengths. Flag: scifi_fable2 at 420 is still in "long speech" mode structurally (11
beats, 7 lines, 97.6 words/line) -- it has not yet made the jump that won it 720, so
don't assume the win repeats here.

**320 (inferred).** public_domain_story again, for the same consistency reasons (18
beats, ratio 1.009 -- the closest any bank comes to hitting its target exactly).
Known fact, not inference: **scifi_sonnet has no valid 320 episode at all (FAIL)** --
it is not eligible for this tier as it stands. scifi_fable2 at 320 is at its most
extreme monologue shape of any bank at any tier (7 beats, 3 lines, 115 words/line) --
almost certainly reads as a single speech, not a play; its 720 ranking is
uninformative here.

## 3. Model analysis

All ten banks ran on one pinned **pair**, not just one model: `aion-3.0-mini` as the
creative/story-writing slot, local `Mistral-Nemo-Instruct-2407` as the technical
slot (structured extraction/validation passes), both held constant across every
bank and every tier. That's confirmed by `docs/2026-07-14-720-bakeoff-hardening.md`,
and it means DATA3 is right: every quality difference in Section 1 is a pure
bank/prompt effect, not a model confound. The Section 4 verdicts are safe to act on
without worrying "maybe it's just this model."

What it does **not** tell you is which underlying model is best for OTR story
writing -- that comparison doesn't exist, because the local *creative-writer*
matrices (mistral, gemma) were smoke-blocked overnight and never ran. (Don't
conflate those with Mistral-Nemo above -- that's the same model family in a
different, technical role, already running on every leg.) Two specific things need
those blocked matrices before "best model" is answerable: whether a lane's ranking
holds across models -- e.g. does scifi_fable2's length-dependent transformation
into full drama happen on mistral/gemma too, or is that aion-specific
instruction-following; and whether the weak lanes (science_news's collapsing
conceit, scifi_gemini's thin output) are prompt problems any model would struggle
with, or aion-specific failures. Until mistral/gemma run the same 10x3 matrix --
ideally through the same blind-read protocol -- there is cross-bank signal on one
model, and no cross-model signal at all.

## 4. Keep / Improve / Leave

| Bank | Verdict | Reason |
|---|---|---|
| scifi_fable2 | **KEEP** | #1 story, and the only lane proven to use extra length for real dramatic payoff. Worth checking whether 320/420 need pacing help so they aren't a stranded monologue. |
| public_domain_story | **KEEP** | #2 story, most structurally reliable bank at every tier -- the closest thing to a safe default lane. |
| scifi_codex | **KEEP** | #3 story, coherent multi-character writing with honest sourcing. Its own spec review already flagged the exact-integer P3 beat table as a re-roll generator and ordered it advisory-only -- worth confirming that fix is what's live, given the tier-to-tier pacing swings. |
| shakespeare | **IMPROVE -- fixable, already half-patched** | Undercut by a device not in the source and no ending. The team caught and patched this exact failure mode on 2026-07-14 (`exchange_system` was forcing invented modern subtext onto the adaptation; rewritten to "put a microphone on the source, don't re-plot it") and ruled the verdict must score the new-seam 720 episode (prompt `c42700e1`, "The Bolted Door"). If DATA1 read that leg, the fix didn't fully take -- worth a second pass, since the sibling lane below took the same fix cleanly. |
| media_archive | **KEEP** | #5, solid and consistent; muddy ending is a light tune, not a rebuild. |
| scifi_sonnet | **IMPROVE** | Best single speech in the set, but a monologue in a bureaucratic frame (the "Continuity Archive audit" format), not a drama. Its own spec review already flagged nine tightly-worded seams as heavy surface area for the technical model -- the likely reason it's also the one bank with an outright FAIL (no valid 320 episode). Seam consolidation is the already-identified fix. |
| original_radio | **IMPROVE** | Strong noir mood undercut by a muddy/opaque plot -- a clarity/throughline fix, not a lane rebuild. Don't lose the mood chasing this. |
| science_news | **IMPROVE -- fixable prompt-craft** | The "ghost climate-scientists rewrite a play" conceit is too ambitious to sustain at 720 and collapses into incoherence. Structurally the lane is steady (consistent 18-beat template every tier) -- constrain the concept rather than retire the lane; it may already hold together better at 320/420, where there's less room for it to unravel. |
| original_codex56sol | **LEAVE** | Consistently the thinnest bank at every tier (fewest words/line, lowest output). The obvious technical culprit -- a hardcoded 8K context window that would have truncated its 720 output -- was caught and fixed before this sweep (live-reverified at ctx=131072), so the thinness is a genuine content/premise problem, not a truncation artifact. The vent-grille puzzle doesn't appear to generate enough dramatic material regardless of length. |
| scifi_gemini | **LEAVE -- check this first** | Worst story and the least reliable generation: smallest asset every tier, worst undershoot every time, a 45-word fragment at 720. Its own spec review (4 days before the sweep) flagged that the spec document itself carries two contradictory engine versions -- an "archived v2" skeleton alongside the corrected v3/8A -- and warned an implementer could build against the wrong one. That's a named, specific suspect worth checking before writing this off as a creative failure. |

## Open verification items

- Confirm which seam version actually produced the judged shakespeare 720 leg
  (prompt `c42700e1`) -- decides whether the device/ending defect needs a second
  prompt pass or was already supposed to be fixed.
- Check the live scifi_gemini implementation against the spec's v3/8A vs the
  archived v2 skeleton before spending any creative-side effort on that lane.
- Unblock the mistral/gemma creative-writer matrices -- the only way to answer
  "best model" rather than "best bank on aion."
