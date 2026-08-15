# Operator eyeball on the overnight episodes, 2026-08-15 -- six findings

Watched by the operator on the pass-1 artifacts. Every claim below was
grounded against the real ledgers in `output\otr\episodes\`, not inferred from
the screenshots. Admissible under the live-artifact rule.

## 1. The SHAKESPEARE lane names nobody. `spoken_coda_source: none`

`ghost_of_elsinore` and `tempests_midnight_revelations` both carry
`spoken_coda_source = none`. The closing announcer says only *"We've lost our
signal. Until next time."* Operator: *"good but does not mention Shakespeare
at the end."*

**This looks like an over-application of the 2026-08-05 ruling.** That ruling
said licensed sources are CREDIT-ONLY -- the announcer names neither the
licence nor the licensor -- *"because Folger publishes the edition and
Shakespeare wrote the play."* The intent was to stop the announcer reciting a
LICENCE. What shipped stops it naming the AUTHOR and the WORK as well, which
is the opposite of what that sentence's reasoning implies.

Naming Shakespeare and the play is not a licence claim. **Needs an operator
ruling on the exact wording, then a fix.**

## 2. PUBLIC_DOMAIN says "public domain work" without naming the work

`midnights_ticktock` (bank `public_domain`, coda type `provenance`) closes on
the literal phrase *"public domain work."* Operator: *"We should mention name
of the work!"* The provenance receipt exists and the coda fires -- it is the
WORDING that is generic. The 65-unit corpus has titles and authors available.

## 3 and 4. VOICE GENDER CONTRADICTS THE CHARACTER (two instances)

* `midnights_ticktock`: **GERTRUDE DEMONGMORENCI MCFIGGIN** speaks with a male
  voice.
* `kindling_the_past`: **JULIANA SIMPSON** speaks with a male voice.

This is the known correctness defect, and it is explicitly CARVED OUT of the
2026-08-04 story-quality directive -- *"fixing 'Malvolio speaks with a woman's
voice' is a bug fix"*. Two live instances in one night on two different banks.

**Operator asked whether an LLM could WEB SEARCH to resolve a character's
gender. No -- and it is not needed.** Web search is barred by the hard scope
rule (100% local, offline-first, no cloud services, no API keys), and the
information is already local: Gertrude is named as a woman in the source text
the lane was handed. The fix is the existing `slot.gender` ladder
(`docs/2026-08-05-character-gender-ladder-SPEC.md`), which derives gender from
the source roster and pins it before casting. `slot.gender` already feeds the
description LLM, the outline prompt, the dialogue cast block and the image
prompt -- what it does not yet reliably do is pin the VOICE.

## 5. MEDIA_ARCHIVE closes with a news brief it never earned

`reel_of_mystery` (bank `media_archive`, coda type `news_close_brief`) closes
on *"Clarisse's gaze meets the reel's enigmatic label"* -- which is drama, not
a source note. Operator: *"What news story???"*

**The finding under the finding:** `media_archive` is an ARCHIVE bank -- its
sources are Library of Congress and Film Preservation blog posts -- yet its
coda type is `news_close_brief`, the shape built for the science-news lanes.
It is being asked to summarize a news story it never had. Either it gets an
archive-shaped close that names the post it adapted, or the news close is not
its owner.

## 6. Title/source mixup on the shakespeare lane -- UNRESOLVED

`tempests_midnight_revelations`. Operator: *"Is this Tempest or Macbeth?
mixup??"* Not yet diagnosed. The episode title is model-authored, so a title
naming the wrong play while the scene adapts another is a FIDELITY defect on a
lane where fidelity outranks arc. **Check the ledger's selected scene against
the title before assuming either way.**

## What is NOT a defect

`kindling_the_past` closes with no source, and the operator asked whether that
is acceptable. **It is.** That episode is bank `original`, whose coda is
correctly `none` -- an original story has no source to credit, and the
standing ruling is that original gets no catalog seeds at all. Working as
designed.

## Shape of the work

Findings 1, 2 and 5 are one family: **the closing announcer does not name what
it adapted**, differently broken on three banks. Findings 3 and 4 are one
defect: **voice gender is not pinned from the source**. Finding 6 is its own
thing and needs diagnosis before it is called anything.
