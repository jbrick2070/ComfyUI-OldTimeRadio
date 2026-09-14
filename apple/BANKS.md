# Source banks -- what kind of episode you get

`source_bank` is the single most consequential control on the writer node. It
decides what the episode IS: where the material comes from, how faithful the
script has to be to it, and what the show sounds like.

Everything else on that node adjusts an episode. This one picks the show.

You do not have to choose. The shipped graphs are set to **roll (any eligible
bank)**, which picks one for you each run -- which is the right setting for
"give me an episode" and the wrong one for "I want to hear the Shakespeare
lane", because a roll will land somewhere else four times out of five.

---

## The six

| Bank | What it makes | Where the material comes from |
|---|---|---|
| **Sci-Fi News Pro** `scifi_news_pro` | Science-fiction audio drama | A science news article, pulled live over RSS |
| **Shakespeare / Folger** `shakespeare` | Shakespeare radio adaptation | A scene from the Folger texts |
| **Public Domain** `public_domain` | Public-domain radio adaptation | A public-domain source text |
| **Media RSS / Archive** `media_archive` | Archive-*inspired* radio drama -- invented, not adapted | An archive / media-history item, pulled live over RSS |
| **My Story** `my_story` | A radio drama from your idea | The four **My Story** boxes on the writer node |
| **Original Radio Drama** `original` | Original radio drama | Nothing -- the writer invents it outright |

A seventh entry, **+ Add Your Own** `custom_source_bank`, is a signpost rather
than a bank. It is deliberately not runnable and never comes up on a roll;
selecting it does nothing. It is there to tell you the door exists --
[apple/EXTENDING.md](EXTENDING.md) is how you walk through it.

---

## The two that need something from you

**My Story** reads four boxes on the writer node -- `story_characters`,
`story_plot`, `story_setting`, `story_author`. Leave them empty and you get a
generic episode, because there is nothing to build from.

**Those four boxes are not ignored by the other banks -- they are refused.** If
you type into any of them while a different source is selected, the run stops
with a message naming the fields and telling you to either select My Story or
clear them. That is deliberate: the alternative is an episode that renders
happily, quietly ignores the characters you described, and looks correct. So
switching from My Story to another bank means clearing those boxes, not just
changing the source.

`custom_premise` is different and is worth knowing about: it is shared by EVERY
bank. Typing a premise on the Shakespeare or archive lane is ordinary, not a
mistake.

**The two RSS banks** -- Sci-Fi News Pro and Media RSS / Archive -- fetch their
material from the internet at run time. On a machine with no network they
cannot find a source to adapt. Shakespeare, Public Domain, My Story and
Original need nothing from the network.

---

## Two of them adapt. The rest invent. The difference is not cosmetic

This is the thing most worth understanding before you pick, because two banks
that both "use a source" treat it in opposite ways.

**Shakespeare and Public Domain are FIDELITY lanes.** Their instruction is to
*adapt the source while preserving its characters, turns, and ending*. The
source has already made the dramatic choices and the job is to put a microphone
on it.

**Media RSS / Archive is NOT.** Its instruction is to *build a fictional story
from* the archive material -- the item is a starting point, not a text to be
faithful to. If you pick this bank expecting a retelling of the archive item you
fed it, you will get an invented story that the item inspired. That is working
as designed; it is just not adaptation.

What follows applies to the two fidelity lanes.

What that means in practice, quoting the rules the packs actually carry:

> Where the source gives these characters words, CARRY THEM. Keep their diction,
> their rhythm, their argument. Compress and trim to fit the beat; do not
> paraphrase into a house style.

Where the source gives only narration or reported speech, it is turned into
something those characters would plausibly say at that moment -- **and nothing
beyond that**. No new protagonist, no bolted-on framing story, no changed
ending, and no imposing subtext on a scene that does not have any.

This also means the author's own language is carried as written. Wells' Editor
may shout *"Story be damned!"* because Wells wrote it. The packs used to forbid
their own sources' content and that was a fidelity defect, not a safety win; it
was removed deliberately.

**Original** is the opposite end and is meant to be: it adapts nothing and
invents the whole thing, with no premise scaffold at all -- pure radio drama.

---

## Picking one on purpose

In the writer node, set `source_bank` to the bank you want instead of leaving it
on **roll (any eligible bank)**.

Worth knowing if you are comparing episodes: a roll is not a fair comparison.
If you are judging a change -- a new voice, a different writer model, a prompt
edit -- pin the SAME bank on both runs, or you are hearing two different shows
and attributing the difference to your change.

## If an episode comes out wrong for the bank

Nearly always one of three things, in this order:

1. **You were on a roll and did not notice.** Check which bank the episode
   actually used; the ledger records it.
2. **My Story with empty boxes.** There was nothing to build from.
3. **An RSS bank with no network.** It could not fetch a source.

---

Adding a bank of your own is a real, supported path with its own acceptance
gates -- see [apple/EXTENDING.md](EXTENDING.md).
