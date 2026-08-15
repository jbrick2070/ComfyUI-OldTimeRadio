# Placed interrupts -- problem statement

**Operator-requested, 2026-08-14. A SMALL improvement, scoped to the banks it
belongs on.** Not started. This exists so the next window can decide in five
minutes instead of re-deriving it.

## The question

Today's `scifi_news` episode is the best story in the corpus, and its remaining
weakness is uniform: **everyone speaks in complete, polished paragraphs.**
Nobody interrupts, nobody is cut off, nobody is inarticulate. It is a good
radio play performed by three orators. The operator asked whether local models
can place interruptions, and whether it is worth an LLM pass.

## What an "interrupt" actually is in this system

The ledger holds ONE ROW PER BEAT, one speaker per row, and beats never cross
speakers -- that is the HuMo clip-fill rule, and it is load-bearing for audio
windows. Every spoken row becomes ONE TTS CLIP.

So an interruption is two consecutive rows, different speakers, where the first
ends mid-thought and the second cuts in. **No schema change is needed.** The
structure already supports it.

## THE BLOCKER IS THE TIMELINE, NOT THE MODEL

`nodes/scene_sequencer.py`:

```
breath_ms = 200          # between every dialogue line
silence_pad = 80ms       # per-clip padding
```

**A 200 ms breath between every line means a written interruption renders as a
polite pause.** You can write a perfect cut-off and hear a normal turn. Any
work that improves the WRITING without touching this is wasted -- the audio
neutralises it.

That is the whole finding. The models are not the constraint.

## DO NOT USE AN EM-DASH. WRITE NOTHING AT ALL.

The obvious convention is to end the interrupted line on an em-dash. **Do not.**
Operator objection, and it is correct: the TTS may PRONOUNCE it.

Verified 2026-08-14 -- `clean_spoken_text` (`nodes/_otr_script_prep.py`) is the
canonical spoken surface, and it passes every dash through untouched:

```
'But the readings are—'   -> 'But the readings are—'
'But the readings are--'  -> 'But the readings are--'
'But the readings are-'   -> 'But the readings are-'
'But the readings are…'   -> 'But the readings are…'
```

So the character reaches the engine verbatim and the engine decides what it
means. That is an unknown that could read as "dash", as an artifact, or as
nothing.

**AND A SMART ENGINE DOES NOT SAVE IT -- IT MAKES IT WORSE.** The operator's
instinct is that modern engines handle punctuation well, and that is probably
true. But a well-behaved TTS renders an em-dash as a **PAUSE**, which is the
CORRECT reading of the symbol and precisely the wrong sound for an interrupt.
The whole point is a hard cut. So the good engines fail this too -- not by
mispronouncing, but by doing the right thing with the wrong signal. There is no
engine quality that turns an em-dash into an interruption.

**AND THERE ARE EIGHT OF THEM.** `bark`, `chatterbox`, `kokoro`, `dia`,
`indextts2`, `google_tts`, `cloud_elevenlabs`, `cloud_sonilo`. Operator, and
this settles it: *"we have so many TTS we can't make all of them happy."* Any
design that depends on how one engine interprets a symbol is a design that
breaks on the other seven, and on the ninth when it arrives.

**THE MECHANISM IS THE ABSENCE, NOT A SYMBOL.** End the interrupted line
mid-clause with NO terminal punctuation at all:

```
row N     "But the readings are"          <- ends mid-clause, no punctuation
row N+1   "They're wrong. I checked."     <- compose_flag: interrupts_previous
```

Nothing can be mispronounced because nothing is there. The abruptness comes
from two places that are both under our control: the sentence stops mid-clause,
and the sequencer gives it zero gap. This is also engine-independent -- it
relies on the absence of a character rather than on Bark, Kokoro or whatever
comes next interpreting a symbol the same way.

`_spoken_text_finding` imposes no terminal-punctuation requirement, so an
unpunctuated line is already legal today.

## Can local models place them? Yes, and cheaply

Stopping a sentence mid-clause is not a reasoning task. A 12B handles it
easily; even the 2B would. The instruction is "end this line unfinished", which
is simpler than asking for a specific punctuation mark.

**And the per-beat schedule shipped 2026-08-14 is already the right place for
it.** `_beat_dialogue_inputs` hands each beat job `rows_so_far`, so the writer
already sees the line it would be cutting into. Telling one beat "the previous
line ends mid-thought, cut in on it" needs no new context.

## DO NOT ADD ANOTHER PASS

Two reasons, and the second is the real one:

1. There is already a per-scene review job. A third opinion about dialogue
   means two passes with overlapping jobs, against the standing "one prompt
   per job" law.
2. **A new pass does not move the 200 ms breath**, so it buys nothing audible.

## The shape, in three steps

| # | change | size |
|---|---|---|
| 1 | The SCORE marks a beat as interrupting the previous one. It already plans beats, speakers and intents. | small |
| 2 | The per-beat job receives that in its existing window. No new pass, no new prompt text -- the JOB varies, not the prompt. | small |
| 3 | The row carries a `compose_flag` (e.g. `interrupts_previous`) and the sequencer drops the breath to ~0 for that row. | **the load-bearing one** |

**Step 3 must be a ROW FLAG, not a text tag.** There is precedent for `[BEAT]`
/ `[PAUSE]` tags driving timing, but the ledger law is that spoken text holds
pure speech -- a timing signal belongs BESIDE the text, never inside it, or it
becomes an F1 defect and TTS may read it aloud.

## WHICH BANKS -- and this is the part that matters

**INVENTION LANES -- yes.** The story is ours, so how characters speak is ours:

- `original` -- invented outright
- `scifi_news`, `scifi_news_pro` -- springboard fiction; the operator ruled the
  drama may depart freely from the real story
- `media_archive` -- invented drama around archive material

**FIDELITY LANES -- NO. Excluded on principle, not on effort:**

- `shakespeare`
- `public_domain`

The standing ruling on those two is that **fidelity outranks arc**: *"Put a
MICROPHONE on the scene; do not re-plot it"*, *"Compression is allowed;
replacement is not."* An interruption Shakespeare did not write is REPLACEMENT
-- new dramatic beats invented on top of the source. It is the same defect
class as the old violence clause that was instructing the model to avoid the
author's own content: a well-meant improvement that damages the thing the lane
exists to preserve.

If a source ALREADY contains an interruption, the passage selector should carry
it as written. That is fidelity, and it needs no feature.

## What would prove it

One live episode per invention lane, graded with
`scripts/otr_ledger_view.py`, plus **listening to the rendered audio** -- this
is the one change whose success cannot be read off the ledger. The ledger will
look identical either way; the question is whether the cut sounds like an
interruption or a pause.

Acceptance: F1 and F2 stay at zero (an em-dash is not stage business, but
confirm the detector agrees), and the interrupted line audibly cuts.

## Risks

- **TTS may not honour the em-dash.** Bark/Kokoro might pause on it rather than
  clipping. Unknown until heard. If it pauses, step 3 has to trim the tail of
  the clip instead, which is more work -- decide then, not now.
- **Overuse.** A play where everyone interrupts is as monotonous as one where
  nobody does. The score should mark FEW beats, not most.
- **Do not chase this into a multi-day pass.** The operator's line: *"we're not
  chasing better output for 2 days like we used to."*

## Sequencing

**Behind the clean stage.** F1 (action in a spoken row) is still a live defect
on five banks at 11-40% of rows; this is polish on a lane that already grades
clean. It does not jump the queue ahead of something that is broken.
