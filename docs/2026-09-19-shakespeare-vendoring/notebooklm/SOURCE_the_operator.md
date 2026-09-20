# The operator: how one person runs a radio station staffed by machines

Source document for the NotebookLM set (segments 1-6 are the findings; the
schema is source 0; this is the person behind them). Written 2026-09-19 by
the coding window that worked the day, from the project's own standing
documents. Every claim cites a file in the repository; nothing here is
inferred from outside it.

## The one thing to take away

Old Time Radio is a one-person open-source project -- a ComfyUI node pack
that writes, casts, voices, scores, illustrates and publishes a radio drama
from a single click, entirely on a laptop, with no cloud and no paid
service [CLAUDE.md, "Scope Discipline"]. The person running it, Jeffrey
Brick, is a self-described vibe coder: *"I can look at colors and words on
a screen"* -- not a fluent reader of code, but a fluent reader of
architecture, tradeoffs and consequences [docs/OTR_STANDING_RULINGS.md, "HOW
TO TALK TO THE OPERATOR"]. The interesting fact for a listener is not that
the code is written by AI. It is that the *management* of the AI is done by
a person who cannot audit the code line by line, and who therefore built a
system of rulings, receipts and adversarial review that does the auditing
instead. The rulings file is 1,500 lines long. It is the real source code.

## How the project is actually run

**Directives, dated, in the operator's own words.** The project's rulebook
does not paraphrase. When a decision is made it is pasted in verbatim,
typos and all, with the date, and then restated plainly beneath it:
*"DONT ASK ME TO COMMIT ALWAYS COMMIT AND PUSH WHEN YOU MAKE CHANGES"*
(2026-09-17) [CLAUDE.md]; *"im noit pa perfetrcuioniust so any natiev even
tough the chaters may be mushged is beter tahn a ai atrasnation"*
(2026-09-19) [docs/OTR_STANDING_RULINGS.md]. The typos are kept on purpose:
the quote is the evidence that the ruling is the operator's and not the
machine's.

**The machines review each other, and the reviewer must be a stranger.**
Every code change is read by a model from a different family than the one
that wrote it, briefed to refute rather than to confirm, because *"a
self-review does not count"* and the two defects that prompted the rule
were invisible to their own author [CLAUDE.md, "ONE CLI REVIEW ON EVERY
CODING CHANGE", 2026-09-07]. On the day this set was written, eight lanes
ran in parallel -- four ChatGPT models, Grok and Composer in Cursor, two
Gemini models -- each on a distinct question, each producing a written
verdict that landed as a file in the repository so the next reader could
check it [docs/2026-09-19-shakespeare-vendoring/README.md].

**Two strikes, then the panel.** A bug that survives two fixes is not a
bug any more; it is evidence that the model of the problem is wrong, and
the third attempt goes to a review panel before a line is written
[CLAUDE.md, "TWO STRIKES, THEN THE PANEL", 2026-07-14].

**Match the review to the task.** The operator corrected the over-use of
the full review arc: *"we aren't running a full kibitz on everything, it
needs to choose the best path for the right task."* The test is one
question -- is there a design choice with more than one defensible
answer? -- and a deterministic fix gets one reader, not four rounds
[CLAUDE.md, "AMENDED 2026-08-17"].

**Never hold a push for a review.** Reviewers read what is already on
`main`; a finding becomes the next commit. This was written the day a
correct fix sat unpushed for a dozen exchanges waiting for a verdict that
never came [CLAUDE.md, 2026-09-19].

## What the operator values, measured by what was refused

**Seeing the episode.** The success signal is a file in a folder, not a
green log: *"if I see it in obs then it's somewhat a success"* and *"if I
don't see it in obs and it took more than 5 minutes, it's a fail."* When a
window tidied seventeen test episodes into a subfolder, the operator had
them restored within minutes -- they were the proof the path worked, and
*"for my dailies it keeps me going to see episodes"* [CLAUDE.md, "otr/obs/
IS THE SUCCESS SIGNAL", 2026-08-17].

**Done means done.** Story quality was declared finished on 2026-08-04
(*"It works. It works. I will publish it as open source"*), music on
2026-09-17, and the scanned-corpus work on 2026-09-19 (*"cut our losses
so we can ship"*). Each closure names what it does NOT close -- a
character speaking in the wrong voice is still a bug -- so the line between
correctness and chasing is drawn in writing, not by mood [CLAUDE.md;
docs/OTR_STANDING_RULINGS.md, top entry].

**The author's words over a cleaner machine.** Asked whether a damaged
century-old scan should lose to a fluent AI translation, the answer was
no: a mangled native text ships, and the gates refuse only a line put in
the wrong mouth [docs/OTR_STANDING_RULINGS.md, "A MANGLED NATIVE TEXT
BEATS A CLEAN AI TRANSLATION"]. The same instinct removed content filters
from the generation path when they were found instructing the model to
avoid "blood, guns, knives" while adapting Macbeth [CLAUDE.md, "NO
CONTENT GUARDRAILS", 2026-08-03].

**Honesty in the receipt.** A campaign one reviewer short is reported as
one reviewer short; a lane that quota-failed is replaced, not waited for;
a self-critique is not a contrarian [CLAUDE.md, "A MISSING REVIEWER NEVER
BLOCKS THE ARC"; "NO FIXED ROUND COUNT"]. The rulebook contains a section
written by a window about its own mistake (moving the obs episodes) and
another about a wrong context-percentage estimate that cost real work --
the project keeps its own errors on the record [docs/OTR_STANDING_RULINGS.md,
"DO NOT REPORT A CONTEXT PERCENTAGE"].

**Local, open, unpaid.** Sixteen gigabytes of laptop VRAM is the ceiling;
no API keys, no cloud writers to improve prose, no paid service adopted to
raise quality [CLAUDE.md, "Scope Discipline"; "STORY QUALITY IS DONE"].

## The day this set comes from

2026-09-19 began as a coding session on a scanned-volume corpus and ended
as a thesis. Over one day the operator: ran eight review lanes by hand,
pasting each verdict back; issued the fidelity ruling above; watched two
reviewers converge on a running head hiding inside a Prospero speech;
named the "in unison dilemma" -- a line Shakespeare gives to several
people at once, marked four different ways by four editions, which a
performance must give to exactly one voice; decided the project had
entered *"an English major meets computer science side-quest"* and turned
that into the goal: a NotebookLM deep-dive on proven oddities of
historical Shakespeare translations, *"doubling as my PhD thesis"*; and
then closed the lane to ship, with one final quest -- a native Japanese
episode, Tsubouchi's balcony scene, rendered through the real pipeline
[docs/OTR_STANDING_RULINGS.md, top entry; PROMPT9_six_lanes_notebooklm_segments.md].

## For the hosts: three hooks and two questions

* A person who cannot read code fluently wrote a 1,500-line rulebook that
  makes eight AI systems audit each other -- and pastes their own typos
  into it as evidence.
* The success metric for a machine-learning pipeline is "did a video file
  appear in this folder"; a window that tidied that folder was overruled
  in minutes.
* The project refuses a clean AI translation in favour of a damaged 1912
  scan, and refuses a filter that would have censored Macbeth.
* Open question: is a rulebook of dated, quoted decisions a better
  specification than code, for a person who reads consequences rather
  than syntax?
* Open question: what does "done" mean when the tool that made the thing
  could keep improving it indefinitely at zero marginal cost?

## Claims register

| claim | evidence | status |
|---|---|---|
| One-person, local, open-source, no paid services | CLAUDE.md "Scope Discipline" | RULING |
| Self-described vibe coder, reads architecture not syntax | docs/OTR_STANDING_RULINGS.md "HOW TO TALK TO THE OPERATOR" (2026-08-17) | RULING |
| Every code change reviewed by a different model family, briefed to refute | CLAUDE.md "ONE CLI REVIEW ON EVERY CODING CHANGE" (2026-09-07) | RULING |
| Two strikes then the panel | CLAUDE.md (2026-07-14) | RULING |
| Match the review to the task | CLAUDE.md "AMENDED 2026-08-17" | RULING |
| Never hold a push for a review | CLAUDE.md (2026-09-19) | RULING |
| obs folder is the success signal; 17 episodes restored | CLAUDE.md "otr/obs/ IS THE SUCCESS SIGNAL" (2026-08-17) | MEASURED (the incident) + RULING |
| Story quality closed 08-04, music 09-17, scan lane 09-19 | CLAUDE.md; docs/OTR_STANDING_RULINGS.md top entry | RULING |
| Mangled native text beats clean AI translation | docs/OTR_STANDING_RULINGS.md (2026-09-19) | RULING |
| Content filters removed from the generation path after censoring Macbeth | CLAUDE.md "NO CONTENT GUARDRAILS" (2026-08-03) | RULING |
| Eight lanes ran on 2026-09-19 with verdicts as files | docs/2026-09-19-shakespeare-vendoring/README.md and RESULT_*.md | MEASURED |
| The in-unison dilemma named by the operator | notebooklm/SCHEMA_speaker_cue_oddities.md; PROMPT9 (Segments 2 and 4) | RULING |
| Final quest: a native Japanese Shakespeare episode | docs/OTR_STANDING_RULINGS.md top entry (2026-09-19 evening) | RULING |

SOURCES READ: 4 (CLAUDE.md, docs/OTR_STANDING_RULINGS.md, docs/2026-09-19-shakespeare-vendoring/README.md, docs/GO_FORWARD_PLAN.md)
