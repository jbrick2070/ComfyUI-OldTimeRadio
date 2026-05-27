# Good Story Writer Architecture — Problem Statement

**Date:** 2026-05-26 (post-Sprint-10A code-completeness)
**Author:** Jeffrey A. Brick + Claude (drafting)
**Sequel to:** `docs/2026-05-26-good-story-writer-architecture__00_question.md`
**Status:** problem-statement only; no implementation decisions yet.

---

## TL;DR

Sprint 10A made the diagnostics work. It did not make the storytelling
work. The post-Sprint-10A pipeline ships audio + video reliably and the
new whole-episode critic correctly flags weak episodes as weak. But
"correctly identified as weak" is not "good story." The story-quality
gap remaining is the actual product gap.

Jeffrey's hypothesis: the current writer pipeline is too structured.
Every LLM call is a one-shot structured-JSON request -- the model is
asked to emit a finished object in a single inference, with no
back-and-forth, no "tell me more," no "that's not quite right -- try
again with this concern in mind." A chat-style multi-turn conversation
with the LLM (Sprint 10A step 6's roleplay multi-turn pattern, but
extended to the entire story-construction process) could produce
substantially better stories.

This document is a problem statement, not a proposal. It is the
honest list of what we know is broken and the design space worth
sketching before any code lands.

---

## 1. Evidence: what the pipeline produces today

### 1.1 Concrete example -- "Spray of Hope" 2026-05-26 16:23

The post-Sprint-10A-code-complete soak run
`signal_lost_spray_of_hope_20260526_162345` shipped the following
script end-to-end. The news seed was *"Scientists say they've reversed
brain aging with a simple nasal spray"* (ScienceDaily,
2026-05-26).

```
ANNOUNCER (16w):  Good evening. This is SIGNAL LOST. Tonight, a signal
                  breaks through the static. Stay with us.
REN BLACK (2w):   Code Red.
REN BLACK (10w):  Where's that damn override? It's got to be here somewhere...
ANNOUNCER (10w):  Breathless, Ren Black mutters, "That's it. Time's not
                  up yet."
ANNOUNCER (14w):  This has been SIGNAL LOST. The report ends, but the
                  signal remains. Good night.
```

The Stage 7 shadow critic (Sprint 10A step 7) verdict on this same
episode, scored against `docs/2026-05-26-sprint-10a-whole-episode-
critic-rubric.md`:

```
verdict       discard
mean_score    2.60 / 5.0
failing axes  premise_clarity, continuity, pacing, emotional_arc,
              resolution, __mean_below_threshold__
```

The critic and a human reader (Claude, in real time on 2026-05-26)
agree this is a weak episode. Pulling apart *why* it's weak:

* **The news premise never appears.** The brain-aging-nasal-spray
  story is not in the script. The title "Spray of Hope" is the only
  trace. Mistral-Nemo invented a different scenario (override panic
  in some unnamed setting) and ignored the seed.
* **No setup.** "Code Red" is a punchline. We have no idea who Ren
  Black is, where we are, or what's threatening.
* **No middle.** "Where's that damn override?" -- override of what?
  No stakes, no question, no opponent.
* **Speaker-leak in beat 4.** The announcer narrates Ren's action
  with embedded dialogue: `Breathless, Ren Black mutters, "That's
  it..."`. This is exactly the prose-leakage shape Sprint 10A step
  5's `validate_speaker_leak` was designed to catch, but Stage 3
  validators run only in the shadow pass today, not in the
  production writer.
* **No resolution.** "Time's not up yet" is a tease, then the
  announcer wraps. Nothing was set up; nothing resolves.
* **Minor PD4 friction.** "damn" tripped no SFW filter but the
  project's safe-for-work rule (`CLAUDE.md` Prime Directive 4)
  should disprefer it.

### 1.2 This is not an outlier

Across the 2026-05-26 operator soak (10+ runs, 4 of which I have
audited line-by-line):

* `pending_20260526_154105` (Dr. Anya, 110-word budget, 2 chars,
  Himalayan pit viper news seed): the news anchor *almost* showed
  up (bioluminescence got mentioned), but the dialogue was thin and
  the legacy story critic flagged 1 line flat in both reroll
  cycles. Reroll loop exhausted -> `needs_full_rerun`.
* `pending_20260526_161845` (REN BLACK, 60-word, 1 char, brain-
  aging news seed): same as 162345 above -- premise invisible,
  3 character lines at 2/8/10 words respectively.
* `pending_20260526_160945` (RAINN CARRUTHERS, NASA Artemis news
  seed): the LLM cast Rainn as gender=female (BUG-LOCAL-279
  family) and the news_interpreter exhausted with `V1: key_term
  'SLS rocket' not in source`. The episode shipped with no key-
  term enforcement, dialogue drifted from the news.

The shape repeats: **the writer produces something that runs end-
to-end but that doesn't read like a coherent story.**

### 1.3 What the diagnostics tell us

This is the new and useful signal:

* **Stage 1 grammar-constrained planner** (step 3): 4/4 valid plans
  on first attempt -- the JSON shape is right, the cast list is
  right, the beat structure is right.
* **Stage 1 cast audit** (step 4): clean 0/0 on the REN BLACK run.
  Names and genders align with the curated pool.
* **Stage 7 whole-episode critic** (step 7): scored the actual
  prose. Verdict matches human read.

So we *can* generate a clean plan, *can* cast valid characters,
*can* score the prose against a rubric -- and the prose is still
weak. The bottleneck is between the plan and the prose.

---

## 2. Mechanism: why is the prose weak?

Three mechanisms compound. None of them is "the LLM is bad."

### 2.1 Mechanism A -- one-shot structured generation

The current writer architecture, for every meaningful pass:

```
SYSTEM:  here's a strict JSON schema. fill it.
USER:    here are some constraints (cast, beat hint, target words).
LLM:     {emits one object, hopes for the best}
```

There is no:
* "your first answer doesn't connect to the news seed -- try again
  with the news in mind"
* "this dialogue line doesn't reveal character -- give me one that
  has a concrete object, a specific verb, and a stake"
* "the previous line said X; now Y must escalate, not restate"

Every refinement that *would* lift quality has to be encoded as
*another schema field with another one-shot call*, or as a
deterministic regex / validator after the call. The model can't
reason iteratively about its own output.

Sprint 10A step 6 (multi-turn roleplay for dialogue composition) is
the only place we already do conversational writing -- and even
there it's 4 fixed turns, not free-form negotiation.

### 2.2 Mechanism B -- the prompts are anchored to structure, not story

Read any of the production prompts (`OTR_LedgerScriptWriter.py`,
`_otr_line_composer.py`, `_otr_outline.py`). They are *exquisite*
at telling the model:

* what shape to emit
* which speaker label to use
* what NOT to do (no music tags, no SFX cues, no parens, no stage
  directions)
* how many words to aim for

They are weak at telling the model:

* what makes this episode different from yesterday's episode
* what the dramatic question is
* whose desire is opposed to whose
* what changes between the open and the close
* what the audience should feel at minute 5 vs minute 30

The model is being asked to perform stage management with no
director's note.

### 2.3 Mechanism C -- the news_interpreter is the only thing
anchoring the LLM to the seed, and it brittles out often

`build_news_briefs` is a structured pass that distills the article
into `key_terms` + `script_brief` + `news_close_brief`. When it
works, downstream passes can be checked for "did the script mention
this key_term?" and the LLM has a 1-2 sentence executive summary
of what the episode is *about*.

When it fails (BUG-LOCAL-264, recurring), the writer falls back to
"raw news_seed for cast + outline (no key_terms enforcement)."
**Without key_terms, there is no contract between the story and
the news.** The model improvises whatever genre fits the surface
texture of the news headline. Brain-aging nasal spray becomes "noir
control-room panic" because nothing prevents that drift.

`news_interpreter` failing is the single highest-impact way to lose
story-news connection. And it fails on every article whose headline
contains a proper noun the article body doesn't repeat verbatim
("Texas A&M" in the brain-aging article, "SLS rocket" in the
Artemis article, etc.) -- a very common shape.

---

## 3. Jeffrey's proposal: chat-style conversation with the LLM

**Original statement, 2026-05-26:**

> "I think if the system had more of a chat-style convo with the LLM
> it could get a better story, but I need a good story."

Read literally and generously, this is the proposal:

> Stop asking the LLM to emit finished objects in a single
> inference. Instead, *talk* to the LLM about the story as it gets
> written. Multi-turn. Conversational. The model gets to ask
> clarifying questions and the system gets to push back when the
> model drifts. The script is the *transcript* of a writers'-room
> conversation, not the result of stamping out cells in a JSON
> schema.

This is consistent with Sprint 10A step 6's multi-turn roleplay
pattern, generalized from "compose one dialogue line" to "construct
the whole episode."

### 3.1 Why this is structurally promising

* **It addresses Mechanism A directly.** A conversational loop can
  do iterative refinement that one-shot generation cannot. "That
  doesn't connect to the news -- try again." "Now make Ren's line
  earn that override." "The middle beat repeats the open; make it
  escalate." These are normal writers'-room moves that map cleanly
  to chat turns.

* **It addresses Mechanism B by changing what the prompt is for.**
  In conversational writing, the system prompt becomes a director-
  brief: "here's the news, here's the cast, here's what we want
  the audience to feel -- now we're going to walk through this
  episode beat by beat." The structure constraints live in
  per-turn user messages, not in a giant schema the model has to
  conform to in one go.

* **It composes with the rubric we already have.** Stage 7 critic
  scores prose against axes (premise_clarity, continuity, pacing,
  emotional_arc, resolution). In a chat-style writer, each axis
  becomes a *checkpoint conversation* between Stage N writer
  passes -- not just a final scorecard. The critic can talk back
  to the writer mid-episode.

* **It scales down naturally.** Today's smoke runs (60-word
  episodes) are pathological for any structured pipeline because
  there's no room for arcs. But a chat-style writer can decide
  on its own that "this budget is too small for an arc; produce
  a vignette, not a 3-act structure, and score it against
  vignette-appropriate axes." Structured pipelines can't make
  that decision; chat-style ones can.

### 3.2 Where the proposal needs refinement

* **Cost.** A multi-turn conversation per episode is N x the
  inference cost of a one-shot pipeline. Mistral-Nemo at ~3-5
  seconds per turn on the RTX 5080 means a 20-turn writer's-room
  session costs ~60-100 seconds in pure LLM time -- not VRAM-
  unaffordable but real.

* **Determinism.** Today's pipeline is reproducible-by-seed (per
  BUG-LOCAL-270 randomization rules). Multi-turn conversations
  are harder to make reproducible without explicit state
  capture. The PD7 byte-identity contract (`tests/
  test_audio_byte_identical.py`) presumes reproducibility on
  C7 seed override. That has to be preserved or explicitly
  relaxed.

* **What's the "system" in the conversation?** Today every
  structured pass has the writer node as the orchestrator and
  the LLM as the worker. Chat-style writing is two-sided -- the
  orchestrator now has to *be* a persona too. Director? Editor?
  Other LLM acting as critic? This is the Candidate B
  "per-character actor agents" architecture from the previous
  design doc (`__00_question.md`), generalized.

* **Failure modes shift.** A one-shot pass either parses or
  doesn't; the failure surface is well-defined. A multi-turn
  conversation can drift, loop, lose context, or get into
  unrecoverable states. Sprint 10A step 6's 4-turn pattern is
  small enough to control; an open-ended conversation isn't.

* **The current pipeline is shipping audio.** Whatever replaces
  it must preserve PD1. A chat-style writer is a large change;
  it would have to ship behind a widget toggle (`use_chat_style_
  writer = False` default), with the structured writer staying
  as the fallback for at least 1-2 sprints.

---

## 4. Adjacent factors that hurt story quality independently

Even if chat-style writing were free, the following issues would
still degrade story quality. They should be on the same plan.

### 4.1 news_interpreter is the load-bearing news anchor and it's fragile

BUG-LOCAL-264 (logged 2026-05-24, recurring). Schema enforcement
on key_terms rejects values that don't appear *verbatim* in the
article body. Articles routinely paraphrase ("NASA" in the body
when the headline says "NASA's Psyche"; "lunar landing" never
literally appears in an article about a lunar landing). The
3-attempt retry ladder retries the same prompt at lower
temperature -- no semantic flexibility -- and exhausts on every
hard article.

**Fix-level work**:
* Switch `key_terms` validation from exact-substring to
  semantic-presence (e.g. each key_term must be derivable from
  the source by a stricter LLM-as-judge check, or
  TF-IDF-style overlap with the body).
* Or: make `key_terms` optional and use the `script_brief`
  alone as the anchor (script_brief is freer-form and rarely
  fails validation).

### 4.2 Stage 3 validators are shadow-only

Sprint 10A step 5 produces `validate_speaker_leak`,
`validate_banned_phrases`, `validate_length`,
`validate_pronoun_consistency`, etc. These run in the shadow
pass on the Stage 1 plan. They DO NOT run in the production
writer's per-line composer. Beat 4 of "Spray of Hope"
(`Breathless, Ren Black mutters, "..."`) is the exact failure
mode `validate_speaker_leak` would catch -- but the validator
never sees production output.

**Fix-level work**: wire Stage 3 validators into
`_otr_line_composer.compose_line` as a post-compose filter.
Lines that fail go back to the model for one repair attempt
before the line is committed.

### 4.3 The writer prompts are structure-heavy, story-light

See Mechanism B above. The system prompts can be edited to
include more dramatic-craft framing without changing the
schema-driven loop. Cheap experiment: add 2-3 sentences to
`_STAGE1_SYSTEM_PROMPT` and the outline prompt about WHAT a
good radio drama beat does (dramatize a *decision*, not a
*setting*; show *opposition*; advance *desire*) and run a
listen-test soak.

### 4.4 The smoke-budget regime distorts every observation

60-word episodes can't have arcs. Every quality check on a
60-word run is partly measuring "is the model capable of
compressing a 3-act structure into 60 seconds of audio" --
which it isn't, full stop. Production-quality assessment
needs production-budget runs.

The Sprint 10A operator-decided next step (full-budget episode
on HEAD for the listen-test verdict) is the right call -- the
60-word soaks confirmed the pipeline is wired correctly; the
real story-quality question is open until production budget
runs.

### 4.5 The Stage 7 critic is shadow-only

Today the critic scores; nothing acts on the score. The verdict
sits on `meta.stage7_shadow_critic`, the operator reads it
later, but the writer doesn't see it. A chat-style writer could
have the critic in the conversation: "Editor sees the draft.
Editor's verdict: discard. Reason: premise_clarity. Writer,
revise."

### 4.6 The legacy reroll loop chases the wrong target

Sprint 5C's reroll loop targets *individual lines* that the
critic flagged as "flat." On a 60-word episode that's the wrong
unit -- the problem is structural (no arc), not local (this
one line is dull). Rerolling a flat line just produces another
flat line because the context didn't change. The reroll loop
exhausts and stamps `needs_full_rerun`. BUG-LOCAL-279 Option A
(landed 2026-05-26 `c23bd6e`) makes the length-band more
forgiving so the reroll loop converges more often, but it
doesn't fix the underlying mismatch between "what the critic
flags" and "what reroll can change."

A chat-style writer could escalate from line-reroll to
beat-reroll to episode-replan based on what the critic says is
broken.

---

## 5. The product requirement

Jeffrey, 2026-05-26: **"I need a good story."**

What "good" means here, unpacked:

* The episode reads like it came from the news. The audience can
  tell what the episode is *about* by minute 1.
* The episode has dramatic shape -- something is at stake,
  something is in opposition, something changes.
* The characters have specificity. Ren Black is not the same
  character as Lawrence Vaughn is not the same character as
  Alice. Their lines are not interchangeable.
* The dialogue advances a thread. The middle beat earns the
  ending, the ending earns the open.
* No structural leaks (no announcer narrating character
  dialogue; no character monologuing announcer copy; no
  stage-direction prose where dialogue should be).
* Safe for work, non-violent, good narrative arc -- per
  `CLAUDE.md` Prime Directive 4.
* Reproducible if the operator pins the seed; randomized if not
  (per BUG-LOCAL-269/270 rules).
* Within current VRAM ceiling (14.5 GB), current production
  budget (~300-800 words / 5-10 minutes of audio), current
  hardware (RTX 5080, no cloud).

"The diagnostics say this is bad" is not "this is good." The
product target is the latter.

---

## 6. Open design questions before any implementation

These are the load-bearing decisions that need explicit answers
before code lands on any of this.

1. **Scope of the chat-style change.** Does the chat conversation
   replace the *whole* writer (premise -> cast -> outline -> beats
   -> dialogue -> polish)? Or just the dialogue pass? Or just the
   beat-construction pass? Each scope choice has different
   ergonomics, cost, and migration story.

2. **Who is the "system" in the conversation?** Single LLM with a
   director-persona system prompt? Two LLMs (writer + editor)?
   Per-character actor agents (Candidate B from `__00_question.md`)?
   The agent topology decision shapes everything else.

3. **Where does the news-grounding contract live?** If
   news_interpreter is too brittle to keep, what replaces it?
   Examples: LLM-as-judge for key-term presence; cosine-similarity
   embeddings between script and article; "tell me what news this
   episode is about" pass after the script and score the answer.

4. **How does Stage 7 critic enter the conversation?** As a turn
   in the writer loop? As a separate editor agent? As a hard gate
   that won't let a draft pass without a passing score?

5. **What's the minimum-viable demo?** "Generate one good 5-minute
   episode about a science news story" is a clear product target.
   Is the demo against a fixed news seed (for A/B) or random
   (for diversity)? How do we measure "good"? Listen-test verdict?
   Critic mean_score >= 4.0? Both?

6. **Reproducibility contract.** What stays reproducible-by-seed
   under chat-style writing? The plan? The cast? The transcript?
   Nothing? PD7 (byte-identical audio on C7) currently assumes the
   whole writer is deterministic.

7. **Backward compatibility.** Does Sprint 10A's shadow critic +
   Stage 1 grammar planner stay in the pipeline as the new writer's
   pre-flight check, or do they retire? They cost ~60s per episode
   of LLM time but produce real signal.

---

## 7. Suggested next step

A single 90-minute Claude design session, working from this
document, that produces:

1. An agent topology diagram (one-LLM director, two-LLM
   writer+editor, or multi-LLM actor agents). One picked, others
   rejected with one-line "why not."

2. A minimum-viable conversation script -- the literal turn-by-
   turn dialogue between the system and the LLM(s) for one
   episode, from news seed to final script. Hand-traced, no
   code yet.

3. A demo plan: produce 3 episodes from 3 different news seeds
   with the new writer, all on a fixed seed pair, listen-test
   verdict from Jeffrey within a week.

4. A migration path: how the chat-style writer ships behind a
   widget (`use_chat_style_writer`, default False), what the
   rollback plan is, what tests pin both modes.

That session output goes into
`docs/2026-05-26-good-story-writer-architecture__02_design.md`.
No code lands until that document is signed off.

---

## 8. What is NOT being asked here

* Not a critique of Sprint 10A. Sprint 10A shipped on plan and
  the diagnostics are now load-bearing. Nothing in this doc
  argues against keeping Sprint 10A's machinery.
* Not a critique of Mistral-Nemo. The model is doing its job
  inside the structured-call framework it was given. The
  framework is the question, not the model.
* Not a request for cloud APIs, paid services, or a different
  model. Per `CLAUDE.md`, 100% local / open source / offline.
* Not a budget request. The chat-style writer is bigger than
  this conversation can scope; the next step is design, not
  build.

---

**End of problem statement.** The design document
(`__02_design.md`) is the next artifact; not in this commit.
