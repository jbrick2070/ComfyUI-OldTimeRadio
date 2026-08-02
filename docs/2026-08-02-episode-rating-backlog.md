# Episode rating: an ORIGINAL open rubric, on the opening and closing credits

**Operator ask, 2026-08-02 (backlog, "at some point"):** an episode rater that
assigns a rating shown in the starting AND ending credits, "based on the same
principles as G / PG / R and such, but NOT copied -- open-source methodology."

Captured now while the reasoning is fresh. NOT scheduled: it sits behind local
6/6, the cloud grounding, and the all-cloud campaign.

## THE CONSTRAINT THAT SHAPES THE WHOLE DESIGN

"Same principles, not copied" is the load-bearing sentence. The letter marks
themselves (G, PG, PG-13, R, NC-17) are administered trademarks of a specific
ratings board, and an unaffiliated product stamping them on its own output
implies a certification nobody granted. So:

* **do not emit those letters**, and do not emit a near-miss lookalike whose
  whole purpose is to be mistaken for them;
* **do** take the underlying PRINCIPLE, which is not proprietary and is the
  actually useful part: name the content dimensions a listener may want warning
  about, band each one, and publish the rubric so the reader can check the
  call themselves.

An open methodology is genuinely BETTER here than a borrowed letter: a letter
compresses several dimensions into one opaque token, and an open rubric can say
*which* dimension earned the band. "Mild peril" tells a parent more than "PG".

## WHAT THIS PIPELINE ACTUALLY VARIES OVER

**CORRECTED 2026-08-02, same day.** This section originally argued the rater
was near-pointless because `CLAUDE.md` bound every episode to "Safe for work.
Non-violent." The operator removed that line as a remnant -- it described a
constraint that never actually bit, since nothing in the pipeline was producing
violent content anyway. So the premise is gone and the argument built on it
does not hold: there is no standing rule flattening every episode into the
mildest band.

What DOES vary, and is worth signalling, is TONE and INTENSITY:
suspense, dread, peril, loss and grief, moral weight, supernatural or body
strangeness, loudness/startle in the audio mix. The fear-cape inversion exists
precisely because some beats are meant to unsettle.

So the honest product is closer to **viewer guidance** than to a gate -- one
episode may be markedly more unsettling than another, and the credits are a
reasonable place to say so. That conclusion survives the correction above: it
rests on what this show IS, not on a rule that no longer exists.

## SHAPE (for the panel to break later, not settled)

* **Dimensions, each banded 0-3**, deterministic from the frozen ledger where
  possible and LLM-judged only where it cannot be: peril/threat, dread/
  suspense, loss/grief, strangeness, startle (measurable from the audio mix).
* **One overall band derived from the dimensions by a published rule** (e.g.
  the maximum, not an average -- an average hides a single intense dimension).
* **An original vocabulary**, e.g. `ALL AGES` / `SOME SUSPENSE` /
  `STRONG SUSPENSE`, chosen to be plainly not a borrowed letter mark.
* **The rubric ships with the repo** and the receipt names which dimension set
  the band, so a rating is auditable rather than an oracle.

## WIRING NOTES (from what this repo already knows)

* **The ledger must carry it, and exactly one owner writes it.** Credits read
  FIELDS, not intentions (`CLAUDE.md` ledger-completeness rule). If the rating
  reaches the opening credits it must be stamped BEFORE the credits node runs,
  and stamped once.
* **Opening AND closing credits is two consumers of one field**, so the value
  must be frozen with the ledger -- not recomputed per credit roll, or the two
  can disagree within a single episode.
* **It must not become a silent gate.** A rating is a LABEL. If a future
  version refuses to publish above a band, that is a separate, explicit
  decision -- exactly the "advisory that quietly became fatal" shape that bit
  the codex lane's cast-coverage field.
* **Determinism**: the same frozen ledger must rate identically on a re-run, or
  the receipt is worthless. Any LLM-judged dimension needs the same
  seed/temperature discipline the other passes use, plus a deterministic
  fallback.

## PRIOR ART -- the shape to borrow (operator asked 2026-08-02)

"Isn't there a public open source rating system people can use that's
self-governed?" Close, with one consistent catch: **the METHODOLOGIES are
public; the MARKS are controlled.**

* **IARC** (International Age Rating Coalition) -- exactly the self-governed
  model: the creator answers a standard questionnaire and ratings are computed
  automatically for several regions. Administered by a coalition, and the marks
  are theirs.
* **Kijkwijzer** (NL) -- questionnaire-driven self-classification with a
  documented, research-based methodology. The best published rubric to study.
* **RTA** -- genuinely free to self-apply, but a single binary adults-only
  flag, not a scale.
* **PEGI / ESRB** -- published criteria and self-declaration at the lower
  tiers, controlled marks.

So the pattern to adopt is theirs and the vocabulary must be ours:
**descriptors -> computed band**, which is the shape this document already
proposes. IARC and Kijkwijzer are the precedent that it is a legitimate,
well-tested structure rather than something invented here.

VERIFY LICENSING before adopting any of them directly -- this was written from
model knowledge with a cutoff, and these bodies revise their terms.

## OPEN QUESTION FOR THE OPERATOR (when this is scheduled)

Is the rating meant to be **informative**
(tone/intensity guidance, which is what this document assumes) or
**gatekeeping** (a band that could block publication)? The two produce very
different designs, and the second one needs a policy decision that is yours,
not mine.

---

# THE PANEL QUESTION (operator, 2026-08-02): which methodology, given a LOCAL LLM judges it?

That constraint is not a footnote -- it should drive the design. The judge is
`gemma-4-12b` at Q4_K_M on a 16 GB laptop, the same model that writes the story.
Everything below follows from what that judge can and cannot be trusted to do.

## WHAT THIS REPO HAS ALREADY LEARNED ABOUT THIS JUDGE (evidence, not opinion)

Hard-won today and yesterday, all from live legs:

* **It drifts from format under load.** It restated speakers with their cast
  role ("Commander Vance (Space Force Tactician)") and broke the markup parser
  for an entire leg. A rater whose output shape is free-form WILL drift.
* **It writes stage direction where dialogue was asked for** -- five lines of
  "(static crackles)" -- and every raw-text check passed it. A rubric asking
  "is this scary?" invites exactly that class of non-answer.
* **It has a hard output ceiling.** `_P0_BASE_OUTPUT_TOKENS = 2800` against a
  context the row now allows at 8192. A rater competing for that budget with
  the writer is a rater that gets truncated first.
* **Its passes already carry a repair ladder** (4-5 attempts) and a
  deterministic collapse guard, because a bare LLM answer is not trusted
  anywhere else in this pipeline. A rater should not be the exception.
* **Determinism is a shipped requirement**: cast/style seeds are pinned under
  `OTR_C7` for byte-identity replay. A rater that answers differently on a
  re-run of the SAME frozen ledger breaks that.

## THE CANDIDATE METHODOLOGIES, and why the choice is not obvious

1. **Holistic judgement.** "Read the episode, return a band." One call, cheap.
   Maximum drift, no auditability, and the failure mode is a confident wrong
   answer with no way to see which part drove it.
2. **IARC/Kijkwijzer-style QUESTIONNAIRE.** Many small CLOSED questions
   ("does a character die on-page? yes/no"), then a PUBLISHED FORMULA computes
   the band. The LLM never chooses the rating -- it only answers observations,
   which is the task a small local model is actually good at.
3. **Rubric-anchored scoring.** Per-dimension 0-3 with worked examples for each
   level. Middle ground; richer than yes/no but reintroduces judgement, and
   anchor examples cost context the writer needs.
4. **Deterministic-first, LLM-last.** Compute everything measurable from the
   frozen ledger and the audio WITHOUT the model (startle from the mix,
   death/peril keywords, act structure), and ask the LLM only for what
   genuinely needs reading comprehension.

The driver's own lean is 2+4: closed questions the model answers, an open
formula the REPO owns, and no dimension asked of the LLM that arithmetic can
answer. But that is a hypothesis, and the panel's job is to break it.

## WHAT THE PANEL MUST ANSWER

1. **Which methodology, specifically, for THIS judge?** Not which is best in
   general -- which survives a 12B local model that has already been observed
   drifting from format and substituting stage direction for content.
2. **Closed questions vs scored dimensions.** Is "did a named character die?
   yes/no" genuinely more reliable from this model than "rate peril 0-3"? If
   yes, how many questions before the token budget bites, and can they be
   batched into one structured call or must they be separate?
3. **Where does the formula live, and who owns it?** If the LLM only answers
   observations and the repo computes the band, the band is deterministic and
   auditable -- but the FORMULA is then a design artefact somebody must defend.
   Max-of-dimensions? Weighted? What makes a band change?
4. **What is computable WITHOUT the model?** Name the dimensions that should
   never reach an LLM: startle/loudness from the mix, runtime, act count,
   speaker count, on-page death via the ledger's own structure. Every one moved
   out of the model is one that cannot drift.
5. **Determinism and the repair ladder.** Same frozen ledger, same rating, every
   time -- what does that require given `OTR_C7` seed pinning? What is the
   DETERMINISTIC FALLBACK when the rater fails its attempts, given a missing
   rating cannot block a render (the credits need a value)?
6. **Token budget.** The writer already fights for output tokens. Where does the
   rating pass run -- inside the writer's budget, after the freeze as its own
   pass, or on the frozen ledger entirely out of band?
7. **Structured output.** This repo has `_otr_structured_call` and a JSON-schema
   `response_format` path. Should the rater use it, and does a schema-constrained
   call actually reduce drift for this model, or just move the failure into
   schema-repair retries?
8. **What is the honest failure mode?** A wrong rating that ships is worse than
   no rating. Under what conditions should the rater REFUSE rather than guess,
   and what do the credits show then?

## CONSTRAINTS

100% local, no cloud, no API. The judge is the same local GGUF the writer uses
and shares its VRAM and token budget. The rating is a LABEL, never a publish
gate. The ledger owns the field with exactly one writer, frozen before the
opening credits so both credit rolls agree. Original vocabulary -- no borrowed
letter marks. Deterministic on a frozen ledger. **Do not launch renders or boot
a server.**

---

# OPERATOR RULING, 2026-08-02: PRESENCE, NOT INTENSITY

> "Better is: does it contain violence, smoking, profanity or sexual content?
> Then say 'For Mature Audiences'."

**This supersedes the 0-3 banded design above, and it is better.** Recording why
so nobody re-derives the complicated version later.

## WHY IT IS BETTER FOR THIS JUDGE

The banded design asked a local 12B model to CALIBRATE -- what separates peril 2
from peril 3? Nothing in the rubric can make that stable, and this repo has
already watched this exact model drift under load (speaker-format drift that
killed a leg; stage direction substituted for dialogue). Intensity scoring is
the single worst task to hand it.

Presence is the opposite kind of question. "Does a character smoke? yes/no" is
an OBSERVATION, and observation is what a small local model is genuinely good
at. It needs no anchor examples eating the writer's token budget, no calibration
across episodes, and two runs over the same frozen ledger can agree.

It also makes the formula trivial and therefore auditable, which is the whole
point of an open methodology:

    any(violence, smoking, profanity, sexual_content)  ->  FOR MATURE AUDIENCES
    none                                               ->  (the default label)

No weighting to defend, no averaging that hides a single strong signal, and a
reader can check the call from the receipt.

## THE FOUR DESCRIPTORS

`violence` / `smoking` / `profanity` / `sexual_content`. Tobacco is a real
descriptor in shipping systems (BBFC and ESRB both surface it), so its inclusion
is not idiosyncratic -- it is one of the categories that most often surprises a
viewer who was not warned.

## ONE ADDITION WORTH MAKING (small, keeps the operator's rule intact)

**Stamp WHICH descriptor triggered, even though the displayed label is binary.**
The credits can still read exactly "For Mature Audiences" -- but the ledger
receipt should carry `{violence: true, smoking: false, profanity: false,
sexual_content: false}`. Costs nothing at judgement time (the model already
answered all four), and it is the difference between a rating you can audit and
an oracle. It also makes a wrong call debuggable: "which question did it get
wrong?" instead of "the rating is wrong."

Whether the CREDITS name the descriptors or just the label is the operator's
call. The LEDGER should carry them either way.

## WHAT THIS CHANGES ABOUT THE PANEL NOW RUNNING

The r1 panel was briefed on the banded design. Its answers on token budget,
structured output, determinism, repair-ladder and deterministic fallback still
apply -- those are about HOW a local model is asked anything. Its answers about
dimension calibration and formula weighting are now moot. Weigh accordingly
rather than adopting wholesale.
