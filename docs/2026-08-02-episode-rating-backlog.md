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

Worth stating early, because it changes the design: `CLAUDE.md` already binds
every episode to "Safe for work. Non-violent. No curse words anywhere." So the
rater is NOT policing a wide range -- everything shipped should already sit in
the mildest couple of bands.

What DOES vary, and is worth signalling, is TONE and INTENSITY:
suspense, dread, peril, loss and grief, moral weight, supernatural or body
strangeness, loudness/startle in the audio mix. The fear-cape inversion exists
precisely because some beats are meant to unsettle.

So the honest product is closer to **viewer guidance** than to a gate: an
episode of this show is not going to be unsuitable, but one may be markedly
more unsettling than another, and the credits are a reasonable place to say so.

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

## OPEN QUESTION FOR THE OPERATOR (when this is scheduled)

Given everything is SFW by rule, is the rating meant to be **informative**
(tone/intensity guidance, which is what this document assumes) or
**gatekeeping** (a band that could block publication)? The two produce very
different designs, and the second one needs a policy decision that is yours,
not mine.
