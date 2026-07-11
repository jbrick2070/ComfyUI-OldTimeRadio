# Source-bank pre-flight -- read this LAST, not first

**Do not read this while you are designing.** It is a list of ways episodes have actually
died, and if you carry it into a blank page you will build something defensive and dull.
Design freely first. Write the thing you want to hear.

Then, when you have:

- **R1 -- designed it** (you know your passes, your artifacts, your roles)
- **R2 -- coded it** (it runs, it fills the ledger)
- **R3 -- wired it** (it is in the workflow JSON and the bank registry)

...come back here and walk the gauntlet. Every item below is a real corpse. Each one cost
a 15-minute render, and most of them were invisible to the unit tests.

Companion docs: `SOURCE_BANK_GUIDE.md` (the contract),
`docs/PROD_BUG_LOG.md` (every live failure, dated), and the Bug Bible in the
`comfyui-custom-node-survival-guide` repo (the cross-project classes).

> **A note on multi-model review (`/kibitz`, `/roundtable`):** do not point a panel at your
> pack until R3. It will regress an original design to the mean. See GATE 3.5 -- bugs are a
> committee problem; taste is not.

---

## GATE 1 -- after R1 (design). Ten minutes, on paper.

**1.1 Can every gate you invented actually be satisfied?**
Take each rule you plan to enforce and ask: could a *good* episode fail it?
- Killed us: a word-count quota demanding exact equality -- 5 words per beat.
- Killed us: "every fact must be integrated", enforced *per scene*, when the facts belong
  to the episode.
- Killed us: an auditor failing a script because a line "adds no new information" -- in a
  30-word episode with a two-fact source.

**1.2 Do your roles have different contracts?**
If one speaker may only state what the source supports and another is licensed to
speculate, then a single validator applied to both is broken before you write it. It will
fail the speculator *for speculating*.

**1.3 Does any rule demand something you cannot enumerate?**
If you cannot list the evidence, you cannot corroborate the finding, and you cannot repair
it. That is not a gate -- it is a coin flip the episode can only lose. (We shipped one for
months. It killed a whole bank.)

**1.4 Is anything in your design a "note" pretending to be a defect?**
Pacing, repetition, thin drama, word choice, register, a warning. Record them. Never block
on them.

---

## GATE 2 -- after R2 (code). Run these before your first live roll.

**2.1 Is your seam's worked example a LEGAL instance of your schema?**
Paste the example JSON out of your prompt and validate it against your Pydantic model.
- Killed us: the seam said `{"fact_ids": ["F01"]}`; the model declared `fact_uses`. The
  LLM obeyed the seam, strict mode rejected it, and the repair then **deleted the model's
  work** to force it to validate. *When the seam and the schema disagree, the model is not
  wrong -- you are.*
- Killed us: a seam asking the cast for `tts_model` / `voice_preset` that the schema
  forbids.

**2.2 Can your schema express what JSON can express?**
- Killed us: `tuple[X, X, X]` in a `strict=True` model fed from `json.loads`. JSON has no
  tuple. That field could **never** validate, no matter what the model wrote. Use a
  length-pinned `list`.
- Killed us: `cites` with `min_length=1`, on a lane whose ceremonial lines cite nothing --
  so the code invented a sentinel id that could not exist. **The schema forced the lane to
  lie.** If a line can legitimately have none, allow none.

**2.3 Does Python author any story text?**
Grep your pack for a literal or f-string assigned to `text=` / `premise=` / `title=`.
- Killed us: `text="The record holds now."` -- Python speaking for a character, while the
  model's own line for that moment sat unused in the artifact and was thrown away.
- There is an AST guard in the suite. It will fail your build. Good.

**2.4 Does every rejection tell you WHY?**
Every gate must name the offending item, the reason, and the evidence.
- Cost us three rolls of pure guessing: `"scene X failed its bounded rewrite"` -- which
  scene? which line? what was wrong with it? Print the accused line next to the accusation.

**2.5 Do you ask twice before you kill?**
A rejected output gets the reason and a retry. "No fallback" means *never ship a canned
line* -- it does **not** mean "no second chance."
- Killed four banks: an announcer intro four characters over a 300-char cap, with no retry
  at all.

**2.6 Do your ids resolve?**
Beat -> shot -> scene -> line -> char. Walk the graph in a unit test. A dangling id dies
in the render, twenty minutes later.

**2.7 Do your creative rolls draw OS entropy?**
Creative RNGs must not seed themselves. Reproducibility comes ONLY from the
`OTR_CAST_SEED` / `OTR_STYLE_SEED` env overrides. A pack that plants its own fixed seed
ships the same episode forever -- and looks perfectly healthy in every test.

---

## GATE 3 -- after R3 (wiring). The arithmetic nobody does.

**3.0 Is your pack actually WIRED?**
Code that is not reachable from `workflows/otr_canonical.json` (plus the bank registry) is
dead, however good it is.
- Killed us: a node and a new blend input shipped, tested green, and ran **dormant in
  production** -- nobody had wired it (2026-06-13).
- Run `OTR_WorkflowValidator`, then audit widget-count vs live `INPUT_TYPES` and link
  referential integrity. `widgets_values` is POSITIONAL: only ever APPEND a new optional
  widget at the END -- inserting mid-list silently shifts every saved value
  (BUG-LOCAL-097).

**3.1 Does your prompt FIT?**
`context_cap` is 8192. `max_input_tokens = context_cap - max_new_tokens`. If your prompt
is bigger, it is **silently left-truncated from the front** -- the system message and the
schema go first.
- Cost us four consecutive rolls: `PROMPT_GUARD: Truncated 5408 -> 4592`. A pass reserved
  3600 output tokens, leaving 4592 for input, and its repair prompt was 5408. The model
  never received the instructions we kept "improving". **It was not ignoring us. We were
  cutting our own instructions off.**
- Your typed-repair prompt is ALWAYS the fat one: it carries the failed artifact *and* the
  validation error *and* the original request.
- Set `prompt_must_fit=True` on any provenance-bearing pass so it fails LOUD instead of
  lying to you.

**3.2 Is your output reservation sized off the right dimension?**
- Killed us: a budget scaled from the word count, when the artifact's size is driven by the
  **line count** (per-line metadata). A 30-word script with 13 lines pays nearly all the
  same cost as a 300-word one.

**3.3 Is your gate reachable-and-passable?**
- Killed us: a finalizer demanding `freeze_verdict == "frozen_clean"` on a lane where the
  freeze runs *before* CastLock assigns voices -- so the cascade always has a note, and
  `frozen_clean` was **unreachable by construction**. The gate could never pass.

**3.4 Warnings are not errors.**
Errors block. Structural verdicts block. Warnings get recorded and the record ships.

---

## GATE 3.5 -- if you have `/kibitz` or `/roundtable`, use them HERE. Not earlier.

If you have the multi-model review tools, they are very good at exactly one thing: finding
bugs in something that already exists. They are dangerous at the one thing you must not
let them do: **design your pack.**

**Never run a panel at R1.** A panel optimises for *agreement*. Ask three models to
critique a blank-page architecture and they will converge on the safe, average, familiar
version of it -- and the strange, specific, original idea that made your pack worth writing
is precisely what gets sanded off first. You will end up with a pack that four other people
could have written. Do not hand your architecture to a committee before it can defend
itself.

**Run them at R3 and at pre-flight**, once the design is committed and the code is wired.
Now the panel is doing what it is actually good at:

- arithmetic you did not do (context caps, token reservations, VRAM)
- contradictions between a seam and its schema
- gates that cannot pass, or cannot fail
- dead levers nobody calls any more
- ids that do not resolve

**Ask it bug questions, not taste questions.**
- Good: *"Where does this silently truncate? Which gate is unsatisfiable? What does the
  ledger expect that I never set?"*
- Bad: *"Is this a good design? What would you do differently? How should the story work?"*

**You are the judge, always.** Panels are confidently wrong. In this repo, on one day: they
caught the seam/schema contradiction that was destroying the model's own fact attribution,
they predicted the warning-as-fatal gate before it fired, and they correctly settled a
one-based/zero-based indexing dispute. They *also* asserted a bug hit a lane it demonstrably
did not, and produced an inflated truncation table that the actual logs refuted. **Ground
every claim against the real code before you act on it.** A finding you have not verified is
a hypothesis, not a bug.

Bugs are a committee problem. Taste is not.

---

## GATE 4 -- the real gate

```
1. pytest -q                     # whole suite, including the cross-lane guards
2. the Bug Bible regression      # comfyui-custom-node-survival-guide
3. a live 30-word canonical run  # must PUBLISH to otr\obs\ -- Test-Path the file
```

**A pack that passes the unit tests and has never published a 30-word episode is not
done.** Nearly every defect on this page was green in CI and died live.

When it fails -- and it will -- read the log before you theorise. Nine times out of ten the
model was doing exactly what you told it, and the bug is in the contract, not the code.

---

## The one-line version

> **A gate may block production only if it is objectively checkable, actually fixable by
> the party you're asking to fix it, and genuinely a defect -- not a note, not a
> preference, and not your own contract being honoured.**

Everything on this page is that sentence, learned the expensive way.
