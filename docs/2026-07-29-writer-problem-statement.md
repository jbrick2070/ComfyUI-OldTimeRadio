# The Writer Kills Half the Episodes — Problem Statement

**For:** Fable (or any panel) to analyze and propose a repair plan
**From:** live evidence, OTR 45-word campaign, 2026-07-29
**Repo:** `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`, HEAD `47c554fa`

---

## Read this first: the vocabulary

The writer turns a source news article into a finished radio script. It does
this in five LLM passes, named P0 through P5. In order:

| Pass | Plain English | Does the finished episode read its output? |
|---|---|---|
| **P0** | Read the article, extract the facts, and quote the exact source text for each one | **No.** Feeds later prompts only. |
| **P1** | Decide the dramatic question | No — feeds later prompts. |
| **P2** | Invent the cast | **Yes** — character names, voices. |
| **P3** | Plan the scenes and beats | **Yes** — shots, scene structure. |
| **P5** | Write the actual spoken dialogue | **Yes** — every spoken line. |

The **ledger** is the finished data structure everything downstream reads:
the cast, the lines, who says what, and when.

---

## The problem, in one sentence

**Half of all episodes die before a single frame is rendered, because a
writer pass refuses its own output and the run has no way to continue.**

---

## The evidence

48 legs of a live GPU campaign, three passes over 19 engines, same code:

| Outcome | Count | Share |
|---|---|---|
| Landed a finished episode | 11 | 23% |
| **Died inside the writer** | **24** | **50%** |
| Died in video render | 8 | 17% |
| Died in the image phase | 1 | 2% |
| Cut off by a timeout | 1 | 2% |

**The writer owns 24 of the 37 failures — 65%.** Of those 24: 15 died in P0,
8 in P5, 1 in P3.

### It is a coin flip, not a bug

Nine engines produced **both a pass and a failure on byte-identical code**.
`still_flat`, `still_word`, `viz_camera`, `viz_green`, `viz_mxc_mandala` all
failed twice and then succeeded on the third roll with nothing changed. Same
commit, same profile, same source. There is no single bad input to find.

---

## Two concrete causes, both verified from live logs

### Cause 1 — the model's answer is cut off mid-sentence

**28 of ~48 legs** logged `no decodable top-level JSON object found`. The
model is given a fixed output budget (P0: 2800 tokens). It writes a long,
verbose fact index, runs out of room, and the JSON object never closes. The
truncated text is then rejected as malformed.

This is not a judgment failure. The model was not given enough room to finish
the sentence it was told to write.

### Cause 2 — the source article contains raw web junk

Verified example, leg `viz_green`, source = an MIT news article:

```
The source text contains:  ...Department of Energy's (DOE)&nbsp;Genesis Mission, with 15...
The model quoted:          ...Department of Energy's (DOE) Genesis Mi...
```

The article has a literal HTML non-breaking space (`&nbsp;`) in it. The model
read it as a space — which is what it is, and what any human reader would do.

P0 requires every quote to be a **byte-exact substring** of the source. A
space is not `&nbsp;`, so the quote was rejected. Then the retry was rejected.
Then a second, different model was brought in, and its answer was rejected for
the same reason. **Four attempts, two models, one non-breaking space, no
episode.**

---

## What makes this structural rather than unlucky

1. **A rejected answer is deleted.** When the second model's repair fails
   validation, its output lives in a local variable that goes out of scope.
   Only a character count and a SHA-256 reach the journal. The bytes are gone,
   so nothing can build on a near-miss.
2. **Validation is all-or-nothing.** The validator returns one error string or
   `None`. There is no way to express "four of five facts are fine, drop the
   fifth." One bad row kills the whole index.
3. **The failure has exactly one exit.** The writer raises, and the call site
   in `OTR_LedgerScriptWriter.py:3473` has no `try`/`except` around it. The
   exception leaves the node and the episode is over.
4. **P5 has no second chance at all.** Only P0 is configured with an alternate
   model. P5 — the pass that writes the actual dialogue — gets one model at
   three temperatures and then dies.

---

## The operator's ruling, which has not been built

> "The writer should not be allowed to kill the run, it just needs to fix the
> ledger." — and restated — "the writer should never veto, the writers should
> keep on passing in a loop to agents to clean up the ledger."

This was recorded and then parked. One slice has since landed (the P0 repair
envelope now trims to fit instead of refusing, and P0's deterministic span
repair is finally reachable). The loop itself does not exist.

---

## The operator's ledger requirements — the target contract

What a workable ledger must contain, in the operator's own terms:

**Per character (the cast):**

- **Character name**
- **Character description**
- **Gender** (age is probably not needed)

**Per line (the script):**

- **Who is speaking** — a cast character, or the announcer
- **The dialogue itself**
- **A marker if the line is a music beat** rather than speech

**ACTION IS NOT RECORDED, AND THERE IS NOWHERE TO PUT IT.** The operator's
recollection here is confirmed by the schema: `ScriptLineV4`
(`nodes/_otr_scifi_codex.py:505`) carries `char_id`, `speaker_role`
(`character` / `announcer` / `music_open` / `music_inter` / `music_close`) and
`text`. There is no action, stage-direction, or parenthetical field anywhere on
a line. Music beats are their own rows, flagged by `speaker_role`.

This sharpens the rule rather than softening it: because action has no home in
the ledger, any action text a model writes **inside a spoken line** is pure
contamination — it will be read aloud by the TTS engine as if it were
dialogue. It must be **stripped from the line**, never used as grounds to
reject the line.

**There is no maximum length.** Whatever word count arrives, the render side
owes enough clips and enough stills to cover it. A long line means more clips,
never a refusal.

---

## Questions for the analysis

1. **Where is the right place to stop the bleeding?** Options seen so far:
   decode the source text before the model ever sees it; give the model room
   to finish; stop demanding byte-exact quotes; retain rejected answers and
   iterate on them; catch the exception at the one call site and degrade.
   Which of these are root fixes and which are treating symptoms?

2. **What is the floor?** If a pass ultimately cannot produce a clean answer,
   what is the minimum ledger that still yields a watchable episode? Every
   field needs exactly one owner — deterministic code, another pass, or an
   explicit default. Which fields can Python own outright?

3. **Should P0 exist in its current form at all?** It is the biggest killer
   and nothing in the finished episode reads its output. It feeds three
   downstream prompts. What breaks if its evidence contract is relaxed from
   byte-exact quoting to something a model can actually satisfy?

4. **What does the "loop to cleanup agents" look like concretely** — how many
   iterations, what carries forward between them, and what ends it?

5. **How does the dialogue/action separation get enforced** without becoming
   another veto? The rule is "strip action out of dialogue," not "reject the
   line."

---

## Constraints that are not negotiable

- **No length gate.** Word count is telemetry, never grounds for rejection.
- **A hole in the ledger is worse than a loud failure.** Any pass that is
  weakened must hand every field it owned to a new owner first.
- **Fail loud, not fatal.** A degrade must say exactly what went wrong and
  what was done about it. Silent substitution is not acceptable.
- 100% local, open source, offline. No cloud services, no API keys.

---

## Verified vs. estimated

**Verified from logs:** the 48-leg tally; the nine engines that flipped; the
`&nbsp;` failure quoted above; the 28 legs with undecodable JSON; the
discarded-repair behavior; the single uncaught call site; P5 having no
alternate model.

**Not established:** the exact split between "wrong coordinates on real text"
and "the model genuinely paraphrased." An automated classification of that was
attempted and discarded as unsound — it compared each quote against the slice
at the model's own bad coordinates, which differs by construction. Determining
this properly needs the source payloads, which are not in the logs.
