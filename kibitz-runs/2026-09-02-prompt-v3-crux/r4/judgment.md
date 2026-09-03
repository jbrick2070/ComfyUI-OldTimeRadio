# r4 judgment -- convergence

Round: r4 (convergence). Reviewer seat: **Sonnet** (Cowork subagent, grounded on
the real Windows files). Driver: Claude (Cowork, 5080).

**VERDICT RETURNED: converged.** No build-breaker survives r2 and r3. Five open
spec gaps were named, all five are real, and all five are closed here before any
code is written.

---

## 1. The finding that matters most, and it is a REGRESSION r2 and r3 both missed

r4 read why `GHOST_MODE_LAWS_V2` exists in the first place, and the comment above
it (`ghost_signal_prompt.py:114-127`) records a measured same-seed pair:

> the v1 arm rendered a recognisable subject on **4 of 4** sampled beats; the
> first-draft v2 arm rendered **0 of 4**, and its LLM-authored sibling managed
> 2 of 8 -- both of them on the only beats whose authored leaf happened to name a
> real object

and draws the rule: **"LEGIBILITY TRACKS CONCRETE NOUNS."**

So the mode law's job was never framing. It was forcing a concrete noun into a
prompt whose other slots might have none. Deleting it is safe **only because the
crux kernel supplies a better noun** -- "film canisters", "a handheld brass
communicator" are more specific than the law's generic "the object".

**Except when the brief failed.** `_failure_sentinel` stamps `key_objects: []`
and `setting: []` (`_otr_story_brief.py:419-454`), W4 says the kernel is then
omitted rather than refused, and on `sci_fi_radio` the pack cue is empty too --
so v3 would compose `world motion + vantage` and **name no thing at all**. That
is the 0-of-4 condition, reproduced deliberately. v2 does not fail this way
because its sigil is cast-derived and survives a brief failure.

**Accepted, and closed with a fourth tier.** `resolve_crux_kernel` becomes a
four-step ladder, and every tier ends in a concrete noun:

1. `key_objects[i]` in the `setting[j]` -- the story's own.
2. `setting[j]` alone.
3. a bounded `story_brief` slice.
4. **the beat's bookend radio object** -- `GHOST_BOOKEND_MOTIFS`
   (`ghost_signal_author.py:580-585`): a bakelite radio set, a glowing radio
   dial, a broadcast console, a spinning turntable.

Tier 4 is always available, always drawable, always on-brand for a radio
programme, and it is never a costume or a cast look. A brief-failed episode draws
the radio world rather than a field of nothing. The kernel therefore **never**
resolves empty, and `kernel_source` records which tier fired.

Rejected as a tier: `meta.specificity_anchors`. It survives a brief failure and
does carry concrete nouns, but on a real episode it reads
`["35mm print", "Copyright Collection", "film preservation", "Mary Pickford
Theater", "Frank Perry"]` -- proper names, including a person's. Handing "Frank
Perry" to the composer as a drawable subject invites exactly the person-in-frame
outcome the modes exist to control.

## 2. The motion pool -- accepted, and already built that way

r4 is right that reusing `GHOST_FALLBACK_CLAUSES` verbatim would reintroduce the
defect: its `figure` entries are person-centric micro-actions and its
`object`/`signal` entries are tabletop-scale, so "a vast cold water reservoir, a
figure turns a page and holds the paper to the lamp" is the old bug with a kernel
in front of it.

The driver's prototype had already authored a **new, subject-agnostic** pool --
clauses with no embedded noun ("drifting slowly", "settling in the still air",
"shifting as the light crosses it") that compose after any kernel. r4 reached the
same design independently.

**Accepted with r4's sizing discipline:** keyed by VANTAGE rather than by the old
mode identity, and sized against the real worst case rather than a guess. The
2026-08-30 exhaustion (`the figure fallback pool is exhausted: 6 clauses`) fired
on a five-act episode, and `GHOST_CHARACTER_CYCLE` makes figure beats the most
common, so figure exhausts first. Under Half A the pool runs on **every** beat,
not only on failed batches. The pool is sized to the longest real episode's beat
count with a test pinning it, not to a round number.

## 3. The receipt/prompt mismatch -- new, real, and closed

`render_driver.py:2937-2939` stamps `ghost_motif_cue` and `ghost_drawable_beat`
straight from the stored row, independent of which composer ran. Under v3 the
stored object is contractually untouched, so those two receipt keys would keep
publishing *"a lean figure in a charcoal coat, carrying a satchel"* on a beat
whose picture contains no coat. **Nobody in r1-r3 flagged it.**

**Closed by making the receipt say what it means, not by dropping it.** Both keys
keep being stamped -- they are the row's authored provenance and Half B needs them
-- and the v3 branch stamps the composed components beside them
(`prompt_slot_tokens`, `prompt_dropped`, `kernel_source`) plus a comment at the
stamp site saying these two describe the AUTHORED object and not the rendered
text. Renaming them would break the allowlist and every historical reader for no
gain.

## 4. The vantage table -- written down, and the wording reconciled

r4 is correct that no file defines it and that the driver's own task text
(`signal -> lit-in-the-dark`) drifted from Cursor's `hand-or-back`. Settled here,
as an explicit table beside `GHOST_MODE_LAWS_V2` and never derived from it:

| stored mode | v3 vantage clause |
|---|---|
| figure | `wide, the people small in the space` |
| object | `the object large in the frame` |
| signal | `lit against the dark, the light moving` |

`figure` follows the operator directly -- his rewrite reads *"characters moving
through a stagnant mass of water at a reservoir"*, people in the world rather
than a costume in close-up. `signal` keeps the light language because that mode's
whole identity is a thing lit in a dark room, and the prototype showed the light
slot must then be DROPPED on that mode or the prompt carries two contradictory
lighting statements ("harsh fluorescent overheads ... lit against the dark").
Cursor's `hand-or-back` is not adopted: it describes a body part, which is the
vocabulary this change is removing.

## 5. The phantom "tail" slot -- cut, explicitly

r4 caught a real drafting inconsistency: r2's accepted MF9 drop order still names
a "tail" carried over from Fable's superseded five-slot shape, while W1 names four
components and no tail. **There is no tail.** The drop order is
`light -> motion -> vantage -> the setting half of the kernel`, and the subject
noun and pack cue are never dropped and never word-sliced.

## 6. Contract after r4 (the build list -- supersedes W1-W7 where they differ)

* **X1** `resolve_crux_kernel` is a FOUR-tier ladder ending in the bookend radio
  object. It never returns empty and never raises. `kernel_source` is receipted.
* **X2** The world-motion pool is NEW, subject-agnostic, keyed by vantage, and
  sized against the longest real episode with a test that pins no-exhaustion.
* **X3** The vantage table is explicit, three entries, and the light slot is
  dropped on `signal` mode.
* **X4** `ghost_motif_cue` / `ghost_drawable_beat` keep being stamped, with a
  comment saying they describe the authored object rather than the rendered text.
* **X5** No tail slot. Drop order: light, motion, vantage, the kernel's setting
  half. Subject and cue are immutable.
* **X6** Everything else in W1-W7 stands unchanged.

**The arc is closed. r1 Fable cold + Antigravity, r2 Codex, r3 Cursor, r4 Sonnet
-- four rounds, four seats, eight grounded reversals of the driver's own plan.
Code starts now.**
