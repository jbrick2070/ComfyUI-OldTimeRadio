# Writing your own OTR source pack

> **Your pack can do whatever it wants, as long as it fills the ledger and uses the LLMs
> obediently to make the best ledger story it can.**

That is the entire contract. Two LLM slots, one ledger. What happens in between is yours,
provided it is bug-free and honest.

"Obediently" is doing real work in that sentence: **you drive the models, they do not drive
you.** Every line the listener hears must be written by a model, and every decision about
what is *acceptable* must be made by your code. Python judges. The LLM writes. Neither does
the other's job.

## Your pack is not a variant of anything

Read this before anything else, because it is the most common misunderstanding:

**A source pack is an ORIGINAL PATH, not a fork of an existing one.** It is not a
"sci-fi lane with different prompts." The lanes that happen to ship today sound the way
they do because that is what somebody wanted to hear -- **the genre is theirs, not the
pipeline's.**

Your pack owns:

- **its source, or none at all.** Anything that can seed an episode can be a source -- a
  feed, a text, a photograph, a single word -- and inventing from nothing is just as
  legitimate.
- **its genre and its form.** Here is the actual brief: **invent a kind of radio drama
  that does not exist yet.** Not a version of one you have heard. Something a listener will
  genuinely enjoy and could not have got anywhere else.
- **its dramatic architecture.** How many passes. What the passes ARE. What artifacts pass
  between them. Who judges what.
- **its cast, its roles, its rules.** How many voices there are, what each one is for, what
  each is allowed to know and to say -- even whether it speaks at all. There is no default
  cast.

One boundary on all of it: **No guns. No blood. No violence. No swearing.** That is not a
content warning bolted on afterwards -- it is the constraint that forces the interesting
answer. Tension is not violence. Stakes are not gore. If your first instinct is a shootout,
a murder, or a monster, that instinct came from somewhere else; go past it. What holds a
listener without those things is exactly what you are here to invent.

The pipeline is old-time radio -- voices, music, a listener in the dark. Beyond that, this
document is the complete list of what you must obey. Everything not in it is a blank page,
and the blank page is the whole point. This is a contract, not a style guide.

---

## 1. THE TWO SLOTS -- you may not invent a third

You are handed exactly two callables:

```python
def run_my_episode(*, pack, payload, resolved, led, meta,
                   creative_fn: GenerateFn, technical_fn: GenerateFn, ...)
```

- `creative_fn` -- the **creative** slot. Story, dialogue, character, image prompts.
- `technical_fn` -- the **technical** slot. Extraction, audits, structured judgement.

Both have the same signature: `fn(messages, *, temperature, max_new_tokens, stop=None) -> str`.

**Hard rules:**

- **You may NOT create a new LLM architecture.** No new loader, no new backend, no
  direct `transformers` / API calls, no second model of your own. The operator chooses
  the models in the workflow; your pack receives them. If your pack imports a model
  library, it is wrong.
- **You may use both slots, or just one.** Using `creative_fn` alone is legitimate.
  Using only `technical_fn` is legitimate. Nothing requires you to use both.
- **Call them as many times as you like**, in any order, with any prompts, for any
  intermediate artifacts you invent. Nobody is counting your passes.

**Why:** the two slots are how VRAM, context caps, prompt truncation, quantisation and
model routing stay under one roof. A pack that reaches around them breaks every one of
those guarantees at once.

---

## 2. THE LEDGER -- this is the deliverable

Your job, and your only obligation, is to **fill the production ledger**. Everything
downstream -- casting, voices, freeze, render, captions, credits, publish -- reads the
ledger and nothing else. It never reads your intermediate artifacts.

You fill it through the `Ledger` object (`nodes/production_ledger.py`):

| call | what it holds | required |
|---|---|---|
| `led.set_cast(rows)` | `char_id`, `name`, `character_description`, `gender`, `tts_model`, `voice_preset` | **yes** |
| `led.set_lines(rows)` | the spoken script -- `line_id`, `char_id`, `speaker_role`, `text`, `beat_id`, `boundary` | **yes** |
| `led.set_scenes(rows)` | `scene_id`, `env`, `description` | yes (>=1) |
| `led.set_shots(rows)` | `shot_id`, `scene_id`, `description`, `visual_prompt` | yes (>=1) |
| `led.set_beats(rows)` | `beat_id`, `shot_id`, `scene_id`, `char_id`, `line_ids` | yes |
| `led.set_music(rows)` | `cue_id`, `placement`, `description`, `generation_prompt` | yes |

**The table is plumbing, not dramaturgy.** It looks like a film crew's anatomy -- scenes,
shots, beats -- because the render tail downstream consumes those rows. Do not mistake it
for a description of what an episode IS. **One scene and one shot is a legal episode**, and
not as a loophole: if your pack's true form is a single unbroken voice in an unchanging
dark, then one scene and one shot is the honest shape of it, and you should say so in the
ledger rather than inventing a crew's worth of structure to look respectable. Fill the
table with the truth about your episode. It will render either way.

**Hard rules:**

- **Every id must resolve.** A beat names a shot that exists; a shot names a scene that
  exists; a line names a beat and a char that exist. Dangling ids are the single most
  common way a new pack dies -- and it dies 15 minutes later, in the render.
- **`text` is what the listener HEARS.** No speaker labels, no stage directions, no
  `(sighs)`, no `[sfx]`, no quotation marks wrapping the whole line.
- **The announcer is a fixed role.** If your cast has one, its `char_id` is `announcer`.
- Return the shape your lane's tail expects, and let the shared writer tail do the rest
  (it stamps delivery text, the episode seed, and the freeze receipts for you).

---

## 3. THE LAW -- Python judges, the LLM writes

This is not negotiable, and it is where most new packs go wrong.

- **Python may NEVER author story text.** No literal assigned to a `text=` field. No
  f-string that builds a spoken line. No template, no canned fallback, no "safe default"
  sentence. If a line must be spoken, a model must have written it.
  *(There is an AST guard in the test suite that will fail your build for this.)*
- **Python MAY repair mechanical metadata** -- ids, ordering, enums, a parent reference,
  a fixed role label, keys the schema forbids. Anything already implied by an artifact
  you have already accepted.
- **If it is ambiguous, FAIL CLOSED.** Never guess. A dead episode is fine; a silently
  wrong one is not.
- **The word count is advisory.** `target_words` is a scale request and a post-hoc
  statistic. It may never cause a trim, a pad, a cull, or a rewrite. Do not build a gate
  on it. (We have been bitten by this four separate times.)

---

## 4. THE ANNOUNCER -- somewhere, someone must ground the story

The listener is in the dark. They cannot see a title card, a set, or a face. If nobody
tells them where they are, they spend the first half of your episode working it out instead
of feeling it.

**The hard rule: at some point, the announcer must GROUND the story.** Give the listener
context for what they are hearing, and bring them back out at the end. An episode that
simply *begins* -- mid-argument, unlocated, unexplained -- does not admit anyone into a
story. It just starts one, and leaves the listener outside it. (We shipped exactly that
episode. It was technically perfect and dramatically inert.)

*Where* the grounding falls, and what shape it takes, is yours. Nothing requires it to come
first: opening cold -- mid-scene, mid-sentence, the listener lost for a moment -- is a
legitimate and often thrilling choice, provided the frame arrives. Early, late, threaded
through, split across the ends, or a shape nobody has used. The only illegal shape is its
absence.

And one thing worth saying plainly, because it is the trap: **the announcer FRAMES, it does
not ARGUE.** The moment the announcer starts taking turns in the scene -- answering a
character, pressing a point, joining the debate -- it stops being the voice of the show and
becomes a fourth person in the room, and the listener loses the only orientation they had.
If your announcer is trading lines with your cast, you have lost the frame.

---

## 5. GATES -- what you are allowed to block on

You will want to validate the model's output. Good. But a gate that blocks production may
**only** block on something that is:

1. **objectively checkable** -- against a schema, a lexicon, an id graph. Not a vibe.
2. **actually fixable** by the party you are asking to fix it, and
3. **genuinely a defect** -- not a note, not a preference, and not your own contract
   being honoured.

Failing that test, it is a **note**: record it, log it, stamp it on `meta`, and ship.

If your gate cannot name the offending item AND the reason AND a way to fix it, it is not
a gate.

*(The graveyard of gates that broke this rule is in `SOURCE_PACK_PREFLIGHT.md`. Read it
when you have finished designing -- not now.)*

---

## 6. RETRIES -- ask twice before you kill

Every structured pass in this pipeline gets a bounded retry ladder: base call ->
structural retry -> typed repair. When a model's output is rejected, **tell it the actual
reason** and ask again ("your line was 337 characters; the limit is 300 -- cut it down").

"No fallback" means *never ship a canned line*. It does **not** mean "no second chance."
An announcer intro once died on a single overlong sentence with no retry at all, and took
four source banks down with it.

---

## 7. SEAMS -- your prompts live in the pack JSON

Your prompts belong in `nodes/story_packs/<your_bank>/<your_bank>.json` under
`prompt_stages`, not in Python.

**The seam and the schema are ONE contract.** If your seam shows the model
`{"fact_ids": [...]}` while your Pydantic model declares `fact_uses`, the model will obey
the seam, strict mode will reject it, and your repair will quietly delete the model's work
to force it to validate. **When they disagree, the model is not wrong -- you are.**

A test in the suite parses every seam's worked example and validates it against the schema
it feeds. Keep them in step.

---

## 8. ROLES -- if your characters have different contracts, judge them differently

If your pack gives two speakers opposite jobs, a validator that applies one rule to both is
broken by construction: it will fail one of them for doing exactly what you licensed it to
do. Make your validators role-aware, or do not give your roles different contracts.

---

## 9. WHAT YOU MUST NOT DO (the short list)

- Do not create or load an LLM. Use the two slots.
- Do not write story text in Python.
- Do not gate on word count.
- Do not block on a warning, a note, or a preference.
- Do not build a gate you cannot enumerate, corroborate, or repair.
- Do not leave an id dangling.
- Do not ship a canned line when the model fails -- retry, then fail closed.
- Do not ship an episode nobody grounds -- and keep the announcer out of the argument.
- Do not reach for guns, blood, violence, or swearing.

---

## 10. HOW TO KNOW YOU ARE DONE

Design freely first. Build the thing you actually want to hear. **Then**, once you have
designed it (R1), coded it (R2) and wired it (R3), walk the pre-flight:

> **`docs/SOURCE_PACK_PREFLIGHT.md`**

That is where every bug this pipeline has ever died of is written down, as a checklist you
run at the gates. It is deliberately NOT in this document: a list of failure modes read
before you start will make you build something defensive and dull, and most of these bugs
were in the *contract*, not the code -- they are invisible until you have a contract to
check.

Then, and only then:

```
1. pytest -q                      # the whole suite, including the cross-lane guards
2. the Bug Bible regression       # comfyui-custom-node-survival-guide
3. a live 30-word canonical run   # it must PUBLISH to otr\obs\ -- Test-Path the file
```

**A pack that passes the unit tests but has never published a 30-word episode is not
done.** Nearly every real defect here was green in CI and only appeared in a live render.
Run the 30-word smoke. It is the real gate.

---

## 11. THE EXISTING BANKS, AS PROOF OF THE FREEDOM

Ignore what these are *about*. Look at how differently they are BUILT. They are evidence,
not a menu. Every one fills the same ledger through the same two slots, and no two work
alike:

- **one** runs a 10-pass ladder: extract facts -> pose a dramatic question -> cast ->
  build a score graph -> write the whole script -> review -> retake -> audit.
- **one** pitches three premises, critiques its own pitches, picks a winner, outlines, then
  drafts and critiques **scene by scene**.
- **one** stages a ceremony: it drafts **line by line**, alternating two speakers with
  *opposite* contracts -- a literalist who may only say what the source supports, and a
  speculator licensed to reach past it -- then runs an audit / challenge / rewrite loop.
- **one** has **no source at all**: it draws atoms from a spark deck and invents the
  episode outright.
- **one** seals its canonical text against a proof map so nothing downstream can touch a
  word of it.

Different pass counts. Different artifacts. Different judges. Different sources -- or none.
One of them is not even in the same *genre* as the others.

Same two slots. Same ledger. Everything between those two facts is yours, and the gap is
enormous. **Do not build a variant of one of these. Build the thing nobody has built.**
