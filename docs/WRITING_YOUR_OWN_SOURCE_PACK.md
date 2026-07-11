# Writing your own OTR source pack

**The whole contract, in one sentence:** you get two LLM slots and a ledger to fill;
what happens in between is entirely yours, as long as it is bug-free and honest.

This is not a style guide. It is the set of hard requirements a new source bank must meet
to run in this pipeline. Everything not listed here is your design freedom -- and the four
banks that ship today (`scifi_codex`, `scifi_gemini`, `scifi_sonnet`, `scifi_fable2`)
each fill the ledger by a completely different route. That is the point.

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

## 4. GATES -- what you are allowed to block on

You will want to validate the model's output. Good. But a gate that blocks production may
**only** block on something that is:

1. **objectively checkable** -- against a schema, a lexicon, an id graph. Not a vibe.
2. **actually fixable** by the party you are asking to fix it, and
3. **genuinely a defect** -- not a note, not a preference, and not your own contract
   being honoured.

Failing that test, it is a **note**: record it, log it, stamp it on `meta`, and ship.

Real gates that killed real episodes here, every one of them removed:

- an exact word-count quota nobody could hit (5 words per beat, equality demanded)
- an episode-level fact rule enforced per scene -- scene 2 failed for not containing a
  fact belonging to scene 1
- an auditor failing a script because a line "adds no new information", in a 30-word
  episode with a two-fact source
- a hygiene check rejecting ALL-CAPS emphasis, in a lane whose characters are *named* in
  all caps by schema -- it rejected a line for obeying the contract
- a content class that could be flagged but never enumerated, corroborated, or repaired:
  an unbounded hunch with a hardcoded `fail`. It was a coin flip the episode could only
  lose.

If your gate cannot name the offending item AND the reason AND a way to fix it, it is not
a gate.

---

## 5. RETRIES -- ask twice before you kill

Every structured pass in this pipeline gets a bounded retry ladder: base call ->
structural retry -> typed repair. When a model's output is rejected, **tell it the actual
reason** and ask again ("your line was 337 characters; the limit is 300 -- cut it down").

"No fallback" means *never ship a canned line*. It does **not** mean "no second chance."
An announcer intro once died on a single overlong sentence with no retry at all, and took
four source banks down with it.

---

## 6. SEAMS -- your prompts live in the pack JSON

Your prompts belong in `nodes/story_packs/<your_bank>/<your_bank>.json` under
`prompt_stages`, not in Python.

**The seam and the schema are ONE contract.** If your seam shows the model
`{"fact_ids": [...]}` while your Pydantic model declares `fact_uses`, the model will obey
the seam, strict mode will reject it, and your repair will quietly delete the model's work
to force it to validate. **When they disagree, the model is not wrong -- you are.**

A test in the suite parses every seam's worked example and validates it against the schema
it feeds. Keep them in step.

---

## 7. ROLES -- if your characters have different contracts, judge them differently

If your pack gives two speakers opposite jobs -- a literalist who may only state what the
source supports, and a speculator licensed to extrapolate -- then a validator that applies
one rule to both is broken by construction. It will fail the speculator for speculating.
Make your validators role-aware, or do not give your roles different contracts.

---

## 8. WHAT YOU MUST NOT DO (the short list)

- Do not create or load an LLM. Use the two slots.
- Do not write story text in Python.
- Do not gate on word count.
- Do not block on a warning, a note, or a preference.
- Do not build a gate you cannot enumerate, corroborate, or repair.
- Do not leave an id dangling.
- Do not ship a canned line when the model fails -- retry, then fail closed.

---

## 9. HOW TO KNOW YOU ARE DONE

```
1. pytest -q                      # the whole suite, including the cross-lane guards
2. the Bug Bible regression       # separate repo
3. a live 30-word canonical run   # it must PUBLISH to otr\obs\ -- verify the file exists
```

A pack that passes the unit tests but has never published a 30-word episode is not done.
Every single defect on the list in section 4 was invisible to the test suite and only
appeared in a live render. **Run the 30-word smoke. It is the real gate.**

---

## 10. THE FOUR BANKS, AS PROOF OF THE FREEDOM

All four fill the same ledger. None of them work the same way:

- **codex** -- a 10-pass ladder: fact index -> dramatic question -> cast -> score graph ->
  whole script -> review -> retake -> audit.
- **gemini** -- pitches three premises, critiques its own pitches, picks one, outlines,
  then drafts and critiques **per scene**.
- **sonnet** -- a far-future archive ceremony: it drafts **per line**, alternating a
  literalist and a speculator, then runs an audit / warden / rewrite loop.
- **fable2** -- a content-owned loop that seals its canonical text against a proof map.

Different pass counts, different artifacts, different judges. Same two slots. Same ledger.
That gap is where your pack lives.
