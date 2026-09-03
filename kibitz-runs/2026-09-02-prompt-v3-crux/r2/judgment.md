# r2 judgment -- Prompt v3, "draw the crux"

Round: r2 (coding plan). Reviewer seat: **Codex** (`gpt-5.1-codex-max`, high).
Driver: Claude (Cowork, 5080). Every claim below was checked against the real
Windows files before it was folded; the line numbers are the driver's own reads,
not the reviewer's.

**VERDICT: the review is accepted almost in full, and it reverses the anchor's
build shape.** Codex was right that the plan as written could not be built --
but grounding its strongest claim (the seed mechanism) turned up the fact that
makes the whole feature *cheaper* than the anchor assumed, not dearer.

---

## 1. The finding that reorganises the build

`compose_ghost_prompt_v2` is **pure and scalar** and takes no ledger
(`nodes/_otr_video_engines/ghost_signal_prompt.py:824-844`). But its caller,
`finalize_ghost_prompt_v2`, **already receives `ledger_meta`**, and the render
driver calls it with the live ledger
(`nodes/_otr_video_engines/ghost_signal_author.py:1295-1317`;
`nodes/_otr_video_engines/render_driver.py:2911-2915`).

So the Ghost prompt is not baked into the ledger at plan time. What the ledger
stores is the small authored object -- `mode`, `motif_cue`, `drawable_beat` --
and the finished prompt is composed fresh on every render, with the whole
episode meta in hand.

**That splits the work in two, and the halves are independent.**

**HALF A -- render-time composition. No LLM, no schema change, no new field.**
The crux kernel is resolved from meta that is ALREADY THERE (`key_objects`,
`story_brief_terms.setting`, `visual_palette`), the costume motif is dropped
from the prompt, and the slot order and drop logic change. Nothing the author
stored changes, so:

* every frozen ledger replays under v3 with **no bundle machinery at all**, and
* the render seed is **bit-identical**, for the reason in section 2.

**HALF B -- plan-time authoring.** The leaf vocabulary (`world / thing / hand`),
the beat's own dialogue reaching the author, subject-coverage validation. This
one really does change the stored object and really does need Codex's
version-dispatch discipline (its must-fix 1) and a re-author path for replay
(its must-fix 2).

**Half A is built and judged first.** It is the half that answers the operator's
actual complaint, it buys the same proof for a fraction of the mechanism, and
Half B's design gets to be informed by looking at Half A's output instead of
guessing at it.

## 2. The seed claim -- confirmed, and it lands the other way

Codex must-fix 3 is CORRECT and the anchor was wrong: the planned
`request_seed: 0` field is inert (`nodes/otr_shot_lock.py:2821`), and the render
seed is recomputed as `_seed_from_hash(render_request_hash, shot_id)`
(`nodes/_otr_video_engines/render_driver.py:3749-3772`). "Copy the seeds across"
was never going to work.

Following it to the source, though:

    "request_hash": _content_hash([brief_hash, cast_hash, beat_id, char_id])
    -- nodes/otr_shot_lock.py:1479-1487

**The prompt text is not an input to the request hash.** Neither is the motif,
the leaf, the style cue, nor anything else Half A touches. So a v2-vs-v3 A/B on
one frozen episode is same-seed *by construction* -- there is nothing to
preserve, only something not to disturb. R9 needs no `--derive-prompt-version`,
no `prompt_override` manifest field and no import stamp for Half A. Codex's
must-fix 2 is real for Half B and premature for Half A.

**The trap this exposes, and it is a live one:** `brief_hash` is
`_content_hash(meta["story_brief_terms"] or meta["story_brief"])`
(`nodes/otr_shot_lock.py:1326`). Putting the crux anywhere inside
`story_brief_terms` would move the request hash, move every seed, and destroy
the comparison it exists to enable. **The kernel is therefore resolved from
`key_objects`, a sibling of `story_brief_terms` and not a member of it.** That
also settles Codex must-fix 5: `crux_subject` has no producer, and it needs
none -- `key_objects` is already produced, already validated, already rich.

## 3. What the live artifact says (this is the whole case)

The last published episode, **"The Faded Ledger"** (media_archive,
`storybook_engraving`, 21:08 tonight, 8/8 clips). Its brief: *a high-security
archive filled with film canisters and dusty ledgers where an archivist and a
cynical consultant race against a security sweep.* Its `key_objects`:
`film canisters, handwritten ledgers, archive shelves, ink pens, security
badges`. Its setting: `high-security archive`.

What the dialogue actually names: the security sweep, handwritten inventory
codes, private journals, faded ink on the **canister markings**, the ledger, a
truck idling at **the gate**, a **reel**, an acquisition **waiver**, a
**signature**.

What the lane drew, from the ledger's own stored objects:

| beat | motif (drawn) | leaf (drawn) |
|---|---|---|
| b002 | a tall figure in a **black shawl**, carrying a ledger | a figure in a shawl holds a ledger toward a desk |
| b003 | a black ledger | a black ledger lies open as a hand flips the pages |
| b004 | a lean figure in a **charcoal coat**, carrying a **satchel** | a figure in a charcoal coat holds a satchel out |
| b005 | a **charcoal satchel** | a charcoal satchel sits on a desk under a moving light |

**Shawl, coat and satchel appear nowhere in the episode.** They are minted by
`MOTIF_FALLBACK_POOLS` -- garment "coat", props lantern / key / ledger /
satchel / chart / telegraph -- and then the authored leaf, whose one instruction
is not to repeat the motif, repeats the motif with a verb attached. Two of the
four character beats are a person holding a bag. That is the operator's sentence
reproduced mechanically: *"very similar shots of humans and bags"*, *"not some
coat or figure that is not even mentioned in the dialogue"*.

Meanwhile canisters, shelves, badges, the gate, the reel and the waiver -- every
one of them drawable, every one of them in the ledger or the dialogue -- reach
the picture never. *"Yes I want to see more story not just the characters."*

## 4. Budget: v3 is CHEAPER than v2, which removes the main build risk

Render refuses over 77 installed SD1 tokens in one window and never trims
(`ghost_signal_author.py:1350-1367`, `:1295-1308`). The anchor treated fitting
the new material as the hard part. Against the shape above it is not: the
costume motif that v3 DELETES is the longest slot in the prompt ("a lean figure
in a charcoal coat, carrying a satchel"), and the crux kernel that replaces it
("film canisters in a high-security archive") is shorter. v3 spends fewer tokens
and says more.

Codex must-fix 11 is nevertheless CONFIRMED and the anchor's number was wrong:
`compact_style_cue` returns **2-4 words, and "" for `sci_fi_radio`**
(`nodes/_otr_visual_styles.py:632-653`) -- not the "4-9 tokens" the anchor
budgeted, and on the default pack there is no cue slot at all. Every budget line
is re-measured with the installed tokenizer before code, and the default pack is
measured as a cue-less prompt.

## 5. Accepted as written

* **MF 9** -- keep `compose_ghost_prompt_v3` assembly-only; token-aware fitting
  lives in the v3 finalizer and drops **whole optional units** (light, then the
  optional vantage qualifier, then the tail) and NEVER word-slices the kernel.
  "data logs" must not become "data". Confirmed: v2's finalizer never trims by
  design, so fitting is new behaviour to be built, not borrowed.
* **MF 10** -- v3 gets one finalizer with the banana route applied exactly once;
  `_ghost_v2_finalized` generalises to a version-neutral flag
  (`render_driver.py:2877`, `:2946-2950`).
* **MF 12** -- `observability.prompt_slots` stays a **list of names**
  (`render_driver.py:2929-2930`); token counts go in a new `prompt_slot_tokens`
  map plus `prompt_dropped`, both added to the trace allowlist.
* **MF 4** -- one total `resolve_crux_kernel(meta)`: bounded key object plus
  setting; else the bounded brief; else a named refusal. `key_objects` and
  `setting` are both legitimately empty when the brief fails
  (`nodes/_otr_story_brief.py:419-432`), so `[0]` indexing is allowed nowhere.
* **MF 13** -- the test matrix grows the failure boundaries. Ordinary frozen v2
  replay keeps asserting no LLM and identical request hashes; the v3 case is a
  sibling test, never a weakening of that one.
* **CUT 1 and CUT 2** -- both accepted. No local VLM subject-hit score (no
  installed model, no callable, and the operator's eye is the verdict), and no
  "location word" classifier (exact normalized containment of the resolved
  setting phrase instead).

## 6. Deferred to Half B, with the reviewer's reasoning kept

MF 1 (a v3 author version and version-dispatched validators), MF 2 (a re-author
path inside the replay branch, which returns before any authoring --
`nodes/otr_shot_lock.py:3092-3124`), MF 6 (a v3 spec and its own hash key set;
v2 hashes thirteen keys including `motif_cue`,
`ghost_signal_author.py:754-758`), MF 7 (subject coverage -- the validator today
checks lettering, abstraction, cast names and human words but never whether the
clause draws the subject, `:902-945`), MF 8 (v3 mode constants, bookend keys and
an explicit `allow_people` flag rather than a blanket `_HUMAN_WORDS`).

None of these are wrong. They are all consequences of changing the stored
object, and Half A does not change it.

## 7. Should-fix disposition

* SF 1 (episode context in the batch header, not per row) -- **accepted**, Half B.
* SF 2 (the plate's protected head vs the crux) -- **accepted**, and it belongs
  to Half A: `compose_plate_prompt` has its own protected head and
  `PLATE_DROP_ORDER`, so the kernel's seat and its receipt are specified there
  explicitly.
* SF 3 (name the distinctness algorithm or drop it) -- **dropped**, with CUT 1.
  Mid-frame MAD from the existing probe report plus the operator's eye.
* SF 4 (no node signature changes; prove the canonical workflow) -- **accepted**.
  Half A changes no `INPUT_TYPES`, no widget and no link, and the JSON
  round-trip, the validator and the link/widget audits run anyway.

## 8. Revised contract (supersedes R1-R10 where they disagree)

* **V1** Half A is render-time only: `resolve_crux_kernel` +
  `compose_ghost_prompt_v3` + a v3 finalizer. No ledger field is added, no
  author output changes, no `crux_subject`, no bundle flag.
* **V2** The kernel comes from `key_objects` and `story_brief_terms.setting`,
  never from anything inside the `brief_hash` input.
* **V3** The costume motif leaves the prompt. The stored `motif_cue` is neither
  deleted from the object nor renamed -- v3 simply does not compose it.
* **V4** The composer stays pure and scalar; the finalizer fits by dropping
  whole optional units and never touches the kernel, the leaf or the law.
* **V5** Slots are re-measured with the installed tokenizer per pack, the
  cue-less default pack included, before any composition constant is chosen.
* **V6** The A/B is one frozen episode replayed twice: same request hash, same
  seed, two prompts. "The Faded Ledger" is the candidate, because its story is
  about objects the lane never drew.
