# Adaptation fidelity: make `shakespeare` and `public_domain` true to source

Status: PLAN, nothing built yet. Driver: Claude (Cowork). Date: 2026-08-03.
Branch: v2.0-alpha.

## Operator rulings (govern; not up for debate)

1. "public_domain and shakespeare don't need any added narrative -- they should be
   true to source."
2. "shakespeare should be a real act from a shakespeare play."
3. "I'm open to multiple characters" -- when a source scene needs more speakers than the
   cast-size widget requests, THE SCENE WINS. Raise the cast; never drop a character.

## What actually happened (two live episodes, 2026-08-03, ledgers on disk)

**Episode A -- `shakespeare`.** Drew Folger *As You Like It* 3.2 ("Rosalind tests Orlando").
Delivered: announcer opens *"In the heart of Verona ... Romeo Montague, lurking in the
shadows of Capulet's garden."* Stamped logline: *"Romeo sneaks into Capulet's garden to woo
Rosaline, but is interrupted by Celia."* Wrong play, wrong city, wrong lovers.

**Episode B -- `public_domain`.** Drew H.G. Wells *The Time Machine* ("arrival").
Delivered: *"the quaint town of Arkham, Massachusetts"* -- Lovecraft's town, a different
author -- and the machine shrunk to "a pocket watch". `arc_shape` stamped "heist".

## Root causes, VERIFIED against the real files

### RC1 -- The source text on disk is a stub, and one of them is fabricated

- `config/source_banks/shakespeare/fixtures/as_you_like_it_act3_scene2_rosalind_orlando.txt`
  is **110 words** of a ~4,000-word scene. Its lines are real Shakespeare but **collaged and
  misattributed**: Celia's "O wonderful, wonderful..." is given to ROSALIND, and Celia's
  Act 4.1 line is imported into "3.2" and rewritten ("misused our hours" for "misused our
  sex in your love-prate"). The fixture itself is not a faithful scene.
- `config/source_banks/public_domain_story/fixtures/time_machine_arrival.txt`
  is **145 words containing ZERO sentences of H.G. Wells** -- invented modern summary prose
  ("lamplight, nervous chairs, and men trying to laugh away a machine they could not
  explain") wrapped in genuine `*** START/END OF THE PROJECT GUTENBERG EBOOK ***` markers so
  it reads as an authentic excerpt. The bank then stamps a source-attribution coda crediting
  Wells for words Wells never wrote. **This is a false-attribution defect, not just a
  quality defect**, and it is the most urgent item in this document.
- Both bank rows point at `*.sample.json` manifests (`nodes/story_packs/banks.json:107,143`)
  and both packs are `"status": "ready_fixture"`. The lane shipped on placeholder data; the
  two failures were honest renders of fabricated inputs.

### RC2 -- No pass that writes spoken words ever sees the source

- The interpreter is the ONLY LLM that sees the stub verbatim
  (`_otr_shakespeare_sources.py:489` and `_otr_public_domain_sources.py:474`, `full_text[:5000]`).
  It compresses to a 2-3 sentence `script_brief` + `casting_brief` + `key_terms`.
- The outline macro receives `script_brief` only; the 1,200-char `seed_text` excerpt built at
  `_otr_shakespeare_sources.py:305-331` reaches the outline ONLY on the degraded
  interpreter-failed path (`_otr_outline.py:562-568`).
- `LineRequest` (`_otr_line_composer.py:193ff`) and `compose_exchange`
  (`_otr_compose_exchange.py:550ff`) have **no source-text field at all**.
- Meanwhile the pack system prompts order the model to "Ground every character line in the
  scene text" and "Where the source gives these characters words, CARRY THEM"
  (`nodes/story_packs/shakespeare/folger_scene_adaptation.json:12-13`). The model is
  commanded to quote a document it is never shown, so it fills the vacuum from pretraining
  gist: Shakespeare+lovers+garden collapses to Romeo/Verona; Victorian+uncanny-machine
  collapses to Lovecraft/Arkham.
- Consequence: the delivered "Shakespeare" lines are **pseudo-Elizabethan pastiche**, not
  quotation. They are not in the fixture and not in the AYL scene.

### RC3 -- Essential cast dropped by prefix truncation

`cast_hints` for AYL 3.2 are `['Rosalind','Celia','Orlando']`. `assemble_pre_locked_rows`
(`_otr_casting.py:1289-1310`) fills `remaining_open` slots from the FRONT of that list, so at
`num_characters=2` **Orlando** -- the scene's other half -- was dropped, and the writer
substituted the most famous Shakespeare lover it knew. `folger-macbeth:act1-scene3-witches`
(`['Macbeth','Banquo','First Witch']`) loses the WITCH at 2: "the prophecy on the heath" with
nobody to prophesy. Ruling 3 settles the policy: raise the cast.

### RC4 -- Content-blind rolls contaminate a fidelity lane

- **Story contract / sound world.** `select_style`'s adaptation branch
  (`_otr_style_catalog.py:846-856`) hashes the cast seed into a 4-item pool, blind to the
  source. Both the Arden forest scene and the Wells parlor drew `candlelit_period_chamber`
  -- "a fire in the grate, a mantel clock, a teacup" (`:456-459`), a domestic interior.
- **arc_shape.** `pick_arc_shape` (`_otr_dramatic_state_llm.py:60-85`), seeded from
  `OTR_STYLE_SEED` + news hash, stamped "heist" on *The Time Machine* and "betrayal" on a
  courtship comedy. Called at `OTR_LedgerScriptWriter.py:4300-4320`; steers `dramatic_state`
  and therefore the writer.
- **visual_style.** Every scene declares `"visual_style_policy": "derive_from_source"`.
  The field is schema-required, validated, and carried into `source_meta`
  (`_otr_public_domain_sources.py:403`) -- and **nothing consumes it**. That is why a Folger
  comedy rendered as `archival_documentary`. `nodes/visual_styles/shakespeare_stage_realism.json`
  exists and is unused. `recommended_word_budget` is dead the same way.
  Note: `visual_style` does NOT reach the writer prompt, so this is a look defect only --
  it did not corrupt the script.

### RC5 -- The announcer is structurally unable to be faithful

`compose_announcer_intro` receives only `SETTING:` and `TIME:` scraped from the
(already-drifted) outline macro. The `HOOK:` line is always empty -- the code reads a `hook`
attribute `SafeOpenBrief` does not have -- and `opening_status_quo`, `cast` and `era` are
captured but never rendered (`_otr_line_composer.py:178-190` vs `1174-1176`). The safe-intro
system prompt says "Use ONLY the proper names in the cast list below" and **no cast list is
ever below** (`folger_scene_adaptation.json:16`). The announcer never learns the play, the
author, the act, or the place. This affects EVERY scaffold-on bank, not only fidelity lanes.

## The plan

Ordered cheapest/highest-impact first. Nothing downstream can be faithful until F1 lands.

**F1. Put real source text on disk. (DATA + trivial code)**
Replace both fixtures with genuine text: Folger AYL 3.2 (CC BY-NC, already stamped
noncommercial by the bank) and real Gutenberg Wells (1895, US public domain). Raise the
`canonicalize_*` `max_chars` and the interpreter's `[:5000]` window to hold a full scene
(~22k chars). Rights are not the obstacle here; the placeholder was.
**Until this lands, `public_domain` is making a false attribution to Wells on every render.**

**F2. Anchor the briefs. (PROMPT-ONLY)**
Harden the interpreter prompts (`_otr_shakespeare_sources.py:492-512`,
`_otr_public_domain_sources.py:478-503`) so `script_brief` MUST begin with work, act/scene and
place -- "As You Like It, Act 3 Scene 2, in the Forest of Arden: ..." -- and `key_terms` MUST
include the place name and the scene's own device. This anchors every paraphrase-only
consumer immediately, before any plumbing changes.

**F3. Announcer SOURCE LOCK + allowlist gate + templated fallback. (PROMPT + small code)**
Give the announcer a SOURCE LOCK block (WORK / ACT+SCENE / PLACE / PERIOD / PEOPLE) and forbid
any proper noun not in it. Do not rely on prompt obedience: extend `validate_announcer_line`
with a proper-noun allowlist built from manifest fields + locked cast; on violation fall back
to the deterministic template that already exists (`fallback_announcer_intro`,
`_otr_line_composer.py:1106`), upgraded for fidelity lanes to
`"Good evening. This is SIGNAL LOST. Tonight: {play_title}, Act {act}, Scene {scene} -- {scene_label}."`
A manifest-templated intro is 100% drift-proof and period-authentic. Also fix the `hook`
attribute mismatch and actually render `cast`/`era`.

**F4. Thread verbatim source into the outline macro. (small code)**
The payload already carries `full_text`; it dies one hop early. Add
"SOURCE SCENE (verbatim -- adapt this, invent nothing)" to the outline user prompt.

**F5. Give the line/exchange composer the scene text. (the substantive code change)**
This is where spoken words are authored -- the only change that turns pseudo-Shakespeare into
Shakespeare. Whole scene, or per-phase slice for context budget.

**F6. Essential cast: the scene sets the floor. (code)**
Per ruling 3, lock every character the scene requires even when that exceeds the requested
cast size, with a ledger receipt distinguishing "raised by the source" from a random roll.
OPEN: exact mechanism (explicit `essential_cast` field vs. treating all `cast_hints` as
required) and the interaction with the "cast voice coverage / no SAYABLE line" guard at low
word targets -- raising the cast must not trade a mashup for a crash. Under review.

**F7. Source-derived sound world; pin arc_shape. (DATA + small code)**
Keep ONE adaptation contract (`faithful_stage_adaptation`, whose engine and ending text are
correct for any source) and take `sound_world` per scene from a new curated manifest field --
14 scenes, one line each ("wind in the oaks of Arden, birdsong, paper pinned to bark").
On fidelity lanes the arc shape is the source's own arc: pin it rather than rolling dice.
CAUTION: bank `original` is scaffold-off so it can INVENT FREELY; these lanes would be gated
so they invent NOTHING. Opposite intents -- do NOT silently share one switch without checking
each consumer.

**F8. Deterministic attribution coda. (small code)**
Compose the source note from manifest fields (title/author/act/scene/label/license) instead of
an LLM-authored `news_close_brief`. Attribution is a fact, not a generation.

**F9. Wire `visual_style_policy: derive_from_source`. (code)**
Decide what it concretely means at the visual-style selection site and implement it, or delete
the field. A schema-required field with no consumer is a lie in the manifest.

## Open questions for the panel

- **Q1 (operator decision, surfaced not guessed).** "A real act" collides with length: the
  manifest validator hard-caps `recommended_word_budget` at 320 words
  (`_otr_shakespeare_sources.py:174-176`), but AYL 3.2 alone is ~4,000 words and a full act is
  several times that. Does "a real act" mean a long-form episode (15-25 min), or one full
  SCENE carried at length? Everything downstream (beat count, VRAM, render time) keys off this.
- **Q2.** Do the fidelity lanes need their OWN pipeline, or is `legacy_many_pass_adapt`
  (shared with generic adaptation) the right shape for verbatim-faithful work?
- **Q3.** Is a validation pass warranted -- a check that the delivered script contains no
  place/person absent from the source? What does it key on, and does it regenerate or halt?
  Project ethos is fail-loud, no silent fallbacks.
- **Q4.** `public_domain` is a one-book lane: one source, one unit, pinned default
  `source_ref`. Every episode is *The Time Machine* "arrival" until the manifest grows.
  Is that in scope now?
- **Q5.** Ledger ownership. Every new field needs exactly ONE producer and a stated default;
  downstream consumers (TTS, per-beat audio slicing, video/shot direction, captions, credits,
  `obs_publish`) read FIELDS, not intentions. Name the owner for each field F1-F9 introduces.

## Constraints

100% local/offline, no paid APIs. Local models: Mistral-Nemo-Instruct-2407 and gemma-4-12b --
extra LLM passes cost VRAM and wall-clock, not money. 16 GB VRAM ceiling, 14.5 GB real-world
target. Say which proposals need a new pass versus prompt-only versus deterministic Python.
