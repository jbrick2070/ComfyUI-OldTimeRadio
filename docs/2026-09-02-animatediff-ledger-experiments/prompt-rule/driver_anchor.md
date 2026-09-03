# Driver anchor -- Prompt v3 for the AnimateDiff lane: draw the crux (campaign item 3, 2026-09-02)

The source is the operator's own hand: `operator_rewrites.md` beside this file (ten beats
rewritten against the prompts the lane actually sent, plus nine rules in his words). This
anchor is the driver's code-grounded position BEFORE the panel. Roster: Fable 5.1 cold read
(creative, in flight at the time of writing) + Antigravity r1, Codex r2, Cursor r3, Sonnet r4.

**SCOPE, widened by the operator twice on 2026-09-02 night.** (a) *"once you are done with
this it's worth reviewing ALL other lane video prompts too, to see we aren't stuffing too much
character description; we should have real action and motion that applies to the action at
hand -- unless it is a TTS speech video that can't handle it."* So the rule is ONE rule across
the lanes, and the exception is named: an audio-driven talking-face lane (`ltx_audio_in`,
`minimax_h3_audio_in`) is driven by the voice track and a moving subject would fight the
lip-sync, so those keep their portrait framing. (b) *"obviously AnimateDiff can't take that
much text, other video models can, so maybe it's OK they have more words."* So the BUDGET is
per lane, not global: this anchor's 69/77 arithmetic is SD1.5's CLIP window and binds the
AnimateDiff lane alone. The LTX / Wan / HuMo / cloud lanes carry their own, larger budgets and
may spend them; what transfers to them is the SHAPE (crux, action, world motion, no costume,
no framing boilerplate), not the token ceiling. Lane-by-lane audit is item 3b, after v3 ships
and is judged on this lane.

**The operator's rule, compressed:** keep it simple; every beat's prompt is the VISUAL STYLE
mixed with the CRUX of the story, in the story's one setting, with the world's own motion --
whether or not that beat's line names the crux. No coat, no figure, no prop the dialogue never
mentions ("unless the story is about a stolen coat and bag"). Fewer variables per prompt. Get
creative. And, on naming: the sentence in the ledger is the STORY BRIEF (`meta.story_brief` +
`meta.story_brief_terms`); the "Meta-Brief" is node 89, `OTR_MetaBriefImagePromptGen`, which
builds the STILL prompts -- two different things, named exactly from here on.

## 1. What the lane does today, at the file

* The composed prompt is `<pack_cue>. <motif>, <leaf>, <law>` -- `GHOST_V2_SLOTS`
  (`ghost_signal_prompt.py:824`), `compose_ghost_prompt_v2` (`:827`).
* `pack_cue` = `compact_style_cue(vstyle)` = two words (`_otr_visual_styles.py:632-653`):
  "storybook engraving", "anime style", "video-art feedback style", "recursive fractal light
  field". The pack authors 80-671 chars; the lane sends 11-29 of them.
* `motif` for a character beat = `motif_for_character` (`ghost_signal_author.py:644-675`):
  `"<silhouette> figure in <colour> <garment>, carrying <prop>"`, every token from a small
  allow-list scanned off the cast row with FALLBACK POOLS (`MOTIF_FALLBACK_POOLS` `:571-577`:
  garment pool leads with "coat"; prop pool: lantern, key, ledger, satchel, chart, telegraph).
  For the object/signal modes of the same character: `"<colour> <prop>"` -- the carried prop
  by itself (`:668-673`). That is the "humans and bags" machine: build + coat colour + carried
  prop, then the prop alone on a table, on every character beat of every episode.
* `leaf` = one LLM line per beat. The batch prompt (`build_batch_prompt` `:789-812`) tells the
  model `id`, `mode`, `motif`, and, when present, `happening=<sanitized_intent>`,
  `mood=`, `story=`. The module docstring (`:8-16`): handed no dialogue, no title, no cast
  prose, no names, BY CONSTRUCTION -- the anti-leak reason v2 exists. MEASURED on the episode
  under study (`the_last_reading_190630`, 31 lines, 27 beats): `beat_intent` is empty on
  every line row and absent from `beats[]`, so `happening=` was blank on all 27 and the model
  was told the mode and the costume and nothing else.
* `law` = `GHOST_MODE_LAWS_V2` (`ghost_signal_prompt.py:129-146`): figure "mid-shot or wider,
  whole figure legible, one clear action, unbroken shot"; object "the object fills the frame on
  a table top, one clear action, unbroken shot"; signal "the object lit against a dark room,
  moving light across it, unbroken shot". Fixed per mode. 43% of the words on the episode.
* The bookends: `GHOST_BOOKEND_MOTIFS` (`:578-583`) -- radio set, glowing dial, broadcast
  console, spinning turntable -- real hardware, kept.
* The leaf validator (`validate_drawable_beat` `:902-1000`) bans medium, camera and quality
  words, lettering words, names, and "a person in object mode"; `deterministic_leaf` (`:1048`)
  is the checked-in fallback after two failed batches.
* Budget: `GHOST_CLIP_WINDOW_TOKENS = 77`, `GHOST_AUTHOR_TOKEN_TARGET = 69`
  (`ghost_signal_author.py:124-129`), measured with the installed SD1 tokenizer;
  `GHOST_PROMPT_MAX_CHARS = 320` (`ghost_signal_prompt.py:71`, chosen for the banana route's
  headroom, not for the model). ComfyUI chunks past 77 rather than dropping; the module's own
  words: "77 is a salience choice, not a transport cliff" (`:35-39`).
* The 2026-08-22 stability finding that keeps the cue short compared the v2 motion module
  against v3 (a MODULE change) and was used to justify a PROMPT-LENGTH choice; nobody has yet
  measured a same-seed pair with a long clause against a short one on the same module
  (PROBLEM_STATEMENT.md:225-226, 279). The instrument now makes that pair cheap.

**What the ledger already carries, unused by this lane:**
* `meta.story_brief` -- ONE sentence, the crux: "A claustrophobic research station monitors a
  mysterious floating landmass that defies the laws of physics." Read today by the cloud video
  lanes, the cloud/Google image lanes, the Meta-Brief still node and the driver
  (`get_story_brief_full` / `get_story_brief_ltx`, `_otr_story_brief_helpers.py:93-131`) -- never
  by the Ghost composer.
* `meta.story_brief_terms` -- `setting` (subsurface research station, remote reservoir,
  industrial monitoring hub, isolated wilderness outpost), `lighting`, `atmosphere`. Read by the
  plate composer (item 2) and the still lanes; never by the video prompt.
* Each beat's spoken text (`lines[*].text` via `shot.source_line_ids`): the nouns of the beat
  (pressure, water, temperature, telemetry, superiors, paperwork).
* `lines[*].beat_intent` -- the outline's "what this beat accomplishes narratively"
  (`_otr_outline.py:99`, required, min length 1, "not dialogue"); copied to the line row by
  `production_ledger.py:1450` as `intent or None`. EMPTY on this episode's 31 lines
  (`story_scaffold_enabled: False`, bank `scifi_news_pro`). Whether the writer path that ran
  here never had intents or dropped them on the way to the lines is an OPEN diagnosis; v3 must
  not depend on the field being filled.

## 2. Decisions (the driver's position; the panel pressure-tests these)

**D1. Prompt v3 shape: `<style clause>. <crux>, <beat subject>, <world motion>`** -- four slots,
no framing law, no costume, no prop pool. The style clause is the pack's authored language
(`positive_tail`, at least; the medium is Python's job on this lane and stays so); the crux is
the story brief's subject, present on EVERY beat; the beat subject is what THIS line is about
(its nouns, from the spoken text, or the crux again when the line has none); the world motion
is the one authored action, and it belongs to the world (drift, sweep, dim, flicker, rise), not
to a figure. Order is the operator's ("mix the visual style with the crux"): style first,
crux second, the beat's specifics last -- the salience order of a CLIP window.

**D2. The character disappears unless the story is about a person.** No `motif_for_character`
on v3. A character beat draws what the character is looking at, handling or standing in: the
instruments, the water, the paperwork, the console -- "characters moving through" the world
only when the line's own nouns are people (superiors, a crowd, a crew). The cast look, the
silhouette, the colour, the garment and the carried prop are gone from the prompt. Identity
recurrence, which the costume motif existed to carry, moves to the CRUX: the same subject in
the same setting on every beat IS the recurrence, and it is the story's, not a wardrobe's.

**D3. Deterministic first, LLM second.** v3's composer builds the whole prompt in Python from
the ledger (brief + terms + the beat's nouns + the pack), so a prompt exists for every beat
with no model call and no fallback pool. The LLM leaf becomes an OPTIONAL enrichment of the
world-motion slot only: told the crux, the setting terms, the beat's nouns and the mode, asked
for one clause of world motion, validated as today (no names, no lettering, no medium words,
no person unless the nouns carry one), replaced by the Python clause on any failure. This
inverts v2 (where the LLM line was the only non-fixed slot) and keeps the anti-leak invariant:
the model still sees no dialogue text, only its extracted nouns and the brief.

**D4. The beat's nouns, extracted not copied.** A small deterministic pass over the beat's
spoken lines: nouns and noun phrases through the SAME allow-list discipline the motif uses
(`_first_allowlisted`'s whole-word rule), stripped of cast names (`strip_cast_names`), numbers
kept as words only when they name a thing ("three degrees" -> "cold"), capped at three
phrases. A beat with no usable nouns (a bare "Yes." or a music bed) takes the crux alone. The
pass is pure and pinned by tests over the episode's 31 lines.

**D5. One setting per episode, threaded.** The first two `story_brief_terms.setting` entries
(the plate composer's `_read_setting` rule) ride every beat as part of the crux clause. Bookends
keep their hardware motif (`GHOST_BOOKEND_MOTIFS`) but gain the setting behind it -- the
operator's own rewrite: "a bakelite radio set, with a background of the reservoir".

**D6. The framing law is CUT, except a three-word floor per mode.** "no people" for the
music bed and "no faces" for the figure beats stay as the only law words (the face floor is the
one thing the sliding context genuinely cannot hold, `GHOST_FRAMING`'s reason); "unbroken shot"
stays as the last two words of every prompt (the one clause that measured as buying stability).
Everything else -- "mid-shot or wider, whole figure legible, one clear action", "the object
fills the frame on a table top", "lit against a dark room, moving light across it" -- goes.
Medium words never enter a law (the 08-22 "real" lesson stands).

**D7. Budget: one measured window, the char ceiling moved.** The finalizer's 77-token /
one-window gate stays and is the real limit; `GHOST_PROMPT_MAX_CHARS` is raised from 320 to a
value that no longer truncates any pack's positive_tail plus the crux (measure the nine packs;
recur_frac's tail alone is 227 chars) -- the banana route's headroom is re-derived from the new
ceiling, not the other way round. Drop order under the window: the beat's third noun phrase,
its second, the setting's second term, the pack's tail after its first clause; never the crux,
never the first clause of the style.

**D8. The receipt records the four slots.** `observability.prompt_slots` becomes the v3 four;
`prompt_version` becomes `ghost_signal_v3`; the prompt profile capability is unchanged so no
engine is unregistered; the shipping haunted lane and the still-in peer both render v3 (same
prompt, both lanes -- the peer's A/B isolates the plate, unchanged).

**D9. The proof.** Same-seed pairs through the replay instrument: one bundle frozen from a v2
render, replayed with v3 prompts on the same lane (a replay-time prompt re-composition switch,
sibling of the engine override: prompts are re-composed from the frozen ledger, seeds and beats
untouched) -- so the operator sees v2 and v3 of the SAME episode, the SAME seeds, on
storybook_engraving, then video_art and recur_frac. His eye is the verdict ("does it draw the
story, in the style, without the coats"); the motion-energy band from the probe runner says
whether the shorter, subject-led prompts moved more or less than the costume prompts did.

**D10. Not in this item.** Filling `beat_intent` on the writer side (a writer/outline item, its
own diagnosis first); per-style checkpoints; any change to the plate composer (item 2 already
reads the setting terms; it gains the crux clause in the same shape when v3 lands, one line).

## 3. Open questions for the panel (r1)

* Q1. D2 is the sharpest call: the cast look leaves the video prompt entirely. Is there a beat
  shape where a person MUST be drawn (a lone announcer at a microphone?) and how is that said
  without a face or a coat?
* Q2. D3 inverts the v2 authority: Python owns the prompt, the LLM decorates one slot. Does the
  creative ask ("get really creative") survive that, or should the LLM own the beat-subject slot
  too, told the crux and the nouns?
* Q3. D4's noun extraction: an allow-list of thousands of nouns is not the same as a vocabulary
  of forty motif words. What is the honest gate against lettering and names -- a stop-list plus
  the existing validator, or a part-of-speech pass?
* Q4. D6 keeps eight words of law. Zero?
* Q5. D9's replay-time re-composition: a second replay override on the instrument. Same seat
  as the engine override (ShotLock's replay branch), or a driver-side switch, since prompts are
  finalized in `build_request_from_shot`?
* Q6. Where is the driver about to go wrong?

## 5. r1 fold, part two: Antigravity (`agy`, Gemini 3.8 Flash (High)) -- five must-fixes, all grounded, all taken, three of them REVERSING the anchor

Judgment with the disposition of every claim: `kibitz-runs/2026-09-02-prompt-v3-crux/r1/judgment.md`.

**The arithmetic that settles the shape.** Measured on the real packs and the real brief
(4 chars/token): recur_frac's FULL positive_tail is ~58 tokens and the brief sentence ~28, so
the anchor's "send the pack's authored language plus the crux" is 86 tokens on the operator's
favourite style -- past the 69 author gate and the 77 render gate, and `assert_shell_fits`
(`ghost_signal_author.py:1409`) would refuse at boot. Antigravity's fix (go back to
`compact_style_cue`) is the defect this campaign exists to remove. **Neither.** The style keeps
the compact FRONT cue (4-9 tokens) and gains a 3-5 word TAIL from the pack's own language
(8-12 tokens); the CRUX is a compact SUBJECT KERNEL -- `key_objects[0]` in `setting[0]`,
<= 15 tokens -- never the brief sentence verbatim. Budget per beat: style 12-20, crux 15,
beat subject 12, world motion 12-18. Inside 69 on all nine packs, pinned by a boot test.

**"no people" / "no faces" may never appear in a positive prompt.** CONFIRMED as a standing
law at `ghost_signal_prompt.py:109-112`: *"There is no `no people` here and there never will
be ... a positive clause that attends to an absent human is a request for the model to think
about one."* The anchor's D6 floor broke it. **D6 becomes ZERO positive law words**;
exclusions stay in `compose_ghost_negative`. "unbroken shot" is held in reserve for the flicker
remeasure only.

**The character does not vanish -- the COSTUME does.** Antigravity reads the operator better
than the anchor did here: his rule 1 is *"characters moving through that world -- small, in
it, never a coat in close-up"*, and his own rewrite of Sarah's beat is *"characters moving
through a stagnant mass of water at a reservoir"*. **D2 is corrected:** the person is never
the SUBJECT and never a wardrobe, but the world is not emptied of people. The grammar is
Fable's: a `hand` vantage draws one hand or a turned back on the story's thing; a `world`
vantage may carry distant plural shapes. `_HUMAN_WORDS` (`:346-355`) already bans
figure/man/woman/silhouette/pronouns while permitting hand, arm, shoulder, back -- v3 applies
that list to every vantage.

**Dialogue nouns are conversational.** "What did you find?" yields nothing drawable, so the
anchor's D4 would collapse to bare crux repetition on the talkiest beats. **D4 is REPLACED**
by `meta.key_objects` (measured present on the episode: handheld brass communicator, telemetry
screens, data logs, hydrographic charts), selected by the beat's stripped nouns and otherwise
rotated by the scheduler. No open-vocabulary scraper, no POS pass -- which also closes the
anchor's Q3.

**A replay may not have its prompts mutated.** CONFIRMED: `render_request_hash` and
`request_sha256` are the admission and cache keys and the replay branch is contracted
immutable. **D9 is REWRITTEN:** the v2-vs-v3 comparison is a FRESH run pinned to the frozen
seeds -- a derived bundle that re-authors prompts and stamps `prompt_version=ghost_signal_v3`
with its own hashes while carrying `seed_bundle.request_seed` from the frozen plan, the exact
sibling of the engine-override bundle item 2 already built. Q5 answered.

**Also taken:** all drop-order logic lives at COMPOSITION time, because
`finalize_ghost_prompt_v2` is contracted "IT NEVER TRIMS AND NEVER REPAIRS" (`:1306-1308`);
the setting term is appended only when the crux kernel lacks a location word (the brief already
says "research station"); per-slot token counts ride `observability.prompt_slots`. On the one
place the reviewers disagree -- who composes the sentence -- Fable's split holds (Python owns
the crux, vantage, light and tail; the model owns one 8-12 word motion clause) with one
concession to Antigravity: the model may NAME the beat's subject from `key_objects` inside its
clause, so it writes "the hydrographic charts sliding off the desk into the water" rather than
being handed the noun. That is where the creativity the operator asked for lives.

## 4. r1 fold, part one: Fable 5.1 cold read (creative; full text in `kibitz-runs/2026-09-02-prompt-v3-crux/r1/`)

Every decision-changing claim below was checked at the file before folding.

**The ledger is richer than the anchor said, and that changes D4.** CONFIRMED on the studied
episode: `meta.key_objects = ["handheld brass communicator", "telemetry screens", "data logs",
"hydrographic charts"]` and `meta.visual_palette = ["slate gray", "oxidized copper", "brushed
steel", "digital amber"]`. The story brief pass already names the story's THINGS and its
colours. So the beat subject does not need a noun-extraction pass over dialogue as its primary
source (anchor D4): it comes from `key_objects`, with the beat's own nouns only as the
selector among them. That removes Q3's whole problem -- no thousand-word allow-list, no
part-of-speech pass -- and it is a strictly safer read: `key_objects` was authored by the brief
model with no names and no lettering in it. **D4 REPLACED.** `crux_subject` is absent today;
Fable's proposal to have the brief pass author it (flat additive, beside `key_objects`) is
TAKEN as the first choice, with `"<key_objects[0]> in <setting[0]>"` as the deterministic
fallback so v3 never depends on a new brief field landing first.

**The seed trap, CONFIRMED and load-bearing.** `otr_shot_lock.py:1326`:
`brief_hash = _content_hash(meta.get("story_brief_terms") or meta.get("story_brief") or {})`,
and that hash feeds `render_request_hash`, which seeds every shot. Putting `crux` or the beat
nouns INSIDE `story_brief_terms` would silently reseed every beat and destroy the same-seed
A/B the whole proof rests on. New fields go TOP-LEVEL in `meta`, beside `key_objects`.
Added to the coding contract as a refusal-level rule.

**The five-slot shape** (`<pack cue>. <crux in its setting>, <world motion>, <vantage + one
lighting term>, <pack medium tail>`) is TAKEN over the anchor's four: the tail returns 3-5
words of the pack's own language at the end, which is the operator's "wrapper at either end"
(his rewrites 1 and 2 put the style at both ends), and it lifts the style's share from 11% to
about 20% without a second CLIP window. `prefix_style_cue` is idempotent
(`_otr_visual_styles.py:656`) so the composer may emit the cue itself. Budget correction,
CONFIRMED: the author gate is 69 MEASURED tokens (`ghost_signal_author.py:129`), roughly 35-40
English words -- the anchor's "~60 words" was wrong; every worked prompt Fable wrote is 31-37.

**The mode scheduler is reused, not replaced.** `figure / object / signal` becomes
`world / thing / world / hand` -- same `schedule_ghost_modes` machinery, same anti-run
guarantee (`:472-520`), new vocabulary. This is what threads one subject across 29 beats
without 29 identical pictures: the crux and the setting are byte-identical every beat (that IS
the recurrence, replacing the costume motif), while vantage, scale, the world's motion and one
lighting term vary on a deterministic cycle.

**People: the `hand` vantage.** CONFIRMED that `_HUMAN_WORDS` (`:346-355`) already bans
figure / man / woman / silhouette / pronouns while permitting hand, arm, shoulder and back
(the clock-hand lesson). v3 applies that list to EVERY vantage, not two, and a person-beat
draws one hand or a turned back on the story's thing -- never a face, never clothing. Plural
distant shapes ("two small shapes on the gantry above the water") are allowed when the story's
own nouns carry people, which is exactly the operator's "unless the story is about it" clause.
This answers Q1: the person is never the subject; a part of them acting on the crux is.

**The framing law: ZERO words.** Fable's argument is stronger than the anchor's eight-word
floor and is TAKEN: today's law is not neutral framing but CONTENT -- "whole figure legible"
REQUESTS a figure and "on a table top" is where every satchel came from
(`ghost_signal_prompt.py:129-147`). Vantage words are per-beat and live in their own slot.
"unbroken shot" is held in reserve: if the A/A pair shows flicker rising without it, those two
words come back alone and are remeasured. One variable at a time. Q4 answered.

**The measurement (Q7 / D9).** Subject hit rate: a local VLM judge names the main subject of
each clip's middle frame; a hit is a match against the crux, the setting or `key_objects`.
v2 scores ~0 on this episode by construction, since coats and satchels appear in no ledger
field. Distinctness second: mean pairwise perceptual-hash distance between mid-frames, v2 vs
v3, same seeds. Both cite `prompt_sha8` off `meta.render_trace`, which the instrument now
carries.

**Two more traps, both TAKEN as contract rules.** (1) Swapping `motif_for_character` to return
the crux while `GHOST_MODE_LAWS_V2` and `MOTIF_FALLBACK_POOLS` survive leaves "on a table top"
and the prop cycle in the prompt -- the same pictures with a reservoir word in front; the pools
and the laws are DELETED in the same change. (2) Feeding the dialogue, or a regex slice of it,
as `happening=` is v1's copy defect reborn (module docstring `:3-7`) and would draw "three
degrees" as lettering; the model receives `key_objects`, the setting, the palette and the
beat's stripped nouns, never the line.

**Hash and version discipline:** `GHOST_REQUEST_HASH_KEYS` (`:754`) gains the crux and the
nouns and `GHOST_AUTHOR_VERSION` (`:67`) bumps, or a stored v2 satchel leaf replays under v3
(replay skips validation). The composed-prompt version becomes `ghost_signal_v3`; the prompt
PROFILE capability string does not move, so no engine is unregistered.

Still open for the code panel (r2): whether the LLM owns the beat-subject slot as well as the
world motion (Q2 -- Fable keeps Python owning crux, vantage and tail, and gives the model one
8-12 word world-motion clause, which the driver accepts); and the replay-time re-composition
seat (Q5).

## 6. Revised decisions after r1 (what r2 plans the code for -- this list governs sections 2-5)

The operator's one-line version, after reading round one: *"yes, I want to see more STORY,
not just the characters."*

**R1. Shape, five slots, in this order:**
`<compact style cue>. <crux kernel>, <beat subject + world motion>, <vantage, one light term>, <pack tail>`
The cue is the existing `compact_style_cue` (4-9 tokens); the tail is 3-5 words lifted from
the pack's `positive_tail` / `image_grade_tail` (8-12 tokens), so the style opens and closes
the prompt without a second CLIP window. Budget per beat, measured with the installed
tokenizer at composition time: cue 4-9, kernel <= 15, subject + motion 20-30, vantage + light
6-8, tail 8-12; the author target 69 and the render gate 77 are untouched; a boot test pins
every one of the nine packs inside 69 with the longest kernel the ledger schema allows.

**R2. The crux kernel** = `meta.crux_subject` when the brief pass supplies it (a new flat
additive field, "the one drawable thing this story is about", authored beside `key_objects`),
else `"<key_objects[0]> in <setting[0]>"`; the setting term is dropped when the kernel already
carries a location word. Identical bytes on every beat of the episode: that IS the recurrence.

**R3. The beat subject** = one of `meta.key_objects`, selected by the beat's stripped nouns
(cast names removed, digits removed, whole-word match against the objects' own words), else
the scheduler's rotation through `key_objects` so a dialogue-only beat still varies. Never an
open-vocabulary scrape of dialogue; never the cast look, silhouette, garment or carried prop.
`motif_for_character`, `MOTIF_FALLBACK_POOLS`, `MOTIF_PROP_WORDS`' carried-prop use and
`GHOST_MODE_LAWS_V2` are DELETED in the same change, not bypassed.

**R4. The world motion** is one 8-12 word clause. The LLM writes it in the existing batch
(told the crux kernel, the setting terms, the palette, `key_objects`, the beat's stripped nouns
and its vantage -- never the line, never a name), and may name the beat's subject inside it;
`validate_drawable_beat` gates it as today plus the `_HUMAN_WORDS` rule on EVERY vantage; the
deterministic fallback composes `<subject> <world-motion verb phrase from a checked-in pool,
seed-selected with the existing collision probing>`. `GHOST_FALLBACK_CLAUSES`' table-top pool
is replaced by a world-motion pool.

**R5. Vantage** reuses `schedule_ghost_modes` unchanged with the vocabulary
`world / thing / world / hand` (the anti-run guarantee kept): `world` = the setting wide,
`thing` = one of the story's objects large with the world behind it, `hand` = one hand or a
turned back acting on the thing. People are never the subject and never clothed; distant
plural shapes are allowed when the story's own nouns carry people. One `lighting` term rotates
per beat. Bookends keep `GHOST_BOOKEND_MOTIFS` with the setting behind the hardware.

**R6. Zero positive law words.** Exclusions live in `compose_ghost_negative` only
(`ghost_signal_prompt.py:109-112` is law). "unbroken shot" is held in reserve for the flicker
remeasure and is not in the first build.

**R7. All drop logic at composition time**, in the composer, never in
`finalize_ghost_prompt_v2` (contracted never to trim). Drop order under the window: the second
light term, the vantage qualifier, the subject's second word, the tail's last words -- never
the kernel, never the cue.

**R8. Receipts and hashes.** `GHOST_V3_SLOTS` (five), `prompt_version = ghost_signal_v3`,
`GHOST_AUTHOR_VERSION` bumped, `GHOST_REQUEST_HASH_KEYS` gains the crux kernel and the beat
nouns, `observability.prompt_slots` carries per-slot token counts; the prompt-profile
capability string is unchanged so no engine is unregistered. New meta fields are TOP-LEVEL in
`meta` (never inside `story_brief_terms`, which seeds every shot through
`otr_shot_lock.py:1326`).

**R9. The proof.** A DERIVED bundle (`--derive-prompt-version ghost_signal_v3`, the sibling of
the engine-override bundle) re-authors prompts from the frozen ledger with new hashes while
carrying each shot's `request_seed` from the frozen plan; the replay branch stays immutable.
One episode, same seeds, v2 beside v3, on storybook_engraving first (the episode he marked
up), then video_art and recur_frac; his eye is the verdict, the VLM subject-hit rate and the
mid-frame distinctness are the numbers, every score citing `prompt_sha8`. THIS is the first GPU
run after the freeze, and nothing renders before it.

**R10. Scope.** Per-lane budgets (this arithmetic is SD1.5's); the other video lanes get the
SHAPE in item 3b after v3 is judged, with the audio-driven talking-face lanes exempt. Filling
`beat_intent` is a writer-side diagnosis, separate. The plate composer (item 2) gains the crux
kernel in the same shape when v3 lands.

## 7. What r2 (coding plan, Codex) must settle

* Where the composer lives: `compose_ghost_prompt_v3` beside v2 in the PURE module, with the
  tokenizer-dependent drop loop in the author module (the purity contract of
  `ghost_signal_prompt.py`).
* How ShotLock hands the new fields to the author batch (the spec dict, its hash keys, the
  batch prompt text) and how `render_driver`'s Ghost branch finalizes v3 without touching the
  v1/v2 paths a frozen ledger may still carry.
* The brief pass edit for `crux_subject` (flat additive; the sha256 receipt on embedded packs
  must not move).
* The derived prompt-version bundle and its ShotLock seat.
* The test matrix: nine packs inside 69; the 31 lines of the studied episode through the noun
  selector; `_HUMAN_WORDS` on every vantage; the deterministic fallback over an empty
  `key_objects`; hash/version bumps; the receipts test for the five slots.

## 8. r2 fold (Codex) -- the build splits in two, and the cheap half is the one he wants

Full review: `kibitz-runs/2026-09-02-prompt-v3-crux/r2/codex.md`.
Driver judgment, with every claim ground against the real files:
`kibitz-runs/2026-09-02-prompt-v3-crux/r2/judgment.md`.

Codex returned thirteen must-fixes, four should-fixes and two cuts, and its
verdict was "not build-ready". Accepted, almost in full. Grounding its strongest
claim reversed the shape of the work in our favour.

**The pivot.** `finalize_ghost_prompt_v2` already takes `ledger_meta`, and the
render driver already calls it with the live ledger
(`ghost_signal_author.py:1295-1317`; `render_driver.py:2911-2915`). The prompt
is composed AT RENDER TIME from a small stored object plus the whole episode
meta -- so the story is already in the room, and nobody invited it in.

* **Half A -- render-time only.** Resolve the crux kernel from `key_objects`
  and `story_brief_terms.setting`, stop composing the costume motif, new slot
  order, drop-to-fit. No ledger field, no author change, no LLM, no bundle flag.
* **Half B -- plan-time authoring.** New leaf vocabulary, the beat's dialogue
  reaching the author, subject coverage. Changes the stored object, so it needs
  the version dispatch and the replay re-author path Codex specified.

Half A is built and proven first.

**The seed, settled.** The planned `request_seed` field is inert; the render
seed is `_seed_from_hash(render_request_hash, shot_id)`. And
`request_hash = _content_hash([brief_hash, cast_hash, beat_id, char_id])`
(`otr_shot_lock.py:1479-1487`) -- **the prompt is not in it.** The v2-vs-v3 A/B
is same-seed by construction, with no derivation machinery at all. The
corollary is a trap worth naming: the crux must NOT live inside
`story_brief_terms`, or `brief_hash` moves and every seed with it.

**The evidence that ends the argument.** "The Faded Ledger", published 21:08
tonight, is an episode about film canisters, archive shelves and security
badges in a high-security archive. The lane drew a black shawl, a charcoal coat
and a charcoal satchel -- three words that appear nowhere in the episode -- on
four of its beats, two of them a person holding a bag. The table is in the r2
judgment, section 3.

**Budget, corrected.** `compact_style_cue` is 2-4 words and EMPTY for
`sci_fi_radio` (`_otr_visual_styles.py:632-653`); the anchor's "4-9 tokens" was
wrong and the default pack has no cue slot at all. Every slot is re-measured
with the installed tokenizer before a composition constant is chosen. The good
news: the motif v3 deletes is longer than the kernel v3 adds, so v3 costs FEWER
tokens than v2 and the 77-token refusal is not the risk it looked like.

**The governing contract is now V1-V6 in the r2 judgment, section 8.** Where
R1-R10 (section 6) disagree with V1-V6, V1-V6 win.

## 9. What r3 (wiring, Cursor) must settle

1. **The render seam.** Half A changes `finalize_*` and the driver block at
   `render_driver.py:2870-2960`. Does anything else read
   `GHOST_V2_SLOTS`, `prompt_version`, `_ghost_v2_finalized` or the
   `ghost_*` observability keys -- the trace allowlist, the receipt's causal key
   set, `otr_stillin_probe_report.py`, the acceptance readers -- such that a v3
   prompt version breaks a consumer that was written to expect exactly "v2"?
2. **The receipt.** `actual_request_sha` covers the prompt. Under Half A it
   MOVES while `render_request_hash` does not. Confirm that is what the
   instrument's A/A rule expects, and that `scripts/otr_verify_replay.py` reads
   an A/B (same request hash, different actual sha) as a legitimate difference
   rather than a corruption.
3. **The still-in peer.** `ghost_plate_prompt.compose_plate_prompt` has its own
   protected head and `PLATE_DROP_ORDER`. Where does the kernel sit, is it
   protected, and does the plate stay inside its own budget when it is?
4. **The canonical workflow.** Half A should touch no `INPUT_TYPES`, no widget
   and no link. Confirm that, and name the audits that prove it.
5. **The other lanes (item 3b).** Every non-Ghost lane composes
   `appearance, setting, expression, motion[, camera]` through
   `motion_common.compose_parts`, and `appearance` is the cast row's face
   paragraph. `ltx25_foley_plus` and `ltx25_mime` already drop it deliberately
   (`_ltx25_parts(include_appearance=False)`), on the reasoning that the
   conditioning still already carries identity. Does that reasoning extend to
   every I2V lane, and which lanes are genuinely exempt (the talking-face lanes
   `minimax_h3_audio_in`, `humo*`, and any lane with a live T2V path such as
   `ltx_video`)?

## 10. The budget, MEASURED (r2 contract V5) -- and the operator's question answered

Run: `measure_slots.py` in this folder, against the installed ComfyUI SD1
tokenizer (`comfy.sd1_clip.SD1Tokenizer`, the same encoder the render uses) and
the real stored objects of "The Faded Ledger". Full output:
`slot_budget_measured.txt`.

**Every Ghost v2 prompt in that episode, per slot, in real SD1 tokens:**

| shot | mode | cue | motif | leaf | law | TOTAL |
|---|---|---|---|---|---|---|
| music_opening | signal | 5 | 5 | 14 | 17 | 38 |
| b001 | object | 5 | 7 | 14 | 18 | 41 |
| b002 | figure | 5 | 13 | 13 | 19 | **47** |
| b003 | object | 5 | 5 | 13 | 18 | 38 |
| b004 | figure | 5 | 13 | 12 | 19 | 46 |
| b005 | signal | 5 | 5 | 13 | 17 | 37 |
| b006 | signal | 5 | 6 | 12 | 17 | 37 |
| music_closing | object | 5 | 5 | 13 | 18 | 38 |
| **mean** | | 5.0 | 7.4 | 13.0 | 17.9 | **40.2** |

**The window is 77 tokens. We are using 40.** Mean headroom 36.8 tokens; even
the longest prompt in the episode leaves 30 tokens unused.

**This answers the operator's question -- "are we limiting the prompt size too
much or is that what the specs say" -- with a number: neither.** The spec is 77
tokens per CLIP window and nothing was pushing against it. The prompts were
short because they had little to say, not because a limit was squeezing them.

**And it re-reads his other complaint mechanically.** Of the mean 40 tokens:

* **45% is the mode law** -- framing boilerplate: *"mid-shot or wider, whole
  figure legible, one clear action, unbroken shot"*. Identical on every beat of
  the same mode.
* **18% is the motif** -- and on the two `figure` beats it is 13 tokens of
  invented costume ("a lean figure in a charcoal coat, carrying a satchel").
* **12% is the pack cue.**
* **32% is the leaf** -- the only slot that is about this beat at all, and on
  half these beats it just restates the motif with a verb.

So roughly **three quarters of every prompt is boilerplate or invention**, which
is precisely *"you are putting way too much ordinary character description,
that's why we're getting the same thing"*.

**What a v3 kernel actually costs, from this episode's own `key_objects`:**

    film canisters in high-security archive        10 tokens
    handwritten ledgers in high-security archive   10 tokens
    archive shelves in high-security archive        9 tokens
    ink pens in high-security archive               9 tokens
    security badges in high-security archive        9 tokens
    [setting alone] high-security archive           6 tokens
    [light] harsh fluorescent overheads             6 tokens

**v3's arithmetic, worst case:** cue 5 + kernel 10 + leaf 13 + light 6 + law 19
= **53 tokens, with 24 still spare.** Dropping the 13-token costume motif pays
for the kernel and the light with change left over. The drop-to-fit order
(r2 must-fix 9) stays in the design as a safety net, but on real material it
should never fire.

**Pack cues, measured (r2 must-fix 11 confirmed):**

| pack | cue | tokens |
|---|---|---|
| sci_fi_radio | *(empty -- the house look)* | 0 |
| anime | anime style | 4 |
| archival_documentary | archival documentary | 4 |
| cartoon | bright cartoon | 4 |
| paper_origami | folded paper | 4 |
| storybook_engraving | storybook engraving | 5 |
| shakespeare_stage_realism | photorealistic Shakespearean | 6 |
| recur_frac | recursive fractal light field | 7 |
| video_art | video-art feedback style | 7 |

The anchor's "4-9 tokens" was wrong at both ends: the range is 0-7, and the
default pack contributes no cue at all. The two packs the operator is most
interested in, `video_art` and `recur_frac`, are the most expensive at 7 -- still
trivial against 24 tokens of spare room.

## 11. The driver's own wiring answers (written BEFORE r3 replies, so r3 is checkable)

Section 9 asked r3 five questions. Three of them the driver can answer from the
files directly, and does, so the reviewer's answer is graded rather than
trusted.

**Q1 -- does anything break when `prompt_version` says v3?** No production
consumer branches on the literal string. `GHOST_PROMPT_VERSION_V2` is read in
exactly one place, `render_driver.py:2923`, where it is STAMPED; the only
equality test against it lives in `tests/test_ghost_prompt_v2_lane.py:217`.
`GHOST_V2_SLOTS` has one production consumer, the fallback at
`render_driver.py:2928`. The `/history` trace allowlist
(`render_driver.py:4981-5006`) carries `prompt_version` and `prompt_slots` as
opaque values and never inspects them.
**So:** the version bump is safe; the work is one test update plus adding
`prompt_slot_tokens` and `prompt_dropped` to that allowlist (r2 must-fix 12).

**Q2 -- what moves in the receipt, and does the verifier accept it?**
`_RECEIPT_CAUSAL_KEYS` includes `text_prompt` and `seed`
(`render_driver.py:4062-4067`), and `actual_request_sha` is the sha256 over
exactly those keys (`:4179-4182`). Under Half A the prompt changes and the seed
does not, so **`actual_request_sha` moves and `seed` holds** -- which is the
signature the A/B wants.

`scripts/otr_verify_replay.py` then splits cleanly:

* its **"seeds equal the source's per shot"** check compares each replay's seed
  against the SOURCE ledger's seed (`:99-104`). Under v3 that still **passes**,
  and it is the check that proves the A/B is honest.
* its **A/A check** (`:105-112`) requires two replays to agree on BOTH `seed`
  and `actual_request_sha`. A v2-vs-v3 pair differs on the sha **by design**, so
  run as-is it would report FAIL on a correct experiment.

**So:** the verifier needs one addition -- an A/B mode that asserts equal seeds
and *unequal* prompt shas, with the plate rule (`:113-122`) still requiring equal
plate hashes only where both rows carry one. That is a script change, not a node
change, and it is the honest way to make the tool state what it proved.

**Q4 -- does Half A touch the canonical workflow?** No. It changes
`ghost_signal_prompt.py`, `ghost_signal_author.py` and the driver block at
`render_driver.py:2870-2960`. No `INPUT_TYPES`, no widget, no link, no node
signature. The JSON round-trip, `OTR_WorkflowValidator`,
`test_widget_value_alignment`, `test_canonical_widget_input_parity` and
`test_workflow_link_target_indexes` all run anyway, and `build_variants --check`
with them.

Q3 (the plate's protected head) and Q5 (the other lanes) are left open for r3.
Q5's measurement is already done and lives in `other_lanes_audit.md`; what r3 is
asked for there is the wiring judgment, not the count.
