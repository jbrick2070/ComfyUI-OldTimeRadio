# Driver anchor -- Prompt v3 for the AnimateDiff lane: draw the crux (campaign item 3, 2026-09-02)

The source is the operator's own hand: `operator_rewrites.md` beside this file (ten beats
rewritten against the prompts the lane actually sent, plus nine rules in his words). This
anchor is the driver's code-grounded position BEFORE the panel. Roster: Fable 5.1 cold read
(creative, in flight at the time of writing) + Antigravity r1, Codex r2, Cursor r3, Sonnet r4.

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
