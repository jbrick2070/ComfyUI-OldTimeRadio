# OTR Story + Cast Fulfillment -- Roundtable Problem Statement (pass00)
2026-06-22. Grounded in the overnight broadcast soak: 17 published episodes + ~11
failures, 5 writer models (mistral / gemma-4-12b / grok-4.3 / gpt-5.5-pro /
deepseek-v4-pro), word tiers 420/560/700/864, all four creativity modes. Every
episode kept its full ledger.

> **Framing (operator):** this is NOT a QA pass that catalogs failures. We have a
> ledger contract and a cast contract. The question is **how do we best get a panel
> of writer LLMs to FULFILL them solidly** -- so a fresh run freezes clean, not
> "ships imperfect." R1 = creative/approach to fulfillment. R2 = the code we add.

---

## 1. The goal (what "solid" means)
A solid OTR episode is one where the writer LLM **fulfills the ledger + cast
contract on the first freeze**: strong/escalating arc, every line doing real
dramatic work (no flat lines), characters staying in their own distinct voice,
continuity respected, and a cast whose voices/roles pass the contract audit with
**zero violations**. Today that essentially never happens (see section 4) -- and
the reroll/critic loop does not recover it. We want fix runs that make clean
freezes the norm.

## 2. THIS IS OUR LEDGER (what the writer must fulfill)
Per-episode `*_ledger.json`. The writer fills a structured contract, not free prose:

- **`lines[]`** -- each row: `line_id`, `char_id`, `text`, `speaker_role`
  (announcer | character | music_* | sfx), `arc_phase` (setup -> complication ->
  resolution), `trait` (e.g. "escalating dread"), `beat_intent` (the director's
  goal for that beat), `dialogue_slot_id`, `target_words`, timing.
- **`meta.slot_drama_contracts`** -- the spine the writer must satisfy. Each slot:
  `{speaker, line_job ("Introduce the central conflict"), hidden_pressure}`. Every
  line is supposed to DISCHARGE its slot's `line_job` while carrying its
  `hidden_pressure`.
- **`meta.dramatic_state`** -- `{dramatic_question, character_a_wants,
  character_b_wants, costly_choice_beat, ...}`. The episode's engine.
- **`meta.continuity`** -- `{location, active_props[], facts[]}` the lines must not
  contradict.
- **Gate = the freeze cascade + StoryCritic axes:** `arc_verdict`
  (strong | uneven), `flat line(s)`, `continuity issue(s)`, `voice-drift note(s)`,
  `reroll target(s)`. Verdicts: `frozen_clean` (ship as-is), `frozen_with_warns`
  (ship, flawed), `needs_full_rerun` (refuse -- only renders tonight because we set
  the `OTR_BYPASS_FREEZE_HALT` stopgap).

## 3. THIS IS OUR CAST (what the writer + casting must fulfill)
`cast[]` rows: `char_id`, `name`, `character_description` (rich face/presence/voice
text used for portraits), `gender`, `tts_model`, `voice_preset`, `voice_params`,
`line_count`. Structure = **lead / foil / support + announcer** (gender + timbre +
role assigned). Bound under `meta.cast_contract` + `allowed_roster`
(`announcer, character, music_close, music_inter, music_open, sfx`), audited by
`OTR_LedgerReviewer:pre`.

## 4. Where the LLMs fall short today (grounded -- context, not the point)
Story:
- **0 / 18 episodes froze clean.** 6 `frozen_with_warns`, 11 shipped via
  "repair-then-ship" after the reroll **bounded out at 2 cycles still naming the
  same targets** -- i.e. the reroll re-composes but does NOT reduce the flagged
  count (cycle1: 3 targets -> cycle2: 3 targets -> bail).
- **Arc rated "uneven" in 50 of 55 critic passes (~90%)**; only 5 "strong."
- **55 flat-line flags, 44 continuity issues, 33 voice-drift notes** across the night.
- **136 stage-direction scrubs (~7.5/episode):** the writer keeps writing
  "(whispering)" / "(typing)" into spoken lines that then must be stripped.
- Line-level prose is often decent noir; the failure is **arc shape + flat beats +
  drift**, plus a reroll loop that can't fix what the critic flags.

Cast:
- **Cast-contract audit found violations in EVERY episode -- 6 to 24 each.** The
  concrete failure is `role_mismatch`: the assignment puts a **voice-engine name**
  (`kokoro`, `bark`) into the **role** field, which isn't an allowed role, so the
  reviewer leaves the row "unrepaired." Engine vs. role are conflated.
- **Voice binding is incomplete:** in the sampled episode 2 of 4 characters had
  `voice_preset=None`. Cast picks gender/timbre/role but does not always bind a
  concrete voice, leaning on a silent fallback.

## 5. R1 QUESTION FOR THE PANEL (fulfillment, not diagnosis)
Given the ledger contract (S2) and the cast contract (S3), and the shortfall (S4):

**How do we best prompt / structure the writer LLMs so a fresh run FULFILLS the
ledger + cast contract solidly enough to freeze clean -- across small local models
(mistral, gemma-12b) AND frontier (grok/gpt/deepseek)?**

Specifically, propose the highest-leverage moves on:
1. **Arc fulfillment** -- how to get a genuinely escalating setup->complication->
   resolution instead of "uneven," from a single writer pass.
2. **Flat-line elimination** -- how each line reliably discharges its
   `slot_drama_contract` `line_job` + `hidden_pressure` (and how the critic should
   define "flat" so the reroll can actually act on it).
3. **An effective reroll** -- why re-composing the same 3 targets twice doesn't
   reduce them, and what a fix-loop that converges looks like.
4. **Voice consistency** -- preventing voice-drift; keeping each character's
   register distinct across the episode.
5. **Cast contract** -- the engine-vs-role conflation, and binding a concrete voice
   for every character (no `voice_preset=None`).
6. **Stage-direction leakage** -- stop it at generation, not by post-scrub.

Out of scope for R1: the video stack (ltx_av/z-image), the wan removal, throughput.
R2 will show the actual writer/critic/cast code for the coding plan; R3 wires it
(workflow JSON + nodes); R4 converges.
