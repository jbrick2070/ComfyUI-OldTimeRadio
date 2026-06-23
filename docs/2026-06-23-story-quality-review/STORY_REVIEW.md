# OTR Story-Quality Review -- R3 flag-ON overnight soak corpus

**Date:** 2026-06-23 | **Reviewer:** Cowork (Claude), grounded against the real ledgers
**Corpus:** 18 episodes, `output/otr/episodes/*`, written 2026-06-22 23:13 -> 2026-06-23 06:34
**Config:** 883-word target, `OTR_STORY_QUALITY_V2=1` (all R3 levers ON, verified `v2_enabled=True` on all 18),
`OPENROUTER_REASONING_EFFORT=low`, writer rotation mistral / gemma-4-12b / grok (via OpenRouter slot-a).
**Raw evidence:** `RAW_LEDGER_DUMP.txt` (full scripts + telemetry + critic reports), `TRANSCRIPTS.txt`.

This is the judge anchor for the two roundtables. Every claim below is grounded in a specific ledger line.

---

## 0. Headline verdict

The R3 spine is **inert as a quality lift, and the operator's instinct is correct: instruction-gates the
weak model can ignore do not move the story.** But the soak data says the real problem is **bigger and more
specific than "imperative command-shouting."** Three things dominate, in order of impact:

1. **Length collapse.** Target 883 words; the corpus median is ~220 (range 154-430). Episodes run ~25-50% of
   the requested length. A 154-word "episode" (EP4) cannot have an arc -- this alone caps quality.
2. **Action-narration-as-dialogue.** The dominant flatness mode is not barked commands -- it is characters
   *speaking their own stage directions*: "Jettisoning module, bracing for impact.", "Initiating dark mode on
   mainframe.", "Rerouteing life support to boost launch sequence.", "Broadcasting Krit's threat, implicating
   him to the board." This is the radio-drama-specific failure and it is everywhere.
3. **Scene monotony across episodes.** All 18 collapse to the SAME scene regardless of premise: 2-3 people in a
   control room / cave / sub fight over a lever / key / drive / console while a gauge climbs into the red and
   something counts down. Knob, casing, lever, key, drive, override, purge, vent, lockdown. The premises differ
   (classroom AI, fossils, spiders, seabed, coal exports); the *drama* is interchangeable. No single-episode
   metric catches this -- but a human watching 3 in a row will.

The model that wrote the best stories was the **local gemma-4-12b**, not the frontier grok. That is the single
most actionable finding (section 4).

---

## 1. Quantified findings (aggregate over 18)

| Metric | Value | Note |
|---|---|---|
| Arc verdicts | strong 3 / uneven 9 / mid_collapse 3 / ungraded "?" 3 | 3/18 strong; all 3 strong are gemma-12b |
| Critic flat_lines | 64 total (~3.5/ep) | the flatness is REAL and critic-visible |
| Critic reroll_targets | 54 | flagged but mostly not acted (bypass/edit-cap) |
| Critic stance_issues | 2 | under-detects reversals (EP1 has an unflagged hard flip) |
| Words/ep | median ~220, range 154-430 | vs 883 target = severe compression |
| L1 objective-literal rerolls (telemetry) | 0 | **but see the bug below -- it actually fired** |
| L7 dialogue|action splits (telemetry) | 0 | nothing matched its narrow quoted-action shape |
| `v2_enabled` | 18/18 | flag-ON confirmed |

---

## 2. The failure modes, ranked, with line evidence

### F1 -- Length collapse (highest impact, fully measurable)
Words per episode: 249, 364, 221, 154, 338, 207, 231, 405, 164, 212, 430, 213, 213, 365, 234, 168, 353, 255.
Median ~220 against an 883 target. The writers treat "883 words" as advisory and stop at the first resolution.
Short episodes are *mechanically* flatter (no room for a turn, a beat of doubt, or subtext). EP4 "Hands-On
Re-entry" is 154 words across 14 character lines -- it is a logline read aloud, not a story.

### F2 -- Action-narration spoken as dialogue (dominant flatness mode)
Characters announce their own operations instead of speaking to each other:
- EP16: "Jettisoning module, bracing for impact." / "Initiating dark mode on mainframe." / "Rerouteing life
  support to boost launch sequence."
- EP4: "Initiating manual override." / "Flipping the main switch. Brace for blackout."
- EP3: "Broadcasting Krit's threat, implicating him to the board."
- EP6: "Lockdown's triggered. No going back." / EP9: "Sealing our fate, initiating total power purge."

This is a distinct class from D1's old "stage direction after a quote" -- here the *entire line* is a narrated
action. It is what the critic keeps flagging as flat (reason: decision/pressure/obstacle). L7 does not catch it
(no quoted-dialogue span to split). This is the single best target for a new deterministic gate.

### F3 -- Bare imperative command-shouting (the operator's stated complaint -- real, secondary)
"Override the lockdown! I need to access the core data now." (EP10) / "Initiate lockdown breach alarm, code
red." (EP10) / "Abort non-essentials." (EP16) / "Lock it down, now." (EP16) / "Initiate worldwide comms
silence." (EP16) / "Do it now!" (EP17). My narrow heuristic caught 14/252; the true rate is higher (critic
flat=64). Concentrated in mistral + grok episodes; near-absent in gemma's strong ones.

### F4 -- Scene/prop monotony across the set (no per-episode metric sees it)
Every episode is the countdown-over-a-control-panel scene. The recurring props: gauge-in-the-red, lever,
key/keycard, drive, override, purge/vent, lockdown, "thumb on the casing/switch." Distinct premises produce
identical staging. This is a CROSS-episode diversity failure -- the outline/beat planner reaches for the same
template every time.

### F5 -- Unearned or absent turns (arc)
EP1 "Sparks Fly": Ming orders "seize Charlie's droid," then two beats later -- with no turn beat -- offers a
"global partnership." The critic's stance detector fired only 2x in 18 episodes, so it is missing most of
these. "uneven" (9) and "mid_collapse" (3) are the arc symptom.

### F6 -- On-the-nose objective recitation (L2's target -- L2 did not move it)
Characters still state their want plainly: "I won't compromise my AI's integrity" (EP1), "These fossils...
they're our only chance" (EP6), "The world deserves better than false hope" (EP15). L2 suppressed the literal
`Objective:` in the prompt and injected a deflection directive (verified in code), but the weak writers ignore
it. Confirms: a soft prompt instruction is not enough.

### F7 -- Meta / director-instruction leak into spoken text (sharp bug, low frequency)
EP18 "Black Cable Live Feed", a spoken character line: **"Nia's voice should maintain its warmth and
calculation, not shift to a more urgent or aggressive tone."** That is a stage/critic *direction* emitted as
dialogue by the writer (grok/slot-a) and frozen into the ledger -- TTS would speak it. The hygiene chain did
not catch it. (Related noise: `phantom_name:*` compose-flags on EP10/EP13/EP16 -- name detector tripping.)

---

## 3. Why the R3 spine did not move the needle (per lever, grounded)

- **L1 (objective-literal floor):** telemetry says `l1_rerolls=0` everywhere, **but EP16 carries two
  `<<objective_literal_retry>>` compose-flags** -- so L1 *did* fire and the telemetry aggregation under-counts
  it (telemetry-vs-breadcrumb mismatch; verify in `scrub_ledger`). Worse: the rerolled line b004 "Jettisoning
  module, bracing for impact." is *still* in that episode's flat_lines. So even when L1 fires it does not fix
  flatness -- its matcher is a narrow content-word overlap, not a flatness detector.
- **L2 (authoring contract / objective suppression):** verified to rewrite the prompt, but the 12B/grok writers
  do not comply (F6). Soft instruction, ignorable.
- **L7 (dialogue|action split):** 0 splits -- the real failure (F2) is a *whole-line* narrated action with no
  quoted-dialogue span, which L7 is not shaped to catch.
- **Net:** the deterministic gates fire ~never; the soft contract is ignored; the flatness the critic plainly
  sees (64 lines) flows through untouched. The spine is safe (byte-identical off, 0 crashes) but does not lift.

---

## 4. Writer comparison (the most actionable result)

Each writer wrote 6 of the 18.

- **gemma-4-12b (LOCAL) = best by a clear margin.** Wrote all 3 "strong" arcs (Copper and Salt, Neural Jack
  Leak, Scorched-Earth Switch) and the 3 "ungraded" episodes were the richest prose in the set ("a graveyard of
  silt and dead calcium", "liquefy us inside these suits before you can even scream"). Longest outputs
  (338-430w). ZERO uneven/collapse verdicts.
- **grok (frontier, via OpenRouter slot-a) = underwhelming here.** 4 uneven + 2 mid_collapse, no strong.
  Compressed hard (164-255w) and produced the meta-leak (F7). With `reasoning_effort=low` it did NOT beat local
  gemma-12b -- which directly challenges candidate (c) "lean on a stronger frontier writer."
- **mistral-nemo = flattest + shortest.** The 154-249w episodes, the most action-narration and bare imperatives.

**Implication:** the cheapest real lift already sitting on the disk is **make gemma-4-12b the default writer and
push length adherence** -- not a new QA gate, and not (on this evidence) a frontier API.

### Harness bug that hides the best writer
gemma-12b's other 3 episodes (Pressure in the Red, Crawling Frost, The Brass Key) hit
`freeze=too_many_edits` -> `arc="?"` -> they were never arc-graded and show flat=0/reroll=0 because the critic
**bailed**, not because they were clean. The critic edit-cap penalizes the richest writer and drops it from
scoring. Fix the cap / the "too_many_edits" abort so the strongest output is actually graded.

---

## 5. Candidate levers (operator's a/b/c) -- pre-roundtable assessment

- **(a) bare-imperative-flatness reroll gate (deterministic):** right direction, but scope it to **F2
  action-narration** (whole-line narrated action), which is bigger than bare imperatives and fully detectable
  (verb-led, first-person operational, no second person / no subtext). This is the one new gate that targets
  what the critic actually flags.
- **(b) L4 best-of-N with a flatness scorer (model-agnostic):** promising precisely because it needs **no model
  compliance** -- generate N, score flatness (reuse the critic's flat dimensions as the scorer), keep the
  least-flat. The risk is cost/time (N renders of the writer) and trusting the scorer. Strongest for the weak
  local end.
- **(c) stronger frontier writer:** the soak **contradicts** this as stated -- grok underperformed local
  gemma-12b. Re-scope (c) to **"pick the best writer we have" = default to gemma-12b + length enforcement**,
  which is free and evidence-backed.

**Two levers the operator did not list but the data demands:**
- **(d) length adherence** (F1) -- the biggest measurable defect; nothing else matters at 154 words.
- **(e) scene/premise diversification at the OUTLINE/beat-planner** (F4) -- break the countdown-control-panel
  template so 16 episodes are not the same scene.

My pre-roundtable lean: **(d) length + default-to-gemma-12b** (free, evidence-backed) FIRST, then **(a) scoped
to F2 action-narration** as the one new deterministic gate, with **(b) best-of-N** considered for the weak end
if cost allows. Take this to the panel rather than asserting it.

---

## 6. Open code-verify items (to ground during the roundtable wiring round)
1. Telemetry under-count: `meta.story_quality.l1_rerolls=0` while `compose_flags` carry
   `objective_literal_retry` (EP16). Find the aggregation in `_otr_ledger_scrub` / `scrub_ledger`.
2. `freeze=too_many_edits -> arc="?"`: where the critic abandons scoring; confirm it is an edit-cap, not a real
   failure, and that it disproportionately hits long gemma outputs.
3. Where the writer length target is set / whether it is even passed as a hard floor.
4. F7 meta-leak path: how a director instruction reached spoken `text` (writer output parser / hygiene).
5. Writer default + selection: where the creative model is chosen, to make gemma-12b the default cleanly.
