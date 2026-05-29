# OTR Better Story Plan v8 (SYNTHESIZED — canonical) — Make the Ledger Carry Drama

**Date:** 2026-05-28
**Status:** canonical go-forward plan. Supersedes v7 (Claude's draft) and the round-robin seed.
**Core idea:** Do not ask the model to write "a better story." Force each ledger slot to perform a dramatic job.

---

## Core decision

Stop trying to make the critic/editor create quality after the fact. The real fix is:

```
Better beat obligation
→ better scene/exchange writing
→ strict craft-floor validation
→ commit only if the lines actually satisfy the drama contract
```

The ledger should not just say "this beat has a costly choice." It should force the written line to *show* the costly choice.

---

## Phase 1 — Immediate prompt fix: de-exposition guardrail (cheap, low-risk, do first)

Add a banned/generic-language block to both `_otr_line_composer.py` and `_otr_story_room.py` (Story Room writer prompt):

```
Do not use generic excitement or importance words.

Forbidden words and phrases:
intriguing, game-changing, unbelievable, transformative, fascinating,
anomaly, critical, unprecedented, remarkable, mysterious signal,
historic discovery, everything changes, this could change everything.

Do not state the emotion directly.
Do not say "I am scared," "this is terrifying," or "this is important."

Show pressure through:
- physical action
- concrete sensory detail
- hesitation
- contradiction
- refusal
- a character avoiding the real question
- a specific object in the room

Every voiced line must include one concrete detail:
a hand, a monitor, a cable, red dust, a paper note, a delayed screen,
a cracked speaker, a sample tray, a blinking light, a chair, a door,
a clock, a breath, a cup, a file, a microphone.
```

## Phase 2 — Add `slot_drama_contract` (main fix)

Before Story Room writes, build a per-slot contract for every voiced ledger line:

```json
{
  "dialogue_slot_id": "d004",
  "speaker": "ANTON CRANSTON",
  "line_job": "Reveal that Anton oversold the discovery and now fears an ordinary result.",
  "hidden_pressure": "Anton promised the press too much.",
  "concrete_detail_required": ["red dust", "delayed monitor", "sample tray"],
  "forbidden_words": ["intriguing", "game-changing", "transformative", "anomaly"],
  "state_before": "Anton is waiting for the rover image with public confidence intact.",
  "state_after": "Anton admits the wait has personal consequences.",
  "must_turn": true
}
```

Writer must output exactly: `d001|ANNOUNCER: ...` (one row per slot).
**Hard rule:** one row in, one row out, same slot IDs, same speakers, no added/skipped slots.

## Phase 3 — Replace isolated `compose_line` with `compose_exchange`

Single-line generation kills subtext. Instead of `compose_line(d002); compose_line(d003); compose_line(d004)`, do `compose_exchange(d002, d003, d004)`:

```
Write a 4-to-6 line exchange that satisfies these three dialogue slots.
Preserve every dialogue_slot_id.
Characters should not answer too directly.
At least one line must avoid the real question.
At least one line must reveal pressure through a concrete object or action.
Do not summarize the situation.
Do not explain the theme.
```

Output: `d002|ANTON CRANSTON: ...` (slot-formatted).

## Phase 4 — Python craft-floor validator (deterministic, before commit)

Fail the draft if:
```
- slot count mismatch
- slot order mismatch
- wrong speaker
- empty voiced line
- line under minimum word count
- forbidden word appears
- costly_choice slot has no decision/refusal/confession
- must_turn=true line does not change the situation
- no concrete detail appears in the line
- character states the dramatic state directly
```

New constraint code: `EXPOSITION_DUMP` — a craft-floor failure, not a taste complaint.

Fail: `ANTON: This discovery could be transformative for science.`
Pass: `ANTON: The color card is still gray, Tariq. If that tray comes up ordinary, I have to call Geneva and admit I sold them a miracle.`

## Phase 5 — One repair pass only

```
If failed: send only the failed slots back to the Writer with the exact failure reason; allow one repair pass.
If still failed: fail loud or fall back to legacy.
```

No three vague editor cycles. Repair prompt: repair only the failed slots, keep slot IDs + speakers, fix the listed failure only, return slot-formatted lines only.

## Phase 6 — Commit only validated Story Room output

```
Stage 1 plan → DramaticState → slot_drama_contract → compose_exchange / Story Room draft
→ craft-floor validator → one targeted repair → StoryRoomCommit → critic → freeze
```

Critic becomes a postmortem tool, not the thing that makes the story good.

## Discard / Keep

**Discard:** more editor cycles; vague rubric scoring; "make it more dramatic"; best-of-N at single-line level; trusting DramaticState alone; letting music interludes hide the missing middle.

**Keep:** banned generic language; concrete sensory requirements; grouped exchange writing; slot-level drama contracts; deterministic validation before commit; one targeted repair pass.

## Build order

```
1. Add banned generic-language prompt block.
2. Add slot_drama_contract generation.
3. Make Story Room output d###|SPEAKER lines.
4. Add Python validator before StoryRoomCommit.
5. Add one repair pass for failed slots.
6. Replace compose_line with compose_exchange for 2–3 voiced beats at a time.
```

---

## Claude's annotations (review notes — not part of the synthesized plan)

These do not change the plan's direction; they de-risk the build.

1. **Sequencing fix.** The validator's *semantic* checks (`must_turn changes situation`, `costly_choice has a decision`, `EXPOSITION_DUMP`) assume exchange-level writing. If step 4 (validator) goes hard-fail before step 6 (compose_exchange), single-line drafts will false-red and force the repair loop every run. Mitigation: land `compose_exchange` before enabling the semantic checks, OR make semantic checks **warn-only** until exchange writing is in. The lexical/structural checks are safe to hard-fail from day one.

2. **Two-tier validator — be honest about what's deterministic.**
   - **Tier A (truly deterministic, hard-fail):** slot count / order / speaker match, empty line, word-count floor, forbidden-word hit, "≥1 required concrete noun present."
   - **Tier B (semantic, NOT deterministic):** costly-choice present, scene turns, exposition dump, "states the dramatic state directly." Python cannot judge these without an LLM, and an LLM judge reintroduces jitter/looping. Start Tier B warn-only, or approximate with crude proxies (e.g., costly-choice beat must contain a first-person commitment verb + a concrete consequence noun) and accept they're heuristics.

3. **Lean on the positive requirement over the ban.** The concrete-detail *requirement* is a stronger, more robust lever than the forbidden-word *list* (which is whack-a-mole and can over-fire — e.g. "anomaly" in legit sci-fi). Keep both; weight the requirement.

4. **The contract pass is itself an LLM call.** `line_job` / `hidden_pressure` / `state_before/after` per slot is a new pass — tag it per project rule 6 (creative vs technical), budget its latency, and sanity-check the contract before trusting the floor built on it (garbage contract → garbage floor).

5. **Phase 0 (re-inserted): N=3 baseline on the current build first.** Or no later phase can be proven to beat the ±0.2–0.4 critic jitter.
