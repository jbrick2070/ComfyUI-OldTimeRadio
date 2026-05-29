# Story Generator — Final Plan

**Date:** 2026-05-26
**Status:** Synthesized from both consultant responses plus the prior Stage 1/2/3 architecture. Opinionated — what I judged worse or wrong is dropped, not preserved as alternatives.

The current pipeline is failing in four stacked ways: JSON gates exhaust their retry ladder, casting inverts gender, LineComposer fights word-count compliance instead of inhabiting characters, and the reroll loop makes scripts worse on cycle 2. The plan addresses all four directly. It is LLM-agnostic and assumes multi-call chat-style generation.

---

## Architecture

Three stages.

### Stage 1 — Structural plan (one call, schema-enforced)

One call generates the entire episode plan as a structured object. Output uses **grammar-constrained generation** (`outlines`, `xgrammar`, `llguidance`, or llama.cpp GBNF — whichever the LLM serving stack supports). This makes invalid JSON structurally impossible at the token-sampling layer.

Schema:

```
{
  "premise": "one sentence",
  "arc": { "setup": "...", "complication": "...", "resolution": "..." },
  "cast": [
    {
      "name": "...",
      "gender": "male | female | nonbinary",
      "pronouns": "he/him | she/her | they/them",
      "voice_id": "v2/en_speaker_N",
      "persona": "2-3 sentences",
      "arc_role": "..."
    }
  ],
  "beats": [
    {
      "beat_id": "b001",
      "speaker": "CHARACTER_NAME",
      "intent": "one sentence",
      "length_target_words": 26,
      "emotional_register": "...",
      "callback_to": "b00X (optional)"
    }
  ],
  "running_facts": [ "established fact 1", "established fact 2" ]
}
```

**Cast audit runs after Stage 1, before any dialogue is generated.** Each cast member's `name → gender → pronouns → voice_id` is validated deterministically (name-gender lookup, voice-roster check). Mismatches are repaired or the plan is regenerated. **Loud failure, not silent.** This is what fixes the Cole-as-female / Reginald-as-female bug — not prompt hope.

### Stage 2 — Dialogue rendering (multi-turn roleplay per line)

Per beat, a short chat conversation walks the LLM into character. The model "warms in" across acknowledgment turns, then speaks at Turn 4.

```
Turn 1 SYSTEM: You are {speaker_name}. {persona}. Reply "Ready" when set.
Turn 1 ASSIST: Ready.

Turn 2 USER: Premise: {premise}. Arc so far: {arc}. Confirm.
Turn 2 ASSIST: Confirmed.

Turn 3 USER: Established facts:
             - {running_facts[0]}
             - {running_facts[1]}
             You: {persona}. Pronouns: {pronouns}. Confirm.
Turn 3 ASSIST: Confirmed.

Turn 4 USER: Previous line ({previous_speaker}): "{previous_line}".
             Your beat: {intent}. Length: ~{length_target_words} words.
             Register: {emotional_register}.
             Speak your line. Just the line.
Turn 4 ASSIST: [DIALOGUE — this is what ships]
```

**Only Turn 4's output lands in the script.** Turns 1-3 do two things: they reshape the prompt into a chat-native sequence the model handles better than a single monolithic prompt, and they give validation breakpoints — if the model doesn't return "Ready" / "Confirmed" (or a close variant) at the right turn, the chain is broken and the line regenerates.

**Best-of-N at Turn 4.** Start N=4. Generate N candidates, score each via Stage 3 validators, ship the highest scorer. Increase N if quality justifies.

**`running_facts` updates after each line ships.** A short separate call (or rule-based extraction) appends new facts to the running list. Subsequent Stage 2 calls see them at Turn 3. This is the state spine — much cheaper than per-character actor agents, and good enough that actor agents are deferred indefinitely.

**Prefix caching must be on** in whichever backend serves the LLM. Without it the multi-turn pattern is expensive for no reason.

**LineComposer mode switch.** Two modes:
- `roleplay_single_turn` (A1: one prompt with everything)
- `roleplay_multiturn` (A2: the chain above)

Default is `roleplay_multiturn`. The single-turn mode stays available for the A/B comparison and as a fallback if A2 misbehaves on a specific model. The old structured/legacy mode is deleted — it's the failure baseline.

### Stage 3 — Validate + ship-or-discard

Each Turn 4 candidate passes through mechanical validators. These are cheap and run on every best-of-N candidate.

1. **Length** — word count in `[target × 0.5, target × 1.7]`.
2. **Pronoun consistency** — references to the speaker match their pronouns from Stage 1.
3. **Speaker leak** — no "As {character}, I would say..." or stage directions.
4. **Banned phrases** — configurable list of purple-prose markers, grown from past failures ("forfeit to the void", "expected toll", etc.).
5. **Continuity** — line does not contradict any `running_facts`.
6. **On-beat** — line at least loosely matches the beat's `intent`.

Lines passing all validators are scored (length closeness, register match). Best score ships. Lines failing all candidates get one final regeneration attempt with the failure reason fed back. If they still fail, the line is stamped `needs_review` and ships with a warning, OR the whole episode is marked for discard — operator call, controlled by a config flag.

**No surgical reroll.** Best-of-N already selected the best candidate per line. After the whole episode is generated, a final critic LLM call returns a binary verdict against the rubric below: `ship` or `discard`. `discard` regenerates the whole episode from Stage 1. The current critic-reroll loop is deleted — the data shows it makes scripts worse, not better.

---

## Failure-to-fix map

| Failure | Fix |
|---|---|
| JSON gates exhaust retry ladder | Stage 1 grammar-constrained generation makes invalid JSON impossible |
| Empty `script_brief` starves announcer / MusicGen / critic | Stage 1 always produces a populated plan |
| Cole=female, Reginald=female | Stage 1 deterministic cast audit; Stage 3 pronoun-consistency validator per line |
| Length drift | Stage 3 length validator with per-beat targets |
| Flat / purple dialogue | Stage 2 multi-turn roleplay + best-of-N + Stage 3 banned-phrase / speaker-leak validators |
| Continuity breaks | `running_facts` carried into every Turn 3; Stage 3 continuity validator |
| Reroll cycle 2 worse than cycle 1 | Surgical reroll deleted; replaced with whole-episode discard |
| Premise/scene mismatch | Stage 1 binds premise + cast + beats in one structured object |

---

## Sprint 10A — Stabilization (ships v2)

Build order. Each step has an exit gate; do not proceed past a failed gate.

1. **Write the rubric for the whole-episode critic.** 5-10 axes scored 1-5: premise clarity, character distinctiveness, continuity, naturalness, pacing, emotional arc, resolution, specificity, SFW adherence, audio-readiness. This is what the critic checks against. Without it, the critic is judging against nothing.

2. **Confirm which LLM is in the slot.** The workflow JSON has one value; the failure logs reference another. Reconcile before anything else — every other decision depends on this. One minute of work.

3. **Implement Stage 1 with grammar-constrained generation.** Test in isolation: 20 runs, count first-attempt valid plans. **Gate: ≥19/20 must pass.** If grammar constraints don't get there for the current model, fall back to API LLM for Stage 1 only — but only after constrained decoding has been honestly tried.

4. **Implement Stage 1 cast audit.** Deterministic name→gender→pronouns→voice validation. Test against 10 generated casts. **Gate: 0 mismatches across 10 runs.**

5. **Implement Stage 3 validators as standalone functions.** Run them against all dialogue lines in existing ledgers (past failed runs). **Gate: every historical flat line, length drift, and pronoun inversion is caught.**

6. **Implement Stage 2 multi-turn roleplay with best-of-N=4.** Wire to validators. Confirm prefix caching is enabled in the LLM backend.

7. **Implement the whole-episode critic against the rubric from step 1.** Binary verdict only — no surgical reroll path exists.

8. **End-to-end run on the same outline as a known-failing past episode.** **Gate: critic verdict is `ship` on first whole-episode attempt; operator listen confirms distinguishable characters, no purple prose, coherent arc.**

If gate 8 passes, this is v2. Ship it.

---

## Sprint 10A-LAB — Empirical A1 vs A2 (parallel to 10A, optional)

Before fully committing to multi-turn (A2) as the default, lab-isolate both modes on identical Stage 1 plans. Render 3 episodes each. Blind operator listen.

Decision rule:
- A2 clearly better → A2 stays the default.
- A1 essentially equal → A1 becomes default, A2 stays as `high_quality_mode` flag (slower, used when ship bar is high).
- A1 clearly better on this model → A1 becomes default, A2 deleted.

The point: don't ship A2 because the literature says so. Ship it because the listen test confirms it on the model actually in the slot.

---

## What's deferred

The following are real options, none of them in 10A:

- **Per-character actor agents with episode-persistent state.** Right architecture long-term, too much scope before the basics are stable. The `running_facts` spine captures most of the value at a fraction of the cost.
- **LoRA fine-tune on canonical episodes.** Premature. Needs 3-5 episodes worth keeping first. Once those exist, revisit.
- **Specialized critic split** (continuity / character / pacing / dialogue critics, Dramaturge-style). Worth it if the whole-episode critic from 10A turns out to be too blunt an instrument. Not on day one.
- **Stage 1 best-of-N.** Premium polish. Worth it if Stage 1 plans turn out to be the bottleneck on creative quality. Not on day one.

---

## Acceptance criteria

10A is done when an end-to-end run satisfies all of:

1. Stage 1 produces a valid plan first-attempt in ≥19/20 trials.
2. Cast audit catches all name/gender/voice mismatches; 0 inversions reach Stage 2.
3. Stage 2 lines pass Stage 3 validators on first or second candidate (not requiring fallback) in ≥80% of beats.
4. `running_facts` entries from Stage 1 are consistent through the final rendered script.
5. Whole-episode critic returns `ship` on first attempt in ≥7/10 trials.
6. Operator listen test: distinguishable characters, coherent arc describable in one sentence, no purple prose, no meta leakage, audio-ready.

---

## Notes

- **LLM-agnostic.** Any chat-tuned model that supports grammar-constrained output and multi-turn conversation with prefix caching can run this pipeline. Quality varies with the model; shape does not.

- **`running_facts` is the state spine.** It replaces actor agents for v2. Update after each line ships, feed into the next Turn 3. Simple, debuggable, sufficient.

- **Stage 1 is where compute is best spent.** A weak plan dooms the episode no matter how good the dialogue rendering. If you have wall-clock budget to spend, spend it on Stage 1 best-of-N before adding it to Stage 2.

- **Discard-and-rerun feels wasteful but is right.** Surgical patches on a bad script produce worse scripts. Discard the whole thing; start over from Stage 1 with the failure reason as context.

- **Two-model selector still applies.** `technical_model` slot serves Stage 1 (structured) and any Stage 3 LLM checks. `creative_writing_model` slot serves Stage 2 (dialogue). Same model can fill both; different models can be assigned.

---

**End of final plan.**
