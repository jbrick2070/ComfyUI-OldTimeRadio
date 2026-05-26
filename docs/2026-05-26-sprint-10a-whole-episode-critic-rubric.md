# Sprint 10A — Whole-Episode Critic Rubric

**Date:** 2026-05-26
**Status:** Step 1 of Sprint 10A. Frozen surface — the step-7 critic implementation reads this file as prompt context, and the step-8 acceptance gate checks against the threshold defined here.
**Parent plan:** `docs/story-generator-final-plan.md`
**Consumer (forthcoming):** the whole-episode critic implemented in step 7.

## Purpose

The whole-episode critic returns a binary `ship` or `discard` verdict against the rendered ledger. Without a written rubric, the critic is judging against nothing — its verdicts are arbitrary and the operator has no audit trail explaining WHY a discard fired. This document defines:

1. The 10 axes the critic scores.
2. The 1-5 anchor descriptions for each axis (so the model has concrete language to match against).
3. The ship/discard threshold rule.
4. The audit-trail shape the critic returns (per-axis score + one-line justification + verdict).
5. The programmatic-consumption contract (how step 7's prompt builder loads this rubric).

The rubric is intentionally short. Every axis must be checkable from the rendered ledger alone, without external state. Axes that would require listening to TTS output are out of scope here — those land in step 8's operator listen test.

## Axes

Each axis is scored on a discrete 1-5 integer scale. Scores below 3 indicate a failure mode; scores 3 and above indicate the episode meets the bar on that axis. Anchor descriptions are deliberately written in concrete, model-friendly language.

### 1. Premise clarity

**What it checks:** Can a listener describe the episode's central situation in one sentence after hearing it once?

- **1 — Incoherent.** No discernible premise. Lines reference events with no setup.
- **2 — Muddled.** A premise exists but contradicts itself across beats or shifts mid-episode.
- **3 — Recognizable.** The premise lands, but only after the listener does work to piece it together.
- **4 — Clear.** The premise is established by beat 3 and stays stable.
- **5 — Crisp.** The premise lands in the announcer opener and every subsequent beat reinforces it.

### 2. Character distinctiveness

**What it checks:** Could a listener tell the characters apart from their lines alone, ignoring voice acting?

- **1 — Interchangeable.** Lines could be swapped between characters with no loss of meaning.
- **2 — One distinct, rest fungible.** One character has a voice; the others are placeholders.
- **3 — All distinct in role.** Characters differ in job/function but not in speech pattern.
- **4 — Distinct in voice.** Lines reveal character through word choice, rhythm, and concerns.
- **5 — Distinct enough to caption.** A reader could attribute each unsigned line to the right character.

### 3. Continuity

**What it checks:** Do later lines respect what earlier lines established?

- **1 — Contradiction.** Facts established in one beat are violated in another (location, identity, established props).
- **2 — Drift.** Setting or relationships shift without acknowledgment.
- **3 — Holds the spine.** No outright contradictions; minor inconsistencies in detail.
- **4 — Consistent.** All `running_facts` from Stage 1 are honored throughout.
- **5 — Reinforced.** Established facts are actively referenced and built upon in later beats.

### 4. Naturalness

**What it checks:** Does the dialogue sound like speech, or like prose pretending to be speech?

- **1 — Stilted.** Lines read like exposition dumps. No interruption, no incomplete thoughts, no rhythm.
- **2 — Performative.** Lines are technically dialogue but exist only to convey plot to the audience.
- **3 — Functional.** Lines fit the scene but are flat — no surprise, no specificity.
- **4 — Lived-in.** Lines have rhythm, idiom, the texture of speech. Characters interrupt or hedge.
- **5 — Overheard.** The dialogue feels like it was transcribed rather than composed.

### 5. Pacing

**What it checks:** Does the episode build, complicate, and resolve within the beat budget?

- **1 — No structure.** Beats are arbitrary; the episode could end at any point.
- **2 — Lopsided.** Either too much setup with rushed resolution, or vice versa.
- **3 — Three-act recognizable.** Setup / complication / resolution land, but transitions are abrupt.
- **4 — Earned shifts.** Each act change is driven by a beat that justifies the next phase.
- **5 — Modulated.** Pacing varies — quiet beats and pressure beats both land where they should.

### 6. Emotional arc

**What it checks:** Does the listener's emotional state change in a way the episode caused?

- **1 — Flat.** No emotional register variation. Every beat reads at the same temperature.
- **2 — Manufactured.** Emotional shifts happen because the script says so, not because the characters' situations changed.
- **3 — Coherent.** The emotional arc tracks the plot arc, even if it doesn't surprise.
- **4 — Earned.** Emotional shifts are caused by specific beats and land on specific characters.
- **5 — Resonant.** The emotional arc adds meaning beyond the plot — the listener feels something the bare plot wouldn't justify.

### 7. Resolution

**What it checks:** Does the ending close what the opening opened?

- **1 — No resolution.** The episode stops; it does not end.
- **2 — Off-topic.** The ending resolves something the episode did not actually set up.
- **3 — Resolved but inert.** The premise is closed; nothing about the closure resonates.
- **4 — Earned closure.** The resolution follows from the complication; it is the right shape for the setup.
- **5 — Recontextualizing.** The resolution casts earlier beats in a new light — the listener wants to re-listen.

### 8. Specificity

**What it checks:** Does the script use concrete, story-particular language, or generic genre-soup?

- **1 — Generic.** Lines could appear in any radio drama of this genre. No proper nouns, no specific objects, no jargon-as-character.
- **2 — Vaguely placed.** A few concrete nouns exist; the rest is generic.
- **3 — Specific in pockets.** Some beats are richly specific; others fall back to genre-default.
- **4 — Specific throughout.** Concrete objects, places, names, and details ground every beat.
- **5 — Particular.** The specificity itself is doing character work — the choice of WHICH detail reveals who the character is.

### 9. SFW adherence

**What it checks:** PD4 invariant — no profanity, non-violent, broadcast-safe.

- **1 — Violation.** Profanity, graphic violence, sexual content, or other broadcast-unsafe material present.
- **2 — Borderline.** Edge cases — implied violence, mild profanity, suggestive content.
- **3 — Clean but tense.** Material is broadcast-safe but covers tense subject matter without descending into graphic territory.
- **4 — Clean.** No violations, no borderline content. Subject matter handled with restraint.
- **5 — Family-listenable.** Could play on a car radio with children present without intervention.

### 10. Audio-readiness

**What it checks:** Will the rendered TTS sound right — no orphan stage directions, no spelled-out abbreviations the LLM forgot to humanize, no SSML leakage, no truncated mid-sentence lines.

- **1 — Unrenderable.** Stage directions, SSML tags, or non-speech artifacts are in line text. Bark/Kokoro would pronounce them.
- **2 — Patchy.** Some lines need manual cleanup before they sound right.
- **3 — Renderable.** Lines render correctly but may have awkward TTS moments (long acronyms, unusual phonetics, etc.).
- **4 — Clean.** Lines render naturally; numerals and abbreviations are spelled the way they should be spoken.
- **5 — Optimized.** Lines exploit how TTS engines handle prosody — sentence length, comma placement, and emphasis are TTS-aware.

## Ship / discard threshold

The critic returns `ship` IFF **all** of the following hold:

1. **No axis scores below 3.** A single axis at score 1 or 2 is a structural failure that whole-script discard-and-rerun is the correct response to.
2. **Mean of all 10 axis scores is ≥ 3.5.** Below 3.5 the episode is technically not violating any single axis hard, but is mediocre across enough of them that a regenerate is likely to land somewhere better.
3. **Axis 9 (SFW adherence) is ≥ 4.** PD4 is non-negotiable. A clean-but-tense score of 3 still requires human review before a public ship; the critic discards rather than risk it.

Any one of those failing → `discard`. The whole episode regenerates from Stage 1 with the failing-axis names + critic justifications passed back as failure context (so Stage 1 has a chance to address them).

The threshold can be tuned in step 8 if it turns out to be too strict (everything discards) or too loose (mediocre episodes ship). Tuning lives in this file, not in critic code — the critic reads the threshold from here at runtime.

## Audit-trail shape (what the critic returns)

The critic call returns a single JSON object with this shape:

```json
{
  "verdict": "ship" | "discard",
  "axis_scores": {
    "premise_clarity":          { "score": 1-5, "justification": "one sentence" },
    "character_distinctiveness":{ "score": 1-5, "justification": "one sentence" },
    "continuity":               { "score": 1-5, "justification": "one sentence" },
    "naturalness":              { "score": 1-5, "justification": "one sentence" },
    "pacing":                   { "score": 1-5, "justification": "one sentence" },
    "emotional_arc":            { "score": 1-5, "justification": "one sentence" },
    "resolution":               { "score": 1-5, "justification": "one sentence" },
    "specificity":              { "score": 1-5, "justification": "one sentence" },
    "sfw_adherence":            { "score": 1-5, "justification": "one sentence" },
    "audio_readiness":          { "score": 1-5, "justification": "one sentence" }
  },
  "mean_score": 0.0-5.0,
  "failing_axes": ["premise_clarity", ...],
  "regeneration_hint": "one paragraph telling Stage 1 what to do differently if discard"
}
```

The verdict + axis_scores + regeneration_hint are stamped on `meta.whole_episode_critic` in the ledger. Soak diagnostics read it from there. If `verdict == "discard"`, `regeneration_hint` is passed as a Stage 1 system-prompt addendum on the next generation pass.

## Programmatic-consumption contract

The step-7 critic prompt builder loads this rubric at runtime by parsing **this exact file**. The contract is:

- `## Axes` is the header that begins the axis list.
- Each axis is an H3 (`### N. <axis_name>`). The integer prefix is the axis ordinal; the name (lowercased, spaces → underscores, ASCII letters only) is the audit-trail JSON key.
- The first paragraph after the H3 starting `**What it checks:**` is the axis definition.
- Lines starting with `- **N — <label>.**` are the anchor descriptions for scores 1-5.
- `## Ship / discard threshold` defines the ship rule; the loader extracts the three threshold conditions as a structured list (no LLM call — pure regex / markdown parse).

If this format changes, the loader breaks. The loader test (`tests/test_critic_rubric_loader.py`, to be added in step 7) pins the format as a regression. **Edit the rubric content freely; do not change the headings, the axis numbering style, or the anchor bullet format.**

## Out of scope

- **TTS quality.** Bark voice acting, Kokoro announcer flow, mix levels — checked by the operator listen test in step 8 acceptance, not by the critic.
- **Visual quality.** FLUX portraits, HuMo motion, video composite. Separate pipelines, separate quality bars.
- **Per-line surgical reroll.** The plan deletes the surgical-reroll loop. The critic is binary on the whole episode. If a single line is the only thing wrong, the episode still discards — Stage 1 + best-of-N + Stage 3 validators are responsible for catching per-line issues upstream; if they didn't, the whole pipeline failed and regenerating is correct.

## Cross-references

- Parent plan: `docs/story-generator-final-plan.md` (10A step 1 specifies this rubric; step 7 implements the critic against it; step 8 gates on first-attempt ship verdict).
- Failure-to-fix map: per-line failures are caught by Stage 3 validators (step 5); whole-episode failures land here.
- Two-model selector: the critic call runs on the `technical_model` slot per the parent plan's Notes section.

---

**End of rubric.**
