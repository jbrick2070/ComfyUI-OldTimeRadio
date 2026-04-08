# SIGNAL LOST v1.2 beta — Peer Review Request

**Date:** 2026-04-07
**Branch:** `v1.2-narrative-beta` @ `e2f210a`
**Status:** First boot test complete. Need help before bug-fix pass.

---

## The Ask

Gemma 4 E4B-it generated a **strong narrative** but **ignored the XML metadata block instructions** in Sections 6–9 of `SCRIPT_SYSTEM_PROMPT`. I need a second opinion on how to fix this before I start hacking the prompt.

---

## What Shipped

All 6 narrative patterns as a prompt-level MVP inside one mega-commit:

- **Pattern 1** — AISM Filter (sensory-first language, forbidden constructs, breath budget)
- **Pattern 2** — `SCAFFOLDING_PREAMBLE` (master dramaturg role, Brick Method 1:5, acoustic spaces)
- **Pattern 3** — `<epilogue_candidates>` verbalized sampling (Stanford 5-response technique)
- **Pattern 4** — Yes-But / No-And act-break escalation
- **Pattern 5** — `<vocal_blueprints>` pipe-delimited character block before Scene 1
- **Pattern 6** — `<locked_decisions>` JSON block between Scene 2 and Scene 3

All delivered as inline instructions in `SCRIPT_SYSTEM_PROMPT` + preamble. Zero new Gemma calls, zero new VRAM.

---

## Boot Test Result

Workflow: `old_time_radio_test.json` (1 min, hard-sci-fi, 3 chars, act_breaks=False)
News seed: "Male octopuses guided through mating by female hormones" (Ars Technica)

### What Worked ✓

| Area | Status |
|---|---|
| Gemma 4 load, inference, VRAM handoff | Clean |
| News fetch, content filter, Lemmy RNG | Clean |
| **Pattern 1 AISM Filter** | Visibly working. Concrete SFX ("high-pitched electrical WHINE"), sensory-first dialogue, natural burstiness, no M-dash crutch, no Rule of Three |
| **Pattern 2 Scaffolding** | Dialogue quality consistent with dramaturg framing |
| Director JSON extraction | Clean gender hints, procedural voice assignment |
| Parser, SceneSequencer, Bark handoff | All green |

### What Failed ✗

**Zero metadata blocks emitted.** Gemma went straight to `=== SCENE 1 ===` and never produced:

- `<vocal_blueprints>` (Pattern 5)
- `<locked_decisions>` (Pattern 6)
- `<epilogue_candidates>` (Pattern 3)

The narrative is strong but the structured metadata that Patterns 3/5/6 depend on for v1.3 Python post-processing is missing.

---

## Hypothesis

Sections 6–9 are **buried at the bottom** of a ~30,000-character `SCRIPT_SYSTEM_PROMPT`. Gemma 4 E4B-it appears to weight recency differently than expected — the AISM filter instructions (Section 5) are being followed because they're behavioral rules the model enforces continuously during generation, but the "emit this block once" instructions at the end of the system prompt are being dropped.

The block instructions are also phrased as **descriptive specs**, not **imperative emission commands**. For example:

```
6. VOCAL BLUEPRINTS ...
   BEFORE writing === SCENE 1 ===, emit a single <vocal_blueprints> block ...
```

Gemma may be treating this as documentation rather than a directive to emit literal text.

---

## What I'm Considering

1. **Reorder**: Move Sections 6–9 to the FRONT of `SCRIPT_SYSTEM_PROMPT`, right after the `SCAFFOLDING_PREAMBLE`, before the canonical audio engine rules.
2. **Rephrase**: Replace "emit a single block" language with explicit emission template:
   ```
   THE FIRST THING YOU WRITE MUST BE THIS EXACT BLOCK, FILLED IN:
   <vocal_blueprints>
   [NAME] | [CLIPPED|MEASURED|RAMBLING] | [sighs|laughs|pants|gasps|sobs] | [foley cue] | [8 word wound]
   ...
   </vocal_blueprints>
   ```
3. **Concrete example**: Give Gemma a full worked example of the block in the system prompt, not just the spec.
4. **Accept defeat for Pattern 3**: Escalate the epilogue selector to v1.3 multi-call architecture immediately. Prompt-only verbalized sampling may be genuinely beyond Gemma 4 E4B-it's following ability.

---

## Questions for Reviewers

1. Has anyone seen Gemma 4 E4B-it follow or ignore XML-block emission instructions in long system prompts? What worked?
2. Is **reorder + imperative rephrase + worked example** the right fix order, or should I try one at a time to A/B isolate the cause?
3. Is the token budget a factor? Script used ~3160 output chars out of `max_new_tokens=1024`. Was there headroom for the blocks?
4. Should Pattern 3 (epilogue sampling) be promoted to v1.3 multi-call immediately given that verbalized sampling is arguably the hardest of the three to coax from a single pass?
5. Any other pattern I should just punt to v1.3 rather than try to rescue in prompt?

---

## Context Files

- `nodes/gemma4_orchestrator.py` @ `e2f210a` — the orchestrator with all six patterns
- `QA_GUIDE_v1.2_beta.md` — full v1.2 design, trade-offs, known gaps
- `ROADMAP.md` — v1.3 backlog N1–N23 (already queued)

Generated script from the boot test is available on request.

---

*Looking for: experience with long Gemma prompts, XML emission reliability, verbalized sampling in single-pass generation. Blunt feedback welcome.*
