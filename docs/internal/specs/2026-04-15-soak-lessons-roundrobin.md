# OTR v2.0-alpha — Post-Soak Problem Statement & Round-Robin Brief

**Date:** 2026-04-15 (morning, post clean shutdown at RUN 289)
**Branch:** `v2.0-alpha` (not merged to main)
**Status:** Soak stopped cleanly. Pipeline is *stable* (ComfyUI server never crashed, VRAM never spiked, audio never corrupted). Narrative layer is broken. No new code until round-robin returns.
**Purpose:** Single shared brief for round-robin AI review (Claude Sonnet thinking → GPT-5 Pro → Gemini 2.5 Pro → repeat). Each reviewer appends to Section 7, does not rewrite earlier sections.

---

## 1. What the overnight soak actually proved

| Metric | Result |
|---|---|
| Runs completed since 18:00 Apr 14 | ~44 (RUN 246 → RUN 289) |
| ComfyUI server crashes | 0 |
| VRAM spikes past 14.5 GB | 0 (held 9.5–10.2 GB) |
| Audio pipeline byte-divergence | 0 observed |
| Episodes titled `signal_lost_the_last_frequency_*` | **100%** |
| Runs with `[WordExtend] Extension pass failed` error | **100%** (first hit 22:34 Apr 14, every run since) |

**Read:** hardware/VRAM/audio layer is shippable. Writer + post-writer layer is not. The green zone from yesterday's decision doc is *necessary but not sufficient* — even pinned to the safest parameters, the writer still hits narrative collapse.

---

## 2. Confirmed bugs (ranked by blast radius)

### 2.1 TITLE_STUCK — filename layer, not writer layer
- **Every** overnight MP4 filename begins `signal_lost_the_last_frequency_`.
- Writer is **not** stuck: the 08:26 run was clearly a "Mars Colony Alpha / Akira Cross / Jimbo Cranston" story. Writer varied. Filename slug did not.
- Hypotheses (pick one, verify):
  1. Title slug picker caches at module-scope and is never invalidated between runs.
  2. Title is computed from a seed that isn't reseeded per run.
  3. Writer emits a title in its output, but the filename path reads a different variable that defaults to a hardcoded fallback when the writer's title doesn't pass a validator.
  4. A recent canonicalization commit broke the title-extraction regex, so every run silently falls back to the same default.

### 2.2 WordExtend `NameError: _false_positives` — 100% failure on every run
- Log line: `[WordExtend] Extension pass failed: name '_false_positives' is not defined`
- Python `NameError`, not a config issue. Variable referenced but never bound in scope.
- Almost certainly the cause of **SHORT_DURATION ~29.9 s**: extension pass dies, script never grows to target length, final episode is stunted.
- Fix is likely <10 lines. Grep for `_false_positives` in `WordExtend` node, restore the missing `_false_positives = []` (or equivalent) in the correct scope.

### 2.3 Density cascade — 2100w / short (3 acts) deadlocks
- Enforced floor of 700w was the only way to keep the pipe alive.
- Root cause unclear: could be LLM inference budget, Bark TTS segment budget, or the LLM getting stuck in a repetition loop trying to hit an unreachable word count.

### 2.4 Cast diversity failures — ALL_SAME_GENDER, SINGLE_LINE_CHAR
- RUN 284 spawned three characters (Ryan Gordon, Andrew Bouvier, FDA Official) with one line each. Writer invented characters it didn't plan to use.
- ALL_SAME_GENDER = voice mapper clumps identical-gender voices from preset list.
- Prompt template does not enforce proportional dialogue distribution or gender balance.

### 2.5 Short-duration vignettes even when "SUCCESS"
- With 700w floor in place, writer produces ~30s vignettes instead of the target 5-minute arc. WordExtend was supposed to bridge — it's now broken (see 2.2), so vignettes slip through as false positives.

---

## 3. Green zone data (from yesterday, still valid as a floor)

Ship-allowed parameter set, empirically derived from post-fix runs 234–245:

- `words`: 350, 700, 1050, 1400
- `length`: medium (5 acts), short (3 acts)
- `genre`: dystopian, hard_sci_fi, first_contact, space_opera, cyberpunk, post_apocalyptic
- `style`: space opera epic, tense claustrophobic, chaotic black-mirror, noir mystery, hard-sci-fi procedural
- `creativity`: safe & tight, balanced, wild & rough
- `profile`: Standard
- **Safest default combo:** words=700, length=medium, genre=dystopian, style=space opera epic, creativity=safe & tight, profile=Standard

Red-zone (blocked until v2.1): words=2100, length=long, style=psychological slow-burn, genre=time_travel, genre=cosmic_horror, profile=Pro.

The overnight soak ran **inside** the green zone and still failed. So parameter guardrailing alone does not ship v2.0-alpha. Code-level fixes are required.

---

## 4. Architectural ideas on the table (for round-robin to evaluate)

### Idea A — Per-character constrained LLM (dialogue line-by-line)
> "Have a more constrained LLM say: you are CHARACTER X in Act N Turn Y, give me dialogue line 1, then step 2 and 3, check each line."

- Replace single monolithic "write me a full script" call with a **director agent** that orchestrates **per-character line generation**.
- Each line is generated against a tight schema: `{character, act, turn, emotion, duration_budget_sec, line}`.
- Schema is enforced via grammar-constrained decoding (outlines / lm-format-enforcer / llama.cpp grammars) — model is *incapable* of emitting malformed output.
- Benefit: eliminates the prose→parser pipeline entirely. Parser becomes a no-op. Most of the current fragile regex work goes away.
- Cost: higher latency per episode (one LLM call per line instead of one per script). Per-call VRAM is smaller, total VRAM same.

### Idea B — Schema-native writer replaces current writer
> "Instead of getting whole script we are having it build a schema set of script outputs including the dialogue of course."

- Single writer call, but output is a fully-typed JSON object conforming to the v2.1 spec schema (scene_break / environment / dialogue / sfx / pause / direction tokens).
- Same grammar-constrained decoding approach; no prose at all.
- Simpler than A, but character-consistency is harder because there's no per-character context window.

### Idea C — Hybrid: schema-native writer + per-character refiner pass
- Round 1: schema-native writer drops a skeleton of scenes + character list + line counts per character.
- Round 2: director loops over each character and asks a constrained per-character LLM to fill that character's assigned lines, given the scene skeleton as context.
- Closes the 58% → 90%+ gap because character voice consistency is enforced structurally instead of by prompt discipline.
- Also fixes SINGLE_LINE_CHAR: the skeleton pre-commits to line counts per character, so the fill pass literally cannot produce one-liner casts.

### Idea D — Title randomization + validation gate
- Minimal surgical fix: inject random seed into title generation, validate title is not a known-stuck value, regenerate if match.
- Doesn't solve the underlying cause but unblocks soak immediately.
- Should be combined with A/B/C, not a replacement.

---

## 5. Open questions for the round-robin

1. **Grammar-constrained decoding on llama.cpp + Blackwell sm_120 + torch 2.10 / CUDA 13 — what's the lowest-overhead library that actually works on this stack today?** (outlines, lm-format-enforcer, llama.cpp native grammars, xgrammar, other?)
2. **Latency budget for Idea C vs Idea A:** at 700w / 5 acts / ~60 lines, how much wall-clock does a per-character pass cost on a 7-8 B model loaded at Q6_K? Is the 10-30 min per-episode envelope still achievable?
3. **Is the TITLE_STUCK cause likely (1) caching, (2) seeding, (3) fallback regex, or (4) recent canonicalization regression?** (Need at least two AIs to look at the writer node code path before we patch blind.)
4. **Voice mapper gender clustering fix:** deterministic round-robin across gender buckets, or is there a better approach that preserves director-chosen casting intent?
5. **Does WordExtend survive the Idea B/C refactor, or does the schema-native writer make WordExtend unnecessary?** (If unnecessary, we delete it instead of fixing the `_false_positives` NameError.)
6. **Ship decision:** patch WordExtend + title slug + cast diversity in v2.0-alpha and ship (~1-2 days), OR skip that and jump straight to Idea C as v2.0.1 (~1-2 weeks)? Which gets us to HyWorld 2.0 integration faster?

---

## 6. Constraints (do not violate)

- C2: no `CheckpointLoaderSimple` in main graph
- C3: visual gen in subprocesses (not relevant here but applies at v2.1)
- C6: no visible lip sync
- C7: **audio output byte-identical to v1.5 baseline at every gate** — any idea that touches the audio path must pass this
- V1: HyWorld subprocess ≤14.5 GB VRAM
- SFW, non-violent, no profanity anywhere (code, comments, output)
- Local, offline, no paid APIs, no cloud
- RTX 5080 Laptop · 16 GB VRAM · Blackwell sm_120 · torch 2.10 · CUDA 13 · sequential only

---

## 7. Round-Robin Review Log (append-only)

Format per entry:
```
### Reviewer N — <model name + mode> — <YYYY-MM-DD HH:MM>
**Ranking of Ideas A/B/C/D (or other):** …
**Strongest argument against the currently-leading idea:** …
**Concrete next step (<3 bullets):** …
**Handoff note to next reviewer:** …
```

Rules:
- Each reviewer reads Sections 1–6 plus *all prior entries* in Section 7 before adding theirs.
- Do not edit prior entries. Append only.
- If you change your mind in a later round, note it as a new dated entry.
- Stop condition: **3 consecutive reviewer entries reach the same top-ranked idea and the same concrete next step.** At that point Jeffrey + Claude execute.

### Reviewer 1 — *pending* —

*(first reviewer fills in here)*

---

## 8. What NOT to discuss in this round-robin

- HyWorld 2.0 integration (separate v2.1 spec, roadmap only)
- Changing the TTS engine, audio mixer, or mastering chain (C7)
- Moving off local/offline (out of scope forever)
- Repo-level refactors unrelated to the writer + post-writer pipeline

---

*End problem statement. Hand off to next reviewer via: "paste this whole file, add your Section 7 entry, return to Jeffrey."*
