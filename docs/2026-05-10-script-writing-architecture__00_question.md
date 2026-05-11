# Round-Robin Brief: From Locked Inputs to Ledger-Locked Beat Dialogue

**Date:** 2026-05-10
**Project:** ComfyUI-OldTimeRadio (OTR) v2.0-alpha — local-only sci-fi audio drama generator
**Author:** Jeffrey Brick (with Claude as scribe)
**Reviewers:** ChatGPT (gpt-4.1 / gpt-5.5) + Gemini (gemini-2.5-pro)
**Stack head:** `2177a0c` on `v2.0-alpha`

---

## 0. What this brief is

We just spent a session locking down the upstream of our script-writing pipeline. Style, cast, briefs, structure inputs all flow cleanly into the outline LLM. The OUTLINE is solid.

We're now at the inflection point where the OUTLINE hands off to the SCRIPT WRITING PHASE — the part that takes the outline's `beats[]` array and turns each beat into actual dialogue text written to a structured ledger. **This is where consistency, character voice, arc payoff, and quality live or die.**

We want external eyes on the architecture for that handoff.

---

## 1. Hardware + model constraints (load-bearing)

- **GPU:** RTX 5080 Laptop, 16 GB VRAM, Blackwell. Single GPU. No cloud, no API, all local.
- **OS:** Windows. ComfyUI Desktop on localhost:8000.
- **LLM:** Mistral-Nemo-Instruct-2407 (12 B params, ~22 GB BF16, ~12 GB INT4). Loaded once per run, kept in VRAM throughout. Sustained throughput ~30-60 tokens/sec on this GPU.
- **VRAM ceiling:** 14.5 GB peak. Validator + reroll is the safety net, not prompt cleverness.
- **Memory rules** (load-bearing for all design suggestions):
  - **Lean prompts:** target ≤250 tokens for control-plane prompts. >400 tokens is a signal to cut, not justify.
  - **Encode constraints in JSON schema, not prose** — pydantic validation > "please follow these rules" instructions.
  - **Validator + reroll over prompt cleverness** — when the LLM drifts, retry with structured feedback rather than over-engineering the prompt.
  - **Fail-loud, no silent fallbacks.** If the LLM can't produce a valid result, raise; do NOT homogenize to a canned default.
  - **C7 byte-identity** — same seed widget value must produce identical output across runs (sample RNG seeded from user seed).
  - **Structured ledger writes only** — each line is a row with `{line_id, char_id, speaker_role, text, ...}`. NEVER parse `[VOICE: NAME]` markers from a text blob — the speaker is locked in via `char_id` at write time.

---

## 2. What's already locked in upstream (just shipped)

By the time we hit the script-writing phase, these have all been resolved by earlier LLM calls + Python + user widgets:

| Artifact | Source | Shape |
|---|---|---|
| `news_seed` | RSS auto-fetch or `custom_premise` widget | str — full article body |
| `style_pick.chosen` | Two-pass `_otr_style_picker` (Inventor + Chooser, GBNF-constrained, fail-loud) | snake_case 2-5 word descriptor, e.g. `decommissioned_dish_archive` |
| `casting_brief` | `news_interpreter.build_news_briefs` | str ≤200 chars: "what kinds of people belong here" |
| `script_brief` | same | str ≤350 chars: "premise arc + central tension + beat hooks" |
| `news_close_brief` | same | str ≤250 chars: closing news read for last announcer beat |
| `key_terms` | same | tuple of 2-6 short strings: people/places/tech that MUST surface in dialogue |
| `cast_rows` (locked cast) | `_otr_casting.lock_cast` (per-character LLM call w/ casting_brief) | list of dicts: `{char_id, name, gender, voice_preset, character_description}` |
| `target_words` | user widget | int (e.g., 350) |
| `target_length` | user widget label | str (e.g., `"long (7-8 acts)"`) |
| `include_act_breaks` | user widget bool | bool — controls whether outline plans music_inter beats between acts |
| `seed` | user widget | int — drives ALL RNG in the run for C7 byte-identity |

All of the above are in `meta` on the in-flight ledger before the outline LLM ever runs. No further upstream LLM work needed before script writing.

---

## 3. The current architecture (post-recent-shipping)

### 3.1 Outline LLM call (happens FIRST — before any script writing)

`nodes/_otr_outline.py:generate_outline()` produces a validated `Outline` pydantic with:

```
Outline:
  title       (str 3-80)
  premise     (str 10-400)
  setting     (str 4-120)
  time_of_day (str 3-40)
  beats       (list[Beat], 4-24 entries)

Beat:
  beat_id      (str pattern b\d{3})
  speaker      (str ALL CAPS, or "NARRATOR" for music/sfx)
  speaker_role (literal: character / announcer / music_open / music_close / music_inter / sfx)
  intent       (str 4-200 — one sentence describing what the beat accomplishes narratively)
  target_words (int 3-80)
  mood         (str 2-40 — tone descriptor)
  sfx_cue      (optional str ≤80)
```

The outline LLM **plans the spine**: who speaks when, what each beat accomplishes narratively, target word count, mood. It does NOT write dialogue. The dialogue text fields don't exist on `Beat`.

**What the outline LLM sees (post-2177a0c):**

```
SYSTEM:
You are a story editor. You produce JSON outlines for short
science-fiction audio dramas grounded in real science stories.

Your job is to plan the episode, not write the dialogue. Each
beat names the speaker, what they accomplish narratively, target
word count, and mood. The dialogue itself is generated by a
separate process and will use whatever register fits the story
and style.

OUTPUT FORMAT
[JSON schema for Outline + Beat]

CONSTRAINTS
- Every beat with speaker_role "character" MUST have a speaker that
  appears in the Cast block of the user prompt. Do not invent new
  characters.
- The first beat is typically speaker_role "music_open" or "announcer".
- The last beat is typically speaker_role "music_close" or "announcer".
- Beats should follow a clear arc: setup, complication, resolution.
- The premise must be grounded in the science story provided in the
  user prompt -- extrapolate dramatically from it, do not contradict it.
- Do not include the dialogue text in the outline. Only the intent.

USER:
Plan a science-fiction audio drama outline.

Story brief: {script_brief}
Required terms (plan beats that surface these in dialogue): {key_terms}
Cast (already chosen -- use exactly these names in character-role beats):
- ALICE (female, weary forensic engineer in her 40s, dry humor)
- BOB (male, ambitious grant officer in his 30s, evasive)
Style: {style_pick.chosen}
Target episode shape: long (7-8 acts). Include music_inter beats between acts.
Target total dialogue length: ~350 words (sum of per-beat target_words should land near this number).

Build a dramatic outline that develops this brief in the chosen style. Return only the JSON outline.
```

Validation: pydantic schema + `_no_duplicate_beat_ids` + post-pydantic per-beat cast-membership check (each `character`-role beat's speaker must be in the LOCKED `req.character_cast`). Reroll-then-repair on failure (3 attempts: 0.7 / 0.8 / 0.3 repair).

### 3.2 Per-beat composer loop (happens AFTER outline)

`nodes/_otr_line_composer.py:compose_line()` is called ONCE PER `character` or `announcer` BEAT. Music + SFX beats use `beat.intent` / `beat.sfx_cue` verbatim — no LLM call.

**What the line composer sees per call:**

```
SYSTEM:
You write a single line of dialogue for an audio drama.

Output ONLY the line the character speaks. Do not include the
character name. Do not include stage directions. Do not wrap the
line in quotes. No prefix, no suffix, no formatting markup.

Match the requested word count approximately. Match the requested
mood. Speak in the voice of the character given the recent
dialogue context and the episode setting.

If you have nothing the character should say, output one short
natural-sounding line that fits the moment. Never refuse, never
explain, never apologize, never output meta commentary. Just the
spoken line.

USER:
EPISODE CONTEXT
TITLE: {canon.title}
SETTING: {canon.setting}
TIME: {canon.time_of_day}
PREMISE: {canon.premise}
[optional: SOUND PALETTE: ...]

RECENT DIALOGUE (most recent at bottom):
[ALICE]: ...
[BOB]: ...
[ALICE]: ...

NEXT LINE
Speaker: ALICE
This line accomplishes: {beat.intent}
Mood: {beat.mood}
Target word count: ~{beat.target_words}

Write the line. Output only the spoken text.
```

Note what the composer **does NOT see today**:

- The locked `style` descriptor (`decommissioned_dish_archive`)
- `script_brief` or `casting_brief` (any of the news_interpreter outputs)
- `key_terms`
- Cast `character_description` per character (just names in `last_lines`)
- The full outline (only THIS beat's fields)
- Future beats (no lookahead — composer is blind to where the arc is heading next)

`canon_header` is the only "episode-wide" context the composer sees. It carries title/setting/time/premise/sound_palette — but NOT cast info, NOT style, NOT briefs.

`last_lines` is the running history: every previously-composed line as `(speaker, text)` tuples, accumulated as we walk through the beats. Currently UNCAPPED — by the last beat of a 15-beat episode the composer sees 14 prior lines in its prompt.

Retry: 2 attempts (0.8 / 0.9 temp). Fail conditions: empty after format-strip; oversized (>1.6× target_words). After all attempts fail → `LineCompositionFailedError`.

### 3.3 Per-beat ledger write

For each composed line, the writer immediately stamps a structured ledger row:

```python
patch_line_fields(led_disk, line_id, {
    "text":       cleaned,
    "char_id":    char_id_by_name[speaker],   # locked at cast-lock time
    "char_count": len(cleaned),
    "word_count": len(cleaned.split()),
    # ... other fields the per-beat consumers (Bark, Kokoro, etc.) will populate
})
```

The text is associated with the speaker by `char_id`, NOT by parsing `[VOICE: NAME]` markers from any text blob. This is the v2.0 LPL ("Local Per-Line Ledger") architecture — already in production. **No "loosely-goosey grepping" of speakers from text anywhere downstream.**

### 3.4 Post-composition phases

After all beats composed:
- `_generate_title_from_script` — one LLM call reads the FINAL assembled dialogue (no news_seed, no style, no outline metadata) and produces a 2-5 word title. Stamped at `meta.episode_title`.
- `_otr_news_wiring.override_announcer_close` — last announcer line's text is replaced with `news_close_brief` verbatim. Deterministic patch, no LLM.
- `_otr_news_wiring.post_assembly_keyterm_check` — walks every voiced line, word-boundary checks each `key_term` against `text`. Stamps diagnostic on `meta.post_assembly_key_terms`. Today warn-only; targeted-repair pass deferred.

### 3.5 Downstream critic

`nodes/script_critic.py` is a separate ComfyUI node that runs AFTER the writer node. Reads the final assembled script + style + anti-slop rubric, returns a verdict + revision suggestions. Already wired, already in production.

---

## 4. The architectural question

Given everything above, **what's the best architecture for the script-writing phase to satisfy these requirements:**

1. **Envision the whole story arc.** The outline does this today via the Beat[] list. Is the outline rich enough for the rest of the pipeline to deliver the arc? Or do we need additional structure (acts as first-class objects? scene boundaries?)?

2. **Number of beats and speakers per beat.** The outline LLM picks both today (constrained by target_length and the 4-24 beats Pydantic bound). Is this the right architectural layer for the decision?

3. **Final go-ahead before line writing.** Today the outline → per-beat-loop transition is unguarded. Should there be a quality gate (LLM critic, deterministic checks, both)? What does "go-ahead" mean concretely?

4. **Per-beat dialogue with consistency.** The current per-beat composer sees `canon_header + last_lines + this beat`. It does NOT see style, briefs, cast descriptions, or future beats. Is this enough? If not, what should be added without bloating per-beat prompts (per-beat = hot path, 10-20× per episode)?

5. **Optional second pass per line.** Yes, we can add a polish pass. But is the cost (2× LLM calls on local hardware) worth the quality gain? When should it fire — every line, only failed lines, only specific roles?

6. **Speaker locked in at ledger write.** This is non-negotiable. Each line's text gets stamped to a structured row with `char_id` set from the locked cast at write time. Confirms current LPL architecture is correct here; no design change needed on this axis. Just want reviewers to verify nothing in their proposals breaks this invariant.

---

## 5. Five architectural options

Each option is described against the current state. Cost is measured in additional LLM calls per episode (assume 15-beat episode for a "short (3 acts)" target).

### Option A — STATUS QUO (current architecture, no further changes)

- Outline LLM: 1 call
- Per-beat composer: 15 calls
- Title regen: 1 call
- Total: **17 calls / episode**

**Pros:**
- Already shipping. Just needs a soak run to validate quality.
- Lean per-beat prompts (~150 tokens each).
- No additional infrastructure.

**Cons:**
- Composer is blind to style + briefs + character descriptions. Every beat reasons from `(canon_header + last_lines + this_beat)` only. Risk: dialogue feels generic-sci-fi regardless of the upstream `style="haunted_broadcast_signal"` selection.
- `last_lines` grows unboundedly. By beat 15 the composer carries 14 prior lines in its prompt → token cost compounds + risk of context-window pressure on long episodes.
- No lookahead. Composer can't set up the next beat because it doesn't know what it is.
- Outline → composer transition has no quality gate. A weak outline silently propagates through the per-beat loop and the only review is the post-hoc `script_critic` (which sees the finished script, can't easily reroll).

### Option B — STATUS QUO + style/brief pass-through to composer

Add three optional fields to `LineRequest`: `style: str`, `script_brief_excerpt: str` (≤200 chars), `character_description: str`. Composer prompt gains 3-4 lines of episode-wide context per call.

- LLM call count: same as A (17/episode). No new calls.

**Pros:**
- Composer can finally write dialogue that actually reflects the locked style.
- Character descriptions inform character voice ("dry humor" character vs. "anxious" character read very differently).
- Cheap (no new LLM calls, ~30-50 tokens added per per-beat prompt).
- Conservative — preserves the current architecture but closes the upstream-leak gap.

**Cons:**
- `_otr_line_composer.py` is currently a "locked v2.0 module" per project conventions — would need a small unlock + sweep to ship.
- Per-beat prompts grow ~30%, compounding across 15× per episode. Still well under the 250-token budget.
- Doesn't address consistency drift across long episodes.
- Doesn't address the no-lookahead problem.

### Option C — Per-beat polish (draft + revise per line)

After the composer drafts a line, fire a SECOND LLM call ("polish pass") that reads:
- The draft line
- The previous 3-5 lines (recent context)
- The next beat's `intent` (lookahead)
- `style` + `character_description` + `mood`

The polish LLM rewrites for: tone, character voice, word count adherence, no info dumps. Output replaces the draft in the ledger.

- LLM call count: 17 + 15 = **32 calls / episode**. ~88% increase.

**Pros:**
- Two-pass design always wins on quality if the second pass has the right context (cf. our two-pass style picker — Inventor + Chooser killed mode collapse).
- Catches per-line drift early instead of letting `script_critic` flag it after the fact.
- Lookahead is the simplest place to insert it (polish knows what comes next; composer doesn't).

**Cons:**
- Doubles LLM time. On RTX 5080 a 15-beat episode goes from ~3-5 min total to ~6-10 min.
- More VRAM pressure (longer context window with previous/next/character/style).
- Quality of the polish depends on prompt design — Mistral-Nemo can over-edit and lose voice.
- Adds another module + tests + soak surface.

### Option D — Outline-gate critic before per-beat phase

Insert a small critic step BETWEEN outline-validate and the per-beat loop. The gate critic reads the outline + locked cast + style + script_brief and produces an `approve | revise(reasons)` verdict.

- LLM call count: 17 + 1 = **18 calls / episode**. ~6% increase.

**Pros:**
- Catches structural issues (poor arc, character monopoly, weak setup) at the cheap (outline) layer instead of expensive (per-beat) layer.
- Fast to add (small module, single LLM call, similar pattern to news_interpreter).
- Composes cleanly with options B/C — orthogonal.

**Cons:**
- Duplicates `script_critic` somewhat (which already does post-hoc review).
- Mistral-Nemo's outline is usually structurally fine after our recent upstream-leverage work; the gate may rarely trigger.
- Adds latency to the front-load of every run.
- Feels like solving a problem we may not have. Soak-once-first might prove the outline is already good enough.

### Option E — HYBRID: Add option B + option C with smart polish (only fires on flagged lines)

- Composer gets style/brief/description pass-through (B).
- Polish pass exists but ONLY fires when the draft line trips a quick post-write check:
  - Word count deviation > 30% from target
  - Repeats a phrase from the prior 3 lines
  - Too short (< 3 words)
  - Mentions characters not in the cast
  - Includes an instruction-token leak (e.g., "OK", "Sure")
- LLM call count: 17 (B's baseline) + ~3-5 (polish on flagged lines, typically 20-30% trip rate) = **20-22 calls / episode**.

**Pros:**
- Get most of C's quality benefit at ~20% of C's cost.
- Composer feels the upstream signals (B).
- Polish only burns cycles where it matters.
- Gradual rollout: ship B first, observe trip rate, decide if polish is worth wiring.

**Cons:**
- Two new code paths to maintain (per-line check + polish).
- Trip-rate-based decisions are sensitive to the heuristic threshold; needs tuning.
- More complex than B alone.

---

## 6. Open subquestions (the real consistency problems)

Independent of which Option (A-E) wins, these tradeoffs matter:

### 6.1 `last_lines` context window strategy

Current: unbounded. Beat 15 sees 14 prior `(speaker, text)` pairs in its prompt.

Three strategies:
- **Sliding window (last N=5):** lean prompt, but long-range callbacks broken.
- **Sliding + summary:** keep last 5 lines verbatim + a "story so far" auto-generated summary (one LLM call per beat after the summary cache stales — caching helps).
- **Full (status quo):** preserves callbacks, but token cost compounds.

For Mistral-Nemo at 32k context, even full is fine. But the longer the prompt, the more "lost in the middle" risk for small models.

### 6.2 Callback handling

Beat 12 might want to reference setup from beat 3. Today the composer sees beat 3 in `last_lines` (verbatim) but with no metadata indicating "this was a setup line worth callback." Should the outline pre-tag setup beats? Or trust the composer to scan `last_lines` for relevant prior content?

### 6.3 Character voice consistency

Today every per-beat call shows the composer the previous lines (so previous speaker turns are visible). But the composer has no compressed "voice fingerprint" for each character — it has to re-derive ALICE's voice from scratch each time.

Could feed `character_description` per beat (Option B). Could also build a per-character "running voice profile" that updates after each line. Latter is a bigger lift.

### 6.4 Pacing / word budget

Per-beat target_words sums to user's target_words, but actual composed word counts can drift ±20%. There's a writer-phase E warning if the SUM drifts >25%, but per-line drift isn't caught.

Should per-line drift trigger a polish pass (Option E)? Or just let `script_critic` flag it post-hoc?

### 6.5 Long episodes (8+ acts, 30+ beats)

Current architecture scales linearly: O(N) LLM calls + O(N²) total token cost (each beat sees all prior lines). For a 30-beat episode, beat 30 sees 29 prior lines = ~1500 tokens of `last_lines` context. Still under budget but pushing it.

Should there be a per-act batching step? Compress completed acts into summaries before starting the next act?

### 6.6 Acts as first-class objects

Today the outline is a flat list of beats. Acts are implicit (music_inter beats mark boundaries when `include_act_breaks=True`). Should the schema explicitly model acts?

```
Outline:
  acts: list[Act]

Act:
  act_number: int
  arc_phase: literal["setup", "rising", "climax", "falling", "resolution"]
  beats: list[Beat]
```

Tradeoff: more explicit structure = better arc-aware reasoning, but bigger schema = more for the small LLM to lift.

---

## 7. Preliminary recommendation (Claude's take)

**Ship Option B FIRST as the immediate next commit.** Add `style`, `character_description`, and a 1-2 sentence `script_brief_excerpt` to `LineRequest`. Composer prompt gains ~50 tokens but actually has the signal it needs. No new LLM calls, no new module, ~1-day refactor. Soak ramp will then tell us whether B is enough.

**Defer Options C/D/E** until we have soak data. C and E are bigger lifts and the trip-rate (E) or quality-gain (C) is unknowable without a baseline. D is a "maybe" — would land cleanly but may not be needed if B + the existing `script_critic` cover the bases.

**For consistency (Section 6):** start with sliding-window-of-5 + keep `script_brief_excerpt` as the always-present arc anchor. Skip the auto-summary infrastructure for now (more LLM calls, more complexity). Re-evaluate after first 8+ act episode.

**Don't model acts as first-class yet.** The implicit-via-`music_inter` model is working and matches the current schema. Promoting acts to a typed object is a bigger refactor that should wait for evidence of need.

**Outline gate (D)?** Skip. The outline already has Pydantic + cast-membership validation + reroll-and-repair. `script_critic` is the post-hoc safety net. A pre-loop critic feels like belt-and-suspenders without strong motivation.

**The non-negotiable architectural invariant:** dialogue text gets stamped to a structured ledger row with `char_id` set from the locked cast at write time. No `[VOICE: NAME]` parsing anywhere in the pipeline. Current LPL architecture preserves this; all options A-E preserve it.

---

## 8. Specific things we want reviewer opinions on

For ChatGPT and Gemini — please weigh in directly on:

1. **Is Option B enough, or do we need C/E?** Is the per-beat composer with `style + character_description + script_brief_excerpt + last_lines + this_beat` rich enough for Mistral-Nemo 12 B to produce consistently good dialogue? What's the failure mode you'd predict?

2. **`last_lines` strategy.** Sliding window of 5 vs. unbounded vs. window+summary — what does the literature say about small-model context-window degradation on long sequences?

3. **Polish pass cost-benefit on local hardware.** If we add Option C, what's the realistic quality gain on Mistral-Nemo specifically (not on GPT-4-class models)? Has anyone benchmarked draft-then-revise on 12 B-class models?

4. **Outline gate value.** Is a pre-loop critic LLM call meaningfully different from a post-hoc `script_critic` for catching structural issues? Or is it the same check at a different point in time?

5. **Long-episode scaling.** For 30+ beat episodes, what's the recommended pattern? Per-act batching? Auto-summarize completed acts? Stay with linear approach until it breaks?

6. **What are we missing?** Any architectural pattern we haven't considered? Any common pitfall in this kind of "outline → per-beat composer" pipeline that we've overlooked?

7. **Pure cosmetic question:** is "beat" the right term? We use "shot" for video-pipeline render units (HuMo, FLUX) and "line" for ledger rows. Beats live at the outline-planning layer. Some confusion has arisen — is there a cleaner naming convention?

---

## 9. Hard rules (do not violate in any proposed alternative)

- **Local-only.** No cloud, no paid API, no remote endpoints.
- **VRAM ≤ 14.5 GB peak.** Must fit alongside the loaded story LLM.
- **C7 byte-identity.** Same seed widget value → same output. RNG always seeded from `int(seed)`.
- **Fail-loud.** No silent fallbacks to canned strings. Validator + reroll, then raise.
- **Lean prompts.** Per-beat hot path stays under 400 tokens including all contexts.
- **Speaker locked at ledger-write time.** Text is associated with `char_id` from the locked cast. No regex `[VOICE: NAME]` parsing anywhere downstream.
- **No `dummy` placeholder names** in code, comments, or tests. Use `placeholder` / `stub` / descriptive names.
- **Safe-for-work.** No profanity. Non-violent. Good narrative arc.

---

## 10. Reference file paths (for context if you need them)

- **Outline LLM:** `nodes/_otr_outline.py` (Outline pydantic schema, Beat schema, system + user prompt, generate_outline function w/ retry loop + cast-membership check)
- **Cast LLM:** `nodes/_otr_casting.py` (per-character casting call w/ casting_brief)
- **News interpreter:** `nodes/news_interpreter.py` (one LLM call → 4 briefs)
- **Style picker:** `nodes/_otr_style_picker.py` (two-pass Inventor + Chooser, GBNF-constrained, fail-loud)
- **Line composer:** `nodes/_otr_line_composer.py` (per-beat dialogue writer)
- **Episode canon:** `nodes/_otr_canon.py` (`render_episode_canon_header` builds the per-beat header)
- **Writer orchestrator:** `nodes/OTR_LedgerScriptWriter.py` (the ComfyUI node that runs the whole pipeline; D.0-D.5 phases for cast contract + outline; per-beat composition loop later in the file)
- **Ledger schema:** `nodes/production_ledger.py` (line row shape with `line_id`, `char_id`, `shot_id`, `beat_id`, `text`, etc.)
- **Script critic:** `nodes/script_critic.py` (post-hoc whole-script review)
- **Project rules:** `CLAUDE.md` (root)
- **Roadmap + bug log:** `ROADMAP.md`, `docs/BUG_LOG.md`

---

## 11. Round-robin process

1. Run `scripts/_consult_round_robin.py` against this brief OR `scripts/_consult_openai.py` first then feed ChatGPT's answer to Gemini (per CLAUDE.md round-robin process).
2. Save responses as `01_chatgpt.md` and `02_gemini.md` next to this file.
3. Loop step 2 if reviewers disagree on something material — re-prompt with the disagreement spelled out.
4. Synthesis goes in `04_synthesis.md` (Claude or Jeffrey writes it after reading both).
5. ADR (if architectural lock-in is warranted) goes at `docs/script-writing-architecture-adr.md`.

---

**End of brief. Reviewers — please answer Section 8 directly, and feel free to propose architectural alternatives outside the A-E options if you see a better one.**
