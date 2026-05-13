# Line-by-Line Script Composer — Optimization Problem Statement

**Date:** 2026-05-11
**Project:** ComfyUI-OldTimeRadio v2.0-alpha
**Branch:** `v2.0-alpha` (HEAD `3b74590`)
**Audience:** External reviewers (ChatGPT, Gemini, Nemotron, prompt-engineering specialists)
**Goal:** Optimize the per-beat dialogue line generator (`nodes/_otr_line_composer.py`) for a small local LLM (Mistral-Nemo Instruct 2407, 12B params, ~14 GB VRAM). The composer is the CORE quality bottleneck — it fires 10-20 times per episode and every dialogue line in the final audio is produced by it.

---

## 1. Pipeline context

Episode generation is a single Mistral-Nemo LLM passing through several distinct passes. In order:

1. **News interpreter** — 1 LLM call, distills raw RSS article → 4 purpose-specific briefs (script_brief, casting_brief, key_terms[], news_close_brief)
2. **Style picker (two-pass)** — 2 LLM calls (Inventor + Chooser), produces a single snake_case style descriptor
3. **Cast contract / locking** — 1+ LLM calls, generates per-character (name, gender, description, voice spec)
4. **Outline** — 1 LLM call (3-attempt retry-with-repair), produces a structured beat list with arc_phase, speaker, intent, mood, target_words per beat
5. **🎯 Line composer (THIS DOCUMENT)** — 10-20 LLM calls per episode, one per voiced beat. Returns the dialogue text for that line.
6. **Reviewer (Phase 3)** — 3 LLM calls (pre-audit, script doctor, post-audit) operating on the assembled ledger after composition

The composer is invoked from `OTR_LedgerScriptWriter.run()` inside a per-beat loop. Its output is written into the ledger row for that beat, then TTS, scene-sequencing, and video rendering consume the ledger downstream.

**Hardware:** RTX 5080 laptop, 16 GB VRAM, Blackwell, single GPU, local-only (no cloud, no API). VRAM ceiling 14.5 GB. KV cache reuse across composer calls is NOT implemented in the model loader (each call rebuilds prefix from scratch; we accept this for v2.0-alpha).

**Quality target:** A 350-word episode = 3 acts ≈ 14 voiced beats × ~25 words each + 2 announcer beats. Each line should be in character, advance the beat intent, fit the arc phase, and be free of phantom names (characters not in the locked cast).

---

## 2. Composer architecture

Module: `nodes/_otr_line_composer.py`
Function: `compose_line(generate_fn, req: LineRequest) -> LineResult`

The composer is a thin wrapper:

1. Build a system prompt (static, the same for every line in every episode).
2. Build a user prompt from `LineRequest` fields (per-beat structure, see §3 + §5).
3. Call `generate_fn(messages, temperature=0.8, max_new_tokens=200)`.
4. `strip_line_formatting()` cleans leaked formatting (speaker prefixes, brackets, markdown, wrapping quotes).
5. Run a heuristic **phantom-name detector** against an `allowed_roster` (cast + ANNOUNCER + news key_terms). Phantom names get flagged on the line (not rerolled).
6. Return `LineResult(text, compose_flags)` to the writer; writer stamps `text` + `compose_flags` onto the ledger row, saves immediately.

Retry strategy:
- **Attempt 1:** temperature 0.8
- **Attempt 2:** temperature 0.9
- Failure conditions that trigger retry: `generate_fn` raises, response is empty after format-strip, response exceeds `3 × target_words` (oversize cap).
- After 2 failed attempts: `LineCompositionFailedError` is raised. **Phantom name violations do NOT trigger retry** (cast is locked early; Phase 3 reviewer handles repair).

---

## 3. Inputs the composer sees TODAY (used)

The `LineRequest` dataclass (frozen) carries the following per-beat:

```python
@dataclass(frozen=True)
class LineRequest:
    # --- core beat info ---
    speaker: str                       # ALL-CAPS character name, e.g. "ALICE"
    intent: str                        # 1-sentence beat purpose, e.g. "hears unusual signal in lab"
    mood: str                          # 1-word mood, e.g. "curious"
    target_words: int                  # target word count for this line, e.g. 25

    # --- shared episode context ---
    canon_header: str                  # TITLE / SETTING / TIME / PREMISE (4 lines)
    last_lines: list[tuple[str, str]]  # rolling window of last N=5 spoken lines [(speaker, text), ...]

    # --- Phase 0 / 1 / 2A enrichment ---
    allowed_roster: frozenset[str]     # UPPERCASE set: cast names + "ANNOUNCER" + news key_terms
    style_descriptor: str              # snake_case episode-wide style, e.g. "closed_room_suspense"
    outline_spine: str                 # full episode outline rendered as one-line-per-beat
    character_voice_card: str          # 1-line speaker blurb: "ALICE (female, weary forensic engineer in her 40s)"
    arc_phase: str                     # current narrative phase: "setup" / "complication" / "climax" / etc.
```

### What each field gives the LLM

| Field | What the LLM gets | Why it matters |
|---|---|---|
| `speaker` | Who's talking | Anchor identity |
| `intent` | What this line must accomplish narratively | Single most important steer |
| `mood` | Emotional register for the line | Tone calibration |
| `target_words` | Approximate length budget | Length control (TTS timing downstream) |
| `canon_header` | Title, setting, time-of-day, premise | Episode-level grounding |
| `last_lines` | Last 5 spoken lines (post-strip), formatted as `[NAME]: text` | Continuity / "what just happened" |
| `allowed_roster` | All names the LLM may safely reference | Phantom-name prevention (post-hoc gate enforces) |
| `style_descriptor` | Episode tonal/atmospheric descriptor | Vibe steer |
| `outline_spine` | Full beat list with `b001 ALICE (curious): hears unusual signal` per row | Arc visibility — the LLM sees where it is in the episode |
| `character_voice_card` | One-line "name (gender, traits)" for the SPEAKER only | Voice differentiation |
| `arc_phase` | Current phase label + a one-line guidance string (e.g. "Escalate or introduce conflict. Make resolution harder, not easier.") | Narrative-direction steer |

The **static block** (style + canon + spine + roster) is identical for every composer call in an episode — designed to be KV-cache-friendly when reuse lands. The **variable block** (character + arc_phase + recent dialogue + write-line directive) changes per call.

---

## 4. Inputs the composer DOES NOT see today (not used)

These exist elsewhere in `meta` but are NOT threaded into `LineRequest`:

| Field | Where it lives | What it carries | Why we don't use it (yet) |
|---|---|---|---|
| `news_seed` | `meta.news_article.seed_text` | Raw original news article text (~1000-3000 chars) | Too noisy; the LLM gets the distilled `script_brief` indirectly via the outline's `intent` strings |
| `script_brief` | `meta.news.script_brief` | News-interpreter's distilled story plan (premise arc + central tension + beat hooks) | Outline already absorbed this; redundancy risk |
| `key_terms` | `meta.news.key_terms` | Verbatim journalistic terms (people, places, tech) the dialogue should surface | Indirect: included in `allowed_roster` so the LLM is allowed to say them, but no positive instruction to USE them |
| `casting_brief` | `meta.news.casting_brief` | The original character-motivation distillation | Subsumed by `character_voice_card` |
| All other characters' voice cards | `cast_rows` on `led.data["cast"]` | Voice cards for non-speaker characters | Only the SPEAKER's voice card is in the prompt — the LLM doesn't see who the OTHER cast members are (other than as names in allowed_roster + outline_spine) |
| `news_close_brief` | `meta.news.news_close_brief` | The journalistic closing text, overrides announcer-close beat verbatim post-composition | Not relevant for character lines |
| `seed` | `meta.episode_seed` | C7 byte-identity seed | Not used in prompt; only for RNG draws elsewhere |
| Prior beats' `arc_phase` / `mood` history | Available from `outline_spine` indirectly | Could give "we're rising into climax; previous beat was tense" steer | Currently inferred only via the spine; no explicit summary |
| Future beats | Available from `outline_spine` | The LLM CAN see what's coming, which is a feature OR a bug (plot-leak risk, ADR watch item) | Trade-off accepted today |
| `beat_id` of the current beat | Passed implicitly via `WRITE LINE: Speaker: ALICE` | Not numerically referenced in prompt | The LLM has to figure out "where am I in the spine" from speaker + intent matching |
| `cast_size`, `act_count`, `target_words` (episode total) | Episode budget | Episode-level shape | Not in composer prompt |

---

## 5. The exact LLM prompts the composer sends (today, verbatim)

### 5.1 System prompt (constant — ~90 tokens)

```
You write a single line of dialogue for an audio drama.

Output ONLY the line the character speaks. Do not include the character name. Do not include stage directions. Do not wrap the line in quotes. No prefix, no suffix, no formatting markup.

Match the requested word count approximately. Match the requested mood. Speak in the voice of the character given the recent dialogue context and the episode setting.

If you have nothing the character should say, output one short natural-sounding line that fits the moment. Never refuse, never explain, never apologize, never output meta commentary. Just the spoken line.
```

### 5.2 User prompt (per beat — assembled by `_build_user_prompt`)

The user prompt is assembled block-by-block. Optional blocks (STYLE, OUTLINE, ALLOWED NAMES, CHARACTER, ARC PHASE) are SKIPPED entirely when their field is empty. Below is the FULL prompt as it ships in production for a typical Phase 2A run:

```
STYLE: closed_room_suspense

EPISODE CONTEXT
TITLE: <episode title>
SETTING: <one-line setting>
TIME: <time of day>
PREMISE: <2-3 sentence premise>

OUTLINE:
  b001 [music_open]: cold open
  b002 ANNOUNCER (steady): introduce the episode
  b003 ALICE (curious): hears unusual signal in lab
  b004 BOB (worried): warns about source location
  b005 ALICE (determined): decides to investigate
  b006 BOB (reluctant): agrees to help
  b007 [music_inter]: transition
  b008 ALICE (afraid): finds something strange
  b009 BOB (urgent): tries to reach her
  ...

ALLOWED NAMES (do not invent any name outside this list; characters outside the cast or news-relevant terms will be flagged): ALICE, ANNOUNCER, BOB, CERN, JPL, Voyager

CHARACTER: ALICE (female, weary forensic engineer in her 40s, dry humor)

ARC PHASE: complication
  Escalate or introduce conflict. Make resolution harder, not easier.

RECENT DIALOGUE (most recent at bottom):
[BOB]: We should call the night supervisor.
[ALICE]: There's no time. The signal's already dropping.
[BOB]: Then take the back stairs — service tunnel access is on level 2.
[ALICE]: I'll meet you on the gantry. Don't wait for me.
[BOB]: That's a five-minute climb in the dark.

WRITE LINE
  Speaker: ALICE
  This line accomplishes: realizes the source is moving
  Mood: afraid
  Target word count: ~25

Write the line. Output only the spoken text.
```

**Prompt budget:** ~700-800 tokens total (system + user). Mostly OUTLINE + RECENT DIALOGUE.

**Generation params:**
- temperature: 0.8 (attempt 1), 0.9 (attempt 2)
- top_p: 0.92 (hardcoded in `_otr_model_loader.make_generate_fn`)
- max_new_tokens: 200
- pad_token_id = eos_token_id

---

## 6. Post-composition phantom-name gate (heuristic)

After `strip_line_formatting`, the composed text is scanned for proper-noun candidates:

1. **Titled names:** `\b(Dr|Mr|Ms|Mrs|Prof|Lt|Capt|Cmdr|Adm|Sen|Sgt|Col|Gen)\.\s+[A-Z][a-z]+\b` (e.g. `Dr. Patel`)
2. **ALL-CAPS tokens, len ≥ 2:** `\b[A-Z]{2,}(?:[-_][A-Z0-9]+)*\b` (e.g. `CARLA`, `NEXUS-7`)
3. **Title-Case bigrams mid-sentence** (skip sentence-start to avoid orthographic false positives): `\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b` (e.g. `Joe Smith`)

Common ALL-CAPS English words are allowlisted (OK, TV, AI, USA, UK, EU, UN, DNA, RNA, AM, PM). Any other candidate not in `allowed_roster` is recorded as `phantom_name:<token>` on `LineResult.compose_flags` and stamped onto the ledger row.

**The composer does not retry on phantom violations.** Cast is locked early per the synthesis §6.A decision; an LLM reroll can't invent the right name. Phase 3 reviewer handles repair downstream via Levenshtein auto-remap + Script Doctor + Step 2.5 deterministic phantom-skip fallback.

---

## 7. Other LLM passes (for context — NOT the primary optimization target)

Briefly, the prompts of the other LLM calls in the pipeline:

### 7.1 News interpreter — 1 call, ~1500-tok prompt

Produces `script_brief` + `casting_brief` + `key_terms[]` + `news_close_brief` from the raw RSS article. JSON schema output via post-hoc validation + reroll. System prompt mandates "purpose-specific briefs, each one short, single-paragraph, no era flavor."

### 7.2 Style picker — 2 calls (Inventor + Chooser)

**Pass 1 (Inventor):** ~800-tok prompt. "Produce 5 distinct snake_case style descriptors grounded in the article. Each must be 2-4 tokens. No duplicates." GBNF + regex grammar. Sampled from 10 seed styles.

**Pass 2 (Chooser):** ~500-tok prompt. Sees 5 candidates + 3 tie-breaker rules; picks the single best.

### 7.3 Cast / casting LLM — 1+ calls

Generates name + gender + character_description + voice_spec per character. JSON schema. Per-character single-pass.

### 7.4 Outline LLM — 1 call with 3-attempt retry-with-repair, ~1000-tok prompt

System prompt mandates: "Return exactly one JSON object." Schema spans title / premise / setting / time_of_day / beats[4..32]. Each beat has `beat_id / speaker / speaker_role / intent / target_words / mood / sfx_cue / arc_phase`.

Per-attempt budget: 1500 tokens output. Validation:
- Pydantic schema
- Cast-membership check (every character-role beat's speaker must be in the locked cast)
- 8 Phase 2A validators against the EpisodeBudget (per-phase word totals, per-phase beat counts, arc_phase ordering monotonic, music_inter count, announcer count, etc.)

On failure: attempt 2 fresh, attempt 3 is a REPAIR call (temp 0.3) with the prior raw response + the exact ValidationError message in the prompt.

### 7.5 Reviewer — 3 LLM calls per episode

**Pass 1 / Pass 3 — Cast Contract Auditor** (same function, called twice; ~2000-tok prompt each, temp 0.2):

```
You are a cast contract auditor for an audio drama script. Your job is to
detect deviations from the locked cast contract in the script ledger.

You DO NOT rewrite dialogue. You only flag violations.

VIOLATION TYPES: bad_casing / alias_used / invented_name / wrong_char_id /
role_mismatch / speaker_unknown.

For every violation output one CastViolation object {line_id, kind, found,
expected, confidence}. Return exactly one JSON PreAuditReport. No prose.
```

**Pass 2 — Script Doctor** (~3500-tok prompt, temp 0.5):

```
You are a script doctor for an audio drama. Cast contract is already
validated and any cast drift has been deterministically repaired.

You may propose: rewrite / skip / annotate (max N edits, scaled by
beat count). You may NOT insert beats, reorder beats, renumber line_ids,
or touch announcer/music/sfx beats.

Output a single JSON ScriptDoctorReport. No prose outside the JSON.
```

Pass 2 only sees CHARACTER beats (filtered before render); apply-time guard rejects edits targeting non-character beats.

---

## 8. The optimization question

We want external eyes on the **line composer prompt + input set** specifically. Concrete questions:

1. **System prompt** — is the current "single line of dialogue, no markup, match mood/words" instruction sharp enough? Too terse? Mistral-Nemo 12B is known to occasionally drift into omniscient narration even when told "spoken text only." Is there a better directive set?

2. **OUTLINE block** — we include the FULL outline_spine on every call (~150-250 tokens). Pro: arc visibility. Con: token cost + theoretical plot-leak risk (the LLM sees future beats). Is "full spine" the right call, or should we trim to current-phase + previous-phase summary?

3. **ALLOWED NAMES** — passed as a bare comma-separated list. Could we get better phantom-prevention by changing the framing (e.g. "These are the ONLY characters in this episode" vs the current "do not invent any name outside this list")?

4. **CHARACTER block** — we pass the SPEAKER's voice card only. Should we also pass voice cards for the OTHER characters in the cast? Cost: ~150 tokens for a 4-cast episode. Benefit: better cross-character voice differentiation, especially when the line is reacting to another character.

5. **ARC PHASE block** — currently `ARC PHASE: complication\n  <one-line guidance>`. Is that the right shape? Should it be richer (e.g. "We are 4/6 beats into the complication phase; the next phase is climax")?

6. **RECENT DIALOGUE window** — N=5 today (was 3 pre-Phase-1). Empirically good? Too narrow for long arcs? Too wide for short ones?

7. **WRITE LINE block** — currently 4 lines (Speaker / accomplishes / Mood / Target word count) + "Write the line. Output only the spoken text." What's the cleanest way to anchor the LLM here?

8. **Unused fields** — see §4. Should `script_brief`, `casting_brief`, full cast voice cards, beat position (beat_id), or arc-phase progress (e.g. "beat 3 of 6 in complication") be added to the prompt?

9. **Two-pass option** — would a separate quality-pass (composer attempt 1 generates, attempt 2 is "polish this line in the same voice, same length, same mood, but better") give meaningful gains for ~2× LLM cost? The Phase 3 reviewer's Script Doctor already does this AT THE END for a few flagged lines; a per-line polish loop would do it for EVERY line.

10. **Generation params** — temperature 0.8 → 0.9 retry ladder. top_p 0.92 hardcoded. min_p / repetition_penalty / frequency_penalty not used. Mistral-Nemo defaults? Or tuned?

### What we want back

- Concrete prompt tweaks (drop-in replacement system prompt, or block-by-block edits)
- Field additions / removals with justifications
- Two-pass vs single-pass tradeoff (does ~2× cost buy ~2× quality on a 12B model?)
- Generation params that work better for in-character dialogue on Mistral-Nemo 12B specifically
- Anything we're missing entirely from a prompt-engineering standpoint

Reviewer note: keep proposals lean. We hit ~800 tokens per call without KV cache reuse; doubling that doubles wall-clock per episode (~3-5 min added per soak). Quality gain has to be worth the latency.

---

## 9. Where the code lives

If you have repo access:

- `nodes/_otr_line_composer.py:562-650` — system prompt + `_build_user_prompt`
- `nodes/_otr_line_composer.py:343-400` — `LineRequest` schema
- `nodes/_otr_line_composer.py:658-770` — `compose_line` (retry loop)
- `nodes/_otr_episode_budget.py:ARC_PHASE_GUIDANCE` — the one-line guidance strings per arc phase
- `nodes/OTR_LedgerScriptWriter.py:1380-1440` — where `compose_line` is invoked per beat, with all LineRequest fields populated

Commit `3b74590` on branch `v2.0-alpha` carries the current shape.

End of brief.
