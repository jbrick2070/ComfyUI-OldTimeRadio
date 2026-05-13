# Script Writer Architecture — How do we stop trying to parse a god-monolith and just compose lines reliably?

> **Round-robin consultation request, fresh problem.** ChatGPT, Gemini — please review independently and respond with: (1) which proposed architecture you'd ship; (2) any architecture we missed; (3) implementation risks per option. Claude will synthesize, flag disagreements, decide.
>
> **TEMPORARY** — `git rm` once architecture is chosen and shipped.

---

## TL;DR

Today's session caught **5 separate failure modes** in `OTR_LLMScriptWriter`, all rooted in the same architectural pattern: **one big LLM call → emit ~256-1024 tokens of "screenplay" → parse with brittle regex → hope it survives critique + revision + format-norm**. Mistral-Nemo Instruct 2407 (local 4-bit NF4) reliably writes flowing prose when asked for screenplay — the parser then has to play whack-a-mole with format variations. Today alone: parser crash (PARSE_FATAL on FormatNorm output), zero-character episodes (all-narrator monologues despite cast=3 declared), 0 scenes / 0 dialogue lines emitted from 835-token outputs.

Jeffrey's instinct is to **stop generating one parseable monolith and instead compose lines one-at-a-time with the LLM, stamping each to the ledger as we go**. Whether that's the right pivot, and what the cheapest version of "right" looks like, is the question.

---

## Hardware + model context

- **GPU:** RTX 5080 Laptop, 16 GB VRAM, single device, Windows
- **LLM:** `mistralai/Mistral-Nemo-Instruct-2407` (12B params), loaded **4-bit NF4** quantized, ~7.7 GiB on GPU, 8192 context cap (truncated from 128k for VRAM headroom)
- **Inference path:** in-process via `transformers.generate()` from `OTR_LLMScriptWriter` ComfyUI node
- **Throughput:** ~13-15 tok/s during generation, ~50-60 s for 800-token output
- **Existing API keys:** `OPENAI_API_KEY` + `GEMINI_API_KEY` in env (used for round-robin consultation, not in-pipeline). Jeffrey's CLAUDE.md says "no cloud services in audio/video render path" — but ScriptWriter doesn't break C7 byte-identity (the audio is regenerated every run anyway)

---

## Today's failure log (real evidence)

### Failure 1 — FormatNorm produces output the parser doesn't recognize (FIXED in commit `485874b`)
- Cyberpunk neon-noir, 100 words, 2 cast, self_critique=ON
- Mistral wrote inline `[VOICE:]` format → critique passed (10 dialogue lines counted) → revision kept it (sim 91.7%) → **FormatNorm reformatted into bare `CHARACTER:\ndialogue\n` screenplay** → parser crashed with `PARSE_FATAL: 0 dialogue from 2768-char input`
- Fix: Pass 3 added to `_parse_script` to handle bare CHARACTER: format. Ships, but is symptom not cause.

### Failure 2 — All-narrator output, zero character dialogue
- Episode `signal_lost_ais_unmasking_20260509_182702`, 110 words, "psychological slow-burn", maximum chaos, 3 cast
- Cast declared: ANNOUNCER + JEHOSHAPHAT + LEV. Final ledger: only ANNOUNCER spoke. Treatment.txt confirmed: open_close bookends + 3 SFX + zero scene-body content. JEHOSHAPHAT and LEV had `line_count=0`.
- Final ledger:
  ```
  l001 announcer  "Welcome to Signal Lost. Tonight's broadcast dives into the mysterious."
  l002 announcer  "Suddenly silenced, our tale concludes here. Heed my warning: Keep sharp."
  ```
- 35.9 MB final mp4 with no actual story. Critique pass approved this as valid because it's grammatically clean.

### Failure 3 — 835 tokens emitted, 0 scenes/lines/chars in streaming counter
- Most recent run, 110 words, "psychological slow-burn", 2 cast, self_critique + open_close ON
- Streaming logger counted matches against `=== SCENE`, `[VOICE:]`, dialogue tags. Counter stayed at **0** across all 835 generated tokens.
- Mistral produced 835 tokens of pure prose paragraphs. No screenplay markers appeared anywhere.
- Run continued through pipeline; final mp4 will likely be another all-announcer episode like Failure 2.

### Pre-existing failures (today's session log + memory references)
- BUG-024: "Zero character dialogue in raw script" → cast roster fallback (existing safety net that masks the root issue)
- BUG-LOCAL-061: Mistral Nemo emits `[EDNA, Female, 40s, urgent] Dammit!` shorthand without VOICE: prefix; parser had to be patched
- BUG-LOCAL-038: permissive 2B-fallback parse (3 fallback passes total now)
- BUG-014: zero/single/double asterisk handling around character names

There are **at least 5 known parser fallbacks** today. Each was added to handle a different Mistral format quirk. Pattern: every time Mistral finds a new way to break the format contract, we patch the parser instead of fixing the contract.

---

## Why "fix the prompt" isn't enough

Today's `OTR_LLMScriptWriter` already has:
- A detailed system prompt enforcing `[VOICE: NAME, traits] dialogue` format
- `1940s old-time radio` period-prompt module (`_otr_period_prompts.py`)
- 2 retry attempts with VRAM flush between
- A critique LLM call that scores the script and demands revision if it lacks dialogue
- A revision LLM call that rewrites based on critique feedback
- A FormatNorm pass that normalises bold/markdown variations
- A WORD_EXTEND fallback that calls the LLM again to add lines if word count is short
- A cast-roster fallback that injects character names if zero are detected
- An ARC_ENHANCER pass for structural coherence
- An LLM_RESCUE pass that re-extracts dialogue from prose-form scripts

Despite all of this, today's runs produced:
- Prose monologues with 0 dialogue (Failure 2, 3)
- Format variations the parser can't read (Failure 1)
- Cast members with `line_count=0` after critique + revision approved the script

The accumulated complexity is a sign the architecture is fighting the LLM, not directing it. **Patching prompt #6 won't be different from patching prompts #1-5.**

---

## The candidate architectures

### Architecture A — **Per-line composition with ledger as source of truth** (Jeffrey's proposal)

```
1. Outline call (1 LLM call, JSON output)
   Input: news_seed, style, cast, target_words
   Output: { opener, beats[{speaker, intent, mood, target_dur_s}], closer }

2. Per-line fill (N calls, ~8-20 per episode)
   For each beat:
     Input: outline + last_2_lines_context + this_beat_spec
     Output: just the dialogue text (string), no format tags
     Stamp ledger: ledger.lines.append({line_id, speaker, text, role, ...})

3. Optional polish pass (1 call)
   Read ledger lines, check continuity, fix only what's broken
```

- **Reliability:** bulletproof per-line. You CANNOT get "0 character dialogue" because each call IS for a specific character.
- **Cost:** 10-22 LLM calls per episode (1 outline + 8-20 lines + 1 polish). At ~5-15s per call on Mistral-Nemo, **~3-5 min total LLM phase** vs current ~1-2 min. Slower but predictable.
- **Crash-resilience:** ledger built incrementally. If line 7 fails, retry line 7. Don't lose lines 1-6.
- **Code surface:** ~500-1000 LOC new node. ~1-2 days dev.
- **Risk:** context window grows; need rolling-window strategy for late lines.

### Architecture B — **Outline-then-fill with single JSON-constrained call**

```
1. Outline call (1 LLM call, JSON output)
   Same as A.

2. Single fill call with strict JSON schema (1 LLM call)
   Input: outline + cast spec
   Output: full script as JSON array of {speaker, text, role}
   Use grammar-constrained sampling (XGrammar / outlines / lm-format-enforcer)
   Stamp ledger from validated JSON.
```

- **Reliability:** JSON schema constraint guarantees format. Cannot output prose.
- **Cost:** 2 calls. Single ~1500-token JSON fill. Total ~30-60s LLM phase. **Faster than current.**
- **Code surface:** ~200-400 LOC. Need grammar-constrained sampling library (xgrammar / outlines / Mistral's structured output).
- **Risk:** grammar-constrained sampling on Mistral-Nemo 4-bit NF4 is untested in this stack. May not work cleanly with NF4 quantization. Fallback path needed.
- **Open question:** does HuggingFace `transformers.generate()` with NF4 model support `outlines`/`xgrammar`? Need to verify.

### Architecture C — **Multi-model staged (Claude/Gemini for outline, Mistral for body)**

```
1. Outline call (Claude API or Gemini API, 1 call)
   Strong models nail the JSON outline first try.
   ~$0.01-0.02 per episode.

2. Per-line fill (Mistral-Nemo local, N calls)
   Same as A's step 2. Constrained per-line, vibey period dialogue.

3. Ledger as source of truth, same as A.
```

- **Reliability:** strongest. Claude/Gemini will not fail at structured output. Mistral can't drift from outline because outline is binding.
- **Cost:** Claude Sonnet 4.6 outline ~$0.01-0.05 per episode + local Mistral compute. **Money cost real but small.**
- **Code surface:** ~700-1200 LOC (A + API client glue). Already have OpenAI + Gemini clients in `scripts/_consult_*.py` to template from.
- **Tension with `CLAUDE.md`:** Jeffrey's rule says "no cloud services in audio/video render path." ScriptWriter is **upstream of audio** — the script is regenerated every run, doesn't break C7 byte-identity. Whether this counts as "in the path" is a judgment call.

### Architecture D — **Pure prompt rewrite with worked few-shot examples** (do nothing structural)

```
Replace the current system prompt with:
  - 2-3 fully-worked example episodes in [VOICE:] format
  - Strong negative examples ("DO NOT WRITE PROSE")
  - JSON output instruction with worked example
Keep current single-call architecture.
```

- **Reliability:** modest improvement. Few-shot examples raise compliance from ~60% to maybe ~85% based on standard LLM prompt-engineering. Not bulletproof.
- **Cost:** 30 min sprint. Zero code changes outside `_otr_period_prompts.py`.
- **Risk:** doesn't actually change the architecture. Failures #1, #2, #3 above could still happen on the long tail. Permanent debt.

### Architecture E — **Function calling / structured output via Mistral's tool-use API**

```
Use Mistral-Nemo's function-calling format (or HuggingFace `tool_calls` API)
to require structured JSON output via a declared `submit_episode` schema.
Single call, schema-locked.
```

- **Reliability:** strong if Mistral-Nemo supports it cleanly.
- **Cost:** 1-2 days dev. Need to verify NF4 quantization preserves tool-call capability.
- **Risk:** Mistral-Nemo's tool-use behavior under 4-bit NF4 in transformers is undocumented for this kind of complex schema. Might fall back to free-text emission.

### Architecture F — **Hybrid: outline + per-line fill + ledger** (Claude's pre-synthesis vote)

Combination of A + the JSON-outline rigor of B without grammar-constrained sampling:

```
1. Outline call (Mistral or Claude/Gemini, 1 call)
   Output: JSON outline. Validate; retry once if invalid JSON.

2. Per-line fill (Mistral local, N calls)
   For each beat in outline:
     Prompt with: outline summary + last 2 lines of dialogue + THIS beat's
     {speaker, intent, mood, target_dur_s}.
     Output: ONLY the dialogue text (no formatting tags).
     Stamp ledger.lines.append({line_id, speaker, text, role}).

3. Stop. Critique/revision become unnecessary because each line was
   constrained at composition time.
```

- **Reliability:** ~95%+. Each call is small + constrained.
- **Cost:** ~3-5 min wall, 10-20 calls. ~700-1000 LOC.
- **Risk:** moderate. Context-window management for late lines. Need solid prompt template per beat type (announcer / character / sfx).
- **Why it wins on paper:** structurally cannot produce the failure modes we're hitting. The current architecture is built around "one call → maybe-parseable output"; F is built around "ledger built one row at a time, every row is a contracted commit."

---

## Comparison matrix

| | A (per-line) | B (single JSON) | C (multi-model) | D (prompt only) | E (tool-call) | F (hybrid) |
|---|---|---|---|---|---|---|
| Reliability | ★★★★★ | ★★★★ | ★★★★★ | ★★★ | ★★★★ | ★★★★★ |
| LLM calls/episode | 10-22 | 2 | 10-22 | 1-3 | 1 | 11-21 |
| Wall time | 3-5 min | 30-60s | 2-4 min | 1-2 min | 30-60s | 3-5 min |
| Code complexity | high | medium | high | low | medium-high | high |
| API cost | $0 | $0 | $0.01-0.05/ep | $0 | $0 | $0 |
| Cloud dep | none | none | Claude/Gemini | none | none | optional Claude |
| C7 risk | none | none | none (script not in audio path) | none | none | none |
| Time-to-ship | 1-2 days | 1-2 days | 2-3 days | 30 min | 1-2 days | 2-3 days |
| Long-tail failure modes | very low | low | very low | medium-high | low | very low |
| Patches BUG-061/14/38 today | obviates them | obviates them | obviates them | doesn't | obviates them | obviates them |

---

## Round-robin questions

1. **Which architecture would you actually ship for OTR's needs** (16 GB local-first, narrative quality matters, occasional API call OK upstream of audio)?

2. **Is grammar-constrained sampling (Architecture B / E) viable on Mistral-Nemo 4-bit NF4** loaded via HuggingFace `transformers`? Specifically: does `outlines` / `xgrammar` / `lm-format-enforcer` work with NF4 quantization in 2026? If yes, B becomes much more attractive than A or F.

3. **Per-line context management**: when fill-call N has lines 1..N-1 already in the ledger, what's the right window? Last 2 lines + outline summary? Last 5 lines? Some embedding-retrieved relevant lines? OTR episodes are 30-90s of dialogue, ~10-20 lines max — so simple "last 3 lines" might be enough.

4. **What's the failure mode of F that could come back to bite us?** I'm worried about subtle continuity drift across per-line calls (character X says contradictory things on line 4 vs line 12). Worth a polish pass at end? Worth shipping without one?

5. **Is OPENAI_API_KEY/GEMINI_API_KEY use upstream of the audio path acceptable** under `CLAUDE.md`'s "no cloud services" rule? My read: yes, because the script is regenerated every run and doesn't break C7 byte-identity, but Jeffrey's call.

6. **Short-circuit option:** is there a quick-win prompt-engineering fix (Arch D) that gets us past tonight's blocker (HuMo 88× test still pending) so we can ship the architectural pivot as a v2.1 sprint rather than a midnight rewrite?

7. **Anything we missed:** is there an architecture not on this list (e.g., agentic with tool calls per beat, a fine-tuned LoRA on screenplay data, an ensemble of small cheap models, etc.) that we should consider?

---

## Disposition

- **Status:** open, second-pass round-robin requested
- **Owner:** Claude (synthesizer) → Jeffrey (decision)
- **Decision deadline:** before next ScriptWriter sprint commits
- **Tonight's bridge:** Architecture D (prompt rewrite) or `custom_premise` widget edit if HuMo 88× test must complete first
- **Delete this doc when:** architecture chosen, BUG_LOG entry written, new node shipped (or current node patched)
