# SIGNAL LOST v1.1 QA Tech Brief

**Version**: 1.1 (Beta)
**Date**: 2026-04-06
**Commit**: `ccb5cb1` + pending voice map fix
**Author**: Jeffrey A. Brick / Claude (Co-Authored)
**Status**: QA IN PROGRESS — not shipped

---

## 1. What Changed in v1.1

Three story-quality features were added to the ScriptWriter node (`gemma4_orchestrator.py`), plus critical Bark TTS stability fixes.

### 1.1 Open-Close Expansion

**What it does**: Before writing the full script, the system generates 3 competing story outlines with different creative priorities, then an evaluator LLM pass picks the winner and optionally merges the best elements from the losers.

**Outline priorities**:

- Outline A: CHARACTER-DRIVEN — interpersonal conflict, secrets, breaking points
- Outline B: SCIENCE-DRIVEN — scientific rigor, technical problem-solving under pressure
- Outline C: ATMOSPHERE-DRIVEN — environmental dread, sensory immersion, sound design

**Key constraint**: All 3 outlines MUST root their premise in the real science news fetched from RSS. The prompt explicitly states: "The science news headlines in the system prompt above ARE your raw material. Your premise MUST be rooted in those real headlines."

**Toggle**: `open_close=True` (default on). Skipped when `custom_premise` is set.

**LLM passes added**: 4 (3 outlines at 600 tokens each + 1 evaluator at 800 tokens)

**Method**: `Gemma4ScriptWriter._open_close_expansion()`

**Extraction logic**: Evaluator output searched for `FINAL OUTLINE:` marker first, then `WINNER: Outline N` regex, then full evaluator output as fallback.

### 1.2 Checks and Critiques Loop

**What it does**: After the initial script draft is generated, the system runs a self-critique pass identifying 5-8 specific weaknesses, then a revision pass that rewrites the script addressing those weaknesses.

**Flow**: Draft (full script) → Critique (analytical, lower temperature) → Revise (full rewrite incorporating critique)

**Safety rails**:

- Critique temperature lowered (analytical mode, not creative)
- If critique output < 50 characters → returns original draft unchanged
- If revision output < 60% of original draft length → assumes LLM summarized instead of rewriting → returns original draft
- All exceptions caught → graceful fallback to original draft

**Toggle**: `self_critique=True` (default on)

**LLM passes added**: 2 (1 critique at ~400 tokens + 1 revision at full script length)

**Method**: `Gemma4ScriptWriter._critique_and_revise()`

### 1.3 Context Engineering (Chunked Generation)

**What it does**: For long episodes (>5 min), instead of dumping raw text as context between acts, the system generates a concise LLM summary of each completed act. These summaries are injected as structured context for the next act's generation.

**Context block format**:

```
STORY SO FAR (summaries of previous acts):
  Act 1: [LLM-generated 200-token summary]
  Act 2: [LLM-generated 200-token summary]
LAST LINES (for dialogue continuity):
  [last 500 chars of previous act]
```

**Why**: The old approach dumped 2000 raw characters which diluted the context window. Summaries preserve narrative continuity without wasting tokens on raw dialogue.

**Continuity instruction**: Each act prompt includes: "CONTINUITY CHECK: Before writing, review the story-so-far summaries above. Ensure characters reference earlier events naturally. No amnesia."

**Summary parameters**: 200 max tokens, temperature 0.3 (factual, not creative)

**Method**: `Gemma4ScriptWriter._generate_chunked()` (enhanced, not new)

### 1.4 English-Only Bark Presets

**What changed**: All 38 foreign-language Bark voice presets (de_speaker, fr_speaker, es_speaker, hi_speaker, it_speaker, ja_speaker, ko_speaker, ru_speaker, pt_speaker, pl_speaker) removed from active rotation.

**Why**: Foreign presets caused Bark hallucinations — the model generates foreign-language phonemes when fed English text, producing unintelligible gibberish. Documented in "Test Signal" QA: Lemmy (`v2/de_speaker_0`) was completely unintelligible. French preset lines also showed artifacts.

**Before**: 60% English / 40% foreign accent pool (48 total presets)
**After**: 100% English pool (10 active presets: 6 male, 4 female)

**Lemmy specifically**: `v2/de_speaker_0` → `v2/en_speaker_8` (gravelly, confident, English-native, 40s-50s). Same character energy, stable phonemes.

**Accent pool**: `_ACCENTS` reduced to single entry: `("neutral", "en", 1.00)`

**Foreign presets preserved as comments** for future reference if Bark's multilingual stability improves.

### 1.5 Director Voice Map Fix

**Bug**: `_randomize_character_names()` was using procedurally generated names (e.g., "BLAKE ARCHER") as dict keys in `voice_assignments`, but the script text still contains the original names (e.g., "HAYES"). When BatchBark looked up "HAYES" in the voice map, it found no match and fell through to a random hash fallback.

**Fix**: Voice assignment dict now uses `upper_name` (the original script character name) as the key. The procedural name is stored in the `notes` field for the treatment file.

**Before**: `new_voice_assignments[profile["name"]]` → key = "BLAKE ARCHER"
**After**: `new_voice_assignments[upper_name]` → key = "HAYES"

**BatchBark lookup confirmed**: `batch_bark_generator.py` line 153 does `voice_map.get(character, {})` where `character` comes from `[VOICE: HAYES, ...]` tags — direct match now works.

---

## 2. File Inventory

| File | Lines | Role |
|------|-------|------|
| `nodes/gemma4_orchestrator.py` | 2,465 | ScriptWriter + Director (core orchestrator) |
| `nodes/batch_bark_generator.py` | 654 | Bark TTS batch processing |
| `nodes/video_engine.py` | 1,251 | CRT video renderer |
| `nodes/scene_sequencer.py` | 947 | Audio assembly + mixing |
| `nodes/audio_enhance.py` | 351 | Vintage radio DSP filter chain |
| `nodes/sfx_generator.py` | 322 | Procedural SFX synthesis |
| `nodes/vintage_radio_filter.py` | 397 | Analog warmth / degradation |
| `nodes/bark_tts.py` | 406 | Single-line Bark wrapper |
| `nodes/parler_tts.py` | 281 | Parler TTS wrapper |
| `nodes/audio_batcher.py` | 134 | Audio batch utilities |
| **Total** | **7,209** | |

### Class/Method Map (gemma4_orchestrator.py)

```
GemmaHeartbeatStreamer
  ├── __init__()
  ├── put()           — token-level heartbeat to otr_runtime.log
  ├── end()           — final stats report
  └── _process_line() — detect SCENE/VOICE/SFX/ENV tags

Gemma4ScriptWriter
  ├── INPUT_TYPES()                    — widget definitions (required + optional)
  ├── IS_CHANGED()                     — always re-execute (news changes daily)
  ├── write_script()                   — main entry point + Phase 1 pre-flight
  ├── _open_close_expansion()          — [v1.1] try/except wrapper
  ├── _open_close_expansion_inner()    — [v1.1] 3 outlines + evaluator + Phase 2a guards
  ├── _critique_and_revise()           — [v1.1] Draft → Critique → Revise + Phase 2b/2c guards
  ├── _generate_chunked()              — [v1.1 enhanced] act-by-act + Phase 3a/3b context hardening
  └── _parse_script()                  — Canonical 1.0 JSON extraction

Gemma4Director
  ├── INPUT_TYPES()
  ├── direct()                     — production plan generation
  ├── _extract_json()              — JSON extraction from LLM output
  └── _randomize_character_names() — [v1.1 fixed] procedural cast profiles
```

---

## 3. Pipeline Flow (v1.1)

```
RSS Science News
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  OPEN-CLOSE EXPANSION (if open_close=True)          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Outline A │  │ Outline B │  │ Outline C │         │
│  │ CHARACTER │  │ SCIENCE  │  │ATMOSPHERE │         │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │
│        └───────────┬───────────┘                     │
│                    ▼                                  │
│              EVALUATOR                                │
│         (picks winner, merges)                        │
└───────────────────┬─────────────────────────────────┘
                    ▼
           FULL SCRIPT GENERATION
          (winning outline as roadmap)
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  CHECKS & CRITIQUES (if self_critique=True)         │
│  Draft → Critique (5-8 weaknesses) → Revise         │
└───────────────────┬─────────────────────────────────┘
                    ▼
           CONTENT FILTER + CITATION GUARD
                    │
                    ▼
           PARSE TO CANONICAL 1.0 JSON
                    │
                    ▼
           DIRECTOR (voice assignments)
           [v1.1: keys = original script names]
                    │
                    ▼
           BATCH BARK TTS
           [v1.1: English-only presets]
                    │
                    ▼
           SCENE SEQUENCER → AUDIO ENHANCE
                    │
                    ▼
           EPISODE ASSEMBLER → SIGNAL LOST VIDEO
```

---

## 4. Widget Configuration

### ScriptWriter Node

| Widget | Type | Default | Notes |
|--------|------|---------|-------|
| `episode_title` | STRING | "The Last Frequency" | Required |
| `genre_flavor` | ENUM | hard_sci_fi | 8 options |
| `target_minutes` | INT | 25 | 1-45, use 1 for quick test |
| `num_characters` | INT | 4 | 2-8 (plus ANNOUNCER) |
| `model_id` | STRING | google/gemma-4-E4B-it | Optional |
| `custom_premise` | STRING | "" | Overrides news generation |
| `news_headlines` | INT | 3 | 1-5 RSS articles |
| `temperature` | FLOAT | 0.8 | 0.1-1.5 |
| `include_act_breaks` | BOOL | True | Sponsor message breaks |
| `self_critique` | BOOL | True | **[v1.1]** Critique loop |
| `open_close` | BOOL | True | **[v1.1]** Outline competition |

---

## 5. Test Run Results

### Test Run: "Test Signal" (2026-04-06 15:07-15:17)

**Configuration**: 1-minute, hard_sci_fi, 3 characters, open_close=True, self_critique=True, custom_premise="" (generative)

**Science seed**: FDA approves gene therapy for severe leukocyte adhesion deficiency-I (real news)

**Characters generated**: DR_VOSS, HAYES, LEMMY (plus ANNOUNCER) — all procedurally generated from news

**Open-Close results**:

- 3 outlines generated (CHARACTER, SCIENCE, ATMOSPHERE-driven)
- Evaluator picked Outline 1, merged science specificity from Outline 2 and atmospheric SFX from Outline 3

**Critique results**:

- 8 weaknesses identified (character archetypes, pacing, Lemmy's confession timing, SFX genericity)
- Revision produced 3,587 chars — accepted (passed 60% length threshold)

**Output**:

- 3 scenes, 16 dialogue lines, 7 SFX cues
- 197.5 seconds audio (3.4 minutes from 1-minute target — richer script expanded output)
- 86.8 MB MP4 at 1280x720 @ 24fps

**QA Findings**:

| Area | Rating | Notes |
|------|--------|-------|
| Script quality | PASS | Strong arc: setup → revelation → climax → payoff |
| Science grounding | PASS | FDA gene therapy drove entire plot premise |
| Generative characters | PASS | Different cast every run, no hardcoding |
| Pacing | PASS | Snappy, frantic dialogue matching urgency |
| Sonic environment | PASS | DSP chain working, vintage warmth present |
| DR_VOSS voice (Bark) | PASS | Compelling, urgent, authentic delivery |
| LEMMY voice (Bark) | FAIL | de_speaker_0 produced unintelligible gibberish |
| French preset lines | FAIL | Artifacts observed in accented presets |
| Voice map matching | FAIL | Director renamed chars; BatchBark couldn't match |
| Visual sync | PASS | Oscilloscope + telemetry overlays flawless |

---

## 6. Bugs Found and Fixed

### BUG-001: Lemmy Bark Hallucination (FIXED)

- **Severity**: High
- **Root cause**: `v2/de_speaker_0` (German-language Bark preset) generates foreign phonemes on English text
- **Fix**: Changed to `v2/en_speaker_8` (English-native, gravelly, confident, 40s-50s)
- **File**: `gemma4_orchestrator.py` line ~260 (`_LEMMY_PROFILE`)
- **Status**: Fixed, awaiting verification

### BUG-002: Foreign Preset Hallucinations (FIXED)

- **Severity**: High
- **Root cause**: All non-English Bark presets (de, fr, es, hi, it, ja, ko, ru, pt, pl) can hallucinate when fed English text
- **Fix**: Removed all 38 foreign presets from `_VOICE_PROFILES`. Set `_ACCENTS` to 100% English. 10 English presets remain (6 male, 4 female).
- **File**: `gemma4_orchestrator.py` lines ~162-233
- **Status**: Fixed, awaiting verification

### BUG-003: Director Voice Map Mismatch (FIXED)

- **Severity**: Medium
- **Root cause**: `_randomize_character_names()` used procedural names as dict keys (e.g., "BLAKE ARCHER") but script text retains original names (e.g., "HAYES"). BatchBark lookup fails, falls to random hash fallback.
- **Fix**: Voice map keys now use `upper_name` (original script name). Procedural name stored in `notes` field.
- **File**: `gemma4_orchestrator.py` line ~2178
- **Status**: Fixed, awaiting verification

---

## 7. Performance Impact

v1.1 features add significant LLM inference time. All timings at Gemma 4 E4B on RTX 5080 (~2-4 tok/s):

| Phase | v1.0 | v1.1 (both on) | Delta |
|-------|------|-----------------|-------|
| Open-Close (3 outlines) | N/A | ~12-15 min | +12-15 min |
| Evaluator | N/A | ~3-4 min | +3-4 min |
| Full script (1 min ep) | ~3-5 min | ~4-7 min | +1-2 min (longer prompt) |
| Critique pass | N/A | ~2-3 min | +2-3 min |
| Revision pass | N/A | ~4-6 min | +4-6 min |
| Director | ~2-3 min | ~2-3 min | unchanged |
| BatchBark | ~3-5 min | ~5-7 min | +2 min (more lines) |
| Video render | ~5 min | ~5 min | unchanged |
| **Total (1 min test)** | **~15-20 min** | **~35-50 min** | **~2x slower** |

Both features can be toggled off (`self_critique=False`, `open_close=False`) for quick iteration.

---

## 8. Diagnostic Logging

v1.1 added runtime log markers for debugging feature activation:

```
ScriptWriter: PARAMS open_close=True self_critique=True custom_premise=(empty) target_min=1 chars=3
ScriptWriter: OPEN-CLOSE CHECK: open_close=True (type=bool), condition=True
OPENCLOSE: Generating 3 competing outlines
OPENCLOSE: Outline CHARACTER-DRIVEN done (2366 chars)
OPENCLOSE: Outline SCIENCE-DRIVEN done (2268 chars)
OPENCLOSE: Outline ATMOSPHERE-DRIVEN done (2289 chars)
OPENCLOSE: Evaluator picking winner
OPENCLOSE: Evaluator done (1847 chars)
ScriptWriter: CRITIQUE CHECK: self_critique=True (type=bool)
ScriptWriter: >>> ENTERING critique_and_revise
CRITIQUE: Starting self-critique pass
CRITIQUE: Critique pass done (1203 chars)
CRITIQUE: Revision done (3587 chars)
CRITIQUE: Revised script accepted
ScriptWriter: <<< EXITED critique_and_revise
```

Monitor live: `Get-Content "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr_runtime.log" -Tail 20 -Wait`

---

## 9. Guardrail System (Phases 1-3)

### Phase 1: Pre-Flight and Input Validation

| Guard | What It Does | Trigger |
|-------|-------------|---------|
| Param sanity | Clamps num_characters to 3 for 1-min episodes; disables act breaks for episodes under 2 min | `PREFLIGHT:` in log |
| Custom premise enforcement | When custom_premise set: bypasses RSS, disables Open-Close, zero context contamination | `PREFLIGHT: Custom premise detected` |
| Global token budget | Normalizes target_minutes into target_words and target_chars at init, passed to all prompts | Always |
| Episode fingerprint | SHA-256 hash of params for reproducibility | `FINGERPRINT` in log |
| RSS fallback seeds | 3 hardcoded real science seeds (deep-sea microbes, quantum entanglement, CRISPR gene drive) if all RSS feeds fail | `PREFLIGHT: RSS_FALLBACK` |
| Headline sanitization | Strip emojis/non-ASCII, cap at 280 chars, normalize whitespace on all headlines | Always |
| Cast map verification | Extract all VOICE tag names from parsed script, log for downstream matching | `CAST_MAP` in log |
| QA debug dump | JSON payload with fingerprint, params, news seed, cast, stats saved alongside MP4 | `QA_DUMP` in log |

### Phase 2: LLM Generation Guardrails

| Guard | What It Does | Trigger |
|-------|-------------|---------|
| Open-Close boundary | Discard outlines outside 200-3000 char range before evaluator | `OPENCLOSE: DISCARDED/TRUNCATED` |
| Open-Close top-level catch | Entire method wrapped in try/except; any crash falls back to v1.0 direct generation | `OPENCLOSE_FALLBACK` |
| Critique format validation | Verify critique has numbered markers and weakness keywords; reject rewrites | `CRITIQUE_SKIPPED — critique format invalid` |
| Revision length floor | Reject revision < 60% of draft (summary, not rewrite) | `CRITIQUE_SKIPPED — revision too short` |
| Revision length ceiling | Reject revision > 250% of draft (runaway expansion) | `CRITIQUE_SKIPPED — revision too long` |
| Levenshtein similarity | Reject if similarity > 95% (lazy copy) or < 35% (hallucinated new story) | `CRITIQUE_SKIPPED — copy/hallucination` |

### Phase 3: Context Chunking and Observability

| Guard | What It Does | Trigger |
|-------|-------------|---------|
| Act summary hardening | If LLM summary < 50 chars, falls back to mechanical summary (scene titles + last 8 dialogue lines) | Warning in log |
| Strict truncation | LAST LINES capped at 500 chars with `... [truncated]` marker | Always |
| Standardized telemetry | All fallback events use grep-able markers: `CRITIQUE_SKIPPED`, `OPENCLOSE_FALLBACK`, `RSS_FALLBACK`, `PREFLIGHT` | Log grep |
| QA debug JSON | Per-episode JSON with fingerprint, outline ID, critique stats, length ratios, cast map | `qa_debug_*.json` in output dir |

---

## 10. Known Limitations

1. **Performance**: v1.1 roughly doubles ScriptWriter phase time. Acceptable for production runs, slow for iteration.
2. **10 English presets only**: Limits voice diversity for large casts (8 characters). De-duplication logic handles this but larger casts may get similar-sounding voices.
3. **Critique may over-expand**: 1-minute target produced 3.4 minutes of output. Now mitigated by 250% ceiling guardrail.
4. **Context Engineering untested**: The act summary feature only activates for episodes >5 minutes. "Test Signal" was 1 minute, so this feature was not exercised in QA.
5. **Wall-clock timeouts**: Not yet implemented for individual LLM phases. Long outlines or revisions could still hang.

---

## 10. Verification Checklist (Next Test Run)

- [ ] ComfyUI restarted after code changes
- [ ] Diagnostic log confirms `open_close=True`, `self_critique=True`
- [ ] OPENCLOSE markers appear in runtime log (3 outlines + evaluator)
- [ ] CRITIQUE markers appear (critique + revision)
- [ ] Lemmy dialogue is intelligible (en_speaker_8)
- [ ] No foreign-language artifacts in any voice
- [ ] Voice map keys match script character names in runtime log
- [ ] All characters have distinct, appropriate voices
- [ ] Science news drives the story premise (not generic sci-fi)
- [ ] Output MP4 renders without errors

---

## 11. Git History

```
ccb5cb1  feat(v1.1): Open-Close + Critique + Context Engineering + English-only Bark presets
963ba5a  perf: Director token budget slashed, Bark CPU fallback, ThreadPool video
ecac016  perf: Director token budget slashed 50-64% + Bark CPU fallback
b124ffc  perf: slash Director token budget 50-64% + prompt tightening
8f811e2  perf: slash Director token budget 50-64% — procedural engine handles casting
0138838  fix: ANNOUNCER pool now 50/50 gender balance (3 male, 3 female)
```

**Pending commit**: Voice map fix (`upper_name` as key) — not yet pushed.

---

## 12. Next Steps

1. Restart ComfyUI, run "Test Signal" again with all fixes
2. Verify Lemmy voice, voice matching, feature activation
3. Commit voice map fix
4. Test a longer episode (5+ minutes) to exercise Context Engineering
5. Evaluate Round 2 features from research doc: CosyVoice 3.0, NeuTTS Air, Stable Audio Open 1.0, LightRAG Story Bible
