# Problem Statement: Multi-Turn Polish & Ledger Coherence for Round-Robin

**Project:** ComfyUI-OldTimeRadio v2.0-alpha
**Date:** 2026-05-11
**Owner:** Jeffrey Brick
**Branch HEAD at time of writing:** `c3f5e3b`

**Round-robin participants requested:** ChatGPT (gpt-5.5 or similar), Gemini (2.5+), Claude synthesis.

**Hard rule:** Don't propose anything that requires cloud, paid APIs, or non-local LLMs. Target is local 7B–14B class (Mistral-Nemo, Gemma-2, Qwen2.5) on a single RTX 5080 Laptop, 14.5 GB VRAM ceiling.

---

## 0. TL;DR

Today the OTR pipeline has a single optional "polish" LLM call that runs once per leaky dialogue line, gated by a small narration-leak regex set, default OFF. That's the entire post-composition cleanup story. The composer is one-shot per beat; there is no scene-level review, no episode-level continuity pass, no audio-readiness check, no video-readiness check, and no explicit ledger gap audit before the writer hands off downstream. The reviewer pass that does exist (`OTR_LedgerScriptReviewer`) does cast-contract auditing, not narrative coherence or surface cleanup.

I want a round-robin recommendation for **how to split the cleanup phase into multiple discrete turns** so an LLM can:

1. iteratively polish the dialogue surface,
2. ensure narrative coherence across the whole script,
3. clean up the ledger so there are no gaps when it is delivered to the downstream audio-gen and video-gen consumers.

A secondary question this surfaces: at one point the writer emitted a `script_parse_json` intermediate that downstream consumers read. I believe the ledger has fully replaced that as the canonical source for the downstream chain, but I want this verified explicitly in the round-robin so my mental model is correct.

---

## 1. Pipeline architecture, ground state

```
   ┌─────────────────────────────┐
   │ OTR_LedgerScriptWriter      │
   │   - news_interpreter (1 LLM)│
   │   - style_picker (2 LLM)    │
   │   - cast lock               │
   │   - outline LLM (1 LLM)     │
   │   - per-beat compose_line   │  <-- ~12-20 LLM calls
   │     (optional polish pass)  │  <-- 0 to N extra LLM calls
   │   - news_close_brief overlay│
   │   - title regen (1 LLM)     │
   └────────────┬────────────────┘
                │  (script_text, script_json, news_used, est_minutes)
                │  + in-memory Ledger singleton via _PL.get_ledger()
                ▼
   ┌─────────────────────────────┐
   │ OTR_LedgerScriptReviewer    │  <-- 3 passes (Auditor / Doctor / Auditor)
   │   - cast-contract audit     │
   │   - deterministic repairs   │
   │   - LLM script doctor       │  <-- ~3 LLM calls
   │   - phantom-skip fallback   │
   └────────────┬────────────────┘
                │  (script_text, script_json, news_used, est_minutes, verdict)
                ▼
   ┌─────────────────────────────┐
   │ OTR_LLMDirector             │
   │   - production_plan_json    │  <-- 1 LLM call
   └────────────┬────────────────┘
                │
                ▼
   Audio chain (Bark / Kokoro / AudioGen / Procedural SFX / MusicGen)
   Sequencer (assembles per-line waveforms + sfx_audio_clips into scene_audio)
   AudioEnhance
   EpisodeAssembler (mp3/wav out)
                │
                ▼
   Video chain (Flux portraits / HuMo character clips / LTX motion / VideoComposite / RTXUpscale)
```

The ledger schema is the L3 format (`schema_version: "l3-2026-05-14"`). The in-memory `Ledger` singleton (in `nodes/production_ledger.py`) is the canonical source — `script_json` is the serialized form, `script_text` is an assembled view. Downstream consumers were rewritten in the 2026-05-09/10 Ledger Consumer Rewrite sprint to read the ledger handle directly (7 of 7 consumers shipped green) but the STRING-wire outputs are still preserved as the public contract.

**Confirmation needed in round-robin:** is there any downstream node still reading the legacy `script_parse_json` shape, or has the ledger fully replaced it?

---

## 2. The current polish phase — full detail

### 2.1 Where it lives

`nodes/_otr_line_composer.py`, inside `compose_line()`. Runs AFTER the composer's retry ladder (2 attempts max) closes with a successful cleaned line, BEFORE the phantom-name detection gate.

### 2.2 Activation

Gated by:

1. The writer's `enable_polish_pass` BOOLEAN widget (default `False`, opt-in per episode).
2. `needs_polish(cleaned)` returning `True`.

Both must be true. When `enable_polish_pass=False`, the polish path is dead code — composer output ships unchanged.

### 2.3 The narration-leak regex set (`_NARRATION_LEAK_PATTERNS`)

Five patterns, case-insensitive:

```python
_NARRATION_LEAK_PATTERNS: tuple[str, ...] = (
    # 1. Pronoun-action narration verbs
    r"\b(?:he|she|they)\s+(?:said|replied|added|asked|whispered|"
    r"shouted|paused|continued|murmured|exclaimed|"
    r"pauses|smiles|nods|shrugs|coughs|looks|turns|leans|stares)\b",
    # 2. Opens with a quote mark (smart or straight, unpaired)
    r'^["“‘]',
    # 3. Markdown / asterisk-wrapped action
    r"\*[^*]+\*",
    # 4. Bracket stage direction
    r"\[[^\]]+\]",
    # 5. Parenthesized cue verb
    r"\([^)]*(?:sigh|pause|beat|laughs?|smiles?|gestures?|nods?|"
    r"shrugs?|cough)[^)]*\)",
)
```

`needs_polish(line)` returns `True` if ANY pattern matches.

### 2.4 The polish prompt (verbatim)

```
You are a script editor cleaning one line of radio drama dialogue.
The line below leaked narration or stage direction. Rewrite it as
pure spoken dialogue.

OUTPUT RULES - strict:
- Only the words the character speaks out loud.
- No name, no colon, no quotes, no brackets, no parentheses.
- No "he said" / "she replied" / narration of any kind.
- Preserve the character's intent. Preserve the speaker's voice.
- Keep within plus or minus 20% of the original word count.

Output the cleaned line and stop. Nothing else.
```

The user-prompt body is:

```
CHARACTER: <speaker_voice_card>
ORIGINAL LINE: <leaked_line>
```

That's it. **The polish prompt knows the speaker's voice card and the leaked line. It does NOT know:**

- the beat the line was meant to accomplish (`beat.intent`)
- the POSITION in the arc (`<phase>, beat N of M`)
- the recent dialogue (`last_lines` window)
- the style / theme / episode context
- the named-entities roster (allowed_people / allowed_things)
- whether the speaker is a character or the announcer
- whether the line was supposed to leak (e.g. an announcer beat where some narration is by design)

### 2.5 Sampling

- `temperature = 0.4` (much lower than composer's 0.7/0.8 baseline — targeted edit)
- `max_new_tokens = max(40, orig_word_count * 3)`
- `top_p`, `min_p`, `repetition_penalty` — inherited from the writer's closure-captured generate_fn (currently 0.92 / 0.05 / 1.03 in default-flipped form). Polish sees whatever the composer was tuned to.

### 2.6 Post-polish disposition

```python
polished_clean = strip_line_formatting(polished or "")
if polished_clean:
    if needs_polish(polished_clean):
        # polish_still_leaky -- keep pre-polish text
    else:
        # word-cap recheck (Tier 2 #14)
        if polished outside band [target*0.5..target*1.7] or > 3*target:
            # polish_overshoot -- keep pre-polish text
        else:
            cleaned = polished_clean  # accept polish
```

Then the phantom-name gate runs over `cleaned` (whichever survived), so any new proper noun the polish introduced gets flagged. That ordering is locked by a test (`TestPolishBeforePhantom`).

### 2.7 Known gaps in the current polish phase (acknowledged but not fixed)

1. **No context beyond voice card** — polish has ~10% of the context the composer had.
2. **Announcer beats mishandled** — polish prompt forbids narration; announcer beats are *by design* narration. No `speaker_role` guard.
3. **No phantom-name awareness** — polish can invent names; only the downstream phantom gate catches them (and only flags them, doesn't fix).
4. **No refusal detector** — `"I cannot rewrite this."` ships as the polished line.
5. **Parameters never validated** — temp 0.4, mnt multiplier 3, ±20% band are declared, not measured.
6. **Polish closure-sampling leak** — composer-tuned min_p / repetition_penalty leak into polish via the closure (deferred fix).
7. **Single-turn** — one LLM call, one shot, no iterative refinement.

---

## 3. What I actually want — the round-robin question

I want **multiple discrete cleanup phases** between the composer and the downstream audio/video chain. Each phase is one or more LLM calls with a tight scope. The phases run sequentially, each operating on the ledger left by the previous one, and each can write back to the ledger so the final hand-off downstream is a single coherent artifact with no gaps.

I have rough ideas about what those phases could be, but I want a round-robin to challenge / reorder / merge / split them and to give me a concrete architecture I can implement.

### 3.1 Phases I am considering (not committed to)

| # | Phase | Scope | Output mutation |
|---|---|---|---|
| A | Per-line surface polish (current) | one leaky line | rewrite `line.text` |
| B | Per-line continuity check | one line + LAST 5 lines + speaker voice card | rewrite if voice drift |
| C | Per-scene cleanup | one full scene between music markers | rewrite any line(s) for scene-internal coherence |
| D | Episode arc check | full ledger | flag / rewrite lines that don't serve the arc |
| E | Cast voice consistency | full ledger grouped by speaker | rewrite line(s) that don't sound like the same character |
| F | Named-entity audit | full ledger + allowed_names roster | replace phantom names, fix typos, normalize aliases |
| G | Audio-readiness pass | full ledger | flag SSML hazards, hard-to-pronounce words, unsafe-for-TTS punctuation |
| H | Video-readiness pass | full ledger | flag visual cues that can't be storyboarded (off-camera referents, etc.) |
| I | Ledger gap audit | full ledger schema | structural check: every line has char_id, traits, speaker_role, start_s/dur_s after assembly, etc. |

The existing `OTR_LedgerScriptReviewer` covers a slice of phase F (cast-contract phantom-skip + deterministic repair) and a slice of A/B (LLM script doctor proposes ≤edit_cap rewrites with cast-locked output). It does NOT cover C/D/E/G/H/I.

### 3.2 What I want the round-robin to decide

1. **Which of A-I are valuable to add and which are noise?**
2. **What is the right ORDER?** Some phases mutate text (could create new phantoms / new audio hazards downstream). Order matters.
3. **What's the right level of GRANULARITY for each phase — per-line, per-scene, or per-episode LLM call?** Per-line is cheap but myopic. Per-episode is global but token-heavy. Per-scene is the middle.
4. **What's the LLM cost budget?** Today a 350-word run is ~20 LLM calls (compose) + ~3 (reviewer) + ~3 (style + outline + title). Adding 5-10 more phases at per-line granularity would 2-3x the wall-clock. Per-scene at 3-5 scenes per episode is more manageable.
5. **Where is the boundary between "writer's responsibility" and "polish phase"?** Today the writer owns surface format (strip_line_formatting), news-close-brief override, and title regen. Should it also own the surface polish, or should that all live in a dedicated post-writer node?
6. **Can a single LLM call cover MULTIPLE coherence checks** (e.g. one call that audits cast voice + named entities + audio hazards), or should each phase be its own specialist call so failures are localized?
7. **How should phases interact with the existing 3-pass reviewer** (Pass 1 cast audit / Pass 2 script doctor / Pass 3 cast audit)? Do they sit before the reviewer, after, or interleaved?
8. **What's the failure mode for each phase?** Hard-fail the episode? Warn-and-proceed? Fall back to pre-phase text? Today's polish does fall-back-to-original; I'd default everything to that policy unless there's a reason to escalate.

### 3.3 Critical constraint: ledger coherence at hand-off

When the writer chain finishes, the ledger is what the downstream audio and video nodes consume. The audio chain reads `led.data["lines"]` per beat for TTS; the video chain reads `led.data` for cast portraits and the HuMo / LTX per-line clips. If a line's `text` mutates AFTER audio renders, the audio is wrong. If a phantom name is left unresolved in the text, TTS will speak it. If the speaker_role is wrong, the wrong voice fires.

**The ledger MUST be byte-stable by the time audio gen starts.** Any cleanup phases that mutate text MUST run before audio. Ideally before the reviewer's Pass 3 final audit, so the audit sees the final shape.

### 3.4 Downstream consumer contract — the legacy `script_parse_json` question

Historically the writer emitted a `script_parse_json` intermediate (legacy `OTR_LLMScriptWriter` ledger format) and downstream nodes parsed it. The L3 sprint (2026-05-09/10) rewrote 7 of 7 consumers to read the in-memory ledger handle (`production_ledger.get_ledger()`) plus the L3 schema, and `script_parse_json` was retired as an output name.

**Today's contract (please verify in round-robin):**

- Writer returns `(script_text, script_json, news_used, estimated_minutes)`.
- `script_text` is the assembled `[VOICE: NAME, traits] text` / `[SFX: text]` view of the ledger (Tier 1 fix derives it from the ledger so it is always in sync with on-disk state).
- `script_json` is `json.dumps(led.data)` — the canonical L3 ledger schema.
- The in-memory `Ledger` singleton is also exposed via `_PL.get_ledger()` for callers that want object access.

Consumers I believe are now ledger-native:

- Director (reads `script_text`)
- Sequencer (reads `script_json` + ledger handle for line-level metadata)
- Bark / Kokoro / AudioGen / Procedural SFX (read ledger lines)
- Episode Assembler (reads ledger)
- Director-driven production plan (reads ledger handle)
- Signal Lost Video (reads ledger + cast)
- HuMo / LTX batch renderers (read ledger + cast)

**Question for round-robin:** confirm or disprove that any path still consumes a legacy `script_parse_json` shape. If yes, that consumer is a bottleneck and a multi-pass cleanup phase that writes only to the ledger would silently desync from it.

---

## 4. Multi-turn polish — what I think the architecture should look like

A first draft for the round-robin to challenge. Phases run in order. Each phase has its own LLM call (or no LLM if pure deterministic). Each phase is allowed to mutate `led.data["lines"][i].text` and `meta.<phase>_record` only. Other ledger fields are immutable post-writer.

```
Writer output (ledger has skeleton + composed text)
   │
   ▼
Phase 1: Per-line surface polish (current polish_line, regex-gated)
   - Optional, default OFF for now
   - Scope: one line at a time
   - LLM: 0..N calls (one per leaky line)
   - Mutates: line.text on accepted polishes
   - Failure: keep pre-polish text
   │
   ▼
Phase 2: Per-line continuity (NEW)
   - Scope: one line + last 5 + voice card + beat intent
   - Trigger: cheap heuristic (voice drift score, e.g. average line length
     for this speaker deviates by >40%, or vocabulary diversity collapses)
   - LLM: 0..N calls
   - Mutates: line.text
   - Failure: keep pre-phase text
   │
   ▼
Phase 3: Scene-level coherence (NEW)
   - Scope: one full scene (lines between music markers)
   - Trigger: always (one call per scene)
   - LLM: 1 call per scene (typically 3-5 calls per episode at 350-word target)
   - Prompt: full scene + outline beats for the scene + cast cards
   - Mutates: any line in the scene
   - Failure: keep pre-phase scene
   │
   ▼
Phase 4: Episode arc + cast voice (NEW)
   - Scope: full ledger
   - Trigger: always (one call per episode)
   - LLM: 1-2 calls
   - Prompt: full lined-up script + arc structure + cast cards + allowed names
   - Mutates: any line, but limited to edit_cap (e.g. 5 edits)
   - Failure: keep pre-phase ledger
   │
   ▼
Phase 5: Named-entity audit (NEW or fold into existing reviewer)
   - Scope: full ledger
   - Trigger: always
   - LLM: 0 (deterministic) for normalization; 1 for ambiguous-cases
   - Mutates: line.text to replace phantoms with allowed names
   - Failure: keep flagged compose_flags as-is
   │
   ▼
Phase 6: Audio-readiness (NEW)
   - Scope: full ledger
   - Trigger: always
   - LLM: 0 (pure regex / pronunciation lookup) preferred; 1 for ambiguous
   - Mutates: line.text to fix unsafe-for-TTS punctuation, expand uncommon
     abbreviations, replace symbol characters with words
   - Failure: keep pre-phase text, log warning
   │
   ▼
Phase 7: Video-readiness (NEW)
   - Scope: full ledger
   - Trigger: always
   - LLM: 0 (deterministic check of cast portrait fields, scene shot
     duration calc, etc.)
   - Mutates: ledger.meta.video_readiness with a hard pass/warn/fail signal
   - Failure: hard-fail or warn depending on severity
   │
   ▼
Phase 8: Ledger gap audit (NEW, pure deterministic)
   - Scope: full ledger
   - Trigger: always
   - LLM: 0
   - Mutates: nothing — read-only audit
   - Checks: every line has non-empty text OR is non-voiced; every voiced
     line has a char_id mapped in cast; every char_id in cast is used at
     least once OR is the announcer; every key_term from meta.news.key_terms
     either landed or is on the missing-but-warn list; ledger.meta.style,
     meta.episode_title, meta.gen_params_initial all stamped
   - Failure: hard-fail if any structural gap; warn for soft gaps
```

Phases 1-5 own dialogue cleanup; phases 6-8 own hand-off readiness. The existing 3-pass reviewer's cast-contract audit could be folded into phase 5, or sit beside it. Question for round-robin.

### 4.1 Token budget envelope I am willing to spend

- Today: ~25 LLM calls for a 350-word episode.
- Acceptable: ~40 LLM calls (60% increase, roughly +3-5 min wall clock on Mistral-Nemo at typical step rates).
- Not acceptable: ~80 LLM calls (3x cost). Polish has to stay smart, not brute-force.

So the round-robin needs to weigh whether each phase pulls its weight at the cost it adds.

### 4.2 What I am NOT asking the round-robin to decide

- The LLM model itself (Mistral-Nemo is the default; see CLAUDE.md).
- The widget surface (we'll wire whatever the round-robin recommends).
- The exact regex set inside any single phase (engineering detail).
- Reviewer's existing 3-pass shape (it works today and is locked).
- The L3 ledger schema (locked).

---

## 5. Specific questions I want the round-robin to answer

1. **Is the multi-phase architecture in §4 the right shape, or am I overengineering?** A single beefier LLM call covering A + C + E in one shot might be more efficient than 3 separate phases.

2. **What is the right LLM-call ORDER between surface polish and coherence checks?** Today's polish runs on raw composer output; a coherence pass would benefit from polished output. But a polish that introduces a phantom needs the coherence pass to catch it. There's a dependency cycle if both can mutate text.

3. **Should announcer beats skip phases 1-2 entirely, or use a different prompt?** Announcer is by design narration; current polish prompt forbids it.

4. **Where should the named-entity audit live — phase 5 (post-coherence) or earlier?** Current reviewer puts it at Pass 1 (pre-doctor) and Pass 3 (post-doctor). My phase 5 sits after coherence rewrites. Two answers possible; pick one.

5. **For phase 3 (scene-level), what's the right scene boundary detection?** Music markers (music_open, music_inter, music_close) are explicit; some episodes may not have music_inter beats, in which case the entire body is one scene. Is that acceptable, or do we need a smarter scene detector?

6. **For phase 4 (episode arc), how do we keep the LLM honest about the edit_cap?** The existing reviewer has `edit_cap = min(8, max(3, voiced_beats // 3))`. Should phase 4 use the same cap?

7. **For phase 6 (audio-readiness), can a pure-deterministic regex set + pronunciation lookup table (e.g. CMU dict + a small custom dict) replace an LLM call?** I'd prefer deterministic for this phase because TTS-safety failures are reproducible.

8. **For phase 8 (gap audit), what's the canonical list of structural invariants to check?** I have a partial list; I want a definitive one.

9. **Is the `script_parse_json` legacy contract fully retired, or is there still a downstream consumer that expects it?** This is a "look at the current code" question, not a design question.

10. **Should any phase be allowed to ADD or REMOVE lines, or only mutate existing line.text?** My instinct is mutate-only — adding/removing lines breaks line_id stability, which breaks audio-clip-to-line bindings. But there may be cases (phantom-skip) where removal is the right answer.

11. **What's the right failure-cascade policy?** If phase 3 fails (LLM crash, malformed output, etc.), do we skip to phase 4 and continue, or abort and ship the pre-phase-3 ledger? I prefer skip-and-continue with WARN logs; the round-robin should validate.

12. **Is there a phase I am missing?**

---

## 6. Reference: code and contracts the round-robin should peek at

(All paths relative to repo root.)

- `nodes/_otr_line_composer.py` — `compose_line`, `polish_line`, `needs_polish`, `_NARRATION_LEAK_PATTERNS`, `_POLISH_SYSTEM_PROMPT`, `detect_phantom_names`, `strip_line_formatting`
- `nodes/_otr_ledger_reviewer.py` — `review_ledger`, `audit_cast_contract`, `run_script_doctor`, `apply_deterministic_cast_repairs`, `apply_phantom_skip_fallback`
- `nodes/OTR_LedgerScriptWriter.py` — `_build_truncating_generate_fn`, the per-beat loop, `_build_line_request_for_beat`, news-wiring overlay, title regen
- `nodes/OTR_LedgerScriptReviewer.py` — the reviewer node wrapper
- `nodes/_otr_news_wiring.py` — `override_announcer_close`, `post_assembly_keyterm_check`
- `nodes/production_ledger.py` — L3 ledger schema, `assemble_script_text_from_ledger`, `init_lines_from_outline`, `update_line_text`, `patch_line_fields`
- `nodes/_otr_episode_budget.py` — `compute_episode_budget`, `ACT_COUNT_CONFIG`, `ARC_PHASE_GUIDANCE`
- `nodes/_otr_outline.py` — `generate_outline`, `Beat`, `Outline`
- `docs/news_interpreter_adr.md` — ADR for the news interpreter stage (LLM-agnostic constraint)
- `docs/script-writing-architecture-adr.md` — ADR for the v2.0 writer architecture

---

## 7. Hard constraints (do not violate)

- **Local-only.** No paid APIs, no cloud, no anything-Anthropic, no anything-OpenAI in production runs. Local LLMs only.
- **14.5 GB VRAM ceiling** on the single 5080 GPU.
- **SFW.** No profanity, no violence beyond what fits classic radio drama. Good arc (beginning, middle, end).
- **C7 byte-identity** for the writer's deterministic path (mocked LLM tests). Cleanup phases that mutate ledger text break C7 by design when they fire; the contract is "C7 holds when polish/cleanup is disabled, otherwise contract is schema-validity + cast-lock preservation."
- **LLM-agnostic for the control plane.** No Mistral-/Gemma-/Qwen-specific branches in code. Prompts may target the small-LLM class but must not name a specific model.
- **Lean prompts.** Per-phase prompt body ≤ 600 tokens. The composer hot-path stays under 1000 tokens including all blocks.

---

## 8. What I will do with the round-robin output

I'll synthesize the round-robin into an ADR (Architecture Decision Record) at `docs/multi-turn-polish-adr.md` and then sprint-plan the chosen phases as a sequence of independent commits on `v2.0-alpha`, each with its own regression tests. Same shape as the news_interpreter sprint and the Tier 1/2/3 forward-plan sprints already shipped this week.

End of problem statement.
