# ComfyUI-OldTimeRadio — Roadmap

**Current shipped:** v1.1 (tagged `v1.1`)
**In flight:** v1.2 narrative beta on `v1.2-narrative-beta` at commit `e2f210a` — all 6 patterns merged as prompt-level MVP, awaiting boot test
**Hardware ceiling:** RTX 5080 Laptop, 16GB VRAM. Gemma 4 E4B only — 27B does NOT fit. No API keys, no paid services, 100% local open-source.

---

## v1.3 — Narrative Hardening + Production Polish (HOT)

**Theme:** Convert v1.2's prompt-level MVPs into real architecture, harden parser, ship production polish. This is where the "honest trade-offs" from v1.2 get paid back.

### v1.3 Narrative Hardening Backlog (from v1.2 beta peer review)

| # | Item | Type | Source |
|---|------|------|--------|
| N1 | **Pattern 3 epilogue selection logic** — Python post-processor that parses `<epilogue_candidates>`, extracts the five `<response>` probabilities, and programmatically swaps in the lowest-probability candidate as the final `=== EPILOGUE ===`. Closes the gap between prompt weight and actual selection. | arch | reviewer #3 |
| N2 | **Parser XML block skip** — `_parse_script` should explicitly skip lines between `<vocal_blueprints>…</vocal_blueprints>`, `<locked_decisions>…</locked_decisions>`, `<epilogue_candidates>…</epilogue_candidates>` rather than catching them as `direction` entries. Clean JSON output. | parser | reviewer #3 |
| N3 | **Act-break decision markers** — prompt Gemma to emit `# ACT_BREAK_DECISION: YES_BUT` / `NO_AND` at every act break, then capture in `_parse_script` for machine-checkable Pattern 4 verification. Strip before feeding SceneSequencer. | prompt + parser | reviewer #3 |
| N4 | **Chunked generation structural validation** — after outline generation inside `_generate_chunked`, verify presence of structural keywords (`ACT 1`, `CHARACTERS`, `SFX`). On miss, force regenerate at lower temperature to prevent downstream act drift. | gen | reviewer #3 |
| N5 | **Parser smoke tests** — unit test harness feeding `_parse_script` a synthetic script with all three metadata blocks. Assert zero `dialogue`/`environment`/`sfx` entries leak from metadata regions and direction count stays within bound. | test | reviewer #3 |
| N6 | **`test_full_patterns.json` workflow** — new workflow: `target_minutes=5`, `include_act_breaks=true`, `self_critique=true`. Exercises all six patterns in a CI-friendly runtime. | workflow | reviewer #3 |
| N7 | **Chunked context slicing** — replace hard `acts[-1][:3000]` / `[-500:]` slicing in `_generate_chunked` with token-aware or sentence-boundary-aware truncation to avoid mid-word summarization context cuts. | gen | reviewer #3 |
| N8 | **Voice regex hyphen support** — widen `_randomize_character_names` regex character class from `[A-Z0-9_ ]` to include `-` so names like `DR-7` or `CHEN-LU` don't silently fail gender mapping. Names stay ALL CAPS by spec. | parser | reviewer #3 |
| N9 | **Defensive `_unload_gemma4()` at Director entry** — call at the start of `Gemma4Director.direct()` as belt-and-suspenders for standalone runs not chained after `write_script`. Already handled in the normal pipeline at line 2229 but worth hardening. | VRAM | reviewer #3 |
| N10 | **Multi-call architecture for Patterns 3/5/6 (original ROADMAP design)** — promote the prompt-level MVPs to real multi-call architectures when A/B shows prompt weight is losing signal. Pattern 5 = +1 Gemma call per character. Pattern 6 = +2 Gemma calls (extract + enforce). Pattern 3 = +1 Gemma call dedicated epilogue tail-sampler. | arch | original ROADMAP |
| N11 | **Strict schema validation gateway** — post-generation validator with 2-retry fallback. Checks: act demarcation count, unique character ID count, SFX+ENV tag density, malformed XML instance count. Abort on second failure. | parser | Generation Improvements §1.1 |
| N12 | **Context window % telemetry** — track `tokens_used / context_window` via native tokenizer, warn at 85%, surface in output treatment log as `Context Efficiency: N%`. Predicts chunk collapse before it happens. | gen | Generation Improvements §1.2 |
| N13 | **Epilogue hard-resample fallback** — if `<epilogue_candidates>` probability parse fails, abandon prompt constraint and resample at `temp=1.2` to force novel token generation and break latent-space repetition. Belt-and-suspenders for N1. | gen | Generation Improvements §1.3 |
| N14 | **TTS audio cache** — hash `(character_id + line_text + voice_preset)` as cache key, probe cache dir before invoking Bark. Cache hit = bypass inference entirely. Transforms script-iteration cycles from minutes to milliseconds. Highest iteration-velocity win in the whole backlog. | perf | Generation Improvements §1.5 |
| N15 | **Dialogue variance enforcer** — post-parse pass: iterate each act, calculate word count of consecutive dialogue blocks. If >80% fall in a narrow 10–20 word range, flag and regenerate that act with explicit burstiness examples prepended to context (4-word panic / 1-word interrupt / 18-word exposition). Fixes Gemma's monotonous predictive cadence. | narrative | Generation Improvements §2.1 |
| N16 | **Character behavioral contracts** — isolate one-line behavioral constraint per character at Pattern 5 blueprint time ("never speaks emotionally", "always interrupts", "highly technical vocabulary"), re-inject into static prompt memory for every subsequent act generation. Hardens blueprints against drift across long episodes. | narrative | Generation Improvements §2.2 |
| N17 | **Dialogue compression pre-TTS filter** — programmatic scrub of LLM filler phrases, hard cap on sentence length, max two sentences per contiguous dialogue node. Forces rapid conversational exchanges instead of theatrical monologues. | narrative | Generation Improvements §2.3 |
| N18 | **TTS temperature scheduling** — per-line temperature injection into Bark invocation: first 3 lines per character at `temp=0.45` (lock prosody to preset), lines with emotional markers `[gasps]`/`[shouts]`/`[crying]` at `temp=0.65` (let Bark explore), default `temp=0.55`. Real fix for voice drift across long episodes. | TTS | Generation Improvements §3.1 |
| N19 | **Micro-prosody text sanitizer** — pre-TTS regex pass: commas → explicit pause markers, ellipses multiplied for longer hesitation, high-impact words (flagged by N15 enforcer) → UPPERCASE for vocal emphasis, emotional tags concatenated to line extremities for non-verbal token generation. ~20 lines of regex, big perceived-quality jump. | TTS | Generation Improvements §3.2 |
| N20 | **Voice preset heatmap + allocation cap** — heatmap counter across cast, max 2 characters per Bark preset, homogenization warning if any single preset exceeds 40% of episode dialogue. Force least-prominent reassignment to maximally contrasting profile on warning. Hardens existing `_randomize_character_names` logic. | TTS | Generation Improvements §3.4 |
| N21 | **Episode hash ID** — SHA-truncate news seed string to 4-digit hex, inject into video title bar as unique episode ID (`"SIGNAL LOST E-38A7"`). Pure aesthetic, ~10 lines, matches show bible. | video | Generation Improvements §5.3 |
| N22 | **Golden Episode Test** — verified structurally-perfect run committed to repo. Every major commit runs comparison generation: duration, character continuity, internal data structures. Catches logic regressions before merge. | CI | Generation Improvements §6.2 |
| N23 | **CI ship-blocker policy** — full integration workflow with all patterns active + `self_critique=True` must render end-to-end within temporal threshold. Failure revokes release candidate status. Pure policy, no code. | CI | Generation Improvements §6.3 |

### v1.3 Production Polish (was the old v1.2)

---

## v1.2 NARRATIVE — Story-First Release

**Bombshell update:** earlier roadmap assumed story was solved. It isn't — it's just good. A research-backed framework of six prompt patterns dropped in from external AI audit can take the script engine from "good" to "broadcast-grade" with **zero new VRAM, zero new dependencies, zero architectural risk**. All six are prompt-engineering upgrades to the existing Gemma4ScriptWriter and Gemma4Director nodes.

Production polish (pedalboard, avectorscope, Kokoro, chapters, LUFS) is important but compounding — every future listener of every future episode benefits more from better scripts than from better EQ. Narrative goes first.

### The Six Patterns

| # | Pattern | What It Fixes | Cost |
|---|---------|---------------|------|
| 1 | **Scaffolding & Parsing Matrix** | Generic structure, flat outlines, lost acoustic environment cues | Prompt rewrite |
| 2 | **Character Interview Pre-Pass** | Character homogenization drift, interchangeable voices, archetype dialogue | +1 Gemma call per character |
| 3 | **Auteur Sandbox AISM Filter** | Abstract emotional summary, "Rule of Three" tics, M-dash crutch, uniform burstiness | Prompt rewrite |
| 4 | **Yes-But / No-And Escalation Engine** | Flat arcs, clean premature conflict resolution, weak Act 2 middles | Prompt rewrite |
| 5 | **Chekhov's Gun State Enforcer** | Context drift, forgotten Act 1 setups, hallucinated climax solutions | +2 Gemma calls (extract + enforce) |
| 6 | **Verbalized Sampling Epilogue Generator** | Ambiguous cliffhanger default, mode collapse, predictable endings | +1 Gemma call (epilogue tail-sample) |

**Total added Gemma calls: ~4-6 per episode** (N characters × 1 for interview, +3 fixed for extract/enforce/epilogue). Estimated wall-time bump: +8-15 min per full episode. All on existing Gemma 4 E4B model, no new model loads.

### Pattern 1: Scaffolding & Parsing Matrix

XML-wrapped system prompt for Gemma4ScriptWriter. Key additions to the existing prompt:

- `<role>` override establishing "master dramaturg, auditory blueprints not prose"
- Brick Method: explicit 1:5 outline-to-script ratio. Generate 1000-word outline, expand to 5000 words
- `<acoustic_spaces>` output tag defining physical volume and material of every room — feeds directly to SceneSequencer room-tone bed synthesizer (cavernous, fluorescent, tiled, storm keywords already parsed downstream)
- `<epilogue>` constraint mandating hard-science grounded conclusion referencing the news seed
- News-as-Spine already implemented, but restructure prompt to reference it as the inciting incident explicitly

### Pattern 2: Character Interview Pre-Pass

New node stage or new Gemma call per character before script generation. For each character in the cast:

- Role: investigative psychologist conducting pre-production interview
- Character answers in first person, demonstrating vocal cadence, regional slang, psychological flaws
- Three probing questions: greatest external motivation, past trauma shaping authority response, physical non-verbal audio cue under stress
- Output: `<vocal_blueprint>` with sentence burstiness, preferred dialogue tags, non-verbal ticks
- Blueprint injected into Gemma4ScriptWriter context during main script generation

This is the single highest-impact narrative upgrade. Character homogenization is the most visible weakness in the current pipeline and this fixes it at the source.

### Pattern 3: Auteur Sandbox AISM Filter

Append to Gemma4ScriptWriter system prompt. Five filter categories:

- **Sensory translation ("Bombs Always Beep"):** No abstract emotion without an audible physical manifestation. Don't write "Rex panicked" — write hissing depressurization + Rex's ragged breathing against the mic
- **Spatial layering:** Bracketed directorial distance notes (`[Off-axis, shouting from tunnel]`) feeding into Gemma4Director spatialization
- **Burstiness:** Radically varied sentence lengths, 1-4 word fragments during panic/shock/failure
- **Dialogue tags:** Strict "said"/"asked" only, action beats imply tone
- **Forbidden constructs:** Negative parallelisms ("not just X, but Y"), Rule of Three adjective lists, stock idioms ("blood ran cold")

### Pattern 4: Yes-But / No-And Escalation Engine

Inject at every major plot juncture / act break. Forbids clean yes/no resolution, forces routing through:

- **Path A (success with complication):** Character achieves goal → immediately introduces physical/environmental complication jeopardizing next step
- **Path B (failure with escalation):** Character fails → cascading failure makes safe haven untenable, escalates danger. Reserved for act breaks and climactic builds

Direct+Explain execution: model outputs binary judgment, explains mechanism, then writes next 10 lines of sensory-first dialogue reflecting outcome.

### Pattern 5: Chekhov's Gun State Enforcer

Dual-prompt architecture with closed-loop memory:

**5A: Extraction Observer** runs after early scenes. Parses script for physical objects, environmental hazards, unresolved psychological states. Outputs strict JSON array to a Locked Decisions Log.

**5B: Callback Enforcer** runs before climax generation. Injects Locked Decisions Log into system prompt with explicit constraint: *"You are strictly forbidden from introducing new technology, unexpected rescue parties, or previously unmentioned abilities. The resolution must be an inevitable consequence of items in the locked state log."*

Fixes the context drift that currently lets Gemma hallucinate Act 3 solutions unrelated to Act 1 setups.

### Pattern 6: Verbalized Sampling Epilogue Generator

Stanford technique. The exact 8-word instruction *"Generate 5 responses with their probabilities"* forces the model to calculate and explicitly verbalize a probability distribution over multiple semantic pathways, bypassing RLHF mode collapse.

- Request 5 distinct epilogues with probabilities
- Response 1: most typical (probability > 0.60) — what an aligned model would default to
- Responses 4-5: sampled from extreme tails (probability < 0.10) — dark, unconventional, tragic, genre-bending
- Automated Python parser in epilogue node reads XML probabilities, discards high-probability generic outputs, injects lowest-probability text into final script

Research-backed diversity increase: up to 2.1x. This alone breaks the "ambiguous cliffhanger every time" problem that currently defines every v1.1 episode ending.

### Implementation Order

1. **Pattern 3 (AISM Filter)** first — pure prompt edit, immediate visible quality jump, no new Gemma calls, sets up the sensory vocabulary the other patterns depend on
2. **Pattern 1 (Scaffolding Matrix)** — restructures existing prompt, adds `<acoustic_spaces>` output, enables downstream room-tone improvements
3. **Pattern 6 (Verbalized Sampling)** — single new Gemma call, highest single-beat impact on perceived quality (endings are what listeners remember)
4. **Pattern 4 (Escalation Engine)** — inject at act breaks, refactor existing chunked generation to route through Path A/B logic
5. **Pattern 2 (Character Interview)** — new pre-pass stage, N extra Gemma calls, biggest wall-time cost but biggest character-depth win
6. **Pattern 5 (Chekhov State Enforcer)** — most architecturally complex (dual-prompt with persistent state), save for last

### Ship Criteria for v1.2 Narrative

- All 6 patterns implemented and verified
- Side-by-side A/B: generate same news seed on v1.1 and v1.2, compare episodes qualitatively
- Character homogenization audit: verify distinct vocal blueprints are actually reflected in dialogue
- Ending variety audit: generate 5 episodes from same seed, verify tonal variance
- Locked Decisions Log verified capturing Act 1 setups and enforcing them in Act 3
- test.json still <20 min (some patterns add cost but test uses minimal cast)
- Tagged `v1.2-narrative` when green

| # | Feature | Bucket | Effort | VRAM |
|---|---------|--------|--------|------|
| P1 | Kokoro Announcer (separate track) | GPU | M | ~0.5GB |
| P2 | pedalboard: tape sat + multiband comp on Announcer bus | CPU | S | 0 |
| P2b | pedalboard: harmonic exciter on Scene bus (fix Bark 24kHz muffle) | CPU | S | 0 |
| P3 | ffmpeg: scanlines + grain + vignette + avectorscope | CPU | S | 0 |
| P4 | pyloudnorm master normalize to -16 LUFS | CPU | S | 0 |
| P5 | Procedural intro/outro sting (scipy chirp) | CPU | S | 0 |
| P6 | GPU/CPU parallel pipeline overlap | arch | M | 0 |
| P7 | MP4 chapter markers (ffmetadata mux) | CPU | S | 0 |

### v1.3 Ship Criteria

- All N1–N10 narrative hardening items landed OR explicitly deferred to v1.4 with justification
- Parser smoke tests passing in CI
- `test_full_patterns.json` runs end-to-end in under 20 minutes
- A/B demonstrates Pattern 3 Python selection > pure prompt weight
- Production polish P1–P7 complete or explicitly deferred
- Tagged `v1.3` when green

---

## v1.4 — Stretch / Experimental / Punted from v1.3

### Original v1.4 stretch list
- Stable Audio Open for SFX cues (VRAM sequencing)
- RVC voice-locking (Bark drift fix)
- XTTS-v2 / F5-TTS / StyleTTS 2 / MeloTTS evaluation
- Convolution reverb on Announcer (small radio-hall IR)
- Whisper-tiny subtitle track
- Sentiment-driven LUT per scene
- Vintage radio dial SVG overlay
- Reactive particle fields (deprioritized)

### Punted from Generation Improvements spec (too fluffy / heavy / architectural for v1.3)
- **Scene-level generative caching** — serialize intermediate JSON state after each act, resume from cache on OOM. Complex, niche use case. (§1.5)
- **Deterministic pipeline fingerprint JSON** — already have `episode_fingerprint`; this is gold-plating. (§1.6)
- **Conflict pressure meter via embedding distances + tension gradient regeneration** — requires embedding infrastructure; tension scoring is hand-wavy. (§2.3, §2.4)
- **Epilogue quality heuristics via vector embeddings** — novelty + polarity + callback scoring across candidates; same embedding infra dependency. (§2.4)
- **Dynamic silence shaping** — inter-clip silence scaled inversely to tension score. Nice but not narrative. (§3.3)
- **Torchaudio GPU waveform ops refactor** — profiling first to confirm this is actually a bottleneck before refactoring. (§4.1)
- **Procedural metallic strike synthesis** (noise + 4th-order Butterworth + exp envelope + convolution) — cool math, not a v1.3 blocker. (§4.2)
- **Exponential sine sweep chapter bumps** — polish, not narrative. (§4.3)
- **Stateful video rendering transitions** (NORMAL/ALERT/SIGNAL_LOSS/RECOVERY state machine) — video polish. (§5.1)
- **PNG sequence export via FFmpeg `image2pipe`** — dev tool, not shipping feature. (§5.2)
- **Streaming XML validation** — catching malformed XML mid-token-stream. Too complex for marginal win; N11 post-gen validator is enough. (§1.2 partial)
- **Async GPU/CPU threading overlap** — architectural refactor, high risk, needs dedicated release. (§6.1)
- **Telemetry breadcrumb JSON trail** — observability overkill for a single-user creative pipeline. (§6.2 partial)

---

## Hard Rules (All Versions)

- No paid APIs, no API keys, no cloud services
- Must coexist with Gemma4 E4B + Bark in 16GB VRAM
- No VRAM fragmentation during Gemma↔Bark handoff
- Gemma 4 27B does not fit — do not plan around it
- Story quality is NOT solved — it is the primary v1.2 target

---

*Roadmap updated 2026-04-07 — v1.2 narrative beta merged (commit `e2f210a`); v1.3 hot with N1–N23 (10 hardening items from v1.2 peer review + 13 items curated from the Generation Improvements spec) plus production polish P1–P7. Fluff and architectural overhauls punted to v1.4.*
