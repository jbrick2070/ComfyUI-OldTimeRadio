# ComfyUI-OldTimeRadio — Roadmap

**Current shipped:** v1.1 (tagged `v1.1`)
**Hardware ceiling:** RTX 5080 Laptop, 16GB VRAM. Gemma 4 E4B only — 27B does NOT fit. No API keys, no paid services, 100% local open-source.

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

---

## v1.3 — Production Polish (was v1.2, demoted)

Everything from the previous v1.2 scope. Now the second release, because better scripts matter more than better EQ.

| # | Feature | Bucket | Effort | VRAM |
|---|---------|--------|--------|------|
| 1 | Kokoro Announcer (separate track) | GPU | M | ~0.5GB |
| 2 | pedalboard: tape sat + multiband comp on Announcer bus | CPU | S | 0 |
| 2b | pedalboard: harmonic exciter on Scene bus (fix Bark 24kHz muffle) | CPU | S | 0 |
| 3 | ffmpeg: scanlines + grain + vignette + avectorscope | CPU | S | 0 |
| 4 | pyloudnorm master normalize to -16 LUFS | CPU | S | 0 |
| 5 | Procedural intro/outro sting (scipy chirp) | CPU | S | 0 |
| 6 | GPU/CPU parallel pipeline overlap | arch | M | 0 |
| 7 | MP4 chapter markers (ffmetadata mux) | CPU | S | 0 |

Attack order unchanged: master bus DSP → video overhaul → Kokoro → chapters → sting → parallel overlap.

---

## v1.4 — Stretch / Experimental

- Stable Audio Open for SFX cues (VRAM sequencing)
- RVC voice-locking (Bark drift fix)
- XTTS-v2 / F5-TTS / StyleTTS 2 / MeloTTS evaluation
- Convolution reverb on Announcer (small radio-hall IR)
- Whisper-tiny subtitle track
- Sentiment-driven LUT per scene
- Vintage radio dial SVG overlay
- Reactive particle fields (deprioritized)

---

## Hard Rules (All Versions)

- No paid APIs, no API keys, no cloud services
- Must coexist with Gemma4 E4B + Bark in 16GB VRAM
- No VRAM fragmentation during Gemma↔Bark handoff
- Gemma 4 27B does not fit — do not plan around it
- Story quality is NOT solved — it is the primary v1.2 target

---

*Roadmap updated 2026-04-07 after narrative framework bombshell. Story promoted to v1.2, polish demoted to v1.3.*
