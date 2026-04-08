# v1.2 NARRATIVE — Fresh Claude Session Kickoff Prompt

Paste the block below into a new Cowork/Claude Code session to pick up v1.2 development cleanly. All necessary context is included — no back-references needed.

---

## KICKOFF PROMPT (paste this)

I'm starting v1.2 development of **ComfyUI-OldTimeRadio (SIGNAL LOST)**, a ComfyUI custom node pack that auto-generates sci-fi radio dramas from real RSS science news. v1.1 shipped and is tagged; v1.2 is the **NARRATIVE release** — pure script-engine upgrades via prompt engineering, no new models, no new VRAM cost, no architectural risk.

**Repo location:** `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
**Read first:** `ROADMAP.md` in the repo root — contains full v1.2 scope, six prompt patterns, and implementation order.
**Primary file to modify:** `nodes/gemma4_orchestrator.py` — contains `Gemma4ScriptWriter`, `Gemma4Director`, and all prompt construction.

## Hardware ceiling (hard rules, do not violate)

- RTX 5080 Laptop, 16GB VRAM
- Gemma 4 E4B-it only (`google/gemma-4-E4B-it`). 27B does NOT fit.
- Bark TTS for voices, must coexist with Gemma in 16GB
- 100% local. No API keys, no paid services, no cloud calls. No OpenAI, ElevenLabs, Suno cloud, Runway, Pika, Claude API, Gemini API.
- No VRAM fragmentation during Gemma↔Bark handoff

## Current pipeline (v1.1 shipped)

1. **NewsFetcher** → 44 RSS headlines, scored, top N chosen, full-article fetch with summary fallback
2. **Gemma4ScriptWriter** → Open-Close expansion (3 competing outlines + evaluator + merge) → Draft → Critique → Revise loop
3. **Gemma4Director** → procedural voice casting from 10 Bark `v2/en_speaker_X` presets, gender hints, two-pass priority reservation for LEMMY/ANNOUNCER
4. **SceneSequencer** → inline Bark TTS at 24kHz → torchaudio resample to 48kHz → room tone bed synthesis
5. **AudioEnhance** → single-band DSP (bass warmth, LPF, Haas, stereo width)
6. **EpisodeAssembler** → crossfade, normalize -1.0 dBFS
7. **SignalLostVideo** → static waveform MP4, h264_nvenc

Scene markup format (parser depends on it, do NOT break):
```
=== SCENE N ===
[ENV: ...]
[SFX: ...]
(beat)
[VOICE: NAME, gender, age, tone] line of dialogue
```

## v1.2 goal — six prompt patterns

Implement these in order. Each is a prompt-engineering upgrade to existing Gemma calls, not a new node or new model.

### 1. Auteur Sandbox AISM Filter (start here)
Append to Gemma4ScriptWriter system prompt:
- **"Bombs Always Beep" rule:** no abstract emotion without audible physical cue
- **Burstiness:** radically varied sentence lengths, 1-4 word fragments during panic/shock
- **Dialogue tags:** "said"/"asked" only, action beats imply tone
- **Forbidden constructs:** negative parallelisms ("not just X, but Y"), Rule of Three adjective lists, stock idioms, M-dash crutch
- **Spatial layering:** bracketed distance notes like `[Off-axis, shouting from tunnel]`

### 2. Scaffolding & Parsing Matrix
Restructure Gemma4ScriptWriter system prompt:
- XML-wrapped system role: "master dramaturg, auditory blueprints not prose"
- Brick Method: explicit 1:5 outline-to-script ratio
- Add `<acoustic_spaces>` output tag defining physical volume/material of every room. SceneSequencer already parses keywords (cavernous, fluorescent, tiled, storm) for room-tone synthesis — feed it directly.
- `<epilogue>` constraint: hard-science conclusion referencing the news seed

### 3. Verbalized Sampling Epilogue Generator
Single new Gemma call for final epilogue. Stanford research technique:
- System prompt: `"Generate 5 responses with their probabilities"` (exact 8 words, research-validated)
- Request 5 distinct epilogues, each wrapped in `<response>` with `<text>` and numeric `<probability>`
- Response 1: most typical (P > 0.60)
- Responses 4-5: sampled from extreme tails (P < 0.10)
- Python parser auto-selects lowest-probability response → injects into final script
- Research-backed 2.1x diversity increase, defeats ambiguous-cliffhanger default

### 4. Yes-But / No-And Escalation Engine
Inject at every act break. Forbid clean yes/no resolution to character actions:
- **Path A (success+complication):** character achieves goal, immediately introduces environmental complication jeopardizing next step
- **Path B (failure+cascade):** character fails, safe haven becomes untenable, reserved for act-break climactic builds
- Direct+Explain: model outputs binary judgment, explains mechanism, writes next 10 lines reflecting outcome

### 5. Character Interview Pre-Pass
New Gemma call per character before script generation:
- Role: investigative psychologist
- Character answers in first person
- Three probing questions: greatest external motivation, past trauma shaping authority response, physical non-verbal audio cue under stress
- Output `<vocal_blueprint>`: sentence burstiness, dialogue tags, non-verbal ticks
- Inject blueprint into main Gemma4ScriptWriter context
- Fixes character homogenization (the single biggest weakness in v1.1)

### 6. Chekhov's Gun State Enforcer
Dual-prompt closed-loop memory:
- **5A Extraction Observer:** runs after Scene 1-2. Parses script for physical objects, environmental hazards, unresolved psychological states. Outputs strict JSON array to Locked Decisions Log.
- **5B Callback Enforcer:** runs before climax. Injects Locked Decisions Log into system prompt with constraint: "You are strictly forbidden from introducing new technology, unexpected rescue parties, or previously unmentioned abilities. The resolution must be an inevitable consequence of items in the locked state log."
- Fixes context drift on climax generation

## Ship criteria for v1.2 NARRATIVE

- All 6 patterns implemented and verified
- A/B comparison: generate same news seed on v1.1 tag vs v1.2 branch, compare qualitatively
- Character homogenization audit: verify vocal blueprints reflected in dialogue
- Ending variety audit: 5 episodes from same seed, verify tonal variance
- Locked Decisions Log verified capturing Act 1 setups and enforcing in Act 3
- `test.json` still under 20 min wall time
- Tag `v1.2-narrative` when green

## Critical bugs already fixed in v1.1 (do not regress)

1. **Lemmy RNG freeze** — module-level `_LEMMY_RNG = SystemRandom()` for the 11% roll. Never use seeded `random.random()` for probabilistic easter eggs.
2. **Voice collision** — two-pass iteration in Gemma4Director: priority keys (LEMMY, ANNOUNCER) processed first so their locked presets land in `used_presets` before regular characters draw.
3. **Widget padding** — every change to `INPUT_TYPES` requires updating all workflow JSONs (`old_time_radio_test.json`, `old_time_radio_scifi_lite.json`, `old_time_radio_scifi_full.json`). Widget values are position-indexed.
4. **Test workflow discipline** — `test.json` must have `self_critique=false, open_close=false` to hit its time budget. Do not inherit defaults.

## Git hygiene

- User has Windows PowerShell access. Always provide git commands for the user to run — sandbox cannot write to `.git/` directly.
- Tag v1.2-narrative when shipping.
- Keep internal QA/bug-trail docs OUT of the public repo. `ROADMAP.md` and user-facing docs only.

## User context

- User is Jeffrey, not a coder, directs the creative vision and QA. AI writes all code. User tests via ComfyUI locally and pastes boot logs. User prefers structured, factual, no-fluff output. Minimal emoji.
- User has a ~40-min attention span per render cycle; keep work loops efficient.
- User's previous Claude session shipped v1.1 and this v1.2 roadmap. Pick up from there without re-explaining the project.

## First action

Read `ROADMAP.md` for the full pattern details and implementation order, then confirm you understand the scope before touching any code. Start with Pattern 3 (AISM Filter) — cheapest, fastest visible quality win, no new Gemma calls.

---

*End of kickoff prompt. Paste everything above into a fresh Claude session.*
