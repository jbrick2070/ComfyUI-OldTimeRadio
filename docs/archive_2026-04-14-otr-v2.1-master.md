# OTR v2.1 — Master Problem Statement: HyWorld Integration + Parser Bridge

**Date:** 2026-04-14
**Author:** Jeffrey Brick (with AI pair)
**Status:** Open for round-robin multi-AI review (see Section 12)
**Parent spec:** `docs/2026-04-12-otr-v2-visual-sidecar-design.md`
**Target branch:** `v2.1-hyworld-prep` (off `v2.0-alpha`, not created until v2.0-alpha ships stable)
**Working assumption:** HyWorld 2.0 input/output schema is identical to HY-WorldPlay 1.5. Release watcher (scheduled daily task `hyworld-2-release-watch`) will confirm or invalidate when 2.0 drops.

This document merges the architecture problem statement and the schema problem statement into a single master. It is designed to travel intact from one AI reviewer to the next via round-robin. No detail is lost between handoffs.

---

## 0. READ THIS FIRST IF YOU ARE AN AI REVIEWER

**Your job:**
1. Read the entire document, including every prior round in Section 12.
2. Append a new numbered round in Section 12 using the template at the end of that section.
3. **Never edit, remove, or rewrite prior rounds.** They are accumulated knowledge. Preserve them verbatim.
4. **Never edit Sections 1 through 11.** Those are the static problem statement. If you believe something in them is wrong, flag it in your round; do not patch it.
5. In your round, directly address the questions in Section 10 AND respond to disagreements or open threads from prior rounds.
6. End your round with a `Handoff to next reviewer:` note — what should the next AI focus on, what's still unresolved.
7. Jeffrey (human) then carries this doc to the next reviewer. Round numbers increment monotonically.

**Stopping condition:** Rounds continue until three consecutive reviewers reach substantive consensus on Section 10 questions 1-4 (parser contract, model choice, prompt design, failure modes). After that, Jeffrey closes the round-robin and commits a parser design spec.

---

## 1. Problem in One Paragraph

OTR v2.0 produces a radio drama whose primary output is byte-identical audio plus a companion LTX-2.3 clip track. The Story/Writer node is tuned for audio-first consumption: dialogue, speaker tags, prose scene descriptions. Those outputs are sufficient to feed TTS and seed LTX clip prompts, but they are not structured enough to drive a world-generation pipeline. A HyWorld-based v2.1 needs a story output whose variables can be mechanically parsed into (a) world prompts, (b) character placements, (c) camera paths, and (d) scene-persistence metadata. The central question: how do we evolve OTR's story representation from an audio-first prose blob into a structured intermediate form that feeds both the existing audio pipeline and a new HyWorld visual pipeline, without breaking byte-identical audio output (v2.0 constraint C7)?

---

## 2. Schema A — OTR Story Output (current, v2.0-alpha)

The Story/Writer node emits a single structured string. Parsed informally today.

```yaml
episode:
  title: string
  acts:
    - number: int               # 1, 2, 3
      scenes:
        - number: int
          setting_prose: string # free-form narrative, mixes spatial + mood + lighting
          beats:
            - speaker: string | null      # CHARACTER: tag, null for SFX-only beats
              line: string | null         # spoken dialogue
              sfx: string | null          # parenthetical sound cues
              # NO mood tag today
              # NO duration today
              # NO shot hint today
```

**Guaranteed format rules (enforced by v2.0 guardrails):**
- `CHARACTER:` prefix on every dialogue line
- Character count clamped per line
- 3-act structure when short preset active
- Dialogue line count scales by preset

**Known noise:**
- Scene settings vary in detail (sometimes 2 words, sometimes a paragraph)
- Time-of-day is often implicit in prose, not a separate field
- Character presence is not declared — only inferred from who speaks

**Downstream v2.0 consumers:**

| Variable | v2.0 usage |
|---|---|
| Episode title | MP4 filename, opening card |
| Act structure | Audio pacing, fades |
| Scene prose | LTX image prompt (lossy) |
| Dialogue line | TTS (byte-identical output — C7 locked) |
| Speaker tag | TTS voice select |
| Sound cues | AudioGen prompts |

---

## 3. Schema B — HyWorld 1.5 (assumed identical for 2.0)

**World generation node:**

```yaml
world_gen:
  prompt: string                # spatial + materials + lighting
  reference_image: image | null
  seed: int
  # Outputs: WorldScene handle (opaque to OTR, passed to renderer)
```

**Camera path node (per shot):**

```yaml
camera_path:
  shot_type: enum [wide, mid, over_shoulder, pov, silhouette, tracking]
  duration_sec: float
  motion: enum [static, slow_push, slow_pull, pan_left, pan_right, handheld_tilt]
  mood: enum [tense, melancholy, action, dialogue, transition]
  # Outputs: CameraPath handle
```

**World renderer (per shot):**

```yaml
world_render:
  world_scene: WorldScene
  camera_path: CameraPath
  # Outputs: VideoFrames for that shot
```

**Optional HY-Embodied character placement (deferred to v2.2):**

```yaml
character_placement:
  character_ids: list[string]
  poses: list[enum]
  facing: enum [toward_camera, away, profile, silhouette]
  lip_sync: false  # MUST be false in OTR (C6 rule)
```

---

## 4. The Gap

OTR emits prose. HyWorld consumes structured fields. The fields HyWorld needs are present in OTR's output but not extracted.

| HyWorld field | Present in OTR output? | Extraction difficulty |
|---|---|---|
| `world_gen.prompt` | Yes, inside `setting_prose` | Low — strip narrative, keep spatial |
| `world_gen.seed` | No | Trivial — derive from episode+scene hash |
| `camera_path.shot_type` | No | Medium — requires mood + speaker inference |
| `camera_path.duration_sec` | No | Medium — derive from TTS-rendered audio length |
| `camera_path.motion` | No | Medium — requires mood inference |
| `camera_path.mood` | Partially — implicit in prose and dialogue | High — needs LLM semantic tagging |

Five of six HyWorld fields require either derivation or LLM-based extraction. That is the parser's job.

---

## 5. Proposed Bridge: Parser LLM

A dedicated LLM pass between Story node and HyWorld nodes. Input: Schema A. Output: Schema B-ready structured script.

**Proposed parser contract:**

```yaml
parser_llm:
  input:
    story_blob: string           # raw Story node output
    target_beat_duration: float  # from TTS audio rendering (passed in after audio phase)
  output:
    scenes:
      - persistence_key: string  # deterministic hash, same key = reuse world cache
        world_prompt: string     # spatial only, stripped of narrative
        time_of_day: string
        characters_present: list[string]
        beats:
          - speaker: string | null
            line: string | null
            sfx: string | null
            mood: enum [tense, melancholy, action, dialogue, transition]
            duration_sec: float
            shot_type: enum [wide, mid, over_shoulder, pov, silhouette, tracking]
            motion: enum [static, slow_push, slow_pull, pan_left, pan_right, handheld_tilt]
```

**Parser rules the LLM must enforce:**

1. **No-lip-sync-visible constraint:** When `speaker != null`, `shot_type` must be one of: `over_shoulder`, `silhouette`, `wide`, `tracking` (back-facing). Never `mid` or `pov` toward the speaker's face during dialogue.
2. **Duration realism:** `duration_sec` must match the TTS-rendered audio for that beat. Parser runs AFTER audio phase so real durations are known.
3. **Persistence determinism:** Same `world_prompt` + `time_of_day` across different scenes must produce the same `persistence_key`.
4. **Mood continuity:** Adjacent beats should not swing mood wildly; parser should prefer gentle transitions unless Story node's prose explicitly signals a jolt.
5. **Fallback:** If any field cannot be confidently extracted, emit a safe default (`shot_type=wide`, `motion=static`, `mood=dialogue`) and log a warning.

---

## 6. Two-Phase Extraction Strategy

Rather than refactor the Story node prompt (high risk of regressing byte-identical audio, C7), introduce a **post-parser** sidecar. Story node output stays unchanged for audio; sidecar feeds visual.

**Phase A — Parser-only (no prompt changes):**

```
StoryWriter (unchanged) → StoryBlob ─┬─→ AudioPipeline (unchanged, byte-identical)
                                     │
                                     └─→ StoryParser (NEW)
                                           ↓
                                     StructuredScript
                                           ↓
                                     HyWorld Pipeline (NEW)
```

**Phase B — Prompt-aware emission (only if Phase A parser proves lossy):**

If extraction accuracy falls below ~85% on a test corpus, amend the Story node prompt to emit structured section markers (e.g. `[SETTING: ...]`, `[MOOD: tense]`) in addition to the narrative prose. Audio pipeline strips markers before TTS; visual pipeline reads them. This is a prompt engineering change, not an architecture change — audio byte-identity still preserved after strip.

---

## 7. Target Full Workflow (v2.1)

```
┌─────────────────────────────────────────────────────────────┐
│ StoryWriter (Obsidian/Kimi LLM)                             │
│   Input: preset, target_length, theme seeds                 │
│   Output: StoryBlob (current format, unchanged)             │
└────────────────────────────────┬────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
┌─────────────────────────┐          ┌──────────────────────────┐
│ AudioPipeline (v2.0)    │          │ StoryParser (NEW)        │
│ - TTS (Bark)            │          │ - Extract structure      │
│ - AudioGen SFX          │          │ - Tag moods              │
│ - Mix to 24/48 kHz      │          │ - Compute persistence    │
│ - Master                │          │   keys                   │
│ OUTPUT: master.wav      │          │ OUTPUT: StructuredScript │
│ (BYTE-IDENTICAL to v2.0)│          └──────────┬───────────────┘
└──────────┬──────────────┘                     │
           │                                    ▼
           │                 ┌──────────────────────────────────┐
           │                 │ OTRWorldGen (NEW subprocess)     │
           │                 │ - Reads Scene.world_prompt,      │
           │                 │   persistence_key                │
           │                 │ - Calls HyWorld 2.0 (or 1.5 stub)│
           │                 │ - Caches worlds by persistence   │
           │                 │   key (reuse across acts)        │
           │                 │ OUTPUT: WorldScene handles       │
           │                 └──────────┬───────────────────────┘
           │                            │
           │                            ▼
           │                 ┌──────────────────────────────────┐
           │                 │ ShotSelector (NEW)               │
           │                 │ - Reads Beat.mood, speaker,      │
           │                 │   duration                       │
           │                 │ - Enforces no-lip-sync grammar   │
           │                 │ - Emits CameraPath per beat      │
           │                 │ OUTPUT: CameraPath list          │
           │                 └──────────┬───────────────────────┘
           │                            │
           │                            ▼
           │                 ┌──────────────────────────────────┐
           │                 │ OTRWorldRender (NEW subprocess)  │
           │                 │ - World + CameraPath → frames    │
           │                 │ - Duration matches audio beat    │
           │                 │ - No character mouths on-frame   │
           │                 │ OUTPUT: VideoFrames per beat     │
           │                 └──────────┬───────────────────────┘
           │                            │
           │                            ▼
           │                 ┌──────────────────────────────────┐
           │                 │ LTX Supplement (v2.0, narrowed)  │
           │                 │ - Used ONLY for action bursts    │
           │                 │ - Crossfade into world frames    │
           │                 │ OUTPUT: VideoFrames supplement   │
           │                 └──────────┬───────────────────────┘
           │                            │
           └────────────────┬───────────┘
                            ▼
                 ┌─────────────────────────┐
                 │ AV Mixer (v2.0 + patch) │
                 │ - master.wav + frames   │
                 │ - FFmpeg mux            │
                 │ OUTPUT: episode.mp4     │
                 └─────────────────────────┘
```

---

## 8. Variable Mapping Table

| OTR variable | v2.0 usage | v2.1 extraction path | HyWorld consumer |
|---|---|---|---|
| Episode title | Filename | Unchanged | Opening card overlay |
| Act number | Audio pacing | Preserved in `Act.number` | Scene re-entry check |
| Scene prose | LTX prompt | Parsed into `Scene.world_prompt` + `Scene.time_of_day` | `OTRWorldGen` prompt |
| Dialogue line | TTS | Preserved in `Beat.line` | Audio only (not visual) |
| Speaker tag | TTS voice select | Preserved in `Beat.speaker` | `ShotSelector` (whose OTS shot) |
| Sound cues | AudioGen | Preserved in `Beat.sfx` | Audio only |
| (new) mood | N/A | Derived by `StoryParser` LLM tagger | `ShotSelector`, `CameraPath.motion` |
| (new) beat duration | Implicit | Computed from TTS-rendered audio length | `CameraPath.duration_sec` |
| (new) persistence key | N/A | Hash of world_prompt + time_of_day | `OTRWorldGen` cache lookup |
| (new) characters_present | N/A | Extracted from dialogue + mention graph | Future HY-Embodied placement (v2.2) |

---

## 9. Constraints (Non-negotiable)

| ID | Rule | Inheritance |
|---|---|---|
| C3 | All visual generation in subprocesses via `multiprocessing.get_context("spawn")` | v2.0 |
| C4 | LTX clips capped at 10-12s; crossfades via ffmpeg | v2.0 (narrowed to action bursts only in v2.1) |
| C5 | LTX uses `torch.float8_e4m3fn` (Blackwell-native) | v2.0 |
| C6 | No visible lip sync — characters may appear but never talking head-on | v2.0 extended in v2.1 |
| C7 | Audio byte-identical to v2.0 baseline at every gate | v2.0 |
| V1 (new) | HyWorld subprocess must fit within 14.5 GB real-world VRAM target | v2.1 |
| V2 (new) | World cache keyed by deterministic hash; identical setting across acts = one world | v2.1 |
| V3 (new) | ShotSelector output must never select a shot that shows lips frontally during speech | v2.1 |

**Hardware floor:** RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows 11, torch 2.10, CUDA 13. Sequential execution only. No cloud. No paid APIs.

---

## 10. Questions for Multi-AI Review

Each reviewing AI should weigh in on these in Section 12:

1. **Is the parser contract (Section 5) complete?** What HyWorld field did this miss?
2. **Parser LLM choice:** Should this be the same model as the Story writer (consistency) or a smaller specialized model (cost, latency)? Propose a specific local model fitting the 16 GB VRAM ceiling.
3. **Parser prompt design:** What system prompt would most reliably produce the output schema from the input blob? Sketch it in code fence.
4. **Failure modes:** Where will the parser hallucinate? What regex or rule-based post-validation should run on its output?
5. **Phase A vs Phase B (Section 6):** Is post-parsing better or worse than in-prompt markers? Defend.
6. **Training data:** Should we hand-annotate 5-10 episodes as a test corpus for parser accuracy evaluation? What accuracy threshold is "good enough"?
7. **Precedence:** If parser output drifts from Story intent (e.g. parser says `mood=action` but prose is melancholy), who wins — parser or prose?

---

## 11. Out of Scope

- HyWorld 2.0 actual release details (release watcher `hyworld-2-release-watch` will resolve)
- HY-Embodied character integration (v2.2)
- Audio pipeline changes (none permitted — C7 protected)
- VRAM / Blackwell compatibility benchmarking (handled separately during 1.5 prototype)
- Full v2.0 stabilization (separate track, must complete before v2.1 branch is cut)

---

## 12. Review Rounds (append-only)

Each reviewing AI appends a new round below. Never edit prior rounds. Use the template at the end.

---

### Round 1 — [Awaiting first reviewer]

**Reviewer (model):** _pending_
**Date:** _pending_
**Review mode (fast / reasoning / deep research):** _pending_

**Response to Section 10 questions:**

1. Parser contract completeness: _pending_
2. Parser LLM choice: _pending_
3. Parser prompt design sketch: _pending_
4. Failure modes and post-validation: _pending_
5. Phase A vs Phase B defense: _pending_
6. Test corpus and accuracy threshold: _pending_
7. Parser vs prose precedence: _pending_

**Disagreements with prior rounds:** _N/A — first round_

**Handoff to next reviewer:** _pending_

---

### Template for subsequent rounds (copy, do not fill this one)

```markdown
### Round N — [Short title, e.g. "Sonnet thinking mode — emphasis on failure modes"]

**Reviewer (model):** Claude Sonnet 4.6 / GPT-5 / Gemini 2.5 Pro / etc.
**Date:** YYYY-MM-DD
**Review mode (fast / reasoning / deep research):**

**Response to Section 10 questions:**

1. Parser contract completeness:
2. Parser LLM choice:
3. Parser prompt design sketch:
4. Failure modes and post-validation:
5. Phase A vs Phase B defense:
6. Test corpus and accuracy threshold:
7. Parser vs prose precedence:

**Disagreements with prior rounds:**
- Round X: [specific point of disagreement and reasoning]

**Agreements / reinforcement of prior rounds:**
- Round X: [specific point you agree with and why]

**New concerns not raised in prior rounds:**
- [concern]

**Handoff to next reviewer:**
[One paragraph — what should the next AI focus on, what's still unresolved, what specific decision is blocked on their input]
```

---

## 13. Round Tracker (Jeffrey updates)

| Round | Reviewer model | Mode | Date | Status |
|---|---|---|---|---|
| 1 | _pending_ | _pending_ | _pending_ | Not started |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

*When three consecutive rounds reach consensus on Section 10 Q1-4, mark round-robin closed and commit parser design spec as a sibling doc `2026-MM-DD-otr-v2.1-parser-design.md`.*

---

*End of master problem statement. Pass to next reviewer via Section 12.*
