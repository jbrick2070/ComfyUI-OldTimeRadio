# OTR v2.1 — Schema Problem Statement for Multi-AI Review

**Date:** 2026-04-14
**Status:** Open for cross-model review (round-robin — see Section 9)
**Working assumption:** HyWorld 2.0 input/output schema is identical to HY-WorldPlay 1.5.

This is a **schema-only problem statement** with embedded round-robin review logic. Jeffrey passes this doc from one AI to the next; each AI appends to Section 9 without modifying prior rounds. No detail is lost between handoffs.

---

## READ THIS FIRST IF YOU ARE AN AI REVIEWER

**Your job:**
1. Read the entire document, including every prior round in Section 9.
2. Append a new numbered round in Section 9 using the template at the bottom of that section.
3. **Never edit, remove, or rewrite prior rounds.** They are accumulated knowledge. Preserve them verbatim.
4. **Never edit Sections 1-6.** Those are the static problem statement. If you believe one is wrong, flag it in your round; do not patch it.
5. In your round, directly address the questions in Section 5 AND respond to disagreements or open threads from prior rounds.
6. End your round with a `Handoff to next reviewer:` note — what should the next AI focus on, what's still unresolved.
7. Jeffrey (human) then carries this doc to the next reviewer. Round numbers increment monotonically.

**Stopping condition:** Rounds continue until three consecutive reviewers reach substantive consensus on Section 5 questions 1-4 (parser contract, model choice, prompt design, failure modes). After that, Jeffrey closes the round-robin and commits a parser design spec.

---

## 1. Schema A — OTR Story Output (current, v2.0-alpha)

The Story/Writer node emits a single structured string. Parsed informally today. Canonical fields below.

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

---

## 2. Schema B — HyWorld 1.5 (assumed identical for 2.0)

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

## 3. The Gap

OTR emits prose. HyWorld consumes structured fields. The fields HyWorld needs **are present in OTR's output but not extracted**.

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

## 4. Proposed Bridge: Parser LLM

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

## 5. Questions for Multi-AI Review

Each reviewing AI should weigh in on:

1. **Is the parser contract complete?** What HyWorld field did this miss?
2. **Parser LLM choice:** Should this be the same model as the Story writer (consistency) or a smaller specialized model (cost, latency)? Propose a specific model.
3. **Parser prompt design:** What system prompt would most reliably produce the output schema from the input blob? Sketch it.
4. **Failure modes:** Where will the parser hallucinate? What regex or rule-based post-validation should run on its output?
5. **Is Phase B (Story prompt markers) a better path than Phase A (post-parse)?** Defend.
6. **Training data:** Should we hand-annotate 5-10 episodes as a test corpus for parser accuracy evaluation? What accuracy threshold is "good enough"?
7. **Handling deletions:** If parser output drifts from Story intent (e.g. parser says `mood=action` but prose is melancholy), who wins — parser or prose?

---

## 6. Out of Scope for This Doc

- HyWorld 2.0 actual release details (release watcher will resolve)
- Full pipeline graph (see `2026-04-14-otr-v2.1-hyworld-problem-statement.md`)
- HY-Embodied integration (v2.2)
- Audio pipeline changes (none — C7 protected)
- VRAM / hardware compatibility (orthogonal; see parent spec)

---

*Send this doc to reviewing models verbatim. Collect their proposals into a comparison table before committing to a parser design.*

---

## 7. Review Rounds (append-only)

Each reviewing AI adds a new round below. Never edit prior rounds. Use the template at the end.

---

### Round 1 — [Awaiting first reviewer]

**Reviewer (model):** _pending_
**Date:** _pending_
**Review mode (fast / reasoning / deep research):** _pending_

**Response to Section 5 questions:**

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
### Round N — [Short title — e.g. "Sonnet thinking mode — emphasis on failure modes"]

**Reviewer (model):** Claude Sonnet 4.6 / GPT-5 / Gemini 2.5 Pro / etc.
**Date:** YYYY-MM-DD
**Review mode (fast / reasoning / deep research):**

**Response to Section 5 questions:**

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

## 8. Round Tracker (Jeffrey updates)

| Round | Reviewer model | Mode | Date | Status |
|---|---|---|---|---|
| 1 | _pending_ | _pending_ | _pending_ | Not started |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

*When three consecutive rounds reach consensus on Section 5 Q1-4, mark round-robin closed and commit parser design spec.*
