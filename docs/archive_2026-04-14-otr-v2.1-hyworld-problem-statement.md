# OTR v2.1 — HyWorld Integration: Problem Statement

**Date:** 2026-04-14
**Author:** Jeffrey Brick (with AI pair)
**Status:** Draft — pre-scaffolding
**Parent spec:** `docs/2026-04-12-otr-v2-visual-sidecar-design.md`
**Target branch:** `v2.1-hyworld-prep` (not yet created; off `v2.0-alpha`)

---

## 1. Problem

OTR v2.0 produces a radio drama whose primary output is byte-identical audio plus a companion LTX-2.3 clip track for b-roll. The current Story node is tuned for audio-first consumption: it emits dialogue, speaker tags, and prose scene descriptions. Those outputs are sufficient to feed TTS and to seed LTX clip prompts, but they are **not structured enough to drive a world-generation pipeline**.

A HyWorld-based v2.1 needs a story output whose variables can be mechanically parsed into (a) world prompts, (b) character placements, (c) camera paths, and (d) scene-persistence metadata. The current Story node output is too unstructured for that without a parsing layer, and the parsing layer is lossy without explicit upstream fields.

**The problem in one sentence:** How do we evolve OTR's current story representation from an audio-first prose blob into a structured intermediate form that feeds both the existing audio pipeline and a new HyWorld visual pipeline, without breaking byte-identical audio output (v2.0 constraint C7)?

---

## 2. Current Story Variables (v2.0-alpha as of 2026-04-14)

The Story/Writer node currently emits a single structured string containing:

| Variable | Type | Example | Consumer |
|---|---|---|---|
| Episode title | string | `"The Last Frequency"` | MP4 filename, opening card |
| Act structure | implicit headers | `ACT I`, `ACT II`, `ACT III` | Pacing, audio fades |
| Scene headers | implicit | `SCENE 1: RADIO ROOM, NIGHT` | Weak — not machine-parsed today |
| Scene prose | free text | `"Rain beats the tin roof. A single bulb swings over the console."` | Fed into LTX as image prompt (lossy) |
| Dialogue lines | `CHARACTER: line` format | `DR. KELLER: The signal's back.` | TTS (this is the critical one) |
| Sound cues | parenthetical | `(static surge, then silence)` | AudioGen prompts |

**Guardrails that already run:**
- Character count clamp per dialogue line
- Obsidian cap on prompt size
- Dialogue line count scaling by preset (BUG-009 fix)
- `CHARACTER:` format enforced in short (3-act) prompt (BUG-007 fix)

**What the Story node does NOT emit as first-class fields:**
- Scene *identity* (is Act III Scene 2 the same location as Act I Scene 2?)
- Scene *setting* as separate spatial description vs narrative prose
- Character list per scene (who is physically present vs mentioned)
- Beat-level mood (tense / melancholy / action / dialogue / transition)
- Shot suggestions (who's speaking, who's listening, what's the camera doing)
- Scene duration target (explicit seconds, not implicit from dialogue length)

These missing fields are what HyWorld needs.

---

## 3. HyWorld Input Requirements (assumed from HY-WorldPlay 1.5)

HyWorld 2.0's input spec is not public at time of writing. This section assumes input shape will be similar to HY-WorldPlay 1.5, which is the currently available reference. If 2.0 differs, the mapping stays the same shape; only field names shift.

**World generation requires:**

| Input | Type | Purpose |
|---|---|---|
| World prompt | string | Spatial + lighting + materials description |
| Reference image (optional) | image | Visual anchor for style consistency |
| Seed | int | Reproducibility, scene caching |
| Persistence key | string | Hash for world cache reuse across acts |

**Camera path requires:**

| Input | Type | Purpose |
|---|---|---|
| Shot type | enum | wide / mid / OTS / POV / silhouette / tracking |
| Duration | float seconds | Matched to audio beat length |
| Motion profile | enum | static / slow push / slow pull / pan / handheld tilt |
| Mood register | enum | tense / melancholy / action / dialogue / transition |

**Character placement (if HY-Embodied integrated) requires:**

| Input | Type | Purpose |
|---|---|---|
| Character list | list of IDs | Who is physically present |
| Pose/gesture | enum | standing / walking / handling object / listening |
| Facing | enum | toward camera / away / profile / silhouette |
| Lip sync policy | bool | MUST be false in OTR (C6 rule, Silent Lip Bug) |

---

## 4. Gap Analysis

| Need | Current state | Gap |
|---|---|---|
| Scene setting as standalone field | Mixed into narrative prose | Need extractor or new Story output field |
| Scene identity (persistence) | None | Need hash of setting + lighting + time-of-day |
| Character-presence list per scene | Implicit from dialogue | Need explicit field (not all mentioned = all present) |
| Beat mood | Not tagged | Need LLM tagger or prompt engineering |
| Shot suggestions | None | Need `ShotSelector` node reading mood + speaker |
| Duration target per beat | Inferred from dialogue length | Need explicit duration field for camera path timing |
| Persistence key for world cache | None | Need deterministic hash function |

---

## 5. Two-Phase Extraction Strategy

Rather than refactor the Story node prompt (high risk of regressing byte-identical audio, C7), introduce a **post-parser** that takes the current story blob and extracts the HyWorld fields as a *sidecar*. Story node output stays unchanged for audio; sidecar feeds visual.

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

`StoryParser` node reads the blob, runs a small structured-extraction LLM pass (or regex + heuristics for fast path), emits `StructuredScript` dataclass:

```python
@dataclass
class StructuredScript:
    title: str
    acts: list[Act]

@dataclass
class Act:
    number: int
    scenes: list[Scene]

@dataclass
class Scene:
    act: int
    number: int
    setting: str              # spatial description only
    time_of_day: str
    characters_present: list[str]
    persistence_key: str      # hash(setting + time_of_day)
    beats: list[Beat]

@dataclass
class Beat:
    mood: Literal["tense", "melancholy", "action", "dialogue", "transition"]
    duration_sec: float
    speaker: str | None
    line: str | None
    sfx: str | None
    shot_hint: str | None
```

**Phase B — Prompt-aware emission (only if parser proves lossy):**

If parser extraction accuracy falls below ~85% on a test corpus, amend the Story node prompt to emit structured section markers (e.g. `[SETTING: ...]`, `[MOOD: tense]`) in addition to the narrative prose. Audio pipeline strips markers; visual pipeline reads them. This is a prompt engineering change, not an architecture change — still preserves audio byte-identity after marker strip.

---

## 6. Target Full Workflow (v2.1)

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
           │                 │ - Reads Scene.setting,           │
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

## 7. Variables: From-To Mapping

| OTR variable | v2.0 usage | v2.1 extraction path | HyWorld consumer |
|---|---|---|---|
| Episode title | Filename | Unchanged | Opening card overlay |
| Act number | Audio pacing | Preserved in `Act.number` | Scene re-entry check |
| Scene prose | LTX prompt | Parsed into `Scene.setting` + `Scene.time_of_day` | `OTRWorldGen` prompt |
| Dialogue line | TTS | Preserved in `Beat.line` | Audio only (not visual) |
| Speaker tag | TTS voice select | Preserved in `Beat.speaker` | `ShotSelector` (whose OTS shot) |
| Sound cues | AudioGen | Preserved in `Beat.sfx` | Audio only |
| (new) mood | N/A | Derived by `StoryParser` LLM tagger | `ShotSelector`, `CameraPath` motion profile |
| (new) beat duration | Implicit | Computed from dialogue char count × TTS rate | `CameraPath.duration` |
| (new) persistence key | N/A | Hash of setting + time_of_day | `OTRWorldGen` cache lookup |
| (new) characters_present | N/A | Extracted from dialogue + mention graph | Future HY-Embodied placement |

---

## 8. Constraints (Non-negotiable)

| ID | Rule | Inheritance |
|---|---|---|
| C7 | Audio byte-identical to v2.0 baseline | v2.0 |
| C6 | No visible lip sync — characters may appear but never talking head-on | v2.0 + extended in v2.1 |
| C3 | All visual generation in subprocesses | v2.0 |
| C4 | LTX clips capped at 10-12s; crossfades via ffmpeg | v2.0 (narrowed to action bursts only in v2.1) |
| C5 | LTX uses `torch.float8_e4m3fn` | v2.0 |
| V1 (new) | HyWorld subprocess must fit within 14.5 GB real-world VRAM target | v2.1 |
| V2 (new) | World cache keyed by deterministic hash; identical setting across acts = one world | v2.1 |
| V3 (new) | ShotSelector output must never select a shot that shows lips frontally during speech | v2.1 |

---

## 9. Open Questions

1. **HyWorld 2.0 input spec** — will the scene prompt shape match 1.5? (Assumption: yes. Release watcher daily task will resolve.)
2. **HY-Embodied integration** — is it required for v2.1, or deferred to v2.2? (Recommendation: defer. Ship v2.1 with environment-only HyWorld first.)
3. **Parser accuracy threshold** — what % extraction accuracy triggers Phase B prompt markers? (Proposal: 85% on test corpus of 20 episodes.)
4. **World cache storage** — disk format for cached worlds? (HyWorld 1.5 uses its own format; defer until we prototype against 1.5.)
5. **Beat duration estimation** — TTS char count × rate is approximate. Do we run TTS first, measure actual duration, then feed CameraPath? (Proposal: yes. Audio leads visual in timing, always.)

---

## 10. Next Actions

1. Create `v2.1-hyworld-prep` branch off `v2.0-alpha` (not before v2.0-alpha ships stable).
2. Install `jtydhr88/comfyui-custom-node-skills` to `.claude/skills/` on that branch.
3. Scaffold `StoryParser`, `OTRWorldGen`, `ShotSelector`, `OTRWorldRender` as empty node classes with correct `INPUT_TYPES` and return signatures.
4. Prototype `StoryParser` first — it's the only node that can be built and tested without HyWorld installed.
5. When daily release watcher pings that HyWorld 2.0 is live: fetch model, verify sm_120 compatibility, fill in `OTRWorldGen` body, run one-scene smoke test.

---

*End of problem statement. Revise as scaffolding proceeds.*
