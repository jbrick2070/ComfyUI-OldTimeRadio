# Question -- 2026-04-17

# HyWorld trio wiring into otr_scifi_16gb_full.json -- design consult

**Date:** 2026-04-17
**Branch:** v2.0-alpha
**Decision owner:** Jeffrey Brick (with Claude synthesis)
**Objective:** Wire 3 registered-but-unused HyWorld nodes into the unified workflow JSON without breaking the audio pipeline or exceeding the VRAM ceiling.

---

## 1. Current workflow state (facts, not opinions)

`workflows/otr_scifi_16gb_full.json` is the ONE unified workflow. 10 active nodes, 20 links, zero orphans, zero muted. There is no separate visual workflow file. There will not be one; the v2.0-alpha rule is *one big JSON*.

Ten active nodes and their data flow:

| ID | Node type | Key outputs |
|----|-----------|-------------|
| 1 | OTR_Gemma4ScriptWriter | `script_json` (STRING) -> [3, 11, 12, 13, 15] |
| 2 | OTR_Gemma4Director | `production_plan_json` (STRING) -> [3, 11, 12, 14, 15] |
| 3 | OTR_SceneSequencer | `scene_audio` (AUDIO) -> [4]; `scene_manifest_json` (STRING) **unused**; `render_log` **unused** |
| 4 | OTR_AudioEnhance | `enhanced_audio` (AUDIO) -> [7] |
| 7 | OTR_EpisodeAssembler | `episode_audio` (AUDIO) -> [12]; `output_path` (STRING) **unused**; `episode_info` **unused** |
| 11 | OTR_BatchBarkGenerator | `tts_audio_clips` (AUDIO) -> [3] (holds GPU ~12-18 min per episode) |
| 12 | OTR_SignalLostVideo | takes final AUDIO + script_json + production_plan_json -> `video_path` (terminal, no consumers) |
| 13 | OTR_KokoroAnnouncer | `announcer_audio_clips` (AUDIO) -> [3] |
| 14 | OTR_MusicGenTheme | `opening_audio` + `closing_audio` -> [7] |
| 15 | OTR_BatchAudioGenGenerator | `sfx_audio_clips` (AUDIO) -> [3] |

**Hard facts:**
- Node 3's `scene_manifest_json` output is currently dangling (no consumers).
- Node 7's `output_path` (STRING, the on-disk final WAV path) is currently dangling.
- Node 12 (`OTR_SignalLostVideo`)'s `video_path` output has no consumers either; it is the de-facto final render today.
- The C7 invariant: audio output must be byte-identical to the v1.5/v1.7 baseline on every run.
- VRAM ceiling: 14.5 GB real peak. Bark TTS pins the GPU during the full audio phase.

---

## 2. The three nodes to wire

All three live in `otr_v2/hyworld/` and are registered in `__init__.py` but absent from the workflow JSON.

### OTR_HyworldBridge (`otr_v2/hyworld/bridge.py`, class `HyworldBridge`)

Inputs:
- **required** `script_json`: STRING (Canonical Audio Token array)
- **required** `episode_title`: STRING
- **optional** `production_plan_json`: STRING (default `"{}"`)
- **optional** `scene_manifest_json`: STRING (default `"{}"`, provides audio offsets)
- **optional** `lane`: enum ["faithful","translated","chaotic"] default `"faithful"`
- **optional** `chaos_ops`: STRING default `""`
- **optional** `chaos_seed`: INT default 42
- **optional** `sidecar_enabled`: BOOLEAN default True

Outputs: `hyworld_job_id` (STRING), `shotlist_json` (STRING)

Behaviour: writes Director plan + scene manifest to `io/hyworld_in/<job_id>/`, generates deterministic shotlist, **spawns the HyWorld worker subprocess fire-and-forget**. On failure returns job_id prefixed `PARSE_ERROR_` or `SIDECAR_UNAVAILABLE` so downstream can fall back.

### OTR_HyworldPoll (`otr_v2/hyworld/poll.py`, class `HyworldPoll`)

Inputs:
- **required** `hyworld_job_id`: STRING

Outputs: `hyworld_assets_path` (STRING), `status` (STRING), `status_detail` (STRING)

Behaviour: blocks reading `io/hyworld_out/<job_id>/STATUS.json` until a terminal status arrives. Short-circuits to `"FALLBACK"` if the bridge returned a `PARSE_ERROR_*` job_id. Tracks PID liveness with a grace window.

### OTR_HyworldRenderer (`otr_v2/hyworld/renderer.py`, class `HyworldRenderer`)

Inputs:
- **required** `hyworld_assets_path`: STRING (path to `io/hyworld_out/<job_id>/` OR literal `"FALLBACK"`)
- **required** `final_audio_path`: STRING (NEVER modified -- audio byte-identical guarantee)
- **optional** `shotlist_json`: STRING default `"{}"`
- **optional** `episode_title`: STRING default `"Untitled"`
- **optional** `crt_postfx`: BOOLEAN default True
- **optional** `output_resolution`: enum ["1280x720","1920x1080","960x540"] default `"1280x720"`

Outputs: `final_mp4_path` (STRING), `render_log` (STRING). Returns empty string on FALLBACK so upstream graph can route to `OTR_SignalLostVideo` instead.

Worker runs CPU-only today (stub mode: solid colour PNG + ffmpeg `zoompan` Ken Burns). Real anchor image generation (SDXL 1.0 base + period LoRA, per the 2026-04-16 Phase B consult) is not yet implemented in worker.py.

---

## 3. The decisions this consult needs to produce

Only one unified workflow will be shipped. Please answer all five.

### Q1. Topology: parallel branches vs router node

Two viable topologies:

**Option A -- Parallel branches, both run, pick at post-step**
- `OTR_SignalLostVideo` stays as-is (fed from EpisodeAssembler AUDIO, Node 12 remains terminal).
- HyWorld trio runs in parallel on the same episode. Renderer's `final_mp4_path` is a *second* MP4 on disk.
- The user / episode notes pick which MP4 is the canonical one.
- If HyWorld fails, renderer returns "" and we ignore that branch.
- Pros: zero risk to existing audio+video path. No new node. No conditional edge. Handles HyWorld failure by ignoring the empty string. Works even if HyWorld never ships for real.
- Cons: Two MP4s on disk per run when HyWorld succeeds. CPU spend on stub Ken Burns we don't use.

**Option B -- Router: HyWorld primary, SignalLostVideo as fallback**
- New `OTR_VideoRouter` (or a simple String-switch) picks between HyWorld `final_mp4_path` and SignalLost `video_path` based on Poll's `status`.
- Single canonical MP4 per run.
- Pros: clean single-output graph. No duplicate renders.
- Cons: requires a NEW node we don't have yet. Router adds a serialisation point. Router failure = no video at all.

**Ask:** Pick one with reasoning. If Option A, confirm we do NOT need a router node for v2.0-alpha. If Option B, sketch the minimum router node interface (one paragraph).

### Q2. Wire the three dangling sockets

Three currently-unused outputs line up with HyWorld inputs:

| Dangling output | Obvious consumer |
|------|------|
| Node 3's `scene_manifest_json` | Bridge's optional `scene_manifest_json` input |
| Node 7's `output_path` (STRING, final WAV on disk) | Renderer's required `final_audio_path` input |
| Bridge's `shotlist_json` output | Renderer's optional `shotlist_json` input |

**Ask:** Are all three of these wirings correct? Any foot-gun -- e.g. does `output_path` exist on disk at the moment the Renderer runs, or only after the whole graph completes? Any reason `scene_manifest_json` should NOT be fed to Bridge (e.g. timing mismatch)?

### Q3. Bridge execution order and the head-start

Bridge's minimum preconditions: `script_json` (from ScriptWriter, Node 1) and `production_plan_json` (from Director, Node 2). That means Bridge is unblocked the moment both those LLM nodes finish -- well BEFORE Bark TTS starts its 12-18 min audio render.

ROADMAP.md P1 item #6 (blocked on Gate 0) proposes `Head-Start async pre-bake` exactly for this reason: kick off HyWorld worker while ScriptWriter/Director/Bark still run, so geometry generation overlaps audio generation.

`scene_manifest_json` is produced LATER (inside Node 3 SceneSequencer, which needs TTS audio). That means either:

- (a) Bridge waits for Node 3 and loses the head-start, OR
- (b) Bridge runs without scene_manifest_json (`"{}"`) and re-reads it later from disk in the worker, OR
- (c) Bridge is duplicated: one "preview" Bridge with no manifest that kicks off early, one "final" Bridge with manifest that amends the shotlist.

**Ask:** Which of (a), (b), (c) is the right v2.0-alpha choice? Head-start is theoretically valuable BUT right now the worker is stub-mode and runs in ~seconds on CPU, so head-start has no wall-clock payoff today. Is the simplest answer (a) until real GPU-bound worker models land?

### Q4. VRAM coordination (audio is king)

- Bark TTS holds GPU during Node 11's entire run (~12-18 min).
- Today the worker is CPU-only (stub). No VRAM contention.
- When real models land (SDXL anchor gen, SPAG4D/ComfyUI-Sharp splats, WorldPlay-5B), the worker WILL want GPU time.
- `otr_v2/hyworld/vram_coordinator.py` already exists (I have not read it in this session; I know the file is there).

**Ask:** For v2.0-alpha wiring, can we leave VRAM coordination out of the workflow JSON entirely (handle it inside `worker.py` via a mutex/file-lock) and only surface it if we later promote it to a real ComfyUI node? Or should we introduce one coordinator node now to future-proof the graph? Prefer whichever minimises JSON churn on v2.0-alpha.

### Q5. Node 12 (OTR_SignalLostVideo) status

Today Node 12 is the terminal "video" node. After HyWorld wires in:

- Under Option A: Node 12 stays. Both renderers run in parallel.
- Under Option B: Node 12 becomes a branch into the router.

**Ask:** Is there any reason to remove or reduce Node 12 from the graph in this commit? I lean toward NO because SignalLostVideo is the proven baseline and the C7 audio-is-king rule says don't break what ships. Please confirm or push back.

---

## 4. Constraints that cannot be violated

- Audio output byte-identical (C7). Renderer must not re-encode the WAV; its only audio op is `-c:a copy` at the mux step.
- `CheckpointLoaderSimple` is BANNED in the main graph (C2).
- All GPU-heavy diffusion work runs in subprocesses with `multiprocessing.get_context("spawn")` (C3).
- 14.5 GB VRAM peak ceiling.
- LTX-style clips max 10-12s, auto-chunk + ffmpeg crossfade (C4). Not directly relevant to HyWorld wiring but constrains Renderer's output pattern.
- v2.0-alpha branch only. Main is frozen.
- No subdirectories under `docs/`. No subdirectories under `workflows/`. One unified JSON.

## 5. Out of scope for this consult

- Whether to replace the stub worker with real SDXL+LoRA anchor gen. That's a separate Phase B/C decision (already consulted 2026-04-16).
- HyWorld Gate 0 empirical measurements. Those are P0/P1 blocked items; not required for JSON wiring.
- Renaming or restructuring any of the 10 existing nodes.

## 6. Output format requested

For each of Q1..Q5, give a direct answer plus 2-5 sentences of reasoning. After the five answers, give a SINGLE "commit plan" section in the form:

```
COMMIT PLAN
  Files to modify:   <list>
  New nodes in JSON: <list of node types + which existing nodes feed them>
  New links:         <from_node.output -> to_node.input triples>
  Existing links:    <unchanged / removed / modified>
  Test after:        <what regression suite + smoke run>
  Rollback:          <revert SHA; anything else needed>
```

End with a single FIRST-MOVE sentence: the very next coding action.
