# RUN 245 TIMEOUT — round-robin consult synthesis (2026-04-17)

## Symptom
- Soak RUN 245 TIMEOUT at 1805s (poll ceiling 1800s).
- Dialogue lines generated: **337** (Script + Director succeeded).
- Treatment file: **NOT** written.
- Error field: empty.
- VRAM peak: 10.04 GB (under 14.5 ceiling).
- ComfyUI queue still shows `running=1, pending=0` for multiple minutes after soak give-up — the run is still chewing on the GPU.
- Active workflow: `workflows/soak_target_api.json` — 10 nodes, NO HyWorld trio.

## ChatGPT (gpt-4.1) verdict — WRONG top candidate
Top pick: `HyworldPoll.execute` infinite loop (`otr_v2/hyworld/poll.py`).
**Incorrect** — the HyWorld trio is not wired into `soak_target_api.json`.
ChatGPT assumed the trio was active because it was in the code bundle.

## Gemini (2.5-flash) verdict — correct scope
Top pick: **`BatchBarkGenerator` Bark model load / CUDA init hang**.
Runner-up: `SignalLostVideo` ffmpeg subprocess deadlock.
Third: `EpisodeAssembler` ffmpeg subprocess deadlock.
Recommends killing ComfyUI PID 52904 now (hang not self-resolving).

## Evidence review (Claude)
Gemini's "Bark load" pick has a flaw: the soak's 337-line count comes from
parsing the Director's `script_json`, so Bark load succeeding proves nothing
about Bark load failing. **BUT**: line 584 of `batch_bark_generator.py` calls
`_load_bark("suno/bark")` which does a blocking HuggingFace download/load — a
documented hang class when the hub returns a partial response or CUDA init
stalls. It is a plausible wedge point. Counter-evidence: VRAM at 10.04 GB
suggests *some* model IS loaded, consistent with Bark+Kokoro+MusicGen
already resident.

## Most likely wedge (synthesized)
Given 10 GB VRAM held + multiple audio models wired (Bark 8 GB + MusicGen
+ Kokoro + AudioGen), the wedge is most likely one of:

1. **BatchBarkGenerator generation loop** (not load): line 604-633 — if a
   single chunk's `model.generate()` hangs on a malformed emotional tag or
   bracket sequence, the whole batch stalls. No per-line timeout.
2. **MusicGen theme generation** — `facebook/musicgen-medium` at guidance
   scale 3.0 can take several minutes; if it hangs on model.generate there's
   no timeout escape.
3. **BatchAudioGen** — similar risk profile; `facebook/audiogen-medium`
   has known long-tail runs.
4. **EpisodeAssembler or SignalLostVideo ffmpeg subprocess** — pipe deadlock
   if stdout/stderr aren't drained while the child blocks.

## Instrumentation plan (next ComfyUI restart)
One log line per node entry + one before each major subprocess call:

- `batch_bark_generator.py:509` — `generate_batch()` entry
- `batch_bark_generator.py:584` — before `_load_bark("suno/bark")`
- `batch_bark_generator.py:604` — loop header with `generated/total`
- `kokoro_announcer.py` — entry
- `musicgen_theme.py` — entry + before `model.generate`
- `batch_audiogen_generator.py` — entry + before `model.generate`
- `scene_sequencer.py:EpisodeAssembler.execute` — entry + before ffmpeg
- `video_engine.py:SignalLostVideoRenderer.execute` — entry + before ffmpeg

All log lines use a consistent prefix `[WEDGE]` so we can grep the next run
log and see exactly the last `ENTER` / `BEFORE_FFMPEG` that printed.

## Kill policy (per Jeffrey)
**Do not kill yet.** Jeffrey explicitly directed: wait for RUN 246. If
RUN 246 reproduces the same 337-lines + no-treatment signature, we have
a deterministic repro and can kill-restart to load instrumentation for
RUN 247.

## Side-finding: "one glorious video" direction
Full A/V workflow (`workflows/otr_scifi_16gb_full.json`) already wires
HyworldBridge → HyworldPoll → HyworldRenderer, with `episode_audio` fed
from EpisodeAssembler into HyworldRenderer. Renderer uses `ffmpeg -shortest`
on mux (line 245 of `otr_v2/hyworld/renderer.py`), which:
- Trims video if it's longer than audio ✓
- **Trims audio if it's shorter than video ✗** (C7 violation risk).

Action: scale shotlist durations to sum to actual audio duration (ffprobe
the audio first), OR add explicit `-t <audio_duration>` on mux output.

Task tracker: #29 (audio-length exact-match), #30 (soak → full rig).
