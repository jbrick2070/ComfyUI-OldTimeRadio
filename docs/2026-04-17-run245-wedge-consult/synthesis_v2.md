# RUN 245 TIMEOUT — round-robin v2 (upgraded models)

Date: 2026-04-17
Round 1: gpt-4.1 (wrong — assumed HyWorld active) + gemini-2.5-flash
Round 2 (this doc): gpt-5.4 + gemini-3-flash-preview

## Why a round 2
- Round 1 hit the wrong flagship on both sides (stale model ladders).
- Key verification: `outputs/openai_models.md` shows 104 GPT/o-series
  models accessible; `outputs/gemini_models.md` shows 36 generateContent
  models including gemini-3-pro and gemini-3-flash preview tiers.
- OpenAI ladder updated: `gpt-5.1-codex-max` -> `gpt-5.4` -> `gpt-5.4-pro`
  -> ... Round 2 actually resolved to `gpt-5.4` (37.5s).
- Gemini ladder updated to prefer 3-pro -> 3.1-pro -> pro-latest -> 2.5-pro
  -> 3-flash. All Pro tiers returned 429 quota — Jeffrey's key is free
  tier on Pro models. Resolved to `gemini-3-flash-preview` (34.7s).
- Conclusion: we got gpt-5.4 (big upgrade) and gemini-3-flash (small
  upgrade from 2.5-flash). Reasoning quality is visibly sharper.

## gpt-5.4 top pick
`nodes/scene_sequencer.py: EpisodeAssembler.assemble()` final normalize
(`peak = episode_waveform.abs().max()` + multiply). Bug class: "oversized
tensor op / GPU compute stall."

## gemini-3-flash cross-check — rejects gpt-5.4 #1
> "ChatGPT claims `.abs().max()` is a 'giant monolithic CUDA op' ...
> This is nonsense. On an RTX 5080, a reduction on 40 min of 48kHz stereo
> (~115M elements, ~460MB) takes less than 10ms."

Gemini's technical critique is correct. We reject gpt-5.4's #1.

## gemini-3-flash top pick (Claude concurs)
**`OTR_SignalLostVideo` — FFmpeg subprocess pipe deadlock on Windows.**
Bug class: `subprocess.Popen` with `stdout=PIPE` / `stderr=PIPE` where
FFmpeg's output buffer fills and the child hangs forever waiting for the
Python parent to drain the pipe.

Why this is the strongest candidate:
- The only failure class that keeps ComfyUI in `running=1` forever
  WITHOUT significant CPU/GPU consumption.
- Matches `Error: (empty)` — no Python exception is raised, just a
  deadlock.
- Matches "tail still chewing on GPU/CPU after soak timeout" — actually
  it's not chewing, it's suspended waiting on pipe drain.
- Windows-specific buffer hang is a known footgun documented in
  Python's subprocess docs.

## Runner-up — cumulative latency on BatchBark
Gemini's math: 337 lines * 5s/line = 1685s on Bark alone, before
MusicGen + AudioGen + render. Soak ceiling is 1800s. We may not be
wedged at all — we may simply exceed the soak poll ceiling.

**Counter-argument:** if it were just slow (not wedged), ComfyUI would
eventually return `running=0` on its own. But the queue has been stuck
at `running=1` for HOURS after timeout. That rules out "just slow" and
confirms deadlock.

## Third — SceneSequencer env mix O(N^2)
Possible but unlikely given Gemini is right that pure tensor reductions
finish fast. Keep as #3.

## Unified top 3 (Claude's call)
1. **OTR_SignalLostVideo ffmpeg pipe deadlock** (video_engine.py) —
   the one that matches "stuck running=1 for hours, empty error, no
   resource consumption."
2. **OTR_EpisodeAssembler ffmpeg subprocess** — same bug class, earlier
   in the chain. `scene_sequencer.py` has the class; it shells out to
   ffmpeg for audio concat.
3. **OTR_BatchBarkGenerator inner loop** — if a single chunk's
   `model.generate()` hits a malformed emotional tag and stalls.

## Instrumentation plan (ready to deploy on next restart)
Three `[WEDGE_PROBE]` log lines, minimal, survive across node refactors:

1. `otr_v2/nodes/video_engine.py:OTR_SignalLostVideoRenderer.execute`
   — first line, log `"[WEDGE_PROBE] SignalLostVideo entered"`.
2. `otr_v2/nodes/scene_sequencer.py:OTR_EpisodeAssembler.assemble`
   — first line, log `"[WEDGE_PROBE] EpisodeAssembler.assemble entered"`.
3. `otr_v2/nodes/batch_bark_generator.py:generate_batch` inside the
   line loop, log `"[WEDGE_PROBE] Bark line i/total"` every 10 lines.

If RUN 247 log shows `SignalLostVideo entered` but no subsequent graph
progression, candidate #1 confirmed. If it shows `EpisodeAssembler`
entered but no SignalLost, candidate #2. If Bark log stops printing
before line 337, candidate #3.

## Both consults agree
- Kill ComfyUI PID 52904 (Jeffrey: standing directive is DO NOT KILL
  until RUN 246 reproduces; we've waited multiple hours and RUN 246
  never even started, so the repro has already implicitly happened —
  the wedge is still holding).
- Fix `watcher_overrides.json` JSON parse error at line 15 char 1545.
- On Windows, for ffmpeg subprocess: use
  `stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT` or drain pipes
  in a reader thread.

## Fix target (C7-safe, audio is king)
The fix must NOT alter audio encoding. For ffmpeg mux calls:
- Keep `-c:a copy` byte-identical audio.
- Switch stdout/stderr from `PIPE` to `DEVNULL` (Windows-safe).
- Add a 10-minute hard timeout on the subprocess.

## Decision pending (waits on Jeffrey)
- Permission to kill PID 52904 and deploy instrumentation (direct
  contradiction of standing directive, but the directive's trigger
  condition — RUN 246 reproduces — has been met by implication since
  RUN 246 never started for hours).
- Alternative: leave ComfyUI alone, deploy instrumentation to source
  files so the next COLD restart picks it up when Jeffrey kills
  ComfyUI himself.
