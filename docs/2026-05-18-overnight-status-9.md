# Overnight status #9 — 2026-05-18 Sprint H §3.7 retest #16

**Status:** HALT — pipeline reached FLUX env stills phase for
the FIRST time in the §3.7 campaign. Audio branch GREEN
end-to-end. Tripped on a pre-existing code defect in
`batch_flux_render.py` (missing function in `_otr_paths`),
NOT a co-residence issue. Per Jeffrey retest #16 branching:
"RED on co-residence elsewhere -> auto-fix per pre-authorization"
applies to co-residence -- this is "anything else" -> halt +
status-9.

**§3.7 audio side CLOSED.** Audio-is-king serialization
proven structurally working: audio runs cold, audio_done emits,
FLUX deferred fires cold. The remaining work is a code-defect
fix in BatchFluxRender's radio-bookend save path.

---

## TL;DR

Three architectural answers landed in ONE retest, end-to-end
through the audio side:

1. **Outline tree (Path C)** -- GREEN. 16 beats, 18 LLM calls,
   no retries. Writer DONE at 269 words.
2. **MusicGenTheme meta-brief (Path F)** -- GREEN. 3 cues
   composed from atmosphere + setting + period descriptor.
3. **OTR_DeferredCheckpointLoader (Path G)** -- GREEN. Fired
   AFTER audio branch finished (third consecutive cold fire
   across retests #14-#16).
4. **EpisodeAssembler audio_done signal (Option D)** -- GREEN.
   Emitted exactly when the assembler finishes the 174s
   episode audio.

The pipeline serialized cleanly:

```
writer (16 lines, 269 words, est_minutes=1.9)
  -> freeze cascade (16 lines)
       -> MusicGenTheme 3 cues from meta brief
            ("survival, tension, symbiosis, evokes station,
              galley, slow attack")
       -> KokoroAnnouncer 2 announcer lines
       -> BatchBark 14 dialogue lines (no FLUX co-residence
          warning -- audio ran cold)
            -> SceneSequencer (14 TTS + 1 SFX + 2 ANNOUNCER)
                 -> AudioEnhance 157.1s episode audio
                      -> EpisodeAssembler 174.13s, 3 segments,
                         crossfaded
                           -> emit audio_done signal at 48 kHz
                              -> ... FLUX deferred fires ...
                                   -> AttributeError in
                                      batch_flux_render.py:910
```

This is the **deepest pipeline reach of the entire §3.7
campaign**. Every architectural fix this Cowork session
landed is now exercised end-to-end on the audio side.

---

## The defect

```
AttributeError: module '_otr_paths' has no attribute
                'otr_legacy_audio_dir'
  File ".../visual/batch_flux_render.py", line 910,
       in _render_and_save_radio_bookend
    [_OTRP.otr_episodes_root(), _OTRP.otr_legacy_audio_dir()]
```

`_OTRP` is `_otr_paths`. The function `otr_episodes_root()` exists
(returns successfully). The function `otr_legacy_audio_dir()`
does NOT exist on the current `_otr_paths` module.

This is a pre-existing code defect that the §3.7 campaign just
unlocked. Before this retest, the workflow never reached
BatchFluxRender (because of upstream co-residence crashes and
hangs). Now that the deferred loaders + audio_done gate finally
let the pipeline through, the missing function trips.

ComfyUI's executor tried to recover (multiple "Unloaded
partially" lines), but the AttributeError eventually exhausts
the executor's per-prompt retry budget. Worker exec timeout
fires at 924s = 15 min.

`worker_iter_001.json`:
```
status:        TIMEOUT
failure_class: timeout
peak_vram_gb:  15.87
wall_time_s:   924.5
```

## Architectural answers proven by this retest

| Path / Option | First fired | Now fired |
|---|---|---|
| Path C outline tree | retest #12 | #12-#16 |
| Path F MusicGen meta brief | retest #12 | #12-#16 |
| Path G FLUX deferred loader | retest #14 | #14-#16 |
| Option A unload_done signal | retest #15 (wired, not fired) | exercised when LTX phase reaches it |
| Option D audio_done signal | THIS RETEST | exercised end-to-end on the audio side |

Five separate architectural commits across the Cowork session
produced a pipeline that runs audio cold, FLUX cold, in series,
with measurable gate-fire telemetry at every junction.

## Recommended next fix (out of pre-authorized scope)

`nodes/_otr_paths.py` is missing `otr_legacy_audio_dir()`.
Either:

A. **Add the missing function** with the expected return value
   (whatever the legacy audio root directory is on Windows).
   Likely 1-3 lines. Smallest fix.
B. **Remove the call site** if `otr_legacy_audio_dir()` is
   stale code from before a path refactor. Edit
   `batch_flux_render.py:910` to drop that element.
C. **Audit the radio-bookend save path** for any other stale
   references. The path was probably touched during a prior
   sprint (e.g. Sprint D voice-path cleanbreak) and one rename
   wasn't propagated.

Recommendation: **Option A**, add the missing function as a
thin wrapper that returns the same value the legacy code
expected. Belt-and-braces; preserves the radio-bookend save
behavior the function was designed to support.

## What's NEXT (out of overnight scope)

- Fix `_otr_paths.otr_legacy_audio_dir` (or its call site).
- Retest #17 expected sequence:
    - audio branch GREEN (proven this retest)
    - FLUX deferred fires cold (proven this retest)
    - FLUX env stills render cleanly
    - FLUX portraits render
    - OTR_UnloadAll fires + emits unload_done
    - LTX deferred fires cold
    - LTX motion clips
    - HuMo Phase A/B/main render
    - ffmpeg composite
    - 3.7 CLOSED

## What we did NOT do (per directive)

- Did NOT modify `_otr_paths.py`.
- Did NOT touch the FLUX render path.
- Did NOT touch any gate / loader.
- Did NOT bump a version label.

## Commits this Cowork session (the full arc)

| Commit | Path / Option |
|---|---|
| `0ce8d2b` | reconcile harness to single workflow source of truth |
| `34f759e` | Path C step 1: upstream LLM audit |
| `252ea1f` | smoke target_words 30 -> 300 + writer_outline classifier |
| `bf554b0` | status-2 doc |
| `0ebef36` | smoke act_count 1 -> 3 + writer_budget classifier |
| `960e376` | status-3 doc |
| `6add3fc` | flip writer to gemma-4-E4B-it |
| `92698ad` | status-4 doc |
| `dd3b5ec` | Path C step 2: outline LLM call broken into tree |
| `6cbdee0` | Path C followup: target_words Python-authoritative |
| `90aeb28` | Path F: MusicGenTheme reads meta brief |
| `d7ffa84` | status-6 doc |
| `1665706` | Path G: deferred-loader wrappers |
| `8e1c608` | status-7 doc |
| `71cfa0b` | Option A: OTR_UnloadAll emits unload_done |
| `ed5e78f` | status-8 doc |
| `d3253ab` | **Option D: serialize audio before FLUX + import-race fix** |
| (this commit) | status-9 doc |

Sixteen commits. Three architectural breakthroughs (outline tree,
meta brief, deferred loader). Two ordering signals (unload_done,
audio_done). One full bug-hunt harness (commit chain pre-`0ce8d2b`
in earlier sessions). Pipeline depth advance from "fails at LTX
text encoder load" to "audio side end-to-end GREEN, FLUX side
unblocked, tripped on pre-existing code defect."

## Halt closed

Awaiting `_otr_paths.otr_legacy_audio_dir` fix direction. Same
posture as status #1-#8. Pre-authorized fixes overnight remain
same-pattern co-residence OOM only; halt-and-report unchanged;
hard stops unchanged.

This is the §3.7 audio-side closure moment. The remaining work
is no longer architectural -- it's code maintenance on a
pre-existing surface the campaign just unlocked.
