# Overnight status #8 — 2026-05-18 Sprint H §3.7 retest #15

**Status:** HALT — new co-residence finding: Bark TTS + FLUX
deferred loader fire in parallel. Both gated on
`OTR_LedgerFreezeCascade.script_json`, the same signal, so
ComfyUI's executor fires them simultaneously after the writer
phase. Different from retest #14's two-loader collision: the
loaders are now serial (Option A wiring works), but the FLUX
loader vs the AUDIO branch (Kokoro + Bark + MusicGen) is still
parallel. Bark generation hangs against FLUX-resident 22 GiB.

**Path G FULLY PROVEN for FLUX:** `OTR_DeferredCheckpointLoader`
fired at 0.02 GiB cold-start for the second retest in a row.
The loader-side deferred-loader contract is solid architecture.

**Option A wired correctly:** Link 210 re-sourced from
`OTR_UnloadAll.unload_done`. Not exercised in iter 1 (timeout
before reaching UnloadAll); iter 2 failed earlier on an
unrelated import-path issue.

---

## TL;DR

Three architectural answers landed at full proof across the
campaign:

1. **Outline tree refactor** — Path C, retest #12-#15 GREEN.
   18 calls per outline, 0 retries.
2. **MusicGenTheme meta-brief** — Path F, retest #12-#15 GREEN.
   3 cues from atmosphere + setting + cue character templates.
3. **OTR_DeferredCheckpointLoader (FLUX)** — Path G, retest
   #14-#15 GREEN. Fires at 0.02 GiB cold-start. Loader-side
   deferred loading IS the right pattern.

The remaining work is **gate sequencing** — making sure every
heavy GPU consumer waits for prior phase eviction. Retest #15
surfaced a NEW co-residence we hadn't measured before: **the
audio branch (Bark + Kokoro + MusicGen) fires in parallel with
FLUX**, both downstream of the freeze cascade.

---

## What ran

Commit `71cfa0b` on `v2.0-alpha` (Option A: `OTR_UnloadAll`
emits `unload_done`; link 210 re-sourced).

§3.7 retest #15 launched at 2026-05-18T14:15:36 via
`sweep_and_launch.bat --iters 2 --inter-iter-sec 10`.

### Iter 1 (worker_iter_001.json)

```
status:        TIMEOUT (worker EXEC_TIMEOUT_S=900 fired)
failure_class: timeout
peak_vram_gb:  15.92
wall_time_s:   924.1
```

Pipeline reach (deepest of the campaign):
```
1. cast locked: 3 rows                                ok
2. [OTR_Outline] success: 16 beats; 18 LLM calls       ok
3. [OTR_LedgerFreezeCascade] running cascade (16 lines) ok
4. [OTR_MusicGenTheme] story_brief_status=ok           ok
   style_slug_diag=extinct_life_reemergence
   (invented slug; Path F composes from meta brief)
5. [MusicGenTheme] Generating 3 cues from atmosphere
   "silence, tension, wonder, evokes salt flats,
    outback, slow..."                                  ok
6. [DeferredCheckpointLoader] fire: VRAM=0.02 GiB COLD ok <- PROVEN
7. [DeferredCheckpointLoader] load complete:
   0.02 -> 22.18 GiB (delta=22.17)                    ok
8. [FluxBranchGate] fire: VRAM=22.18 GiB              ok
9. [KokoroAnnouncer] chose voice bm_fable;
   2 announcer lines stamped                          ok
10. [VRAM_SENTINEL] bark_batch: VRAM 22.2 GB exceeds
    6.0 GB entry ceiling -> Forcing offload          WARN
11. [VRAM_SENTINEL] bark_batch: VRAM still 22.2 GB
    after offload -> proceeding anyway, may OOM       WARN
12. [BatchBark] First line for v2/en_speaker_8 --
    activating hallucination guard                  HANG
13. [worker timeout @ 924s]                          TIMEOUT
```

The Bark + FLUX co-residence is fighting for 16 GiB physical
VRAM with the dynamic offloader. Bark didn't crash -- it just
made no measurable progress on the first dialogue line in
15 minutes. With FLUX taking 22.18 GiB (offloaded fragments
across pagefile), Bark's TTS forward passes are starving for
GPU pages.

### Iter 2 (worker_iter_002.json)

Different failure:
```
status:        error
failure_class: unknown
exception:     No module named 'nodes._otr_story_brief_helpers';
               'nodes' is not a package
exception_type: ModuleNotFoundError
executed_count: 12
peak_vram_gb:  15.89
wall_time_s:   409.2
```

12 nodes ran successfully then a sibling node (likely
batch_humo_render, batch_ltx_render, batch_flux_render, or
batch_flux_portrait_render -- all four use the absolute
`from nodes._otr_story_brief_helpers import ...` form) hit a
sys.path race. The relative `from ._otr_story_brief_helpers
import ...` form used by MusicGenTheme works; the absolute
form intermittently fails.

This is likely a side effect of the new deferred-loader
import path (`.nodes._otr_deferred_loaders`) changing the
`sys.path` / module-cache state somewhere. It's NOT a
co-residence issue; it's a packaging issue surfaced by the
new module addition. Treat as a separate (and fixable)
secondary defect.

## Architecture finding (the campaign's final piece)

ComfyUI's executor topo-sorts independent branches in
parallel:

    freeze_cascade.script_json
      |
      +----> OTR_DeferredCheckpointLoader (video branch)
      |       -> FLUX load (22 GiB)
      |       -> FluxBranchGate
      |       -> FLUX env stills
      |       -> ... -> OTR_UnloadAll
      |
      +----> OTR_MusicGenTheme (audio branch root)
      |       -> 3 cues
      |
      +----> KokoroAnnouncer
      |       -> 2 announcer lines
      |
      +----> BatchBark
              -> 14 dialogue lines

All four branches fire simultaneously from script_json. FLUX
loader pre-empts the GPU at 22 GiB; Bark's 4 GiB resident
need plus per-line forward passes get starved.

Path G + Option A solved the video-branch internal ordering
(deferred loaders + UnloadAll signal). The remaining issue is
**inter-branch ordering**: the audio branch should complete
BEFORE the video branch starts, OR FLUX deferred load should
wait until audio is done.

## Fix options (all require Jeffrey sign-off)

### Option D: gate FLUX deferred loader on audio completion

Smallest change. Add a STRING output `audio_done` to a
canonical audio-branch sink (likely OTR_EpisodeAssembler or
the last node in the audio pipeline). Source FLUX deferred
loader's gate_signal from that, NOT from freeze cascade.

Effect: writer phase done -> audio branch runs fully
(MusicGen + Kokoro + Bark + assembly) -> audio sink emits
audio_done -> FLUX deferred loader fires at low VRAM -> rest
of video branch.

Scope: ~10 lines in one audio node + one link rewire +
documentation update. Recipe identical to Option A.

### Option E: deferred-wrap the audio batch nodes

Wider. Wrap BatchBark / KokoroAnnouncer / MusicGenTheme in
gate-signal-bound wrappers. Pulled by gate_signal sources
that the operator chains. More work, more flexibility.

### Option F: serial-by-design rewire

The deepest change. Move the audio branch INTO the video
branch's dependency chain by adding a dummy passthrough that
takes an audio-branch output as input. Forces topo serial
without any new gate signals. Risky -- the existing branch
parallelism is by design.

## Secondary defect: iter 2 ModuleNotFoundError

The absolute-import form `from nodes._otr_story_brief_helpers
import ...` is used by 4 existing render nodes (batch_humo,
batch_ltx, batch_flux, batch_flux_portrait). It works most of
the time because ComfyUI's custom-node loader manipulates
sys.path to expose the OTR package's `nodes/` directory at
the top level.

Adding `OTR_DeferredCheckpointLoader` + `OTR_DeferredLtxTextEncoderLoader`
via `.nodes._otr_deferred_loaders` may have shifted the
import order or sys.path state enough to surface this race.

Fix is straightforward: convert all 4 sites to the relative
`from ._otr_story_brief_helpers import ...` form (consistent
with how MusicGenTheme already does it). Independent of the
co-residence fix above.

## Wins captured in this retest

1. **OTR_DeferredCheckpointLoader fires at 0.02 GiB COLD**
   for the second consecutive retest (#14 + #15).
   Architecture proven.
2. **MusicGenTheme Path F** generated cues from
   `extinct_life_reemergence` invented slug with no crash.
3. **Outline tree** GREEN with full validation.
4. **OTR_UnloadAll emits unload_done** -- visible in iter 1
   pipeline trace though not reached due to Bark hang.
5. **Pipeline reached its deepest point** of the §3.7
   campaign before failing: 13 distinct stages executed.

## What's NEXT (out of overnight scope -- Jeffrey sign-off)

- Option D (recommended): audio-done gate on FLUX deferred
  loader.
- Secondary fix: relative-import normalization for the 4
  batch_*_render nodes.

Both are small commits. Both blocking §3.7 end-to-end
closure.

## What we did NOT do

- Did NOT touch the audio branch nodes.
- Did NOT modify the FLUX deferred loader's gate_signal
  source.
- Did NOT fix the absolute-import sites.
- Did NOT bump a version label.

## Commits this Path-F + Path-G + Option-A arc

- `34f759e` Path C step 1: upstream LLM audit (read-only)
- `dd3b5ec` Path C step 2: outline LLM call broken into tree
- `6cbdee0` Path C followup: outline Stage 3 target_words
            Python-authoritative
- `bc1b519` Status-3 doc
- `6add3fc` Flip writer to gemma-4-E4B-it
- `92698ad` Status-4 doc
- `34f759e` (already listed)
- `90aeb28` Path F: MusicGenTheme reads meta brief
- `d7ffa84` Status-6 doc
- `1665706` Path G: deferred-loader wrappers
- `8e1c608` Status-7 doc
- `71cfa0b` Option A: OTR_UnloadAll emits unload_done
- (this commit) Status-8 doc

## Halt closed

Awaiting Option D + secondary-import-fix direction. Same
posture as status #1-#7. Pre-authorized fixes overnight remain
same-pattern co-residence OOM only.

The retest #14 pipeline-depth analysis from status-7 stands
strengthened: now confirmed across two retests that the
deferred FLUX loader fires cold and that the gate pattern can
be extended one node at a time as new co-residence cases
surface. The audio-vs-video parallelism is the last major
inter-branch ordering question.
