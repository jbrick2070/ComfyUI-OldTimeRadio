# Overnight status #7 — 2026-05-18 Sprint H §3.7 retest #14

**Status:** HALT — Path G deferred loaders work as designed
INDIVIDUALLY, but the LTX text encoder loader needs a different
gate source than the FLUX loader. Both currently source
`gate_signal` from `OTR_LedgerFreezeCascade.script_json`, which
fires simultaneously, leaving co-residence at the second loader.

Per Jeffrey's retest #14 branching rule:
> "RED on co-residence OOM after deferred load -> check
> OTR_UnloadAll eviction telemetry; may need explicit
> MODEL/CLIP/VAE patcher dereference"

This is exactly that branch. Halt + status-7.

---

## TL;DR

Path G is **half-proven and half-not** in one retest:

**Half-proven:** `OTR_DeferredCheckpointLoader` (FLUX) fires at
**0.02 GiB** — completely cold start. Writer phase fully unwound
before the loader materializes. This is the data-point Jeffrey
asked for. Deferring loader memory residency works.

**Half-not:** `OTR_DeferredLtxTextEncoderLoader` fires at
**22.18 GiB** because both deferred loaders share the same
`gate_signal` source (`OTR_LedgerFreezeCascade.script_json`).
After script_json arrives, the executor fires both deferred
loaders in parallel. FLUX wins the race, allocates 22.17 GiB,
then LTX text encoder fires into the same context and the load
crashes the process with the same `access violation` retest #7
+ #13 produced.

Fix: re-wire LTX text encoder loader's `gate_signal` to source
from a signal downstream of `OTR_UnloadAll`. The unload runs
between FLUX consumers (env stills) and LTX consumers; if the
LTX loader depends on something `OTR_UnloadAll` emits, the
loader can't fire until FLUX has been evicted.

---

## What ran

Commit `1665706` on `v2.0-alpha` (Path G deferred-loader
wrappers). Pre-flight verification:
- AST + JSON parse: clean.
- 170 passed, 3 skipped, 2 xfailed.

§3.7 retest #14 launched at 2026-05-18T13:46:40 via
`sweep_and_launch.bat --iters 2 --inter-iter-sec 10`. Both iters
crashed with `crash_process / access_violation` at peak VRAM
15.89 GB. Supervisor halted on `2 consecutive crash_process
failures`.

### Iter 1 (worker_iter_001.json)

```
status:        CRASH_PROCESS
failure_class: crash_process
crash_subclass: access_violation
peak_vram_gb:   15.89
wall_time_s:    374.95
prompt_id:      73dfbdc8-f4c2-4182-8e73-bba9bc9521ed
```

Key markers from `comfy_session_iter_001.log`:
```
[OTR_Outline] success: 16 beats; 1 macro + 3 phase + 14 beat = 18 total
[OTR_LedgerFreezeCascade] running cascade on ledger (16 lines)
[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[]
                    style_slug_diag=genetic_blueprint_heart_strain
[MusicGenTheme] Generating opening (12s): tension, ambition,
                decay, evokes lab bench, chamber, slow...
[DeferredCheckpointLoader] fire: VRAM allocated=0.02 GiB;
                gate_signal len=25964; ckpt=flux1-dev-fp8.safetensors
[DeferredCheckpointLoader] load complete: VRAM allocated=
                0.02 -> 22.18 GiB (delta=22.17);
                ckpt=flux1-dev-fp8.safetensors
[FluxBranchGate] fire: VRAM allocated=22.18 GiB; writer
                ledger signal received (len=25964)
[DeferredLtxTextEncoderLoader] fire: VRAM allocated=22.18 GiB;
                gate_signal len=25964; text_encoder=
                gemma_3_12B_it_fp4_mixed.safetensors;
                ckpt=ltx-2.3-22b-dev.safetensors; device=default
Windows fatal exception: access violation
  File "torch/storage.py", line 468 in __getitem__
  File "comfy/utils.py", line 136 in load_torch_file
  File "comfy/sd.py", line 1241 in load_clip
  File "nodes_lt_audio.py", line 203 in execute   <- the loader
                                                     wrapped by
                                                     DeferredLtx
                                                     TextEncoder
                                                     Loader
```

### Iter 2 (worker_iter_002.json)

Confirmed identical:
```
status:        CRASH_PROCESS
failure_class: crash_process
crash_subclass: access_violation
peak_vram_gb:   15.89
wall_time_s:    381.0
```

Supervisor halt: `STOP_DECISION: halt: 2 consecutive
crash_process failures`.

## Architecture finding

The deferred-loader contract is sound: `gate_signal`
(`forceInput=True`) successfully makes the ComfyUI executor
topologically defer the loader's `execute()` call. The
FLUX-side proof:

```
allocated 0.02 GiB at fire time -> writer phase fully unwound
allocated 22.18 GiB at load complete -> FLUX resident
```

The Path G design issue: both deferred loaders share the SAME
upstream signal node. ComfyUI's executor satisfies both
dependencies the moment `OTR_LedgerFreezeCascade.script_json`
emits. From there it fires in topo order, but both loaders
are at the same topo depth. FLUX wins (because graph layout
or topo deterministic order), allocates 22 GiB, then LTX
text encoder fires immediately after into the same context.

The OTR_UnloadAll node (node 24, IMAGE-typed) runs after
FLUX env stills produce an IMAGE -- so it IS downstream of
FLUX execution. But its output type is IMAGE (links 83 + 200
go to downstream LTX-side consumers). It does NOT emit a
STRING signal that the LTX text encoder loader could currently
consume.

## Fix options (all require Jeffrey sign-off)

### Option A: Add STRING telemetry output to OTR_UnloadAll

Easiest. Add a second output port to OTR_UnloadAll (slot 1,
"unload_signal", STRING). Emit a small JSON like
`{"delta": "X.XX GiB", "after": "Y.YY GiB"}` from execute().
Re-wire LTX text encoder loader's `gate_signal` (link 210) to
source from this new output instead of node 62 script_json.

After the rewire, the executor topo-sort puts the LTX text
encoder loader STRICTLY downstream of OTR_UnloadAll. The unload
runs after FLUX env stills; LTX text encoder loader can't fire
until OTR_UnloadAll completes.

Scope: ~10 lines in `nodes/visual/unload_all.py` (add the
STRING output + telemetry payload) + one workflow link rewire
(change link 210 source from `[62, 1]` to `[24, 1]`).

### Option B: Make LTX text encoder loader accept IMAGE gate

Wider surface (gate accepts any-typed sentinel) but smaller
workflow change (no new output port). Add a second IMAGE input
to OTR_DeferredLtxTextEncoderLoader; route link from node 24
slot 0 (IMAGE) into it. ComfyUI executor still enforces
ordering via dependency.

Scope: ~5 lines in `_otr_deferred_loaders.py` (add IMAGE input
+ silent-pass) + one workflow link addition (new link from
node 24 slot 0 to node 57's new IMAGE input).

### Option C: Explicit MODEL/CLIP/VAE patcher dereference

Per Jeffrey's branching note. After the FLUX consumer phase
completes, explicitly del the MODEL/CLIP/VAE patcher and
trigger gc + torch.cuda.empty_cache() before the LTX phase
fires. This is effectively what OTR_UnloadAll already does --
but maybe it's not running for the LTX path because LTX text
encoder isn't downstream of node 24 IMAGE.

Scope: would dig into model_management's reference graph.
Larger commit. Probably not necessary given Option A is so
small.

### Recommended path

**Option A.** Smallest, cleanest, surfaces useful telemetry for
future debugging. The unload signal is exactly the kind of
"checkpoint marker" that downstream consumers should be able
to depend on.

## Wins captured this retest

1. **FLUX deferred loader confirmed effective:** fires at
   0.02 GiB. The architectural answer Jeffrey was asking for
   in status-6 now has its proof point. Loader-side deferred
   loading IS the right pattern.
2. **MusicGenTheme Path F still GREEN:** generated 3 cues from
   meta brief (`tension, ambition, decay, evokes lab bench,
   chamber, slow...` etc). Style slug
   `genetic_blueprint_heart_strain` was invented + logged
   diagnostically only.
3. **Outline tree still GREEN:** 16 beats, 18 LLM calls, no
   retries.
4. **FluxBranchGate now redundant safety net:** it still fires
   correctly, but the deferred loader does the real work.

## What we did NOT do (per directive)

- Did NOT add the STRING output to OTR_UnloadAll.
- Did NOT modify OTR_DeferredLtxTextEncoderLoader.
- Did NOT rewire link 210.
- Did NOT touch FluxBranchGate / LtxBranchGate (still in place).
- Did NOT bump a version label.

## Halt closed

Awaiting Option A / B / C direction. Same posture as
status #1-#6. Pre-authorized fixes overnight remain
same-pattern co-residence OOM only; halt-and-report unchanged;
hard stops unchanged.

---

## Note on the iter 1 success span

Even though both iters crashed, iter 1 advanced through the
deepest path of the §3.7 campaign:

```
writer phase (Gemma-4-E4B-it, smoke profile, ~3 min)
  -> style picker
  -> news interpreter
  -> cast lock (3 rows)
  -> outline tree (18 calls, 16 beats)
  -> freeze cascade (16 lines)
  -> MusicGenTheme (3 cues, ~3 GiB GPU during gen, then unload)
  -> DeferredCheckpointLoader (FLUX) fires at 0.02 GiB <- COLD
  -> FLUX load complete (22.18 GiB)
  -> FluxBranchGate fires (correct timing, FLUX consumers wait)
  -> DeferredLtxTextEncoderLoader fires (22.18 GiB) <- collision
  -> access violation -> CRASH_PROCESS
```

One link rewire away from full GREEN. The campaign produced
its core architectural answer + working demonstrator across
retests #12, #13, #14.
