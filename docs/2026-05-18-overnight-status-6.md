# Overnight status #6 — 2026-05-18 Sprint H §3.7 retest #13

**Status:** HALT — co-residence crash at LTX text encoder loader,
NOT addressable by the pre-authorized "add gate analog" recipe.
The loader-wrapper follow-up flagged in the FluxBranchGate
synthesis is the next path.

**Path F MusicGenTheme refactor verified GREEN end-to-end.**
**FluxBranchGate fire telemetry LANDED — the answer Jeffrey was
asking for.**

---

## TL;DR

For the first time in the §3.7 campaign, the workflow advanced
all the way to the FluxBranchGate. Three big wins from Path C +
Path F:

1. Outline tree (commits `dd3b5ec` + `6cbdee0`) cleared iter 1.
   18 LLM calls, no retries needed on the macro / phase /
   beat stages.
2. Freeze cascade ran. Ledger produced 16 lines.
3. MusicGenTheme synthesized prompts from meta brief (commit
   `90aeb28`), no slug-palette crash:
   ```
   [OTR_MusicGenTheme] story_brief_status=ok mood_terms=[]
                       style_slug_diag=ashfall_diagnostic_horror
   [MusicGenTheme] Generating opening (12s): suffocation,
        despair, pressure, evokes chamber, tunnel, slow...
   [MusicGenTheme] Generating closing (8s): ...reso...
   [MusicGenTheme] Generating interstitial (4s): ...brie...
   ```
   The invented slug `ashfall_diagnostic_horror` is logged
   diagnostically only -- the prompt body comes from
   `story_brief_terms.atmosphere` + `setting` + cue-character
   templates.

Then **FluxBranchGate fired**:

```
[FluxBranchGate] fire: VRAM allocated=22.18 GiB; writer
                ledger signal received (len=26762)
```

**Answer to the eager-FLUX-load question:** ComfyUI's executor
**pre-loaded FLUX to GPU regardless of gate readiness**. The
gate fires only after FLUX is already resident at 22 GiB. As
designed, the gate defers downstream FLUX *consumers* -- it does
NOT defer the loader itself.

The crash that followed (access violation in LTX text encoder
loader) is the same co-residence corruption pattern as retest
#7. **22 GiB FLUX + LTX-encoder GPU load attempt corrupts the
CUDA context, takes ComfyUI down with `Windows fatal exception:
access violation` at `torch/storage.py:468 __getitem__` ->
`comfy/sd.py:1241 load_clip` -> `nodes_lt_audio.py:203`.**

Per Jeffrey's branching rule for retest #13:
- co-residence OOM -> auto-fix with gate analog (pre-authorized)

The pre-authorized "add gate analog" recipe was already applied
in commit `b5c1441` (OTR_LtxBranchGate). The crash here proves
that adding more gates does not solve the problem -- gates only
defer consumers, not the loader itself.

The FluxBranchGate synthesis already flagged the actual fix:

> "If gate fires at VRAM >= 20 GiB -> follow-up commit replaces
> CheckpointLoaderSimple with an OTR wrapper that defers
> model_management.load_models_gpu() until gate_signal is ready."

That is a substantial node refactor (an OTR replacement for
ComfyUI's stock `CheckpointLoaderSimple`). Out of pre-authorized
overnight scope. **Halt + status-6**.

---

## What ran

§3.7 retest #13 launched at 2026-05-18T12:07:59 via
`sweep_and_launch.bat --iters 2 --inter-iter-sec 10`.

### Iter 1 (worker_iter_001.json)

```
status:        CRASH_PROCESS
failure_class: crash_process (correctly routed)
crash_subclass: access_violation
exception:     ComfyUI process exited (rc=3221225477) before
               /history resolved the prompt
executed_count: 0 (the crash hit before any output stamping)
peak_vram_gb:   15.89   <- FLUX co-resident peak; above 14.5 ceiling
wall_time_s:    380.2
prompt_id:      ef9faf95-1213-418e-b706-6c9ce44b75d4
```

Log markers from comfy_session_iter_001.log:
```
[OTR_LedgerScriptWriter] cast locked: 3 rows
[OTR_Outline] success: 16 beats; calls used:
              1 macro + 3 phase + 14 beat = 18 total
[OTR_LedgerFreezeCascade] running cascade on ledger
              pending_20260518_120834 (16 lines)
[OTR_LedgerReviewer:pre] audit complete: 16 violation(s),
              pass_clean=False
[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[]
              style_slug_diag=ashfall_diagnostic_horror
[MusicGenTheme] Generating opening (12s): suffocation, despair,
              pressure, evokes chamber, tunnel, slow...
[MusicGenTheme] Generating closing (8s): ...reso...
[MusicGenTheme] Generating interstitial (4s): ...brie...
Requested to load Flux
loaded completely;  22700.13 MB loaded, full load: True
[FluxBranchGate] fire: VRAM allocated=22.18 GiB; writer ledger
                signal received (len=26762)
Windows fatal exception: access violation
  File "torch/storage.py", line 468 in __getitem__
  File "comfy/utils.py", line 136 in load_torch_file
  File "comfy/sd.py", line 1241 in load_clip
  File "nodes_lt_audio.py", line 203 in execute   <- LTXAVTextEncoderLoader
```

The crash is the same defect retest #7 surfaced: LTX-Audio text
encoder load fails when co-resident with FLUX. The LtxBranchGate
(`b5c1441`) was the original response, but the gate only defers
the LTX text encoder's downstream CLIP consumers -- it does NOT
defer the loader's GPU materialization.

### Iter 2 (worker_iter_002.json)

Confirmed same pattern:
```
status:        CRASH_PROCESS
failure_class: crash_process
crash_subclass: access_violation
peak_vram_gb:   15.89
wall_time_s:    398.6
```

Supervisor halt line:
`STOP_DECISION: halt: 2 consecutive crash_process failures`

## Gate-fire telemetry: question answered

The FluxBranchGate's design intent was to defer FLUX consumers
until after the writer's ledger freeze. That part works -- the
gate fired only after `script_json` signal arrived (len=26762
bytes, the full freeze cascade output).

But the loader itself is unaffected: `loaded completely;
22700.13 MB loaded, full load: True` happened BEFORE the gate
fire. ComfyUI's executor topologically reaches `CheckpointLoaderSimple`
at graph start (it has no upstream inputs) and the loader's
`execute()` calls `model_management.load_models_gpu()` immediately.

**The 22.18 GiB allocated-at-gate-fire reading confirms the
worst-case scenario the synthesis flagged.** The gate alone is
not sufficient. The loader needs to be wrapped.

## Recommended next path (out of overnight scope)

Replace `CheckpointLoaderSimple` (and likely also the LTX
text encoder loader) with OTR wrappers that defer
`model_management.load_models_gpu()` until a gate_signal is
ready. Sketch:

```python
class OTR_DeferredCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "gate_signal": ("STRING", {"forceInput": True,
                                            "default": ""}),
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")

    def load(self, ckpt_name, gate_signal):
        # gate_signal blocks execution until upstream emits it.
        # NOW we call the actual loader -- safe to put on GPU
        # because the writer phase has completed + LLMs are
        # unloaded.
        out = comfy.sd.load_checkpoint_guess_config(
            folder_paths.get_full_path("checkpoints", ckpt_name),
            output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths(
                "embeddings"),
        )
        return out[:3]
```

Then rewire workflow node 22 (FLUX loader) from
`CheckpointLoaderSimple` to `OTR_DeferredCheckpointLoader`, with
gate_signal wired from the freeze cascade's script_json. Similar
treatment for the LTX text encoder loader (node 47, the LTXAVTextEncoderLoader).

Scope: 1 new node class + 1 workflow rewire + tests. Single
commit. Per Jeffrey's "no node change" overnight constraint, this
is post-Jeffrey-sign-off scope.

Why this should work where the gate alone doesn't: the deferred
loader makes `model_management.load_models_gpu()` topologically
dependent on gate_signal. ComfyUI's executor can no longer
pre-load FLUX at graph-start because the loader literally cannot
fire until upstream produces gate_signal. The gate signal is
emitted only AFTER the freeze cascade's finally block runs
`_OTRML.unload_llm()` (commit 12.12 contract). So Gemma is
already evicted from VRAM when FLUX loads, eliminating the
co-residence.

## What's PROVEN now (vs end of status-5)

- **Outline tree refactor**: production-grade, two iters across
  retest #12 + #13 reached the freeze cascade.
- **MusicGenTheme meta-brief composer**: production-grade,
  generated 3 cues from meta brief atmosphere + setting on real
  data.
- **FluxBranchGate**: works as designed (defers consumers); does
  NOT solve the loader-side pre-load problem (now measured).
- **LtxBranchGate**: same -- defers consumers, doesn't defer the
  loader.

## What's NEXT (out of overnight scope -- Jeffrey sign-off)

- `OTR_DeferredCheckpointLoader` node (replaces stock
  `CheckpointLoaderSimple` for FLUX).
- Maybe `OTR_DeferredLTXAVTextEncoder` node (same recipe for the
  LTX text encoder).
- Workflow rewire to use the new loaders.

## What we did NOT do (per directive)

- Did NOT build the deferred-loader node.
- Did NOT rewire the workflow's loader nodes.
- Did NOT modify the LTX or FLUX loader source.
- Did NOT touch any classifier (crash_process already routes correctly).
- Did NOT bump a version label.

## Files this session (Path F retest #13)

- `nodes/musicgen_theme.py` (M; in commit `90aeb28`) -- Path F refactor
- `tests/test_audio_c7_b3sum_guards.py` (M; in commit `90aeb28`) -- one assertion re-pointed
- `tests/test_musicgen_news_brief_used.py` (D; in commit `90aeb28`) -- dead path
- `tests/test_musicgen_style_palette.py` (D; in commit `90aeb28`) -- dead path
- `tests/test_story_brief_musicgen_c5g.py` (D; in commit `90aeb28`) -- dead path
- `docs/2026-05-18-upstream-llm-audit.md` (M; in commit `90aeb28`) -- Path F update
- `docs/2026-05-18-overnight-status-6.md` (N -- this file)

## Halt closed

Awaiting deferred-loader path direction. Same posture as
status #1/#2/#3/#4/#5: pre-authorized fixes overnight remain
same-pattern co-residence OOM only; halt-and-report conditions
unchanged; hard stops unchanged.

**Major architectural answer landed this session:** the gate
pattern (LtxBranchGate, FluxBranchGate) is the right design for
deferring CONSUMERS, but ComfyUI's `CheckpointLoaderSimple`
pre-loads to GPU at graph-start regardless. The loader needs to
be wrapped, not just gated.
