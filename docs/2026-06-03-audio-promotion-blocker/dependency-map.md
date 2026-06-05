# OTR audio dependency map -- is the old code holding us back? (2026-06-03)

**Short answer: no.** The codebase is well-factored. The promotion blocker is ONE
architectural choice -- the conflict engines import their library *in the ComfyUI
process* -- and it is isolated to **3 adapter files behind a clean registry seam**.
Nothing pervasive needs rewriting.

## Matrix A -- package-pin conflict (the external constraint)
What each engine demands vs the protected stack (torch 2.10+cu130 / numpy 2.4 /
transformers 5.5 / sm_120). Measured via `pip install --dry-run` + git reqs.

| engine | torch | numpy | transformers | sm_120 | main-venv safe? |
|---|---|---|---|---|---|
| bark | uses stack | 2.x ok | 5.x ok | yes | YES (installed, renders) |
| kokoro | uses stack | ok | ok | yes | YES |
| musicgen (transformers) | uses stack | ok | 5.x ok | yes | YES |
| **chatterbox** | ==2.6.0 | <2.0 ->1.26 | ==5.2.0 | NO | NO -- downgrades stack |
| **indextts2** | ==2.8.* cu128 | ==1.26.2 | ==4.52.1 | maybe | NO -- downgrades stack |
| **stable_audio_tools** | unresolvable | -- | -- | -- | NO -- build fails |
| **SA3 (native)** | ComfyUI's own | ok | ok | yes | YES (no pip dep) |

## Matrix B -- code coupling (audio subsystem x heavy lib; M=module-top, L=lazy)
From a static AST scan of `nodes/`.

| module | torch | transformers | numpy | conflict lib |
|---|---|---|---|---|
| registry.py | . | . | . | . (imports NOTHING -- pure seam) |
| base.py | M | . | . | . |
| eng_bark.py | L | . | L | . |
| eng_kokoro.py | L | . | L | kokoro[L] |
| eng_musicgen.py | L | L | L | . |
| **eng_chatterbox.py** | L | . | . | **chatterbox[L]** |
| **eng_indextts2.py** | L | . | . | **indextts[L]** |
| **eng_stable_audio.py** | L | . | . | **stable_audio_tools[L]** |
| batch_character_voices / announcer / cast_lock | . | . | . | . (0 heavy libs) |

Good news in this matrix: (1) all heavy imports are **lazy** (inside functions) ->
box-fresh ComfyUI import never pulls a conflict lib (C-5 is actually satisfied).
(2) The voice/music NODES import **nothing heavy** -- they route through the
registry. (3) The conflict surface is exactly **3 files**, all behind the seam.

## The in-process conflict surface (the whole blocker)
```
eng_chatterbox.py     imports chatterbox            (in ComfyUI process)
eng_indextts2.py      imports indextts              (in ComfyUI process)
eng_stable_audio.py   imports stable_audio_tools    (in ComfyUI process)
```
"Lazy" defers the crash to *engine-selection* time, but the import still fires in
ComfyUI's interpreter -- so the engine's deps must live in the main venv, which is
the conflict. The laziness is not a fix; the *process* is the problem.

## Load-bearing modules (the real "old code" risk, separate workstream)
Top internal fan-in (119 modules, 156 edges). These are WRITER-side, not audio:

| module | importers | note |
|---|---|---|
| _otr_structured_call | 10 | the LLM structured-call helper -- where the gemma failures originate |
| production_ledger / _otr_paths / _otr_repair_prompts | 8 each | core infra |
| _otr_model_loader | 7 (fan-out 6) | the only high-coupling hub (loads + offloads) |
| registry | 6 (fan-out 0) | audio seam -- depended-on, depends on nothing (ideal) |

Nothing in the audio subsystem is a coupling hotspot. `registry` has 6 importers
and 0 dependencies -- the textbook shape of a healthy seam.

## Verdict + future-proof direction
- The architecture is NOT holding us back; a single in-process decision in 3 files
  is. The registry/contract/dispatch layer is clean and reusable.
- **Future-proof fix = a subprocess boundary at the 3 conflict adapters.** Each
  becomes a thin client that hands text/seed to a sidecar process (its own venv
  with torch 2.8-cu128 / numpy 1.26) and gets audio back. The main stack is never
  touched; the seam above is unchanged. Scope: ~3 adapter rewrites + 1 sidecar
  runner + IPC contract -- not a rewrite of the codebase.
- Cross-validated by the roundtable panel (Opus 4.8 + Sonnet 4.6, grounded):
  both independently flagged the in-process commitment, the wrong
  `eng_stable_audio.py` (imports `stable_audio_tools` + points at the
  non-commercial `stable-audio-open-1.0`, not native SA3), and that promotion
  needs a CODE mechanism to flip `default_roles` (the adapters hardcode `()`),
  not a workflow-widget flip.
- Separate, real follow-up (writer side, not audio): `_otr_structured_call` is the
  highest-fan-in module and the origin of the weak-LLM failures -- if model
  robustness matters, that is the lever, independent of the audio plan.
