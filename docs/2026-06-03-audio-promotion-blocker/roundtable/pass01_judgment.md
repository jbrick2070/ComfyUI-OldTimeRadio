# Roundtable pass01 -- judged synthesis (Claude as judge)

Panel: Opus 4.8, Sonnet 4.6, GPT-5.5, Gemini 3.1 Pro (4/6 valid; Grok+DeepSeek
failed on invalid `~latest` slugs). Every claim below was GROUNDED against the
real adapters/registry before acceptance.

## CONFIRMED (consensus + verified in code) -- these are real
1. **In-process adapters are the blocker.** `eng_indextts2.py`/`eng_chatterbox.py`
   import their lib inside the ComfyUI process -> cannot be promoted on torch
   2.10+cu130/numpy 2.x. (all 4 models; matches the dependency-map scan.) The
   process must not import `indextts`/`chatterbox`/their torch/their numpy at all.
2. **`eng_stable_audio.py` is the WRONG path.** It imports `stable_audio_tools`
   and defaults to `stable-audio-open-1.0` (non-commercial), named
   `stable_audio_music` -- NOT the ComfyUI-native SA3 the plan promotes. (all 4.)
3. **No promotion mechanism.** Adapters hardcode `default_roles=()` + `requires_flag`;
   flipping the default is a CODE change (flip `default_roles`, clear flag), gated
   on F -- not a workflow-widget flip. (Opus, Sonnet, GPT.)
4. **Determinism hole.** `supported_kwargs` silently drops `generator=` if the real
   signature rejects it -> renders on global RNG yet stays selectable. Bit-exact
   mode must fail closed when the generator isn't bound. (Opus, Sonnet, GPT.)
5. **Offline/C-7 risk.** `from_pretrained`/`get_pretrained_model` are network entry
   points inside `load()` at execute time; no `local_files_only`. (Opus, Sonnet, GPT, Gemini.)
6. **Teardown + naming.** `unload()` lacks `gc.collect()` (I-7); engine id is
   `stable_audio_music` vs the plan's `stable_audio_3` -- unify everywhere. (Opus, GPT.)

## Build-critical specifics the panel added (the value of asking)
- **IPC carries the SEED INT, never a torch.Generator** -- a bound generator can't
  cross a process boundary; the sidecar reconstructs it locally on its device.
  This is the missing piece that makes a sidecar byte-identical. (Gemini -- decisive.)
- **GPU sidecar may still fail on sm_120:** torch 2.8-cu128 on Blackwell risks
  PTX-JIT hangs or no support. So F must prove sm_120 execution FIRST; if it fails,
  the fallbacks are (a) a CPU-only voice sidecar as the first supported mode
  (GPT) or (b) ONNX/torch-export run under onnxruntime-gpu with no torch pin
  (Gemini). (this is the real "outside-the-box" answer to plan Question 1.)
- **Sidecar IPC contract checklist (GPT):** process lifecycle, request/response
  schema, audio transfer format, seed fields, timeout->error mapping, local
  model-path validation, stderr capture, Windows process-tree kill on timeout,
  VRAM residency policy + health-check (version/model-hash/device/dtype/det-capable).
- **SA3 must use ComfyUI's `comfy.model_management`** to load/unload (respects the
  14.5 GB ceiling), not raw `empty_cache()`. (Gemini.)
- **Missing dep/model/token -> 6-class `EngineUnusable`** (MISSING_MODEL/
  MISSING_HF_TOKEN/MALFORMED_CONFIG), never a bare RuntimeError. (GPT.)

## CUT for the first sprint (lean)
Parallel multi-worktree orchestration (serial avoids venv/cache/log races); the
R0a legacy manifest + legacy baseline (if first sprint is SA3-only or sidecar);
native-stereo machinery; the migration framework beyond reject-and-rerender;
re-render-twice-legacy after the writer refactor. (GPT, Gemini.)

## Judged direction (smallest correct first sprint)
1. **SA3-native music adapter** (`eng_stable_audio_3.py`) driving ComfyUI's
   `comfy.model_management` SA3 path; keep `eng_stable_audio.py` as the non-default
   `stable_audio_open` only. No torch/numpy conflict -> shippable headless +
   one GPU validate. This is the one promotion with no architecture change.
2. **Voice stays bark** until a sidecar is built. Then: one sidecar runner (own
   venv), the 3 voice adapters become thin IPC clients passing seed-int + text,
   F proves sm_120 (else CPU-mode or ONNX). Add the promotion mechanism
   (default_roles flip + 6-class fail-closed) at that point.

## Open verify-at-build
ComfyUI Desktop actual version (>= v0.22.0?) + native SA3 node availability;
whether cu128 PTX-JITs or runs native on sm_120; the real `infer`/`generate`
signatures (do they accept a generator?) -- all GPU/operator checks.
