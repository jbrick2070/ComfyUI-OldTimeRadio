# Polish review: the BUILT chatterbox + Dia sidecars + metadata refactor

This is a CODE review, not a design review. The build below is DONE and the full
test suite is green (3771 passed / 0 failed) + Bug Bible green. Review the ACTUAL
grounding files for real bugs. Be concrete: cite the file + the failure mode.

## What shipped (all grounded in the attached files)
- **base.py**: `AudioEngineAdapter` gained `requires_voice_ref=False`,
  `voice_ref_kind=None`, `missing_ref_fallback=None`.
- **_otr_voice_node_common.py**: deleted the `_OTR_CLONE_ENGINES` tuple; added
  `_engine_requires_voice_ref(adapter)` + `_engine_missing_ref_fallback(adapter)`;
  `_render_per_line` now branches on `requires_voice_ref` (both the ref-resolve
  block AND the bark-fallback block) and resolves the fallback engine from
  `missing_ref_fallback`, keeping the `self.ROLE == "char_voice"` guard and the
  `get_engine(None)`-skip guard.
- **eng_chatterbox.py**: rewritten from an in-process import to a Popen sidecar
  (mirrors eng_indextts2.py). roles char+announcer, flag OTR_ENABLE_CHATTERBOX,
  sample_rate FIXED 24000, requires_voice_ref/kind/fallback set. Sends
  exaggeration/cfg_weight/temperature; loads the WAV via soundfile.
- **eng_dia.py**: new Popen sidecar. char_voice only, flag OTR_ENABLE_DIA,
  sample_rate 44100, optional transcript resolved from
  `config/dia_ref_transcripts.json` keyed by WAV basename.
- **_otr_chatterbox_worker.py / _otr_dia_worker.py**: indextts2 fd1->fd2 dance;
  `_seed_everything` per request; readiness line; chatterbox uses
  `_supported_kwargs` + `torchaudio.save` + resample-to-24000; dia builds
  `[S1] <transcript?> [S1] <text>`, `model.generate(prompt, audio_prompt=ref)`,
  `save_audio` with a soundfile fallback.

## Invariants to guard (reject any "fix" that breaks one)
C-5 import-time clean (no chatterbox/dia/torch import at package import); C-7
fail-closed named errors; PD1 always-renders (bark fallback); byte-identical bark
+ indextts2 paths; model-agnostic dispatch (no engine-name branches); ZERO shared
torch (sidecar venvs); 16 GB VRAM with I-7 teardown-in-finally; UTF-8/no-BOM/ASCII.

## Specifically hunt for
1. Subprocess protocol RACES / deadlocks: a worker that emits a non-protocol line
   on the JSON channel; readline() blocking forever if the worker dies mid-request;
   stdout/stderr buffering; a crash leaving the worker resident (VRAM leak).
2. Determinism: is per-request `_seed_everything` enough for byte-identical
   render-twice given no external generator? Any global-state leak across lines?
3. ComfyUI quirks: the engine combo / widget serialization; does anything here
   touch INPUT_TYPES or add a forbidden widget (only engine + stereo_policy may
   serialize; no seed/model_id widget)?
4. The refactor: did removing the tuple change behavior for indextts2 or bark on
   ANY path (announcer vs char, ref present vs absent, fallback engine None)?
5. Chatterbox: sample_rate must stay 24000 for pack_audio_batch -- is the worker's
   resample correct? PerTh watermark interactions?
6. Dia: `[S1]` prompt construction; `save_audio` vs raw-tensor; 44100 vs 22050 ref
   clips; audio_prompt-only quality; the basename transcript lookup.
7. Teardown: unload() across both workers -- any path that leaks the subprocess or
   the stderr file handle? The finally-unload of the bark fallback.

## Already known (do NOT re-raise as new) -- verify-at-build on the GPU box
chatterbox torch on sm_120; chatterbox external-Generator (bit_exact, deferred);
Dia audio_prompt-only quality; Dia 0626 vs Dia2; exact library `generate()`
signatures (workers use supported_kwargs / documented calls). These need the GPU
box, not a code reading.
