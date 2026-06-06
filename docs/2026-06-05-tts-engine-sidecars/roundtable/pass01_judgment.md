# Roundtable pass01 judgment (Claude = sole judge, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro. Spend ~$0.25. All four
returned VERDICT: not build-ready -- the SHAPE is right (Path-B sidecar + adapter
metadata) but several seams were underspecified against the real code. Every claim
below was checked against the actual files; MISREADs discarded.

## CONFIRMED -> folded into pass01_plan.md
- **Refactor has TWO sites, not one.** `_render_per_line` tests
  `_OTR_CLONE_ENGINES` twice: (1) the resolve-ref block, (2) the bark-fallback
  block, which ALSO gates on `self.ROLE == "char_voice"`. Both must be swapped to
  `getattr(adapter,"requires_voice_ref",False)` AND the char_voice guard kept.
  (GPT, Grok, DeepSeek -- grounded in the excerpt I read.)
- **Add the 3 metadata attrs to `AudioEngineAdapter` (base.py) with defaults**
  (requires_voice_ref=False, voice_ref_kind=None, missing_ref_fallback=None).
  Duck-typed legacy adapters (bark/kokoro) read the getattr default -> correct.
  (DeepSeek, Grok, GPT.)
- **Guard `get_engine(None)`**: skip the fallback when missing_ref_fallback is
  falsy; only call get_engine on a truthy name. (GPT, Grok, DeepSeek, Gemini.)
- **Seed INSIDE the workers.** Main-process `deterministic_inference` cannot seed
  a subprocess; copy `_seed_everything(seed)` from the indextts2 worker into both
  new workers. (GPT -- grounded.)
- **Chatterbox sample_rate stays 24000** (adapter + profile). `_render_per_line`
  packs at the profile rate and `pack_audio_batch` raises on any clip whose rate
  differs -- so the worker must emit 24000, not a dynamic `m.sr`. (GPT, DeepSeek,
  Gemini.)
- **`supported_kwargs` belongs IN the worker** (it has the real model to
  introspect; the main-venv adapter cannot). The worker drops any kwarg the real
  `generate` signature rejects. (DeepSeek, GPT.)
- **Chatterbox worker saves via `torchaudio.save` in the SIDECAR venv** (the
  README's documented path; sidecar torch is fine), adapter loads via soundfile in
  the main venv (mirrors indextts2 `_load_wav`). Resolves the tensor-shape
  contract. (GPT.)
- **Dia transcript policy DECIDED: audio_prompt-only is the official path this
  pass.** Worker protocol gets an OPTIONAL `ref_transcript` field; the adapter
  populates it only if `config/dia_ref_transcripts.json` exists, keyed by WAV
  BASENAME (the adapter only receives `ref_clip_path`, not voice_ref_id -- Gemini,
  grounded). Mandatory faster-whisper is CUT this pass (deferred quality upgrade).
  (GPT, Grok, DeepSeek, Gemini.)
- **Engine profiles are required + chatterbox already has one.**
  `char_chatterbox_v1` exists but is `runtime: in_graph` -> flip to `oop_venv`.
  ADD `char_dia_v1`. Import `eng_dia` in the package `__init__`. (GPT, Grok --
  grounded in audio_engine_profiles.yaml.)
- **Bank mirror needs unique voice_ref_id** (cb_/dia_ prefix); the `engine` field
  already disambiguates the caster lookup, so NO lookup change is needed. Drop the
  5 dangling placeholder chatterbox rows (0 WAVs on disk). (Grok, DeepSeek.)
- **C-5 import-safety test**: importing the engines package must not import
  chatterbox/dia/sidecar torch/CUDA/spawn a worker. (GPT.)

## ACCEPTED as defensive (lighter form)
- Validate "requires_voice_ref implies voice_ref_kind" -- as a TEST, not runtime
  (keep the registry IO-free). (Grok.)
- Log the producing engine name on the bark-fallback line. (Grok.)
- Dia worker robust to a raw-tensor return (soundfile fallback) even though
  `m.save_audio` is the documented API. (DeepSeek.)

## REJECTED (MISREAD against the real code)
- "Bank rows may carry engine-specific fields (emo_vector) that break mirroring"
  -- the bank schema is uniform (voice_ref_id/engine/gender/timbre/roles/age_band/
  ref_path/ref_sha256/commercial_clean). Safe to mirror. (DeepSeek.)
- "Mirrored rows cause voice_ref_id collisions; the lookup must filter on prefix"
  -- the caster filters by `engine`; unique ids alone suffice, no lookup change.
  (Grok.)

## SCOPE decision
chatterbox keeps its existing char + announcer profiles (add ONE real on-disk
announcer ref so announcer-via-chatterbox is not dangling); Dia is char_voice ONLY
this pass (no Dia announcer). Announcer-via-clone-engine generalization is future
work. (Refines GPT #4.)

## CUTS adopted
chatterbox `--variant base|turbo` (base only); `supports_external_generator` /
`generator=` handling (keep False, defer to the GPU pilot); the warm-worker
debate (use the indextts2 lifecycle as-is); mandatory faster-whisper.

## Disagreement noted
Panel split on a dedicated mirror script vs hand-edit. I keep a lean idempotent
`scripts/_otr_mirror_clone_refs.py` (with --dry-run): mirroring 72 rows by hand is
error-prone, and "re-mirror after adding an indextts2 ref" is a recurring need.

## Convergence
One pass. All four converged on the same grounded seam-fixes (no conflicting
must-fixes); the only remaining unknowns are EMPIRICAL verify-at-build items that a
second panel pass cannot resolve (only the GPU pilot can). Not looping.

## Verify-at-build (operator GPU pilot)
1. chatterbox-tts pinned torch on Blackwell sm_120 -- runs, or needs a cu128
   override in its isolated venv?
2. chatterbox `generate()` external `torch.Generator` binding (bit_exact G1).
3. Dia audio_prompt-only clone quality (decides whether transcripts are added).
4. Dia 1.6B-0626 vs Dia2 (2025-11-19) -- target 0626 now; evaluate Dia2 later.
5. 16 GB VRAM headroom: one clone worker + later HuMo video.
