# TTS-engine sidecars -- hardened build spec (pass01, grounded)

Supersedes pass00 where they differ. Every item is grounded against the real
files; see pass01_judgment.md for the accept/reject log. Invariants from pass00
still hold (C-5 / C-7 / PD1 / byte-identical bark / model-agnostic dispatch / zero
shared torch / 16 GB / UTF-8-no-BOM).

## A. Foundation refactor (base.py + _otr_voice_node_common.py)
- base.py `AudioEngineAdapter`: add `requires_voice_ref=False`,
  `voice_ref_kind=None`, `missing_ref_fallback=None`.
- Clone adapters (indextts2, chatterbox, dia): set `requires_voice_ref=True`,
  `voice_ref_kind="wav_path"`, `missing_ref_fallback="bark"`.
- `_render_per_line`: replace BOTH `engine in _OTR_CLONE_ENGINES` tests with
  `getattr(adapter,"requires_voice_ref",False)`. KEEP the `self.ROLE ==
  "char_voice"` guard on the bark-fallback block. Resolve the fallback engine via
  `fb=getattr(adapter,"missing_ref_fallback",None)`; only `get_engine(fb)` when fb
  is truthy (no `get_engine(None)`). Log the producing engine on the fallback line.
- Delete the `_OTR_CLONE_ENGINES` tuple. (Referenced only in this file; no test
  depends on it.)

## B. chatterbox Path-B sidecar (MIT, char + announcer)
- Rewrite `eng_chatterbox.py` to the indextts2 Popen template (lifecycle: spawn in
  load(), reuse across lines, teardown in unload()/finally). env
  OTR_CHATTERBOX_VENV / _WORKER. `commercial_clean=True`,
  `requires_flag="OTR_ENABLE_CHATTERBOX"`, `interface="per_line"`,
  `sample_rate=24000` (FIXED -- must match the profile so pack_audio_batch stays
  single-rate). Adapter computes `exaggeration` from the delivery vector
  client-side (keep `_project`) and sends it in the request; adapter `_load_wav`
  via soundfile (main venv).
- `scripts/_otr_chatterbox_worker.py`: indextts2 fd-dance; `_seed_everything`;
  readiness line `{"ready":true}`; request `{text, ref_clip, exaggeration,
  cfg_weight, temperature, seed, out_path, verbose}`; body
  `ChatterboxTTS.from_pretrained(device="cuda")` +
  `supported_kwargs(m.generate, audio_prompt_path=ref, exaggeration, cfg_weight,
  temperature)`; save via `torchaudio.save(out, wav, m.sr)` (sidecar torch); reply
  `{ok, out_path, sample_rate:24000}` (assert/resample if m.sr!=24000).
- Profile: flip `char_chatterbox_v1` (and `announcer_chatterbox_v1`) `runtime`
  in_graph -> oop_venv. PerTh watermark documented.

## C. Dia Path-B sidecar (Apache-2.0, char_voice ONLY)
- New `eng_dia.py` on the same template. env OTR_DIA_VENV / _WORKER (+ optional
  _MODEL id, default nari-labs/Dia-1.6B-0626). `commercial_clean=True`,
  `requires_flag="OTR_ENABLE_DIA"`, `roles=("char_voice",)`,
  `interface="per_line"`, `sample_rate=44100`. Adapter resolves an OPTIONAL
  transcript from `config/dia_ref_transcripts.json` keyed by
  `os.path.basename(ref_clip_path)`; sends it as `ref_transcript` (may be "").
- `scripts/_otr_dia_worker.py`: fd-dance; `_seed_everything`; readiness line;
  request `{text, ref_clip, ref_transcript, seed, out_path, verbose}`; body
  `Dia.from_pretrained(model_id, compute_dtype="float16")`; build
  `[S1] <ref_transcript> [S1] <text>` when transcript given else `[S1] <text>`;
  `out=m.generate(prompt, audio_prompt=ref_clip)`; save via `m.save_audio(out_path,
  out)` (fallback `soundfile.write(out_path,out,44100)` if out is a raw array);
  reply `{ok,out_path,sample_rate:44100}`. Official clone path = audio_prompt-only;
  transcript optional (faster-whisper deferred).
- ADD profile `char_dia_v1` (role char_voice, engine dia, commercial_clean=true,
  allowed_voice_banks:[default], sample_rate:44100, runtime:oop_venv,
  needs_ref_clip:true, rank:2, is_default:false, requires_hf_token:false). Add dia
  to `_LEGACY_FIRST_ENGINES["char_voice"]`. Import eng_dia in package __init__.

## D. Reference bank wiring (`scripts/_otr_mirror_clone_refs.py`, idempotent)
- Drop the 5 dangling placeholder chatterbox rows (0 WAVs on disk). Mirror the 36
  indextts2 char_voice rows to engine=chatterbox (cb_*) and engine=dia (dia_*):
  copy ref_path/ref_sha256/gender/timbre/roles/age_band/commercial_clean, new
  unique voice_ref_id. Add ONE chatterbox announcer row (cb_announcer_male ->
  vz_bill_boerst.wav) so announcer-via-chatterbox is not dangling. Bank JSON
  hot-reloads. Effective commercial_clean = engine AND ref (engine on the adapter;
  ref CC0=true) -- chatterbox/dia + CC0 = clean; indextts2 + CC0 = NOT (MUST-FIX #2,
  separate).

## E. ComfyUI quirks / widget / IO (verify, don't regress)
- chatterbox/dia are ENGINES, not new nodes -> they appear in the existing voice
  node's `engine` combo (built from `legacy_first_engines("char_voice")`). Confirm
  the combo lists indextts2, chatterbox, dia, bark; index 0 stays indextts2
  (byte-identical default). No new widgets; opt-in via requires_flag at queue time.
- Confirm INPUT_TYPES/RETURN unchanged; the only serialized widgets stay `engine`
  + `stereo_policy` (no seed/model_id widget -- CLAUDE.md rule 6). Fail-closed at
  queue time when flag unset (EngineUnusable named error).

## F. Tests (full suite was 3758/0; + Bug Bible/core/dropdown)
registry registration+roles+flag-gating; load() fail-closed (missing venv/worker);
refactor metadata replaces tuple + bark fallback still fires for a ref-less clone
char engine + non-clone engines never fall back; worker JSON contract via a stub
(no GPU); bank mirror loads/validates + caster assigns + no dangling on-disk ref;
C-5 import-safety (no chatterbox/dia/torch import on package import);
"requires_voice_ref implies voice_ref_kind" duck-type check.

## G. Operator-gated (heavy; RESTART): the two isolated-venv installs + first-run
model download + set USER env + RESTART ComfyUI + (optional) Dia ref transcription
+ set node 81 engine + node 80 voice_bank=default + live smoke render + confirm
main venv untouched. Docs: chatterbox_pathb_setup.md, dia_pathb_setup.md.

## Verify-at-build (GPU pilot): chatterbox torch on sm_120; chatterbox external
Generator (bit_exact); Dia audio_prompt-only quality; Dia 0626 vs Dia2; VRAM with
HuMo.
