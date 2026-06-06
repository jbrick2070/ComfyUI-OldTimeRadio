<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: build-ready as-is? no. The plan has unresolved policy/API choices and contradictions that will break dispatch, model loading, or claimed fail-closed behavior.

MUST-FIX BEFORE BUILD:
1. [Part B / Part C / Part E / Part F / C-7] Model loading policy contradicts itself. Part B says Chatterbox uses HF “auto-download on first from_pretrained”; Part C hardcodes Dia `from_pretrained("nari-labs/Dia-1.6B-0626", ...)`; Part E expects “missing venv / worker / weights -> NAMED RuntimeError”; Part F says operator performs first-run downloads. As written, render-time can hit network or silently populate cache instead of failing closed. Concrete fix: workers/adapters must take an explicit local model/cache path, run in local-files-only/offline mode where supported, preflight required files/cache before spawning or before `from_pretrained`, and raise a named RuntimeError if absent. Install scripts, not render, perform downloads.

2. [Part C / Dia voice-clone wrinkle / Open question 3] Dia transcript policy is unresolved but listed under SHIP. The plan says Dia clone requires `[S1] <ref transcript> [S1] <target>`, then proposes “audio_prompt-only” if transcript is absent, then asks the panel to pick audio_prompt-only vs bark. This is not buildable as a final spec. Concrete fix: choose one policy before implementation. If transcript is required, missing transcript must follow `missing_ref_fallback="bark"` or raise a named fail-closed error. If audio_prompt-only is allowed, define it as the official degraded path and add tests/logging for it.

3. [Part C / Dia worker protocol] The Dia request protocol omits the transcript even though the worker behavior depends on it. Part C defines worker input only implicitly around `text` and `audio_prompt=ref_clip`; “if a clone transcript is supplied prepend it” has no request field or adapter responsibility. Concrete fix: add a protocol field such as `ref_transcript` or `prompt_text`, have `eng_dia.py` resolve `config/dia_ref_transcripts.json`, send the transcript to the worker, and have the worker construct the exact Dia prompt from that field.

4. [Part A / Part B / Part C / grounded `_render_per_line`] Announcer support for clone engines is underspecified and likely broken. Chatterbox and Dia declare `roles=("char_voice","announcer_voice")`, but the existing fallback branch is guarded by `self.ROLE == "char_voice"`, and `_resolve_clone_ref_path()` hardcodes `role="char_voice"` when assigning a bank reference. Part D mirrors the 36 `char_voice` rows, not announcer rows. Concrete fix: either remove `announcer_voice` from Chatterbox/Dia for this pass, or generalize clone-ref resolution/fallback to the active role and add/reference bank rows whose roles include `announcer_voice`.

5. [Part A / grounded `_render_per_line`] The metadata refactor must preserve both existing clone behaviors, not just replace tuple membership mechanically. There are two current clone checks: one resolves missing refs from the bank; the second triggers Bark fallback only for `char_voice`. Concrete fix: implement explicit logic:
   - `requires_ref = getattr(adapter, "requires_voice_ref", False)`
   - if `requires_ref` and no ref, resolve from bank using active role/engine
   - if still no ref and `missing_ref_fallback` is set, load that fallback engine
   - if no fallback is set, raise a named fail-closed error
   - keep Bark selected path untouched.

6. [Part B / grounded `_render_per_line` + `pack_audio_batch`] Chatterbox “return `m.sr` dynamically; do not hardcode 24000” conflicts with the current pack path. `_render_per_line` fixes `sr` from `adapter.sample_rate` and then `profile.sample_rate`; `pack_audio_batch(clips, sample_rate=sr)` raises on any clip whose returned sample rate differs. Concrete fix: either verify and set Chatterbox profile/adapter sample rate to the actual constant `m.sr`, or resample Chatterbox worker outputs to the profile sample rate before packing, or change dispatch to derive `sr` from the first generated clip and require all clips match.

7. [Part B / Part C / worker design] The sidecar workers receive `seed` but the proposed bodies do not seed inside the sidecar. Main-process `deterministic_inference()` cannot affect RNGs in a subprocess. IndexTTS2’s grounded worker has `_seed_everything(seed)` for this reason. Concrete fix: copy the seed routine into both `_otr_chatterbox_worker.py` and `_otr_dia_worker.py` and call it per request before generation.

8. [Part B / Part C / env config] `OTR_CHATTERBOX_MODEL` and `OTR_DIA_MODEL` are specified but not used by the worker bodies. Chatterbox body calls `ChatterboxTTS.from_pretrained(device="cuda")`; Dia body hardcodes `"nari-labs/Dia-1.6B-0626"`. Concrete fix: either remove these env vars from the spec or implement them end-to-end as model id/cache/local path inputs, including adapter validation and worker CLI args.

9. [Part B / Part C / registry/profile hidden dependency] Adapter files alone are not sufficient. Grounded `_render_per_line` calls `require_resolver()`, `resolver.resolve_casting_plan(role=self.ROLE, engine=engine)`, `assert_token_for_profile(profile)`, and `assert_model_available(profile)` before generation. [ASSUMPTION] If profiles/imports are not already present for Dia and sidecar Chatterbox, dispatch will fail before the adapter runs. Concrete fix: add/update engine profile entries for `chatterbox` and `dia`, including role compatibility, sample rate, commercial flag, model availability checks suitable for sidecar installs, and ensure the package import path imports `eng_dia.py` so registration occurs.

10. [Part D / Part C] “Reference bank wiring (no new files, no downloads)” contradicts the proposed `config/dia_ref_transcripts.json` and `scripts/_otr_dia_transcribe_refs.py`. Concrete fix: reword the section or split it: bank mirroring uses no new audio refs/downloads, while Dia transcript generation creates a new metadata JSON file.

SHOULD-FIX:
1. [Part B / Part C] Verify and pin the real library APIs before coding workers. The plan assumes `ChatterboxTTS.from_pretrained(device="cuda")`, `generate(..., audio_prompt_path=..., exaggeration=..., cfg_weight=...)`, `Dia.from_pretrained(..., compute_dtype="float16")`, `m.generate(audio_prompt=...)`, and `m.save_audio(...)`. Concrete fix: record the verified signatures in the worker comments/tests and adapt the protocol accordingly. Mark any remaining uncertainty as build-blocking, not post-build.

2. [Part A] `missing_ref_fallback` should be validated before use. `get_engine(getattr(adapter, "missing_ref_fallback", None))` will raise a raw `KeyError` if the fallback name is wrong unless explicitly skipped/wrapped. Concrete fix: if fallback is `None`, skip; if set but unregistered, raise `EngineUnusable(... MALFORMED_CONFIG ...)` or a named RuntimeError.

3. [Part E] Add an explicit import-safety test for the rewritten Chatterbox and Dia adapters: importing `nodes._otr_audio_engines` must not import `chatterbox`, `dia`, sidecar `torch`, touch CUDA, spawn workers, or access model paths. This directly protects C-5 and “ZERO shared torch”.

4. [Part D] Define whether mirrored bank rows for Chatterbox/Dia should retain the original `roles` from IndexTTS2 rows or expand to match each engine’s declared roles. Current text says “same ... roles” but the adapters claim announcer support.

5. [Part B] Chatterbox worker output saving needs a defined tensor-to-WAV normalization/shape contract. Concrete fix: specify handling for `[T]`, `[1,T]`, `[C,T]`, CPU transfer, float32 conversion, clipping policy, and `soundfile.write` layout.

6. [Part C] Dia output sample rate is stated as 44100. Add a runtime assertion or worker-side check that saved output is actually 44100, otherwise return the actual rate and let adapter/profile handling resample or fail clearly.

7. [Part F] The operator sequence should explicitly say that environment variables must be visible to the ComfyUI process after restart and that the sidecar workers are tested from the same account/session that runs ComfyUI.

OPTIONAL / NICE-TO-HAVE:
- Add a small shared sidecar protocol helper to avoid duplicating readiness-line parsing, stop handling, JSON error replies, temp-WAV loading, and stderr-file management across IndexTTS2, Chatterbox, and Dia.
- Add docs explaining that Chatterbox outputs are PerTh-watermarked and whether that affects downstream mastering/delivery.
- Add a smoke test that intentionally corrupts a sidecar readiness line and verifies the adapter reports the stderr log path.

CUT THESE (over-engineering):
1. [Part B] Cut `--variant base|turbo` unless the adapter exposes it or the worker actually uses it. Current body ignores variant, so it only adds an untested configuration surface.

2. [Part D] Cut a new standalone `_otr_mirror_refs.py` if extending the existing ref-download/bank tool or committing the generated JSON change is enough. The goal is deterministic mirrored rows, not another script.

3. [Part C] Cut mandatory faster-whisper transcription from this pass if the chosen Dia policy is officially audio_prompt-only degraded mode. Keep transcript support as a later quality upgrade. If the chosen policy is transcript-required, then do not cut it.

4. [Part B / Part C] Cut optional `_MODEL` env vars unless they are made authoritative for local-only model loading. Half-specified env knobs are worse than fixed documented install paths.