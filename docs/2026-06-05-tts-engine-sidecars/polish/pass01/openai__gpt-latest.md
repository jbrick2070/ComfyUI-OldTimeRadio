<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Sidecar lifecycle can hang indefinitely and leak handles/processes; Chatterbox announcer/ref-missing path is not made safe by the metadata refactor.

MUST-FIX BEFORE BUILD:
1. [eng_chatterbox.py:ChatterboxEngine.load / eng_dia.py:DiaEngine.load] `proc.stdout.readline()` has no timeout. If the worker hangs during import/model load before emitting readiness, the Comfy render thread blocks forever. Concrete fix: read readiness with a bounded timeout while polling `proc.poll()`; on timeout kill, wait, close pipes/stderr, and raise a named `... worker startup timed out` error. Make the timeout configurable if model load can be long.

2. [eng_chatterbox.py:generate_voice / eng_dia.py:generate_voice] `self._proc.stdout.readline()` has no timeout for per-request responses. If `model.generate()` hangs or a worker stays alive but stops responding, the node blocks forever and I-7 teardown never runs. Concrete fix: wrap request/response in a bounded wait loop or reader thread/queue with timeout; on timeout kill+wait the worker, close handles, clear `_proc`, remove the temp output if present, and raise a named `... worker request timed out` error.

3. [eng_chatterbox.py:load/unload/generate_voice / eng_dia.py:load/unload/generate_voice] Process/file-handle cleanup is incomplete. Failure modes:
   - `load()` opens `self._stderr` before readiness; on bad/no readiness it kills but does not `wait()` and does not close `_stderr`.
   - `load()` starts a new process when `self._proc` is dead but does not close the old stderr handle first.
   - `generate_voice()` EOF branch sets `self._proc = None`; later `unload()` returns early and never closes `_stderr`.
   - `unload()` calls `proc.kill()` but does not `proc.wait()` after kill.
   Concrete fix: centralize `_close_proc(kill: bool)` that always closes stdin/stdout where possible, kills if needed, waits after kill, closes `_stderr`, and clears both `_proc` and `_stderr`. Use it in load failure, dead-worker replacement, EOF, timeout, and unload.

4. [_otr_voice_node_common.py:_render_per_line + eng_chatterbox.py metadata] Chatterbox declares `roles = ("char_voice", "announcer_voice")` and `requires_voice_ref = True`, but missing-ref fallback is guarded by `self.ROLE == "char_voice"`. For `announcer_voice` with no usable `voice_ref_path`, the path eventually calls `ChatterboxEngine.generate_voice(..., ref_clip_path=None, ...)`; `_otr_chatterbox_worker.py` then returns `{"ok": false, "error": "ref_clip missing: None"}`, failing the whole announcer render. Concrete fix: either remove `announcer_voice` from Chatterbox until announcer references are guaranteed, or implement an announcer-specific reference/fallback policy. Do not leave a selectable announcer engine that predictably fails when no ref is present.

5. [_otr_voice_node_common.py:_resolve_clone_ref_path] Ref resolution is hard-coded to `role="char_voice"` when assigning a fallback reference. This function is called for any adapter with `requires_voice_ref`, including Chatterbox announcer because the ref-resolution block is not role-gated. That can assign a character reference for an announcer render, or fail in a role-confusing way. Concrete fix: pass `self.ROLE` into `_resolve_clone_ref_path(...)` and use that role for `assign_voice_for_slot`, or explicitly skip this resolver for announcer engines unless an announcer ref bank exists.

6. [eng_chatterbox.py:generate_voice / eng_dia.py:generate_voice] `json.loads(resp_line)` is unguarded. If the worker protocol is ever corrupted despite fd redirection, the user gets a raw `JSONDecodeError`, not a named fail-closed engine error, and cleanup relies on outer teardown only. Concrete fix: catch `ValueError`, kill/close the worker because stream state is no longer trustworthy, and raise `Chatterbox/Dia worker protocol error: bad response line ...`.

SHOULD-FIX:
1. [eng_chatterbox.py:generate_voice / eng_dia.py:generate_voice] `tempfile.mktemp()` is race-prone and leaves stale paths on failures. Concrete fix: use `tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=...)` or `mkstemp()`, close the fd, pass the path to the worker, and remove it in all failure paths.

2. [eng_dia.py:_load_wav / _otr_dia_worker.py:_save] `_otr_dia_worker.py` always emits `sample_rate: 44100`, but when `model.save_audio(out_path, out)` succeeds, `_save()` does not verify the actual WAV sample rate. The adapter then reads the actual `sr` from soundfile and returns it, so a non-44100 `save_audio` output will later trip `pack_audio_batch(... sample_rate=44100)` with a mixed-rate error. Concrete fix: after saving, read/validate the WAV sample rate or resample/rewrite to `_SR` before emitting success.

3. [_otr_voice_node_common.py:_render_per_line / eng_chatterbox.py:ChatterboxEngine.generate_voice] Chatterbox’s delivery projection is effectively disabled: `_render_per_line` calls `adapter.generate_voice(prepared, voice_ref, None, engine_seed)`, so `ChatterboxEngine._project(delivery_vector)` always receives `None` and sends `exaggeration = 0.5`. Concrete fix: if the ledger/resolved request carries a delivery vector, pass it through; otherwise remove/comment the dead projection path so behavior is explicit. [ASSUMPTION] This matters if shipped ledgers contain per-line delivery metadata.

4. [_otr_chatterbox_worker.py:_seed_everything / _otr_dia_worker.py:_seed_everything] Worker seeding does not set deterministic backend flags inside the subprocess. Parent-side `deterministic_inference()` cannot affect the sidecar. Per-request `random`/`numpy`/`torch.manual_seed` is not enough to guarantee byte-identical CUDA output if the model uses nondeterministic kernels. Concrete fix: either set deterministic flags in the workers where supported, or explicitly mark sidecar outputs as seeded-but-not-bit-exact until the external-generator pilot lands.

5. [eng_chatterbox.py:load / eng_dia.py:load] If `subprocess.Popen(...)` itself raises after `_stderr` is opened, the file handle is leaked. Concrete fix: wrap Popen/readiness setup in try/except and close `_stderr` on every exception before re-raising a named install/startup error.

6. [eng_chatterbox.py:_load_wav / eng_dia.py:_load_wav] Output temp files are removed only after successful `sf.read`. If `sf.read` fails on a corrupt/partial WAV, the temp file remains. Concrete fix: move `os.remove(path)` into a `finally`.

7. [_otr_dia_worker.py:_build_prompt] If `ref_transcript` already contains Dia speaker tags, `_build_prompt()` produces duplicated tags like `[S1] [S1] ...`. Concrete fix: either document that transcripts must be raw text only, or normalize by stripping a leading `[S1]` from transcript/text before composing.

OPTIONAL / NICE-TO-HAVE:
- Add a request id to the JSON protocol and echo it in responses. It is not required for the current synchronous one-request-at-a-time design, but it makes future recovery from protocol desync checkable.
- Include the worker stderr path in per-request failure messages, not only startup/EOF messages.
- Log the resolved absolute ref path and fallback count at debug level for clone engines.

CUT THESE (over-engineering):
1. [base.py:supported_kwargs] There are now two similar helpers: `base.supported_kwargs` and `_otr_chatterbox_worker.py:_supported_kwargs`. If `base.supported_kwargs` is not used by grounded adapters, cut or consolidate it to avoid maintaining duplicate signature-filtering logic. [ASSUMPTION] Verify no other file imports `supported_kwargs` before cutting.

2. [eng_chatterbox.py / eng_dia.py `verbose` request field] Both adapters send `"verbose": False`, but neither worker reads it. Safe to cut from the protocol unless an unshown worker/library path consumes it. [ASSUMPTION] Grounded workers do not use it.

3. [eng_dia.py:_model_id env override in adapter plus no UI widget] Keeping `OTR_DIA_MODEL` is fine for dev, but if the release invariant is a fixed Dia model and no model widget, this override is an extra support matrix. Safe to cut for the build unless GPU-box validation explicitly requires model switching.