<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: build-ready as-is? no. Hardened paths still have deterministic handle leaks and malformed-protocol cases that bypass teardown/named errors.

MUST-FIX BEFORE BUILD:
1. [eng_chatterbox.py:ChatterboxEngine.load / eng_dia.py:DiaEngine.load] Readiness JSON is only guarded for invalid JSON syntax, not valid JSON with the wrong shape. `json.loads("[]")` succeeds, then `ready.get("ready")` raises `AttributeError` outside the `except`, so `_SC.close_worker(proc, stderr)` is not called and the worker/stderr handle can leak. Concrete fix: after `json.loads`, validate `isinstance(ready, dict)` inside the protected block; raise `ValueError` on non-dict. Also require `ready.get("ready") is True` rather than truthiness.

2. [eng_chatterbox.py:ChatterboxEngine.generate_voice / eng_dia.py:DiaEngine.generate_voice] Response JSON has the same shape hole. `resp = json.loads(resp_line)` is inside the `try`, but `resp.get("ok")`, `resp["out_path"]`, and `_load_wav(...)` are outside. A valid non-dict JSON response, or `{"ok": true}` without `out_path`, bypasses the request-failure cleanup path: no worker kill/clear, no temp removal, no named error. Concrete fix: validate `resp` is a dict, `resp.get("ok") is True`, and required success fields exist inside the same `try`; on any protocol-shape failure, run the existing close/clear/remove path and raise the named `RuntimeError`.

3. [eng_chatterbox.py:ChatterboxEngine.load / eng_dia.py:DiaEngine.load] A dead-but-still-referenced worker is overwritten without teardown. Current top-of-load check returns only when `self._proc.poll() is None`; if `_proc` exists but has exited while idle, the method opens a new stderr file and assigns a new `Popen` without closing the old `_stderr` or old stdin/stdout pipe objects. Concrete fix: at the start of `load()`, if `self._proc is not None` and `self._proc.poll() is not None`, call `_SC.close_worker(self._proc, self._stderr)` and clear both fields before starting a new worker.

4. [_otr_sidecar.py:close_worker] The helper claims to prevent handle leaks, but it only closes `stderr_handle`. For every successful `Popen`, stdin and stdout are also pipe file objects. They are never explicitly closed on normal unload, readiness failure, timeout, EOF, or mid-request crash. Concrete fix: after graceful/kill/wait handling, best-effort close `proc.stdin` and `proc.stdout` as well as `stderr_handle`, guarding each close and preserving the “never raises” contract.

5. [_otr_voice_node_common.py:_resolve_clone_ref_path] The role-aware fix is incomplete. The `role` argument is only passed to `assign_voice_for_slot`; the explicit `voice_ref_id` lookup still filters only `voice_ref_id` + `engine`, and the gender-agnostic fallback `cands = ... if e.engine == engine` is still role-blind. [ASSUMPTION] If voice-bank entries are role-scoped, this can still select a char ref for an announcer role or vice versa. Concrete fix: filter those two branches by role as well, or delegate both paths to the same bank API that understands role. Preserve the current char_voice behavior by defaulting role to `"char_voice"`.

SHOULD-FIX:
1. [_otr_sidecar.py:_env_float/startup_timeout/request_timeout] Negative or zero timeout env values are accepted. `queue.get(timeout=<negative>)` raises `ValueError`, which is not one of the named timeout/EOF protocol failures the callers expect. Concrete fix: clamp configured timeouts to a positive minimum, or raise a clear configuration `RuntimeError` before use.

2. [_otr_sidecar.py:read_protocol_line + eng_chatterbox.py/eng_dia.py request path] There is no serialization around write/read pairs on a shared adapter process. `_render_per_line` calls are sequential in the shown code, but if `get_engine()` returns a shared adapter across concurrent node executions [ASSUMPTION; verify registry lifecycle], two threads can write requests and start independent reader threads against the same stdout, racing responses and desynchronizing the protocol. Concrete fix if adapters can be shared: add a per-adapter `threading.Lock` covering stdin write/flush through response parse, or ensure `get_engine()` returns an instance that is never shared concurrently.

3. [_otr_dia_worker.py:_ensure_rate] On `sf.info(out_path)` failure, `_ensure_rate` returns silently but the worker still emits `"sample_rate": 44100`. That weakens the “guarantee exactly 44100” contract. Concrete fix: if `sf.info` fails, either attempt full `sf.read`/rewrite anyway or emit `{"ok": false, "error": ...}` so the adapter fails closed instead of declaring a rate it did not verify.

4. [eng_chatterbox.py:generate_voice / eng_dia.py:generate_voice] `tempfile.mktemp()` is race-prone and creates a path without reserving it. Concrete fix: use `tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=...)`, close it immediately, pass its name to the worker, and keep the existing remove-in-finally behavior.

OPTIONAL / NICE-TO-HAVE:
- The timeout orphan-reader mitigation is sound for the shown timeout paths: after `read_protocol_line` timeout, both adapters call `close_worker`, clear `_proc/_stderr`, and therefore do not intentionally reuse that same process for a later request.
- `close_worker(None, None)` is a clean no-op in the shown helper, so outer `_teardown(adapter)` after a request-path failure will not double-kill or double-close through adapter state once the adapter has cleared its fields.
- Dia `_ensure_rate` does early-return on the common 44100 path before importing torch/torchaudio or doing interpolation.

CUT THESE (over-engineering):
1. None recommended. The sidecar helper abstraction is justified by duplicated lifecycle risk across both new adapters; the numpy fallback in Dia resampling is also justified because it is the only non-torchaudio fallback shown.