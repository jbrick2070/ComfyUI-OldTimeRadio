<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? **yes-with-fixes**  
Two defects break invariants (resource leak + PD1 gap). They are not gated on GPU-side unknowns; they will manifest on any box.

MUST-FIX BEFORE BUILD

1. **[eng_chatterbox.py + eng_dia.py] Stderr file-handle leak on worker start failure / unexpected death**  
   In `load()`, `self._stderr` is opened before `Popen`. If `Popen` raises or the readiness check fails, `_stderr` is never closed.  
   In `generate_voice()`, if the worker dies (empty response), `self._proc` is set to `None` but `_stderr` is left open.  
   Both paths leak an open file handle per failed start or crash, eventually exhausting descriptors on repeated failures.  
   **Fix**: In `load()`, wrap the spawn + readiness block in a `try/finally` that closes `self._stderr` on error and sets it to `None`.  
   In `generate_voice()`, after detecting dead worker and before raising, close `self._stderr` if not None and set `self._stderr = None`.  
   (Apply ident