<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

**VERDICT**: yes-with-fixes. The sidecar isolation and fallback routing are solid, but both workers leak their stderr file handle if they fail to start or crash mid-request.

**MUST-FIX BEFORE BUILD**:
1. **[eng_chatterbox.py / eng_dia.py]** Teardown file handle leak. In `unload()`, the early return `if proc is None: return` skips the `finally` block that closes `self._stderr`. If `load()` fails (raising an exception before `self._proc` is set) or if the worker crashes (setting `self._proc = None`), the `_stderr` file handle is leaked permanently.
   **Fix**: Move the `if proc is None` check inside the logic so `_stderr` is always closed:
   ```python
       def unload(self):
           proc, self._proc = self._proc, None
           try:
               if proc is not None and proc.poll() is None:
                   proc.stdin.write(json.dumps({"stop": True}) + "\n")
                   proc.stdin.flush()
                   proc.wait(timeout=10)
           except Exception:
               pass
           finally:
               if proc is not None and proc.poll() is None:
                   try:
                       proc.kill()
                   except OSError:
                       pass
               sd = getattr(self, "_stderr", None)
               if sd is not None:
                   try:
                       sd.close()
                   except OSError:
                       pass
                   self._stderr = None
   ```

**SHOULD-FIX**:
1. **[eng_chatterbox.py / eng_dia.py]** Lost zombie reference. In `generate_voice`, if `readline()` returns empty (worker crashed), the adapter does `self._proc = None` before raising `RuntimeError`. This loses the `Popen` reference, meaning `unload()` (called in the `finally` block of `generate`) cannot wait on or kill the zombie process, leaving it to Python's garbage collector.
   **Fix**: Remove `self._proc = None` from the `if not resp_line:` block in `generate