# Polish review pass 2: verify the pass01 fixes are sound

Round 1 found real defects (stderr leaks, no read timeout, unguarded json, Dia
44100, role-blind ref resolution); all are now FOLDED and the full suite is green
(3778 passed / 0 failed) + Bug Bible green. This pass re-reviews the HARDENED code.
Be concrete: cite file + failure mode. Confirm the fixes are correct AND introduced
no new bug. Do NOT re-raise the already-known verify-at-build items.

## What changed since round 1 (all in the attached files)
- NEW `_otr_sidecar.py`: `read_protocol_line(proc, timeout, what)` (daemon reader
  thread + queue, because `select` is not usable on Windows pipes) and
  `close_worker(proc, stderr)` (idempotent: graceful stop -> kill -> wait ->
  always close stderr) + `remove_quietly`.
- `eng_chatterbox.py` / `eng_dia.py`: load()/generate_voice()/unload() now use the
  helper; Popen failure closes stderr; request path catches
  Timeout/EOF/ValueError/OSError, kills the worker, clears `_proc`/`_stderr`,
  removes the temp, raises a NAMED error; `_load_wav` removes the temp in finally.
- `_otr_dia_worker.py`: `_ensure_rate()` forces 44100; `_build_prompt` strips a
  leading speaker tag.
- `_otr_voice_node_common.py`: `_resolve_clone_ref_path(..., role=self.ROLE)`.

## Specifically pressure-test the NEW concurrency
1. `read_protocol_line` reader-thread: on TIMEOUT the thread is left blocked in
   `readline()`. Is the documented mitigation (caller kills the proc so the
   pending readline returns EOF) actually guaranteed? Any path where an orphaned
   reader thread later steals a line and desyncs the protocol on a SUBSEQUENT
   request to the SAME proc? (Note: callers clear `_proc` on timeout -> respawn.)
2. `close_worker`: double-close of stderr (unload after a generate_voice already
   closed it)? `proc.stdin.write` after the pipe is closed? Is `poll()` after
   `kill()` reliable enough to gate the stderr close?
3. After a request-path failure clears `_proc=None`, the OUTER `_render_per_line`
   `finally` calls `_teardown(adapter)` -> `adapter.unload()` -> `close_worker(None,
   None)`. Is that a clean no-op (no double-kill, no exception)?
4. Determinism unchanged: per-request `_seed_everything` only; still correct to
   leave `supports_external_generator=False`?
5. Any remaining handle/zombie leak across: Popen raises; readiness time-out;
   readiness bad-json; worker dies mid-request; normal unload; GC of a never-loaded
   adapter.
6. Dia `_ensure_rate`: torchaudio-missing fallback (numpy interp) correctness;
   does it ever run on the common 44100 path (it should early-return)?
7. The role-aware `_resolve_clone_ref_path`: does threading `self.ROLE` change the
   indextts2 char_voice behavior in any way (it must not)?

## Known verify-at-build (GPU box; do NOT re-raise)
chatterbox torch on sm_120; chatterbox external Generator; Dia audio_prompt-only
quality; Dia 0626 vs Dia2; exact library generate() signatures.
