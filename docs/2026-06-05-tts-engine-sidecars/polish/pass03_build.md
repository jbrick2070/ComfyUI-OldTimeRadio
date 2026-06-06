# Polish review pass 3: convergence check

Rounds 1 + 2 found and FIXED real defects. Round-2 fixes are now folded; full
suite green (3781 passed / 0 failed) + Bug Bible green. This pass is a convergence
check: confirm the round-2 fixes are correct and find any REMAINING material
(MUST-FIX) bug. If you only find style/nits, say so plainly. Do NOT re-raise the
known verify-at-build items (GPU-box only).

## Round-2 fixes now in the attached files
- `_otr_sidecar.close_worker`: now also closes `proc.stdin` + `proc.stdout` and
  the stderr close is `except Exception` (tolerates double-close ValueError on
  3.11+). `_env_float` clamps non-positive timeouts to the default.
  `read_protocol_line` reader catches `BaseException`.
- `eng_chatterbox.py` / `eng_dia.py` `load()`: reaps a dead-but-referenced worker
  (poll() not None) before respawn; validates readiness is a dict + `ready is
  True` INSIDE the try (wrong-shape JSON now hits teardown, not an uncaught
  AttributeError).
- `eng_chatterbox.py` / `eng_dia.py` `generate_voice()`: validates the response is
  a dict and (when ok) has `out_path` INSIDE the try, so a malformed response runs
  the kill/clear/remove path and raises a named error.
- `_otr_dia_worker._ensure_rate`: `sf.info` failure now falls through to
  read+resample (or raises in sf.read -> ok:false) instead of silently reporting
  44100.
- `_otr_voice_node_common._resolve_clone_ref_path`: the gender-agnostic last
  resort now prefers role-matching refs, then falls back to any (PD1).

## Invariants (reject any "fix" that breaks one)
C-5 import clean; C-7 fail-closed named errors; PD1 always-renders; byte-identical
bark + indextts2; model-agnostic dispatch; ZERO shared torch; I-7 teardown;
UTF-8/no-BOM/ASCII.

## Known verify-at-build (GPU box; do NOT re-raise)
chatterbox torch on sm_120; chatterbox external Generator; Dia audio_prompt-only
quality; Dia 0626 vs Dia2; exact library generate() signatures.
