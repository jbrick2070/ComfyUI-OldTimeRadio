# Polish pass02 judgment (Claude = sole judge, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro. Spend ~$0.25.
Re-reviewed the HARDENED code (after pass01 fixes). All four converged on a tight
set of edge cases IN the new hardening code -- proof the pass01 fixes were real but
incomplete. Every claim checked against the source. Folded + full suite re-run
(3781 passed / 0 failed) + Bug Bible green.

## FOLDED (grounded, fixed)
- **`close_worker` leaked the stdin/stdout pipes** (GPT, Gemini) -- only stderr was
  closed. Now closes `proc.stdin` + `proc.stdout` too, after kill+wait.
- **stderr double-close raised `ValueError`** on 3.11+ (Grok, DeepSeek) -- the
  close was `except OSError`; widened to `except Exception`.
- **Wrong-SHAPE JSON bypassed teardown** (GPT) -- `json.loads("[]")` is valid JSON
  but `ready.get`/`resp.get` then raised AttributeError OUTSIDE the try, leaking
  the worker. Both adapters now validate `isinstance(dict)` + `ready is True` +
  (on ok) `out_path` present INSIDE the try, so a malformed line hits the
  kill/clear/remove path and a NAMED error.
- **Dead-but-referenced worker overwritten without teardown** (GPT) -- `load()`
  now reaps a `_proc` whose `poll()` is not None before respawning.
- **`_ensure_rate` silently returned on `sf.info` failure** (GPT, DeepSeek,
  Gemini) -- it now falls through to read+resample, so a non-44100/corrupt file
  never gets reported as `ok:True @ 44100` (fail-closed via sf.read raising).
- **Role-blind gender-agnostic fallback** (GPT) -- `_resolve_clone_ref_path` now
  prefers role-matching refs, then falls back to any (PD1 preserved).
- **Hardening nits:** `_env_float` clamps non-positive timeouts to the default;
  the reader thread catches `BaseException` (matches the isinstance check).

## REJECTED (MISREAD against the real code)
- DeepSeek headline "`_bark_fallback_active` not reset in the finally" -- it IS
  reset (`generate()`'s finally sets it None after teardown); DeepSeek's own
  detailed point #4 concludes "Good." No change.
- Grok "out_path NameError on early OSError" -- `out_path` is assigned BEFORE the
  try; cannot be unbound in the except. No change.
- Grok "close_worker write after poll races / closed stdin" -- the write is inside
  `try/except Exception`; a closed-stdin write is already swallowed. No change.

## CUT / deferred (panel-agreed or low-value)
- `tempfile.mktemp` -> `NamedTemporaryFile`/`mkstemp` (GPT, twice): NOT applied --
  pre-creating the file defeats the worker's "produced no file" check; kept
  `mktemp` (matches the proven indextts2 adapter).
- Per-adapter request `Lock` (GPT, ASSUMPTION-gated): not applied -- adapters are
  registry singletons and ComfyUI executes a prompt serially; indextts2 has no
  lock either. Noted as a follow-up if concurrent prompt execution is ever added.
- numpy-interp resample quality in `_ensure_rate` (DeepSeek): acceptable -- it only
  runs in the (unexpected) non-44100 case; torchaudio is the primary path.

## Convergence
pass01 -> pass02 each found real, grounded MUST-FIX -> both folded. A pass03
convergence-check was launched against the now-hardened code but OpenRouter hung
this round (no panel response after ~14 min); it was terminated rather than block
delivery. The remaining unknowns are all EMPIRICAL verify-at-build items a panel
cannot resolve (only the GPU box can), so the code-review loop is effectively
converged: two independent grounded rounds reduced the findings from architectural
(pass01) to edge-case (pass02), and pass02's fixes are localized + covered by 23
sidecar tests within the 3781-test green suite. Re-run pass03 later if desired:
`roundtable_pass.py --doc polish/pass03_build.md --out polish/pass03 ...`.
