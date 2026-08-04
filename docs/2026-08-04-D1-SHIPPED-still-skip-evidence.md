# D1 -- FINAL (post-r3). Shipped state.

Observability only. No gate weakened, no fallback revived, no behavior change in
which prompts are refused. Campaign: explicitly scoped r2 -> r3 (see
`../scope_receipt.md`; r1 and r4 deliberately not run).

## What shipped

**D1a -- the completion gate raises with structured evidence.**
`nodes/otr_image_gen_dispatcher.py`. A keyed `skip_evidence_by_oid` map is
populated at both silent-skip branches with a stable `reason`
(`no_engine`, `prompt_path_guard`). The gate emits one deterministic status per
missing target, in `required_scene_targets` order: `no_row`,
`historical_row_only` (a row exists but was not appended THIS dispatch --
`images` is seeded from prior ledger state), or `dead_path` with `path=...`.
Evidence is joined by KEY, never by substring: `still_b1` is a prefix of
`still_b12`.

**D1b -- both silent skips log at skip time, and the guard names its arm.**
`path_guard_arm()` reports `arm` / `token` / `index` / `excerpt` /
`prompt_len` / `prompt_hash`. The excerpt is centred on the match (not
`prompt[:200]`, which misses a late slash) and `repr`-escaped so a newline in an
LLM-authored prompt cannot forge a log record. The hash is the dispatcher's
CANONICAL sha256 `_prompt_content_hash`, so evidence joins the ledger's own
`prompt_hash`. The original predicate is preserved EXACTLY -- same
short-circuit precedence, no `rstrip()` -- because this change may not move the
set of refused prompts.

**Structured record on the production channel.** The canonical runner renders a
failure as `str(status["messages"])[:500]` (`scripts/otr_api.py:749`), so a rich
exception message is truncated before an operator reads it. The gate now emits
one compact JSON `log.error("[OTR_ImageGenDispatcher] MISSING_TARGET ...")` per
target BEFORE raising. That is the production contract; the
`err.missing_targets` attribute exists for the regression tests, which assert
the evidence schema rather than substring-matching a sentence.

**D1c -- the launcher stops destroying the prior run's log.**
`scripts/_otr_rotate_log.ps1` (new, bounded, independently testable without
booting ComfyUI) moves an existing log aside under a sortable, locale-
independent `yyyyMMdd_HHmmss_fff` stamp with collision suffixing.
`_otr_soak_server_launch.cmd` calls it, then truncates `%~1` ONLY on success,
writes a `LOG ROTATION FAILED` marker when it could not rotate, and APPENDS
server output with `>>`. **The caller's log path is unchanged** -- eight
harnesses read exactly `%1`. A locked log is preserved and the failure is loud;
a boot is never killed to save a log file.

**D1d -- CUT.** It could not preserve the refused prompt (that prompt
`continue`s at `:896` and never reaches row construction), and `CanonicalImage`
(`nodes/_otr_image_engines/schemas.py:73-94`) is `extra="forbid"` without a
`prompt` field, enforced by `tests/test_image_platform_c1.py:1132`.

## Proof

- `tests/test_d1_still_skip_evidence.py` -- 11 tests: guard arms and position,
  repr-escaping, predicate preservation (trailing-space `.png` still passes;
  precedence unchanged), canonical hash, the incident in miniature (gen_fn never
  reached, raise names object + branch + arm + offending text), `no_row` with no
  evidence, prefix-id isolation, and skip-time logging.
- Rotation proven standalone across four cases including a LOCKED log:
  exit 1, file preserved, evidence intact.
- Full suite + Bug Bible (17 passed / 24 skipped / 3 xfailed).
- AST parse clean; `workflows/otr_canonical.json` untouched (no node, widget or
  link change).

## Acceptance for the next live leg (revised)

A 320-word Shakespeare still leg must produce EITHER a published episode OR a
fail-closed carrying complete D1 evidence. The second outcome is not a
regression -- at an admitted ~1-in-6 rate it is the proof D1 works. Gating an
observability change on a coin flip was the original error.

## Open, deliberately NOT done here

1. **`rows_by_object` is built from `images` (historical), not `ep_rows`.** A
   stale row from a prior image revision whose file still exists can satisfy the
   gate without being minted this dispatch. Real defect; fixing it changes WHICH
   episodes fail, so it belongs to D3, not to an observability change.
2. **`OTRImageGenDispatcher` has no `IS_CHANGED`** while depending on external
   file existence. ComfyUI caching concern; separate change.
3. **Rotated-log retention** is unbounded. Follow-up.
4. **`extension_suffix` is unreachable inside `dispatch_images`** --
   `append_visual_safety_clause` appends after the prompt, so a `.png` tail is
   never the last thing the guard sees. Pinned by test so it is not
   rediscovered as a bug. It also narrows the live suspect: the incident, if it
   was this guard, was a SEPARATOR arm.

## D2 / D3

D2: reproduce at 320 words (~1 in 6); the failure now names its own branch,
prompt hash and offending text in the server log, which survives the reboot.
D3: fix the branch D2 names, at its root -- either where the separator ENTERS
the prompt, or by making the predicate detect real paths rather than any
separator. `PROD_BUG_LOG.md` gets its entry then, recording a mechanism rather
than a guess.
