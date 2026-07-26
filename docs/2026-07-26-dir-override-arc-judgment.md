# 2026-07-26 -- the `*_DIR` override arc: judgment

HEAD at entry `823b9929` (v2.0-alpha, == origin). Written for the next window;
forward-only. The panel ran BEFORE any code, at the operator's instruction, and
then again AFTER.

## The defect

`_build_graph` hands ComfyUI's `CheckpointLoaderSimple` / `CLIPLoader` a BARE
BASENAME. ComfyUI resolves that through `folder_paths`. But
`wan_shared._resolve_model_file` lets a `*_DIR` env var win on file EXISTENCE
ALONE and never consults `folder_paths`. So `OTR_LTX_8GB_CKPT_DIR` produced a
green preflight against one file and a render against another -- the receipt
describing one weight while a different one rendered. Same identity lie C-1
closed for the explicit `OTR_LTX_8GB_CKPT` path, one level up.

`*_DIR` has NEVER worked. It has only ever moved the preflight check.

## What was decided, and what was refuted

Pre-code fan-out: kibitz r3 (codex `gpt-5.6-sol` high + agy `Gemini 3.6 Flash
(High)`), one Sonnet lens, one Fable pass. UNANIMOUS on **Option A**: the token
is the authority, a disagreeing `*_DIR` is TERMINAL, framed as a deprecation
tripwire rather than a feature.

- **Option C REFUTED** -- register the folder from preflight via
  `folder_paths.add_model_folder_path`. ComfyUI ships no unregister function.
  A CHECK would have permanently mutated global process state, and every later
  engine on the same server would have inherited it.
- **Silently ignoring `*_DIR` REFUTED** -- it keeps the lie, just quieter.
- **Fixing WAN in the same chunk REFUTED, with evidence.**
  `tests/test_wan_loader_preflight.py:6-10` says in its own docstring that the
  `*_DIR` envs are its MOCK SEAM for a box with no ComfyUI runtime. Fixing WAN
  means migrating those fixtures first. The `wan_shared` change here is
  therefore ADDITIVE ONLY: `_resolve_model_file_by_token` was split out and
  `_resolve_model_file` still calls it, behaviour unchanged.

Grounding that decided it: nothing in `config/`, `scripts/` or `workflows/`
sets any `*_DIR`, and `extra_model_paths.yaml` is already live on this box
(`C:/ComfyUI-Models/`). So the tripwire fires on no shipped configuration, and
the remedy it names already works here.

## Shipped

- `wan_shared._resolve_model_file_by_token(categories, name)` -- where the
  LOADER would find a token, ignoring every `*_DIR`. Split out of
  `_resolve_model_file`, which still calls it.
- `eng_ltx_8gb._loader_token_path` -- a `*_DIR` that disagrees with the loader's
  own resolution, or that resolves nowhere by token, is now
  `MALFORMED_CONFIG`. The message confesses the build's bug ("That is this
  build's bug, not your setup") and points at `extra_model_paths.yaml`.
- The DIR verdict is reached BEFORE the explicit-path verdict, so the operator
  hears about the deprecated channel first.
- `tests/test_ltx_8gb_dir_override_tripwire.py` -- 16 tests, 9 of them controls.
  Suite 7055 -> 7071, Bible 17, canonical `9872624A` unmoved.

## Post-code QA found four real defects in the green code

Two independent lenses (Sonnet, agy) converged on the same two message defects
without seeing each other's output:

1. **The tool's own remediation text led into the new refusal.** Both T5
   `MISSING_MODEL` messages said "fix OTR_LTX_8GB_T5_NAME / OTR_LTX_8GB_T5_DIR",
   and `assert_usable`'s checkpoint message said "or set OTR_LTX_8GB_CKPT". An
   operator following either would have hit `MALFORMED_CONFIG` on the very next
   preflight. All three now name `extra_model_paths.yaml` / `models/`.
2. **The module docstring still called `*_DIR` a plain "dir override"** -- an
   operator reading the documented contract would meet an undocumented refusal.
3. **The order-pinning test was DECORATIVE** (Sonnet). It pointed both env vars
   at the SAME decoy, which makes the explicit guard's condition trivially
   false, so it would have passed under a real branch swap -- in the one test
   whose entire subject is which branch runs first. Fixed with a third distinct
   decoy, and mutation `SWAP_explicit_guard_runs_first` now proves it.
4. **`_installed()`'s bool contract was unpinned for the new branch.** The
   tripwire is a SECOND place that can raise out of `_ckpt_path()`; only the
   explicit-path guard had a test. Added.

## Mutation proof: 8 mutants, 0 control breaks

`tmp/_kbA_dir_mutate.py`. Baseline failed=0 first, so a blind harness cannot
pass silently.

| mutant | caught by |
|---|---|
| REVERT_tripwire_never_fires | the three terminal tests |
| HALF_only_catches_a_missing_token | the disagreement tests |
| SELF_AGREEING_asks_the_dir_question_twice | the disagreement tests |
| ORDER_explicit_wins_when_both_are_set | the order test |
| SWAP_explicit_guard_runs_first | the order test (post-fix) |
| INSTALLED_lets_the_dir_refusal_escape | the predicate test |
| SPLIT_broke_wan_dir_precedence | the WAN control |
| BY_TOKEN_honours_a_dir | the by-token contract control |

The last two name CONTROLS as their target: they prove the controls have teeth
rather than merely being green.

## Still open (separate tickets, deliberately not this chunk)

- `eng_wan_ti2v` / `eng_wan_i2v` carry the SAME `*_DIR` lie AND the pre-C-1
  explicit-path lie. Blocked on migrating `tests/test_wan_loader_preflight.py`
  off `*_DIR` as its mock seam.
- No test creates a real NTFS junction; `_same_file`'s junction behaviour is
  covered only by its own unit tests plus a dot-segment stand-in.
- Live-box verification that `extra_model_paths.yaml` folders come back from
  `folder_paths.get_full_path` (agy's VERIFY-AT-BUILD 3) -- true by design, not
  yet observed in-process this session.
- `_file_receipt` returning a list still survives the suite; `mtime_ns` is never
  isolated from size.
