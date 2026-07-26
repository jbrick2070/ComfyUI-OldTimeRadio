# POST-CODE QA FINDINGS -- B1a / B2a / B2b (already pushed)

Fan-out run 2026-07-26 against HEAD `7d7e9a4e`, on commits `8caf3516`,
`55c8a811`, `582dfbd8`. Panel: codex `gpt-5.6-sol` (high) + agy
`Gemini 3.6 Flash (High)` via kibitz, plus FOUR Sonnet lenses (run_graph
lifecycle / the path guard / identity stability / test quality). All verified
against the real Windows files.

**This QA should have run BEFORE each push and did not. The operator caught the
omission. It found five code defects and six test defects in code that was
green, mutation-proven with controls, and already on origin.** That is the sixth
time a panel has found real defects in already-green code in this project.

---

## CODE DEFECTS -- ranked by production impact

### C-1 CONFIRMED, LIVE FALSE REFUSAL -- the path guard rejects correct configs
`eng_ltx_8gb.py:252-253`. `os.path.abspath` normalises separators but does NOT
fold case and does NOT resolve junctions/symlinks. Windows/NTFS is
case-insensitive, and **this box reaches the repo through a live junction**
(`C:\Users\jeffr\ComfyUI-Installs\...` -> the repo, `LinkType: Junction`,
recorded at `GO_FORWARD_PLAN.md:266`).

Failing input, reproduced live by the lens:
`OTR_LTX_8GB_CKPT=C:\Models\ltxv-2b-0.9.8-distilled.safetensors` while the token
resolves to `c:\models\...` -- the SAME file -- raises `MALFORMED_CONFIG` on
every multi-segment `ltx_8gb` beat, forever. Same for either spelling of a
junctioned model dir, and `extra_model_paths.yaml` (a shared/NAS store) is the
standard way that arises.

**Found independently by all four sources** (codex MUST-FIX 4, agy MUST-FIX 1,
Sonnet lens B, and lens D via the weak test). Fix: `os.path.samefile()` in a
try/except (it raises when either path is missing), falling back to
`os.path.normcase(os.path.realpath(...))` comparison.

**Why it shipped: the control test that should have caught it is misnamed.**
`test_the_override_guard_is_case_and_separator_tolerant`
(`tests/test_ltx_8gb_session_config.py:122`) only swaps `os.sep` for `/` --
which `abspath` DOES normalise. It never varies case despite its name.

### C-2 CONFIRMED -- `terminal` is validated against `results`, not `graph`
`wrapper_bridge.py:413-415`. Because `results` is now seeded with the externals,
an id that exists ONLY in `external_results` can masquerade as the terminal: a
graph whose real terminal node is missing or mistyped returns SUCCESSFULLY with
the caller's own handle tuple instead of failing closed. (codex MUST-FIX 5;
Sonnet lens A traced it and rated it cosmetic -- codex's read is the correct one,
because it converts a caller typo into a silent wrong return.) Fix: validate
`terminal` against `graph` membership, not `results`.

### C-3 CONFIRMED -- a raising identity read STRANDS the GPU lease for the life of the server
`beat_session.py:169` + `motion_common.py:423-478`. `open()` reads the identity a
second time AFTER `prepare()` has already acquired the cross-process heavy-engine
lease. `session_identity()` now does file I/O and can raise (that is new -- B2b
made it so). A raise there propagates out of `__enter__`, and **when `__enter__`
raises, `__exit__` is never called** -- so `close()`, `teardown()`, and the
`finally: _GR.release(lease)` never run. The lease owner is the live ComfyUI
process, so the stale-lock reclaim (`gpu_residency.py:65`, PID-liveness based)
never fires either. Every later heavy render blocks its full timeout and raises
`LeaseTimeout`. Recovery is killing ComfyUI or deleting the lock dir by hand.

None of the 38 tests in `test_beat_session.py` construct an engine whose identity
succeeds once and raises on the next call. (Sonnet lens C.)

### C-4 CONFIRMED -- `_file_receipt`'s `os.stat` is unguarded
`eng_ltx_8gb.py:109`. A file vanishing between resolution and stat raises a raw
`FileNotFoundError` instead of the named `EngineUnusable(MISSING_MODEL)` used
everywhere else in the same function. `assert_usable` two functions away already
guards the equivalent call (`:397-400`). (codex SHOULD-FIX 4, lens B, lens C.)

### C-5 CONFIRMED -- `_resolve_value` violates its own fail-closed contract
`wrapper_bridge.py:157-164`. `out = results[val.src]` sits OUTSIDE the try, so a
missing source raises a bare `KeyError`, not the NAMED `GraphExecutionError` the
module docstring promises. Currently unreachable through `run_graph` (the topo
check catches it first), but it is exactly the "segment 1 dies on output slot
unavailable" class this arc already had to fix once -- a future regression would
lose its diagnostic. (Sonnet lens A.)

---

## THE STRUCTURAL ONE -- two resolvers over one fact, and the fix protects the wrong path

**CONFIRMED by codex MUST-FIX 1, lens B finding 3/4, lens C finding 5, and lens D
gap A -- four independent reads, and lens D PROVED it live.**

`resolve_session_config()` is reachable ONLY from `session_identity()`, which
`BeatSession` calls ONLY `if self.is_multi_segment`. The ordinary single-clip
path (`render_driver.py:2782`, no `segment_count=`) never runs it. That path is
still gated by `assert_usable()`, which still uses the OLD `_ckpt_path()` --
existence + size floor, no token cross-check.

Lens D reproduced it: with `OTR_LTX_8GB_CKPT` pointed at a decoy that clears the
4 GiB floor, `assert_usable()` returned green while `resolve_session_config()`
raised, same environment. **So the identity-lie bug B2a was written to close is
still fully open on the single-clip path -- the common case.** Commit `55c8a811`
was purely additive (271 insertions, 0 deletions); `_ckpt_path`, `_installed`
and `assert_usable` were never touched.

codex's CUT 1: retire `_ckpt_path()`/`_t5_path()` as competing authorities and
route `assert_usable` through the same resolver. That is the actual fix, and it
is bigger than the guard itself.

---

## TEST DEFECTS (from the Sonnet test-quality lens, mutation-verified)

| # | Test | Verdict |
|---|---|---|
| T-1 | `test_the_override_guard_is_case_and_separator_tolerant` | **WEAK, and it is why C-1 shipped** -- name promises case, body only varies separator; final assert is truthy-only |
| T-2 | `test_CONTROL_free_after_use_still_frees_an_ordinary_intermediate` | **FAKE CONTROL for this patch** -- deleting `keep \|= set(ext)` does not move it, because it never passes externals |
| T-3 | `test_the_same_external_handle_survives_repeated_graph_runs` | **WEAK** -- uses `terminal=`, so it never observes internal eviction; the dict-identity assert cannot fail |
| T-4 | `test_a_malformed_render_knob_fails_before_any_file_work` | **NAME OVERSELLS** -- reordering the method so file work runs first still passes |
| T-5 | `test_an_identity_read_that_cannot_resolve_its_weights_RAISES` | **WEAK** -- `pytest.raises(Exception)` is maximally broad at the one place the docstring insists on an unwrapped type |
| T-6 | env-clean fixtures | **LEAKY** -- both `_ENVS` tuples omit `OTR_LTX_8GB_CFG`, `_SAMPLER`, `_BASE_SHIFT`, `_TERMINAL`, `_NEGATIVE`, so host env leaks in |

**Mutation survivors the suite does NOT catch** (empirically run by the lens):
* `keep |= set(ext)` -> `keep = set(ext)`: **62/62 still pass.** In production
  that silently wipes the caller's `keep` on EVERY call -- `render_clip` passes
  `keep={"ckpt","modelsampling",terminal}` with `free_after_use=True` -- freeing
  the MODEL patcher before the post-loop teardown grab. A real leaked-patcher
  path, invisible to the whole suite. **`keep=` has zero direct coverage
  anywhere.**
* `_file_receipt` returning a list instead of a tuple: 24/24 pass.
* Reordering `resolve_session_config`'s range-check after file work: 24/24 pass.

Also uncovered: `terminal` naming an external-only id (C-2's hole); `mtime_ns`
isolated from size (every "swapped weight" test changes the byte count too);
the `*_DIR` directory-override axis.

---

## WHAT IS NOT WRONG (stated so a later window does not re-litigate)

Sonnet lens A traced and empirically stress-tested the `run_graph` lifecycle and
found the mechanics sound: `remaining` cannot go negative and cannot
cross-contaminate (increments are per-distinct-consuming-node, deduped at
`:361`); `keep` is never read before it is fully built; the caller's dict and
tuples are never mutated across repeated calls; `{}` and `None` are provably
identical; `on_result` cannot be undone by the same iteration's `free_after_use`
(a node never depends on itself); no degradation on calls 2 or 3.

Lens C confirmed `repr()` of the receipt is a stable identity key (every element
is already a str or a deterministically-formatted float), and that
`begin_segment()`'s identity re-check -- unlike `open()`'s -- unwinds correctly,
because a raise there is inside the `with` suite and `__exit__` still runs.

---

## FIX ORDER

1. **C-1** (live false refusal) + **T-1** (the test that let it through).
2. **C-3** (stranded lease) -- resolve/validate before the lease is taken, or
   make `open()` unwind what it acquired when the baseline read raises.
3. **C-2**, **C-4**, **C-5** -- small, named, fail-closed corrections.
4. **T-2/T-3/T-4/T-5/T-6** + the `keep=` coverage hole.
5. The structural one: route `assert_usable` through the single resolver and
   retire `_ckpt_path`/`_t5_path`, so the guard protects the single-clip path
   too. Own chunk; it is the real close of the identity-lie defect.


---

# STATUS -- 2026-07-26, same session

**All five CODE defects are FIXED, mutation-proven with controls, and pushed.**
Suite 7023 -> **7045 passed** / 27 skipped / 1 xfailed. Bible 17. Canonical
byte-identical `9872624A` throughout.

| # | What | Commit |
|---|---|---|
| C-1 | path guard falsely refusing correct configs (`_same_file`: samefile + normcase/realpath fallback) | `ea1652f9` |
| C-4 | `_file_receipt`'s unguarded `os.stat` -> named `EngineUnusable` | `ea1652f9` |
| T-1 | the misnamed control that let C-1 through -> three real controls (separator, CASE, redundant spelling) + a direct fallback unit test | `ea1652f9` |
| T-6 | env-clean fixtures completed (6 vars were leaking the host env in) | `ea1652f9` |
| C-3 | a raising baseline identity read stranding the GPU lease for the life of the server | `f33c5e15` |
| C-2 | `terminal` validated against `results` -> validated against `graph` | `fdeee600` |
| C-5 | `_resolve_value`'s lookup outside its guard -> NAMED error | `fdeee600` |
| T-3 | repeated-runs test could not observe internal eviction | `fdeee600` |
| `keep=` | the mutation survivor with real teeth -- now covered | `fdeee600` |

**Mutation results, every fix, zero controls broken:** reverting C-1 to
`abspath` FAILS the new case control (the direct proof the fix is real);
`samefile` without its fallback and the fallback without `normcase` each fail
their own test; removing the C-3 unwind fails the teardown assertion and
narrowing `BaseException` to `Exception` fails the interrupt case; `terminal`
against `results` again, the lookup back outside the guard, `keep` overwritten
instead of unioned, and externals not kept at all each fail exactly their own
test.

**Two process notes worth keeping.**

1. The first C-4 test monkeypatched `os.stat`, which is process-wide -- it broke
   pytest's own traceback machinery with an INTERNALERROR. Model the real race
   (a resolver returning a genuinely absent path); never patch the interpreter
   out from under the runner.
2. The first version of
   `test_CONTROL_without_an_explicit_keep_an_intermediate_still_frees` ALSO
   asserted the external survived -- which made it a second test of the feature,
   so deleting `keep |= set(ext)` broke it and the harness correctly reported
   `CONTROLS_broken`. **A control must fail under OVER-tightening and pass under
   correct behaviour; it must never mirror the feature it bounds.** Caught by
   the mutation harness, not by review -- which is the argument for running the
   harness on the controls too, not just the fixes.

## STILL OPEN -- the structural one, and it is its own chunk

`resolve_session_config()` is reachable ONLY from `session_identity()`, which
`BeatSession` calls ONLY `if self.is_multi_segment`. So the identity-lie defect
B2a was written to close **remains fully open on the single-clip path -- the
common case.** `assert_usable()` still uses the old `_ckpt_path()` (existence +
size floor, no token cross-check). The QA lens reproduced it live: with
`OTR_LTX_8GB_CKPT` pointed at a decoy clearing the 4 GiB floor, `assert_usable()`
returned green while `resolve_session_config()` correctly raised, same
environment.

Closing it means routing `assert_usable` through the ONE resolver and retiring
`_ckpt_path`/`_t5_path` as competing authorities (codex CUT 1). Two resolvers
over one fact is the exact pattern chunk 4's lesson names, so this is not
cosmetic -- it is the actual close of the defect.

Also still open, lower: `_file_receipt` returning a list instead of a tuple
survives the suite; `mtime_ns` is never isolated from size in any drift test;
the `*_DIR` directory-override axis is uncovered; and
`test_a_malformed_render_knob_fails_before_any_file_work` still oversells its
name (reordering the method survives).
