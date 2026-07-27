# B3 QA record -- the LTX-only effective frame contract

Two fan-outs, one before a line was written and one before the push: 3 Sonnet
lenses + 1 agy (`Gemini 3.6 Flash (High)`, via kibitz) each round, every lens
launched in ONE block so they ran concurrently. $0 external. Claude wrote the
anchor design first, grounded every panel claim against the real Windows files,
and judged. No codex spend -- the architecture was already decided in
`docs/2026-07-26-8gb-1080p-arc-judgment.md` and no fix needed a second attempt.

## The pre-code panel killed three of its own recommendations

Two of agy's three MUST-FIXes did not survive grounding, and saying so matters
more than the fixes that did land -- an adopted-but-wrong panel fix is a defect
with a second opinion attached.

1. **"Pass `engine_id` through `public_engines.resolve_engine_id` inside
   `effective_frame_contract`." REJECTED.** Grounded:
   `registry.is_registered("ltx_8gb (16:9)")` is `False`, and
   `_stamp_coverage_plan` returns at its registration gate before the
   derivation is ever reached -- so a suffixed id already gets NO plan, not an
   uncapped one. Adopting it would have added a SECOND normalization authority
   that disagrees with the registry: an id the registry rejects would start
   behaving as `ltx_8gb`, and `frame_contract.py` (documented "cold-import
   clean: stdlib only") would gain a dependency it exists not to have.
2. **"A required keyword-only `max_render_frames` breaks existing test
   callers." REJECTED.** The only occurrence of `_stamp_coverage_plan` in
   `tests/` is inside a module docstring; `build_execution_plan` is the sole
   caller in the tree. The proposed remedy -- default it to 0 -- is exactly the
   silent-fallback shape this build removes: a caller that forgets the ceiling
   would plan unpinned and nothing would say so.
3. **"Exact receipt equality will crash every legacy/soak run where the force
   map mutates `engine_id`; compare everything EXCEPT `engine_id`." REJECTED
   on both halves.** No shipped profile pins a ceiling on an `ltx_8gb`-routing
   tier, so no receipt exists to mismatch today. And a plan built under one
   engine's ceiling, executed by another, MUST refuse -- which is what
   `test_the_legacy_path_validates_the_plan_against_the_FINAL_engine` already
   establishes one contract further down.

**Adopted from the pre-code panel, and each one improved the design:** the
derivation sits OUTSIDE both broad `except Exception` blocks (three seats named
that line independently); `PlanningCapError` is wrapped into `RenderError` at
the render boundary, because it is a `ValueError` and NOT a
`CoveragePlanError`, so the existing narrow catch would have let it escape
unformatted; and the ceiling is normalized by ONE function rather than three
hand-copied `max(0, int(x or 0))` expressions.

## The post-code panel found four real defects in green, mutation-proven code

All four were in code that had already passed 24 focused tests, 12 mutants and
the full 7122-test suite. This is the sixth time a panel has done that here.

1. **The unresolved-engine branch compared only the CEILING, never the ENGINE.**
   Two seats built the same live repro independently: a stale `ltx_8gb` receipt
   on a shot whose engine was swapped to an unregistered id, ceiling unchanged,
   sailed through to an arithmetic-only check -- a plan built for an 8n+1
   ladder accepted for an engine never checked against any length. The branch
   now holds the ledger to BOTH facts the receipt states about its own
   provenance.
2. **A malformed receipt read as no receipt.** `shot.get(...) or None` collapses
   `{}` to "nothing narrowed", so a damaged ledger was indistinguishable from an
   ordinary unpinned one. Now terminal and named.
3. **The `discrete_frames` guard broke the function's own documented
   guarantee.** It refused ANY nonzero ceiling on a menu contract, including one
   far above the whole menu -- while the docstring promises that a non-binding
   ceiling returns the contract unchanged. A tier that never constrained the
   engine would have failed. Now: refuse only when the ceiling actually bites.
4. **`motion_common.profile_max_render_frames` was a fourth hand-copied
   normalization** -- and it is live: `eng_wan_ti2v._floor_length` reads it to
   resolve WAN's native cap at render time. The test whose name promised
   "exactly one normalization" never touched the one site its own docstring
   cited as justification. That is the decorative shape this build keeps
   finding, in a test written to guard against decorative tests. Both fixed.

**Also caught, by me, mid-write:** the first draft of the stamp site rebound one
variable and fed the ALREADY-NARROWED contract into
`coverage_contract_receipt`. A narrowed contract narrows to itself, compares
equal, and returns `None` -- so the receipt would have silently never existed
and the render boundary would have had nothing to check. Every test in the file
would still have passed. It is now pinned by name.

**And by mutation:** validating the plan against the NARROWED contract was
unobservable -- the receipt equality fires first in every scenario the tests
covered. `test_the_render_boundary_validates_against_the_NARROWED_contract`
(a receipt-valid ledger whose PLAN was tampered with) is what makes that
argument load-bearing.

## Mutation proof

17 mutants: 15 DEFECT, all red; 2 CONTROL, both green; baseline and restore
green. The controls move values the recipe is entitled to move --
`_LTX8_MAX_FRAMES_DEFAULT` (the env-read default, which the static contract
deliberately does NOT derive from) and `_TI2V_DEFAULT_FRAMES` -- and prove the
assertions read the DECLARED contract rather than secretly pinning an env knob.

## Known, accepted, and recorded rather than fixed here

- **B3 is production-inert today.** No shipped profile pins
  `max_render_frames` on a tier routing to `ltx_8gb`; the only profile that
  pins it is `config/profiles/otr_8gb_wan.json` at 17, and WAN is deliberately
  outside the allowlist. So B3 changes zero bytes of rendered output on any
  shipped tier, and it cannot be proven on a live leg by itself.
- **DO NOT PIN AN LTX CEILING BEFORE B4 LANDS.** `ltx_8gb`'s CLIP-FILL
  ping-pong currently LAUNDERS a plan-vs-adapter frame-count disagreement:
  `_ltx8_frame_length` clamps a segment down to the env cap, the ping-pong pads
  it back up to the requested length, and `render_driver`'s
  `got != segment.render_frames` assertion then compares equal and passes. The
  claim that this disagreement is "already terminal" -- which the B3 brief made
  and the arc judgment implies -- is FALSE until B4 deletes that extension.
- **A very low ceiling makes long beats structurally uncoverable.** At a ceiling
  of 9 the ladder collapses to a single legal length and `max_segments=64`
  bounds coverage at roughly 576 visible frames, so a beat past ~23 s refuses
  outright where the static 161 ceiling covered it. Fail-closed and intended,
  but a tier author needs to know it before pinning near the floor.
- **`schemas.py`'s `ShotRow` is a closed (`extra="forbid"`) model that no
  boundary enforces and that no longer matches production shot rows** -- it is
  missing `beat_id`, `role`, `char_id`, `start_s`, `dur_s`, `coverage_plan`
  (since chunk 3b) and now `coverage_contract`. Nothing validates a real ledger
  through it, so this is not a live break; it is a contract that other docs in
  this tree cite as a live safety net. Recorded as an open defect rather than
  quietly widened inside a chunk that is not about it.
- **`docs/ENGINE_MATRIX.md` reports the DECLARED contract only.** Correct today
  and by its own stated design (every number is read from the live registry),
  but the moment a profile pins an `ltx_8gb` ceiling the matrix will print
  `9-161 step 8` for a tier whose real window is narrower, and the `--check`
  drift gate will not notice because it diffs the registry, which B3 never
  touches. Owed when a ceiling is actually pinned.
- **The `run_episode` / `run_gpu_soak` / `render_single` validation bypass is
  UNCHANGED by B3** and currently inert for it (those harnesses never stamp a
  coverage plan). It stays an open item from the 7b judgment.

## Records

Pre-code brief `tmp/_kbA_b3_brief.md`, post-code brief
`tmp/_kbA_b3_postbrief.md`, agy runs under
`kibitz-runs/2026-07-26-b3-precode-agy/` and
`kibitz-runs/2026-07-26-b3-postcode-agy/`.
