# PLAN -- clear the four 7b blockers, reach the 7d live slice

**Optimisation target, operator's words: "the cleanest architecture for the
future set." Not the smallest diff.** Where a clean end state costs more churn
now, take the churn. Where a blocker's "obvious" fix leaves a second authority
standing, say so and propose the collapse instead.

HEAD `ade0b938` on `v2.0-alpha`. Suite **6925 passed / 27 skipped / 1 xfailed**.
Bible 17. Canonical `5377914B` byte-identical. `engine_matrix --check` OK.

**GPU IS FREE THIS WINDOW** -- RTX 5080, 2306 MiB of 16303 used, port 8000
unbound, no resident ComfyUI. 7d is reachable. Nothing has ever rendered
through this architecture.

Background, in order: `docs/2026-07-27-multiclip-7b-fork-judgment.md` (the
settled decision), then GO_FORWARD CURRENT STEP. The A-vs-B framing in
`docs/2026-07-26-chunk-7b-window-prompt.md` is DEAD; do not revive it.

---

## 0. WHAT IS ALREADY SETTLED (do not relitigate)

* Option A (refuse on env-vs-declaration) is CUT.
* `render_driver.py:2952-2958` already makes the divergence terminal on the
  MULTI-segment path by comparing rendered OUTPUT to the plan. That predicate
  is the architecture; everything below serves it.
* The SINGLE-segment path has no such proof, and it is the only path
  production runs today.
* `frame_contract_for` stays STATIC -- `tools/engine_matrix.py:145-152`
  generates a committed doc from it under a `--check` drift gate.
* The boomerang (`ltx_video` declares 169, returns 193) is 7c's, tripwired.
* `OTR_FORCE_ENGINE_MAP` is already closed; no second check.

## 1. THE FOUR BLOCKERS -- all verified against source

**B1 -- the canonical workflow never wired `max_render_frames`.**
`OTR_VideoDirector.INPUT_TYPES` declares it last in `optional`
(`otr_video_director.py:302-318`). Node 87 in `workflows/otr_canonical.json`
has NO input descriptor for it: its `inputs` array ends at `dtype_policy`, yet
`widgets_values` carries a 15th trailing `0` -- the default, unbound. The
profile-ceiling channel (widget -> ledger `video.max_render_frames` ->
`build_episode_render_policy` -> `prepare` ->
`motion_common.profile_max_render_frames()`) is therefore **dead in the real
workflow.** Fix IN `otr_canonical.json`, same commit, then re-validate:
`OTR_WorkflowValidator`, JSON round-trip, link/widget audit (widget count vs
live `INPUT_TYPES`, every wired input name in `INPUT_TYPES`, link referential
integrity). `widgets_values` is POSITIONAL -- only ever APPEND (BUG-LOCAL-097).

**B2 -- ComfyUI serves a STALE plan across a frame-cap env change.**
`OTR_ShotLock.IS_CHANGED` (`otr_shot_lock.py:1412-1422`) fingerprints only
`route_freeze.routing_env_snapshot()`, which is exactly two variables:
`OTR_FORCE_ENGINE_MAP`, `OTR_ENABLE_HUMO_HOSTS` (`route_freeze.py:50-51,
63-76`). Change `OTR_WAN_TI2V_MAX_FRAMES` and ShotLock does not re-lock, so the
cached coverage plan AND its stamp survive the change that was meant to move
them. `route_freeze.py:46-48` already warns: *"Any new routing env var MUST be
added here AND to the snapshot, or it escapes the freeze ... an env read
outside the snapshot is a hole in the ledger."*

**ARCHITECTURE QUESTION, and this is the one that matters most for the end
state:** does the frame-cap env set belong INSIDE `routing_env_snapshot` -- the
existing route authority -- or in a SECOND sibling snapshot with its own
fingerprint, both folded into `IS_CHANGED`? Putting frame caps inside a thing
named "routing" makes the name a lie and couples the frozen-route equality
check (which is TERMINAL on drift, `otr_shot_lock.py:1460-1475`) to values that
are not routing. A second snapshot keeps the names honest but creates two
authorities to keep in sync. **Pick one and justify it as the durable shape.**

**B3 -- both plan boundaries SWALLOW what this work makes terminal.**
`_stamp_coverage_plan` catches contract exceptions and emits no plan at all
(`otr_shot_lock.py:1150-1155`, `except Exception: return`);
`assert_coverage_plans` catches live-contract exceptions and degrades to
arithmetic-only validation (`render_driver.py:3430-3438`, `contract = None`).
A resolver that raises is absorbed at BOTH ends and the build looks fine.
Chunk 1a's lesson verbatim: *when you make something newly terminal, grep every
caller for a broad catch in the SAME change.* Both catches have legitimate
uses today (an unregistered engine, a stub) -- the fix must separate "this
engine is not resolvable" from "this engine's config is wrong", not delete the
catch.

**B4 -- `frame_count` is an ESTIMATE for 13 of 31 engines.**
`eng_cloud_video.canonicalize` computes
`int(round((asset.duration_s or 0.0) * (asset.fps or 0.0)))`
(`eng_cloud_video.py:491`); the media probe reads format duration, not counted
frames (`_otr_shared/cloud_media_canonical.py:402-436`).
`docs/ENGINE_MATRIX.md` puts the roster at 18 local / 13 provider. So the
output-equality proof compares the plan against a number derived from the same
duration a provider clamp would move -- it cannot see the clamp, and rounding
alone can trip it spuriously. `wan_shared.ffprobe_counted_frames` already
exists (chunk 6a, for the assembly boundary) and decodes. This is wiring.

**Coupled to B4:** the proof's own guard is fail-OPEN.
`render_driver.py:2952`: `got = int((clip or {}).get("frame_count") or 0)`
then `if got and got != ...` -- a clip reporting 0 or absent skips the check
entirely, and `CanonicalClip.frame_count` defaults to 0
(`schemas.py:216-235`). Fail-open guard inside a fail-closed function. Fix in
the same change as B4; they are one predicate.

## 2. THE END-STATE SHAPE THIS SHOULD CONVERGE ON

State it as a target so the panel can attack it rather than infer it:

1. **ONE resolver** -- `resolved_frame_contract_for(engine, *, env,
   profile_max_frames, request_template=None)` -- the single place any
   frame-ceiling environment variable or profile value is read.
   `frame_contract_for` stays static and untouched.
2. **ONE counted number.** Every engine's canonicalized clip carries a
   `frame_count` that was COUNTED, not derived. No caller ever divides a
   duration to get frames again.
3. **ONE proof, on every path.** The plan-vs-output equality at
   `render_driver.py:2952` becomes a single predicate that both the
   single-segment and multi-segment paths call. Not two copies.
4. **ONE cache key.** Whatever the resolver reads is fingerprinted into
   `IS_CHANGED`, so a plan can never outlive its inputs.
5. **The stamp is a receipt, not a second authority.** It exists so a replayed
   or cross-process ledger can be checked, and `assert_coverage_plans`
   validates against BOTH stamped and freshly-resolved
   (`tests/test_multiclip_coverage_stamp.py:248-262` pins the live guard --
   stamped-only deletes that test's reason to exist).

**Reject any proposal that leaves two places computing one number.** That is
the defect shape this build has removed five times (chunk 1a routing mirrors,
chunk 4 still-spine inference, QA4 route ordering, QA6 segment length, QA7 the
terminal-frame key), and re-introducing it for the resolver would be the
sixth.

## 3. PROPOSED ORDER

| # | slice | why here |
|---|---|---|
| C1 | B1 canonical wiring + validator + widget/link audit | nothing downstream is reachable without it, and it is self-contained |
| C2 | B4 counted `frame_count` + the fail-open predicate | one predicate; makes the existing multi-segment proof honest for 13 lanes |
| C3 | the resolver + B3's two catches, same commit | the catches are what would hide it |
| C4 | the stamp beside `coverage_plan` + B2's `IS_CHANGED` | the stamp is only trustworthy if its inputs are keyed |
| C5 | the single-segment proof | AFTER the resolver: on an 8GB box with `OTR_WAN_TI2V_MAX_FRAMES=49` a 177-frame beat is a legal single plan today, renders 49, and this would refuse it with no remedy until C3 lands |
| C6 | boundary comparison: keep the live guard, ADD stamped-vs-live | last, because it needs C4's stamp |

**Challenge this order.** In particular: can C2 land before C1, and does C5
really need C3, or is the 8GB case better handled by making that box's plan
multi-clip at C3 time anyway?

## 4. THE 7d QUESTION -- what can run on the GPU NOW

The GPU is free and nothing has ever rendered through this architecture. The
plan is to run a live leg IN PARALLEL with this arc rather than after it.

**Say whether that is safe, and what the smallest honest first leg is.**
Specifically:

* Is a live multi-clip beat reachable at HEAD, with none of C1-C6 landed? The
  multi-clip machinery (chunks 1-6, 7a) is landed and green; the blockers are
  all about the env/profile ceiling, and with NO frame-cap env set on this box
  the static contract and the resolved one are the same number.
* 7d as specified is a 169-frame beat (`161 + (9-1)`, `169 mod 8 == 1`), >= 2
  forward-only clips, ONE heavy load, no ping-pong, plus a 162-frame CPU
  tail-trim case. Acceptance: `RESULT SUCCESS` + `obs_publish OK` + the asset
  on disk confirmed by `Test-Path` at `otr\episodes\<ep>\` / `otr\obs\`.
* **Is 169-on-`ltx_video` the right first leg at all?** `ltx_video`'s boomerang
  is ON by default and returns `2N-1`, so a 169 ask comes back 193 and the
  multi-segment per-segment equality check at `render_driver.py:2952` would
  refuse it. Name the engine and beat length whose first live multi-clip leg
  is actually clean, or say that leg must set `OTR_LTX_LOOP_VIA_REVERSE=off`
  and why that is honest rather than a workaround.

## 5. INVARIANTS -- reject any fix that breaks one

* `workflows/otr_canonical.json` is THE workflow; any node/wiring/widget change
  goes in it in the SAME commit as the code. `widgets_values` is POSITIONAL --
  append only.
* `frame_contract_for` stays static or the generated `ENGINE_MATRIX.md` becomes
  machine-dependent.
* No fallbacks, no silent re-plan, no fallback assets, no truncation, no
  arbitrary provider caps.
* THE LAW: an audit may improve a story, never fail one for length, language,
  style, visual vocabulary, or quality.
* Every slice green and pushed on its own; mutation-prove every fix.
* Never blanket-kill Python (severs the MCP pythons, incl. this bridge).
  Selective CIM kill by CommandLine only.
* UTF-8 no BOM, ASCII, SFW. $0 external spend.

## 6. GROUND RULES FOR REVIEWERS

Cite `file:line` from the real tree at
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`. A claim
with no line number is dropped; a claim whose line number does not say what it
is said to say is dropped louder. In the previous arc on this build, the panel
refuted two of the driver's own load-bearing claims and was right both times,
and the driver caught a `render_driver.py:2952` predicate both seats missed.
Both directions happen. Prefer the claim you can point at.
