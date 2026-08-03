# QA: the mirror / ping-pong rip -- did I leave a hole?

**Operator directive, 2026-08-02:** "you must fix bugs, kill mirrors and
ping-pong -- true video for every second of audio."

This is a RIP, and this repo's own rule about rips is the thing to check me on:
**removing a mechanism is allowed; leaving its job unowned is not.** Enumerate
what the removed code was responsible for, confirm each responsibility has a new
owner, and say plainly if one is now unowned.

**Do not boot a server, do not launch renders, do not edit code. READ-ONLY.**
Use the file tools -- they read the real Windows files. Verify with `git diff`.

## What was removed

| Where | What |
|---|---|
| `wrapper_bridge.py` | `extend_frames_to_target` -- the ping-pong/mirror extender -- **DELETED**, and dropped from `__all__` |
| `wrapper_bridge.fit_frames_to_target` | the `allow_mirror` parameter -- **REMOVED**; a short render now always raises `MirrorExtensionForbidden` |
| `eng_ltx_video._loop_via_reverse` | now returns **False unconditionally**; `OTR_LTX_LOOP_VIA_REVERSE` is inert for every value |
| `beat_session.session_ctx` | `multi_clip` now means COVERAGE-PLANNED, not `segment_count > 1`; new `coverage_planned` parameter |
| `render_driver` (x2) | per-segment AUDIO window now gates on `requires_coverage_execution`, not `is_multi_clip` |

Suite: 8289 passed, 131 skipped, 1 xfailed, 0 failures.

## What the removed code was responsible for -- check each owner

1. **Filling a beat the engine could not render in one affordable pass.**
   Claimed new owner: coverage planning splits it into native forward-rendered
   segments. **Verify this holds for every engine that previously relied on the
   extender** -- particularly the WAN tiers, since a test used to assert the
   extender "still EXISTS for the lanes that need it" and named the shipped
   8 GB WAN tier specifically.
2. **`ltx_video`'s half-render-and-mirror loop fill.** Claimed new owner: the
   same coverage planning, since `ltx23_16gb_video` chains. **Verify a beat that
   used to be covered by a 193-frame mirrored source is now covered natively**,
   and that `_ltx_loop_source_length` / `_loop_fill_allowed` going unreachable
   leaves no other behaviour (e.g. the 169-frame decode floor) stranded.
3. **The degenerate single-frame case.** The extender repeated a still to fill a
   beat. That now raises. **Is there a lane where a 1-frame render is legitimate
   and this refusal breaks it?**
4. **Credits.** Believed untouched -- they render a black background and called
   neither path. **Confirm.** The operator's exception was explicit: "no
   boomerangs ... except for credits."

## The specific things I most want broken

* **Is `MirrorExtensionForbidden` now reachable on a path with no useful
  message?** Its one caller wraps it with routing advice; if the rip exposed it
  somewhere unwrapped, a beat fails with a bare error and no remedy.
* **`beat_session`'s changed meaning.** `multi_clip` now reports
  coverage-planned. Any consumer that genuinely wanted "more than one segment"
  is now silently getting a different answer. **Find every reader of
  `session_ctx()["multi_clip"]` and say which question each is actually asking.**
* **Did I miss a mirror?** Search the whole tree -- `nodes/`, `scripts/`,
  `config/profiles/*.json`, `workflows/otr_canonical.json` -- for any other
  frame-reuse, hold-last-frame, freeze-fill, loop-fill or reverse path. The
  composite's `_warn_clip_underrun` / loop-fill machinery in
  `otr_silent_composite` is worth a hard look: **does anything still LOOP a
  short clip at composite time?** That would defeat the whole rip.
* **Does anything still hold a frozen frame** to cover audio a render did not
  reach? A freeze is not a mirror, but it violates the same rule.

## Do not reinvent the wheel

Before proposing a fix, check whether the archive already settled it:
`docs/*.md` (2026-07*, 2026-08*), `BUG_LOG.md` and `BUG_LOG_2026-06.md` in the
repo ROOT, `docs/PROD_BUG_LOG.md`, `kibitz-runs/**`, and
`comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`. This session has twice
re-derived a worse answer to an already-solved problem.

## Report format

Worst-first, max 10 findings: **WHAT** (one sentence), **WHERE** (`path:line`),
**STATUS** (UNOWNED RESPONSIBILITY / BROKEN NOW / RISKY / WORKS AS INTENDED),
**PRIOR ART** (cite it, or "none found"), **FIX** (only if needed and not
already prescribed).

Anchor everything to a real `path:line`. UNVERIFIED is a legitimate answer.

## CONSTRAINTS

100% local, offline-first. 16 GB RTX 5080, 14.5 GB real-world ceiling. Every
second of audio gets ORIGINAL video -- no mirrors, no ping-pong, no held frames,
credits excepted. `wan_8gb`'s sampler recipe is FROZEN. The only workflow JSON is
`workflows/otr_canonical.json`.
