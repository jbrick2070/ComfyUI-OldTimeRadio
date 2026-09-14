# 4060 regression after the widget tier

Paste the fenced block into the Cursor window on the 4060. Everything above it
is why this run exists.

Written 2026-09-14 from the 5080, after the widget tier landed on `main`.

---

## Why this supersedes the 2.1.2 brief

`docs/4060_TEST_BRIEF_2.1.2.md` was written for main @ 2.1.2. Since then the
**widget tier** landed and it changed **all 17 shipped graphs**, including the
four this box owns. So the receipts from `shipping_set_20260913_*` and the
overnight AnimateDiff run are now against a tree whose graphs no longer exist in
that shape.

What changed, in one list:

* Ten controls removed across the pack -- `perfect_run_spacesaver`, the `ffmpeg`
  widget on five nodes, `seed_mode` + `request_seed` on `OTR_VideoDirector`, and
  the duplicate `episode_title` on the Assembler and SignalLost.
* `OTR_LedgerScriptWriter` reordered: 36 widgets, and `source_bank` is now the
  only `required` entry (ComfyUI renders `required` before `optional`, so that
  move is what lets the order start there).
* Every saved graph migrated to match, link tables repaired by identity.

**Nothing was renamed**, so a graph either loads correctly or fails loudly --
there is no silent-wrong-value path. That is what this run confirms on 8 GB.

## Why this box specifically

The 4060 is the only machine that can answer whether the 8 GB profiles still
fit. The 5080 can run the same graphs and prove the shape, and it did -- but a
24 GB-class card proving an 8 GB graph proves the graph, not the fit, which is
the one thing an 8 GB graph exists for.

## Two traps this box has already been bitten by

1. **`-ObsDir` is not the server's `--output-directory`.** OTR pins
   `OTR_OUTPUT_DIR` to `D:\output` here, so episodes land in
   `D:\output\otr\obs`. Twice now the harness has printed `obs=0` on a run that
   really did publish, purely because `-ObsDir` was handed the launch flag
   instead. Take the obs directory from the server log's pinned line.
2. **Never pass `--title`.** The harness label becomes the on-screen title card.

---

## PASTE THIS INTO THE 4060 WINDOW

```
Regression-test the four 8 GB graphs on this RTX 4060 after the widget tier.
Self-contained; assume no memory of anything before this message.

WHY: the widget tier landed on main and rewrote all 17 shipped graphs. Ten
controls were removed across the pack and OTR_LedgerScriptWriter was reordered
to 36 widgets with source_bank as its only `required` entry. Every saved graph
was migrated to match. Your existing receipts predate that, so they describe
graphs that no longer exist in that shape. Nothing was renamed, so a graph
either loads correctly or fails loudly -- confirming that on 8 GB is the job.

START CURRENT:
  git fetch origin main
  git log --oneline HEAD..origin/main
  git pull --rebase origin main
Say what came down before running anything. If your remote's fetch refspec is
still narrowed to v2.0-alpha, widen it:
  git config --add remote.origin.fetch '+refs/heads/main:refs/remotes/origin/main'

STEP 1 -- THE FREE CHECK, AND DO IT FIRST. Boot the server, then convert every
shipped graph to an API prompt against the live /object_info WITHOUT rendering:

  python scripts/otr_canonical_api_run.py --workflow workflows/otr_canonical.json --dry-run --comfyui-url http://127.0.0.1:8000
  ...and the same for each workflows/variants/*.json

That is 17 conversions, seconds each, no GPU. It is the check that actually
catches a widget mismatch: a wrong count or a wrong name fails there rather
than 40 minutes into a render. The 5080 got 17 ok / 0 failed. If ANY graph
fails here, stop and report it -- do not proceed to renders.

STEP 2 -- THE FOUR 8 GB LEGS, cheapest first:
  powershell -File scripts\otr_shipping_set_legs.ps1 `
    -Url http://127.0.0.1:8000 `
    -Graphs otr_8gb_low,otr_8gb_still,otr_8gb_video,otr_8gb_animatediff `
    -ObsDir D:\output\otr\obs `
    -Python <this box's ComfyUI venv python>

-ObsDir MUST be D:\output\otr\obs -- the directory OTR_OUTPUT_DIR pins on this
box, NOT the server's --output-directory. Handing it the launch flag has twice
made the harness print obs=0 on a run that really published. Confirm the obs
path against the server log's pinned line before you start.

NEVER pass --title: the harness label becomes the on-screen title card.

THE SUCCESS SIGNAL IS AN EPISODE IN obs, NOT A GREEN LOG. For each leg, grep the
leg log for "obs_publish OK ->" and confirm the file on disk with its byte size.
If a leg has run over 5 minutes with nothing in obs, stop waiting and read the
leg log.

WHAT COUNTS AS A FAIL: a death only -- OOM, traceback, or hang. A warning is not
a failure and a slow leg is not a failure. An OOM on this card is the single
most valuable result you can produce, because 8 GB fit is the one question only
this box can answer. Capture the full traceback if you get one.

IF A LEG DIES: do not stop the run. Note it, let the harness clear the queue,
and continue -- three current receipts plus one named failure beats nothing.

REPORT, per leg: RESULT, minutes, the obs filename, its byte size, and the
engine markers in the filename. Also report peak VRAM if you have it, because
this run is about fit. Append to docs/4060_DRILL_LOG.md -- it is append-only and
merge=union, so it is safe to write while the 5080 is also pushing.

DO NOT edit workflows/, nodes/, or config/profiles/ -- the 5080 owns the
shipping surface and is actively changing it. You own docs/4060_DRILL_LOG.md.
Push that file only.
```

---

## What the 5080 has already proven, so you can compare

Three of four 16 GB legs on `main` at the widget tier, all SUCCESS, all
published:

| graph | minutes | markers |
|---|---:|---|
| `otr_16gb_low` | 9.8 | `vart vcam koko orig g412 sa3` |
| `otr_16gb_still` | 23.8 | `cart stmo zimg koko news g412 sa3` |
| `otr_16gb_animatediff` | 28.1 | `vstb adhv koko orig g412 sa3` |

Plus 17/17 on the submit-shape pass. The fourth leg, `otr_16gb_video` (LTX 2.5),
was still running when this was written.
