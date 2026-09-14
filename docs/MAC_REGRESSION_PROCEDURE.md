# Standing procedure: re-prove the Mac graphs on a rented machine

**This file is meant to be re-used unedited.** The Mac is a RENTAL, not a
machine we own, so this runs a few times a year: rent, paste the block at the
bottom, collect four receipts, let the rental lapse. Nothing in the pasted block
names a particular week's reason, so it does not go stale between rentals.

The dated instances live at the bottom, as a log. Add a row; do not rewrite the
procedure.

---

## When this is worth renting for

Not on a schedule, and not "every major update". The trigger is a **stale
receipt**, and the repo already computes the thing that decides it.

`semantic_master_hash` (`nodes/_otr_workflow_apply.py:451`) hashes node types +
links + the profile-MANAGED widget values. By construction it EXCLUDES pos/size
and UI keys, the node-63 stamp widgets, and every creative widget -- so a canvas
tidy-up, a title edit or a premise change does not move it, and a wiring change,
a node swap or a managed-value change does.

That is exactly the line a receipt should expire on:

* **Hash unchanged since the last Mac receipt** -> the receipt still stands. Do
  not rent. Six canvas-layout commits on 2026-09-13 moved no hash.
* **Hash moved on one or more `otr_mac16_*` graphs** -> those receipts are dead
  and the Mac surface is shipping on a dated claim. Rent when convenient, or
  ship with the dated label (below) and say so.
* **About to publish a registry version** -> rent first if any Mac hash has
  moved. A publish reaches strangers and cannot be taken back.

To see where you stand, compare each graph's current hash against the hash
recorded with its last receipt in `docs/MAC_LAB_LOG.md`. If a receipt has no
hash recorded, it predates this procedure -- treat it as stale.

## What only this machine can answer

Worth being precise about, because it decides whether renting is the right spend
at all.

**Already covered without a Mac.** Graph shape and wiring -- each Mac variant's
links are byte-identical to its nearest NVIDIA sibling; the differences are
device strings, quant, canvas and the engine pick. Widget order and count,
dropdown validity, and submit shape are all covered by the static gates and the
API-prompt conversion, on any machine.

**Only a Mac can answer:** whether the picked engine actually FITS and RUNS on
MPS at that canvas. Bounded by `device_backends`, never measured by it. The
worked example is PBUG-20260913-04, an AnimateDiff VAE decode OOM on MPS that no
static test could have seen.

**Two of the four graphs are the only place something exists at all:**

| graph | why nothing else covers it |
|---|---|
| `otr_mac16_low` | `viz_camera` + `sd15` on MPS -- cheapest, so it runs first |
| `otr_mac16_still` | `still_motion` + `sd15` on MPS |
| `otr_mac16_video` | `ltx098_low_video` at 832x480 -- the 8 GB sibling runs it at 512x288 |
| `otr_mac16_animatediff` | the ONLY graph anywhere using `animatediff15_lightning_video` |

## An OOM on a Mac reboots the machine

The one operational difference from the NVIDIA boxes, and the reason the order
is cheapest-first: a reboot after leg 4 costs one leg, a reboot during leg 1
costs the night. `apple/MAC.md` section 2 is the reading.

## If you do not rent, say so honestly

Shipping the Mac graphs on a dated receipt is a legitimate choice -- it is a
rental, and the graphs are static-clean. What is not legitimate is implying the
exact file was proven when it was not. The honest form, which
`build_variants.py` can carry into each `.launch.md`:

> Published an episode on a Mac mini M4 / 16 GB on `<date>` (one act,
> `<minutes>` minutes). This file has been regenerated since (`<what changed>`);
> the graph shape and every engine it selects are proven on that machine, the
> exact file is not.

---

## PASTE THIS INTO THE MAC WINDOW

```
Re-prove the four otr_mac16_* shipped graphs at HEAD on this Mac, and publish
every one to obs.

FETCH BEFORE YOU BELIEVE ANYTHING, INCLUDING YOUR OWN INSTRUCTIONS. A rental
machine is stale by construction -- it was last up whenever it was last rented --
and a stale checkout carries a stale CLAUDE.md, so the rules you are reading may
themselves be out of date. This has already cost one night's rental.

  git fetch origin
  git log --oneline HEAD..origin/main
  git rev-list --left-right --count origin/main...HEAD

THE SPECIFIC TRAP: a checkout from before 2026-09-13 carries a CLAUDE.md saying
`main` is a stale v1.7 branch 3,923 commits behind and must never be treated as
current. That WAS true, and it was reversed on 2026-09-13 by commit `4e8acae3`
("v2 is promoted: main is the branch, v2.0-alpha is retired"). The old v1.7 tip
was preserved at `archive/main-v1.7`, which is what that warning now refers to.
`v2.0-alpha` is a strict ancestor of main -- `git rev-list --left-right --count
origin/main...origin/v2.0-alpha` returns commits-ahead on the left and ZERO on
the right -- so moving to main is a fast-forward and strands nothing.

Verify that yourself with the two commands above rather than taking it from this
file, which is also a document that can go stale. Then:
  git pull --rebase origin main

Say what came down before you run anything.

THEN ESTABLISH WHAT IS ACTUALLY STALE, BY MEASUREMENT. For each of the four
otr_mac16_* graphs, compute semantic_master_hash
(nodes/_otr_workflow_apply.py:451) and compare it with the hash recorded beside
that graph's last receipt in docs/MAC_LAB_LOG.md. Report the four comparisons
BEFORE running anything, and say plainly which receipts are dead and which still
stand. A graph whose hash has not moved does not need a leg; if all four are
unchanged, say so and stop -- that is a successful outcome, not a wasted rental.

IF A RECEIPT HAS NO HASH RECORDED, it predates this procedure and you cannot
compare. Do not guess and do not re-run on faith: diff the graph files between
the commit that receipt was taken at and HEAD --

  git diff --stat <receipt-commit>..HEAD -- workflows/variants/otr_mac16_*.json

-- and read what actually changed. Four files with one changed line each is a
single managed value moving across the set, and the diff will name it. This is
how the 2026-09-14 instance was settled: the four graphs carried `musicgen` at
the older commit and `stable_audio_3` at HEAD, so legs run at the older commit
had proven the wrong music engine. Prove staleness from the FILES; a commit
message is a claim, a diff is evidence.

THEN RUN THE STALE ONES, CHEAPEST FIRST. An OOM on a Mac reboots the machine, so
the cheap legs bank their receipts before an expensive one can take the box
down. Typical minutes from the last run, for ordering only:
  1. otr_mac16_low          (~15)
  2. otr_mac16_still        (~26)
  3. otr_mac16_video        (~56)
  4. otr_mac16_animatediff  (~65)

USE THE EXISTING HARNESS -- do not write a new one:
  scripts/otr_shipping_set_legs.sh <comfyui-url> <obs-dir> <python> <graph>...

It runs one act per leg by default, writes one log per leg plus a SUMMARY.txt
under otr/legs/shipping_set_<stamp>/, reuses one warm server, and clears the
queue after a leg that did not reach SUCCESS so the next leg does not stack up
behind a wedged render. Never pass --title: the harness label becomes the
on-screen title card.

CHECK THE OBS DIR MATCHES THE SERVER'S OWN OUTPUT DIRECTORY before you start. A
mismatched obs dir reports obs=0 on a real SUCCESS, which reads as a failure and
is not one.

THE SUCCESS SIGNAL IS otr/obs/, NOT A GREEN LOG. A leg counts only when the
episode lands there -- grep the leg log for "obs_publish OK ->" and confirm the
file on disk. If a leg has run more than 5 minutes with nothing in obs, stop
waiting and go read the leg log.

WHAT COUNTS AS A FAIL: only a death -- an OOM, a traceback, or a hang. A warning
is not a failure and neither is a slow leg.

IF ONE LEG DIES: do not stop the run. Note it, let the harness clear the queue,
and keep going -- three current receipts plus one named failure is a far better
night than nothing. A Mac OOM may reboot the box; if it does, restart the
harness with only the graphs that still have no receipt (every row in
SUMMARY.txt without RESULT SUCCESS).

WATCH FOR A SILENT ENGINE FALLBACK. This is the failure most likely to be missed,
because the leg still SUCCEEDS: an engine that cannot initialise on MPS may fall
back to a CPU or legacy path, and the only tell is the obs filename's engine
marker. For every leg, report the obs filename and confirm its markers match the
engines the graph actually asked for. A mismatch is the single most valuable
thing you can tell me.

REPORT BACK, per leg: RESULT, minutes, the obs filename, the engine markers in
it, and the semantic_master_hash the receipt should be recorded against.

DO NOT edit workflows/, nodes/, or config/profiles/ -- the Windows box owns the
shipping surface. You own docs/MAC_LAB_LOG.md: append your results there,
INCLUDING the hash beside each receipt so the next rental can tell what is
stale, and push that file only.
```

---

## Instance log

| date | graphs run | why the receipts were stale | outcome |
|---|---|---|---|
| 2026-09-13 | all four | first full Mac proof | 4/4 to obs (15 / 26 / 56 / 65 min) |
| 2026-09-14 | pending | `958c7d1d` made Stable Audio 3 the default music engine on every shipped graph AFTER the 09-13 legs finished. **Proven by diff, not by the commit message:** at `c490e62b` all four graphs carry `musicgen`; at `56c2bd30` all four carry `stable_audio_3`, and `git diff --stat` over the four files shows one changed line each. So the 09-13 legs proved musicgen on MPS. SA3 on MPS has never run end to end in a real graph -- that is what these legs buy. | rental expires 2026-09-14 |

### What went wrong on the first attempt, 2026-09-14

The first version of this prompt was refused by the Mac, correctly, and the
refusal is the reason the branch section above exists.

The Mac reported that commit `958c7d1d` "doesn't exist anywhere" and that
`origin/main` was the stale v1.7 branch, and declined to pull. Both statements
were false **of the repo** and entirely reasonable **from where it stood**: its
checkout was 86 commits behind, so it genuinely could not see the commit, and
its CLAUDE.md was the pre-promotion copy that still described main as stale. It
was reasoning correctly from stale inputs.

Two lessons, both now in the block above:

* **A rental machine's own instructions are part of what is stale.** Telling it
  to read CLAUDE.md and then follow the branch rule is circular. It has to fetch
  and MEASURE the branch relationship first.
* **Cite evidence a stale machine can verify, not a commit hash it cannot see.**
  The driver of the first attempt passed on a commit id taken from a subagent's
  report without checking it, which is what made the exchange expensive. A
  `git diff` over the graph files settles the same question from either side.

