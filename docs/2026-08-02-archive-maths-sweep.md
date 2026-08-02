# Sweep the archive: what maths did we already solve and then forget?

**Operator, 2026-08-02.** Today I re-derived a WORSE answer to a problem this
repo had already solved well, one day earlier, with the same model. The lesson
was in `docs/` the whole time and I did not read it. This sweep exists to find
every other instance before I repeat the pattern.

**Your job is SEARCH, not design.** Read the archive and report what is already
decided. Do not propose new architecture. Do not launch renders or boot a server.

## The failure mode to hunt for

A number, formula, or decision that was MEASURED or RULED ON in a doc, and is
now either (a) contradicted by current code, (b) re-derived differently
elsewhere, or (c) simply forgotten -- sitting in a doc nobody reads.

The known instance, as the template:

* `docs/2026-08-01-fastwan-frame-cap-TWO-STRIKES.md` measured VRAM FLAT at
  6563.1 / 6531.1 / 6563.1 MiB for 17 / 49 / 81 frames @832x480, identified
  `vae_temporal=16` as the mechanism, and concluded "the linear `per_frame` term
  does not describe this engine at all."
* Its kibitz ruling then said the 6563 figure is `peak_delta_mib`, NOT
  machine-wide usage, so it may NOT be written into `FRAME_COST_MODEL` as
  overhead -- and forbade refitting coefficients from bench data at all.
* `FRAME_COST_MODEL` / `_DEFAULT_FRAME_COST` in
  `nodes/_otr_video_engines/motion_common.py` still carries `(7000.0, 185.0)`, a
  LINEAR model, today.

## What to search

Everything under `docs/` and `kibitz-runs/`, plus `docs/PROD_BUG_LOG.md`,
`docs/PRODUCTION_SPRINT_LESSONS.md`, and
`C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`.

Topics where a forgotten number would hurt most:

1. **VRAM / frame budgets** -- any measured peak, ceiling, overhead, per-frame
   cost, safe rung, or free-VRAM reference. Especially any number tied to a
   named engine or canvas.
2. **Frame contracts and caps** -- min/max/quantum/stride decisions, why a
   ceiling is what it is, and any "qualified at X" statement.
3. **Coverage / multi-clip partitioning** -- segment counts, drop_head/trim_tail
   arithmetic, join-mode rulings, audio-slice arithmetic per segment.
4. **Canvas / resolution** -- which engine renders at what, why, and any measured
   quality or VRAM consequence of changing it.
5. **Continuity** -- chain vs jump rulings, reference-image semantics, seed
   pinning, identity across cuts, and the FEAR CAPE seam.
6. **Any explicit "do not do X" ruling** -- standing prohibitions that current
   work might violate.

## Report format

For each finding, give exactly:

* **The claim** -- the number/formula/decision, quoted.
* **Where** -- `path:line`.
* **Date + who ruled it** (model/panel if the doc says).
* **Status** -- one of: STILL TRUE IN CODE / CONTRADICTED BY CODE /
  SUPERSEDED BY A LATER DOC / FORGOTTEN (no code implements it).
* **Why it matters** -- one line.

Rank by how much damage forgetting it would cause. Be concrete: a wrong number
with a file:line beats a general observation about documentation hygiene.

## What NOT to report

Do not report doc-hygiene opinions, naming, or formatting. Do not report a
finding you cannot anchor to a real `path:line`. If a claim appears in a doc but
you cannot tell whether code implements it, say so explicitly rather than
guessing -- "UNVERIFIED" is a legitimate status and more useful than a wrong one.

## CONSTRAINTS

100% local, open source, offline-first. 16 GB RTX 5080, 14.5 GB real-world
ceiling. The only workflow JSON is `workflows/otr_canonical.json`. Every second
of audio gets original video -- no mirrors, no ping-pong. **Do not launch
renders, do not boot a server, do not edit code.**
