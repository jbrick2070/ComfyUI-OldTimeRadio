# Problem statement: getting a non-coder from zero to an episode

**Date:** 2026-08-22
**Status:** problem statement. No solution is committed here; this is the
argument for which solutions are worth building.

## The problem in one sentence

A person who wants to make a radio episode has to install a node pack, choose
one of 32 dropdown options with no basis for choosing, and assemble a model set
from two entirely different distribution systems -- and nothing in the product
tells them whether they have succeeded until a render fails.

## What we actually know, measured today

Not opinions. These are the facts that should drive the design.

* **The model store is 854 GB in total, and no user needs that.** One lane
  needs far less: Ghost Signal is **two files, 3.9 GB**. The 854 GB is the
  union of every lane ever built. Presenting it as "the models" is the single
  most discouraging thing we could do, and it is not true for anybody.
* **Models arrive by two different mechanisms, and only one is manual.** The
  writer LLM, TTS and music **self-download via HuggingFace hub on first use**
  -- no user action at all. Only the image/video weights are manual files in
  `folder_paths`. So the manual burden is much smaller than the total suggests,
  and we have never said so.
* **`folder_paths` only exists inside a running ComfyUI.** Any check run from a
  terminal reports "not installed" even when the file is on disk. We hit this
  today: the box has all four Ideogram artifacts and a CLI probe still said they
  were missing. **A user debugging their install from a terminal will be lied
  to.** This is the single most important constraint on any tooling we build.
* **Host RAM is a real ceiling and it is undocumented.** The 8 GB clamp run
  peaked at **27.56 GiB of system RAM**. A 16 GiB machine cannot run this
  pipeline regardless of its GPU. No VRAM table warns anyone.
* **Two shipped filenames actively mislead.** `otr_8gb_wan.json` selects an
  engine measured at **12.1 GiB** and `otr_8gb_fastwan.json` one at **12.8
  GiB**. The dropdown names are honest -- both say "high VRAM" -- but a user
  picks the FILE, and the file says 8gb.
* **The dropdown offers 32 options**: 7 are paid cloud services, 2 require
  their own boot lane, 1 is explicitly one-GPU-only, and 1 is not an engine at
  all. A newcomer has no way to know any of that from the menu.
* **Multi-model installs are normal, so our shape is not the problem.** Of the
  522 official ComfyUI templates, 75 declare models and **57 of those declare
  more than one**; the largest declare ten across five directories. A template
  needing two files is unremarkable.

## Why the obvious fixes are not enough

**"Write better install docs."** Docs do not survive contact with a machine
whose state differs from the author's. The Ideogram case above is the proof: a
correct document plus a terminal check produced a confident wrong answer.

**"Ship a one-click template."** Necessary, not sufficient. It gets the
*workflow* onto their canvas; it does not tell them the render will fail in
four minutes because one encoder is missing. And the operator has ruled that we
will **not** dumb the pipeline down to fit a packaging format.

**"Auto-download everything."** Two thirds of it already does. The remaining
third is exactly the part ComfyUI's own `properties.models` standard covers
(`name` / `url` / `directory`), and we have that data with hashes.

## The core insight

**The check must live INSIDE ComfyUI, because that is the only place model
resolution tells the truth.**

Everything else -- documentation, matrices, manifests -- is a map. A map is
worth having, and we now have a good one. But the thing that converts a
frustrated user into a working one is a component that runs where
`folder_paths` is real and says, in their words:

> For the lane you picked you need 2 files. You have 1. Here is the other one,
> here is where it goes, and here is how big it is.

Nothing we can write in Markdown does that job, because Markdown cannot see
their disk.

## What "optimised for a non-coder" would mean concretely

Ranked by how much friction each removes, highest first.

1. **An in-ComfyUI readiness check.** A node (or a startup report) that resolves
   the selected lane's declared `model_requirements` through the live
   `folder_paths` and reports present/missing with the URL and target directory
   for each miss. This is the only item that can be *correct*; the rest are
   documentation.
2. **One recommended starting lane, stated as such.** Not 32 equals. "Start
   here: Ghost Signal -- two files, 3.9 GB, no image model needed." The README
   matrix is half of this; the missing half is a recommendation.
3. **`properties.models` on shipped templates**, so ComfyUI's own download flow
   handles the manual third. We have URLs, sizes and SHA-256 for Ghost already;
   most official templates ship no hash at all.
4. **Rename the two misleading variant files.** Small, and it removes an
   actively harmful signal.
5. **State the host-RAM floor** anywhere the VRAM matrix appears.
6. **A machine-readable model manifest** covering every lane -- URL, category,
   bytes, SHA-256 -- so an AI assistant can execute the install. The operator's
   position is that expecting some know-how is acceptable and AI assistance is
   the modern path; a manifest is what makes that assistance reliable rather
   than improvised.

## What success looks like

A person with a supported GPU and 32 GB of RAM installs ComfyUI, installs this
pack, opens one recommended workflow, and is told **before rendering** exactly
what is missing and where to put it. They fetch two files. They click Run and
get an episode.

Not "it works if you already know". Not "read these six documents first".

## The test that settles it

`docs/2026-08-22-4060-CLEAN-ROOM-TEST.md`. Wipe an independent machine, install
from the README alone, and record where a stranger would get stuck. Until that
has been run, every claim in this document about difficulty is inference --
including mine.
