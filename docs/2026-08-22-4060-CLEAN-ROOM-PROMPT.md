# Prompt for the agent running ON the RTX 4060

Paste everything below the line into Codex / Cowork **on the 4060 laptop**.
Do not run it on the 5080.

---

You are running on Jeffrey's **RTX 4060 laptop (8 GB VRAM, ~32 GB RAM)**. This
machine is completely independent of his main 5080 workstation.

Your job is a **CLEAN-ROOM INSTALL TEST**: wipe ComfyUI and all local models,
reinstall from scratch the way a stranger would, and record exactly where a
newcomer gets stuck.

## READ THIS FIRST -- the test is SUPPOSED to fail

A fresh install has no models. **The finding is not "did it render" -- it is
HOW it failed.** A clear message naming which files to fetch and where they go
is a shippable product. An inscrutable stack trace, a silent hang or a black
video is the bug we are hunting.

**Therefore: do NOT repair a clean failure mid-test.** If something is missing,
write down exactly what the software said, then continue. Fixing it as you go
destroys the only result this test produces. You are a witness, not a mechanic.

## Hard boundaries -- do not cross these

- **Only touch this laptop.** Never write to `10.55.0.1`, the 5080, or anything
  reached over the network.
- There is a share on this machine (`4060-TRANSFER`) mapped from the 5080. **Do
  not delete its contents** without checking with Jeffrey -- it may hold the
  only copy of something.
- **Confirm with Jeffrey before deleting anything.** State what you are about to
  remove and roughly how much disk it frees, and wait for a yes.
- No cloud services, no API keys, no paid endpoints. This project is
  offline-first.

## Step 1 -- inventory before you destroy

Record, so the wipe is reversible in principle and the cost is known:

- ComfyUI install path and version.
- Model directory paths and their total size (this is the honest "what does
  this cost a user" number).
- GPU name, total VRAM, total system RAM.

## Step 2 -- wipe

After Jeffrey confirms: remove the ComfyUI install and the local model store.
Report how much disk was freed.

## Step 3 -- fresh ComfyUI

Install ComfyUI the way a new user would -- the official download. **Do not copy
anything from the 5080.** Start it once and confirm it boots to an empty UI
before adding anything.

## Step 4 -- install the node pack

Repo: `https://github.com/jbrick2070/ComfyUI-OldTimeRadio` (branch
`v2.0-alpha`).

**Follow its README literally.** If a step is missing, ambiguous or wrong, that
is a finding -- record it and do the most obvious thing a newcomer would do,
then note what you had to guess.

Restart ComfyUI. Report:

- Do the OTR nodes appear?
- Any red nodes, import errors or missing dependencies? **Quote them verbatim.**
- Does anything tell you which models you need?

## Step 5 -- open a workflow and click Run

In this order. Cheapest first, so a failure is unambiguous:

1. `workflows/variants/otr_8gb_lite.json` -- procgen and stills only, no video
   model. If this fails, the problem is not VRAM.
2. `workflows/variants/otr_ghost_signal.json` -- the smallest video lane: two
   files, 3.9 GB total (`v1-5-pruned-emaonly-fp16.safetensors` in `checkpoints`,
   `mm-p_0.5.pth` in `animatediff_models`). Needs the
   `ComfyUI-AnimateDiff-Evolved` custom node pack.
3. `workflows/variants/otr_8gb_ltx.json` -- `ltx098_low_video`, measured
   6.8 GiB.

**DO NOT open `otr_8gb_wan.json` or `otr_8gb_fastwan.json`.** Their filenames
say 8gb; their engines measure 12.1 and 12.8 GiB. They cannot fit this card.
That mismatch is a known defect, already recorded -- you do not need to
rediscover it.

For each workflow, record:

1. What happened -- rendered, errored, or hung.
2. If it asked for models: **did it name them precisely enough to go and get
   them?** This is the single most important question in the whole test.
3. How long the first run took, including any downloads.
4. Whether anything reached `otr/obs`.

## Step 6 -- report

Write a short report covering:

- Disk freed by the wipe, and disk consumed by the reinstall.
- Every error message, verbatim.
- Every point where you had to guess because the docs did not say.
- Peak VRAM and peak system RAM if you can capture them. **System RAM matters
  here**: an 8 GB clamp run on the 5080 peaked at 27.56 GiB of host RAM, and we
  do not yet know how that behaves on this machine.
- Your honest answer to: *could a non-coder have done this?*

## What we already believe, so you can confirm or destroy it

State plainly if any of these turns out false -- that is more valuable than
agreement:

- The writer LLM, TTS and music **self-download** via HuggingFace on first use;
  the user should not have to fetch them by hand.
- Only the image/video weights are manual files.
- Model resolution goes through ComfyUI's `folder_paths`, so a check run from a
  terminal will report "not installed" even when the file is present. **Trust
  what ComfyUI says in the app, not a CLI probe.**
- Ghost Signal needs no image model at all -- it mints no stills.
