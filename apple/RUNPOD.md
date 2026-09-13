# Renting a GPU

For the lanes your own card cannot hold. Written for RunPod, but nothing here is
RunPod-specific beyond the paths.

This needs the **git clone**, not a Manager install — the provisioning script is
not in the registry bundle.

---

## 1. Size the pod

| What you want to run | Start with |
|---|---|
| The default AnimateDiff episode path | 8 GB NVIDIA |
| HuMo 14B | 16 GB+ NVIDIA, 32 GB+ host RAM |
| LTX 2.5 at the shipped 1664x960 | 32–48 GB NVIDIA, 100 GiB+ cgroup RAM — take a 48 GB L40S first |

A 24 GB card reached the decode stage on LTX 2.5 and ran out of GPU memory there,
so it is not the cheap answer it looks like.

For the heavy lanes, also give yourself:

- **100 GiB effective cgroup RAM** and **150 GiB free** on whichever filesystem
  holds the models;
- a 200 GB container disk, or a network volume under `/workspace` with the model
  tree on it;
- ports 8188 (ComfyUI) and SSH.

Those figures cover the whole stack — weights, writer cache, isolated voice
runtime, page cache and output room. No single engine wants 100 GiB by itself.

## 2. Provision

One bootstrap does everything: finds the template's real ComfyUI tree, pins
ComfyUI core and the partner packs, repairs the CUDA mismatch these images
usually have, clones or fast-forwards OTR from `v2.0-alpha`, downloads the
automatic lanes, warms the writer, verifies the manual tiers, and prints one
receipt.

For automatic selection by VRAM, clear the overrides first:

```bash
unset OTR_PROVISION_PROFILE OTR_PROVISION_MACHINE OTR_WITH_INDEXTTS2
```

Or force a specific row instead of letting it detect:

```bash
export OTR_PROVISION_MACHINE=16gb
```

The `--machine` keys are the same ones in [MACHINES.md](MACHINES.md).

Read the receipt it prints before launching anything. It is the difference
between "the weights are there" and "the weights were attempted."

## 3. Weights that need a licence click

Some of the heavy video lanes are gated. Accept the licence on each model's
Hugging Face page while signed in, then `hf auth login` on the pod. **More than
one owner may be involved** — a lane can need clicks on two different accounts,
so do not assume one acceptance covers the set.

Which files, from which repository, into which folder:
[MACHINES.md](MACHINES.md) section 3.

## 4. Launch and prove it

Start ComfyUI, load `workflows/otr_canonical.json`, set the dropdowns for the
lane you rented the box for, and queue one episode.

**The proof is a file in `otr/obs/`.** Not a green log, not a finished queue — a
published episode. If more than five minutes have passed with nothing there, stop
waiting and read the log.

## 5. Pull your work off before you stop the pod

Copy `otr/obs/` down first. Then stop it. A stopped pod's disk is not a place to
keep anything you want.

---

## Two things worth knowing

**The registry install works now.** Older notes say ComfyUI-Manager cannot
install this pack reliably; that stopped being true at `2.0.0-alpha.30`. On a pod
you still want the git clone, but for the reason above — the scripts — not
because Manager is broken.

**Do not put licensed local-only weights on rented hardware.** If a model came to
you under terms that assume your own machine, a rented box is not your own
machine.

---

`docs/RUNPOD_INSTALL.md` in the GitHub tree carries the rest: the failure atlas,
the unattended sweep and soak procedure, driving a pod from a second machine, and
the evidence ledger. It does not ship in a Manager install.
