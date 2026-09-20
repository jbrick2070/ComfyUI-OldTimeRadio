# Apple Silicon

It works, and the short version is that you do not have to do anything special.
Install as in [INSTALL.md](INSTALL.md), open a Mac graph, press Queue. Every
Mac graph has published an episode on a Mac mini M4 / 16 GB:

```
workflows/variants/otr_mac16_low.json          procedural lanes, no weights
workflows/variants/otr_mac16_still.json        sd15 stills with motion
workflows/variants/otr_mac16_video.json        ltx098 video diffusion
workflows/variants/otr_mac16_animatediff.json  SD 1.5 motion from the prompt
```

Start with `otr_mac16_low` if you want the fastest proof that it works at all.
Each is the canonical with Mac-appropriate dropdowns already saved. The
canonical itself also runs — it names no vendor and resolves your device at run
time — but these pick the lanes that have receipts here.

(Each graph is cut from the profile of the same name in `config/profiles/`.
`otr_mac_mps` is a separate lab profile, not the parent of these graphs.)

What runs on a Mac and what does not is the **Mac 16 GB** column in
[MACHINES.md](MACHINES.md). That table is generated from the same data the code
uses, so it is the one to trust; this page is only the things that are different
*because* it is a Mac.

---

## The one warning that matters

**An out-of-memory on Apple Silicon can take the machine down, not just the
render.** Unified memory means the GPU allocation and the system's are the same
pool, so an overcommit does not fail politely the way a discrete card's does —
it can reboot you.

This is why the Mac column is conservative, and why lanes that merely *fit* on a
16 GB NVIDIA card read as **not offered** here. They are not omissions.

## The writer is the memory hog, not the video

Counter-intuitive but consistently true: the lane most likely to push a 16 GB Mac
over is the language model writing the script, not anything that draws pixels.
The Mac graph enforces a **10 GB ceiling** on it for that reason, and ships
`Qwen/Qwen3.5-4B` unquantized.

**Close other applications before a run.** A browser with many tabs is a real
factor on a 16 GB machine, and it is the cheapest thing you can change.

## Images

The Mac still and video graphs mint stills and ship **SD 1.5** for it --
`otr_mac16_still` and `otr_mac16_video` — one 2 GB checkpoint, ungated, no
account, fetched on first use. `otr_mac16_low`, the one to start with, draws its
own frames and never fetches an image model at all. `otr_mac16_animatediff`
renders from the prompt alone, and its SD 1.5 checkpoint is a MANUAL fetch --
a node pack supplies code, not 2 GB of weights. [MACHINES.md](MACHINES.md#where-do-the-manual-weights-come-from)
names the files that lane needs and where each goes.

Z-Image-Turbo is what the NVIDIA graphs use and is a ~19 GB download; leaving it
selected on a Mac with a still-consuming video lane starts that download. If you
did not mean to, that is the usual cause.

## Video

The Mac graph ships **procedural lanes** -- they draw their own frames and
download nothing. Every `still_*` lane is proven here too, plus the
AnimateDiff Lightning lane.

**LTX 0.9.8 is a swap, not a default.** It is real video diffusion and it is
proven on Apple Silicon, but selecting it starts a ~16 GiB download. It is
image-to-video, so it consumes the SD 1.5 still — which the Mac graph already
selects, so that half costs you nothing extra.

## Python and voices

3.12 or earlier runs Kokoro on torch and is the multilingual path. 3.13 runs
English through `kokoro-onnx` on the CPU automatically; this pack does not use
that backend for non-English rows. 3.14 has no Kokoro build and is refused
rather than half working. Details in [INSTALL.md](INSTALL.md#python-versions) and
[MULTILINGUAL.md](MULTILINGUAL.md).

Kokoro is the shipped voice on every platform. The cloning engines need a
Windows-only installer and are not available here.

## Fonts

Captions and credits need a real font. macOS has plenty, so this is normally a
non-issue — unlike Linux, where a headless image often has none.

---

## The receipts behind this page

Every row above is backed by a named run on a physical Mac mini M4 / 16 GB: the
the shipping graphs, the AnimateDiff re-tests, and the MPS decode fix that came
out of them. Two lab notebooks in the GitHub tree hold the evidence —
`docs/MAC_LAB_LOG.md` (harness run ids, per-leg logs, MPS allocation numbers,
tracebacks) and `docs/MAC_PORTABILITY_GUIDE.md` (every measurement and dead end
behind the rows above). Neither ships in a Manager install, and neither needs to:
what you need to run OTR on a Mac is above this line.
