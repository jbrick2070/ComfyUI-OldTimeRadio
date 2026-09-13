# Apple Silicon

It works, and the short version is that you do not have to do anything special.
Install as in [INSTALL.md](INSTALL.md), open the Mac graph, press Queue.

```
workflows/variants/otr_mac_mps.json
```

That graph is the canonical with Mac-appropriate dropdowns already saved. The
canonical itself also runs — it names no vendor and resolves your device at run
time — but the Mac variant picks the lanes that have receipts here.

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

The Mac graph ships **SD 1.5**, which downloads itself on first use — one 2 GB
checkpoint, ungated, no account.

Z-Image-Turbo is what the NVIDIA graphs use and is a ~19 GB download; leaving it
selected on a Mac with a still-consuming video lane starts that download. If you
did not mean to, that is the usual cause.

## Video

The Mac graph ships **three procedural lanes** — they draw their own frames and
download nothing. All four `still_*` lanes are proven here too, plus the
AnimateDiff Lightning lane.

**LTX 0.9.8 is a swap, not a default.** It is real video diffusion and it is
proven on Apple Silicon, but selecting it starts a ~16 GiB download. It is
image-to-video, so it consumes the SD 1.5 still — which the Mac graph already
selects, so that half costs you nothing extra.

## Python and voices

3.12 or earlier runs Kokoro on torch; 3.13 runs it through `kokoro-onnx` on the
CPU automatically; 3.14 has no Kokoro build and is refused rather than half
working. Details in [INSTALL.md](INSTALL.md) section 5.

Kokoro is the shipped voice on every platform. The cloning engines need a
Windows-only installer and are not available here.

## Fonts

Captions and credits need a real font. macOS has plenty, so this is normally a
non-issue — unlike Linux, where a headless image often has none.

---

## Going deeper

`docs/MAC_PORTABILITY_GUIDE.md` in the GitHub tree is the full record — every
measurement, every dead end, and the reasoning behind each row above. It is a lab
notebook rather than a guide, which is why the useful half is here instead. It
does not ship in a Manager install; read it on GitHub.
