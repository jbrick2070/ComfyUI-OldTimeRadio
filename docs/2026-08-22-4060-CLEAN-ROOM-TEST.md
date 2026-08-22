# The 4060 clean-room test -- can a stranger actually run this?

**Date:** 2026-08-22
**Machine:** the RTX 4060 (8 GB), reachable at `10.55.0.2`. **Completely
independent of the 5080** -- its own ComfyUI, its own models. Wiping it does not
touch `C:\ComfyUI-Models` on the main box.

## What this test is FOR

Every other test in this repo runs on a machine that already works. This one
asks the only question an open-source release actually turns on: **a stranger
installs ComfyUI, installs OTR, opens a workflow, clicks Run -- what happens?**

**IT IS SUPPOSED TO FAIL, AND THE FAILURE IS THE RESULT.** A fresh install has
no models. The finding is not "did it render" -- it is *how* it failed:

* a clear, actionable message naming exactly which files to fetch and where; or
* an inscrutable stack trace, a silent hang, or a black episode.

The first is a shippable product. The second is the bug. Do not "fix" a clean
failure by installing something mid-test -- write down what it said, then
continue.

## Before you wipe

- [ ] Confirm nothing on the 4060 is needed elsewhere. It is independent, but
      the `U:` share (`\\10.55.0.2\4060-TRANSFER`) is mapped from the 5080 --
      make sure nothing you want is sitting only there.
- [ ] Note the current ComfyUI version if you can, so "fresh web install" can be
      compared against what was there.

## The run

### 1. Nuke

- [ ] Remove the ComfyUI install.
- [ ] Remove the model store completely.
- [ ] Record roughly how much disk that freed -- it is the honest "what does
      this cost a user" number.

### 2. Fresh ComfyUI, from the web

- [ ] Install ComfyUI the way a new user would -- the official download, not a
      copy from the 5080.
- [ ] Start it once and confirm it boots to an empty UI before adding anything.

### 3. Fresh OTR

- [ ] Install this node pack the way the README says. **Follow the README
      literally.** If a step is missing or wrong, that is a finding -- record it
      rather than working around it from memory.
- [ ] Restart ComfyUI. Confirm the OTR nodes appear.
- [ ] **Record any red node, missing dependency or import error verbatim.**

### 4. Open a 4060-friendly workflow and click Run

Use these, in this order -- cheapest first, so a failure is unambiguous:

| Order | File | Why |
|---|---|---|
| 1 | `workflows/variants/otr_8gb_lite.json` | procgen + stills only; no video model at all. If THIS fails, the problem is not VRAM. |
| 2 | `workflows/variants/otr_ghost_signal.json` | smallest video lane you own -- 3.9 GB of weights (SD1.5 + `mm-p_0.5.pth`) at 512x288. |
| 3 | `workflows/variants/otr_8gb_ltx.json` | `ltx098_low_video`, measured 6.8 GiB. |

- [ ] For each: open it, click Run, and record exactly what happens.

**DO NOT open `otr_8gb_wan.json` or `otr_8gb_fastwan.json` on this box.** Their
names say 8 GB and their engines measure **12.1 GiB** and **12.8 GiB**. They
will not fit, and the filename is the reason a stranger would try. Fixing those
two names is a separate task this test justifies.

## What to write down

For each of the three workflows:

1. Did it render? If not, what EXACTLY did it say -- copy the text.
2. If it asked for models: did it name them precisely enough to go and get them?
3. How long did the first run take, including any downloads?
4. Did anything reach `otr/obs`?

## What "good" looks like

Not "it rendered". A fresh install with no models CANNOT render, and pretending
otherwise is how a project ships an install guide nobody can follow. Good is:

* the missing pieces are named exactly, with where they go;
* nothing crashes with a bare stack trace;
* the node pack loads clean even with zero models present;
* and once the named models are fetched, it renders without further guesswork.

## Known-suspect areas going in

Recorded now so the test can confirm or clear them, rather than being read into
the results afterwards:

* **The `8gb` variant names are wrong for two of four files** (above). Already
  known, listed here so it is not "discovered" twice.
* **Model resolution goes through `folder_paths`**, which only exists inside a
  running ComfyUI. Any preflight run outside it reports "not installed" even
  when the weights are on disk -- so trust the in-app message, not a CLI probe.
* **Two lanes need their own boot** (`h3_low_video`, `h3_low_audio_in`). They
  are not in this test for that reason.
* **`ltx25_high_video` is 5080-only** by its own label. Not in this test.
