# AMD / ROCm

It ran. On 2026-09-14 an outside tester put `otr_amd_still.json`
through end to end on a Radeon and got a finished episode -- script, cast,
voices, music, stills, captions, credits, muxed. Clean pass, no traceback, no
edits to the graph.

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700 -- 32 GB, **RDNA4 / gfx1201** |
| OS | **Ubuntu 24.04.4 LTS**, in Docker on Unraid |
| ROCm | 7.2 (HIP 7.2.53211) |
| PyTorch | 2.9.1+rocm7.2.4 |
| ComfyUI | 0.35.1 |
| Graph edits | **none** -- loaded and queued exactly as shipped |
| Pack commit | `0b38424` |

One graph ships for AMD, and it is the one that ran:

```
workflows/otr_amd_still.json    still image with motion, Kokoro voices, Stable Audio 3
```

It is the canonical with AMD-appropriate dropdowns already saved: a
Qwen3.5-4B writer unquantised, `still_motion` over Z-Image Turbo stills,
Kokoro, Stable Audio 3. It has not yet run on AMD hardware.

The profile reads `draft`. That field records **promotion**, not proof -- the
graph has a stranger's published episode behind it and the field has not been
moved, which is a separate decision. For what is proven, read the **AMD ROCm**
column in [MACHINES.md](MACHINES.md). That table is generated from the same
data the code uses, so it is the one to trust; this page is only the things
that are different *because* it is AMD.

---

## What one receipt does and does not say

It covers the **still tier on a 32 GB RDNA4 card under Linux**. These engines
carry **proven** in the AMD column, each attested by a file the tester
published rather than by their summary: `viz_mxc_cpu` and `still_motion` from
the per-beat engine ids in their `qa_report.json`, `kokoro` and `z_image_turbo`
named in their `CHANGES.md`. Their music line read "musicgen/stable-audio",
which names two engines, so **no music engine is marked** -- an ambiguous note
is not a receipt. The writer cell is `?` for the same reason: the artifacts
attest engines, not the model that wrote the script.

Still unmeasured, and the honest word for each is **?**: RDNA3, Windows, the
8 GB profile, and every lane past the still tier -- video and AnimateDiff on
AMD are unknown. The engines these profiles select were chosen to be pure
PyTorch and that was verified by reading every one of them; whether ROCm's
kernels agree with them at runtime is a different question, and one card
answering it once is not the same as knowing.

**And it is dated.** The run was at `0b38424`, more than a hundred commits
back, and the shipped graphs have been regenerated since -- the writer lost one
inert widget and was reordered, and other nodes dropped widgets in the
same window. What is proven is that the
PIPELINE and every engine this graph selects run on ROCm -- not that today's
file, byte for byte, has been through a Radeon.

## The one thing that is different

**A ROCm build of torch presents itself as `cuda`.** That is normal and the
pack expects it -- the ROCm profiles declare `device_backend: "cuda"` for
exactly this reason. Everything above torch reads your Radeon as a CUDA device,
which is also why an AMD cell in MACHINES.md can read as *offered* without
anyone having run it: the column answers "is this vendor-locked?", and only a
**proven** cell answers "has this run on AMD?".

Two things follow from it:

* **Check torch again after every `pip install`.** Installing another project's
  requirements can replace your ROCm wheel with a stock PyPI one, and the
  symptom is a card that silently stops being visible.
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  must print `True` before anything else is worth trying.
* **Do not install `sageattention`.** It is CUDA-only and the pack refuses to
  run if it finds it patched in. You do not need `pycairo` either -- no AMD
  graph selects the one engine that wants it.

**Whether `bitsandbytes` quantisation works under ROCm is UNTESTED, not
settled.** The loader used to assume it did not, and that assumption was found
to be a policy rather than a measurement -- it cost an 8 GB AMD card a 2.90 GiB
NF4 load it would have fitted, forcing 8.06 GiB unquantized and missing the
tier by 0.07 GiB for no hardware reason (`nodes/_otr_model_loader.py`). The
installed bitsandbytes advertises cpu, cuda, hpu, mps, triton and xpu backends.

So the files disagree deliberately, and both are right: the SHIPPED graph
`otr_amd_still` asks for `quant_policy: none` and loads the writer at bf16,
because it is the one with a receipt and nothing about it should move. The two
LAB profiles ask for `bnb_nf4` like every other machine and let the runtime
decide. If the probe below reports a working bitsandbytes backend on your card,
the shipped graph is leaving speed on the table -- and that is a finding worth
sending, not a knob to turn mid-render.

## What fetches itself

Nothing on this graph needs a hand fetch. Z-Image Turbo (19.3 GiB) and Stable
Audio 3 download on the first queue; both are ungated -- no account, no licence
click, no token. Stable Audio 3 declares CUDA and Metal in the engine registry,
and ROCm arrives as CUDA, which is how it scored the tester's episode.
About 31 GB in total for a complete episode, of which the image model is
19.3 GiB -- this graph's `still_motion` lane consumes a still, which is what
makes that download live. Every file, with its repository and destination
folder, is in [MACHINES.md](MACHINES.md#where-do-the-manual-weights-come-from).
`scripts/otr_fetch_lane_weights.py --list` shows the lanes,
and `... z_image` / `... stable_audio_3` only save you the wait. All of it
exists only in a GitHub clone -- `scripts/` is not in the registry package.

**ffmpeg and ffprobe, both, 6.1 or newer, on PATH.** This is the one dependency
that fails at the END: the final mux writes the master audio into the MP4 as
PCM, which older builds cannot, so the episode renders in full and then dies
with `Could not find tag for codec pcm_s16le`. The pack refuses in about a
second at the start of a run instead. `winget install Gyan.FFmpeg` on Windows;
`sudo apt install ffmpeg` on Ubuntu 24.04 or newer -- 22.04's apt build is 4.4
and fails, so take a static build there. Check with
`ffmpeg -version; ffprobe -version`.

## Python and voices

Kokoro is the shipped voice on every platform and is what the tester's episode
used. The cloning engines need a Windows-only installer and are not available
here. [INSTALL.md](INSTALL.md#python-versions) has the Python-version rules; nothing
in them is AMD-specific.

---

## Running it on a card the receipt does not cover

RDNA3, Windows and the 8 GB profile are the open cells, and an evening on any
of them is worth having.

| | |
|---|---|
| **GPU** | RDNA3 (7900 XT / XTX, W7900), RDNA4 (RX 9070 / 9070 XT, and the R9700 that ran), or MI-series. **Check your card's gfx target, not its tier** -- ROCm grants support per target, so two cards of the same generation can differ: gfx1100 (7900 XT / XTX / GRE) has been supported for years while gfx1102 (7600 XT) waited until 7.14. Look yours up in AMD's [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) before buying for this. RDNA2 is the ragged edge: gfx1030 appears via the PRO W6800, while the consumer gfx1031 (6700 XT) and gfx1032 (6600) reach it only through community builds |
| **VRAM** | 16 GB for the shipped graph, 8 GB for the small lab profile |
| **OS** | Linux has the receipt. **Windows is untried** and AMD ships a ROCm build of PyTorch for Radeon for it |
| **ROCm** | 7.2.x is what ran; 6.x is still fine on Linux |
| **Disk** | About 31 GB of weights, so 50 GB free with working room |
| **Time** | An hour, most of it downloads |

A rented box works fine -- this is a couple of dollars of cloud GPU, not a
hardware purchase.

Five minutes first, before any download. **Install a ROCm torch before you run
the probe** -- it checks torch and prints nothing useful without one. On
Windows take AMD's PyTorch for Radeon from the
[ROCm on Radeon guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html)
rather than the pip line:

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio custom_nodes/ComfyUI-OldTimeRadio
python custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_rocm_probe.py
```

It prints a short receipt and stops -- no weights, no render, nothing written
outside the checkout. These lines decide whether the rest is worth your
time: `vendor()` must come back `amd` (if it says `nvidia` or `unknown` the pack
cannot tell your card from a GeForce and that is a bug to fix first);
`is_amd()` is what device resolution branches on, and it has executed on one
Radeon, once; and if `bitsandbytes` imports with a working backend, every AMD
graph is leaving speed on the table by refusing to quantise, which nobody can
discover without your card.

Then the episode. Match the wheel index to your ROCm -- the tester's 7.2 box
used `2.9.1+rocm7.2.4`; the 6.x index below is the one that can be named with
confidence, so take the matching `--index-url` from
[pytorch.org](https://pytorch.org/get-started/locally/) for 7.2:

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
pip install -r requirements.txt
pip install -r custom_nodes/ComfyUI-OldTimeRadio/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True, and YOUR card's name
python main.py --listen 127.0.0.1 --port 8188
```

In a second terminal, one act, no timeout:

```bash
python custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_canonical_api_run.py \
  --comfyui-url http://127.0.0.1:8188 \
  --workflow custom_nodes/ComfyUI-OldTimeRadio/workflows/otr_amd_still.json \
  --act-count 1 \
  --timeout 0
ls -la output/otr/obs/
```

**Use `--workflow`, not `--machine amd`** -- that shortcut expands to an
unaudited in-memory bundle, not the graph this page is about.

Success is an mp4 in `output/otr/obs/`. Win or lose, an issue titled
`ROCm: <your card>` with the leg log, the ComfyUI terminal, the ledger
(`output/otr/episodes/<episode>/episode_canon.json`), `rocm-smi`, the torch
version line, your card, your ROCm version and your distro, and the first
traceback with everything above it, is the whole ask. Send the mp4 too if you
got one -- we would like to hear it. A failure in the first ninety seconds is a
result. Send that.

One trap worth knowing before you spend the evening: **an 8 GB card is tight
even on NVIDIA**, so a 16 GB Radeon is the better first test.

---

## The receipts behind this page

Commit `0fc0fb90` records the run above from the tester's own artifacts, marks
the proven engines in `apple/dropdown_matrix.json`, and fixes the bug they
hit: `scripts/otr_fetch_lane_weights.py` resolved the models directory through
an import that only works inside a running ComfyUI, swallowed the failure, and
returned a hardcoded Windows path -- on Ubuntu. An explicit
`OTR_COMFYUI_MODELS_ROOT` now wins outright, the ordinary ComfyUI layout is
detected, and a real failure says so. The tester's full report is at
`drearburh.uk/otr-amd-report/`; the drafted reply and the recruiting post are
under `docs/` in the GitHub tree, which does not ship in a Manager install and
does not need to. What you need to run OTR on a Radeon is above this line.
