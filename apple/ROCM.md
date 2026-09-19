# AMD / ROCm

**IT RAN. 2026-09-14: the first full episode off a Radeon, clean pass, no
traceback.** An outside tester put `workflows/variants/otr_amd_still.json`
through end to end and it produced a finished episode -- script, cast, voices,
music, stills, captions, credits, muxed.

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700 -- 32 GB, **RDNA4 / gfx1201** |
| OS | **Ubuntu 24.04.4 LTS**, in Docker on Unraid |
| ROCm | 7.2 (HIP 7.2.53211) |
| PyTorch | 2.9.1+rocm7.2.4 |
| ComfyUI | 0.35.1 |
| Graph edits needed | **none** -- loaded and queued exactly as shipped |
| Pack commit | `0b38424` |

**Two things about that surprised us, and both are corrections to this page.**
It landed on **Linux**, where this page had assumed Windows would be first. And
it landed on **RDNA4**, which the table below has always listed but which the
public call for testers did not mention -- so the ask was filtering out the one
architecture that now has a receipt.

**WHAT IS AND IS NOT PROVEN.** One receipt is one receipt. It covers the STILL
tier on a 32 GB RDNA4 card under Linux. It does not cover RDNA3, Windows, the
8 GB profile, or any lane past still-image -- the video and AnimateDiff lanes on
AMD remain unknown.

**AND IT IS DATED.** The run was at `0b38424`. The shipping graphs have been
regenerated since (the widget cleanup removed ten controls and reordered the
writer), so what is proven is that the PIPELINE and every engine this graph
selects run on ROCm -- not that this exact file, byte for byte, has been through
a Radeon. That is an honest distinction and worth keeping until someone re-runs
it.

**What the tester also found**, and it is fixed: `scripts/otr_fetch_lane_weights.py`
resolved the models directory through an import that only works inside a running
ComfyUI, swallowed the failure, and silently returned a hardcoded Windows path.
On Ubuntu. An explicit `OTR_COMFYUI_MODELS_ROOT` now wins outright, the ordinary
ComfyUI layout is detected automatically, and a genuine failure says so instead
of guessing.

`bitsandbytes` has no ROCm build and is not needed: the loader degrades to bf16
with a warning, which this run confirmed in the wild rather than in theory.

**It is open source, and it is fair game.** If you have the card, everything
needed to try is already in the repo and nothing is waiting on us:

* **A graph is built and shipped** -- `workflows/variants/otr_amd_still.json`,
  generated from `config/profiles/otr_amd_still.json` the same way as every
  working graph. It reads `draft` because `status` records PROMOTION, not
  proof -- an outside tester published an episode from this exact graph on
  2026-09-14. It is the
  still-image tier: Qwen3.5-4B writer (unquantised), `still_motion` over
  Z-Image Turbo stills, Kokoro voices, Stable Audio 3. Two older lab profiles,
  `otr_amd16_rocm` and `otr_amd8_rocm`, still load with `--profile` for a 16 GB
  or 8 GB variant of the same idea. Every engine any of them selects is plain
  PyTorch: no sageattention, no flash-attn, no bitsandbytes, no fp8, no
  CUDA-only GGUF kernels, no custom CUDA at all. On paper they should work. On
  paper is exactly the problem.
* **A five-minute probe exists** that answers most of the open questions without
  downloading a model or rendering anything.
* **The unknowns are written down** rather than hand-waved -- see the end of this
  page.

One person has done this: a Radeon AI PRO R9700 (RDNA4) under ROCm 7.2 on
Ubuntu 24.04 published an episode from `otr_amd_still` on 2026-09-14, with no
edits to the graph. What is still unclaimed is everything that run did not
cover -- RDNA3, Windows, the 8 GB profile, and every lane past the still tier.
A traceback is worth as much as a success there. Either way an issue with the
probe output pasted in is the whole ask.

---

## What you need

| | |
|---|---|
| **OS** | **Windows first** -- the pack is Windows-native and AMD ships a ROCm build of PyTorch for Radeon on Windows (ROCm 7.2.x, one installer for both OSes) -- or Linux. Neither has a receipt here |
| **GPU** | One AMD card: RDNA3 (7900 XT / XTX, W7900), RDNA4 (RX 9070 / 9070 XT, official in ROCm 7.2), or MI-series. RDNA2 may work; nobody knows |
| **VRAM** | 16 GB for the full profile, 8 GB for the small one |
| **ROCm** | 7.2.x (Windows or Linux); 6.x still fine on Linux. The pip lines below show the 6.x wheel index -- match it to what you install |
| **Disk** | About 31 GB of weights, so 50 GB free with working room |
| **FFmpeg** | **And ffprobe -- both binaries, 6.1 or newer.** The one dependency that fails LATE; see the step below |
| **Time** | An hour, most of it downloads |

A rented box works fine. This is a couple of dollars of cloud GPU, not a
hardware purchase.

---

## Before any of that: five minutes that might save you an evening

**You do not have to commit to the whole mission to help.** Most of what we
need to know is answerable without downloading a single model or rendering
anything. Install a ROCm torch -- on Windows, AMD's PyTorch for Radeon from the
[ROCm on Radeon guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html)
instead of the pip line below; on Linux, the pip line -- clone this pack, and
run one file:

**Put FFmpeg and ffprobe on PATH first -- both binaries.** This is the one
dependency that fails at the END: the final mux copies the master audio into the
MP4 as PCM, which builds older than 6.1 cannot write, so the episode renders in
full and then dies with `Could not find tag for codec pcm_s16le`. The pack
refuses in about a second at the start of a run instead. That check is
ffmpeg's; ffprobe is wanted for the optional engines and for duration reads
where PyAV is not enough. `winget install Gyan.FFmpeg` on Windows;
`sudo apt install ffmpeg` on Ubuntu 24.04+ (22.04's apt build is 4.4 and FAILS,
so take a static build there). Check with `ffmpeg -version; ffprobe -version`.

**And every step here wants the GitHub clone, not a Manager install** --
`scripts/` is not in the registry package, so the probe and the runners below
exist only if you cloned the repository.

**Match the wheel index to YOUR ROCm.** The line below is the 6.x index,
which is the one we can name with confidence; on 7.2 take the matching
`--index-url` from [pytorch.org](https://pytorch.org/get-started/locally/)
instead. Nobody here has run either, so the URL is yours to confirm, not ours
to promise.

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio custom_nodes/ComfyUI-OldTimeRadio
python custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_rocm_probe.py
```

It prints about thirty lines and stops. No weights, no render, nothing written
outside the checkout. Paste the output into the issue and you are done -- on a
card or an OS the 2026-09-14 run did not cover, that alone is new evidence.

**Three of those lines decide whether the rest of the mission is even worth your
time.** `is_amd()` is what our device resolution branches on; it executed on a
Radeon for the first time on 2026-09-14, and on one card only. `vendor()` must come back `amd`; if it says `nvidia` or
`unknown`, the pack cannot tell your card apart from a GeForce and we have a bug
to fix before you spend an evening rendering. And if `bitsandbytes` imports with
a working backend, then every AMD graph we ship is leaving speed on the table by
refusing to quantise -- which is a good problem, and one nobody can discover
without your card.

If those come back sane, the full mission below is worth it. If they do not, you
have saved yourself the evening and taught us more than a failed render would.

## The mission, in six commands

**1. PyTorch for ROCm first.** A ROCm build of torch presents itself as
`cuda` to everything above it -- that is normal and the pack expects it. On
Windows, install AMD's PyTorch for Radeon per the
[ROCm on Radeon guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html)
and skip the pip line; on Linux:

**Match the wheel index to YOUR ROCm.** The line below is the 6.x index,
which is the one we can name with confidence; on 7.2 take the matching
`--index-url` from [pytorch.org](https://pytorch.org/get-started/locally/)
instead. Nobody here has run either, so the URL is yours to confirm, not ours
to promise.

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That line must print `True` and your card's name before anything else is worth
trying.

**2. ComfyUI, then this pack inside it.**

```bash
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
pip install -r requirements.txt
cd custom_nodes
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
cd ComfyUI-OldTimeRadio
pip install -r requirements.txt
```

**Check torch again before going further.** Installing another project's
requirements can replace your ROCm build with a stock PyPI wheel, and the
symptom is a card that silently stops being visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that no longer prints `True`, reinstall the ROCm wheels from step 1 before
continuing. Everything after this point assumes it is still True.

**Do not install `sageattention`.** It is CUDA-only and the pack actively
refuses to run if it finds it patched in. You do not need `pycairo` either --
no AMD graph or profile selects the one engine that wants it.

**3. Fetch the weights.** About 31 GB for a complete episode -- the image model is 19.26 GB of that, because this graph's `still_motion` lanes consume a still and therefore make it live. Every file, with its repository and destination folder, is in [MACHINES.md](MACHINES.md) section 3.

```bash
python scripts/otr_fetch_lane_weights.py --list
python scripts/otr_fetch_lane_weights.py z_image
python scripts/otr_fetch_lane_weights.py stable_audio_3
```

The shipped graph needs none of this by hand: Z-Image Turbo and Stable Audio 3
fetch themselves at the first queue (the music weights are ungated -- no
account, no licence click, no token), so both lines above only save you the
wait. Stable Audio 3 is what the AMD graph and every other shipped graph score
with; only the CPU preset is on MusicGen, because Stable Audio 3 declares CUDA
and Metal only.

**4. Start ComfyUI.**

```bash
cd ../.. && python main.py --listen 127.0.0.1 --port 8188
```

**5. Run one episode, headless.** This is the mission. The shipped graph is
the one to try first on a 16 GB+ card:

```bash
python custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_canonical_api_run.py \
  --comfyui-url http://127.0.0.1:8188 \
  --workflow custom_nodes/ComfyUI-OldTimeRadio/workflows/variants/otr_amd_still.json \
  --act-count 1 \
  --timeout 0
```

The two lab profiles are the alternates: `--profile otr_amd16_rocm` instead of
`--workflow` for the 16 GB lab preset, `--profile otr_amd8_rocm` for an 8 GB
card.

**Use `--workflow` or `--profile`, not `--machine amd`.** The `--machine amd`
shortcut resolves to a different, unaudited engine set. The graph and the two
profile ids above are the ones this file is about.

**6. Success looks like an mp4.**

```bash
ls -la output/otr/obs/
```

One file, a few minutes long, with a title card, voices, music and moving
pictures. Play it. If it plays, you won.

---

## What to send back

Win or lose, this is the payload. Please include all of it:

1. **The leg log** -- everything the command in step 5 printed.
2. **The ComfyUI server log** -- the terminal from step 4, whole.
3. **The ledger**: `output/otr/episodes/<episode>/episode_canon.json`
4. `rocm-smi` output, and `python -c "import torch; print(torch.__version__)"`
5. **The first traceback**, if there was one, with everything above it. The
   first failure is worth more than a summary of five.
6. Your card, your ROCm version, your distro.
7. The mp4 itself, if you got one. We would love to hear it.

Open an issue on the repo titled `ROCm: <your card>` and paste it in.

---

## What is genuinely unknown

Being straight with you, because you are the one spending the time:

* **One card, one tier, one day.** `otr_amd_still` has a published episode
  from 2026-09-14; the two lab profiles `otr_amd16_rocm` and `otr_amd8_rocm`
  still read `UNVERIFIED on hardware` in their own files and genuinely are.
  Everything past the still tier on AMD remains unmeasured.
* The engines were chosen to be pure PyTorch, and that was verified by
  reading every one of them. Whether ROCm's kernels agree with them at
  runtime is exactly what nobody knows.
* The 8 GB profile is tight even on NVIDIA. If one of them works, the 16 GB
  one is the likelier.
* `llama-cpp-python` needs a HIP build if you want the GGUF writer lane. The
  profiles do not use it -- they pin a transformers model -- so skip it.

If it fails in the first ninety seconds, that is still a result. Send it.

---

## What you get

Your name and card in this repo's hardware-support notes, in the commit that
widens AMD past one card and one tier, and an AMD-made
episode of a show that did not previously exist on your hardware.

*This message will not self-destruct. It will sit here until somebody with an
AMD card gets curious.*
