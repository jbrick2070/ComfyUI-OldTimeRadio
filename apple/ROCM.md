# AMD / ROCm

**AMD ships EXPERIMENTAL in v2.0: one graph, no receipts. This page is the map,
and the first episode off a Radeon is what makes it v2.1.**

This pack writes a full old-time-radio episode -- script, cast, voices, music,
pictures, video, credits -- locally, offline, on one GPU. It has run on NVIDIA
(16 GB and 8 GB), on Apple Silicon, and on a rented Linux box. It has **never
once run on ROCm**, for a boring reason: nobody on the project owns an AMD card,
and we were not going to claim a platform we could not put an episode through.

So AMD ships EXPERIMENTAL in v2.0 -- one graph, no receipts, while the other
fifteen shipping graphs each carry an episode. **It becomes v2.1 the day one
of them runs on a Radeon**, and that is not a promise we can keep by
ourselves: nobody here has the card.

**It is open source, and it is fair game.** If you have the card, everything
needed to try is already in the repo and nothing is waiting on us:

* **A graph is built and shipped** -- `workflows/variants/otr_amd_still.json`,
  generated from `config/profiles/otr_amd_still.json` the same way as every
  working graph and marked `draft` because it has no receipts. It is the
  still-image tier: Qwen3.5-4B writer (unquantised), `still_motion` over
  Z-Image Turbo stills, Kokoro voices, MusicGen. Two older lab profiles,
  `otr_amd16_rocm` and `otr_amd8_rocm`, still load with `--profile` for a 16 GB
  or 8 GB variant of the same idea. Every engine any of them selects is plain
  PyTorch: no sageattention, no flash-attn, no bitsandbytes, no fp8, no
  CUDA-only GGUF kernels, no custom CUDA at all. On paper they should work. On
  paper is exactly the problem.
* **A five-minute probe exists** that answers most of the open questions without
  downloading a model or rendering anything.
* **The unknowns are written down** rather than hand-waved -- see the end of this
  page.

If you get an episode out of one, you are the first person to do it, and the
credit is yours in this repo. If you get a traceback instead, that is worth just
as much: it is the first real ROCm signal this project has ever had. Either way
an issue with the probe output pasted in is the whole ask.

---

## What you need

| | |
|---|---|
| **OS** | **Windows first** -- the pack is Windows-native and AMD ships a ROCm build of PyTorch for Radeon on Windows (ROCm 7.2.x, one installer for both OSes) -- or Linux. Neither has a receipt here |
| **GPU** | One AMD card: RDNA3 (7900 XT / XTX, W7900), RDNA4 (RX 9070 / 9070 XT, official in ROCm 7.2), or MI-series. RDNA2 may work; nobody knows |
| **VRAM** | 16 GB for the full profile, 8 GB for the small one |
| **ROCm** | 7.2.x (Windows or Linux); 6.x still fine on Linux |
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
refuses in about a second at the start of a run instead -- but only if both
binaries are reachable. `winget install Gyan.FFmpeg` on Windows;
`sudo apt install ffmpeg` on Ubuntu 24.04+ (22.04's apt build is 4.4 and FAILS,
so take a static build there). Check with `ffmpeg -version; ffprobe -version`.

**And every step here wants the GitHub clone, not a Manager install** --
`scripts/` is not in the registry package, so the probe and the runners below
exist only if you cloned the repository.

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio custom_nodes/ComfyUI-OldTimeRadio
python custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_rocm_probe.py
```

It prints about thirty lines and stops. No weights, no render, nothing written
outside the checkout. Paste the output into the issue and you are done -- that
alone is more AMD evidence than this project has ever had.

**Three of those lines decide whether the rest of the mission is even worth your
time.** `is_amd()` is what our device resolution branches on and has never once
executed on an AMD card. `vendor()` must come back `amd`; if it says `nvidia` or
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
cd ComfyUI-OldTimeRadio && git checkout v2.0-alpha
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

The shipped graph needs none of this by hand: Z-Image Turbo and MusicGen fetch
themselves at the first queue, so the `z_image` line only saves you the wait.
The `stable_audio_3` line is for the 16 GB lab profile alone; the shipped graph
and the 8 GB profile are on MusicGen.

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

* **Nothing here has ever touched ROCm.** Both profiles are marked `draft`
  and `UNVERIFIED on hardware` in their own files. That is not modesty, it is
  the literal status.
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
turns two `draft` profiles into a supported platform, and the first AMD-made
episode of a show that did not previously exist on your hardware.

*This message will not self-destruct. It will sit here until somebody with an
AMD card gets curious.*
