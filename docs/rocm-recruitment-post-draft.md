# Draft r/ROCm post -- for Jeffrey to post, or not

**A window must never post this.** It is written to be pasted by you, edited
however you like. Two versions below: a short one and a longer one. Reddit's
r/ROCm skews toward people who enjoy making things work on hardware everyone
else ignored, so both lead with the honest gap rather than with a pitch.

Flair: whatever the sub uses for "help wanted" or "project". The image to
attach is `docs/images/rocm_mission_hero.jpg`.

---

## Short version

**Title:** Has anyone run a full local text-to-video pipeline on ROCm? I have
two AMD profiles nobody has ever been able to test.

> I maintain an open-source ComfyUI pack that writes and renders complete
> old-time-radio episodes locally -- script, cast, voices, music, pictures,
> video, credits, no cloud, no API keys. It runs on NVIDIA 16 GB and 8 GB, on
> Apple Silicon, and on rented Linux boxes.
>
> It has never run on ROCm, for the boring reason that I do not own an AMD
> card.
>
> There are two AMD profiles in the repo, built and generated the same way as
> every working profile. Every engine they select is plain PyTorch: no
> sageattention, no flash-attn, no bitsandbytes, no fp8, no custom CUDA
> kernels. On paper they should just work. On paper.
>
> If anyone here has an MI-series or RDNA3 card and an hour, there is a
> step-by-step in the repo (apple/ROCM.md) -- six commands, about
> 20 GB of weights, and success is an mp4 you can actually watch. A traceback
> is just as useful to me as a success; either way you would be the first
> person to point this thing at an AMD GPU, and I will credit you in the
> commit that turns two draft profiles into a supported platform.
>
> Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
> `v2.0-alpha`)

---

## Longer version, if the sub prefers detail

**Title:** Two ROCm profiles, zero ROCm testers: looking for someone to try a
local AI radio-drama pipeline on AMD

> **What it is.** An open-source ComfyUI node pack that generates a whole
> old-time-radio episode end to end on one GPU, offline: it writes the script
> with a local LLM, casts the characters, speaks the lines with a local TTS,
> composes and renders the music, generates stills, animates them, burns
> captions and credits, and muxes the lot into an mp4. No cloud services, no
> API keys, no subscriptions.
>
> **Where it runs today.** NVIDIA 16 GB (an RTX 5080 is the dev box), NVIDIA
> 8 GB, Apple Silicon, and rented Linux boxes. Published episodes from all of
> those.
>
> **Where it has never run.** ROCm. Not once. I do not own an AMD card, and
> that is the entire reason -- not a technical objection, not a "we tried and
> it broke". Nobody has ever pointed it at one.
>
> **What is already prepared for you.** Two profiles, `otr_amd16_rocm` and
> `otr_amd8_rocm`, generated from the same source as every working profile
> and marked `draft` / UNVERIFIED because that is literally the status. Each
> pins a deliberately conservative engine set, and I went through every one of
> them to confirm it is plain PyTorch:
>
> * video: a still-motion lane (numpy/PIL, no diffusion video)
> * images: Z-Image Turbo at bf16 (stock ComfyUI loaders, no fp8/nvfp4/GGUF)
> * voices: Kokoro
> * music: Stable Audio 3 on the 16 GB profile, MusicGen on the 8 GB one
> * writer: a transformers model, no bitsandbytes, no llama.cpp needed
> * sage-attention, flash-attn, fp8 and cuda-malloc: all off
>
> A ROCm torch build presents as `cuda` to everything above it, which the
> pack expects. The one thing to avoid is installing `sageattention` -- it is
> CUDA-only and the pack refuses to run if it finds it patched in.
>
> **What I am asking for.** An hour on an MI-series or RDNA3 card (16 GB for
> the full profile, 8 GB for the small one), ROCm 6.x, about 50 GB of disk.
> A rented box is fine and cheap. The repo has `apple/ROCM.md`
> with six commands start to finish and a list of exactly what to send back.
>
> **A traceback is a win.** If it dies ninety seconds in, that failure is the
> first real ROCm signal this project has ever had and I will act on it. If it
> produces an episode, you are the first person in the world to have made one
> on AMD hardware, and I will say so in the repo.
>
> Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
> `v2.0-alpha`). Happy to answer anything here.

---

## Notes for you before posting

* Check r/ROCm's rules on self-promotion; some hardware subs want a
  "[Project]" or "[Help]" tag, and some prefer no link in the title.
* The repo is MIT-ish open source and the ask is genuine testing, not
  traffic, which is usually the distinction those rules care about.
* If r/ROCm is quiet, r/LocalLLaMA and r/StableDiffusion both have AMD
  users who enjoy exactly this kind of "nobody has tried it" problem.
* Expect the first reply to be "which card" and the second to be "does it
  need flash-attn". Both answers are in the mission file.
