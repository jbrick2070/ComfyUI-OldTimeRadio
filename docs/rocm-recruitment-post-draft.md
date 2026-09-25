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
an AMD graph nobody has ever been able to test.

> I maintain an open-source ComfyUI pack that writes and renders complete
> old-time-radio episodes locally -- script, cast, voices, music, pictures,
> video, credits, no cloud, no API keys. It runs on NVIDIA 16 GB and 8 GB, on
> Apple Silicon, and on rented Linux boxes.
>
> It has never run on ROCm, for the boring reason that I do not own an AMD
> card.
>
> There is a generated AMD graph in the repo (`otr_amd_still.json`, the
> still-image tier, plus two lab profiles for 16 GB and 8 GB cards), built the
> same way as every working graph. Every engine it selects is plain PyTorch: no
> sageattention, no flash-attn, no bitsandbytes, no fp8, no custom CUDA
> kernels. On paper it should just work. On paper.
>
> If anyone here has an RDNA3, RDNA4 or MI-series card -- Windows first, Linux
> welcome -- and an hour, there is a step-by-step in the repo (apple/ROCM.md) -- six commands, about
> 20 GB of weights, and success is an mp4 you can actually watch. A traceback
> is just as useful to me as a success; either way you would be the first
> person to point this thing at an AMD GPU, and I will credit you in the
> commit that turns the draft graph into a supported platform -- the AMD
> column in the machine table is yours.
>
> Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
> `main`)

---

## Longer version, if the sub prefers detail

**Title:** One ROCm graph, zero ROCm testers: looking for someone to try a
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
> **What is already prepared for you.** A shipped graph,
> `workflows/variants/otr_amd_still.json`, generated from the same source as
> every working graph and marked experimental / no receipts because that is
> literally the status, plus two lab profiles (`otr_amd16_rocm`,
> `otr_amd8_rocm`) for 16 GB and 8 GB cards. They pin a deliberately
> conservative engine set, and I went through every engine to confirm it is
> plain PyTorch:
>
> * video: a still-motion lane (numpy/PIL, no diffusion video)
> * images: Z-Image Turbo at bf16 (stock ComfyUI loaders, no fp8/nvfp4)
> * voices: Kokoro
> * music: MusicGen (the 16 GB lab profile swaps in Stable Audio 3)
> * writer: Qwen3.5-4B through transformers, unquantised -- no bitsandbytes,
>   no llama.cpp needed
> * sage-attention, flash-attn, fp8 and cuda-malloc: all off
>
> A ROCm torch build presents as `cuda` to everything above it, which the
> pack expects. The one thing to avoid is installing `sageattention` -- it is
> CUDA-only and the pack refuses to run if it finds it patched in.
>
> **What I am asking for.** An hour on an RDNA3, RDNA4 or MI-series card
> (16 GB for the full profile, 8 GB for the small one), ROCm 7.2.x on Windows
> (AMD's PyTorch for Radeon) or Linux, about 50 GB of disk.
> A rented box is fine and cheap. The repo has `apple/ROCM.md`
> with six commands start to finish and a list of exactly what to send back.
>
> **A traceback is a win.** If it dies ninety seconds in, that failure is the
> first real ROCm signal this project has ever had and I will act on it. If it
> produces an episode, you are the first person in the world to have made one
> on AMD hardware, and I will say so in the repo.
>
> Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
> `main`). Happy to answer anything here.

---

## Notes for you before posting

* Check r/ROCm's rules on self-promotion; some hardware subs want a
  "[Project]" or "[Help]" tag, and some prefer no link in the title.
* The repo is MIT and the ask is genuine testing, not traffic, which is
  usually the distinction those rules care about.
* Where to post, ranked by who owns the card and reads posts like this:
  r/LocalLLaMA first (the biggest local-AI crowd, real ROCm users, lead the
  title with "one GPU, offline"), then cross-post -- not repost -- a day apart
  to r/ROCm (small, every reader has the card) and r/comfyui (the pack's
  home audience). For the gamer subs, r/radeon and r/Amd, reframe the title
  for them ("your 7900 XTX has never made a radio drama") and keep the same
  body. Skip r/StableDiffusion (stricter self-promo filter, off-topic ask)
  and r/AMDHelp (support only). The Comfy-Org Discord's custom-nodes channel
  is the non-Reddit option. Put the GitHub link in the body, not the title.
* Expect the first reply to be "which card" and the second to be "does it
  need flash-attn". Both answers are in the mission file.
