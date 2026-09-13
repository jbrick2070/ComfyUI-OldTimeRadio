# Reply draft -- the first AMD volunteer (u/k8-bit, R9700 32 GB)

Posted 2026-09-13 in reply to the ROCm recruitment comment: *"I've got a 32gb
r9700 setup for Comfy as a secondary machine to an Nvidia primary. Love doing
audiobook/narrative stuff, so happy to help."*

An R9700 is RDNA4 with 32 GB. Paste the reply below; it asks for exactly one
result and nothing else.

**On the Windows push, revised 2026-09-13, from the post itself.** The
operator supplied the text of `r/ROCm/comments/1wfb4qe`. Its author validates
ZLUDA + ROCm/HIP on ONE card, an RX 9060 XT (gfx1200, RDNA4), with nvcuda,
cuBLAS/rocBLAS, cuBLASLt/hipBLASLt, cuSPARSE/rocSPARSE and cuFFT all passing,
and a 2,216,347-parameter PPO run completing a full training iteration.
**His own stated limit is the one that decides this for us:** *"cuDNN is also
still a limitation with the current stable Windows HIP stack, so
convolution-heavy workloads may not work yet."* `otr_amd_still` is SD1.5
stills plus a VAE decode -- convolutions end to end -- and a PPO policy
network exercises none of that. So the route is not wrong, it is simply not
this workload, and saying that plainly is fairer to the project than "reports
say it is painful". The pack's own AMD row already assumes the NATIVE path
(`device_backend: "cuda"`, because PyTorch ROCm presents through the CUDA
API), so nothing in the code changes. He is also collecting RX 9000-series
Windows reports, and our volunteer's R9700 is one -- worth pointing at, since
it costs the volunteer a scan and is not our test.

**It deliberately names no other graph.** The operator owns no AMD hardware, so
what this tester runs on 32 GB is the only AMD fact anyone will have. Pointing
him at the graphs named for NVIDIA VRAM tiers would invite him to prove
something about cards nobody can check, and the tier names mean nothing on his
hardware anyway. One card, one graph, one result.

---

That is the exact card I was hoping someone would have, and an audiobook habit
means you will hear things in the output I would not. Thank you.

There is one graph I need run and it is the small one. Here is the whole ask:

**1. Install from git, not from Manager, for this one.** ComfyUI with a ROCm
build of PyTorch (AMD's own instructions for your OS), then
`git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio` into
`custom_nodes/`. Either branch is fine -- `main` and `v2.0-alpha` are the same
commit. Manager pulls the registry copy, and the graph below was added to the
repo today, so no published version carries it yet. I will ping you when one
does, because the one-click install is its own thing worth testing and I would
rather learn about it from someone who is not me.

**2. A current ffmpeg, with ffprobe.** Every episode is mixed and muxed
through it, and older builds cannot write the final file -- Ubuntu 22.04 ships
4.4, which fails. I learned that yesterday the expensive way, on a rented box
that rendered two complete episodes and then wrote zero bytes. The pack now
muxes a fifth of a second of silence up front and refuses in about a second if
your build cannot do it, so you will not be guessing; a current build just
saves you the conversation.

**3. Fetch the image weights once, by hand.** From the checkout:

    python scripts/otr_fetch_lane_weights.py z_image

About 19 GB in three files from Comfy-Org, all ungated. This is the one thing
the pack does not fetch for you, and I would rather you heard it from me than
from an error message. Take the bf16 lane, not int8 -- int8 goes through
bitsandbytes, which is not a road I would send you down on ROCm as the first
thing you try.

**4. Load `workflows/variants/otr_amd_still.json` and queue it.** The first
queue pulls about 12 GB more on its own -- a 4B writer, the music model, the
Kokoro voices -- and none of it is gated, so no Hugging Face account is
needed. Then it writes a script, casts it, speaks every part, scores it, draws
the stills, moves them, burns captions, rolls credits, and drops a finished
MP4 in `output/otr/obs`.

**5. Tell me what happened.** Either way is a result. There is an issue open
for exactly this -- https://github.com/jbrick2070/ComfyUI-OldTimeRadio/issues/2
-- and what I need in it is: your OS, your ROCm and PyTorch versions, whether
an MP4 landed in `otr/obs`, and the first traceback if one did not. A failure
is worth as much to me as a pass; nothing has ever run on AMD hardware, so
every line of that is new.

**Take whichever gets you to a working ComfyUI without a fight.** I had been
saying Windows is the more interesting answer, and it is -- the pack is
Windows-native and nobody has pointed it at a Radeon there. But I am not going
to spend your evening on my curiosity, and if Linux is the shorter road for
you, take it. A Linux result is worth far more to me than a Windows result I
never get.

**If you do go Windows, use AMD's native ROCm PyTorch, not ZLUDA.** Not a
knock on that project -- its author is careful and says the limit out loud:
cuDNN is still missing from the stable Windows HIP stack, so convolution-heavy
workloads may not work yet. What this graph does is convolution-heavy by
definition. It draws SD1.5 stills and decodes them through a VAE, which is
nothing but convolutions. The training run that project proves out is a small
policy network, which does not touch that path at all.

**Unrelated to me, and worth thirty seconds of your time:** that project is
asking specifically for RX 9000-series Windows reports to build a real
compatibility matrix, and yours is one. Its scanner would tell you whether
cuBLAS, cuBLASLt, cuSPARSE and cuFFT come up on your card the way they do on
the 9060 XT. Different question from mine, useful to someone either way.

Fair warning on the time: one act is a whole episode, and this graph takes
somewhere between fifteen minutes and an hour depending on the card.

---

# Reply draft 2 -- the Vulkan comment (u/Dodgy_Past)

Posted 2026-09-13: *"As a comment, I tried ROCm for LLM work on a Halo Strix
395+ and it was very buggy, OTOH Vulkan is working very well for me."*

**Grounding, so the reply does not overclaim.** Vulkan is a real compute
backend for llama.cpp and not one for ComfyUI: ComfyUI is PyTorch, and PyTorch
has no general Vulkan compute device (the one it had was mobile-focused and has
been wound down). The single overlap is genuine, though -- this pack has a
`gguf_native` writer transport that drives `llama-cpp-python` in process, with
`n_gpu_layers`, already handling Metal alongside CUDA
(`nodes/_otr_gguf_backend.py`). A Vulkan-built llama-cpp-python would therefore
accelerate the writer, which is the largest single model in the AMD lane, while
everything downstream stayed on CPU. The shipped GGUF dropdown rows are Gemma 4
12B, not the Qwen the AMD graph names, so this is an experiment and not a
supported configuration -- say so.

Also worth keeping separate: a Strix Halo APU (RDNA 3.5, unified memory) is not
a discrete R9700 (RDNA4). One person's bad ROCm week on the first does not
predict the second, and implying otherwise would talk the other volunteer out
of the test that matters.

---

That is a useful data point, thank you -- and it matches what I have read from
other Strix Halo owners.

Worth separating two things, because they do not travel together. Vulkan is a
real backend for llama.cpp, so for pure LLM work your experience makes sense.
ComfyUI is PyTorch, and PyTorch has no general Vulkan compute device -- the one
it had was mobile-focused and has been wound down -- so for the image, video
and audio half of this there is no Vulkan road today. ROCm or CPU.

There is exactly one place the two meet, and it is the interesting one. The
script writer in this pack can run in process through llama-cpp-python rather
than through torch, so a Vulkan-built llama.cpp would accelerate the biggest
single model in the pipeline while everything downstream ran on CPU. I have
never seen anyone try that split and I would read the numbers with real
interest. Fair warning that it is an experiment, not a supported setup: the
GGUF rows I ship are Gemma 4 12B, not the smaller Qwen the AMD graph names.

And for anyone reading this next to the other comment -- a Strix Halo APU is
not a discrete RDNA4 card. Your ROCm experience is worth knowing and it does
not decide what an R9700 will do.

---

# Reply draft 1b -- the SAME reply with every URL removed

2026-09-13: the first attempt came back as held for automatic approval review.
A long comment carrying three URLs is the classic automod trigger, and none of
those links is load-bearing -- a repo is findable by name, and the issue number
is enough. Same content, no URLs, nothing shortened to hide a link (which is
what automod is actually looking for).

---

That is the exact card I was hoping someone would have, and an audiobook habit
means you will hear things in the output I would not. Thank you.

There is one graph I need run and it is the small one. Here is the whole ask:

**1. Install from git, not from Manager, for this one.** ComfyUI with a ROCm
build of PyTorch (AMD's own instructions for your OS), then clone the repo --
it is jbrick2070/ComfyUI-OldTimeRadio on GitHub -- into `custom_nodes/`.
Either branch is fine, `main` and `v2.0-alpha` are the same commit. Manager
pulls the registry copy, and the graph below was added to the repo today, so
no published version carries it yet. I will ping you when one does, because
the one-click install is its own thing worth testing and I would rather learn
about it from someone who is not me.

**2. A current ffmpeg, with ffprobe.** Every episode is mixed and muxed
through it, and older builds cannot write the final file -- Ubuntu 22.04 ships
4.4, which fails. I learned that yesterday the expensive way, on a rented box
that rendered two complete episodes and then wrote zero bytes. The pack now
muxes a fifth of a second of silence up front and refuses in about a second if
your build cannot do it, so you will not be guessing; a current build just
saves you the conversation.

**3. Fetch the image weights once, by hand.** From the checkout:

    python scripts/otr_fetch_lane_weights.py z_image

About 19 GB in three files from Comfy-Org, all ungated. This is the one thing
the pack does not fetch for you, and I would rather you heard it from me than
from an error message. Take the bf16 lane, not int8 -- int8 goes through
bitsandbytes, which is not a road I would send you down on ROCm as the first
thing you try.

**4. Load `workflows/variants/otr_amd_still.json` and queue it.** The first
queue pulls about 12 GB more on its own -- a 4B writer, the music model, the
Kokoro voices -- and none of it is gated, so no Hugging Face account is
needed. Then it writes a script, casts it, speaks every part, scores it, draws
the stills, moves them, burns captions, rolls credits, and drops a finished
MP4 in `output/otr/obs`.

**5. Tell me what happened.** Either way is a result. There is an issue open
for exactly this -- issue 2 on that repo, the AMD/ROCm adaptation one -- and
what I need in it is: your OS, your ROCm and PyTorch versions, whether an MP4
landed in `otr/obs`, and the first traceback if one did not. A failure is
worth as much to me as a pass; nothing has ever run on AMD hardware, so every
line of that is new.

**Take whichever gets you to a working ComfyUI without a fight.** I had been
saying Windows is the more interesting answer, and it is -- the pack is
Windows-native and nobody has pointed it at a Radeon there. But I am not going
to spend your evening on my curiosity, and if Linux is the shorter road for
you, take it. A Linux result is worth far more to me than a Windows result I
never get.

**If you do go Windows, use AMD's native ROCm PyTorch, not ZLUDA.** Not a
knock on that project -- its author is careful and says the limit out loud:
cuDNN is still missing from the stable Windows HIP stack, so convolution-heavy
workloads may not work yet. What this graph does is convolution-heavy by
definition. It draws SD1.5 stills and decodes them through a VAE, which is
nothing but convolutions. The training run that project proves out is a small
policy network, which does not touch that path at all.

**Unrelated to me, and worth thirty seconds of your time:** that project is
asking specifically for RX 9000-series Windows reports to build a real
compatibility matrix, and yours is one. Its scanner would tell you whether
cuBLAS, cuBLASLt, cuSPARSE and cuFFT come up on your card the way they do on
the 9060 XT. Different question from mine, useful to someone either way.

Fair warning on the time: one act is a whole episode, and this graph takes
somewhere between fifteen minutes and an hour depending on the card.
