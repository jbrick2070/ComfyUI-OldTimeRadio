# Reply draft -- the first AMD volunteer (u/k8-bit, R9700 32 GB)

Posted 2026-09-13 in reply to the ROCm recruitment comment: *"I've got a 32gb
r9700 setup for Comfy as a secondary machine to an Nvidia primary. Love doing
audiobook/narrative stuff, so happy to help."*

An R9700 is RDNA4 with 32 GB, which is the best card that could have answered
that post. Paste the reply below; it asks for exactly one result and offers a
second only if the first works.

---

That is the exact card I was hoping someone would have, and an audiobook habit
means you will hear things in the output I would not. Thank you.

There is one graph I need run and it is the small one. Here is the whole ask:

**1. Install.** ComfyUI with a ROCm build of PyTorch (AMD's own instructions
for your OS), then the pack -- either through ComfyUI Manager, searching for
**Old Time Radio**, or `git clone -b v2.0-alpha
https://github.com/jbrick2070/ComfyUI-OldTimeRadio` into `custom_nodes/`. The
branch matters; `main` is a stale v1.7.

**2. ffmpeg 6.1 or newer, with ffprobe.** Every episode is mixed and muxed
through it. On Linux check `ffmpeg -version` first, because Ubuntu 22.04 ships
4.4 and that one cannot write the final file. I learned this yesterday the
expensive way, on a rented box that rendered two complete episodes and then
wrote zero bytes. The pack now refuses in the first second instead, but a
current build saves you the conversation.

**3. Load `workflows/variants/otr_amd_still.json` and queue it.** First run
pulls about 12 GB: a 4B writer, a music model, and the Kokoro voices. Nothing
is gated, so no Hugging Face account is needed. It writes a script, casts it,
speaks every part, scores it, draws the stills, moves them, burns captions,
rolls credits, and drops a finished MP4 in `output/otr/obs`.

**4. Tell me what happened.** Either way is a result. There is an issue open
for exactly this -- https://github.com/jbrick2070/ComfyUI-OldTimeRadio/issues/2
-- and what I need in it is: your OS, your ROCm and PyTorch versions, whether
an MP4 landed in `otr/obs`, and the first traceback if one did not. A failure
is worth as much to me as a pass; nothing has ever run on AMD hardware, so
every line of that is new.

**Windows is the more interesting answer** if you have the choice. The pack is
Windows-native and AMD ships a Radeon ROCm build for Windows now, but nobody
has pointed the two at each other. Linux is welcome too.

**If the still graph works, there is a bigger one.** With 32 GB you are past
every card I own. `otr_16gb_video` and `otr_16gb_animatediff` use real video
diffusion instead of moved stills, and whether those run under ROCm is a
genuinely open question -- the still graph exists because I could not answer
it. That would be the second episode, not the first.

Fair warning on the time: one act is a full episode and takes roughly fifteen
minutes to two hours depending on the graph and the card. The still one is at
the fast end.
