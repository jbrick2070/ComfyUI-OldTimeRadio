# Reply draft -- the first AMD volunteer (u/k8-bit, R9700 32 GB)

Posted 2026-09-13 in reply to the ROCm recruitment comment: *"I've got a 32gb
r9700 setup for Comfy as a secondary machine to an Nvidia primary. Love doing
audiobook/narrative stuff, so happy to help."*

An R9700 is RDNA4 with 32 GB. Paste the reply below; it asks for exactly one
result and nothing else.

**It deliberately names no other graph.** The operator owns no AMD hardware, so
what this tester runs on 32 GB is the only AMD fact anyone will have. Pointing
him at the graphs named for NVIDIA VRAM tiers would invite him to prove
something about cards nobody can check, and the tier names mean nothing on his
hardware anyway. One card, one graph, one result.

---

That is the exact card I was hoping someone would have, and an audiobook habit
means you will hear things in the output I would not. Thank you.

There is one graph I need run and it is the small one. Here is the whole ask:

**1. Install.** ComfyUI with a ROCm build of PyTorch (AMD's own instructions
for your OS), then the pack -- either through ComfyUI Manager, searching for
**Old Time Radio**, or `git clone -b v2.0-alpha
https://github.com/jbrick2070/ComfyUI-OldTimeRadio` into `custom_nodes/`. The
branch matters; `main` is a stale v1.7.

**2. A current ffmpeg, with ffprobe.** Every episode is mixed and muxed
through it, and older builds cannot write the final file -- Ubuntu 22.04 ships
4.4, which fails. I learned that yesterday the expensive way, on a rented box
that rendered two complete episodes and then wrote zero bytes. The pack now
muxes a fifth of a second of silence up front and refuses in about a second if
your build cannot do it, so you will not be guessing; a current build just
saves you the conversation.

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

Fair warning on the time: one act is a whole episode, and this graph takes
somewhere between fifteen minutes and an hour depending on the card.
