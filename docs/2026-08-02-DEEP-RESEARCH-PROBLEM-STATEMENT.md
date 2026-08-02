# Problem statement for deep research

*Paste this as the research prompt. The companion file
`2026-08-02-DEEP-RESEARCH-BRIEF.md` has the full measured detail and twelve
numbered questions; this states the problem and what a useful answer is.*

---

## The problem

I run a fully local, offline AI pipeline that generates old-time-radio-style
episodes on a single **RTX 5080 Laptop (16 GB VRAM, Blackwell sm_120, Windows,
PyTorch 2.10 / CUDA 13, no Flash Attention 2)**. It writes a script, casts
characters, synthesises speech, then generates video for each dialogue beat.

**Hard rule: every second of audio must receive original generated video.** No
mirroring, no ping-pong/boomerang, no held or frozen frames. When a beat's audio
outruns what one render call can produce, the beat is split into segments that
are generated separately and cut together.

Two model-behaviour questions decide how good the finished episodes are, and
**neither can be answered from my own code** -- they are questions about how
these models actually work. I need external evidence.

### Question 1 -- Does render orientation change VRAM cost at an identical pixel count?

I run **HuMo-14B in fp8** (the Kijai conversion
`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`, a Wan 2.1-based
audio-driven talking-face model, conditioned by Whisper-large-v3 and a
lightx2v 480p distill LoRA). The **same checkpoint** is configured two ways:

* **480x832 portrait** -- allowed up to **177 frames** per clip.
* **832x480 landscape** -- capped at **49 frames** per clip.

Both are **399,360 pixels**. The landscape cap claims the tier "rides ~15.9 GB
at 832x480", above my 14.5 GB working ceiling, and cites an internal measurement
document that no longer exists, so I cannot check it.

That 49-frame cap is expensive: it turns a 17.7-second character beat into **ten
separate ~2-second clips**, each restarting from the same reference portrait, so
the character visibly snaps back to the same pose ten times mid-speech. If the
cap is wrong, that beat becomes two or three clips and the problem vanishes.

**My assumption, which I want confirmed or destroyed:** attention cost in these
DiT video models follows latent *sequence length*, which is identical at 480x832
and 832x480, so a cap that differs by orientation alone is a bookkeeping error
rather than physics. Is that right? Is there any mechanism -- positional/rotary
encoding, tiling granularity, a non-square attention or decode window, memory
fragmentation -- by which one orientation genuinely costs more at equal pixels?
And what peak VRAM do others actually observe for HuMo-14B fp8 at each
orientation and clip length on 16 GB consumer cards?

### Question 2 -- The lip-sync onset error

On the same model, **the audio consistently leads the lips by 100-200 ms**, and
the face is visibly static for roughly the first 3-6 frames while the first
phonemes are already playing. Audio arriving *early* is the perceptually
unforgiving direction, and this lands on the single most important beat of every
episode.

Is this a **known, documented characteristic** of HuMo specifically, or of
Whisper-conditioned audio-driven face models in general? **What is the accepted
fix, and what is the correct magnitude?** My own notes prescribe padding the
audio with leading silence, generating the extra frames, then discarding those
frames after VAE decode -- but they never state a pad length, and guessing wrong
simply moves the error in the other direction. I would rather adopt a number
someone has measured than invent one. Is a conditioning-offset approach better
than padding? Does the correct pad change with frame rate or clip length?

## Secondary questions, if the research has room

3. Is it established that **tiled VAE decode with a bounded temporal window makes
   peak VRAM independent of total frame count** for Wan-family VAEs? I measured
   flat peaks (6563 / 6531 / 6563 MiB) at 17 / 49 / 81 frames on a Wan 2.2 5B
   lane at 832x480, and an untiled LTX lane that scaled instead. Does flat-to-81
   license flat-to-177, or does some other allocation begin to scale?
4. **HuMo takes a `ref_image`, not a `start_image`.** Confirm the reference is an
   identity hint with no first-frame lock, so consecutive clips cannot be
   frame-chained -- and say whether any supported method chains them continuously.
5. Peak VRAM for **`ltx-2.3-22b-dev` with its audio VAE on a single ~449-frame
   clip** at 16 GB. This is the largest untested allocation in my system.
6. Best current practice for **identity preservation across separately generated
   clips of the same character** in a fully local pipeline (shared reference,
   fixed seed, IP-Adapter, PuLID, InstantID, or newer). I currently share one
   portrait across segments with no identity conditioning at all.

## Constraints -- please do not propose these

* Any mirroring, ping-pong, boomerang, or frozen-frame fill. Ruled out
  unconditionally.
* Flash Attention 2 -- no wheel exists for this stack.
* Cloud or API services of any kind. Offline-first, no keys, no paid services.
* Async CUDA streams or queue refactors. Sequential execution only.
* Upgrading past PyTorch 2.10 / CUDA 13, which the rest of the stack is pinned to.

## What a useful answer looks like

For each finding: **the claim, the source, and whether it was measured on
comparable hardware.** A cited measurement on a 16 GB consumer card outweighs a
confident generalisation. Model cards, GitHub issues, ComfyUI node source,
benchmark posts, and papers are all fair game.

Where nobody has published the answer, **say so plainly.** "This has not been
measured publicly" is a genuinely useful result -- it tells me to spend the GPU
hours measuring it myself, which is expensive but decisive, rather than
continuing to search.
