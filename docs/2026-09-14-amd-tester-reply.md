# Reply to the AMD tester (issue #2)

**Revised after actually reading their report** at drearburh.uk/otr-amd-report/,
including CHANGES.md. The first draft was written from the forum excerpt alone
and got two things wrong: it assumed Windows, and it did not know they had found
a real bug.

Paste the block. Notes underneath.

---

```
This is the best bug report this project has had, and I want to deal with the
useful part first.

FIXED: the models-root bug you found in scripts/otr_fetch_lane_weights.py. You
were exactly right about the mechanism -- models_root() tried the in-process
import first, that import needs ComfyUI's folder_paths and so can only work from
inside a running ComfyUI, and a bare `except Exception` swallowed the failure and
returned the hardcoded C:\ComfyUI-Models. On Ubuntu. Silently. The function's own
docstring said "never a hardcoded guess" directly above the hardcoded guess,
which is the kind of thing you only see when someone else runs your code.

Three changes rather than the doc note you suggested:
  - An explicit OTR_COMFYUI_MODELS_ROOT / COMFYUI_MODELS_ROOT now wins outright,
    before anything else is tried. If you name a path you meant it.
  - If neither is set, it looks for the models/ directory that sits beside
    custom_nodes/ in any normal ComfyUI checkout -- so your Docker case should
    now work with no environment variable at all.
  - And if it genuinely cannot tell, it says so loudly on stderr and names the
    variable, instead of returning a Windows path with a straight face.

You are also the first person to run this on Linux. The AMD page says "Windows
first" because that is where the ROCm build was expected to land; you have shown
the Ubuntu/ROCm 7.2/PyTorch 2.9.1 path works end to end, on RDNA4 no less. And
your bitsandbytes note matches the design intent exactly -- there is no ROCm
build, the loader is supposed to degrade to bf16 with a warning, and you have
confirmed it does that in the wild rather than just in my head.

On the dissonance between audio and visuals: you are right, and I would rather
say so plainly. They are produced by stages that agree on the script and not
much else, so you get two individually-correct things that were never made to
answer to each other. That is the honest weak point of the pipeline.

What fixes it is not better prompts, it is an EDITOR pass -- something that looks
at the finished audio and finished picture together and makes the cuts a human
editor would: hold this shot, trim that one, put the music under the line instead
of over it. A person could do that today. I am waiting on local models good
enough to do it inside a ComfyUI workflow, because this pack runs on your own
machine with no cloud and no keys and I am not trading that away for a shortcut.
When that lands it is the next real step, not a tweak.

On voices -- genuinely useful, and noted. Kokoro ships on every slot for one
reason: it is the only engine that is a single click on every platform the pack
claims to support, about 0.3 GB, fetched automatically. That makes it the floor,
not a verdict. The engine is a dropdown, so Fish Speech, Qwen3-TTS and Omnivoice
are all the same size of experiment. Fish is the one I find most interesting, for
the reason you gave: a radio drama lives on delivery, and consistency across a
cast matters more than any one good take. Bark is already in the box if you want
the contrast -- fair warning, 4.2 GB, and it will out-of-memory a 16 GB Mac.

One practical note: you cloned v2.0-alpha at 0b38424, which was right when you
started and is now about a hundred commits behind. main is the branch; v2.0-alpha
is retired but kept alive so old image links do not break. Worth a pull before
you try anything else -- a fair amount has changed, including your fix.

Thank you. Genuinely.
```

---

## What changed from the first draft, and why

* **It leads with the fix, not with thanks.** They filed a precise, correctly
  diagnosed bug. The respectful reply is evidence it was acted on.
* **The Windows/Linux fact was wrong in the first draft.** CHANGES.md: Ubuntu
  24.04.4 in Docker on Unraid, ROCm 7.2, PyTorch 2.9.1+rocm7.2.4, Radeon AI PRO
  R9700 (32 GB, RDNA4/gfx1201). `apple/ROCM.md` says "Windows first", so this is
  the first Linux ROCm receipt as well as the first AMD one.
* **The branch note is new and worth saying.** They cloned `v2.0-alpha` at
  `0b38424`, 103 commits behind current main. The shipped docs are clean -- the
  only mention correctly calls it retired -- so they simply started before the
  promotion. Telling them costs nothing and saves them a confusing next run.
* **The first-Radeon-receipt claim survives**, and `apple/ROCM.md:3-4` backs it:
  "the first episode off a Radeon is what makes it v2.1."
* **Still no promise on Fish/Qwen3/Omnivoice.** Adding a voice engine has real
  gates, and a voice change is settled by ear.

## Follow-ups this report earned

1. The `models_root` fix is done. Consider whether `_models_root()` itself, in
   `nodes/_otr_gguf_backend.py`, should also stop defaulting to a Windows path.
2. `apple/ROCM.md` should record this receipt: card, OS, ROCm and PyTorch
   versions, and that it was Linux. Ask them first about being named.
3. Their `qa_report.json` and `episode_canon.json` are downloadable and worth
   reading against our own expectations for that graph.
