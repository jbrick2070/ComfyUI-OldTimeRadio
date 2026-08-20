# CFG: THE PROBLEM STATEMENT

**Written 2026-08-20, at the operator's request, after a false explanation of
CFG was found sitting in the locked recipe.**

Operator, in his words: *"WOAH BE CAREFUL W/ CFG THAT WAS A LOT OF TROUBLE IN
THE VRAM AND WE FOUND OUT LATER."* This file exists so the next person does not
have to find out later again.

---

## 1. WHAT CFG IS, IN ONE LINE

CFG is how hard the model is pushed toward your prompt. Low = the model wanders
and does its own thing. High = it clamps hard to the words.

There is a second thing CFG does, and it is the one that matters here: **at
normal settings the model runs TWICE per step** -- once imagining your prompt,
once imagining the absence of it -- and subtracts one from the other. Two runs
means roughly twice the memory.

## 2. WHERE WE ARE

All three CFG values on the LTX 2.5 lane are **1.0** and they are **LOCKED** by
operator directive:

```
LTX25_CFG_VIDEO    = 1.0
LTX25_CFG_AUDIO    = 1.0
LTX25_CFG_MODALITY = 1.0
```

Nothing in the 2026-08-19/20 work changed any of these. Verified: no commit
touched a CFG value line.

## 3. WHY THEY ARE LOCKED -- the part that is still true

The lab measured that raising any of them pushes the render **past 16 GiB**,
which is an instant crash on a 16.3 GiB card. That is an OBSERVATION and it
stands.

## 4. THE PART THAT WAS WRONG, AND IT WAS WRONG IN OUR OWN FILES

Three comments -- two in `eng_ltx25.py`, one in `ltx25_recipe.py`, inherited
from the lab's notes -- explained the lock like this:

> "CFG 1.0 evaluates batch size 1. Any value above 1.0 forces batch size 2
> (positive + negative evaluated together)... it doubles the batch."

**That is the normal ComfyUI rule and it does NOT apply to this recipe.**

ComfyUI does skip the second run at CFG 1.0 -- `sampling_function` sets
`uncond_ = None` when the scale is ~1.0 (`comfy/samplers.py:609`). But our
locked sampler is `euler_ancestral_cfg_pp`, a **CFG++** sampler, and CFG++
explicitly turns that optimisation back OFF:

* `comfy/k_diffusion/sampling.py:1284` -- passes `disable_cfg1_optimization=True`
* `comfy/k_diffusion/sampling.py:1297` -- then USES `uncond_denoised` in its
  own step maths

**So we are already running both passes, at CFG 1.0, every step.** The saving
the comment claimed we were getting, we were never getting.

## 5. THE OPEN QUESTION -- THIS IS THE ACTUAL RISK

If we are already paying for both passes at 1.0, then **nobody currently knows
what raising CFG would actually cost.** The old mental model said "1.0 is cheap,
above 1.0 doubles it". That model is broken. The truthful position is:

* the lab's "above 1.0 goes past 16 GiB" is a **measurement** and is trusted;
* the **reason** given for it is wrong;
* therefore **no one may reason about CFG cost from first principles any more.**
  Only a measurement decides.

This is exactly the shape of the trouble the operator remembers: a CFG-related
VRAM belief that felt solid, was acted on, and turned out to be false later.

## 6. WHY THE FALSE BELIEF WAS DANGEROUS, CONCRETELY

Believing "the negative does nothing at CFG 1.0" made an obvious optimisation
look free: feed the POSITIVE conditioning into both slots of the guider and skip
an entire 12-billion-parameter text encode per shot. Faster, less memory, no
downside.

**It would have silently changed every render on this lane.** The negative is
live; the sampler consumes it.

That proposal was made during the OOM panel, **one reviewer approved it**, and
it was killed only because another reviewer checked which sampler was actually
selected. It came within one seat of shipping.

## 7. THE RULES GOING FORWARD

1. **Do not change any CFG value.** Locked by operator directive. If a change is
   ever wanted it is a measured experiment with a receipt, never an inference.
2. **Do not reason about CFG memory cost from the "1.0 means one pass" rule.**
   It is false on this lane. Check which sampler is selected first -- any
   `*_cfg_pp` sampler forces the second pass back on.
3. **Do not "optimise away" the negative conditioning.** It is live here. The
   empty STRING is the locked recipe value; the empty string is not the same
   thing as an unused input.
4. **A CFG belief needs a file and a line.** Every claim in section 4 above is
   cited to real ComfyUI source. Anything that cannot be cited that way is a
   hypothesis.

## 8. WHAT IS STILL UNMEASURED

* The true VRAM cost of CFG > 1.0 on THIS lane with THIS sampler. The lab's
  ">16 GiB" is the only datum and its explanation is unreliable.
* Whether the CFG++ double pass is part of why the text encode straddles the
  card. Unverified; the encode spike was measured at ~13.76 GiB allocated
  independently of sampling.

Both are open. Neither blocks the current work, and neither should be guessed at.
