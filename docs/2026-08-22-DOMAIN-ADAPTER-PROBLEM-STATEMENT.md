# Problem statement: what the removable domain adapter could do for Ghost

**Date:** 2026-08-22
**Status:** problem statement and a testable hypothesis. Nothing is built. No
graph change is authorised by this document.

## What the adapter actually is

AnimateDiff v3 ships an optional companion to the motion module:
`v3_adapter_sd_v15.ckpt`, a **LoRA applied to the IMAGE model** (SD1.5), not to
the motion module. The authors describe it in one sentence:

> a Domain Adapter LoRA for image model finetuning, which provides more
> flexibility at inference

and state its purpose plainly: it exists to **"fit defective visual artifacts
(e.g., watermarks) in the training dataset."** It can be removed at inference
entirely, or integrated with an adjustable LoRA scalar.

Read that mechanism carefully, because it is the whole point. During training
the model learned the training set's *defects* along with its motion. The
adapter is where those learned defects are parked. At inference you decide how
much of that parked behaviour to let back in -- none, some, or all.

## Why this is interesting for THIS lane specifically

Ghost's single worst content tendency is **lettering**. The adapter's own file
says so:

> SD1.5 volunteers lettering into any scene that smells like a sign, a poster or
> a radio dial -- which is most of this show.

That is why the lane's negative leads, unconditionally, with
`text, watermark, caption, lettering, subtitles`, and why the plan refused an
AnimateLCM checkpoint outright: at CFG 1-2 the negative goes inert, and the
lettering defence goes with it.

**So today we fight lettering with tokens.** We spend part of a 320-character
budget, every beat, telling the model not to do something it learned to do.

The domain adapter is a **different mechanism aimed at the same defect**: rather
than arguing with the model at inference, it isolates the learned artifact
behaviour in a removable component. `watermarks` is the authors' own example,
and a watermark is lettering.

## The hypothesis, stated so it can be killed

**If Ghost runs v3 with the domain adapter DOWN-WEIGHTED or ABSENT, the model's
learned tendency to paint lettering should be structurally reduced -- and the
negative's job gets easier rather than harder.**

Three things would confirm it, and any one of them failing kills it:

1. Lettering incidence drops at a fixed seed and prompt, adapter off vs on.
2. It drops *without* the motion getting worse -- the adapter is on the image
   model, so motion should be largely unaffected. If motion degrades, the
   mechanism is not what we think.
3. The effect is monotonic in the scalar. If 0.0 and 1.0 differ but 0.5 behaves
   randomly, we are looking at seed noise, not the adapter.

## What we would have to change, and what it costs

**This is not free, and it is a graph change.** Ghost's internal graph is eight
nodes and ten links, frozen by the r4 plan, which excludes Motion LoRA and every
multival/keyframe/per-block socket. A domain adapter needs a **ninth node** -- a
LoRA loader between the checkpoint and the ADE loader:

```
ckpt -> LoraLoader(v3_adapter) -> ADE -> sampler
```

Three honest consequences:

* The coding plan says that if discovery proves a new node is necessary, **stop
  and re-plan**. That clause is about `otr_canonical.json`, which is untouched
  here -- this is adapter-internal -- but the spirit applies: the eight-node
  contract is pinned by tests and would need amending deliberately, not
  incidentally.
* It only makes sense on a **v3 peer lane**. The golden lane's `mm-p_0.5` has no
  matching adapter, and the operator's condition stands: golden stays untouched.
* It adds a third artifact to a lane whose whole pitch is **two files, 3.9 GB**.
  For a ComfyUI community template that matters -- a third file is a third thing
  a stranger must fetch.

## What it might buy, ranked

1. **A structural fix for lettering** rather than a token-budget fight. If it
   works, the negative could shrink and the positive gets its characters back.
2. **A knob rather than a switch.** The scalar means we are not choosing between
   "artifacts" and "no adapter" -- there is a dial, and dials can be tuned once
   and frozen, which is how every other recipe value in this lane works.
3. **It is the intended configuration.** We are currently planning to run v3
   *without* the component its authors designed alongside it. If v3 comes back
   looking flat in the bakeoff, this is the first thing to try before writing
   the module off.

## What would make this NOT worth doing

Stated up front so the answer is not motivated:

* If the bakeoff shows `mm-p_0.5` still looks best, the licence is the only
  reason to move, and adding a third file to chase a lettering fix on a module
  we did not prefer is bad value.
* If lettering is not actually a live problem on shipped episodes. **We have not
  measured it.** The published Ghost episode was not audited for painted text;
  the operator's verdict was "hyperreal" and "its perfect". A fix for a problem
  nobody is having is the definition of chasing.
* The footer/lettering detector this repo already built was calibrated on dark
  test cards and flagged 5 of 7 real production stills -- so we do not currently
  have a trustworthy automatic way to measure lettering incidence. Without one,
  the confirmation test above is an eyeball exercise across many seeds, which is
  expensive.

## The cheap first step

Before any graph change: **finish the bakeoff.** If v3 is not in contention on
looks, this whole document is moot. If it is, the next cheapest move is a
fixed-seed A/B of v3 with and without the adapter on a handful of card-heavy
beats -- the ones with signs, dials and posters, where lettering actually shows
up -- rather than a full episode.
