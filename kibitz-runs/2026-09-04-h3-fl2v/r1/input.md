# H3 FL2V -- pin a LAST frame on lane 19 so a beat ends where the next begins

**Status: DESIGN FORK, no code written.** Feasibility is confirmed YES against the
installed ComfyUI. This document is the thing under review.

**Operator's goal, verbatim:** *"Pinning a last frame too would give real
beat-to-beat continuity: a shot could end where the next one begins, instead of
the model inventing a fresh scene every beat."*

**Explicitly OUT OF SCOPE, do not propose any of these:** the W4A8 quantized H3
DiT, Qwen3-VL 4B INT8 + ClipProj, INT8 ConvRot VAE, ComfyKitchenAttention, the
Spectrum node pack. H3 already fits the physical 4060 at 7.21 GiB cold / 6.79
warm (864x480x90, 20 steps -- `docs/4060_DRILL_LOG.md`), so VRAM headroom is not
wanted, and ComfyKitchenAttention is the same silent-noise class as Sage on H3
(Comfy-Org/ComfyUI#15263); H3 is `SAGE_SENSITIVE` for exactly that reason and
calls `_MC.assert_sage_not_patched` at `eng_minimax_h3.py:742`.

---

## 1. FEASIBILITY -- settled, with the evidence

The install is **NOT** under `Documents\ComfyUI` (that path is the data root:
`custom_nodes/`, `models/`, `output/`, and no ComfyUI code at all). It is:

```
C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\        ComfyUI 0.34.3
  comfy_extras\nodes_minimax_h3.py                      (dated 2026-09-02)
```

**FL2V is not a new node.** `last_frame` is already an optional input on
`MiniMaxH3ImageToVideo` -- the exact node lane 19 wires as `"h3"` today. Verbatim
from the installed file:

```python
class MiniMaxH3ImageToVideo(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent."""
    ...
    io.Image.Input("first_frame", optional=True),
    io.Image.Input("last_frame", optional=True),
    ...
    if first_frame is not None:
        # geometry anchor: plain stretch to canvas
        img = _resize(first_frame[:1], width, height, "disabled")
        keyframes.append({"resolved_frame_index": 0, "image": img})
    if last_frame is not None:
        # follower: aspect-preserving cover-crop
        img = _resize(last_frame[:1], width, height, "center")
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})
    ...
    cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
```

Registered and live via `MiniMaxH3Extension.get_node_list()`.

A **more general** alternative also ships: `MiniMaxH3AddGuide` anchors an image,
a clip, or audio at an arbitrary `frame_idx` (negative counts from the end) and
chains for several anchors.

So the wiring cost is roughly one line. That is not what makes this a fork.

## 2. WHAT MAKES IT A FORK -- the pack already rejected this, on the record

`nodes/_otr_video_engines/eng_minimax_h3.py:1058`, in the docstring of
`_conditioner_nodes`:

> ``last_frame`` is deliberately never wired. It is H3's first/LAST
> interpolation endpoint, which is a different capability from the first-frame
> chaining this lane declares -- supplying it would make the render interpolate
> toward a frame the coverage planner never chose.

**Any plan that wires `last_frame` must answer that objection, not step around
it.** The question is not "can we" but "is the objection still true".

## 3. THE FACT THAT MAY DEFEAT THE OBJECTION

The objection assumes nothing chose the last frame. In the canonical graph, that
looks wrong -- **every beat's still is minted BEFORE any video renders**:

```
87 VideoDirector -> 88 ImageDirector -> 89 MetaBrief -> 90 ShotLock
   -> 91 ImageGenDispatcher   <- ALL stills minted here
   -> 92 VideoRenderBatch     <- ALL video rendered here
```

If that ordering holds, then while rendering beat *N* the still for beat *N+1*
already exists on disk, and it is a frame the planner **did** choose -- it is the
next beat's own opening image. That is the candidate answer to the 2026-09-04
objection, and **verifying or refuting it is review question A below.**

## 4. WHAT THE CONTRACT SAYS TODAY

`nodes/_otr_video_engines/frame_contract.py`:

```python
def can_chain(engine) -> bool:
    """True iff segment N+1 may begin exactly on segment N's terminal frame."""
    return frame_contract_for(engine).continuity == CONTINUITY_STRICT_FIRST_FRAME
```

Lane 19 (`minimax_h3_video`, and its sibling `h3_low_video`) declares
`continuity=CONTINUITY_STRICT_FIRST_FRAME`, so it ALREADY chains
**within** a beat: segment N+1 opens on segment N's terminal frame
(`coverage_plan.py:162`). What does not exist is continuity **across beat
boundaries**, which is exactly what the operator is asking for.

`minimax_h3_audio_in` uses `MiniMaxH3ReferenceToVideo`, which has NO
`first_frame` input at all, and is `CONTINUITY_SOFT_REFERENCE` on purpose. It is
NOT in scope; any proposal must say whether it stays untouched.

## 5. THE ASYMMETRY THE NODE IMPOSES

The two inputs are not mirror images and a plan that treats them as such is
wrong:

| input | resize mode | node's own word | pinned at |
|---|---|---|---|
| `first_frame` | `"disabled"` -- plain stretch | *geometry anchor* | index 0 |
| `last_frame` | `"center"` -- aspect-preserving cover-crop | *follower* | `frame_count - 1` |

A still that is not already canvas-aspect will be **cropped** as a last frame
where it would be **stretched** as a first frame. So the same image used as beat
*N*'s last frame and beat *N+1*'s first frame does not necessarily produce the
same pixels at the join -- which is the entire point of the feature.

## 6. THE OPTIONS ON THE TABLE

**A. Do nothing.** The recorded rejection stands. Cost: no beat-to-beat
continuity, which is the thing the operator asked for.

**B. Wire `last_frame` = the NEXT beat's minted still.** Needs the planner (not
the adapter) to own "what is the next beat's opening image", and needs section 3
to be true. Turns the last frame into a planner-chosen value, answering the
recorded objection head-on.

**C. Use `MiniMaxH3AddGuide` instead**, anchoring at an arbitrary `frame_idx`.
More general and chainable; more wiring; unclear whether the extra generality
buys anything for a two-endpoint problem.

**D. Wire it only where the next beat shares a setting/shot family**, so the
model is not asked to interpolate across a hard scene change.

## 7. QUESTIONS FOR THE PANEL

**A. (blocking) Is section 3 true?** Verify in `workflows/otr_canonical.json`
and the node implementations that EVERY beat's still is minted at node 91 before
node 92 runs any video. If stills are minted lazily per beat inside the render
batch, option B collapses and the recorded objection stands.

**B. Does pinning a last frame VOID the `strict_first_frame` claim?**
`can_chain` promises segment N+1 may begin exactly on segment N's terminal
frame. If H3 interpolates toward a pinned last frame, is the terminal frame
still the frame the next segment will open on -- or does the in-beat chain break
to buy the cross-beat one? Name the failure the coverage planner would see.

**C. What does the asymmetry in section 5 do at the join?** Is a cover-cropped
last frame and a stretched first frame a visible discontinuity, and does that
defeat the feature? Is there a canvas-aspect precondition that must gate it?

**D. Is a NEW continuity mode needed** (e.g. `strict_both_endpoints`), or does
this ride inside `strict_first_frame`? A new mode touches `frame_contract.py`,
`coverage_plan.py`, and every adapter that declares one.

**E. What happens on the LAST beat of an episode**, which has no successor
still? And on a beat whose successor is a different engine, bank, or canvas?

**F. Does this reach `h3_low_video` and `minimax_h3_audio_in`**, or is it lane-19
only? Both share the implementation class -- name what each gets.

## 8. INVARIANTS A PROPOSAL MAY NOT BREAK

* H3 stays `SAGE_SENSITIVE`; no attention swap, no quantization, no VRAM work.
* The render recipe is never traded for speed or memory.
* A wrong STRICT claim is worse than an honest jump cut -- the pack's own rule:
  *"A jump cut is honest; that is what SOFT means."*
* Nothing may reduce tomorrow's `otr/obs/` episode count.
* An `IS_CHANGED` / replay path must stay deterministic on the same seed.
