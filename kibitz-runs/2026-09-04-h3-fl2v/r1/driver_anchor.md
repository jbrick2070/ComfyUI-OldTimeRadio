# Driver anchor -- r1, H3 FL2V on lane 19

**Driver:** Claude (Opus 5) in Cowork. Sole judge. Written BEFORE the fan-out.
**Every claim below is labelled CONFIRMED / MISREAD / UNVERIFIABLE against files
I actually read on this Windows box.**

## VERDICT

**The plan's feasibility section is CONFIRMED and its central objection is
ANSWERABLE.** I would not ship option A (do nothing). Option B is viable and the
2026-09-04 rejection docstring that forbids it rests on a premise I can now show
is false. But B is NOT a one-line wiring change, and anyone who prices it that
way has missed the contract question in section 7B.

## CONFIRMED

1. **The installed node exposes `last_frame`.** CONFIRMED by reading
   `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_minimax_h3.py`
   (ComfyUI 0.34.3). `MiniMaxH3ImageToVideo.define_schema` declares
   `io.Image.Input("last_frame", optional=True)`; `execute` pins it at
   `resolved_frame_index = frame_count - 1` into `minimax_keyframes`. The class
   docstring names the mode: `"t2va and fl2va"`. It is registered in
   `MiniMaxH3Extension.get_node_list()`.
   **Note for reviewers: this file is OUTSIDE the repo.** `Documents\ComfyUI` is
   the data root and contains no ComfyUI code. Do not report the node as absent
   because you looked in the repo.

2. **Lane 19 wires only `first_frame` today.** CONFIRMED,
   `eng_minimax_h3.py:1074` -- `"first_frame": W("loadimage", 0)`, and the
   `"h3"` class maps to `MiniMaxH3ImageToVideo` at `:1015`.

3. **The rejection is on the record.** CONFIRMED, `eng_minimax_h3.py:1058-1063`.
   Its stated reason: supplying `last_frame` "would make the render interpolate
   toward a frame the coverage planner never chose."

4. **THE OBJECTION'S PREMISE IS FALSE, and this is my main contribution to r1.**
   CONFIRMED at `otr_video_render_batch.py:502-509`: there is an explicit
   `image_done` gate, wired from `OTR_ImageGenDispatcher.image_done`, whose own
   tooltip says the video render "cannot start before every episode still exists
   on disk (W4)". So while beat *N* renders, beat *N+1*'s still is already on
   disk **and it was chosen by the planner** -- it is the next beat's own opening
   image. The 2026-09-04 docstring is not wrong about the danger of an arbitrary
   endpoint; it is wrong that no such planner-chosen frame exists.
   **This does not by itself authorize the change** -- see MUST-FIX 1.

5. **Lane 19 declares `CONTINUITY_STRICT_FIRST_FRAME`.** CONFIRMED,
   `eng_minimax_h3.py` frame_contract block; and `frame_contract.py:327`
   `can_chain()` returns True only for that mode, meaning "segment N+1 may begin
   exactly on segment N's terminal frame."

6. **The two inputs are asymmetric.** CONFIRMED in the node source:
   `first_frame` -> `_resize(..., "disabled")` (plain stretch, commented
   "geometry anchor"); `last_frame` -> `_resize(..., "center")`
   (aspect-preserving cover-crop, commented "follower").

7. **`MiniMaxH3AddGuide` is real and more general.** CONFIRMED -- arbitrary
   `frame_idx`, negative counts from the end, chainable, needs the video VAE.

## MUST-FIX (the plan may not proceed to code without these)

1. **The in-beat chain and the cross-beat pin may be in direct conflict, and the
   plan does not say so.** `can_chain` already promises that segment N+1 opens on
   segment N's TERMINAL frame. If beat *N*'s final segment is also pinned to
   interpolate toward beat *N+1*'s still, then that segment's terminal frame is
   being pulled toward a different image than the in-beat chain expects. Either
   the cross-beat pin belongs ONLY on a beat's LAST segment, or the two
   mechanisms fight. The plan must state which segment carries `last_frame` and
   prove the other segments are untouched.

2. **A pinned last frame may not silently keep the STRICT claim.** If H3
   interpolates toward an endpoint, "frame 0 is locked to the supplied image" may
   still hold while "the terminal frame is what the next segment will open on"
   quietly stops holding. The pack's own rule is that a wrong STRICT claim is
   worse than an honest jump cut. The plan must either prove STRICT survives or
   propose the honest contract change (question 7D).

3. **Section 5's asymmetry is a correctness issue, not a cosmetic one.** The
   SAME still used as beat *N*'s `last_frame` (cover-cropped) and beat *N+1*'s
   `first_frame` (stretched) yields DIFFERENT pixels unless it is already
   canvas-aspect. The join would then be visibly discontinuous -- defeating the
   entire feature -- on exactly the lanes whose stills are not canvas-aspect. The
   plan needs either a canvas-aspect precondition or a pre-resize owned by the
   planner.

## SHOULD-FIX

4. **The last beat, and heterogeneous successors.** No successor still exists for
   the final beat; and a successor on a different engine, canvas, or bank is not
   a legitimate interpolation target. Needs an explicit "no pin" path, not a
   crash.

5. **Name the blast radius across the three H3 ids.** `minimax_h3_video`,
   `h3_low_video` and `minimax_h3_audio_in` share one implementation class.
   `minimax_h3_audio_in` uses `MiniMaxH3ReferenceToVideo`, which has NO
   `first_frame` input at all -- so it cannot take this change even in principle.
   The plan must say what each of the three gets.

6. **Determinism and replay.** A pinned last frame becomes part of the render
   inputs. Replay and `IS_CHANGED` must both see it, or a replayed episode
   silently renders differently from the one that was approved.

## UNVERIFIABLE from here -- do not let the panel assert these as fact

* Whether H3 actually produces a *seamless* join when both endpoints are pinned.
  That is a render result, not a code fact; it needs a leg on the 5080. No
  reviewer may claim it either way from source.
* Whether `minimax_keyframes` with two entries changes step scheduling or cost.
* Whether the trained frame range (`~124-362`, per the node's own `length`
  tooltip) interacts with two-endpoint conditioning.

## OUT OF SCOPE -- reject any proposal touching these

W4A8 quantized H3 DiT, Qwen3-VL 4B INT8 + ClipProj, INT8 ConvRot VAE,
ComfyKitchenAttention, the Spectrum pack. H3 fits the 4060 at 7.21 GiB cold /
6.79 warm already; ComfyKitchenAttention is the same silent-noise class as Sage
(Comfy-Org/ComfyUI#15263) and H3 is `SAGE_SENSITIVE` for that reason.
