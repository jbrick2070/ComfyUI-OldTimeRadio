# r1 judgment -- H3 FL2V

**Driver:** Claude (Opus 5), Cowork. Sole judge.
**Panel (2 lanes, the operator's cap):** Codex `gpt-6-astra` @ reasoning `ultra`;
Cursor `cursor-grok-4.6-high` (ask mode). Antigravity was DELIBERATELY NOT RUN:
it is the only unsandboxed lane and the tree carried an uncommitted migration.
Claude did not review itself.

**Both lanes returned VERDICT: no, independently.** So did I, on re-reading. The
capability is there; the plan was not.

## What I got WRONG in my own anchor -- all three CONFIRMED against the files

1. **My section-3 finding does not defeat the objection.** I claimed the
   `image_done` barrier answers "a frame the coverage planner never chose".
   Both lanes caught the gap and Cursor put it best: *"Coverage plans lengths and
   joins, not images."* Node order proves the still EXISTS; it does not prove the
   PLANNER CHOSE IT AS AN ENDPOINT. ShotLock/ImageGenDispatcher choose it as beat
   N+1's OPENER. The objection stands until an owner stamps it as beat N's END.
   CONFIRMED -- links 260/267 do wire 91 -> 92, so the barrier half was right;
   the inference from it was not.
2. **My node-order diagram was REVERSED.** I wrote `89 MetaBrief -> 90 ShotLock`.
   Link 255 is `node 90 (OTR_ShotLock) -> node 89 (OTR_MetaBriefImagePromptGen)`.
   CONFIRMED by parsing the canonical JSON. (I copied the ordering from
   `GO_FORWARD_PLAN.md`, which carries the same reversal -- worth fixing there.)
3. **`h3_low_video` is an ALIAS, not a sibling.** `public_engines.py:128` maps
   `"h3_low_video": "minimax_h3_video"`. One class, two public ids. CONFIRMED.

## CONFIRMED findings that change the design

**C1 -- THE PINNED FRAME IS NOT ALWAYS PUBLISHED (Codex).** Verified by executing
the pack's own functions, not by reading them. H3 renders at 24 fps and publishes
at 25; `canvas_frames_for_model` FLOORS. On 3 of the first 8 legal rungs the last
canvas frame maps to a model frame BEFORE the pinned one:

```
model 141 -> canvas 146, last canvas frame = model 139, pinned 140  NOT SHOWN
model 158 -> canvas 164, last canvas frame = model 156, pinned 157  NOT SHOWN
model 209 -> canvas 217, last canvas frame = model 207, pinned 208  NOT SHOWN
```

This is decisive on its own: on those rungs you buy FL2V conditioning and the
viewer never sees the pinned frame, so the join is still discontinuous -- and
silently so. **Any build must define the endpoint at the FINAL VISIBLE boundary
and map it back through segmentation and frame conversion.**

**C2 -- THE FEAR CAPE OWNS THAT EXACT SEAM (Cursor).**
`render_driver.py:4873-4891`: on a beat with enough segments the still handed to
the FINAL segment is INVERTED (`negate_image`), and the comment says it
*"knowingly breaks the seamless cut at that one seam -- that IS the effect."*
Default ON. So FL2V would try to manufacture continuity at precisely the seam the
pack deliberately breaks, interpolating inverted-current -> next-beat still.
CONFIRMED. These two features are in direct conflict and one must yield.

**C3 -- `accepts_last_frame` ALREADY EXISTS AND FAILS OPEN.** `render_driver.py:2148`
resolves it through `_accepts(attr, default=True)`; the only engine that declares
it is `eng_google_veo_video.py:544`. H3 declares nothing, so **H3 is currently
treated as ACCEPTING a last frame it never wires.** Today that is inert because
nothing populates `asset_refs.last_frame` -- but the safety rests on absence, not
on a declaration. CONFIRMED. Worth an explicit `accepts_last_frame = False` on
H3 regardless of whether FL2V is ever built.

**C4 -- SEGMENT GRANULARITY (both lanes).** Intra-beat chaining replaces each
successor segment's init_image with the predecessor's extracted terminal frame
(`render_driver.py:4731-4758`). A naive `last_frame` on every segment would pull
every segment toward the same future image, and on a chained segment first and
last could become the same image -- which `render_driver.py:1216-1217` already
forbids as the held-frame case. `can_chain` / `join_mode_for` never inspect
`last_frame`, so the planner would not see the failure. CONFIRMED.

**C5 -- A TEST PINS TODAY'S BEHAVIOUR.**
`tests/test_minimax_h3_video.py::test_the_graph_wires_the_still_as_the_FIRST_frame_and_never_the_last`
asserts `"last_frame" not in h3_node`. Any build rewrites it deliberately, with
the reason. CONFIRMED.

**C6 -- MY ASYMMETRY SECTION STOPPED TOO EARLY (both lanes).** H3 already routes
its still through `_materialize_init_image`, which applies the canvas pad/crop
policy before the node sees it (`eng_minimax_h3.py:1052`, `wan_shared.py:568-612`).
So the stretch-vs-cover-crop difference is a no-op IF both sockets are
materialized, and a real join bug only if `last_frame` is raw-loaded. The fix is
"materialize both", not "add a canvas-aspect precondition". CONFIRMED.

**C7 -- SUCCESSOR STILLS ARE NOT UNIVERSAL (Codex).** The dispatcher skips
non-consuming roles and permits recorded model-refusal GAPS; the renderer skips
gap beats while keeping their timeline positions. So "every beat has a still" is
false, and a build must resolve the immediate VISIBLE successor after image
receipts are known -- never search past a gap, never turn a sanctioned gap into
an episode failure. CONFIRMED in principle from the cited call sites;
[VERIFY-AT-BUILD] the exact gap-skip semantics.

**C8 -- DETERMINISM (both lanes).** The render receipt hashes only the OPENING
still, so changing beat N+1's image would NOT invalidate beat N's cached render.
`IS_CHANGED` hashes the whole ledger, so the successor identity must be stamped
on the shot, not derived inside the adapter. CONFIRMED.

## ACCEPTED CUTS (both lanes agreed; I agree)

* **No new continuity mode.** `strict_both_endpoints` would force
  `coverage_plan.py` and every adapter to model a join the partitioner does not
  have. Lane 19 keeps `CONTINUITY_STRICT_FIRST_FRAME`; a last frame is an
  optional last-segment conditioner, not a continuity mode.
* **Cut `MiniMaxH3AddGuide`.** It consumes an existing AV latent + conditioning
  and is not in this lane's `_EXTRA_NODE_CANDIDATES`. `last_frame` is the native
  socket for two endpoints.
* **Cut setting/shot-family gating.** No FL2V family classifier exists and
  `scene_id` is not a motion-join signal; building a "same setting" oracle is a
  new product. The engine/canvas/still gate is enough.

## UNVERIFIABLE -- not facts, and no lane may assert them

* Whether H3 produces a perceptually seamless join with both anchors. Needs a
  leg. The SSIM 0.900 / PSNR 33.4 dB the lane cites is an INTRA-beat first-frame
  number and is not a cross-beat FL2V measurement.
* VRAM cost of a second never-denoised keyframe. The 7.21/6.79 GiB ladder is
  first-frame-only; [ASSUMPTION] flagged by Cursor, and it must be measured at
  the same 864x480 rung before anyone says the 4060 fit is unchanged.

## VERDICT CARRIED TO r2

**Option B is not dead, but it is not "one wire" and the plan as written is
rejected.** The shape that survives r1:

> A PLANNER-OWNED boundary decision, resolved after image receipts, that stamps a
> successor-boundary image on the shot; applied ONLY to the segment owning the
> final visible boundary; defined at the final VISIBLE frame and mapped back
> through the 24->25 conversion; mutually exclusive with the fear cape on that
> beat; materialized through the same path as the init image; with the successor
> identity bound into the render receipt and replay identity.

r2 (coding plan) must price that, and must answer C1 and C2 first -- if the
endpoint cannot be guaranteed visible, or cannot coexist with the fear cape,
the honest answer is to stop.
