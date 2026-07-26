# JUDGMENT -- O1 the render canvas (kibitz r1 -> r4)

Full 4-round local arc, 8 agent calls: codex `gpt-5.6-sol` (high) + agy
`Gemini 3.6 Flash (High)`, both pinned and VERIFIED from
`codex_model_selected.txt` / `agy_model_selected.txt` every round. $0 external.
Driver anchor written before the r1 fan-out. Every claim below re-read against
the real Windows files at HEAD `1732896f`. Run:
`kibitz-runs/2026-07-26-o1-canvas/r{1,2,3,4}/`.

**No code was written. This is the plan and the corrections that produced it.**

---

## 1. THE BUG, STATED CORRECTLY (it took four rounds to say this right)

`eng_ltx_8gb.py:420-421`, the engine's own words:

> the NVML peak probe spans the whole render window (**telemetry only -- no
> ceiling enforcement; the operator's tier JSON owns the OOM budget**)

**The engine explicitly delegates its OOM budget to the tier JSON. The tier
JSON's canvas never arrives.** `config/profiles/otr_8gb_ltx.json:64-66` asks for
512x288; `build_request_from_shot` gives every non-face engine 1472x832
(`render_driver.py:2260-2273`) with deliberate branches after it for `ltx_video`
and `ltx_audio_in` and none for `ltx_8gb`. 8.3x the pixels, on the tier that
exists because 8 GB cannot afford them.

Root cause: the profile declared its canvas through
`render.canvas_w/h -> _otr_workflow_apply.py:519-523 -> otr_video_director.py:456
-> otr_shot_lock.py:1478,1537-1541 -> ledger.video.canonical_canvas`, which
**nothing on the render path consumes** (only `build_clip_manifest`,
`render_driver.py:3845,3963`). `otr_8gb_wan.json` closed the same gap through a
different channel -- `launch.env.OTR_VIDEO_LANDSCAPE_CANVAS: "832x480"`
(`:82`) -- and `otr_8gb_ltx.json`'s `launch.env` is empty (`:76-80`).

**This is `PBUG-20260723-02` again** (`docs/PROD_BUG_LOG.md:2679-2713`), whose
Bible rule already answers it: *a contract declared only in a process-launch
environment cannot bind work submitted to an already-running server; a per-tier
constraint has to ride the artifact the run loads.* That bug's fix defined THE
PROVEN CHANNEL, which is the one C1/C1b finished wiring three days ago.

## 2. FIVE CHANNELS NAME A RENDER CANVAS. FOUR ARE LIVE, ONE IS DEAD

| # | channel | default | scope | live |
|---|---|---|---|---|
| 1 | `OTR_VIDEO_LANDSCAPE_CANVAS` | 1472x832 | every non-face engine, episode path (`:2268`) | YES |
| 2 | `OTR_LTX_RENDER_CANVAS` | 832x480 | `ltx_video` only (`:2284`) | YES |
| 3 | `OTR_LTX_AV_RENDER_CANVAS` | 512x288 / 832x480 | `ltx_audio_in`, by talking register (`:2312`) | YES |
| 4 | `OTR_VIDEO_RENDER_CANVAS` | 832x480 | the single-engine HARNESS (`:4169`) | YES |
| 5 | `render.canvas_w/h` -> `canonical_canvas` | 832x480 | episode, per profile | **DEAD** |

**Correction to `docs/2026-07-27-7b-blockers-arc-judgment.md` section 2.** The
7d-preflight ("GPU IS PROVEN": `ltx_8gb`, 25 frames, 3004 MB) ran through
channel 4, at **832x480**. `render_single` (`render_driver.py:4151-4178`) never
calls `build_request_from_shot`. **The harness that proved the GPU renders at a
different canvas than the production path it was proving** -- the production
canvas for `ltx_8gb` has never been exercised live.

## 3. WHAT SHIPS -- the lean 7d tranche

Panel-converged (codex r4 MUST-FIX 1 + CUT 3). Touches **`ltx_8gb` only**;
`ltx_video` and `ltx_audio_in` keep their existing branches until the general
resolver lands.

1. **The resolver seam + `RenderCanvasStamp`.** One typed, serialized contract
   -- field name, keys, version, precedence, observation rule -- validated
   identically at the ShotLock WRITE boundary and the `build_request_from_shot`
   READ boundary. It validates positive dims, **/32 latent-grid divisibility**
   (the `ltx_av` comment records 1280x720 failing that gate), and
   engine-supported values. Nothing today does: `schemas.py:103-109` declares
   bare ints, `otr_video_director.py:259-261` allows anything >= 16, `_dims`
   returns them unchecked (`wan_shared.py:364-370`).
2. **`ltx_8gb` declares its render canvas STATICALLY** -- beside its existing
   `frame_contract`, `render_aspect` and `target_fps`. Not an env var, not a
   ledger read, not a fourth inline branch.
3. **ShotLock builds + stamps the REAL rows BEFORE cast preflight.** Today
   `_assert_family_inputs_satisfiable_cast_time` runs at `:1256-1261`, before
   real rows exist at `:1266-1326`, against a temporary shot (`:978-985`) that
   carries no stamp. Both seats found this independently; it would have crashed
   every run.
4. **`build_request_from_shot` consumes ONLY the stamp for stamped lanes**, and
   raises `RenderError` on missing/malformed -- never a fall-through to the
   1472x832 default. **No catch surgery is needed:** `otr_shot_lock.py:990-1003`
   already re-raises `RenderError` and swallows only `ValueError` / broad
   `Exception`. (This CORRECTS the driver's own r2 plan, which called for
   un-swallowing.)
5. **A REAL observation.** `eng_ltx_8gb.py:463` builds `render_canvas` from the
   REQUESTED dims computed at `:412` -- comparing a stamp against it is a
   tautology, and a wrong-size render would pass the gate. Probe after
   `canonicalize()`, populate the EXISTING typed `CanonicalClip.w/h`
   (`schemas.py:216-247`) with probed dims, keep the legacy `render_canvas`
   STRING for compatibility, and probe the assembled multi-clip output again
   after `assemble_beat_segments` -- copying the last segment's metadata is not
   an observation of the assembly. `ffprobe_clip_fields` already returns
   `width`/`height` from the stream read `eng_ltx_8gb` already performs at `:456`.
6. **A drift guard on the dead channel.** A test pinning
   `otr_8gb_ltx.json`'s `canvas_w/h` equal to the engine declaration, so the two
   cannot silently disagree while the profile channel stays unconsumed.

**Multi-clip:** ONE shot-level stamp inherited by every segment; only
observations are per-segment (codex r3 CUT 4).

## 4. WHAT IS DEFERRED, AND WHY (each its own chunk)

* **The general profile-override widget.** `canvas_w`/`canvas_h` are REQUIRED
  Director widgets with unconditional 832/480 defaults
  (`otr_video_director.py:259-260`), so "absent" is not representable and
  `max_render_frames`' `0 = unpinned` sentinel has no analogue. Shipping the
  widget properly means `capability_profiles.py`, `widget_mapping.json`,
  `_otr_workflow_apply.py`, variant generation, canonical node 87 (15 -> 16
  positional values, appended LAST) and **all eleven variant workflows**. Real
  work, not the 7d blocker.
* **`render_single` parity + the 129 direct `build_request_from_shot` calls in
  `tests/`.** Useful, not on the authoritative ledger path.
* **The recipe/register ownership conversion.** GO_FORWARD carries it as an OPEN
  OPERATOR DECISION; codex r4 is right that the canvas fix may not decide it by
  accident. Only the recipe SELECTOR needed to resolve a canvas is recorded, and
  compared live before rendering.
* **`VideoLedgerSection`** -- a DEAD schema (driver finding, neither seat).
  Referenced only by `tests/test_video_retry_taxonomy.py:156` and
  `tests/test_video_schemas_additive.py:135`; it is `extra="forbid"` yet ShotLock
  stamps eight fields it never declares (`policy_version`, `device_policy`,
  `dtype_policy`, `max_render_frames`, `roles_effective`,
  `routing_env_snapshot`, `clip_budget`, `warnings` -- agy's inventory).
  Tombstone or repair as its own item; O1 uses a narrow boundary validator.
* **`canonical_canvas`** -- legacy parsing only, explicitly non-authoritative,
  NEVER a fallback for a missing stamp.

## 5. CLAIMS THAT DID NOT SURVIVE

**Driver, refuted by the panel (5) -- all verified:**

1. **"`ltx_8gb` is gated by `compute_real_frame_budget`" -- FALSE, and it
   demolished the driver's whole r1/r2 arithmetic.** That function is called by
   exactly ONE engine, `eng_wan_ti2v.py:399`. `eng_ltx_8gb` declares "NO
   VRAM/NVML/vendor gate" (`:33`, `:264`) and treats the NVML probe as telemetry
   only. So the 43,276 / 19,657 / 12,455 MB budget table does not apply to this
   engine: **the real failure mode at 1472x832 x 161 frames is a CUDA OOM
   mid-render, not a clean preflight refusal.** Worse, not better.
2. **"22 of 23 profile stamps are wrong" -- FALSE.** `otr_16gb_ltx_video.json`
   routes every visual role to `ltx_video`, which IS forced to 832x480, so its
   stamp is CORRECT; likewise `otr_16gb_ltx_audio_in.json` in one register. The
   true statement: the stamp is wrong for every profile whose routed engines have
   no per-engine branch.
3. **"1472x832 is the deliverable" -- FALSE.** `widget_mapping.json:343-348` maps
   `render.composite_w/h` onto `OTR_SilentComposite`, and profiles ship
   `composite_res 1920x1080`. 1472x832 is only the node's class default.
4. **"512x288 was adopted with no recorded ladder walk" -- FALSE**, but thin:
   `docs/2026-07-20-OTR-video-tiers/ltx_8gb_discovery.md:59-69` records a PASS at
   512x288 x **9 frames**, "the legal minimum". Functional, never qualified at
   production length.
5. **"Compare the stamp against the adapter's `render_canvas`" -- CIRCULAR.** It
   is derived from the request, not the media.

**Driver self-correction, before the panel saw it (1):** the 7d-preflight
rendered at 832x480, not 1472x832 (section 2).

**Panel, refuted by the driver (4) -- all verified:**

1. **agy r1 MUST-FIX 1+2 -- make `launch.env` the render-time canvas authority.
   REFUTED by `PBUG-20260723-02` with a production receipt.** agy proposed as
   the FIX the exact defect this project already paid for.
2. **agy r1 CUT 2 -- "fallback rendering" instead of failing closed. REJECTED
   against project law.** S4 (2026-07-10) made the budget RAISE rather than
   resize; the coverage block exists to remove silent degradation. The one panel
   line that would have done real damage.
3. **agy r2 SHOULD-FIX 1 -- `profile.get("canvas_w")` reads in
   `render_driver.py`. MISREAD:** zero occurrences of `canvas_w`/`canvas_h` in
   that file.
4. **agy r3 SHOULD-FIX 2 -- default `render_canvas` on every test fixture.
   ACCEPTED ONLY WITH A CARVE-OUT:** a blanket default masks the very
   missing-stamp failure being introduced. The missing-stamp control must use a
   fixture that deliberately omits it.

**Also rejected:** adding `FRAME_COST_MODEL["ltx_8gb"]` (dead code -- no caller);
nested `observed_canvas` (`CanonicalClip.w/h` already own that); folding the
canvas into `render_request_hash` (it becomes the render seed at
`render_driver.py:2623-2630` -- it would silently re-seed every shot).

## 6. THE CLOUD TRAP (codex r3 MUST-FIX 4 -- would have been a real regression)

`requested == observed` is **FALSE BY OPERATOR DESIGN** on cloud lanes.
`eng_cloud_video.py:458-476` calls `cloud_delivery_wh(...)`: *"TRUE 1080p cloud
delivery (operator 2026-07-03): conform the provider clip to a real 1080p canvas,
NOT the smaller per-family request canvas."* Directory clips
(`eng_mesh_stage.py:782`) cannot be ffprobed at all. Hence three typed canvases
-- `request_canvas` / `expected_delivery_canvas` / `observed_canvas` -- with each
adapter declaring which one observation must equal, and delivery dims resolved
BEFORE any paid call.

## 7. ACCEPTANCE / CONTROL MATRIX

Must PASS (a tightening that refuses honest input is not a fix): `ltx_video`
@ 832x480; BOTH `ltx_audio_in` registers; `still_pan` / `viz_*` @ 1472x832;
`audio_driven_face` portrait @ 480x832; cloud lanes whose delivery canvas
deliberately differs from the request.

Must FAIL, each its own mutation: missing stamp; malformed type; non-positive
dims; non-/32 dims; stamped-vs-observed mismatch on a lane declaring equality;
a legacy ledger with no stamp; a surviving ambient override on the `ltx_8gb`
path. Plus a non-square directory clip (e.g. 640x352) proving the observer never
reverses w/h, and multi-clip segment AND final-assembly equality.

## 8. QUALIFICATION OWED BEFORE THE VALUE IS LOCKED

Both seats, independently. `ltx_8gb` at 512x288 vs 832x480, walked upward from
short legal lengths, fresh-server VRAM peak, completion, and an **operator look
call** -- 512x288 upscales ~3.75x to a 1920x1080 deliverable and BUG-LOCAL-412
puts LTX-2B's quality floor at 832x480, so the cheapest canvas may not be the
right one. 1472x832 is a REFUSAL/OOM control, never a production-length forward.

## 9. HOUSEKEEPING

`.kibitz/comfyui.local.md` claims 57 canonical links; `otr_canonical.json` has
**56** (23 nodes; node 87 = `OTR_VideoDirector`, 15 positional widget values,
`max_render_frames` last). Stale local profile -- it feeds every future arc.

## 10. DOCTRINE FOR THE LOG

**A configured value with a severed channel is this build's recurring defect
class -- this is the third instance in four days.** C1 (node 87's
`max_render_frames` descriptor), C1b (the orphan `17` in eleven variants), and
now the canvas. The portable rule already exists in `PBUG-20260723-02`; what is
new is the corollary: **when one tier closes a gap through a different channel
than another tier, the channel that "worked" is not vindicated -- it is the next
bug.** `otr_8gb_wan.json` worked by accident of picking the live channel.

**And: a harness that exercises a different code path than production does not
prove production.** The GPU proof rested on `render_single`, which resolves its
canvas through a channel no episode ever touches.
