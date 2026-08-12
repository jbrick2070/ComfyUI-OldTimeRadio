# VIDEO_LANE_PREFLIGHT receipt -- lane 10, `mesh_stage`

`VIDEO_LANE_PREFLIGHT receipt: mesh_stage | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane10_mesh_stage/ | verdict PASS`

The most defective lane left in the roster -- four red gates, and the only lane
in the tree that delivers a frame DIRECTORY rather than an mp4, which is what
made its audio gate a design question rather than a copy of what nine lanes
before it did.

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 weights resolve | **RED** | PASS | node gate at preflight + `folder_paths` and the configured models root in the resolver |
| G2 canvas truth | **RED** | PASS | `render_canvas = (1472, 832)` declared, the inline sniff deleted, the profile moved to agree |
| G3 contract vs runtime | **RED** | PASS | `continuity=CONTINUITY_NONE` passed at the shared declaration, with the reason |
| G4 admission honesty | PASS | PASS | untouched |
| G5 audio law (V-1) | **RED** | PASS | the directory contract PROVES each frame from its magic bytes; G5 taught that name |
| G6 guards | PASS | PASS | untouched |
| G7 public surface | PASS | PASS | `ENGINE_MATRIX.md` regenerated (the canvas row moved to `declared`) |

Four `EXPECTED_RED` entries deleted for this lane, plus **four more** for
`still_motion` / `still_pan` / `still_flat` / `still_word` -- see G3 below. The
strict unexpected-pass gate requires exactly that: a shared fix that flips other
lanes green must remove their entries in the same commit.

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures** --
green, so nothing red was inherited and everything below is this lane's.

## G1 -- two defects in one row, and the second one was live on this box

**(a) S8b-16, the node gate.** `_node_candidates()` named ten core hy3d classes
and they resolved ONLY inside `load()`, so `assert_usable` passed on a box
without them and the render died mid-beat, after the checkpoint had been paid
for. The gate now runs in `assert_usable`, and three properties are pinned
rather than asserted-by-existing (lane 8 built this shape first and its lessons
are why): it runs BEFORE weight resolution, it reads the ACTIVE candidate set,
and it collects EVERY miss before raising. The ordering test makes weight
resolution RAISE `RuntimeError`, so a mis-ordered gate fails with the wrong
exception type instead of passing quietly.

**(b) Lesson L1 / Bug Bible 12.88 -- and THE FIRST VERSION OF THIS SECTION WAS
WRONG. Corrected 2026-08-11 after the operator said 3D had been working.**

The original claim here was "this lane was DEAD on this box". **It was not.**
The operator was right and the diff proves it: `_ckpt_path` is BYTE-IDENTICAL to
the version running at `37254f39` when mesh_stage was rendering in June, so
nothing regressed -- and under the soak launcher the old resolver FOUND the
weight every time:

```
launcher sets  HF_HOME = C:\ComfyUI-Models\huggingface
old 2nd probe  dirname(HF_HOME) + "checkpoints" = C:\ComfyUI-Models\checkpoints
               hunyuan3d-dit-v2-mv.safetensors   exists = True     <-- found
old 1st probe  <comfy_root>/models/checkpoints    exists = False
```

**What my probe actually measured** was a BARE SHELL, where `HF_HOME` is unset
because the launcher (not the User env) sets it. That is a real defect, but a
narrower and different one -- and it is precisely what Bible **12.88** is about:
*"where would the LOADER find this?"* (`folder_paths`, in-process only) and
*"is this weight on this box?"* (must be answerable OUTSIDE one) were sharing a
single probe, so every off-runtime caller -- the CPU suite, the preflight
matrix, any "is this lane installed?" tool -- got a confident wrong NO. A
checkpoint registered only via `extra_model_paths.yaml` was invisible in-process
too, since `folder_paths` was never asked at all.

**The fix is unchanged and still correct** -- env pin, then `folder_paths`, then
the historical dirs, then `configured_models_root()` LAST, additive by
construction so it can only turn a false negative into truth. What changes is
the SEVERITY and the lesson: this was never a production outage, and the smoke
below is not "the first time this lane rendered". It is the first time the lane
answers honestly to something that is not a running ComfyUI.

**Why the wrong version got written, since that is the transferable part:** the
probe was run in the environment that was convenient (a bare shell) rather than
the one production uses (the launcher), and a False came back. L1's own runnable
check says to resolve "with NO environment variables set at all" -- which is the
right test for the OFF-RUNTIME question and the wrong one for "is the lane
broken". Both questions needed asking; only one was.

The fix reuses lane 1's shared answer rather than writing a third resolver:
env pin -> `folder_paths` -> the historical dirs -> `wan_shared.configured_models_root()`
LAST. Additive by construction, so it can only turn a false negative into truth.

## G2 -- the canvas, and an inline branch that had to die

The lane declared no `render_canvas` and chose its size with a magic-number
sniff inside `render_clip`: *if the request says 832x480 and carries no explicit
canvas, rewrite it to 1472x832*. That is the same shape lane 7 deleted. It fired
only on paths that reached `render_clip`, and everything upstream that reads
`request.canvas` -- admission, still sizing, composite scaling -- still saw the
number the sniff was silently overruling.

`render_canvas = (1472, 832)` is now declared. It describes the RUNTIME, which
is L2's rule: Blender is told exactly these dimensions and `validate_frame_dir`
refuses to publish a frame that is any other size, so the declaration and the
pixels cannot disagree without a raise. /32-legal on both axes (46x32, 26x32).
L13's /64 rule does not reach this lane -- no halved stage, no fixed-x2
upsampler -- though 1472x832 satisfies it anyway.

`config/profiles/otr_w45_mesh_stage.json` said **832x480**. It now carries
1472x832 as a DRIFT GUARD, and the variant was regenerated in the same change.

**CORRECTION, traced end to end at the start of lane 11 and folded back here:
that profile channel is NOT dead, and the corpus wording -- "read by nothing" --
is wrong.** `_otr_workflow_apply.py` flattens `render.canvas_w/h` into the
node-87 `OTR_VideoDirector` widgets, and `otr_video_director.py` turns those
widgets into `request["canvas"]`. Measured on this lane's own regenerated
variant: node 87's widgets moved from `25, 832, 480` to `25, 1472, 832`. What
really happens is that `build_request_from_shot` then OVERWRITES the request
canvas to the landscape default for every non-face family, and the declaration
overrules that in turn.

Same OUTCOME as a dead channel -- the profile number never decides the render,
so the fix and the reasoning for declaring were both right -- but a materially
different failure mode, and the next lane needs the accurate one: an operator
who edits that field watches the node-87 widget change and reasonably concludes
it took effect. "Dead" would have told lanes 11-18 to document a channel they
should instead keep truthful. Per lane 4's G2.3 every profile resolving to
this engine was enumerated, not just the obvious one -- there is exactly one,
and the test asserts the enumeration is non-empty so G2.3 cannot go vacuous.

**The "declares NOTHING" differential control moved again**, for the fourth
time: wan_ti2v -> wan_i2v -> mesh_stage -> **still_pan**. The invariant outlives
every occupant, so the test is edited and never deleted.

## G3 -- fixed at the shared mechanism, which took four other lanes with it

`_CheapFamilyBase.frame_contract` never passed `continuity=`, so
`CONTINUITY_NONE` was a dataclass DEFAULT. The comment above it had REASONED
about continuity for months, which is exactly what makes this a declaration bug
rather than a wrong value -- the right answer arrived at by nobody deciding it
has the same shape a wrong one would have had.

L13 says fix the shared mechanism and sweep every adapter sharing it before the
lane closes. Six lanes read this contract, so the keyword landed on the base and
the four still lanes' G3 rows came out of `EXPECTED_RED` in the same commit.
**The four visualizers and `google_omni_video` are NOT reached** -- each
inherits the default through its own contract -- and they stay red, because a
shared fix that did not reach them must not be reported as though it had.

For `mesh_stage` NONE is honest for a reason of its own, and it is written at
the declaration rather than implied: `build_blender_cmd` takes `start_angle` /
`arc_degrees`, so a chained successor would need the predecessor's terminal
ORBIT ANGLE threaded forward, and nothing threads it. A chain would snap the
camera back to the arc's start at every segment boundary.

## G5 -- the interesting one: do NOT bolt an mp4 probe onto a PNG directory

G5 is a LEXICAL gate -- it greps the canonicalize path for
`validate_silent_clip_contract`, which ffprobes an mp4 for audio streams. This
is the only directory-clip lane in the roster (`"type": "directory"`,
straight-alpha PNGs). There is no container to probe, and adding an ffprobe call
to satisfy a string match would have proved nothing about anything.

What was actually wrong was one level down. `validate_directory_clip` did check
`has_audio`, but it read the field off **the dict this adapter itself wrote** --
a declaration checking a declaration, which is precisely L4's complaint. And
`list_directory_frames` accepted frames by FILENAME EXTENSION, so a file named
`0001.png` containing an mp4, or a WAV, counted as a frame and shipped as proof
of silence.

The root fix makes the contract PROVE the artifact:
`prove_frame_is_a_silent_image` reads each frame's MAGIC BYTES and refuses
anything that is not really a PNG/EXR. A still image has no audio stream to
carry, so silence becomes a structural fact about the bytes rather than a naming
convention. It lives inside `list_directory_frames`, so every consumer inherits
it -- including the tolerant `frame_dir_summary` that the manifests and
`_clip_summary` read, which would otherwise have let a receipt call an impostor
directory real.

Then G5 was **taught the name**, per lane 10's mapping
`DIRECTORY_CLIP_AUDIO_LAW = {"mesh_stage": "validate_directory_clip"}`.
Teaching a gate a new name is the sanctioned move (L9: G1 was taught
`_resolve_unet` rather than widened); accepting "some validator, whatever it is
called" would let a future lane launder a missing proof past it. And because the
teaching buys a green row for a STRING, a twin assertion exercises the named
function directly: a file called `0001.png` whose bytes are an mp4 `ftyp` box
must be REFUSED. Revert the magic-byte read and that test goes red.

## The solo smoke -- LIVE PASS

Stock `default` boot (no token needed: `requires_flag is None`, the registry IS
the menu). Box reset first per CLAUDE.md section 4 -- no resident server, port
8000 clear, VRAM at 1,265 MiB.

**All ten hy3d classes confirmed PRESENT on the RUNNING server** via
`/object_info` (2,206 classes) before submitting, so a fail-closed refusal and a
missing install could never be confused (lane 7's lesson 4).

**The still is `mesh_fodder`, not a scene still.** `requires_mesh_fodder = True`
exists because handing this lane a cinematic scene still meshes the whole
environment into a clay blob -- and a smoke fed the wrong still renders a blob
and still exits 0. So the fodder was minted from the lane's OWN `still_plan`
row's `framing_geometry`, verbatim, through the shipped `z_image_turbo` recipe.

| Item | Value |
|---|---|
| Harness | `_otr_single_engine_smoke.py --engine mesh_stage --frames 50 --portrait <fodder>` |
| Prompt id | `0bb42bcf-6c67-4a31-9834-eb7abd06fb60` |
| Wall time | **41.5 s** cold (mesh cache MISS -- the mesher actually ran) |
| Canvas PROBED | **1472x832 on every one of the 50 frames** -- equals the declaration |
| Frames PROBED | **50**, exactly the ledger target |
| Pixel format | RGBA on every frame, straight alpha |
| Audio | **structurally impossible** -- all 50 frames magic-byte proved PNG |
| Chain | cache MISS -> hy3d-2mv mesher -> VRAM barrier -> cube self-test PASSED -> Blender |
| Clip | `type: directory`, 21,236,673 bytes |
| Artifact | `.../_shared/mesh_cache/stages/stage_uncast__308bd10d1ca6819b__hy3d2m_d3sb91ix` |
| Frames sha256 | `be34304891745f3d8c8f73110ddf5e6650713e566035bbf1f6e670ae3a294100` (all 50, name order) |
| Viewable | `.../_lane_smokes/lane10_mesh_stage/lane10_mesh_stage_preview_f50.mp4` (alpha flattened over black; the DELIVERED clip is the directory) |

The `subject_id='uncast'` warning in the log is CORRECT and not a defect: a solo
smoke builds `shot_id="single_0000"` with no `char_id`, and the adapter is
supposed to say so LOUDLY rather than cache a real subject under a shared id.

## Deliberately NOT done here, and why

**No cost row, no VRAM peak.** `_directory_clip` returns no `vram_peak_mb` /
`recipe` / `quant` / `render_canvas`, so the smoke report carries
`vram_peak_mb: null` and this receipt claims **no peak at all**. That is L4's
RECEIPTS half; G4 is green because the manifest already records this lane as
admission-unenforced in words, which is honest. Recorded as an open observation
rather than fixed, because the lane's four red gates were the packet and the
peak needs a `VramPeakProbe` threaded through a torch mesher AND a Blender
subprocess -- two different measurement surfaces (L7), which is a measurement
design question, not a passthrough. **Do not seed a cost row from anything in
this receipt.**

**No naming change.** `mesh_stage` has no public row and no `low`/`high` marker,
and inventing one would need a measured peak -- see above. The internal id
stands.

**The canonical workflow is untouched.** No node, wiring or widget change: this
lane's registration, roles and node-87 string are all unchanged.
