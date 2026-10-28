# Proposal: getting each beat's own objects onto the AnimateDiff picture

Status: review arc complete (four rounds, Codex and Antigravity on every round, Claude as anchor and
sole judge). Build authorized by operator relay on 2026-10-27 as opt-in, default off. ARM R IS BUILT AND
PUSHED (856ce263, fixes cb233177): two lab engine ids, tests, and a first paired replay, results in
section 6A. Arms P and L are NOT built. Written 2026-10-27 against main at 0aa61970; this replaces the
first version of the file, which is in git history.

## 0. What changed since the first version of this file

1. The recipe's reference pictures ARE on this box, and they are not abstract colour fields. The first
   version said otherwise and was wrong (section 3).
2. The operator's priority moved. Style is a light touch; what matters is that the objects, the place
   and the dramatic situation the beat's dialogue names are what is on screen. That reorders the
   recommendation: a text-side technique from the recipe (regional prompts) comes first, beat-matched
   photographs at low weight second, and the look technique the proposal first led with (the LCM
   ladder) is deferred.
3. The review arc corrected a dozen mechanics (checkpoint seam, multi-pass lifecycle, canvas grid,
   two-pack dependency schema, provisioning order, arm resolution order). They are folded in below.

## 1. In plain words: what Jeffrey would see

Today a beat's picture is drawn from one short text prompt (320 characters at most) in which the style
takes the front and the back and the beat's own object gets one clause in the middle. The picture
often shows the console or a generic scene, not the lantern, the ledger or the harbour the dialogue
names. The lab lanes proposed here keep everything else the same (story, voices, music, seeds, the
shipping lanes' defaults) and change only how the beat's named objects reach the sampler:

- Arm R, regional prompts: each object the dialogue names gets its own part of the frame and its own
  text, so it appears instead of competing inside one prompt. Core ComfyUI nodes only; nothing to
  download.
- Arm P, beat-matched photographs: a real public-domain or CC0 photograph of the object or place the
  beat is about nudges the picture at low strength. Style is not carried by these photos.
- Arm L, the LCM ladder (deferred): the recipe's sharper, three-pass look.

Each is a separate opt-in engine id. None is a default of any role. Success is judged on one question
first: per beat, did the dialogue's objects and situation show up on screen.

## 2. What ships today (read from the files)

### 2.1 Where the AnimateDiff graph lives

The graph is not in any workflow JSON. The engine assembles it in Python and runs it through OTR's
in-process executor (`nodes/_otr_video_engines/eng_ghost_signal.py:1331-1453`). The shipping variants
`otr_16gb_animatediff`, `otr_8gb_animatediff` (lane `animatediff15_v3_haunted_video`) and
`otr_mac16_animatediff` (lane `animatediff15_lightning_video`) only carry the director's dropdown
value, canvas 512x288, and `rtx_vsr` delivery (`config/workflow_matrix.json:138, 492` and the
`otr_mac16_animatediff` row; workflows `otr_16gb_animatediff.json` nodes 84 and 87). They are
generated from `otr_canonical.json`, which itself uses the viz lanes.

### 2.2 The haunted lane's recipe

| Cell | Value | Source |
| --- | --- | --- |
| Checkpoint | plain SD 1.5, `v1-5-pruned-emaonly-fp16.safetensors` | `eng_ghost_signal.py:69` |
| Motion module and adapter | `v3_sd15_mm.ckpt`, v3 adapter LoRA at 0.5 | `eng_ghost_signal_official.py:51, 112` |
| Sampler | one KSampler: 20 steps, cfg 8, euler, normal, denoise 1.0; negative live | `eng_ghost_signal.py:106-111, 1385-1398` |
| Context | non-looped Standard Static, length 16, overlap 4 | `eng_ghost_signal.py:115-120` |
| Canvas, cadence | 512x288, hold 2 (12.5 fresh frames per second), 25 fps delivered | `eng_ghost_signal.py:93-94, 253` |
| Delivery | Lanczos or RTX VSR to 1920x1080 | `otr_silent_composite.py:229, 254` |
| Stills | none (`accepts_still = False`) | `eng_ghost_signal.py:674-676` |

The lightning lane is the same graph on a distilled module: 8 steps, cfg 1.0 (negative inert), hold 3
(`eng_ghost_signal_lightning.py:123-158`). Absent from both, by the engine's own docstring
(`eng_ghost_signal.py:3-18`): IPAdapter, ControlNet, a second sampler, any upscale ladder.

### 2.3 How the beat's subject reaches the prompt

`resolve_crux_kernel` (`nodes/_otr_video_engines/ghost_signal_author.py:2006`) picks the beat's subject
by tiers: a `key_objects` entry the beat's own dialogue cleanly names (measured on 26.3% of beats),
then the object the batched author ranked for the beat, then the object by position, then the setting,
the brief, and finally the bookend radio object. The composer then writes
`<pack cue>. <kernel>, <light>, <motion>, <vantage>, <trailing style>` under a 320-character ceiling
(`ghost_signal_prompt.py:87, 939-1000`; example at `ghost_signal_author.py:2438`). For video_art the
front cue is `video-art feedback style` and the trailing cue `luminous recursive image field, phosphor
bloom, prismatic halos`; for recur_frac `recursive fractal light field` and `pristine raster geometry,
nested self-similar depth, emissive shader surfaces` (real output of `compact_style_cue` and
`trailing_style_cue`, `nodes/_otr_visual_styles.py:662-741`). So the dialogue-derived noun exists in
the prompt; one clause among several is all it gets.

## 3. The recipe and its sample, as found

The converted graphs are `FaderFace\workflows\animatediff_lcm_part1.json` and `_part2.json`; they are
local, paid originals and nothing from them is committed. At class level:

- Part 1: an LCM community checkpoint (`photon-LCM`), the AnimateLCM motion module, cfg 1, three
  IPAdapter Plus applications (one picture each, limited to a colour band of a mask), three sampler
  passes at denoise 1.0 / 0.6 / 0.4 with a 4x model upscale between them.
- Part 2: three ControlNets, background removal and REGIONAL TEXT CONDITIONING: an RGB mask is split
  into three region masks, each paired with its own text phrase, combined into the positive.

The sample `output\AnimateDiff_00016.mp4` (1008x1464, 16 frames, 15 fps) was made from four files in
`ComfyUI\input`: `faderface_top.png` (a bright minimalist room with a green curtain and a palm),
`faderface_middle.jpg` (a curved white architectural corridor carrying a third-party watermark),
`faderface_bottom.png` (a gallery view of a large squeegee-style abstract painting) and
`faderface_rgb_bands.png` (red / green / blue thirds as masks). The output shows the corridor's
structure, the room's curtain and palm, and the painting's palette on the panels. Its prompt was "flat
2d illustration of a quiet street at dusk", which the output does not depict: at IPAdapter weight 1.0,
"linear", cfg 1, the pictures decided the content and the text did not. Measured on this 5080 from
today's ComfyUI log: 281 s for the 16-frame clip (about 17 s per source frame), 77% of it in the second
upscale rung and the third pass. Those three pictures cannot ship (one carries another firm's
watermark, the painting's provenance is unknown).

Consequences for OTR: references carry CONTENT. That is a leak when the goal is style and the right
tool when the goal is "show the object the dialogue names", provided the weight is low and the picture
matches the beat. The recipe's 1.0 weight is wrong for that goal.

## 4. What explains the look, and what carries over

| Recipe part | Effect | Carries into OTR? |
| --- | --- | --- |
| Reference pictures (IPAdapter) | Decides content and palette; demonstrated by the sample | Yes, but beat-matched and low weight (arm P); recipe pictures cannot ship |
| Regional text (part 2) | Puts each named object in its own region | Yes; core nodes only; masks generated in code (arm R) |
| Three-pass ladder | Sharper, more detail than 512x288 plus enlargement | Yes, expensive (arm L); deferred |
| Photon-LCM, AnimateLCM, cfg 1 | Motion character; kills the negative | Optional, lab-only until Photon has a pinned source |
| Banded IPAdapter masks per picture | Surreal layering the operator wants LATER | Not built now |
| Background removal from a source video | Needs footage the lane does not have | No |
| ControlNets from a driving video | Needs footage and masks | No |

Honest bound: the sample proves references decide content. It does not prove references explain the
operator's preference for the look, and nothing here was separated by experiment before the arc.

## 5. Recommendation

Build, opt-in and default off, in this order: arm R first, arm P second, arm L later. Sibling engine ids
of the two existing AnimateDiff lanes so a frozen haunted bundle can be replayed on them, and so all
three AnimateDiff workflows can select them from the existing video dropdown:

- `animatediff15_v3_haunted_regions_video` (sibling of the haunted lane, used by the 16 GB and 8 GB
  variants) and `animatediff15_lightning_regions_video` (sibling of the lightning lane, used by the Mac
  variant). Registered for cuda only; the Mac path is unproven.
- No change to what any shipping workflow renders by default; no new director widget; no change to
  `otr_canonical.json` for a lab lane. Registration puts the ids in every user's dropdown.

Why R first: it answers the stated problem (objects named in the dialogue appear) with core ComfyUI
nodes, no new node pack, no model download, no new install story, works at cfg 8 and cfg 1 alike, and
leaves the unchanged global prompt in place so a beat with nothing to add renders exactly as today.

## 6. Arm R: regional prompts for the beat's named objects

DATA, as built. The episode's own key objects that belong to THIS beat, at most three: pool members the
beat's words refer to as whole words and do not negate (a bare "no", "not", "without" or "don't" within
four words before the mention, inside its sentence, counts as a negation, which the kernel's own cues
miss), plus the object the batched author ranked for the beat from its intent. The kernel's positional
(odometer) pick never gets a region, and neither does a bookend. Pool entries ending in the same word
collapse to one. Source: `ghost_subject_candidates(meta)` and the beat text `render_driver` joins on
`line_id` (`nodes/_otr_video_engines/ghost_scene_regions.py`). Style words never go into a region. The
place is not a region: the kernel already carries it in the global prompt.

FIELD. conditioning must be a declared request field, never observability (`schemas.py:158-170`).
One new optional field on the request model (`scene_objects: list[str]`, no place field), default
empty, so frozen bundles and every other lane are unchanged. The driver populates it only for an
engine that declares a capability (`wants_scene_regions`), the way it populates `plate_prompt` for
`wants_plate_prompt` (`render_driver.py:3977-3988`); it works in a derived replay bundle because the
driver runs at replay time. Request-model change is shared code and is covered by the other lanes'
golden pins.

ENGINE. In the encode stage, when the field is non-empty and the lane enables regions: keep the
unchanged global positive (so the style cue and the kernel stay); add one `CLIPTextEncode` per object;
build one mask per object in code (one object: a centred soft ellipse; two: left and right halves;
three: thirds; feathered edges; static for now, drifting masks are the recipe's refinement and deferred);
apply `ConditioningSetMask` (strength dial, default 1.0) and chain `ConditioningCombine`; the negative
is untouched. Core node classes only. No object, no regions: the graph is byte-identical to the base
lane, which is what keeps the lab ids honest controls.

RECEIPT. `sampler_inputs_for` gains `scene_regions` (objects, layout, strength, mask hash) only when
active; existing lanes' receipts do not change.

KNOWN LIMITS. Whether ADE's sliding context windows carry static area masks correctly over a long beat
is unmeasured; overlapping regions add weight; hold-2 cadence and the 16-frame floor are unchanged. At
cfg 1 (lightning) the negative is inert, so a region phrase cannot be steered away from lettering by the
negative; count lettering.

### 6A. Built, and what the first paired replays showed (2026-10-27 night)

Built and pushed: `animatediff15_v3_haunted_regions_video` and `animatediff15_lightning_regions_video`
(siblings, cuda only, registered in every roster `apple/VIDEO_LANE_PREFLIGHT.md` lists), the
`scene_objects` request field, the driver hook, `ghost_scene_regions.py`, and 53 new tests. The strength
dial is `OTR_SCENE_REGION_STRENGTH` (default 2.0, range 0.1-10, read by the server at run time). Sonnet QA
of the pushed diff found a real defect in the first push (the encode externals dict held the CLIP handle
on every Ghost lane) and gaps in object selection; all fixed in cb233177. No shipping workflow, variant or
default changed.

Paired replays on the canonical path (a fresh default-lane episode, frozen, replayed on the regions
sibling: same story, voices, music, seeds and prompts), all published to `otr/obs`. Judged per beat on
one question: is each object the dialogue names recognisable on screen, default versus regions.

- video_art, haunted lane, three arms of one bundle (default, strength 2, strength 4), key objects hearth,
  ledger, floorboard. Default arm: none of the named objects on any beat. Regions at strength 2: floorboards
  and, in the last frame, a fire on stone, on the one beat that names floorboard and hearth. Regions at
  strength 4: that beat is a clear win (floorboards throughout, a wood-burning stove and flames); the beat
  that names "ledger" shows a crowd of figures at both strengths and no ledger; the beat that names
  "floorboards" stays the style's abstract burst at both strengths. About one beat in four at both strengths,
  stronger at 4. The style's global prompt still dominates when a region cannot win.
- video_art, lightning lane (cfg 1), strength 2: no win. Key objects were artifact, table, microphone, while
  the dialogue talks about "that thing", "the stitching", "this covering": none of those words is in the
  pool, so by the repo's ruling they can never become a region. Only two beats got a region at all ("artifact"),
  and both arms look alike.

Cost: the regions replay took 17 minutes against about 18 for the default arm's video phase on the 8 GB
workflow's haunted lane on the 5080 (the lightning replay took 5.6 minutes), so the added time is not
measurable from wall clock; VRAM was not measured and no weight is added.

What limits it, in order: (1) the pool ruling: only the brief's key objects can ever be regions, and the
dialogue's own physical nouns are often not in the pool; (2) the region text is the bare pool word, and
"ledger" alone reads to the model as a crowd; (3) on abstract packs the style words dominate unless the
dial is high; (4) at cfg 1 the effect is weak. Levers not yet pulled, each a decision: a richer region
phrase ("a ledger book on a desk"), a pool source that includes the dialogue's own nouns (the ruling forbids
it today), and a higher default dial.

## 7. Arm P: beat-matched real photographs at low weight

A small bank of real photographs, public domain or CC0 only (NASA, Library of Congress, Smithsonian
Open Access CC0, Wikimedia Commons marked PD or CC0), resized small, tagged by the concrete objects and
places radio drama names (radio set, telephone, lamp, door, window, street at night, harbour, train,
desk, letter, sea, hills ...). A manifest beside the images records for each: source URL, author,
licence, tags, content hash. Never the recipe author's pictures. Selection: tag match against the
beat's dialogue-named objects (same field as arm R); one picture per beat, unmasked; no match, no
reference (baseline-equal). IPAdapter Plus weight low (sweep 0.3-0.5), `end_at` about 0.6 so the
photograph steers the first part of the denoise only. Style is not carried by these pictures.

WHERE THE PICTURES COME FROM, updated 2026-10-28 by the operator (relayed): the pictures may be minted per
beat from the dialogue-named objects instead of, or beside, public-domain photographs, with `flux3_klein`
on the NVIDIA tiers and `sd15` on the Mac workflow. What the profiles map today, checked in
`config/workflow_matrix.json`: `otr_8gb_animatediff` maps `flux3_klein`, `otr_16gb_animatediff` maps
`moss_image`, `otr_mac16_animatediff` maps `sd15`. Klein is to serve the 16 GB reference arm too, scoped to
the arm only: the 16 GB profile's own still-phase mapping must not change, so the arm would name its picture
engine itself instead of reading the role mapping. Sizes: `flux3_klein` 10.3 GiB, `moss_image` 12.1 GiB,
`sd15` 2.0 GiB (`apple/IMAGE_MODELS.md`); per-beat time and added VRAM on the 5080 are not measured, and
the 8 GB and Mac costs are unmeasured. This arm is not built: the Ghost lanes mint no stills
(`accepts_still = False`), so a minted picture needs either the image phase or an in-lane image engine call.

COST OF THE ARM. A new node pack (ComfyUI_IPAdapter_plus) and three model files, so the install story
changes: the one-pack-per-lane frontend schema must be migrated first (`js/lane_node_packs_core.js:58-64`,
`js/lane_node_packs.json`, `tests/test_lane_node_packs.py:40`, in a backward-compatible form with the
legacy single-pack keys as a fallback); the ADE-only hint must name both packs
(`eng_ghost_signal.py:286-288`, `_otr_visual_assets.py:1605-1615`); `scripts/otr_provision.py` needs an
explicit route (the lightning precedent, before the haunted set at 1396-1414), an explicit no-still
entry (1234-1243) and a pinned `ensure_*` like ADE's (line 52); ordinary replay does not provision
(`_otr_visual_assets.py:615-637`), so everything is in place before the replay server boots; the CLIP
vision file's generic basename needs a remote-to-local mapping (`_otr_visual_assets.py:341-344,
760-779`). IPAdapter ownership: the patch keeps the adapter module and the embeddings and ignores casts
to CPU (`IPAdapterPlus.py:497-507`, `CrossAttentionPatch.py:36-44`); only the CLIP vision model can be
released after apply; every retained allocation is released on success, cancellation and failure. The
engine tracks only base, adapter and ADE patchers today (`eng_ghost_signal.py:1420-1447, 1597-1666`).
`assets/` is excluded from the registry bundle (`.comfyignore:266`), so a shipped bank needs another
shipped path, or stays local-only for the lab.

## 8. Arm L (deferred): the recipe's ladder

Frozen first arm for when it is wanted: Photon-LCM (lab-only local file) or plain SD 1.5, AnimateLCM
module at cfg 1; pass 1 512x288, 10 steps, lcm, sgm_uniform, denoise 1.0; decode, 4x-UltraSharp, exact
Lanczos to 768x432, encode; pass 2 denoise 0.6; same to 1024x576; pass 3 denoise 0.4; decode. The
declared canvas must be /32 and 16:9 (`render_driver.py:434-436, 507-513`), so the lane declares
1024x576 and every arm ends there (an arm that stops sooner gets one recorded terminal Lanczos resize).
Slices of 8 frames through decode, upscale, resize and encode, dropping image tensors per slice (the
upscale node keeps its whole input batch on the device and allocates its whole output on the
intermediate device, `comfy_extras/nodes_upscale_model.py:79-95`). Needs: a checkpoint seam on the
parent (it hard-loads one constant in six places, `eng_ghost_signal.py:746, 856, 896-899, 930-938,
975-979, 1130-1133`, proven byte-identical on the existing lanes first), a multi-pass lifecycle (the
inherited render clears base and conditionings after one sample, 1456-1463, 1597-1666), its own canvas
override including `canonicalize` (1668-1692), and the arm resolved by one lazy pure resolver because
provisioning calls `_weight_tokens()` before `assert_usable` (`_otr_visual_assets.py:929-941`).
Delivery note: 1024x576 to 1080p is a single VSR pass (<= 2x, `eng_rtx_vsr.py:244, 448-454`) while the
512x288 baseline takes x2 plus resize, so a delivery control arm is needed to separate recipe from
delivery.

Cost, measured then extrapolated. Measured on the 5080: 281 s for one 16-frame recipe clip; per step
0.82 / 1.8 / 10.5 s on passes of 0.28 / 0.65 / 1.5 MP. EXTRAPOLATION, unmeasured, by the driver, for the
1024x576 geometry with sampling x1.4 for 11 context windows at 125 source frames: about 0.5 / 2.3 / 7
seconds per source frame with no ladder / two passes / three, so one 10 s beat (250 delivered = 125
source frames) about 1 / 5 / 15 minutes, and a 23-beat episode of 1,368 source frames about 0.2 / 0.9 /
2.7 hours. Peak VRAM and the 8 GB card are unmeasured; pass 3 stays off on 8 GB until measured. No
decision should be taken from these numbers.

## 9. Build order (every step green and pushable)

1. Request field pair, helper, capability; populated for capable engines; no engine uses it; the
   other lanes' golden pins unchanged.
2. The regions seam on the Ghost base, off by default; mask layouts; graph; receipts; tests, including
   "no object means byte-identical graph".
3. The two sibling lab ids plus the registration surface in `apple/VIDEO_LANE_PREFLIGHT.md:252-300`
   (import, shortcodes, bookend roster, evidence sentence, parity fixtures, generated matrices).
4. Replay proof (section 10).
5. Arm P: pack schema migration, then prefix row, then provisioning and rows, then the arm.
6. Arm L, only if wanted: checkpoint seam proven identical, then the lane.
7. Shipping any lab id into a matrix row also needs `render.canvas_w/h` for that row
   (`config/workflow_matrix.json:152-153`, G2.3 in `tests/test_lane_preflight_matrix.py:630-644`), then
   `scripts/build_variants.py --all` and `--check`; only if Jeffrey picks it. A registry version bump
   for any new download is his call (CLAUDE.md 7A).

## 10. Experiment and judging

Replay, on the canonical path: freeze a haunted-lane video_art bundle BEFORE registration
(`scripts/otr_freeze_replay_bundle.py`); derive once per lab id (`nodes/_otr_replay_freezer.py:380-425`
refuses a second); a Ghost sibling is the only engine ShotLock accepts (`nodes/otr_shot_lock.py:
2355-2363`); replay with `scripts/otr_canonical_api_run.py --replay-from`, one server boot per arm, arm
settings in the server-launch environment, every leg published to `otr/obs/`. The frozen bundle pins
writer, cast, voices, music, seeds and prompts, so the arms differ only in recipe. Published names stay
neutral; arms are attributed by receipt, leg log and a timestamp index with the delivered file's
SHA-256.

THE FIRST QUESTION, per beat: did the objects and the situation the dialogue names show up on screen.
The report lists each beat and says which did and which did not. Then: lettering, flicker, subject
legible, seams across context windows, render time, and the operator's eye.

## 11. What only Jeffrey can decide or supply

- D2, first: whether to run a few one-off renders of the recipe in plain ComfyUI (references removed or
  swapped, an OTR-style prompt) before more OTR code. CLAUDE.md 0A strikes a bench harness, so it is
  his call. It is the cheapest way to learn what carries the look.
- Photographs for arm P: approve the bank (public domain or CC0, sources recorded) or supply his own.
- D1: a time budget per episode and an exception to the 5-minute obs rule for long legs, only if arm L
  is wanted.
- Photon-LCM's origin, or plain SD 1.5 for anything that ships (arm L only).
- Any registry version bump for new downloads.
- Added by the build (2026-10-28): the region strength default (2.0 in code; the overnight pairs run at 4.0);
  whether to richen the region phrase; whether the dialogue's own nouns may ever become regions (today's
  ruling says no, and it limits how often the arm can help); and whether the regions lanes, or any picture
  arm, should ever replace a shipping default, which is his eye's call on the published pairs in `otr/obs`.
- Whether to build the picture-reference arm at all, and if so with minted pictures (Klein on the NVIDIA
  tiers, `sd15` on the Mac) or public-domain photographs.

## 12. What the review arc changed, and what it discarded

Changed the recommendation: the reference pictures exist and carry content (so low weight and
beat-matching, not a style mood board); the two-pack schema must be migrated first; the declared
canvas must be /32 (so 1024x576, not the first draft's sizes); ordinary replay does not provision; the
checkpoint is a hard-coded constant in six places; the arm must resolve before provisioning; and,
weighed after the arc, regional text conditioning is a better first arm than any look technique.

Discarded reviewer claims, with the reason: a "20+ hour" render and a watchdog stall (the window factor
is about 1.4x and the watchdog keys on the harness heartbeat, `scripts/otr_render_watchdog.ps1:5-6`);
"multi-pass ladders rely on ControlNet" (the sample's own graph has none); an invented
`acquire_gpu_lease()` (the lease is taken inside `MotionEngineBase.prepare`); ">30 GB VRAM, immediate OOM"
on the upscale node (the output tensor is on the intermediate device); reviewer byte counts that do not
match the files; "multiples of 64 pixels needed" (the sample is 1008x1464); "Design C2 raises
ValueError" (the sibling check does not compare `accepts_still`); a proposal to tag published
filenames (blind judging is the point, and the name builder is shared code); and a proposal to avoid the
request-model change (conditioning must be a declared field).

## 13. Evidence index

`nodes/_otr_video_engines/`: `eng_ghost_signal.py`, `eng_ghost_signal_official.py`,
`eng_ghost_signal_lightning.py`, `eng_ghost_signal_stillin_lab.py`, `ghost_signal_prompt.py`,
`ghost_signal_author.py` (2006 and 2438), `ghost_plate_prompt.py`, `render_driver.py` (434-436, 507-513,
2887-2890, 2925-2932, 3977-3988, 4234-4238, 4294-4311), `schemas.py` (158-170), `wrapper_bridge.py`
(95-100, 477-560); `nodes/_otr_visual_assets.py` (341-344, 615-637, 760-779, 929-941, 1605-1615);
`nodes/_otr_visual_styles.py` (662-741); `nodes/otr_shot_lock.py` (2318-2363);
`nodes/_otr_replay_freezer.py` (380-425); `nodes/_otr_upscale_engines/eng_rtx_vsr.py` (244, 428-454);
`js/lane_node_packs.json`, `js/lane_node_packs_core.js` (58-64); `tests/test_lane_node_packs.py:40`;
`tests/test_lane_preflight_matrix.py` (626-646); `scripts/otr_provision.py` (52, 250-292, 1234-1243,
1396-1414); `scripts/otr_canonical_api_run.py` (280-300, 740-760); `config/workflow_matrix.json`;
`.comfyignore:266`; `apple/VIDEO_LANE_PREFLIGHT.md:252-300`; `apple/SPEC_haunted_image_to_video.md`;
`apple/RUNPOD_INSTALL.md:764`. Outside the repo: ComfyUI `comfy_extras/nodes_upscale_model.py:79-95`,
`comfy/utils.py:1236-1249`; `custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py` (307-338, 497-507,
783-793, 854-855), `CrossAttentionPatch.py` (36-44, 126-158); `custom_nodes/ComfyUI-AnimateDiff-Evolved`
(`nodes_gen1.py:17-75`, `nodes_deprecated.py:74-92`, `utils_model.py:150-157`); the ComfyUI log
`user\comfyui_8000.log` (145-193); `output\AnimateDiff_00016.mp4`; the `faderface_*` inputs; and the
converted recipe graphs under `FaderFace\workflows`.
