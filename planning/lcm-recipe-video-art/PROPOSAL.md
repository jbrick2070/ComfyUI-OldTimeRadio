# Proposal: a reference-conditioned LCM lane for video_art and recur_frac

Status: PROPOSAL ONLY. Nothing here is built, no workflow JSON is edited, no
model is registered. Written 2026-10-27 against main at a0b70cab.

Next step this document asks for: the full kibitz arc on the design choice in
section 6 (CLAUDE.md, 2026-08-17 amendment: more than one defensible answer
means the four-round arc BEFORE code), then Jeffrey's eye on a paired replay
(section 7). Whether any of it ships is his call.

A note on provenance. The recipe this studies is a paid Patreon workflow. This
file names node classes, public model files and the numeric settings needed to
reason about cost; it does not reproduce the recipe's graphs, wiring, prompts or
reference pictures. If a lane is ever built it is original code in OTR's own
engine layer, with reference pictures that Jeffrey authors or OTR generates.

## 1. In plain words: what this would do to what Jeffrey sees

Today, on video_art and recur_frac, the AnimateDiff lane draws a 512x288 picture
from a short text prompt (320 characters at most) and never sees a single
reference image. The style reaches the model as three or four words at the front
of the prompt plus up to ten at the back. The recipe he prefers does two things OTR does not: it shows
the model actual pictures of the look (reference-image conditioning), and it
draws each clip three times at rising resolution, re-painting detail on each
pass (the upscale ladder). The sample he likes is a 1008x1464 frame made that
way; OTR's own AnimateDiff pixels are 512x288 before any enlargement.

If this were built and won the A/B, the same story, voices and music would come
out with richer, crisper, more clearly "video art" or "fractal light" pictures
under the narration. The price is time: one 16-frame clip of the recipe took
281 s on this 5080 (section 4.5), and the ladder, not the model swap, is most of
it. A full-depth ladder over a whole episode is estimated in hours, so the
design question is as much "how much ladder can an episode afford" as "which
reference pictures".

## 2. What ships today (read from the files)

### 2.1 Where the graph lives

The AnimateDiff graph is NOT in any workflow JSON. It is assembled in Python at
render time by the engine and run through OTR's in-process executor
(`nodes/_otr_video_engines/eng_ghost_signal.py:1331-1453`, `_wb.run_graph`).
The shipping workflows `otr_16gb_animatediff.json` and `otr_8gb_animatediff.json`
carry only the director's dropdown values: `OTR_VideoDirector` (node 87) holds
`"animatediff15_v3_haunted_video (16:9)"` three times, canvas 512x288, and
`OTR_SilentComposite` (node 84) holds `1920, 1080, 25, "", "rtx_vsr", "cuda"`.
Those two files are generated variants: `config/workflow_matrix.json` rows
`otr_8gb_animatediff` (line 138) and `otr_16gb_animatediff` (line 492) project
`role_overrides.*_visual = animatediff15_v3_haunted_video`, canvas 512x288 and
`upscale_stage.engine = rtx_vsr` onto `otr_canonical.json`, which itself uses the
viz lanes (`viz_mxc_cpu`, `viz_green`, `viz_camera`; canonical node 87).

### 2.2 The engine and its recipe

Three AnimateDiff engines are registered (`nodes/_otr_visual_assets.py:362-366`).
video_art and recur_frac run on the haunted lane when a workflow selects it.

| Cell | Value | Source |
| --- | --- | --- |
| Engine id | `animatediff15_v3_haunted_video` | `eng_ghost_signal_official.py:151` |
| Checkpoint | `v1-5-pruned-emaonly-fp16.safetensors` (plain SD 1.5) | `eng_ghost_signal.py:69` |
| Motion module | `v3_sd15_mm.ckpt` | `eng_ghost_signal_official.py:51` |
| Domain adapter | `v3_sd15_adapter.ckpt` LoRA, model path only, strength 0.5 | `eng_ghost_signal_official.py:89,112`; `eng_ghost_signal.py:1410-1418` |
| Sampler | one KSampler: 20 steps, cfg 8.0, euler, normal, denoise 1.0 | `eng_ghost_signal.py:106-110, 591-596, 1385-1398` |
| Negative prompt | LIVE (cfg 8 evaluates the unconditional pass) | `eng_ghost_signal.py:102-105, 696` |
| Context | non-looped Standard Static, length 16, overlap 4, pyramid | `eng_ghost_signal.py:115-120, 1356-1367` |
| Latent | `EmptyLatentImage` 512x288, batch = source frames | `eng_ghost_signal.py:93-94, 1380-1384` |
| Cadence | hold 2: `U = ceil(T/2)` fresh frames, each shown twice, 25 fps delivered (12.5 fresh frames per second) | `eng_ghost_signal.py:253, 387-400` |
| Delivery | Lanczos to 1920x1080, or RTX VSR x2 then exact resize (all NVIDIA workflows) | `eng_ghost_signal.py:96-100`; `otr_silent_composite.py:229, 254` |
| Stills | none: `accepts_still = False`, `subject_ownership = "prompt"` | `eng_ghost_signal.py:674-676` |

### 2.3 What is deliberately absent

The engine's own docstring lists it so nobody adds it by accident
(`eng_ghost_signal.py:3-18`): "no IPAdapter, no ControlNet, no reference folder",
no second KSampler, no hires fix, no ESRGAN or SeedVR2 upscaler, no looped
context. A grep of `nodes/` finds IPAdapter only in that docstring. The only
upscale OTR applies to these clips is the delivery step in 2.2, which enlarges
pixels and cannot add detail.

AnimateLCM was refused on 2026-08-22 because its cfg 1-2 regime "kills the live
negative" (`eng_ghost_signal_official.py:33-35`). That reason has since been
overtaken: `animatediff15_lightning_video` runs cfg 1.0 on purpose, records the
negative as inert (`negative_effective` false), and published a full episode
(`eng_ghost_signal_lightning.py:71-102, 147-158`; `registry.py:673-690`). The
lettering risk the refusal guarded against is still real (SD 1.5 volunteers text
into anything sign-shaped) but it is now a look defect to measure, not a rule.

### 2.4 The prompt for these two styles

The text path is the weak link the operator's notes already name.

- Front cue: `compact_style_cue` returns two to four words and stops at the word
  "style" (`nodes/_otr_visual_styles.py:662-683`). Calling the real function on
  the two shipped packs gives `video_art -> "video-art feedback style"` and
  `recur_frac -> "recursive fractal light field"`.
- Trailing cue: up to ten words of the rest of `positive_tail`, whole comma
  units only (`_otr_visual_styles.py:693, 696-741`). Real output:
  `video_art -> "luminous recursive image field, phosphor bloom, prismatic halos"`;
  `recur_frac -> "pristine raster geometry, nested self-similar depth, emissive shader surfaces"`.
- Hard ceiling 320 characters (`ghost_signal_prompt.py:87`) and a 69-token author
  target inside one 77-token CLIP window (`ghost_signal_author.py:138`). The
  composed shape, from the composer's own example, is
  `"recursive fractal light field. a bakelite radio set, distance opening toward
  the back, wide, the people small in the space, pristine raster geometry, nested
  self-similar depth, emissive shader surfaces"` (`ghost_signal_author.py:2438`).
- Motion register: a pack's 20-29 word register is compacted to 14 words
  (`_otr_visual_styles.py:762`); for the announcer role that is
  `"radio console phosphor bloom breathes, slow sculptural push forward"` on
  video_art. Colour language the packs author elsewhere ("red blue green
  phosphor overlap", "laser cyan, spectral magenta, acid green") lives in fields
  this lane never reads (`video_art.json`, `recur_frac.json`).
- Negative: the pack's `negative_tail` after five lane-hygiene tokens, 320
  characters (`ghost_signal_prompt.py:90, 107-109`).

The plate prompt of the still-in lab is the one place the full pack language is
used, and it is cut by the same ceiling: `recur_frac`'s `positive_tail` is 227
characters and `plate_look` 129, so head plus `plate_look` is 358 against 320 and
`plate_look` is dropped; the plate reduces to the `positive_tail` alone
(`ghost_plate_prompt.py:17-20, 75-78, 118-148`; lengths measured from
`nodes/visual_styles/recur_frac.json`). `video_art` fits head plus `plate_look`
(146 + 103 = 251 characters) plus at most its story setting.

### 2.5 Two relatives already in the tree

- `animatediff15_lightning_video`: distilled module, 8 steps, cfg 1.0,
  `sgm_uniform`, hold 3, external VAE, no adapter
  (`eng_ghost_signal_lightning.py:123-158, 214-215`). It proves a low-cfg
  distilled module runs through OTR's graph and adapter path.
- `animatediff15_v3_stillin_lab_video`: mints one in-family plate in the same
  graph from the pack's full language and feeds it to the sampler as an INIT
  LATENT at denoise below 1, default 0.65 (`eng_ghost_signal_stillin_lab.py:8-27,
  61-62`). Its own spec expects a repeated init latent to suppress motion
  (`apple/SPEC_haunted_image_to_video.md:77-95`). I found no recorded verdict for
  the lab's denoise sweep in the docs I read; the file calls itself a probe.
  This matters below: IPAdapter conditions through attention, not through the
  starting latent, so it avoids that suspected motion penalty.

## 3. The recipe, at class level

From the converted workflows on this box (`FaderFace\workflows\animatediff_lcm_part1.json`
and `_part2.json`) and today's ComfyUI log.

Part 1 (the part this proposal is about):

- Checkpoint `photon-LCM.safetensors` (by its name an SD 1.5 photographic
  community checkpoint with LCM applied; its provenance is not recorded on this
  box), motion module `AnimateLCM_sd15_t2v.ckpt`, beta schedule
  `sqrt_linear (AnimateDiff)`, motion scale 1.15 on the loader.
- Looped Uniform context, length 16, closed loop off, overlap 4, pyramid fuse.
- Base latent 440x640 (portrait), 24 frames in the saved graph, 16 in the sample.
- IPAdapter Plus (`ip-adapter-plus_sd15.bin`) with the SD 1.5 CLIP vision encoder,
  weight 1.0, three reference pictures, each limited to a colour band of a
  painted mask (RGB bands via ColorToMask).
- Three KSampler passes, seed fixed, 10 steps, cfg 1, `lcm`, `sgm_uniform`,
  denoise 1.0 then 0.6 then 0.4.
- Between passes: VAE decode, `4x-UltraSharp` model upscale, resize back down so
  each rung nets about 1.5x per side, VAE encode. Output 15 fps H.264. Sample on disk:
  `output\AnimateDiff_00016.mp4`, 1008x1464, 16 frames, 15 fps (ffprobe).

Part 2 adds three ControlNets (lineart, depth, tile), a video-driven preprocessor
chain, background removal and regional text conditioning. All of it is fed by a
driving video and masks a person supplied. It is out of scope here (section 5).

## 4. What explains the look, and what carries over

### 4.1 Comparison

| Recipe part | OTR today | Likely explains the look? | Carries into OTR? |
| --- | --- | --- | --- |
| Reference pictures through IPAdapter (3 refs, banded masks) | none; style is 3-4 front words + 10 trailing words | Strongest candidate. The sample's palette and surface language are carried by pictures, which a 320-character SD 1.5 prompt cannot summon. | The mechanism yes. The pictures no: they are a person's. A source must be chosen (section 6). |
| Three-pass ladder, denoise 1.0 / 0.6 / 0.4 with a 4x model between passes | one pass at 512x288; delivery upscale only enlarges | Strong. The recipe generates about 1.5 MP of real content per frame; OTR generates 0.15 MP and then enlarges. Re-denoising at 0.6 and 0.4 is what paints new detail. | Yes, pure compute, no outside source. Dominant cost (4.5). |
| Photon-LCM checkpoint instead of plain SD 1.5 | `v1-5-pruned-emaonly-fp16` | Plausible and untested in isolation. A quality merge of SD 1.5 can matter as much as the module. | Yes technically (file is on this box); its public source and licence are unestablished (4.3). |
| AnimateLCM module, cfg 1, 10 steps | v3 module, cfg 8, 20 steps | Affects motion character and speed more than palette. Cfg 1 also makes the negative inert. | Yes. Lightning lane is the precedent for cfg 1 (2.3). |
| Looped Uniform context | Standard Static context | Cannot explain the sample: 16 frames is one window, ComfyUI log: "latents passed in (16) less or equal to context_length 16". It matters only for OTR's 100-plus-frame beats, where it is untested. | Yes; ADE ships the node (installed pack, `nodes_context.py`). |
| Base resolution 440x640 portrait | 512x288 landscape | Pixel budget yes (about 2x base area); portrait composition is a different picture and not transferable to 16:9. | Budget yes: 704x384 matches the recipe's base area in 16:9. |
| 15 fps, every frame fresh | 12.5 fresh frames per second, each shown twice (hold 2) | Possible smoothness contributor; cheap to test with the existing hold knob. | Yes (`OTR_GHOST_HOLD_FACTOR`, `eng_ghost_signal.py:125-174`). |
| Part 2: ControlNets, driving video, masks, regional prompts | none | Not part of the sample Jeffrey liked in its simplest form. | No: needs footage and masks OTR does not have. A self-conditioned tile ControlNet on a pass's own output is a possible later extension and is not studied here. |

Honest bound: I could not separate these causes by experiment in a read-only
study. The ranking above is reasoning from the settings and the sample frame, and
section 7 is built so the replay isolates each cause instead of testing the bundle.

### 4.2 The lettering and negative risk

cfg 1 deletes the unconditional pass, so a text-defence negative does nothing
(`eng_ghost_signal_lightning.py:71-102`). The recipe's own pictures are abstract
colour fields, which may keep lettering down; OTR's would need to be too. Count
lettered beats in every arm. A cfg sweep knob (`OTR_LIGHTNING_CFG` precedent)
buys the negative back at double the UNet cost.

### 4.3 Models the lane would need

Sizes are bytes on disk under `C:\ComfyUI-Models` (the models root,
`nodes/_otr_models_root.py`). "Registered" means present in
`nodes/_otr_visual_assets.py` rows (line 321-326 for the current AnimateDiff set).

| Artifact | Folder | Bytes | On this box | Registered | Source / licence status |
| --- | --- | --- | --- | --- | --- |
| `photon-LCM.safetensors` | checkpoints/SD1.5/LCM | 2,133,817,176 | yes | no | Public repo NOT established; registry rows need a pinned Hub repo and SHA. Licence unknown. |
| `AnimateLCM_sd15_t2v.ckpt` | animatediff_models | 1,813,041,929 | yes | no | wangfuyun/AnimateLCM; licence not confirmed (search found no licence field). |
| `ip-adapter-plus_sd15.bin` | ipadapter | 158,033,179 | yes | no | h94/IP-Adapter family; not re-checked. |
| CLIP vision `model.safetensors` | clip_vision/SD1.5 | 2,528,373,448 | yes | no | h94/IP-Adapter image encoder; the generic filename needs a unique registry name. |
| `4x-UltraSharp.pth` | upscale_models | 66,961,958 | yes | no | Third-party listings give CC BY-NC-SA 4.0 (unconfirmed against the model card). |

That is about 6.7 GB of new downloads. The shipping SD 1.5 base stays for the
existing lanes. Commercial licence is not a gate in this project (operator
ruling, memory `commercial-licence-is-not-a-gate`); the unknowns above are
recorded so the registry listing can state them, not so they block.

Nodes beyond what the haunted lane uses (the haunted lane's seven classes are at
`eng_ghost_signal.py:290-298`):

- `ComfyUI_IPAdapter_plus` (cubiq), installed here: `IPAdapterModelLoader`,
  `IPAdapterBatch` or `IPAdapterAdvanced`. A NEW third-party pack dependency.
  It needs a `_PACK_FOR_PREFIX` row (`wrapper_bridge.py:95-100`), an entry in
  `js/lane_node_packs.json`, and the queue-time gate
  (`_otr_visual_assets.py` around line 1590-1625) then names it on a fresh install.
- ComfyUI core only: `CLIPVisionLoader`, `UpscaleModelLoader`,
  `ImageUpscaleWithModel`, `ImageScale`/`ImageScaleBy`, `VAEEncode`.
  `ADE_LoopedUniformContextOptions` is already in the pack the lane requires.
- No KJNodes, VideoHelperSuite or ColorToMask are needed: OTR writes its own MP4
  and can build the blend masks in Python.

### 4.4 What changes in the canonical workflow

Short answer: for a lab lane, nothing in `otr_canonical.json`.

- The engine id appears in the director dropdown automatically, because the menu
  is "every REGISTERED video engine" (`nodes/otr_video_director.py:176-178`).
- The canonical graph does not contain the AnimateDiff graph (2.1), so there are
  no nodes or links to add. The lane is chosen by the director's first three
  widget values.
- To SHIP the lane, the change is a `config/workflow_matrix.json` delta (the
  `role_overrides.*_visual` ids and `preflight.required_models`), then
  `python scripts/build_variants.py --all` and `--check`, then the four checks in
  CLAUDE.md section 0 (variants check, widget-value alignment, widget/input
  parity, link target indexes). No widget shape changes, so no
  `widgets_values` reindexing.
- If Jeffrey wants a visible control (for example ladder depth or reference
  strength on the director), it is appended at the END of `OTR_VideoDirector`
  `widgets_values` (today 17 values, last is `still_roll_pool`; node 87 in every
  AnimateDiff variant) and at the end of its `INPUT_TYPES` optional block, in
  canonical and all variants in the same change. Recommendation: no widget in the
  first version. Lab dials in this repo are environment knobs stamped into the
  receipt (`OTR_LIGHTNING_CFG`, `OTR_STILLIN_LAB_DENOISE`,
  `OTR_GHOST_HOLD_FACTOR`), and a recipe frozen on the class is what a shipped
  lane does.
- The other registration surface a new engine id trips is documented and long:
  `apple/VIDEO_LANE_PREFLIGHT.md:252-300` (shortcode row at
  `nodes/_otr_shared/shortcodes.py:152-154`, evidence-manifest sentence, parity
  fixtures, the literal rosters, bookend roster, fetch bundle, no-still set,
  `_ANIMATEDIFF_WEIGHT_ENGINES`, `_PACK_FOR_PREFIX`). Budget it in the
  registration commit.

### 4.5 Cost: measured, extrapolated, unmeasured

Measured, this 5080 (16 GB, `Total VRAM 16275 MB`), today's recipe run, one
16-frame clip, ComfyUI log `user\comfyui_8000.log` lines 145-193:

| Stage (from log timestamps) | Seconds | Per source frame |
| --- | --- | --- |
| Model loads before sampling | 6 | n/a (once per beat session in OTR) |
| Pass 1 sampling (10 steps, 440x640) | 8 | 0.5 s |
| Decode + 4x upscale + resize + encode (ladder 1) | 27 | 1.7 s |
| Pass 2 sampling (10 steps, denoise 0.6, about 670x970) | 18 | 1.1 s |
| Ladder 2 | 107 | 6.7 s |
| Pass 3 sampling (10 steps, denoise 0.4, about 1010x1480) | 105 | 6.6 s |
| Final decode and write | 11 | 0.7 s |
| Whole prompt | 281.37 | about 17 s |

Reading it: the second ladder rung and the last pass are about 77% of the run.
Each pass runs 10 steps (ComfyUI runs the full step count from partial noise when
denoise is below 1), so per step pass 1 is about 0.8 s, pass 2 about 1.8 s and
pass 3 about 10.5 s. Pass 3 has 2.3 times the pixels of pass 2 and took 5.8 times
as long per step, close to what attention scaling with token count predicts
(2.3 squared is 5.3). So cost climbs faster than pixel count at this size, and the
last pass is where a smaller canvas saves the most. Whether VRAM pressure on the
16 GB card also played a part was not diagnosed.

Extrapolated (assumes cost per frame is constant, which is unproven beyond 16
frames; a 125-frame beat adds context-window overlap):

| Ladder depth | Per source frame | One 10 s beat (125 source frames) | 23-beat episode (1,368 source frames at hold 2) |
| --- | --- | --- | --- |
| Pass 1 only | about 1.2 s | about 2.5 min | about 27 min |
| Passes 1-2 | about 4 s | about 8 min | about 1.5 h |
| All three passes, recipe base area | about 17 s | about 36 min | about 6.5 h |

The 23-beat, 2,736-frame episode is the Mac proof of the lightning lane
(`registry.py:677-679`); it is a size reference, not a claim about video_art.
A smaller base (512x288) would cut the pass and ladder costs by at least the pixel
ratio, 0.52, and more on the last pass given the superlinear step time above;
again unmeasured. For comparison the existing haunted lane's whole
one-act leg took 21 minutes on a rented 4090 including writer, voices and music
(`apple/RUNPOD_INSTALL.md:764`) and 35.7 minutes on the 4060 clean-room run
(`apple/HANDOFF_LOG.md:1691`); its per-frame sampling cost on the 5080 is not
recorded anywhere I found.

Host memory, computed and unmeasured: a model upscale returns an image-domain
batch. For 125 frames the second rung's 4x output at about 669x973 input is
2676x3892x3 float32 per frame, about 15.6 GB for the batch. It must be chunked
(frames are independent in decode, upscale and encode, so chunking changes no
pixel) and only the small latents kept between passes.

VRAM:

| Item | Status | Value |
| --- | --- | --- |
| UNet (Photon LCM) | measured, log | 1,639 MB staged |
| Text encoder | measured, log | 235 MB |
| VAE | measured, log | 159 MB |
| AnimateLCM module | measured, log | 864 MB loaded |
| CLIP vision | measured, log | 1,209 MB staged |
| 4x-UltraSharp | measured, log | 65 MB |
| IPAdapter Plus | not logged | 158 MB on disk |
| Resident weights, all of the above | measured sum | about 4.3 GB |
| Existing haunted lane weights | recorded | 2.4 GB (`apple/PROD_BUG_LOG.md:11994`) |
| Peak at pass 3, 16 GB card | NOT MEASURED | log has no peak line |
| Anything on the 8 GB card | NOT MEASURED | 4060 owns this (CLAUDE.md 0B) |

No existing AnimateDiff lane has a qualified cost row either; admission is
"NOT enforced" by operator decision (`config/workflow_matrix.json`, haunted
lane note), so this lane would be recorded the same way until measured.

Profile implications, as claims to test, not findings:

- 16 GB (`otr_16gb_animatediff`): weights fit with room; the open question is
  time and the pass 3 slowdown above.
- 8 GB (`otr_8gb_animatediff`, writer ceiling 6.8 GB): resident weights of about
  4.3 GB leave roughly 3.5 GB for activations. A first 8 GB arm should be one
  or two passes at 512x288 base, not the recipe's 1008x1464 final.

## 5. What cannot be carried over

- The recipe's reference pictures and its painted band mask: a person's work.
  The reference pictures are not on this box (the converted graphs name
  `pasted\...` files that do not exist under `ComfyUI\input`), so even their
  content is unknown to this study; the only evidence of the look is the sample
  frame.
- Part 2: ControlNets fed by driving video, background removal, regional prompts.
- The portrait 440x640 composition.
- The recipe's Patreon graphs themselves: not to be copied into any tracked file.

## 6. Designs

All three are a new lab engine id that subclasses the unregistered base
`GhostSignalEngine`, as the lightning lane does (`eng_ghost_signal_lightning.py:203-205`),
not the haunted peer: no v3 adapter, since the adapter is v3-paired
(`eng_ghost_signal_lightning.py:64-69`). Staying a Ghost sibling (same family,
roles, prompt profile and frame contract) is what lets a frozen haunted bundle
be replayed on it (`nodes/otr_shot_lock.py:2318-2363`). Working name
`animatediff15_lcm_ladder_lab_video`; the name is for the arc to settle.

### Design A: ladder only, no reference pictures

Photon-LCM + AnimateLCM at cfg 1, then 1 to 3 passes with the model upscale.
Prompts stay exactly as today.

- For: no outside source, no new third-party pack, no assets to ship, replay-safe
  by construction. Answers "how much of the look is resolution and checkpoint".
- Against: it cannot add information the 320-character prompt lacks. If the pictures
  are what Jeffrey loves, this gives a sharper version of today's look, not the
  recipe's look. The negative is inert.

### Design B: pictures from a shipped style-anchor bank (recipe-faithful)

Everything in A, plus IPAdapter Plus conditioned on a small bank of reference
pictures per pack (video_art and recur_frac first), blended spatially or used one
per beat.

- For: closest to the recipe, so the likeliest to reproduce the look; fully
  deterministic and replay-safe; the pictures are Jeffrey's eye made concrete.
  A new pack is a folder of pictures, not a code change.
- Against: needs pictures from Jeffrey (his own photos are cleared to ship;
  AI-generated frames he makes are fine; the recipe author's are not). Adds a
  shipped asset class and a third-party node pack. Every beat shares the same
  anchors, so the look is uniform across an episode, which may be a feature for
  these two styles and a flaw for the rest. Nine packs means nine banks to
  maintain, though only two matter now.

### Design C: pictures from the episode itself

Everything in A, with references taken from the episode's own stills (character
or scene stills the image phase mints in the pack's style) or from a plate minted
in the same graph.

- C1, plate in the graph (`compose_plate_prompt`, wired for the lab lane at
  `render_driver.py:3977-3987`): zero shipped assets, works for all nine packs
  and carries the story's setting. But the plate is drawn from the same
  320-character prompt, so on recur_frac it is the `positive_tail` alone (2.4):
  it can re-express the medium, it cannot add what the text lacks. It likely
  gives the smallest uplift for the same reason today's look is weak.
- C2, the episode's stills: richest and most episode-specific, and it would carry
  characters across beats. It reverses the lane's design
  (`accepts_still = False`), makes the image phase mandatory (flux3_klein on 8 GB,
  moss_image on 16 GB per the matrix rows), and a Ghost bundle contains no stills
  (`eng_ghost_signal.py:674-676`), so a paired replay needs a new bundle from an
  image-bearing episode and cannot reuse the haunted-lane bundles.

### Recommendation

Build one lab id with the recipe frozen on the class and the arm chosen by
environment variables stamped into the receipt (the stillin-lab precedent,
`eng_ghost_signal_stillin_lab.py:57-62`). Order the arms A, then B, then C1, and
let the replay (section 7) decide whether any reference source earns its
complexity. If A alone gets the look, ship A and skip the reference question. If
B wins, ship B for the two packs it was proven on. C2 is a separate, larger
decision and should not be bundled into this one.

Open design questions for the arc: how a ladder of this cost is bounded (depth
by profile? by beat length? only on bookend beats?), whether the final pass
should be skipped on 8 GB, and whether the lane should keep its own prompt
budget or widen it when references carry the style.

## 7. Paired-replay plan (stays on the canonical path)

CLAUDE.md section 0 and 0A: every run loads `workflows/otr_canonical.json`, no
bench harness. A replay does: `scripts/otr_canonical_api_run.py --replay-from
<bundle>` renders only the video phase and the publish tail from a frozen
bundle (`scripts/otr_canonical_api_run.py:740-743`).

Mechanics, all existing:

1. Freeze one bundle from a real haunted-lane episode per style with
   `scripts/otr_freeze_replay_bundle.py`. Writer, cast, voices, music and the
   planned prompts and seeds are pinned (memory `replay-pins-the-prompt-too`: the
   Ghost prompt is composed at plan time and a replay reuses it). That is the
   right property here: it holds the 320-character prompt constant so the arms
   differ only in recipe.
2. Derive one bundle per arm: `python scripts/otr_freeze_replay_bundle.py
   <bundle> --derive-engine <lab id>` (`nodes/_otr_replay_freezer.py:380-425`).
   ShotLock accepts it only for a Ghost sibling of the frozen plan's engine
   (`otr_shot_lock.py:2355-2363`), which is why section 6 keeps the family,
   roles, prompt profile and frame contract. A bundle frozen from a non-Ghost
   plan (for example one frozen from an LTX episode) cannot be derived onto it.
3. One arm per server boot, arm knobs from the environment, receipt stamping
   the resolved values; same shape as the existing arm runs.
4. Every leg publishes to `otr/obs/` (operator rule: a leg that does not reach
   obs did not pass, and more than 5 minutes with nothing there is a fail).

Arms, on video_art first, then recur_frac with only the arms that survived:

| Arm | What changes against the frozen haunted baseline | Isolates |
| --- | --- | --- |
| 0 | frozen haunted lane, replayed | baseline and A/A null |
| 1 | Photon-LCM + AnimateLCM, cfg 1, 512x288, no ladder | checkpoint and module |
| 2 | arm 1 plus ladder, passes 1-2 | resolution and re-denoise |
| 3 | arm 2 at the recipe's base pixel area (16:9, 704x384) | base resolution |
| 4 | best of 1-3 plus anchor-bank references (Design B) | reference pictures |
| 5 | best of 1-3 plus in-graph plate references (Design C1) | plate vs bank |
| 6 | best arm with the third pass | what the last rung buys |
| cfg | best arm with cfg 1.5-2 | whether the negative matters |

Step zero, before any arm is judged: one 125-frame beat per candidate depth with
`nvidia-smi` sampled every 5 seconds (the 4060's portable Python has no NVML, so
the engine's own peak reads unknown, per memory `recipe-ab-by-paired-replay`).
That replaces every extrapolation in 4.5 with a number and decides which arms are
affordable at all.

Judging is Jeffrey's eye, as radio drama, per style, on published obs episodes
(memories `his-ear-settles-a-lane-cut-rows-that-reask-it`,
`model-swap-on-shipping-surface-is-an-ear-decision`,
`judge-it-as-radio-drama`). Two stale pointers to fix, not to rely on:
`recipe-ab-by-paired-replay` still names `scripts/otr_replay_ab.py`, which was
deleted in 939167bf (2026-10-14, "the operator ended bake-offs on 2026-08-23"),
and its distance-from-the-still metric means nothing for a prompt-owned lane
anyway. Report lettering count, flicker and any beat that fell back.

Machines: the 5080 runs the arms first (shipping surface and workflow matrix are
its files). The 4060 then measures the 8 GB arm. Blast radius to state in the
registration commit: a new engine id leaves every existing lane's cells
byte-identical (the haunted and lightning pins), and the shared files it touches
are additive rows (`registry.py`, `shortcodes.py`, `_otr_visual_assets.py`,
`lane_node_packs.json`, `_PACK_FOR_PREFIX`). Prove the golden pins stay green
(`test_the_golden_lane_still_declares_hold_2` and neighbours) rather than assert
it.

## 8. Risks and unknowns, in one place

- The ranking of causes in 4.1 is reasoning, not measurement.
- Cost scaling past 16 frames is unmeasured; pass 3's per-step jump is
  undiagnosed.
- Photon-LCM has no recorded public source; a registry row needs one.
- Licences for AnimateLCM and 4x-UltraSharp are unconfirmed (4.3).
- A new third-party node pack (IPAdapter Plus) enters the install story; its
  upkeep status was not checked in this study.
- cfg 1 disables the lettering defence (4.2).
- Publishing a lane with about 6.7 GB of new downloads is a registry version
  bump, which is Jeffrey's call (CLAUDE.md 7A).
- The converted graphs and sample are Patreon-derived; nothing from them goes
  into a tracked file.

## 9. Evidence index

Existing code and data cited above, by path (all under the repo root unless
absolute):

- `nodes/_otr_video_engines/eng_ghost_signal.py` (recipe constants 56-130,
  class attributes 591-690, sample graph 1355-1453)
- `nodes/_otr_video_engines/eng_ghost_signal_official.py` (haunted lane 118-173)
- `nodes/_otr_video_engines/eng_ghost_signal_lightning.py` (cfg 1 precedent)
- `nodes/_otr_video_engines/eng_ghost_signal_stillin_lab.py` (plate lab lane)
- `nodes/_otr_video_engines/ghost_signal_prompt.py:87,90,107-114,1032`
- `nodes/_otr_video_engines/ghost_signal_author.py:138,2430-2448`
- `nodes/_otr_video_engines/ghost_plate_prompt.py:1-29,75-78,118-148`
- `nodes/_otr_visual_styles.py:662-741,762`
- `nodes/visual_styles/video_art.json`, `nodes/visual_styles/recur_frac.json`
- `nodes/_otr_video_engines/registry.py:600-701`
- `nodes/_otr_visual_assets.py:305-326,359-380,1590-1625`
- `nodes/_otr_video_engines/wrapper_bridge.py:95-100`
- `js/lane_node_packs.json`
- `nodes/otr_video_director.py:176-178`
- `nodes/otr_silent_composite.py:229,254`
- `nodes/otr_shot_lock.py:2318-2363`
- `nodes/_otr_replay_freezer.py:380-425`
- `scripts/otr_canonical_api_run.py:280-300,740-760`
- `config/workflow_matrix.json:138,492` and the haunted-lane admission note
- `workflows/otr_16gb_animatediff.json`, `workflows/otr_8gb_animatediff.json`
  (nodes 84 and 87), `workflows/otr_canonical.json` (node 87)
- `apple/VIDEO_LANE_PREFLIGHT.md:252-300`
- `apple/SPEC_haunted_image_to_video.md:77-95`
- `apple/RUNPOD_INSTALL.md:764`, `apple/HANDOFF_LOG.md:1691`,
  `apple/PROD_BUG_LOG.md:11994`, `apple/MAC_PORTABILITY_GUIDE.md:580`
- Outside the repo: `C:\Users\jeffr\Documents\ComfyUI\user\comfyui_8000.log`
  (lines 145-193), `C:\Users\jeffr\Documents\ComfyUI\output\AnimateDiff_00016.mp4`,
  `C:\Users\jeffr\Documents\ComfyUI\FaderFace\workflows\animatediff_lcm_part1.json`
  and `_part2.json`, model files under `C:\ComfyUI-Models`.
