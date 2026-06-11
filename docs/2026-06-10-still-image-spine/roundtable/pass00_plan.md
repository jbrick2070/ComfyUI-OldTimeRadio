# The 2D-still spine -- problem statement (operator-ordered roundtable, 2026-06-10 night)

## The operator's complaint (verbatim intent)

The June-5 era episodes opened on GORGEOUS macro radio stills (brass speaker
grilles, rings of glowing amber tubes -- FLUX still quality) and the operator
wants that back. Somewhere in the video-platform refactor "the 2D still got
lost in the shuffle": today the LTX opens are TEXT-ONLY (no image conditioning
at all), the only stills minted are character portraits, they are stored in a
GLOBAL dir (`output/otr/stills/`) instead of the episode folder, and their
prompts no longer follow the June-5 composition. Operator directives:

1. **Stills must feed the video models** -- a 2D still not populating LTX (and
   peers) for input "is THE problem."
2. **Still prompts must mimic the June-5 style** (the legacy 5-layer
   composition that produced the macro radio look).
3. **Stills are per-episode artifacts**: stored under
   `output/otr/episodes/<episode>/stills/` (portraits, opens, everything),
   not only a global pool.
4. Roundtable the design: "2D image gen and how it feeds into ALL the 3D and
   2D video models" -- Fable sits as an independent panelist AND the judge.

## Current state (probed tonight, all git/log-anchored)

- **Image platform exists** (C1): `otr_image_director.py` (policy),
  `otr_meta_brief_image_prompt.py` (per-CHARACTER prompts, temp=0, person
  guard, gear scrub as of d087bfa), `otr_image_gen_dispatcher.py` ->
  `nodes/_otr_image_engines/` registry (flux_gen1 default: 20 steps, cfg 1.0,
  832x1216; peers: flux2_klein, qwen_image, sd35_large, chroma_hd, hidream_i1,
  lumina_image, z_image_turbo). Portraits write back into the live ledger
  `images.images[]` keyed by object_id and SAVE to `output/otr/stills/<hash>.png`.
- **No scene/open stills are minted at all** -- `meta.visual_plan.scenes == []`
  post-CW-1 (the writer's per-scene derivation died with the legacy plan; the
  round-5 fix composes LTX TEXT prompts per-beat instead).
- **ltx_video is text_to_video**: graph = checkpoint -> CLIPLoader(T5) ->
  CLIPTextEncode x2 -> EmptyLTXVLatentVideo -> LTXVConditioning -> KSampler ->
  VAEDecode (`eng_ltx_video._build_graph`). NO image input. required_inputs =
  ("text_prompt",). The installed wrapper MAY expose an img2vid conditioning
  node (LTXVImgToVideo or equivalent) -- VERIFY-AT-BUILD.
- **wan_i2v is image_to_video** (roles incl. music_visual + scene_broll +
  character_video) -- it NEEDS an init image by family; ckpt on disk
  (capstone night), env-gated OTR_ENABLE_WAN_I2V.
- **humo** consumes portrait init_images already (the only still->video seam
  alive today). **still_kenburns** floor drifts over a still/procgen frame.
- **3D (PARKED)**: the 3D plan's image-routing must-fixes (VIDEO_OPTIN_
  GOFORWARD_PLAN Phase 5) already specify character-level image routing into
  mesh-gen -- the SAME stills should feed it when 3D reopens; the
  ImageDirector.video_policy_json fail-closed requirement is recorded there.
- **June-5 reference look** (signal_lost_toolwielding_tentacles_20260605):
  full-frame FLUX macro stills of brass radio hardware, Ken-Burns motion,
  per-episode palette (ocean episode -> brass/teal). The composer is preserved
  at `docs/2026-06-10-brief-downstream-gaps/legacy_otr_video_plan_e74a3ce.py.txt`:
  `portrait_prompt + scene.visual_prompt + shot_hint + era_tail + style_tail`.
- **Era tail today is heavy** (~280 chars: full atmosphere sentence + palette
  + 5 lighting + 5 atmosphere terms) -- uncapped prompts (portraits/M4) carry
  it whole; tonight's Mars episode painted everything red (correct-to-story
  but operator-noticed).
- Round-5 state (shipped tonight, 1351d78): LTX text prompts are per-beat
  brief+beat composed, frame band 169, diversity gate, person anchors,
  attribution repair, portrait gear scrub. Suite 3863/0.

## The design question for the panel

Design the **still-image spine**: per-episode 2D stills as first-class
artifacts that (a) look like June-5 (macro radio opens + in-character
three-quarter portraits, per-episode palette), (b) are stored in the EPISODE
folder, and (c) CONDITION every downstream video/3D engine that can take an
image. Specifically:

- **S1 -- Scene/open stills**: mint a still per text-engine beat (the open,
  announcer, outro at minimum) via the image platform, prompts composed
  June-5-style: concrete SUBJECT (macro radio hardware for open roles) +
  brief setting/atmosphere TERMS + era tail (probably TRIMMED) + style tail.
  Where do the prompts come from -- the round-5 driver composition moved INTO
  the image-prompt node, or a shared helper both call?
- **S2 -- Episode-folder storage**: every still (portrait + scene) saved under
  `episodes/<ep>/stills/` (operator inspectability) -- global pool kept or
  dropped? Ledger images[] entries must carry the episode-local path.
- **S3 -- Conditioning matrix**: still -> engine:
  * wan_i2v: init_image = the beat's scene still (native i2v).
  * ltx_video: img2vid conditioning IF the installed wrapper exposes it
    (VERIFY-AT-BUILD; if absent, LTX stays text-only and wan_i2v/kenburns
    carry the image-conditioned look).
  * still_kenburns: drift over the scene still (the June-5 fallback look,
    zero new GPU cost).
  * humo: portraits (unchanged).
  * 3D mesh-gen (FUTURE, parked): consumes character stills via the
    ImageDirector routing -- design the still spine so the 3D plan's
    must-fixes slot in without rework.
- **S4 -- The M4->HuMo creative seam** (found tonight): cast-beat requests
  carry no M4 prompt in the live graph. In-scope to wire here, or separate?
- **S5 -- Era-tail diet**: trim to atmosphere line + top-2 palette + top-2
  lighting? Per-surface tails (stills full, video capped)?
- **Routing/owner**: does OTR_ImageDirector own which beats get stills
  (policy), with the dispatcher minting and the render driver consuming by
  beat_id? Image_done gating (C1) interaction? VRAM: stills mint BEFORE heavy
  video engines load (the gpu_residency lease)?

## Constraints (binding)

Frozen audio spine / byte-identical / mux-LAST; fail-soft never fail-episode
(a missing still degrades to today's text path LOUDLY, never blocks); single
resident heavy engine <= 14.5GB NVML; engine-agnostic (no model is "primary");
fixes land in node code + the SAVED `workflows/otr_scifi_16gb_full.json`
in-place when graph wiring is needed (operator directive -- no runner patches,
no second json); UTF-8 no BOM; SFW; suite + Bug Bible green; commit AND push
every green commit (operator git policy 2026-06-10).

## Acceptance sketch (the panel hardens this)

ONE 30w production render: episode folder contains `stills/` with the open
still + portraits; the open still is macro-radio June-5-style (operator
eyeball); at least one video engine consumed a still as init (trace-stamped
`init_image` on a text-engine beat OR wan_i2v lane proof); suite + Bug Bible
green; audio byte-identical; all round-5 gates stay green.
