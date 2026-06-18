# Coverage architecture -- WIRING decision (pass00)

## Operator ask (verbatim intent)
"We need an architecture where ALL video models -- and possibly all 3D -- accept
whatever image-gen is selected for their purpose (music / announcer / beats). We
do NOT want to whitelist the image source for EVERY video model. It should be one
and done, decided in ONE place." Plus a second idea: "maybe just say: every
video/3D model always looks at <episode>/stills for its image -- no init, no
gatekeeper function."

So pick the architecture + the exact wiring. Claude is the judge and will ground
every claim against the real code; the panel proposes.

## Grounded facts (verified against the repo, 2026-06-18)
- Image-MODEL selection is ALREADY one place: OTR_VideoDirector (shipped b8bb388).
  Each role's image engine is chosen there; OTR_ImageDirector consumes it.
- The dispatcher gate `_still_needed_for_role` (nodes/otr_image_gen_dispatcher.py
  ~292-315) decides whether to GENERATE a still for a role. It keys on:
  `"init_image" in tuple(getattr(eng, "required_inputs", ()))` of the role's
  SELECTED video engine. Bare `except: return True` (silent force-still on unknown
  engine -- violates the no-silent invariant).
- Video engines + required_inputs (the 20 registered):
  - Pure procedural floors that IGNORE a still: `abstract` (req=()), `visualizer`
    (req=("audio_ref",)). These synthesize their picture; a still is wasted.
  - Optional-still floors: `station_card`, `flux_still`, `still_kenburns` (use a
    still if present, else synthesize a fallback).
  - Require a still: `still_parallax`, `mesh_stage`, `triposr`, `ltx_av_talk`,
    `humo`/`humo_1.7B`/`humo_1.7B_169`/`humo_14B_169`, `wan_i2v`, `wan_ti2v`,
    and the dark 3D talkers (`triposg_talk`/`hunyuan3d_talk`/`trellis_talk`).
  - `ltx_video`: req=("text_prompt",) -- TEXT-only in required_inputs, BUT the
    adapter READS an optional init image (eng_ltx_video.py:431-452, used at 829;
    env OTR_ENABLE_LTX_I2V default ON). So it CAN do image->video but does not
    declare it -> the gate skips its still TODAY. This is the exact bug behind
    "flux2 images on silent LTX render nothing."
  - `ltx_av_music`: req=("text_prompt","audio_ref") -- no init in tuple.
- 3D lock: `requires_mesh_portrait=True` on the 3 character_3d talkers
  (eng_character_3d.py:264/333/405); the granularity lock
  (otr_image_director.py:133-174 three_d_locked_slots/enforce_3d_granularity_lock,
  re-checked in the dispatcher ~343-351) forbids per_beat mesh rebuild.
- Bases: `MotionEngineBase` (motion_common.py:301), `_CheapFamilyBase`
  (cheap_families.py:28, already carries a `uses_still` bool). 3D talkers + triposr
  are dark scaffolds with no base.
- role_compat.engine_fits_role (role_compat.py:107-130) gates role x required_inputs
  (a still-needing engine only fits a role whose ROLE_AVAILABLE_INPUTS supplies it).
- VALIDATED_ENGINES: video registry.py:277-297; image registry.py:154-160.

## Invariants the answer MUST preserve
Model-agnostic / no primary; role_compat stays the role filter; no silent fallback
(every skip/degrade is LOUD); single-resident <=14.5GB unchanged (wiring/metadata
only); workflow JSON otr_scifi_16gb_full.json stays source of truth; cold-import
clean (plain attrs, no NVML at import); UTF-8 no BOM; determinism.

## Candidate A -- capability flag + central usability (roundtable pass01 design)
Add to the video/3D adapter protocol: `accepts_still: bool`,
`still_input_name="init_image"`, `still_kind in {init_image, mesh_portrait}`.
Default DERIVED so real lanes accept the selected still; floors (abstract,
visualizer) + audio-only (ltx_av_music) opt OUT explicitly (`accepts_still=False`).
`_still_needed_for_role` dual-reads `accepts_still` (else falls back to
"init_image" in required_inputs). Centralize approval behind ONE helper
`image_engines.registry.usable(name, role) -> (ok, reason)` that both directors +
the dispatcher call; coverage = usable-image x accepts_still-video, filtered by
role_compat (no NxM table). Unify the 3D lock onto `still_kind=="mesh_portrait"`.
- Pro: keeps the "all-procedural episode invokes NO image model" accessibility win
  (floors skip the still). One declaration per engine; new engines inherit the base
  default -> no per-model whitelist. Forward-compatible with 3D.
- Con: still a (one) capability flag per engine; floors must remember to opt out.

## Candidate B -- convention, no gatekeeper (operator's second idea)
Delete `_still_needed_for_role`. The dispatcher ALWAYS generates the role's
selected image into <episode>/stills/<role>.png. EVERY video/3D engine ALWAYS
looks at that path for its init image and uses it if its render path wants one;
pure floors simply ignore the file. No capability flag, no gate.
- Pro: dead simple; "one and done"; literally no gatekeeper; a new engine needs
  zero metadata -- it just reads the stills folder.
- Con: ALWAYS renders an image even for a pure-procedural episode (visualizer /
  abstract) -> breaks the "accessible: no image model needed" floor + spends a
  flux render per beat that nothing consumes; the 3D mesh-portrait-vs-init kind
  distinction + the per_beat mesh-rebuild lock still need SOMETHING to express
  granularity (can't be pure convention); "always look at a fixed path" re-introduces
  the fixed-filename overwrite risk (S7) for same-dim beats.

## The decision to harden
1. A, B, or a hybrid (e.g. B's "always make the still available" + a tiny
   `consumes_still=False` opt-out ONLY for the 2 pure floors to keep the accessible
   no-image path)? Which best satisfies "one place, no per-model whitelist" WITHOUT
   breaking the accessible all-procedural floor and the 3D granularity lock?
2. Exact wiring for the chosen option: where the capability/convention lives, how
   the dispatcher decides, how ltx_video starts consuming the still, how 3D's
   mesh_portrait kind + per_beat lock are expressed, and how the "approval in one
   place" helper is shaped (or removed).
3. Name the correctness traps (silent-fallback, fixed-path overwrite, role_compat
   interaction, the dark 3D scaffolds) and the smallest reversible build order.
