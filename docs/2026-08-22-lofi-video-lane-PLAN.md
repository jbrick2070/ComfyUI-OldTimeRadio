# A lo-fi, deliberately artistic video lane for very low VRAM

**Status:** PLAN, pre-code. Driver anchor for a full four-round kibitz arc.
**Date:** 2026-08-22

**Operator ask, verbatim:** *"a new video lane ... think creative artistic for
people with really low vram, yeah its low res, its experimental"*; *"especially
a new 'video' dropdown entry that could occupy announcer, music, or character
beats"*; and the pointer to *"docs/VIDEO_LANE_PREFLIGHT.md ... and the how-to-add
walkthrough is docs/EXTENDING_OTR.md plus the 'HOW TO ADD YOUR OWN VIDEO ENGINE'
docstring now at the top of nodes/_otr_video_engines/__init__.py."*

---

## 0. What this lane IS, and the framing correction that produced it

The driver's first answer to "AnimateDiff for low VRAM?" was NO, on three
grounds: face drift, flicker, and 2023-era motion. **That answer measured the
candidate against a photoreal bar it was never trying to clear, and the operator
corrected it.** Read as a deliberate look -- rotoscope wobble, breathing
identity, scratchy dream-sequence drift -- those same properties are the
product. A radio drama scored to a hand-drawn, unstable, dreaming image is
on-genre in a way a clean LTX pan is not.

The lane's thesis, from which every declaration below follows:

> **This lane does not preserve. It dreams.** It is a stylizer, not a
> reconstructor. It is chosen BECAUSE it restyles, and its receipts must say so
> in words -- or a future reader will file the restyle as a defect and "fix" the
> lane into being a worse LTX.

That is mechanical, not editorial: gates G2, G3.6 and G4 are answered
differently by a lane that admits it restyles than by one claiming fidelity.

**Scope guard.** New capability + new dropdown row + new preflight receipt. This
is NOT story-quality work and does not reopen the 2026-08-04 freeze.

**Creative method:** commissioned separately and cold from Fable (r1), so the
artistic thesis is not framed by this engineering anchor. Its verdict lands
before r2 and may overturn section 2's leanings.

---

## 1. The contract, grounded

The walkthrough is a light code drop (`EXTENDING_OTR.md:9-18`): create
`nodes/_otr_video_engines/eng_<name>.py`, decorate with `@register` from the
namespace `registry.py`, add a `CAPABILITIES` row, and the dropdown
auto-populates on next boot.

| Surface | File / anchor | What must change |
|---|---|---|
| Adapter | `nodes/_otr_video_engines/eng_<new>.py` | new; implements the `VideoEngine` Protocol (`registry.py:58`) |
| Parent | `motion_common.py:606` `MotionEngineBase` | subclass it -- see 1a |
| Family | `schemas.py:31` `FAMILIES` | **`text_to_video`** -- requires `text_prompt`, NO `init_image` (R1). An earlier draft said `image_to_video`; that predated the no-still ruling and was caught by the Cursor pass. |
| Registration | `_otr_video_engines/__init__.py` | guarded `try: from . import eng_<new>` |
| Public menu | `_otr_shared/public_engines.py` | ONE row in `_PUBLIC_ENGINES`; a module-scope bijection assert fires at IMPORT and empties the ComfyUI menu if two public ids collide |
| Content family | `_otr_shared/content_oracle.py:55` | engine -> family row |
| Node 87 strings | `OTR_VideoDirector` | GENERATED, never hand-typed (G7.2) |
| Matrix doc | `docs/ENGINE_MATRIX.md` | regenerate `python tools/engine_matrix.py` (G7.3) |
| Canonical JSON | `workflows/otr_canonical.json` | SAME commit as the code (rule 0) |
| Profile | `config/profiles/` | a lane profile across the three visual roles |
| Preflight | `docs/VIDEO_LANE_PREFLIGHT.md` | 8 gates; `tests/test_lane_preflight_matrix.py` enforces |

### 1a. Why `MotionEngineBase` is the right parent

It already supplies four things this lane would otherwise have to reinvent:

- **`accepts_still = True` by inheritance** -- the one declaration the
  `__init__.py` docstring calls out by name, because silence resolves to False
  through a getattr fallback and the lane then never invokes the operator's
  chosen image model, *and the episode renders anyway so nothing reports it*.
- **AS-3 single-heavy-engine lease** on `prepare`, so it serialises with every
  other motion lane.
- **V-4 patcher-detach `teardown`** -- never `unload_all_models`.
- **`compute_real_frame_budget` -- MUST NEVER RUN ON THIS LANE.** An earlier
  draft of this bullet called its loop/ping-pong extension "the single most
  valuable inheritance for a low-VRAM lane." **That was wrong and it
  contradicts R3**: `acceptance.py` sets
  `DELIVERABLE_EXTENSION_MODES = ("none",)` and `grade_no_mirror` grades every
  beat. Caught first by Fable, restated by the Cursor pass. The base's real
  value here is the LEASE and the TEARDOWN, not clip extension -- and the
  wiring round must prove the extension path is unreachable, not merely
  unused.

### 1b. Hard invariants inherited from the namespace

- **Cold-import clean (V-12).** Module scope imports nothing heavy. torch and
  the AnimateDiff node classes are lazy, inside `load` / `render_clip`.
  `test_cold_import_no_heavy_libs` asserts it.
- **Capture node classes from a LIVE `/object_info` BEFORE coding.** This is the
  established pattern, stated in `__init__.py` for both `wan_ti2v`
  ("captured from a live /object_info before coding") and `ltx_8gb`. It is not
  optional here: AnimateDiff-Evolved is **not installed** (`ls custom_nodes/`),
  so its real class names and input signatures are currently unknown to us.
- **Registration is guarded** so a packaging quirk never breaks the namespace
  import -- but G1.2 still requires a NAMED `EngineUnusable` from
  `assert_usable`, never a swallowed import.

### 1c. Roles

Operator named all three. `role_compat.py:41-42` declares
`ANNOUNCER_VISUAL = "announcer_visual"` and `MUSIC_VISUAL = "music_visual"`.

**RESOLVED by the Cursor pass, verified against the file:** the third was never a
mismatch, it is an explicit MAPPING. `nodes/_otr_shared/slot_matrix.py`
`ROLE_TO_PROFILE_KEY` contains `"character_video": "character_visual"` -- so the
ROLE token is `character_video` and the PROFILE key is `character_visual`, by
design. Question closed; it leaves the wiring list.

### 1d. Naming is blocked on measurement

`public_engines.py` fixes `<model><version>_<low|high>_<capability>`, and lane 8
explicitly refused to let `low` mean "we think it runs on an 8 GB card" -- the
token must come from a measurement receipt. **So this lane gets no low/high
token until its own smoke measures one.** Any id before that is a placeholder.

---

## 1e. THE HARD CEILING: UNDER 4 GB VRAM (operator ruling 2026-08-22)

Operator: *"again we need to keep under 4gb vram."* This is the tightest
constraint on the lane and it outranks every aesthetic preference below.

**The arithmetic is brutal and must be faced before anything else.** Approximate
fp16 weights for this stack: SD1.5 UNet ~1.7 GB + AnimateDiff motion module
~1.7 GB + CLIP ~0.25 GB + VAE ~0.16 GB = **~3.8 GB of WEIGHTS, at the ceiling
before a single activation.** (Figures are estimates; measurement #1 replaces
them.) Consequences, all binding:

* **Sequential load/unload is MANDATORY, not an optimisation.** Encode text ->
  free; sample -> free; decode -> free. `MotionEngineBase`'s V-4 patcher-detach
  teardown matters more here than on any existing lane, and
  `unload_all_models` is still never the mechanism.
* **Motion-module SIZE becomes a primary backbone-selection criterion**, ranking
  alongside step count and the CFG/negative interlock -- not a footnote.
* **The smallest viable canvas wins.** This strengthens the 384x216
  recommendation: activations scale with pixel count, and it is 44% fewer
  pixels than 512x288.
* **A stage that runs after teardown is a SEQUENTIAL peak, not additive.** An
  enlarge/upscale stage does not sum with the render peak, provided teardown is
  real and proven.

### The upscale-vs-enlarge decision, and why ESRGAN is OUT

Operator's own vocabulary fix: *"enlarge could be more accurate."* The
distinction is load-bearing. **Enlarge** (lanczos / nearest) makes the picture
bigger and INVENTS NOTHING -- every delivered pixel traces to something really
rendered. **Upscale** (ESRGAN, SeedVR2) HALLUCINATES detail that was never
rendered, which is the exact lie a lane about honest degradation exists not to
tell.

Three independent reasons converge on pure enlargement:

1. **`spandrel_esrgan`'s VRAM is UNMEASURED.** The `~64 MB` in the adapter is
   the CHECKPOINT size, not VRAM; no measured figure exists anywhere in the
   repo. Adopting it under a 4 GB ceiling would be adopting an unknown.
2. **Real-ESRGAN x2plus is a LIVE SUSPECT in an unresolved artifact the
   operator personally reported** (`docs/GO_FORWARD_PLAN.md:1136`): tiling/mesh
   at 00:37 of `beneath_the_silvery_boughs`. The raw 832x480 render was CLEAN;
   the mesh appeared in the 1920x1080 composite. Suspects are ESRGAN x2plus,
   the bicubic landing resize, or the encode -- **still unpinned, because the
   ledger never records which upscale engine ran.** A new lane whose identity
   IS a clean coarse pixel grid must not enter that artifact class.
3. **It contradicts the identity.** A super-resolver's job is restoring
   sharpness the signal never had.

**So: the upscale engine stays OFF for this lane, and the enlargement is the
delivery chain in section 5.** `RealESRGAN_x4plus_anime_6B` (on disk, tuned for
flat illustration) is noted as the ONLY variant that would ever merit
re-opening this, and only after the 1136 artifact is pinned -- not before.

## 1f. CONVERGED DESIGN -- "Ghost Signal" (2026-08-22, post-panel)

Panel: Fable (r1 cold + r2), Codex (blind r1 + an operator-driven pass). The
driver is sole judge; every claim below was checked against the real files.

**IDENTITY.** One dominant form -- figure, radio, prop, emblem -- coalesces from
noise, becomes briefly readable, and partly loses itself again. Contours crawl,
colours misregister, shadows become objects. Instability tracks the audio's
dramatic pressure. Spoken name **Ghost Signal**.

**FULL FRAME. NO BLACK BARS. OPERATOR RULING 2026-08-22, and it is his stated
problem statement:** *"I'm for full frame I hate black bars that's my problem
statement lol."* The lane renders 16:9 and fills 1920x1080 edge to edge. No
pillarbox, no letterbox, no matte.

**LOW RESOLUTION IS ACCEPTED AND IS THE DECLARED IDENTITY. OPERATOR RULING
2026-08-22:** *"yeah I know it's not high res I am fine with that look."*
This is a RULING, not a tolerance, and it is written here because the failure
mode is a future contributor repairing the lane into a worse LTX. Consequences,
all binding:

* **DRIVER CORRECTION, same day.** An earlier draft of this bullet read "no
  quality-recovery stage, EVER -- the upscale engine stays `off`." That was the
  driver over-inferring a hard ban from a soft statement, and the operator
  corrected it within the minute: *"ok well have Fable do another pass -- I'd
  love a full 1920x1080."* The two statements are not in conflict. He ACCEPTS
  the low-res look; he would still LOVE real 1080p if it can be had honestly.
  So the reaching-1080p question is OPEN and is a live Fable pass, not closed.
  What stays closed is chasing sharpness the lane never had: no second
  diffusion pass, no hi-res fix, and RIFE never as coverage.
* **The `unsharp` pass is DROPPED.** `_scale_filter` couples lanczos to
  `unsharp=5:5:0.4:5:5:0.0`, whose stated purpose is the "soft-native fix" --
  recovering sharpness on lanes softer than their delivery. On THIS lane
  softness is the product, so the sharpener is working against the intent.
  Decoupling the two flags is the required change (smaller than adding a
  nearest mode).
* **No pressure to raise the canvas.** 512x288 is not a compromise to be
  escaped later; a bigger canvas is only ever justified by a defect, never by
  "it would look better".
* **Acceptance judges the look as INTENDED, not as degraded.** The preflight
  receipt says in words that low resolution and softness are the declared
  identity, so a later reader cannot file them as a bug.

**THIS KILLED THE ORIGINAL GHOST SIGNAL FRAMING.** Codex's section H -- a
256x256 square nearest-upscaled to 1024x1024 and centred on 1920x1080 with 448px
black margins, "a dark room around the receiver aperture" -- is REJECTED. That
aperture had been doing real identity work: it justified the tiny native
resolution as deliberate framing rather than limitation. Full frame removes that
alibi, so the identity must now survive on the SURFACE alone -- the coalescence,
the crawl, the misregistration. Re-anchoring is Fable's round-3 job.

**THE CANVAS ARITHMETIC, worked rather than wished.** 1080 = 2^3 x 3^3 x 5 --
only three factors of 2 -- so **no exact-integer nearest upscale to 1080 can
land on a /32-legal height. That option does not exist.** Candidates:

| Canvas | To 1920x1080 | /32-legal | Note |
|---|---|---|---|
| 512x288 | x3.75 uniform | BOTH axes | 4,4,4,3 block pattern; ltx_8gb's canvas |
| 640x360 | x3 exact integer | 360 is NOT | even blocks, needs a G2.1 exemption, 1.56x the pixels |
| 512x288 hybrid | x2 nearest -> resample | BOTH axes | hard edges without the 4,4,4,3 irregularity |

**THE UNLOCK -- one beat, one timeline.** 16 frames is the CONTEXT WINDOW, not
the clip length. Non-looped sliding Context Options (Standard Static, length 16,
overlap 4) process a whole beat in ONE render, with VRAM bounded by context
length rather than beat length. A long beat costs TIME, not memory. This kills
the chaining seam problem outright and satisfies "original seconds of render for
every beat" natively. **UNVERIFIED -- upstream doc claim about a package that is
not installed. This is measurement #1 after install, before any adapter code.**
If it fails, the fallback is Fable's panel grammar (below), which is retained
for exactly that reason.

**RATE.** Operator ruling 2026-08-22: **12.5 fps native, uniform hold-2, exact
25 fps delivery.** `ceil(D * 12.5)` unique frames per beat. 12.5 is exactly half
of 25, so the cadence is uniform -- no ragged 3-3-3-3-4. Chosen over 8 fps
(Fable/Codex) and over 25 fps native (3x compute, loses the cadence).

**PROMPT.** Own budget, ~320 chars / ~65 encoder tokens, target 260-300; verify
the real tokenizer after install. `_LTX_MOTION_PROMPT_MAX = 240` is UNTOUCHED --
its name says whose it is. Slot order:

> pack cue -> subject identity -> framing lock -> beat action -> motion ->
> emotion -> one story accent -> shot law

Sources, all confirmed to exist: announcer -> `announcer_subject_face`; music ->
`announcer_subject_object` or `open_subjects`; character -> `char_id` -> cast
appearance distilled to one invariant identity spine. Motion: character ->
`resolve_motion_clause_text`; announcer/music -> the pack's `motion_registers`.
Emotion -> `traits`, `beat_intent`, `arc_phase` (all real: 21 / 7 / 9 modules).

**THE NEGATIVE ACTUALLY BINDS HERE** (Fable, and it is the sharper claim). This
lane runs real CFG (~7); the distilled LTX lanes run CFG 1.0 where negatives are
INERT. So the lettering guard genuinely works -- and per G3.5 it must be
verified against the sampler THIS lane selects, never inherited from a sibling.

**STYLE PACKS -- compose.** Not override, not a tenth pack. Both lanes agreed
independently. Pack owns medium, palette, lighting, rendering vocabulary,
surface movement, negative. The lane owns aperture, one dominant form, coarse
pixels, imperfect coalescence, kinetic intensity, black field.

**CHARACTER ROLE.** One heraldic actor: exact silhouette, one asymmetrical
landmark, one costume item, one prop, mid-shot, one action. Both lanes converged
on this. **Acceptance judges silhouette + costume + action, never facial
identity.** The named failure is semi-photorealistic face soup.

**THE MOTION APERTURE** (Codex, untested): masked `scale_multival` /
`effect_multival` -- calmer motion at centre, stronger mutation at the
perimeter, so a face can hold while the surrounding ether stays alive.
Unverified at 256x256.

**DELIVERY.** 16:9 native, FULL FRAME to 1920x1080, no bars (ruling above).
Canvas: see the SPEC (`docs/2026-08-22-GHOST-SIGNAL-SPEC.md` section 5), which is the document of record for delivery.
No crop, no stretch, no blur-fill, no second diffusion pass. **Requires a NEW
mode in `otr_silent_composite._scale_filter`** (`nodes/otr_silent_composite.py:172`),
which today offers only lanczos+unsharp (`sharpen=True`) or bilinear
(`sharpen=False`) -- there is no nearest. Note the existing chain also applies
`force_original_aspect_ratio=decrease` then `pad=...:color=black`; a full-frame
16:9 source pads to nothing, so the bars problem disappears at the source rather
than needing the pad suppressed.

**NAME.** Register `animatediff15_video`, display label "AnimateDiff -- Ghost
Signal". Codex proposed `animatediff_p05_video`; the driver moved the module
token out of the public id because swapping `mm-p-0.5` later would make the id
lie or need a legacy alias, and that table trips a bijection assert at IMPORT.
No low/high token until a measurement receipt exists.

**RETAINED FALLBACK -- Fable's panel grammar.** If Context Options does not
bound VRAM as documented: beats partition into cells, `CONTINUITY_NONE`, joined
as deterministic-seeded panels cut on holds, bound by byte-identical subject
text, a named light direction, and a rotating panel directive (medium shot /
detail insert / profile). Not dead -- contingent.

## 2. The forks the panel must break

This is why the item earns a full arc rather than a grep-and-fix. Each has more
than one defensible answer and the driver does not have a confident pick.

### F1 -- CLOSED BY OPERATOR RULING, 2026-08-22. No still.

*"AnimateDiff-Evolved can generate video directly from a text prompt and noise
-- no still image required ... so it needs to be sure not to try to generate
stills."* He accepts the cost in his own words: *"greater abstraction, subject
mutation, and flicker -- which may suit your low-quality experimental-art
aesthetic."*

So: **`text_to_video` family** (`schemas.py:31`, requires `text_prompt`), no
init image, no ControlNet, no IPAdapter. Prompt plus noise.

**This does NOT violate G3.6, and the reason is worth writing down** -- the
driver checked rather than assumed, because a no-still lane is exactly what that
invariant was built to catch.
`tests/test_still_spine_engine_coverage.py::test_no_video_engine_is_silently_exempt_from_the_image_dropdown`
polices **silence, not the value**, in its own words: *"refusing the dropdown
must always be a DECLARED choice ... Derived from the live registry, never a
whitelist ... cannot be exempted by omission."* The four `viz_*` lanes are
exempt **because they say so**, not because they are visualizers. A lane that
declares `accepts_still = False` out loud, with its reason, satisfies the gate.
No test edit, no whitelist entry, no invariant bent.

**Consequence that must be declared, not discovered:** this lane never invokes
the operator's chosen image model for its roles. That is now intended. It goes
in the adapter docstring and the receipt, or a future reader repairs it into
being a worse LTX.

### F2 -- Motion backbone (operator gave a starting recipe)

Operator's start: **SD1.5 + the `mm-p-0.5` motion module, 256x256, 16 frames,
8 fps, 8-12 steps.** AnimateDiff-Evolved also serves SD1.5 v1/v2/v3, AnimateLCM
(4-8 step) and AnimateDiff-Lightning. Open: whether to hold `mm-p-0.5` or let
the panel argue AnimateLCM for the step count. This choice fixes VRAM, steps,
wall clock and the F4 licence answer at once. **Nothing is confirmed until the
node pack is installed and its real classes are read from a live
`/object_info`** -- the module name above is the operator's, unverified by us.

### F3 -- fps: the recipe lands ON the approved pattern

G3.1: `native_fps == target_fps == 25`, or declare 25 and CONVERT AT DELIVERY
(the Veo/H3 pattern) -- never relabel. The gate's origin is a lane that shipped
192 frames labelled 25 fps = 7.68 s against an 8.00 s window.

**16 frames at 8 fps is exactly 2.000 s, which is exactly one beat.** That is a
clean number, and it means the lane renders its native 16 and converts to 50
canvas frames at delivery -- squarely the Veo/H3 route the gate blesses, not a
relabel. Open: what the conversion IS (duplicate, blend, interpolate), because
that choice is visible on screen -- duplication gives a hard 8 fps stutter that
may be exactly the look, interpolation smooths it away.

### F6 -- NEW: canvas, aspect, and a tension the operator should see

`_scale_filter` composites with `force_original_aspect_ratio=decrease` then
`pad=...:color=black`. So **a 256x256 square render delivers PILLARBOXED** --
a square image floating in black inside 1920x1080.

That may be a feature: a small square window, a peephole, a silent-film iris, is
a deliberate and attractive presentation for an experimental lane. But it must
be a choice, not a surprise.

The arithmetic behind it, which constrains the whole lane: for a canvas to be
true 16:9 AND /32-legal on both axes (G2.1), the smallest legal size is
**512x288** -- which is already `ltx_8gb`'s canvas. There is no smaller
true-16:9 /32-legal canvas. So "smaller than the cheapest existing lane" and
"fills the 16:9 frame" cannot both hold. Three ways out: accept the square and
frame it as style; take 512x288 and win on look rather than size; or accept a
near-16:9 like 448x256 and letterbox a hair.

### F4 -- Licence, and it lands in the FLOOR tier

`tools/audit_model_license.py` exists; `docs/H3_LICENSE_ATTESTATION.md` sets the
pattern. AnimateDiff-Evolved itself is permissive, but a usable SD1.5 checkpoint
is CreativeML OpenRAIL-M -- use restrictions, not OSI-open. **This is the lane a
stranger with a weak card runs first when the repo goes open source.** Panel:
acceptable at the floor, or does it need an Apache/MIT-licensed SD1.5
derivative?

### F5 -- Which roles, honestly

Character beats are AnimateDiff's worst case -- a held face is exactly where
identity wander reads as defect rather than style, and "a character's face
changing between beats" is a NAMED open correctness defect in CLAUDE.md. Music
beats are its best case. Options: ship all three and let the look carry it; ship
music + announcer and declare character out of scope; or ship all three with a
per-role recipe that damps motion on character beats. Driver leans third, has no
evidence. **Fable's creative method (section C of its brief) speaks directly to
this and should be read before the panel rules.**

---

## 3. What is NOT in question

- `accepts_still = True` -- G3.6 invariant; not a viz lane.
- Silent clip: `validate_silent_clip_contract` on the lane's OWN emitted file
  (G5.1). A `has_audio: False` literal is not evidence.
- `continuity` declared EXPLICITLY, never defaulted (G3.3; the lane-10 lesson --
  six lanes once inherited the right value because nobody decided it).
- `render_canvas` declared, both axes /32-legal (G2.1). `ltx_8gb` is at
  `(512, 288)` and is the current cheapest; this lane sits at or below it.
- `still_plan` declared and audit-clean (G7.4).
- Canonical JSON changes in the SAME commit as the code (rule 0).
- **Publication to `otr/obs/` is the success signal.** A leg that does not reach
  it did not pass, however green its logs are.

---

## 4. Order of work

1. **r1 arc** -- is a lo-fi stylizer lane the right shape at all, given
   `still_pan` / `still_motion` already deliver motion at zero VRAM? If a
   diffusion lane wins, does it win on the LOOK rather than on the VRAM?
   *(Fable is answering the creative half of this cold, in parallel.)*
2. **r2 coding** -- F1 / F2 / F3 resolved; adapter shape; base class confirmed.
3. **r3 wiring** -- registry, `CAPABILITIES` row, public-menu bijection, node 87
   generation, canonical JSON, profile, ENGINE_MATRIX regeneration.
4. **r4 convergence** -- no new must-fix.
5. Install the node pack; capture node classes from a live `/object_info`;
   build; run the 8 preflight gates; solo smoke; measure for the low/high token;
   **publish to `otr/obs/`.**

---

## 5. The question behind the question

The operator's real ask is *people with really low VRAM should get something
worth watching.* AnimateDiff is one answer. The repo already ships `still_pan`
and `still_motion` at zero VRAM, zero flicker and zero face drift. The panel
should say plainly whether a lo-fi diffusion lane beats a well-directed moving
still -- and if it does, it wins on being ALIVE, not on being cheap.
