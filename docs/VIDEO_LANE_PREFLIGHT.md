# Video Lane Preflight

Run this checklist whenever a video lane is added or materially changed.
Format and acceptance protocol follow `SOURCE_BANK_PREFLIGHT.md` (the house
pattern): every hard item receives `PASS`, `FAIL`, or an explicitly allowed
`N/A`, plus evidence - a file and line, test name, validator output, or
receipt path. Save an `ID | status | evidence` matrix; the final receipt
names that matrix and its SHA-256. Any hard `FAIL` stops the lane. Machine
enforcement lives in `tests/test_lane_preflight_matrix.py` (spec S8c); this
document narrates those checks and is never a substitute for running them -
the `vram-recipe-lab/PREFLIGHT.md` rule.

Every gate below exists because a real lane failed it (2026-08-09/10 audits:
16 defects across 18 lanes; receipts in the lab repo and kibitz-runs/).

## Gate 1 -- Weights resolve

- G1.1 Every declared weight resolves via `folder_paths` or a documented env
  pin; no bare `os.path.exists` on a hardcoded default.
  *Origin: wan_i2v shipped dead - default path absent on this box.*
- G1.2 A missing weight produces a NAMED `EngineUnusable` from
  `assert_usable`, never a swallowed import.
  *Origin: registry imports swallow exceptions; a lane can vanish silently.*
- G1.3 **A GUARD CALIBRATED FOR ONE ARTIFACT IS A FALSE ACCUSATION AGAINST
  EVERY OTHER.** Any per-artifact constant -- byte floors, model filenames,
  recipe receipts, quant tokens -- must be a CLASS attribute a sibling lane can
  override, never a module-level constant read from inside a method. Two
  separate failures of this one class, and the second cost a live leg:
  * `eng_fastwan_8gb` records the first. Its parent's recipe accessors read
    module-level constants, so "a subclass declaring its own recipe would have
    SILENTLY rendered with `wan_ti2v`'s and stamped a FastWan receipt on the
    result" -- wrong pixels under a confident label.
  * The Ghost v3 peer hit the second on 2026-08-22. The module NAME had been
    moved to a class attribute; the BYTE FLOOR beside it had not. v3's official
    module is 1,673,262,583 bytes -- 144 MB smaller than the golden lane's
    `mm-p_0.5.pth` -- so a byte-perfect download was refused as *"only
    1673262583 bytes (< the 1700000000 floor) ... truncated or wrong file"*.
    Six minutes and a full leg to learn that a guard had inherited the wrong
    number.
  **The floor must sit BELOW its own artifact (or it refuses a perfect file)
  and within ~15% of it (or a badly truncated fetch slips through).** Both
  directions are pinned by test; a floor of 1 byte passes everything and guards
  nothing. When you add a sibling lane, ask what ELSE was sized for the parent.

## Gate 2 -- Canvas truth

- G2.1 GPU lanes with a fixed render size declare `render_canvas`; both axes
  /32-legal.
- G2.2 The declaration equals what the graph actually emits.
  *Origin: humo_14B_169 requested 1472x832 and rendered 832x480 - 3.07x.*
- G2.3 Every profile canvas either matches the declaration or the dead
  profile channel is documented for that lane.
  *Origin: nine lanes carry profile canvases read by nothing.*
- G2.4 Derived/intermediate canvases (two-stage halves, upscaler inputs) are
  also /32-legal. *Origin: ia2v stage-A 416x240; 240 % 32 == 16.*

## Gate 3 -- Contract matches runtime

- G3.1 `native_fps == target_fps == 25`; a 24 fps model declares 25 and
  converts at delivery (the Veo/H3 pattern), never relabels.
  *Origin: 192 frames labeled 25 fps = 7.68 s against an 8.00 s audio window.*
- G3.2 Discrete menus in FRAMES, boundaries pinned by test at both ends;
  menu arithmetic derived from the installed node's real limits, not a doc's
  rounded seconds. *Origin: the 107-vs-124 floor correction.*
- G3.3 Continuity declared explicitly on every adapter, never defaulted.
  *Origin: default CONTINUITY_NONE refuses chaining silently.*
- G3.4 Multi-clip partition literals for the lane's menu are pinned as test
  literals derived by running the real `partition_beat`.
- G3.5 **Any claim that depends on the SAMPLER is verified against the sampler
  this lane actually selects** - never against ComfyUI's general behaviour, and
  never inherited from a sibling lane or a vendor's notes. Every model is
  different; that is the point of the gate. The check is one grep: find the
  lane's `sampler_name`, open its implementation, and read what it does before
  writing down what it costs.
  *Origin: `ltx25_video`, 2026-08-19/20. Three comments - two in the adapter,
  one in the locked recipe, inherited from the lab - stated that negative
  conditioning is INERT at CFG 1.0 and that CFG 1.0 evaluates batch size 1.
  Both are true of ordinary ComfyUI (`comfy/samplers.py`: `sampling_function`
  sets `uncond_ = None` near cfg 1.0) and both are FALSE for that lane, because
  its locked sampler is `euler_ancestral_cfg_pp` - a CFG++ variant that passes
  `disable_cfg1_optimization=True` (`comfy/k_diffusion/sampling.py:1284`) and
  consumes `uncond_denoised` in its own step derivative (`:1297`). The lane had
  been running BOTH passes at CFG 1.0 the whole time.*
  *Why it is a gate and not a note: the false premise made an optimisation look
  free - feed the positive conditioning into both guider slots and skip a whole
  12B encode per shot - which would have silently changed every render. It was
  proposed during a panel, SURVIVED one reviewer, and died only because another
  checked which sampler was selected. Full write-up: `docs/OTR_STANDING_RULINGS.md`, section
  "CFG ON THE LTX 2.5 LANE".*

- G3.6 **`accepts_still` declared explicitly on every adapter, never left to
  the `required_inputs` fallback.** The operator picks a video model and an
  image model in two separate dropdowns (`OTR_VideoDirector` node 87), and the
  video lane is expected to render the still that the SELECTED image engine
  mints -- per role, independently. That join is one capability read:
  `otr_image_gen_dispatcher.engine_consumes_still` returns `accepts_still` when
  the lane declares it, and only otherwise falls back to looking for
  `init_image` in `required_inputs`. A lane that declares NEITHER resolves to
  False, mints no still, and never invokes the operator's chosen image model --
  the episode still renders, so nothing reports it. Visualization lanes are the
  legitimate exemption (`viz_camera`, `viz_green`, `viz_mxc_cpu`,
  `viz_mxc_mandala` declare `accepts_still = False`: procedural, no image gen
  at all) -- but they are exempt because they SAY so, not because they stayed
  quiet. Swept over the live registry by
  `tests/test_still_spine_engine_coverage.py::
  test_no_video_engine_is_silently_exempt_from_the_image_dropdown`, so a new
  lane is covered the day it registers.
  *Origin: operator invariant, 2026-08-21 -- "all video dropdowns should obey
  the image gen dropdowns unless of course viz, there is no image gen for
  visualization video models." Audited clean at that date (26 lanes obey, the
  4 viz lanes exempt, 0 silent); the gate keeps it that way.*

- G3.7 **The lane's `still_plan` is DECLARED and TRUE -- it is what tells the
  image phase what to mint.** The video lane is the authority on the image
  assets it needs, and it says so through a `still_plan` of `StillPlanRow`:
  `kind` (portrait / scene_open / scene_beat / scene_character / mesh_fodder /
  background_plate), `cardinality` (how MANY -- per_beat / per_subject /
  per_recurring_subject / per_bookend_role), `aspect` (the dimensions --
  wide / portrait / inherit_engine), `required` (always / never / conditional),
  `framing_geometry` and `style_tail_policy`. That declaration must match what
  the lane's renderer ACTUALLY consumes: `wan_ti2v` declares portrait `never`
  and is family `image_to_video`, so `render_driver` overrides its init with the
  per-beat scene still; `humo` declares portrait `always` and is
  `audio_driven_face`, so it keeps `init_image = portrait` and drives a mouth
  from it. **Check the declaration against `_SCENE_INIT_FAMILIES` and the lane's
  own render path, not against intuition** -- "it's an i2v lane so it needs a
  portrait" is exactly the wrong inference.
  *Origin: 2026-08-22. `still_word` had declared `kind="portrait"
  required="never"` since it was written, and nothing read it -- the module
  says so in terms: "Nothing in this module reads the plan for production."
  So every still_word episode minted a portrait per cast member that no
  consumer on that lane ever loads. FREE on an engine that will draw a face,
  which is why it hid for months; FATAL on `ideogram4_local`, which returns a
  safety placeholder for a person close-up and killed a live leg. The lane was
  right, the enumerator was not asking.*
  *The seam is the lane-derived role-set family that already carries the other
  four decisions into the image phase -- `still_aspects` (dimensions),
  `mesh_fodder_roles` (kind), `talking_roles` (framing), `still_word_roles`
  (which composer) and now `_portrait_free_roles_from_policy` (whether a
  portrait is minted at all). Swept by `tests/test_portrait_free_roles.py`;
  a new lane is covered the day it declares its plan, with no edit anywhere.*

- G3.7-SCOPED **A PROMPT-OWNED, STILL-FREE LANE MUST DECLARE WHAT IT OWNS.**
  Added 2026-08-22 with Ghost Signal (`animatediff15_video`), the first lane
  that takes NO still at all. The check applies **only** when `family ==
  "text_to_video"` AND `accepts_still is False`; every older still-owned
  text-to-video lane legitimately inherits its subject from a minted image and
  owes none of this. Once it applies, the lane must declare:
  * `subject_ownership = "prompt"` -- if the still does not own the subject,
    say what does;
  * a `prompt_profile` and a positive `prompt_budget_chars` -- so a composer is
    selected BY CAPABILITY and the driver cannot fall through to a generic seed;
  * a `style_join` (`compose`, `override`, or `pack:<id>`) -- how the visual
    style pack reaches the prompt;
  * a `motion_source` -- a lane with no still has no movement to inherit, so it
    must name its motion authority;
  * a `negative_prompt_binding`, **and the CODE is checked against it**: the
    render module must read the request's negative by name and must not carry
    the `get("negative_prompt") or <engine constant>` idiom. A declaration
    checking a declaration proves nothing (lesson L4).

  Enforced by `gate_g3_7` in `tests/test_lane_preflight_matrix.py`, invoked over
  the whole live roster. It is deliberately NOT a new matrix column -- the live
  matrix is G1-G7 and the docs already use Gate 8 for the solo smoke -- and
  deliberately not folded into `gate_g3_contract`, which is about frame
  contracts and applies to every lane.

  **WHICH SEAM ACTUALLY COVERS A NO-STILL LANE, stated exactly, because the two
  look alike.** `_portrait_free_roles_from_policy` looks for a plan ROW saying
  `kind=portrait required=never`, so it is INERT for a lane whose `still_plan`
  is EMPTY and it returns nothing for Ghost. That is correct: it exists to
  exempt a lane that consumes SOME stills but never a portrait (`still_word`).
  A lane that consumes none is covered by the stronger `accepts_still = False`
  gate at the image dispatcher, which mints nothing of any kind. Adding a
  portrait/never row to an empty plan to make the role set light up would be a
  declaration the lane cannot honour -- the exact unread-declaration defect
  G3.7 exists to end. Pinned by
  `tests/test_ghost_signal_lane.py::test_which_g3_7_seam_actually_covers_ghost_and_which_does_not`.

## Gate 4 -- Admission honesty

- G4.1 The lane has a QUALIFIED cost row / envelope key, OR its receipts say
  "admission NOT enforced" in words, on disk, reachable in the manifest.
  *Origin: a disqualified row enforced on one path and not the other; four
  lanes with no refusal at all; `vram_admission` written but read by nothing.*
- G4.2 The envelope key states engine, recipe/quant, canvas, frame rung, and
  boot lane; a key miss reports unenforced rather than borrowing a number.

## Gate 5 -- Audio law (V-1)

- G5.1 The adapter's canonicalize path runs `validate_silent_clip_contract`
  on its OWN emitted file. A `has_audio: False` literal is not evidence.
  *Origin: H3 natively produces audio; literals lie.*
- G5.1a A DIRECTORY-CLIP lane satisfies G5.1 through the NAMED twin
  `validate_directory_clip`, which proves every frame is really a PNG/EXR
  from its MAGIC BYTES -- a still image has no audio stream to carry, so the
  silence is a fact about the bytes. The gate is taught that name per lane
  (`DIRECTORY_CLIP_AUDIO_LAW`), never widened to accept any validator, and a
  twin assertion checks the named function actually refuses a mis-named
  non-image. *Origin: `mesh_stage` is the only directory-clip lane; its audio
  check read `has_audio` off the dict the adapter itself wrote, while frames
  were accepted by FILENAME EXTENSION -- so a file named `.png` containing a
  WAV counted as proof of silence (lane 10, 2026-08-11).*
- G5.2 A keeps-audio lane (the standalone music runner) declares a NAMED
  exemption here and never registers into episode assembly without a
  standalone-only boundary.

## Gate 6 -- Guards fire early and by name

- G6.1 Sage-sensitive lanes call `assert_sage_not_patched` inside
  `assert_usable`. *Origin: ltx_8gb shipped with no gate on the exact family
  BUG-070 names.*
- G6.2 Boot requirements are declared and probed against the RUNNING
  server's `comfy.cli_args.args` at ShotLock plan time; render-time checks
  are defence in depth only. *Origin: refusals firing after writer/TTS/
  master-freeze/stills were already paid for.*
- G6.3 Module-scope env reads go through the guarded numeric parser; a
  malformed env var must not delete the lane from the registry.
  *Origin: OTR_LTX_AV_RESERVE_VRAM_GB deleted ltx_audio_in, silently.*

## Gate 7 -- Public surface

- G7.1 Exactly one live menu id per internal engine
  (`exact_menu_option_for` proves 1:1); legacy aliases resolve via
  `_LEGACY_ENGINE_ALIASES` and never render as menu options.
- G7.2 Node-87 / variant workflow strings are GENERATED, never hand-typed;
  variants regenerate in the same commit as any profile change.
- G7.3 `ENGINE_MATRIX.md` regenerated in the same commit as ANY
  canvas/contract/registration change (the doc is a live drift gate).
- G7.4 `still_plan` declared and audit-clean; naming states what the lane
  is: audio-conditioned lanes say `audio_in`, portrait lanes say `portrait`,
  the `low`/`high` marker comes from a measurement receipt, never a guess.

## Gate 8 -- Solo smoke

- G8.1 One real render on the lane's declared boot lane: canvas probed,
  frame count exact, silence probed (or audio present for a G5.2-exempt
  lane), VRAM peak receipted, trim ratio logged when tail-trim fired.

## Receipt

`VIDEO_LANE_PREFLIGHT receipt: <lane> | <date> | matrix sha256 <...> |
suite run <test output path> | smoke receipt <path> | verdict PASS/FAIL`

## The family (create each sibling when its subsystem is next touched,
never as an empty paper checklist)

- `SOURCE_BANK_PREFLIGHT.md` - exists (the format authority).
- `VIDEO_LANE_PREFLIGHT.md` - this file; enforced by the S8c suite.
- `TTS_VOICE_PREFLIGHT.md` - exists (2026-08-16); enforced by
  `tests/test_tts_voice_preflight_matrix.py`. Seeded from the cross-engine
  Lemmy work, so its gates are the ones that actually bit: a degraded dropdown
  two engines short, a generator one command from deleting rows it could not
  recreate, and a route tier that would have raised at render time.
- Future, each backed by its own enforcement code before the doc is written:
  `STILL_LANE_PREFLIGHT.md`, `MUSIC_AUDIO_PREFLIGHT.md`,
  `LLM_WRITER_PREFLIGHT.md`, `UPSCALER_PREFLIGHT.md`. Seed their gates from
  this file's shape plus the Bug Bible's per-subsystem entries.
