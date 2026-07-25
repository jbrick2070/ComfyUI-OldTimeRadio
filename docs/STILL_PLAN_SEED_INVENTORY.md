# Still-plan seed inventory -- what each path asks for TODAY

**Captured 2026-07-25 at HEAD `9d1874f1`** by driving the real producer
(`nodes/otr_meta_brief_image_prompt.derive_image_prompts`), not by reading it.
Every prompt below is the ACTUAL composed string for a representative episode.

**Why this file exists (operator, 2026-07-25):** the per-model still-plan
architecture replaces five role-indexed capability maps with one independent
plan per video model. This is the map of what to TRANSPLANT -- the still
requirements and the prompt content -- so nothing learned is re-derived or
lost. Treat it as the seed data for the new plans and as the fixture the parity
test compares against.

Representative inputs: setting `a derelict orbital station`; one cast row
(`c01` / BABA, "a tall weathered spacer with a scar"); announcer, character and
music-close beats; visual style `sci_fi_radio`.

## The three prompt LAYERS (operator law, 2026-07-25)

> "still and video prompts must respect the visual model and contain the
> [essence] of the meta brief, of the actual script story or beats."

Every composed prompt today is three layers, in this order. The new
architecture changes only WHO supplies layer 2:

1. **SUBJECT** -- from the meta brief / story pack / cast. The story-pack
   `open_subjects` template, the brief's `setting`, or the cast row's
   appearance. Stamped in `prompt_field_source`
   (`open_subjects:<key>`, `cast:appearance`, `plate_look`, `brief:radio_form`).
2. **FRAMING / GEOMETRY** -- per image KIND: what shape, what must be visible,
   what must be excluded. *This is the layer the per-model still plan owns.*
3. **STYLE TAIL** -- the visual-style authority (`_vstyle`, resolved ONCE at the
   image-prompt entry), e.g. `sci_fi_radio` contributing "35mm film grain,
   broadcast-distressed cinematic aesthetic, centered composition, no on-screen
   text" after the shared "timeless cinematic aesthetic, cinematic, 35mm film
   look, subtle film grain, volumetric lighting, anamorphic lens, heavy
   vignette, muted color grade, sharp focus".

A model's plan may only contribute layer 2. It may never replace layer 1 or
layer 3, and it may never decide style -- that stays the style authority's job
(and under THE LAW, style is never a reject reason).

## Per-KIND inventory (verbatim current output)

### `portrait` -- 832x1216 (PORTRAIT)
- Subject: cast `portrait_prompt` + brief setting.
- Framing: "in-character cinematic three-quarter portrait, full head and face
  clearly visible with natural headroom above the head (never crop the top of
  the head), period-accurate costume and environment, dramatic film lighting".
- Style tail: present.
- **GAP FOUND:** this kind carries `prompt_field_source = None` and
  `visual_style = None` -- the portrait path is the one kind with NO provenance
  and NO style stamp, even though its text ends with the style tail. Fix while
  transplanting.

### `scene_open` -- 1472x832 (WIDE), `open_subjects:synthetic`
- Subject: story-pack open subject ("a sleek space-station communications
  console warming up on a table, glowing dials and tubes").
- Framing: "warm filament glow, full-frame macro, centered subject".
- Style tail: full `sci_fi_radio` tail.

### `scene_beat` -- 1472x832 (WIDE), `open_subjects:announcer`
- Subject: story-pack announcer subject ("a sleek space-station communications
  console in a broadcast booth, glowing warmly, lit dials and tubes").
- Framing: "cinematic three-quarter framing, people shown with full heads and
  clear headroom inside frame, faces unobstructed, balanced composition".
- Style tail: full.

### `scene_character` -- 1472x832 (WIDE), `cast:appearance`
- Subject: the character's appearance, LEADING the prompt.
- Framing: "cinematic medium shot, the character framed within a wide 16:9
  environment, full head and shoulders with clear headroom inside frame, face
  unobstructed, balanced landscape composition".
- Style tail: full + "no on-screen text".
- Nugget (BUG 1, 2026-06-20): this is a 16:9 CHARACTER shot, never the vertical
  portrait and never a generic radio scene still.

### `mesh_fodder` -- 832x1216, `brief:radio_form`, `mesh_subject_id=radio_host`
- Subject: the brief's radio form ("a sleek space-station communications
  console").
- Framing (the "clay blob" lesson, 2026-06-21, and the most engine-specific
  text in the tree): "single centered subject, simple clean unbroken
  silhouette, smooth solid form, plain matte solid-colour clothing, short tight
  neat hair, neutral symmetrical forward stance, full unoccluded three-quarter
  view, entire head and body clearly visible, plain seamless neutral mid-grey
  studio backdrop, even soft diffuse frontal lighting, no hard shadows, no
  props, sharp focus, full natural color".
- Style tail: DELIBERATELY MINIMAL -- only "timeless cinematic aesthetic". A
  mesh subject must not carry film grain / vignette / colour grade, because
  those bake into the mesh. **Transplant this restraint explicitly**; it is the
  clearest case of an engine needing its own prompt rules.

### `scene_background_plate` -- 1472x832 (WIDE), `plate_look`
- Subject: the brief setting.
- Framing: "empty establishing environment, no people, no subject, no
  characters, wide 16:9 cinematic scene, atmospheric depth, period-accurate
  set".
- Style tail: full + "no on-screen text".

## Which model asks for what (live registry, 2026-07-25)

| Video model | Still plan as it behaves today |
|---|---|
| `wan_ti2v`, `wan_i2v`, `ltx_8gb`, `word_razzle`, `cloud_*` i2v | 1 `scene_*` still per beat (WIDE) + `portrait` per subject |
| `still_word`, `still_pan`, `still_flat`, `still_motion` | 1 `scene_*` still per beat (WIDE) -- and note they declare only `text_prompt` in `required_inputs`, so requiredness must be DECLARED, never derived |
| `humo`, `humo_1.7B` | `portrait` per subject, PORTRAIT aspect + scene still kept as OOM-fallback insurance |
| `humo_1.7B_169`, `humo_14B_169` | same, WIDE aspect |
| `ltx_audio_in` | scene still per beat + the radio-face path; lip-sync means it wants a FACE-FORWARD portrait |
| `mesh_stage` | `mesh_fodder` + `scene_background_plate` per beat -- BOTH required, never a scene still |
| `viz_camera`, `viz_green`, `viz_mxc_cpu`, `viz_mxc_mandala` | NOTHING. An all-procedural episode invokes no image model at all |
| `ltx_video`, `google_*` t2v | declare `accepts_still=True`; still enumerated per beat |

## Other content that must ride along

- **Seeds:** the opening radio still has its own seed lane
  (`OTR_RADIO_BOOKEND_SEED`) so the open is stable across runs;
  `mesh_subject_id` keys the mesh cache on a stable per-subject file rather than
  the per-beat still hash.
- **Negatives:** `visual_safety_negative(...)` is applied at the dispatcher, and
  the prompt hash is recomputed AFTER that augmentation -- so any new plan must
  keep the final hashing boundary at the dispatcher, not the producer.
- **Dimensions:** WIDE = 1472x832, PORTRAIT = 832x1216 as composed today
  (the render canvas is a separate profile knob).

Regenerate this inventory with `tmp/_kbA_harvest.py` (throwaway probe; the
recipe is: drive `derive_image_prompts` per model and print one exemplar per
kind with its `prompt_field_source`, `visual_style` and dimensions).

## ADDENDUM 2026-07-25 -- the MECHANISM map (captured at HEAD `90e52f13`)

Everything above records what each path REQUIRES. This section records HOW
that requirement is decided, engine by engine, because the mechanism is what
the migration deletes and the requirement is not enough to rebuild it.
Captured by driving the live registry and calling
`render_driver._still_spine_requires_scene` for all 31 registered internal
engines (probe recipe at the end of this file).

### `_still_spine_requires_scene` has FOUR fall-throughs, not one list

They fire in this order, first match wins:

1. **`id_list`** -- the hardcoded `("still_pan", "still_flat", "still_word",
   "ltx_audio_in")` at `render_driver.py:635-637`.
2. **`family`** -- `_SCENE_INIT_FAMILIES = frozenset({"image_to_video",
   "static_motion"})` (`render_driver.py:791`).
3. **`required_inputs`** -- `"init_image" in required_inputs`.
4. **`provider_side and accepts_still`**.

Families `audio_driven_face` and `character_3d` return False before 3 and 4.

Live result -- the 21 engines that require a scene still, BY MECHANISM:

| Mechanism | Engines |
|---|---|
| `family` (9) | `wan_ti2v`, `wan_i2v`, `ltx_8gb`, `mesh_stage`, `still_motion`, `word_razzle`, `cloud_wan_i2v`, `cloud_vidu_q2_pro_fast_720p`, `cloud_vidu_q2_pro_fast_720p_sfx` |
| `id_list` (4) | `ltx_audio_in`, `still_flat`, `still_pan`, `still_word` |
| `provider_side` (6) | `google_veo_video`, `google_omni_video`, `google_vid_sfx_omni`, `google_vid_sfx_veo_lite`, `google_vid_sfx_veo_fast`, `google_vid_sfx_veo_pro` |
| `required_inputs` (2) | `cloud_seedance_2`, `cloud_wan_i2v_audio` |

**TRAP 1 -- `still_motion` is NOT in the id list.** The four `still_*`
engines are one row in the table above and TWO different mechanisms here:
`still_pan` / `still_flat` / `still_word` are family `static_image_gen`
(not in `_SCENE_INIT_FAMILIES`, which is exactly why the id list exists),
while `still_motion` is family `static_motion` and rides branch 2. Deleting
the id list fixes three of the four and silently leaves the fourth on an
untested branch. The parity fixture asserts PER ENGINE across all four
branches -- never against the id list.

**TRAP 2 -- `mesh_stage` DOES require a scene-slot row.** Its class declares
`family = "image_to_video"` (`eng_mesh_stage.py:306`), which is in
`_SCENE_INIT_FAMILIES`, so `_still_spine_requires_scene` returns True and
`validate_and_repair_still_spine` validates BOTH the mesh fodder row AND the
beat's scene row -- the background plate is what fills that scene slot.
`requires_mesh_fodder = True` is the capability that stops the cinematic
scene still being fed to the mesher, and the engine's own comment
(`:317-325`) forbids re-deriving that routing from an engine-name/family
check or from `required_inputs`/`uses_still`. The line "never a scene still"
above is true of the MESHER'S INPUT and false of the validator. If the plan
drops the plate's scene-slot role, the plate's requirement disappears.

**TRAP 3 -- `_still_spine_requires_scene` is NOT the complete oracle for
"required". Two more gates live in the init-selection branch, and BOTH
default ON.** Corrected 2026-07-25 after a kibitz r2 grounding (codex
`gpt-5.6-sol` high); the first version of this entry read the helper as
authoritative and was wrong.

- **`ltx_video` -- the LTX-I2V gate.** The helper returns False for it
  (family `text_to_video`, no `init_image` in `required_inputs`, not
  `provider_side`). But `render_driver.py:1801-1817` requires the beat's
  scene still whenever `OTR_ENABLE_LTX_I2V` is set, and it **defaults to
  "1"** -- `os.environ.get("OTR_ENABLE_LTX_I2V", "1") == "1"`. A missing
  still raises `RenderError` with "NO FALLBACK to text-only rendering".
  So `ltx_video` is `required=True` in the default environment, by a
  mechanism the helper cannot see. This is the fifth mechanism.
- **`ltx_audio_in` -- the IA2V portrait gate.** `render_driver.py:1709-1721`
  computes `_s4_portrait = (engine == "ltx_audio_in" and role ==
  "character_video" and _ia2v_talking_register_active(engine))`, and with no
  portrait in the ledger raises "IA2V TALKING register: character beat ...
  has NO portrait -- NO FALLBACK to the wide scene still (face too small to
  lip-sync; proof7 A/B 2026-07-02)". So `ltx_audio_in` requires a real cast
  PORTRAIT on character beats under the talking recipe, deliberately
  vertical into the wide canvas (center-crop covers it; "no pillarbox, no
  squash"). This is the sixth mechanism.

Consequence: a plan row's `required` column may not be transplanted from
`_still_spine_requires_scene` alone. The parity fixture must freeze THREE
outputs per engine -- objects the producer authors, targets the dispatcher
materializes, and assets the render path actually validates or raises on --
because those three disagree for `ltx_video` and `ltx_audio_in` today.

**TRAP 4 -- the HuMo portrait requirement is keyed on FAMILY, in a seventh
place.** `validate_and_repair_still_spine` requires a portrait row via
`if family == "audio_driven_face"`, not via any engine declaration. That
covers `humo`, `humo_1.7B`, `humo_1.7B_169`, `humo_14B_169` and
`cloud_kling_avatar`, all of which return scene=False on the face branch.

**TRAP 5 -- there are TWO distinct "face" mechanisms, not one lips flag.**
`wants_talking_prompt` is declared by exactly ONE engine in the whole
registry: `ltx_audio_in` (True). No HuMo variant declares it. So the HuMo
PORTRAIT face rides the family string (trap 4) while the `ltx_audio_in` WIDE
mouth-forward radio face rides the hook. Both are "this model asks for a
face-forward still"; they differ only in aspect and in which mechanism
currently expresses it.

### Per-engine facts from the live registry (31 engines)

`still` = declared `accepts_still` (`None` = falls through to the
`init_image in required_inputs` dual-read in `engine_consumes_still`).
`talk` = `wants_talking_prompt()` where callable.

| engine | family | aspect | still | mesh | talk | scene | branch |
|---|---|---|---|---|---|---|---|
| `cloud_kling_avatar` | audio_driven_face | wide | None | - | - | no | face |
| `cloud_seedance_2` | audio_conditioned_video | wide | None | - | - | yes | required_inputs |
| `cloud_vidu_q2_pro_fast_720p` | image_to_video | wide | None | - | - | yes | family |
| `cloud_vidu_q2_pro_fast_720p_sfx` | image_to_video | wide | None | - | - | yes | family |
| `cloud_wan_i2v` | image_to_video | wide | None | - | - | yes | family |
| `cloud_wan_i2v_audio` | audio_conditioned_video | wide | None | - | - | yes | required_inputs |
| `google_omni_video` | text_to_video | wide | True | - | - | yes | provider_side |
| `google_veo_video` | text_to_video | wide | True | - | - | yes | provider_side |
| `google_vid_sfx_omni` | text_to_video | wide | True | - | - | yes | provider_side |
| `google_vid_sfx_veo_fast` | text_to_video | wide | True | - | - | yes | provider_side |
| `google_vid_sfx_veo_lite` | text_to_video | wide | True | - | - | yes | provider_side |
| `google_vid_sfx_veo_pro` | text_to_video | wide | True | - | - | yes | provider_side |
| `humo` | audio_driven_face | portrait | True | - | - | no | face |
| `humo_1.7B` | audio_driven_face | portrait | True | - | - | no | face |
| `humo_1.7B_169` | audio_driven_face | wide | True | - | - | no | face |
| `humo_14B_169` | audio_driven_face | wide | True | - | - | no | face |
| `ltx_8gb` | image_to_video | wide | True | - | - | yes | family |
| `ltx_audio_in` | audio_conditioned_video | wide | True | - | **True** | yes | id_list |
| `ltx_video` | text_to_video | wide | True | - | - | **no** | default |
| `mesh_stage` | image_to_video | wide | None | **True** | - | yes | family |
| `still_flat` | static_image_gen | wide | True | - | - | yes | id_list |
| `still_motion` | static_motion | wide | True | - | - | yes | **family** |
| `still_pan` | static_image_gen | wide | True | - | - | yes | id_list |
| `still_word` | static_image_gen | wide | True | - | - | yes | id_list |
| `viz_camera` | abstract | wide | False | - | - | no | default |
| `viz_green` | abstract | wide | False | - | - | no | default |
| `viz_mxc_cpu` | abstract | wide | False | - | - | no | default |
| `viz_mxc_mandala` | abstract | wide | False | - | - | no | default |
| `wan_i2v` | image_to_video | wide | True | - | - | yes | family |
| `wan_ti2v` | image_to_video | wide | True | - | - | yes | family |
| `word_razzle` | image_to_video | wide | None | - | - | yes | family |

Drift check: no engine's class `family` disagrees with
`render_driver.engine_family()`.

### Plan keys must be INTERNAL ids

`nodes/_otr_shared/public_engines.py` is the single normalization table.
Eight strings resolve INTO the ids above and must never carry a plan of
their own: public `ltx_8gb`, `wan_8gb` -> `wan_ti2v`,
`ltx23_16gb_audio_in` -> `ltx_audio_in`, `ltx23_16gb_video` -> `ltx_video`;
legacy `flat_still` -> `still_flat`, `flux_still` -> `still_pan`,
`still_kenburns` -> `still_motion`, `visualizer` -> `viz_green`.

### Stale prose to fix during transplant

`eng_humo.py:497` still says "Degrades on to the zero-VRAM still floor
(humo -> humo_1.7B -> still_motion)" while `:502` sets
`fallback_engine = None  # NO FALLBACKS`. The degrade chain was ripped
2026-07-02 (`render_driver.py:46-49`; `render_shot` at `:2468-2495` states
there is no engine swap and no still-image floor). That stale sentence is
where this file's "scene still kept as OOM-fallback insurance" line came
from. Whether HuMo should still mint that scene still is a SEPARATE
operator decision, deliberately not folded into the migration.

Regenerate this addendum with `tmp/_kbA_sp_branches.py` (throwaway probe).

## PARITY MATRIX 2026-07-25 -- what the producer actually emits, per engine

Captured at HEAD `90e52f13` by driving the real producer
(`derive_image_prompts`) once per registered internal engine, with every video
role set to that engine and the director's own `_role_aspects` / `_role_talking`
maps fed in. Default environment. Pure CPU. Probe: `tmp/_kbA_sp_parity.py`,
full record `tmp/_kbA_sp_parity.json`.

### The producer is very nearly ENGINE-INVARIANT

27 of 31 engines produce the IDENTICAL fingerprint: 4 required targets
(`still_b000_music_open`, `still_b001`, `still_b002`, `still_b005`), object
kinds `portrait` + `scene_open` + `scene_beat` + `scene_character`, dims
1472x832 and 832x480. Only three shapes exist in the whole registry:

| Shape | Engines | Required targets | Object kinds |
|---|---|---|---|
| scene spine | 26 | 4 | portrait, scene_open, scene_beat, scene_character |
| mesh fork | `mesh_stage` | 8 | mesh_fodder, scene_background_plate, portrait |
| nothing | the 4 `viz_*` | 0 | none |

`humo` and `humo_1.7B` are the ONLY engines whose objects include the
832x1216 PORTRAIT dim; their `_169` wide siblings mint 832x480 instead, which
is the 2026-06-17 nugget holding exactly as recorded.

This is the strongest argument for the operator's simplicity directive: the
per-engine variation the five-authority scatter exists to express is, in
practice, three shapes and one aspect knob. The plan table should be boring.

### The picked-vs-effective defect, measured

In the DEFAULT environment (`OTR_ENABLE_HUMO_HOSTS` unset) the bookend
redirect fires for all four HuMo variants:

| Picked | Roles redirected | Effective engine | `_role_aspects` still says |
|---|---|---|---|
| `humo` | announcer_visual, music_visual | `ltx_audio_in` | portrait |
| `humo_1.7B` | announcer_visual, music_visual | `ltx_audio_in` | portrait |
| `humo_1.7B_169` | announcer_visual, music_visual | `ltx_audio_in` | wide |
| `humo_14B_169` | announcer_visual, music_visual | `ltx_audio_in` | wide |

`humo` and `humo_1.7B` are therefore minting PORTRAIT-aspect bookend stills
for a renderer that is WIDE. No other engine in the registry has a
picked != effective row in the default environment.

**This is why exact HEAD parity and effective-engine resolution cannot both
be unconditional.** Resolving the plan from the effective engine CHANGES
those two rows from portrait to wide -- which is the bug fix, and therefore
a behaviour change that a "no exceptions" HEAD fixture would forbid. The
fixture must be split:

1. **Unforced byte-parity** -- every engine whose picked == effective for all
   three roles. Exact, no exceptions, this is the safety net.
2. **Named corrections** -- the four HuMo rows above, recorded individually
   with their before/after dims, gated on the operator's eyeball because
   minted still dimensions change on a shipped lane.

A migration that hides case 2 inside case 1 is a behaviour change wearing a
refactor's coat; a migration that skips case 2 ships the bug.
