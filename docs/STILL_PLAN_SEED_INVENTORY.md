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

**TRAP 3 -- `ltx_video` requires NO scene still today.** It declares
`accepts_still = True` (so the producer enumerates one and the dispatcher
mints it) but its family is `text_to_video`, it has no `init_image` in
`required_inputs`, and it is not `provider_side` -- so the validator returns
False. The six `google_*` t2v engines DO reach True, via `provider_side`.
So "t2v engines declare `accepts_still=True`; still enumerated per beat"
covers both, while their REQUIRED column differs: `ltx_video` is
`required=False`, `google_*` is `required=True`. Transplanting them as one
row changes behaviour on the `ltx_video` lane.

**TRAP 4 -- the portrait requirement is keyed on FAMILY, in a sixth place.**
`validate_and_repair_still_spine` requires a portrait row via
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
