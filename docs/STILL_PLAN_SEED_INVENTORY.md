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
