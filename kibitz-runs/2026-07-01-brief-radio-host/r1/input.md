# Brief-Driven Humanoid Radio-Host -- Plan (2026-07-01)

Operator ask: the radio-host imagery does NOT reflect the true nature of each story.
Make EVERY radio brief-driven -- a humanoid radio-host (an expressive human-like face
embedded in a radio/console body) whose radio FORM + face + palette come from the
episode meta brief, and that HuMo can animate (the radio hosts / sings in-world).
"Not a baby -- choose a face and radio style based on the meta brief of the story."

## 1. Problem (grounded)

The radio imagery is HARDCODED 1940s regardless of story:
- `ANNOUNCER_PORTRAIT_ANCHOR` = "vintage 1940s radio announcer at a large chrome ribbon
  microphone..." (`nodes/otr_meta_brief_image_prompt.py:125`).
- `_mesh_fodder_subject` music_visual = "a vintage 1940s tabletop radio receiver, wood
  cabinet..." (`:587`); announcer = "a vintage 1940s radio announcer in a tailored suit
  and tie" (`:583`).
So the space-station-sabotage episode (`pending_20260701_101425`: style
`automated_space_docking`, palette cool_blue/harsh_white/metallic/grimy_grunge, "saboteurs
... graveyard shift") would still get a 1940s wooden-radio host. That is the mismatch.

PROOF (3 Flux renders, `docs/2026-07-01-overnight/ComfyUI_0000{1,2,3}.png`): generic warm-wood
baby radio (v1) vs the same "radio-host" idea driven by THIS episode's real brief -- a wary
adult face in a grimy space-station comms console (v2/v3). The v3 speaker-port console is the
target look. The difference is exactly what the operator wants.

## 2. Goal

One BRIEF-DRIVEN radio-host still builder feeding every radio surface:
- Radio FORM (1940s bakelite / mid-century / sci-fi comms console / etc.) derived from the
  brief's era + `meta.style` + `meta.story_brief_terms.setting`/`lighting`.
- FACE = expressive human-like face (adult by default, NEVER baby), mood from
  `meta.atmosphere_line` / `meta.music_mood_terms`. A REAL face so HuMo can animate it.
- Palette/lighting tail from `meta.visual_palette`.
The face-in-radio still is HuMo-animatable -> the radio hosts / sings, in the story's world.

## 3. Design

1. **New builder** `build_radio_host_prompt(meta, *, aspect)` in
   `otr_meta_brief_image_prompt.py`: template "humanoid radio-host: an expressive
   human-like face embedded in <RADIO_FORM>, <FACE_MOOD>, <PALETTE_TAIL>". RADIO_FORM
   DERIVED from the brief -- REUSE the existing era machinery (`get_era_tail` /
   `finish_visual_prompt` from `_otr_story_brief_helpers`) rather than a new hand-maintained
   era->form map (no-drift, same principle as the E4 label fix). Default face age = adult;
   hard-negative "baby".
2. **Replace the fixed anchors** with the builder:
   - `ANNOUNCER_PORTRAIT_ANCHOR` -> `build_radio_host_prompt(meta)` (keep the radio-grounding
     gate `:764` -- the prompt must still read as a radio, now era-appropriate).
   - `_mesh_fodder_subject` music_visual/announcer -> brief-driven radio form (keeps the
     single canonical `MESH_RADIO_HOST_SUBJECT_ID` `:225` for identity continuity).
   - (optional) the `ltx_audio_in` radio-console motion prompt -> same brief-driven form.
3. **HuMo-hosts (operator's real goal) -- ONLY HuMo gets a FACE.** The
   humanoid-radio-WITH-A-FACE still is HuMo-specific (HuMo is the face-animation engine, so
   only it needs a face-bearing still). Route announcer/music beats to HuMo animating the
   brief-driven radio-FACE still (the v3 look), as an ALTERNATIVE to today's radio-is-host ->
   `ltx_audio_in` animated-console redirect (`render_driver._enforce_radio_is_host:821`).
   Operator toggle -- KEEP the animated radio selectable ("may bring it back"). When on, the
   guard redirects HuMo-on-bookends to a HuMo render of the FACE-radio still instead of
   swapping engines. Operator 2026-07-01: "only humo should have a face."
4. **Per-engine radio representation (operator: only HuMo gets a face).** The BRIEF drives the
   radio's era/form/palette for EVERY engine, but the FACE is HuMo-only:
   - **HuMo** -> brief-driven humanoid radio with an expressive human FACE (v3), which HuMo
     animates (the radio hosts / sings).
   - **mesh_stage** -> brief-driven 3D radio OBJECT, NO face (keeps its faceless
     `_mesh_fodder_subject` radio; just era/form now brief-driven instead of fixed-1940s).
   - **ltx_audio_in** -> DEFERRED: keep the animated console for now; a face still is a
     later experiment to test, not this pass ("maybe ltx audio in but not yet").
   - **viz_*** -> no still at all (flux-skip fix `945707f5`).

## 4. Integration points (grounded)

- `nodes/otr_meta_brief_image_prompt.py`: `ANNOUNCER_PORTRAIT_ANCHOR:125`,
  `_mesh_fodder_subject:564`, radio-grounding gate `:764`, `_otr_story_brief_helpers`
  (`get_era_tail`/`finish_visual_prompt`).
- `nodes/_otr_video_engines/render_driver.py`: `_enforce_radio_is_host:821` (redirect
  target choice + the new HuMo-hosts toggle).
- `nodes/otr_image_gen_dispatcher.py`: mints the still per role; already gates on
  `accepts_still` (+ the force-map flux-skip fix) -- the radio-host still is minted for
  humo/mesh, skipped for viz.

## 5. Invariants

- Audio byte-identical (visual-only; `test_audio_byte_identical` GREEN).
- Determinism seed-keyed (brief/episode-keyed, like `OTR_RADIO_BOOKEND_SEED` 4242).
- No-fallback LOUD; radio-grounding gate still enforces "reads as a radio".
- NEVER baby; SFW; era-appropriate.
- UTF-8 no BOM; workflow-JSON edited in the same change as code IF node wiring changes
  (likely none -- prompt-builder + routing only).
- Keep `ltx_audio_in` animated-radio as a selectable alternative (operator).

## 6. Open questions (for kibitz)

- RADIO_FORM source: reuse `get_era_tail` (no new map) vs a tiny `style`->form lookup vs an
  LLM call from the brief? Prefer derive/reuse (no-drift).
- HuMo-hosts vs animated-radio default? (Operator leaning: experiment with the HuMo still
  now, keep the animated radio available.)
- Does the radio-grounding gate (`:655` token regex: radios/microphones/broadcast/studio)
  need widening to accept a sci-fi "comms console" as a valid radio, so era-shifted forms
  pass the gate?
- Face identity across bookends (open/inter/close): one canonical radio-host face per
  episode (seed-pinned) so the host is consistent -- mirror `MESH_RADIO_HOST_SUBJECT_ID`.

## 7. Reference renders

`docs/2026-07-01-overnight/ComfyUI_00001_.png` (v1 generic baby wooden radio -- the anti-example),
`_00002_.png` (v2 brief-driven, astronaut-helmet lean), `_00003_.png` (v3 brief-driven
space-station comms-console -- the target look). All flux1-dev-fp8, from the
`automated_space_docking` brief.
