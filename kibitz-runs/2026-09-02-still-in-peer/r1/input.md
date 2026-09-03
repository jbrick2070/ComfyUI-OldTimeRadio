# Driver anchor -- item 2, the still-in LAB PEER for the AnimateDiff v3 lane (2026-09-02)

Campaign: `docs/2026-09-02-animatediff-ledger-experiments/` (PROBLEM_STATEMENT.md, fresh-eyes/,
instrument/). Judged order: (0) the instrument -- CODED and merged 2026-09-02 (969f2578, dc4beea5);
(1) the adapter sweep -- running; **(2) this item**; (3) pack language under the 77-token budget;
(4) one timeline per shot. This anchor is the driver's code-grounded position BEFORE the panel;
the four readers' fact digest it rests on is reproduced in section 2 with file:line.

Operator rulings that bind this item, verbatim where recorded:
* "it is paramount that the prompting for this new AnimateDiff workflow must obey the visual
  style ... I think we lost the visual style." (PROBLEM_STATEMENT 5B)
* "anime is not the only target; all visual styles need to craft the episode as well when
  selected." (GO_FORWARD_PLAN 1.14, 2026-09-02 evening)
* "always remember when adding a new video pack to follow the preflight video guide" --
  `docs/VIDEO_LANE_PREFLIGHT.md` gates 1-8 + `tests/test_lane_preflight_matrix.py` BEFORE the id
  is registered or a profile names it.
* No damping words in motion prompts; flicker is a recipe problem, never a wording one; story
  quality is done; every arm is an ADDITION judged against the shipped recipe with a same-seed
  A/A null first; recipes are hard-won and never traded for VRAM or speed.

## 1. What the item is, in one paragraph

A NEW engine id beside the shipping `animatediff15_v3_haunted_video`, subclassing it, that renders
the same recipe (v3 module + adapter, 20 / 8.0 / euler / normal, 512x288 hold-2, static 16/4
pyramid, the live negative) but starts the sampler from a STILL instead of an empty latent. The
still is the lane's own **in-family plate**: one 512x288 frame minted in the same graph from the
same SD1.5 checkpoint with the adapter at 0.0, prompted with the style pack's full language and
the ledger's setting, palette and lighting, then VAE-encoded, repeated to the sampler batch and
sampled at denoise < 1.0 (E1). The plate carries the STYLE into every frame at the pixel level,
which the two-word cue cannot; the motion recipe is untouched. ADE image injection (E2) is a
later knob behind its own VRAM probe. Nothing about the shipping engine, its graph, its tests,
its receipt id or the 4060 provisioning set moves.

## 2. Code facts the design rests on (readers 2026-09-02, verified file:line)

**The registry contract and the gates a new lane owes.**
* Registration is `@register` + a unique `name` (`nodes/_otr_shared/engine_registry_base.py:141-150`);
  the pack's own gates then demand a `CAPABILITIES` row (`registry.py:273-275, 688-721`), an explicit
  `accepts_still`, a declared `still_plan` tuple (`()` valid), exactly one live menu option
  (`exact_menu_option_for`), and a row in the generated `docs/ENGINE_MATRIX.md`
  (`tests/test_lane_preflight_matrix.py:940-964`, `tests/test_engine_matrix_doc.py:43-49`).
* The preflight suite iterates `all_engine_names()` with `EXPECTED_RED` empty, so registration IS
  the gate (`test_lane_preflight_matrix.py:176-273, 383-393`). Files a new lane touches for the
  matrix to pass: `eng_<name>.py`, `registry.py` (CAPABILITIES), `_otr_video_engines/__init__.py`
  (guarded import), `scripts/build_video_evidence_manifest.py` (a >= 6-word `admission_unenforced`
  sentence, then regenerate the manifest -- never hand-edit it), `docs/ENGINE_MATRIX.md`
  (`tools/engine_matrix.py`), and a `tests/test_<lane>.py` naming the id AND its exact canvas
  (G2.2 pin, `:627-636, 711-731`). G8 (the solo smoke) is the only live gate and is prose-only
  (`docs/VIDEO_LANE_PREFLIGHT.md:234-243`).
* A sibling of an existing lane also owes the G1.3 per-artifact CLASS attributes with byte floors
  below and within 15% of the real files (`tests/test_ghost_signal_peers.py:106-127`).

**The shipping lane's seams.**
* `GhostSignalV3HauntedEngine` is the only registered Ghost engine; base classes are unregistered
  and their ids tombstoned (`eng_ghost_signal_official.py:68-175`). The graph is built INLINE in
  `render_clip` (`eng_ghost_signal.py:781-970`) from a closed `GHOST_NODE_CANDIDATES` dict of seven
  classes (`:154-162`); `ADE_ADE_OMITTED_SOCKETS` pins `sample_settings` and friends as OMITTED
  (`:166-172`); tests pin 7 resolver names, 8 executed node instances on the haunted lane, the
  omitted sockets, `sampler_inputs.latent == 'EmptyLatentImage'`, `init_image is None`, and
  `model_artifacts` == [checkpoint, motion_module, adapter] (`tests/test_ghost_signal_lane.py:226-243,
  416-520`; `tests/test_ghost_signal_haunted.py:165-172`; `tests/test_render_receipts.py:129-143`).
  => the peer is a NEW subclass that overrides `name`, `recipe_receipt_id`, `_node_candidates`,
  `render_clip`, `sampler_inputs_for`, `model_artifacts`, `session_identity`, `shot_cache_identity`;
  it never edits the shipping class.
* `_build_render_request` reads only shot/request ids, text_prompt, negative_prompt,
  timing.target_frame_count and seed_bundle.request_seed (`eng_ghost_signal.py:732-778`); the seed
  is `_seed_from_hash(render_request_hash, shot_id)` and does not move when a still, a denoise or an
  adapter strength is added (`render_driver.py:548-553, 3768-3772`) -- same-seed A/Bs stay valid.
* `GHOST_CHECKPOINT_NAME` is a module constant read inside methods (`:411, 535-536, 715, 466, 497`),
  not a class seam; the peer keeps the checkpoint, so this stays a note (promote to a class seam
  only when a per-style checkpoint arm E11 is opened).
* `prepare()` runs only the `ckpt` node and registers the base MODEL patcher; STAGE 2 encodes the
  two prompts and releases the CLIP; STAGE 3 builds context -> lora -> ADE loader -> EmptyLatentImage
  -> KSampler; STAGE 4 decodes with the external `vae` handle (`eng_ghost_signal.py:678-729,
  849-945`). `run_graph` accepts `external_results` as legal Wire sources (`wrapper_bridge.py:460-560`).
  The plain checkpoint MODEL stays usable beside the ADE-cloned one (ADE clones,
  `ADE nodes_gen1.py:85`), so a single-frame plate branch on `Wire('base_model')` is legal in the
  same STAGE 3 graph.
* No image node (LoadImage, VAEEncode, ImageScale, RepeatLatentBatch, LatentFromBatch) is in the
  lane's Phase-0 `/object_info` capture (`docs/2026-08-22-ghost-signal-object-info.json` holds the
  seven shipping classes); the peer needs its own live capture before pinning any class id -- ONE
  NAME PER ALIAS, no probing at render time.

**How a still reaches an engine today, and why the peer does not go that way first.**
* A still reaches an engine ONLY as `request['asset_refs']['init_image']` (`render_driver.py:342-380,
  4389-4410`); `render_driver` joins the beat's SCENE still when the family is image_to_video /
  static_motion or `init_image` is in `required_inputs` (`:2240-2275, 1307-1313`), and raises
  `DeferredImageGapError` with no portrait fallback when the row is missing. Whether the image
  phase MINTS anything is `engine_consumes_still` = `accepts_still` (`otr_image_gen_dispatcher.py:640-657`).
  Scene stills are 1472x832 from a foreign image family (`otr_meta_brief_image_prompt.py:471-484`).
* The lane is in `_NO_STILL_VIDEO_ENGINES` (`scripts/otr_provision.py:1546-1551`) precisely so the
  8 GB box is not gated on the ~11 GB Klein bundle (PROBLEM_STATEMENT 7; judgment 1.2). A peer that
  flips `accepts_still=True` would be pulled out of that set on the 4060 unless its id is excluded
  by hand, and would have to declare the G3.7-scoped ownership fields differently
  (`test_lane_preflight_matrix.py:1036-1038`).
* The receipt already hashes `still_sha256` from `asset_refs.init_image` content
  (`render_driver.py:4095, 4132`); an engine-minted plate is NOT on `asset_refs`, so the peer
  must put the plate's identity into `sampler_inputs` itself (section 4.5).

**Conditioning mechanics available on this install (ComfyUI 0.34.3, ADE 1.6.0).**
* Stock init-latent path: `VAEEncode(pixels, vae)`; `RepeatLatentBatch(amount <= 64)`
  (`ComfyUI/nodes.py:1304-1330`); `LatentFromBatch(samples, batch_index, length)`; `KSampler.denoise`
  0-1. `ADE_UpscaleAndVAEEncode(image, vae, latent_size, scale_method, crop)` resizes + encodes in
  one node (`animatediff/nodes_animatelcmi2v.py:133-159`). `VHS_DuplicateLatents` exists
  (`comfyui-videohelpersuite`) but VHS is NOT in `otr_provision.py`'s pinned set -- not a dependency
  the peer may take.
* Ghost source batch = `max(ceil(T / 2), 16)` latents with `max_frames = 0` (`eng_ghost_signal.py:
  201-211, 385-392`): a beat over 128 delivered frames needs more than one RepeatLatentBatch.
* `ADE_NoisedImageInjection` needs `ADE_AnimateDiffSamplingSettings` on the loader's
  `sample_settings` socket (the socket the shipping lane pins OMITTED) and decodes + re-encodes the
  WHOLE batch through the VAE at each injection point (`animatediff/sampling.py:645-693`) -- E2 is
  a second arm behind a VRAM probe, exactly as the judgment ordered.
* ComfyUI-Advanced-ControlNet is not installed; SparseCtrl (E7) stays deferred.

**Stills, layout, style packs.**
* Episode stills live at `<output>/otr/episodes/<episode_id>/stills/{object_id}_{hash12}.png`,
  with a `stills_manifest.json` beside them (`nodes/_otr_shared/portrait_ledger.py:63-108`,
  `otr_image_gen_dispatcher.py:2099-2139`). The replay bundle freezes `stills/` and `portraits/`
  whole (`scripts/otr_freeze_replay_bundle.py`, FROZEN_DIRS).
* A planned shot carries `ghost_prompt` (with `mode` in figure / object / signal and the bookend),
  `beat_id`, `role`, `char_id`, `subject_sigil` -- no still path (`otr_shot_lock.py:1662-1690`).
  `GHOST_CHARACTER_CYCLE = ('figure','object','figure','signal')` (`ghost_signal_author.py:101,
  472-505`).
* Packs: nine files under `nodes/visual_styles/` (+ the run-time `visual_storybased`); prompt
  LANGUAGE only -- `positive_tail`, `image_grade_tail`, `broadcast_tail`, `era_tail`,
  `negative_tail`, eleven look strings incl. `plate_look`, `scene_instruction_look`,
  `portrait_look`, four dicts (`_otr_visual_styles.py:559-618`; PROBLEM_STATEMENT 154-162).
  The video cue is `compact_style_cue()` = two words (`_otr_visual_styles.py:632-653`); the leaf
  validator bans medium/camera/quality words -- "style is Python's job on this lane"
  (`ghost_signal_author.py:316-330`); the finalizer enforces one 77-token window and 320 chars on
  BOTH prompts (`:1348-1367`).

## 3. Decisions (the driver's position; the panel pressure-tests these)

**D1. A new lane, id `animatediff15_v3_stillin_lab_video`, menu label "AnimateDiff v3 still-in lab
(512x288)", recipe receipt `animatediff_sd15_v3_haunted_stillin_e1_512x288_lab_v1`.** Subclass of
`GhostSignalV3HauntedEngine`; every G1.3 constant a class attribute it inherits or overrides;
`status: lab` in its profile; 5080 only until measured. The word "lab" is in the id on purpose:
it is never a default, never in a shipping profile, and its receipt id can never be mistaken for
the shipping one.

**D2. The plate is minted IN-GRAPH by the peer, not taken from the image phase.** Consequences,
all deliberate: `accepts_still` stays `False`, `required_inputs` stays `('text_prompt',)`,
`still_plan` stays `()`, `subject_ownership` stays `'prompt'`, the id stays OUT of any dispatcher
minting and is added to `_NO_STILL_VIDEO_ENGINES` alongside its parent so the 4060 is never gated
on an image bundle by it. The plate is a recipe cell of the VIDEO lane, hashed by its receipt, not
an asset of the image phase. The dispatcher-fed still (portrait-derived `scene_character` at
1472x832 through `resolve_aspect_transform`, `motion_common.py:135-166`) is arm E1b, a LATER knob
on the same peer, opened only after E1a is judged; it is the one that would need
`accepts_still=True` and its own provisioning exclusion, and it is not built now.

**D3. Plate prompt = the pack's language, the ledger's world, never the leaf.** Positive:
`<positive_tail>. <plate_look or scene_instruction_look>. <setting phrase from the ledger:
place, era, palette, lighting>. <mode subject>` measured by `measure_clip_tokens` to <= 77 with
the pack clauses first (they are the point) and the ledger phrase trimmed last; negative = the
composed lane negative (`LANE_HYGIENE_NEGATIVE` + pack `effective_negative`). NO character name,
no dialogue, no camera words, no lettering: the plate owns MEDIUM and WORLD; motion and framing
stay with the video prompt (mode law) exactly as today. The plate branch runs with the adapter at
0.0 (plain checkpoint MODEL): the plate is the pristine image-domain picture of the style, the
adapter then haunts the motion on top of it.

**D4. Mode routing by `ghost_prompt.mode`, not a frozen cycle.** The plan's one-liner ("cycle
frozen to figure") predates the r2/r3 folds that preferred routing; the plate makes routing free:
figure -> a figure plate (a person of the character's described build in the world, no name),
object -> the object with no person, signal -> the environment with no person, bookend ->
`scene_open`. Freezing the cycle would change the shipping author's schedule for the peer only and
hide a variable; routing keeps the schedule identical between the two lanes so the A/B isolates
the still. GO_FORWARD_PLAN 1.14's one-liner is corrected in this change.

**D5. E1 mechanics, stock nodes only.** STAGE 3 gains, before the sampler:
`plate_latent = EmptyLatentImage{512x288, 1}` -> `plate_sampler = KSampler{model: Wire('base_model'),
seed: plate_seed, steps 20, cfg 8.0, euler, normal, denoise 1.0, positive: plate_positive, negative:
negative}` -> `plate_decode = VAEDecode{vae}` (the plate PNG is written to
`<episode>/stills/plate_<shot_id>_<sha12>.png` for the record, section D7) -> `plate_encode =
VAEEncode{pixels, vae}` -> `repeat = RepeatLatentBatch{amount: min(U, 64)}` (+ a second
`RepeatLatentBatch` and `LatentFromBatch{0, U}` when `U > 64`; U = the lane's `source_request`) ->
`sampler.latent_image`, with `sampler.denoise = 0.75` (class attribute `E1_DENOISE`, sweep 0.6 /
0.75 / 0.9 later; 0.9 only if motion is damped). The plate prompt is a THIRD CLIPTextEncode in
STAGE 2 before the CLIP is released. Node candidates: the seven shipping classes + `VAEEncode`,
`RepeatLatentBatch`, `LatentFromBatch` (one name per alias, pinned from a fresh live capture).
Resize policy: none needed -- the plate is minted at the canvas. `ADE_UpscaleAndVAEEncode` and
`VHS_DuplicateLatents` are rejected: the first is an ADE-private helper for a foreign-size still
(E1b's problem, not E1a's), the second is an unprovisioned dependency.

**D6. The seed.** `plate_seed = (request_seed * 1000003 + 0x5EED) & 0x7FFFFFFF` -- derived, so a
same-ledger replay reproduces the plate by construction and a changed request seed changes it; the
video sampler keeps `request_seed` untouched so the shipping lane and the peer share the video
seed on the same ledger (the A/B isolates the init latent).

**D7. Reproducibility and the record.** The plate is deterministic from (checkpoint, plate prompt,
plate seed, canvas, steps/cfg/sampler) on one box, and the peer ALSO writes it to
`<episode>/stills/plate_<shot_id>_<sha12>.png` and records `plate_sha256` on the clip -- so the
replay bundle carries the exact frame, `otr_verify_replay.py` can compare plate hashes across the
A/A, and a bitwise-unstable GPU kernel shows up as a plate-hash disagreement rather than as an
unexplained video difference. The peer does not READ the file back on a replay in the first build
(the plate is re-minted; the hash is the check) -- reading it back is a second knob if the null
proves noisy.

**D8. The receipt (the r4 pin, applied to the peer).** `sampler_inputs_for` returns the parent's
dict plus `latent: 'VAEEncode'`, `init_image: <plate_sha256>`, `denoise: E1_DENOISE`, `plate_seed`,
`plate_prompt`, `plate_negative`, `plate_steps/cfg/sampler/scheduler`, `plate_adapter_strength: 0.0`,
`repeat_amount: U`; `model_artifacts` is the parent's (no new weight). Two A/A replays agree by
construction (every field derives from the ledger + seed); a changed plate prompt, denoise, seed
or canvas disagrees. `session_identity` / `shot_cache_identity` gain the plate sha so a cache hit
can never hand a still-free render to the peer.

**D9. The proof.** All legs on the canonical graph, published to `otr/obs/`, clean runner, no
harness title (the 17:45 lesson), engine selected per role by a `lab` profile
`otr_stillin_lab_5080.json` (copy of the haunted profile with the three video roles on the peer,
`status: lab`, variants regenerated). Sequence:
1. G8 solo smoke on one beat: canvas probed 512x288, frame count exact, silence probed, VRAM peak
   receipted (the plate branch adds one 512x288 single-frame sample and one VAE encode; expected
   well under the lane's 13.3-14.2 GB peak -- MEASURED, not assumed).
2. Same-ledger A/B per style via the replay mode: freeze one episode rendered on the SHIPPING lane
   (anime), replay it on the peer; repeat for `storybook_engraving` and `paper_origami` (three
   styles, two non-anime). Each pair carries its A/A null (the peer replayed twice).
3. Verdict, in the operator's words: "looks like <the selected style>", stills and video the same
   show, name / gender / portrait / voice agreeing -- judged as radio drama, by his eye, with the
   render trace to attribute any difference.
Stop the item if the null is not a null (plate hashes or seeds differ across two replays), or if
E1 at 0.75 damps motion visibly on all three styles (then 0.9 is tried once; a second failure ends
E1 and opens E2's probe).

**D10. Not in this item.** E2 (image injection) and its VRAM probe; E1b (dispatcher-fed still);
E7 (SparseCtrl, pack not installed); per-style checkpoints; any change to the shipping engine,
its graph, tests, receipt, or the 4060 provisioning set; any prompt change on the video prompt
(that is item 3).

## 4. The build list (what one window codes after r4)

1. `nodes/_otr_video_engines/eng_ghost_signal_stillin_lab.py`: `GhostSignalV3StillInLabEngine(
   GhostSignalV3HauntedEngine)` with `name`, `recipe_receipt_id`, `E1_DENOISE = 0.75`,
   `PLATE_*` class constants, `_node_candidates()` (ten names), `render_clip()` (STAGE 2 with the
   plate prompt, STAGE 3 with the plate branch), `_plate_prompt(request, vstyle, ledger_world)`,
   `_plate_seed(request_seed)`, `sampler_inputs_for`, `session_identity`, `shot_cache_identity`,
   `canonicalize` adding `plate_sha256` to the clip (schemas.py: one new optional key, since
   `CanonicalClip` is `extra='forbid'`).
2. `registry.py` CAPABILITIES row (copy of the haunted row, `status: lab`); `__init__.py` guarded
   import; `scripts/otr_provision.py` `_ANIMATEDIFF_ENGINES` gains the id (so it stays no-still on
   the 4060); `scripts/build_video_evidence_manifest.py` sentence + regenerate;
   `tools/engine_matrix.py` regenerate `docs/ENGINE_MATRIX.md`.
3. `config/profiles/otr_stillin_lab_5080.json` + `scripts/build_variants.py --all/--check`.
4. Tests: `tests/test_ghost_signal_stillin_lab.py` (canvas pin, node candidates, exactly N node
   instances, `sampler_inputs` pins, plate-seed derivation, plate-prompt budget <= 77 on all nine
   packs, no name / camera / lettering word in any plate prompt, `session_identity` moves with the
   plate sha); `test_render_receipts.py` gains the "plate changes the sha" case; the preflight
   matrix and `test_ghost_signal_peers.py` floors run green.
5. A `docs/evidence/lane_receipts/animatediff15_v3_stillin_lab_video.md` G8 receipt after the smoke.

## 5. Open questions for the panel (r1)

* Q1. Is the in-graph plate (D2) the right FIRST build, or should E1b (dispatcher still) go first
  because it reuses the existing still spine? The driver says D2: it isolates the style variable,
  needs no image model, cannot gate the 4060, and cannot pull a foreign-family look into the lane.
* Q2. The plate prompt's ledger phrase: is "setting, era, palette, lighting" the right slice of the
  ledger, and which existing composer (the still lane's scene prompt builder in
  `otr_meta_brief_image_prompt.py`) should the peer borrow from rather than write new prose?
* Q3. `denoise 0.75` first -- or 0.6, given the adapter at 1.0 already pulls toward its domain?
* Q4. Should the plate be minted at the adapter's LIVE strength instead of 0.0 (the plate would
  already carry the haunt)? The driver says 0.0: the plate is the style's own picture; the adapter
  is a motion-domain knob and the sweep is measuring it separately.
* Q5. Mode routing (D4) versus the plan's frozen cycle -- any reason routing hides a variable?
* Q6. Anything in `render_driver`'s coverage / jump-still / segment path that assumes an
  EmptyLatentImage lane and would break a subclass that repeats latents (multi-segment beats,
  `U > 64`)?
