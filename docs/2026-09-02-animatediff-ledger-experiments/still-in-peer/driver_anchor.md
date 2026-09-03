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

## 6. r1 fold (Fable 5.1 cold read + Antigravity `agy` Gemini 3.8 Flash (High); judgment in `kibitz-runs/2026-09-02-still-in-peer/r1/judgment.md`)

The two reviews split on the still source. Antigravity: reverse D2 to the dispatcher still
(identity continuity, unity with the episode's portraits; "the 4060 justification contradicts
D1"). Fable: keep the in-graph plate (on an all-Ghost policy the dispatcher mints nothing, so
"reuse the still" is a new image phase; at denoise 0.75 SD1.5 repaints the medium and an init
survives as layout, palette, lighting). What decided it is a document neither the anchor nor
the 09-02 judgment had folded: **`docs/SPEC_haunted_image_to_video.md` (2026-08-30) already
took this lane through r1 (Antigravity) and r2 (Codex)** and closed on two findings, both
re-confirmed at the file today:

1. **Route A -- a still VAE-encoded and repeated as the init latent -- plausibly fails BY
   CONSTRUCTION**: identical cross-frame keys in the motion module's temporal attention
   suppress the trajectory while high-frequency texture boils. So Route A is a PROBE, not a
   build: denoise 0.35 / 0.50 / 0.65 / 0.80, disqualified if <= 0.50 shows no macro-motion with
   texture boil, or >= 0.65 loses the still within 2-3 frames. The mechanism is the same
   whichever still is fed, which makes the still SOURCE a secondary question for the probe.
2. **Upstream recommends the animated image come from the SAME SD1.5 model used for
   animation**; a Z-Image still into a haunted SD1.5 checkpoint invites cross-model drift
   (spec 11.5). This is the in-family plate's strongest reason, and it replaces the 4060
   reasoning the anchor gave.

The 08-30 spec's own recommendation ("measure the image phase on the 4060 first, qualify
Advanced-ControlNet, build SparseCtrl") is superseded by the 09-02 campaign order, which
deferred SparseCtrl (E7: a new pack pin, 1.99 GB, 323 MB of overnight headroom) and put the
still-in probe here. Its Route A probe grid and disqualification rule are adopted verbatim.
CLAUDE.md 0A forbids a standalone stock-node probe graph, so the probe VEHICLE is the lab
peer itself rendering canonical episodes with a denoise knob, published to `otr/obs/`.

## 7. Revised decisions after r1 (what r2 plans the code for)

**D1 (id, receipt, lab status) -- unchanged.** `animatediff15_v3_stillin_lab_video`,
recipe `animatediff_sd15_v3_haunted_stillin_e1_512x288_lab_v1`, `status: lab`, 5080 only,
never a default, never a tombstoned id (`public_engines.py:398-402`).

**D2 (in-family plate) -- STANDS, reasoning replaced.** The plate is minted in-graph from the
same checkpoint because (a) upstream says the animated image should come from the animating
model, (b) it needs no image phase (on 8 GB the image role is a native abort,
PBUG-20260829-03; on the 5080 it is a cost the probe does not need to pay), and (c) it is the
cheapest vehicle for the Route A probe, whose outcome is the same for either still source.
`accepts_still` stays `False`, `required_inputs` `('text_prompt',)`, `still_plan` `()`,
`subject_ownership` `'prompt'`, `prompt_profile` stays `GHOST_PROMPT_PROFILE` (or ShotLock
stamps no `ghost_prompt` and render falls to v1, `otr_shot_lock.py:2137, 2246-2247`); the id
joins `_ANIMATEDIFF_ENGINES` (`otr_provision.py:1516-1522`) so it stays in the no-still set.
**E1b (dispatcher still, `accepts_still=True`, the 08-30 spec's contract corrected to
`required_inputs=('text_prompt','init_image')` and a truthful non-empty `still_plan`) is the
SECOND engine id, opened only if Route A survives the probe.**

**D3 (plate prompt) -- replaced by the plate-prompt composer, modelled on
`_compose_background_plate_prompt` (`otr_meta_brief_image_prompt.py:1839-1865`).** Protected
head, in order: the FULL `positive_tail` (never `compact_style_cue` / `prefix_style_cue`,
`_otr_visual_styles.py:632-685` -- that IS the defect), `plate_look`, `_read_setting` top-2
(`:40-49`), `get_era_tail(profile="still")` (`_otr_story_brief_helpers.py:321-341`, capped
120 chars). Droppable in fixed order: `era_tail`, `image_grade_tail`, atmosphere terms, the
second setting term. NEVER: `scene_instruction_look` (an instruction sentence),
`broadcast_tail`, any `motion_registers` entry (camera words; storybook's carry "gently" --
a damping word), `open_subjects` / `announcer_*` / `portrait_*` (subject, identity), the leaf,
the mode law, `NO_TEXT_CLAUSE`. Negative: the request's own composed Ghost negative
(`ghost_signal_prompt.py:622-651`), verbatim. Budget: COUNTED with `measure_clip_tokens`
(`ghost_signal_author.py:1234-1249`), `GhostBudgetError` if the protected head exceeds 69,
pinned by a test over all nine packs plus a synthetic long `visual_storybased`. **The
composition lives in `render_driver`'s Ghost branch (where `_vstyle` and ledger meta exist,
`:2881-2950`) as a declared optional `VideoRequest` field `plate_prompt`** -- the engine has
no ledger, `sampler_inputs_for(request)` is pure over the request, and `observability` is
never conditioning (`eng_ghost_signal_official.py:144-149`); this is the schema design item
that file names, taken deliberately.

**D4 (mode routing) -- DROPPED: one subject-free scene plate per beat, plan untouched.**
`mode` is planned by ShotLock and reused on replay; an engine-side override would make the
rendered prompt differ from the planned row and break the same-prompt A/B. The plate carries
WORLD and MEDIUM only; the figure keeps coming from motif + leaf + law exactly as today
(`ghost_signal_prompt.py:863-867`), so Antigravity's "random face per beat" cannot occur --
there is no face in the plate. Figure plates are knob two. GO_FORWARD_PLAN 1.14's "cycle
frozen to figure" is corrected to this in the same change.

**D5 (E1 mechanics) -- latent-direct, stock only, zero new node classes.** STAGE 2 encodes
the plate prompt as a third CLIPTextEncode before the CLIP is released. STAGE 3a (a bounded
`run_graph`): `plate_latent = EmptyLatentImage{512x288, 1}` -> `plate_sampler = KSampler{model:
Wire('base_model'), seed: request_seed, steps 20, cfg 8.0, euler, normal, denoise 1.0,
plate_cond, negative_cond}` -> the LATENT object returned in-process. Python:
`init = samples.repeat(U, 1, 1, 1)` with `RepeatLatentBatch`'s `batch_index` rule
(`ComfyUI/nodes.py:1314-1330`), U = `source_request`; no cap, fixed topology. STAGE 3b: the
shipping sampler graph with `latent_image` = the repeated latent as an `external_results`
entry and `denoise = E1_DENOISE`. STAGE 4: decode the beat as today; decode the PLATE once,
off the critical path, for the PNG. No `VAEEncode`, no `RepeatLatentBatch`, no
`VHS_DuplicateLatents` (installed, unprovisioned), no `ADE_UpscaleAndVAEEncode`. The seven-class
object_info capture stands; the peer gets its own instance-count pin (the haunted lane pins 8,
`tests/test_ghost_signal_haunted.py:165-170`).

**D6 (seed) -- the plate uses `request_seed` itself.** Nothing new on the request; the plate is
a pure function of (checkpoint, plate prompt, seed, canvas, plate steps/cfg/sampler).

**D7 (record and reuse).** The plate PNG is written to
`<episode>/stills/ghost_plates/<shot_id>_<sha8 of plate INPUTS>.png`; the replay bundle
already freezes `stills/` whole, so a replay REUSES a present sha-named plate (encoding it
instead of re-sampling) and the A/A is exact rather than exact-up-to-GPU-noise. Never a
fabricated dispatcher `images[]` row (those carry `content_hash` / `pool_path` / `provenance`
and node 91 verifies them). Verify at build that node 91 tolerates the unlisted file.

**D8 (receipt) -- inputs only in the causal hash.** `sampler_inputs_for` returns the parent's
dict with `latent: 'ghost_plate_init'`, `init_image: None` (kept -- `accepts_still` is False),
`denoise: E1_DENOISE` (overriding the parent key), `plate_prompt` (full text), `plate_seed`,
`plate_steps / cfg / sampler / scheduler`, `plate_adapter_strength: 0.0`, `plate_canvas`,
`init_repeat_method: 'torch_repeat'`, `init_repeat_count: U`. NEVER the plate's rendered
sha, a path, a timestamp or minted/reused -- an OUTPUT in the causal hash makes the A/A false
by construction the first time SD1.5 is not bit-stable. `plate_sha256` rides beside the hash
like `vram_peak_mb` and in `CanonicalClip.qc` (`schemas.py:244`); no schema key on the clip.
`shot_cache_identity` gains plate-inputs sha8 + denoise. `E1_DENOISE` is a class attribute
with an env override read inside a method through the guarded parser (like `lora_strength`,
`eng_ghost_signal_official.py:158-174`), stamped on the receipt.

**D9 (the proof) -- the 08-30 probe, run through the canonical graph.**
1. G8 solo smoke (one beat): canvas probed, frame count exact, silence probed, VRAM peak
   receipted; split STAGE 3a/3b further if the peak exceeds 14.0 GB (Antigravity S2).
2. Per style (anime, storybook_engraving, paper_origami): one canonical episode rendered on the
   lab profile at denoise 0.65 becomes the frozen source; replays: peer x2 at 0.65 (its OWN
   null -- the plate adds sampler noise), peer x1 each at 0.35 / 0.50 / 0.80, and the SAME
   bundle rendered on the SHIPPING lane via the replay engine override (D11) as the baseline.
   Every leg publishes to `otr/obs/`, clean runner, no harness title; `otr_verify_replay.py`
   on every pair.
3. Measurement beside the eye: per-beat mean inter-frame difference (motion energy) for
   baseline, each denoise and both nulls, printed with a triptych card (plate | baseline frame
   0 | peer frame 0). E1 helped only if the operator's blinded style call rises while motion
   energy stays inside the null band; below the band is damping.
4. Stop rules, in order: the plate PNGs do not read as the style (SD1.5 base with the full pack
   language is E1's ceiling; the next arm is E11, per-style checkpoints); no denoise both
   moves and keeps the plate (the 08-30 disqualification; go to the E2 probe); peak over
   14.5 GB (a 64x36 plate cannot do that; it would mean a patcher held twice).

**D10 (not in this item) -- unchanged**, plus: no figure plates, no E1b, no ADE upgrade, no
Advanced-ControlNet.

**D11 (NEW, on the instrument): a replay-time engine override.** The reused plan's shot rows
carry `engine_id`, so a replay renders on the FROZEN engine and a same-ledger cross-engine A/B
is impossible today. `otr_canonical_api_run.py --replay-engine <id>` (plus the writer's
`replay_from` path) re-stamps `engine_id` on every reused shot row for the roles the
override names, records `meta.replay_engine_override`, and leaves prompts, seeds, beats and
the frozen audio untouched. Default (no flag) stays the pure A/A. r2 designs where the
re-stamp lives (ShotLock's replay reuse branch is the natural seat) and how ShotLock's
per-engine prompt policy is re-applied without re-authoring.

## 8. What r2 (coding plan, Codex) must settle

* The `VideoRequest.plate_prompt` field: optional str, default "", composed only for engines
  that declare `wants_plate_prompt = True`; where in `render_driver`'s Ghost branch; the
  budget test's fixture set.
* STAGE 3a/3b split: `run_graph` return shape for the LATENT, the `batch_index` rule, patcher
  ownership across the two calls (the base MODEL patcher is registered in `prepare`; the plate
  sampler must not detach it).
* D11's seat and its receipt; whether the override is per role or whole-plan.
* The instance-count pin and the `_node_candidates` map (no new classes; the map is the
  parent's).
* Tests: plate-prompt budget over ten packs; plate-prompt word bans (no name / camera /
  lettering / damping word); `sampler_inputs` pins; cache identity moves with plate inputs;
  receipts test "plate inputs change the sha, plate output does not".

## 9. r2 fold (Codex gpt-5.6-sol, coding plan; judgment in `kibitz-runs/2026-09-02-still-in-peer/r2/judgment.md`) -- ten must-fixes, all grounded, all taken (two by simplification)

* `status: lab` is not a value the schema accepts (`capability_profiles.py:64`, `_DECL_KEYS`
  `:443-455`): the profile is `draft`; the CAPABILITIES row carries no status; "lab" is in the id.
* Fresh minting and PNG reuse cannot share a seven-class map: the first build RE-MINTS every
  time and PROVES stability by equal plate hashes across the A/A instead of reusing the file.
  Reuse is the next knob, with its own encoder branch, only if the hashes ever disagree.
* The engine cannot locate the episode from a request: a NON-causal `VideoRequest.plate_path`
  is filled by `build_request_from_shot` (which has the ledger); the engine never reads the
  ledger singleton.
* No transport exists for a replay engine override, and re-stamping only `shot.engine_id`
  would trip the render boundary (`roles_effective` `render_driver.py:5158`; the coverage
  contract re-derivation `:5355-5362`): see D11 below, rewritten.
* `run_graph` returns the terminal tuple and the lane reads `sampled[0]` as the LATENT dict;
  the live `RepeatLatentBatch` (`ComfyUI/nodes.py:1317-1330`) is the repeat contract to copy in
  Python (dict copy, `samples` repeated along batch, `noise_mask` and `batch_index` handled).
* `plate_sha256` was not durable anywhere: it now rides `CanonicalClip.qc`, the receipt's
  non-causal block, and the verifier.
* `session_identity` has no request and BeatSession refuses drift mid-beat
  (`beat_session.py:280-290`): no override; plate identity lives in `shot_cache_identity`.
* The denoise env, the plate identity object, positive execution evidence
  (`audit_node_ids`), the existing `otr_ltx_mad.py::mad_of`, cleanup ownership and the
  cold-import promise are all specified below.

## 10. The coding contract as revised by r2 (what r3 wires and r4 converges on)

**Engine** `nodes/_otr_video_engines/eng_ghost_signal_stillin_lab.py`,
`GhostSignalV3StillInLabEngine(GhostSignalV3HauntedEngine)`:
* `name = "animatediff15_v3_stillin_lab_video"`, `recipe_receipt_id =
  "animatediff_sd15_v3_haunted_stillin_e1_512x288_lab_v1"`, `wants_plate_prompt = True`,
  `E1_DENOISE_DEFAULT = 0.65` (the probe's centre), `PLATE_STEPS/CFG/SAMPLER/SCHEDULER` = the
  lane's own cells, `PLATE_ADAPTER_STRENGTH = 0.0`; `accepts_still`, `required_inputs`,
  `still_plan`, `subject_ownership`, `prompt_profile`, `frame_contract`, `_node_candidates`
  INHERITED (seven classes, unchanged).
* `assert_usable` (super first) resolves `OTR_STILLIN_LAB_DENOISE` once: finite float in
  [0, 1] or a NAMED `EngineUnusable`; refuses a request whose `plate_prompt` is blank (the
  plate is the lane's whole point) with a named error.
* `plate_identity(request, denoise)` -> `(dict, sha256)`: canonical sorted JSON of
  {checkpoint digest, plate positive, plate negative, seed, steps, cfg, sampler, scheduler,
  canvas, plate adapter strength}; used by the filename (`<sanitised shot_id>_<sha16>.png`),
  by `shot_cache_identity` (which also folds the resolved denoise) and by `sampler_inputs_for`.
* `render_clip`: STAGE 2 encodes positive, negative AND the plate prompt before the CLIP is
  released. STAGE 3a (bounded `run_graph`, `audit_node_ids` = plate sampler): EmptyLatentImage
  {512x288, 1} -> KSampler{`Wire('base_model')`, request_seed, plate cells, denoise 1.0} ->
  LATENT. Python: assert batch 1; copy the dict; `samples.repeat(U,1,1,1)`; replicate
  `noise_mask` / `batch_index` by the live rule; U = `source_request`. STAGE 3b: the parent's
  sampler graph with `latent_image` = the repeated latent via `external_results` and
  `denoise = resolved`. STAGE 4: the beat decode as today; then the PLATE decode (one frame,
  `audit_node_ids`) -> PNG to `request.plate_path` via temp sibling + `os.replace`;
  `plate_sha256` = sha256 of the PNG bytes. `finally`: clear plate cond / plate latent /
  repeated latent / decoded plate; reclaim after 3a and after the plate decode. All PIL /
  numpy / torch imports inside methods.
* `sampler_inputs_for`: parent's dict + `latent: "ghost_plate_init"`, `denoise`, `plate_prompt`,
  `plate_negative`, `plate_seed`, `plate_steps`, `plate_cfg`, `plate_sampler`, `plate_scheduler`,
  `plate_canvas`, `plate_adapter_strength`, `init_repeat_method: "torch_repeat"`,
  `init_repeat_count`; `init_image` stays `None`. NEVER an output hash.
* `canonicalize`: parent's clip + `qc = {"plate_sha256", "plate_name", "plate_source":
  "minted", "plate_identity_sha256", "graph_exec": [the two audit records]}`.

**Request and driver** (`schemas.py`, `render_driver.py`): `VideoRequest` gains two optional
fields, `plate_prompt: str = ""` (CAUSAL through `sampler_inputs`) and `plate_path: str = ""`
(non-causal; excluded from `_RECEIPT_CAUSAL_KEYS` by construction since receipts hash
`sampler_inputs`, not the request). `build_request_from_shot`, in the Ghost branch where
`_vstyle` is resolved (`:2118-2123, 2878-2882`), fills both only when the engine declares
`wants_plate_prompt`; the plate-prompt composer is a pure function
`compose_plate_prompt(vstyle, ledger_meta) -> (positive, negative)` beside
`_compose_background_plate_prompt`'s pattern, counted with `measure_clip_tokens`, the
protected head raising `GhostBudgetError` above 69, the drop order of D3. `plate_path` =
`<episode>/stills/ghost_plates/` through `_otr_paths` with the ledger's episode id.

**Receipt** (`render_driver.build_actual_receipt`): non-causal `plate_sha256`, `plate_name`,
`plate_source`, projected from `clip["qc"]`. **Verifier** (`otr_verify_replay.py`): when a
trace row carries `plate_sha256`, the A/A check requires the two replays' rows to carry
non-empty EQUAL plate hashes, and, given an episode dir, the named file's bytes must hash to
the receipt.

**D11 rewritten -- the replay engine override travels in the bundle.**
`scripts/otr_freeze_replay_bundle.py --derive-engine <id> <bundle>` writes a sibling bundle
`<bundle>__engine_<id>` (same files, same hashes, a new manifest with `engine_override` and
`derived_from`), immutable like any bundle. `import_replay_bundle` reads `engine_override`,
validates the id is a registered Ghost sibling of the frozen plan's engine (equal family,
roles, `prompt_profile`, `frame_contract`), and stamps `meta.replay_engine_override`.
ShotLock's replay reuse branch, when the meta carries it, rewrites the WHOLE plan
atomically: `roles_effective`, every shot's `engine_id` and family, every execution group,
and re-derives each shot's coverage contract through the same function the render boundary
uses (`render_driver.py:5355-5362`), refusing (named) on any mismatch. No new widget, no
canonical edit, no whitelist change; every receipt names the override through
`meta.replay_from`. Default (no override) is the pure A/A, byte for byte as today.

**Registration surface**: CAPABILITIES row (copy of the haunted row; no status key); guarded
import in `_otr_video_engines/__init__.py`; `_ANIMATEDIFF_ENGINES` gains the id
(`otr_provision.py:1516-1522`); `scripts/build_video_evidence_manifest.py` sentence +
regenerate; `docs/ENGINE_MATRIX.md` regenerate; `config/profiles/otr_stillin_lab_5080.json`
(`status: draft`, three video roles on the peer) + `build_variants --all/--check`;
`tests/fixtures/still_plan_head_parity.json` regenerated if the roster test demands it.

**Tests** (`tests/test_ghost_signal_stillin_lab.py` + edits): canvas pin (G2.2); the seven
candidates are the parent's; 11 render-time node instances on the fresh path; plate-prompt
budget over `list_style_ids()` + one embedded `visual_storybased` (both the 69 target and the
77 window); plate-prompt bans (no name / camera / lettering / damping word; never
`compact_style_cue`); `sampler_inputs` pins (`latent == "ghost_plate_init"`, `init_image is
None`, no output hash key); `shot_cache_identity` moves with plate inputs and denoise;
`OTR_STILLIN_LAB_DENOISE` invalid -> named `EngineUnusable`; receipts: "plate inputs change
`actual_request_sha`, plate output does not"; `qc` propagation; the verifier's plate-hash
rule; the bundle derivation + import + ShotLock whole-plan rewrite + boundary agreement; cold
import; the preflight matrix and `test_ghost_signal_peers.py` floors green.

**Probe runner** `scripts/otr_stillin_probe_report.py`: given the source and the replay
episode dirs, matches trace rows to clips by `shot_id`, computes `mad_of` per clip
(`scripts/otr_ltx_mad.py:26`), prints the null band (interval between the two A/A nulls'
per-beat MAD widened by 10% of its width), flags below-band beats as damping, writes one
triptych card per beat (plate | baseline frame 0 | peer frame 0), and exits non-zero on any
missing row, clip or plate.
