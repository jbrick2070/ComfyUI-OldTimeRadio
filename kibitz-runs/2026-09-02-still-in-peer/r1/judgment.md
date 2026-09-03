# r1 judgment -- still-in lab peer (2026-09-02)

Roster, r1: Fable 5.1 cold read (Cowork subagent, no sight of the anchor) + Antigravity
(`agy`, Gemini 3.8 Flash (High), `antigravity.md`). Driver: Claude (Cowork). Every claim below
was checked against the real Windows files before disposition. Codex is r2, Cursor r3, Sonnet
r4 (one seat per round, the operator's 2026-09-02 ruling).

## The fork the two reviews disagree on

Antigravity M1: reverse D2 -- take the episode's dispatcher still (`accepts_still=True`, the
portrait-anchored `scene_character`), because the in-family plate gives no identity continuity
(M3) and the 4060 argument is moot for a 5080 lab peer. Fable 1: keep the in-graph plate --
on an all-Ghost policy the dispatcher mints NOTHING (`tests/test_ghost_signal_lane.py:145-159`;
haunted profile's `z_image_turbo` rows inert), so "reuse the still" is a NEW image phase with a
Klein/Z-Image dependency; at denoise 0.75 SD1.5 repaints the medium anyway, so what survives an
init is layout, palette and lighting, which a plate supplies.

Grounding that decides it:
* `docs/SPEC_haunted_image_to_video.md` (2026-08-30, r1 Antigravity + r2 Codex, both grounded)
  already took THIS lane through an arc and closed on "do not build as specified": Route A (a
  still VAE-encoded and repeated as the init latent) plausibly fails by construction -- identical
  cross-frame keys in the temporal attention suppress motion while texture boils -- so it is a
  PROBE (denoise 0.35 / 0.50 / 0.65 / 0.80, disqualified if <= 0.50 shows no macro-motion or
  >= 0.65 loses the still in 2-3 frames), not a build. CONFIRMED at the file (sections 10-11).
* The same r2 fold records upstream's guidance: the animated image should come from the SAME
  SD1.5 model used for animation; a Z-Image still into a haunted SD1.5 checkpoint invites
  cross-model drift (section 11.5). CONFIRMED. This is the in-family plate's strongest reason,
  and neither the anchor nor Antigravity's r1 had it.
* Antigravity's "the 4060 justification contradicts D1" is CONFIRMED as written, and the
  reasoning in D2 is replaced (section 7 of the anchor): the plate is chosen because it is the
  same model (upstream), needs no image phase (PBUG-20260829-03 makes the image role a native
  abort on 8 GB; on the 5080 it is a cost, not a block), and is the cheapest vehicle for the
  Route A probe, which is the SAME mechanism whichever still is used.
* Antigravity's "base SD1.5 cannot render the styles" is its own flagged ASSUMPTION; the plate
  prompt carries the pack's FULL language where the video prompt carries two words, so the plate
  is a strict increase of style signal on this checkpoint. Fable's stop rule covers the case where
  that is still not enough: "if the plate PNGs do not read as the style, E1 cannot exceed it; the
  next arm is E11". TAKEN as the first stop rule.

DISPOSITION: D2 stands (in-family plate), reasoning corrected; E1b (dispatcher still) is the
second engine id, opened only if Route A survives the probe. Identity (Antigravity M3) is
answered by Fable 3: the first build mints SUBJECT-FREE scene plates (world + medium only); the
figure keeps coming from motif + leaf + law exactly as today, so there is no face to pop. Figure
plates are knob two.

## Claims, grounded

Antigravity
* M1 (reverse to dispatcher still) -- CONFIRMED as a coherent alternative, REJECTED for the
  first build (above). `docs/SPEC_haunted_image_to_video.md:47-87` is the Route A description,
  CONFIRMED.
* M2 (VAEDecode -> VAEEncode round trip is lossy and a Stage-3 VRAM spike) -- CONFIRMED; the
  plate KSampler already emits a LATENT. TAKEN: latent-direct (Fable 4 agrees), decode once for
  the PNG off the critical path.
* M3 (random face per beat at denoise 0.75) -- CONFIRMED for subject-bearing plates. TAKEN via
  subject-free plates (Fable 3).
* M4 (two chained RepeatLatentBatch cap at 128; U > 128 underruns; dynamic topology breaks the
  instance-count pin) -- CONFIRMED (`ComfyUI/nodes.py:1310` amount <= 64; the lane's
  `source_request` is unbounded, `eng_ghost_signal.py:385-392`). TAKEN: repeat the latent in
  Python between the bounded `run_graph` calls the lane already uses (`:917-921, 941-945`
  pass results in-process as `external_results`) -- zero new node classes, fixed topology.
  `ADE_EmptyLatentImageLarge` exists (`animatediff/nodes_extras.py:63`) but is an empty-latent
  node; not the fix.
* M5 (`CanonicalClip` pollution; `asset_hashes` / `qc` exist) -- CONFIRMED (`schemas.py:243-244`).
  TAKEN: `plate_sha256` goes in `qc` (and the receipt beside the hash, non-causal); no schema key.
* S1 (77-token ordering; the subject must have a reserved floor) -- CONFIRMED in principle;
  with subject-free plates the protected head is the pack language; the drop order and a
  COUNTED budget over all nine packs + a long `visual_storybased` is Fable 2. TAKEN.
* S2 (Stage-3 concurrency; measure at G8, split if > 14.0 GB) -- CONFIRMED as a measurement
  rule. TAKEN.
* S3 (`init_image` key ambiguity on the receipt) -- CONFIRMED. TAKEN: `init_image: None` stays;
  the plate rides its own keys (Fable 5).
* S4 (frozen plan authored for the shipping lane; uncontrolled variable) -- MISREAD as stated
  (the video prompt is deliberately IDENTICAL between the lanes; that is the A/B), but it
  uncovers a REAL gap: the reused plan's shot rows carry `engine_id`, so a replay renders on
  the FROZEN engine (`otr_shot_lock.py` replay reuse returns the planned section unchanged), and
  a bundle frozen from the shipping lane cannot feed the peer's plates anyway. TAKEN as a new
  must-fix on the instrument: an explicit replay-time engine override (r2 designs it).
* Optional 1 (seed constant) -- the plate seed is dropped in favour of `request_seed` itself
  (Fable 3); nothing new on the request.

Fable
* 1 (plate; no dispatcher still exists on this policy; write the PNG) -- CONFIRMED. TAKEN.
* 2 (plate prompt: full `positive_tail` + `plate_look` + `_read_setting` top-2 +
  `get_era_tail(profile="still")`, never `compact_style_cue`; drop order; measure with
  `measure_clip_tokens`; pin all packs) -- CONFIRMED at `otr_meta_brief_image_prompt.py:1839-1865`
  (`_compose_background_plate_prompt` precedent) and `_otr_story_brief_helpers.py:321-341`.
  TAKEN, with the composition living in `render_driver`'s Ghost branch as a declared
  `VideoRequest` field (Fable 5's trap: `sampler_inputs_for(request)` is pure over the request;
  the engine has no ledger; `observability` is never conditioning). The schema change is the
  design item `eng_ghost_signal_official.py:144-149` names.
* 3 (no freeze, no routing: one subject-free plate per beat; plan untouched) -- CONFIRMED that
  `mode` is planned by ShotLock (`otr_shot_lock.py:1690, 2499`) and reused on replay; an
  engine-side override would make the rendered prompt differ from the planned row. TAKEN.
* 4 (latent-direct; motion-energy measurement; 0.9 next) -- TAKEN; the 08-30 grid
  0.35 / 0.50 / 0.65 / 0.80 replaces the anchor's 0.75-first (the probe's own grid).
* 5 (never hash the plate OUTPUT sha into `sampler_inputs`; A/A false by construction if SD1.5 is
  not bit-stable) -- CONFIRMED against `otr_verify_replay.py`'s row-for-row check. TAKEN.
* 6 (mint per render, write `stills/ghost_plates/<shot>_<sha8 of inputs>.png`, reuse a present
  plate on replay; never fabricate a dispatcher `images[]` row; verify node 91 tolerates an
  unlisted file) -- TAKEN; the freeze already copies `stills/` whole.
* 7 (first legs from sweep bundles) -- MISREAD in one detail: the sweep bundles were rendered on
  the shipping lane and carry no plates, which is fine for the PLATE lane (it mints per render)
  -- so this stands; the override in S4 is what lets the same bundle also render on the shipping
  lane as the baseline. Ten legs, all to `otr/obs/`. TAKEN.
* 8 (G3.7-scoped applies; `prompt_profile` must stay; no new node classes; instance-count pin;
  add the id to `_ANIMATEDIFF_ENGINES`; denoise env through the guarded parser inside a method;
  no tombstoned ids) -- CONFIRMED (`otr_shot_lock.py:2137, 2246-2247`; `otr_provision.py:1516-1551`).
  TAKEN.

## Verify-at-build
* Node 91 (`verify_replay_images`) tolerates an unlisted PNG under `stills/ghost_plates/` on a
  replay -- it checks listed rows; confirm before relying on plate reuse.
* `run_graph`'s returned LATENT object shape for `samples.repeat`; the `batch_index` rule.
* `test_ghost_signal_peers.py` floor tests pick the peer up automatically or need a roster edit.
