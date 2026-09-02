# Fresh-eyes round on the AnimateDiff v3 + ledger problem statement -- the driver's judgment

**Driver and sole judge: Claude (Fable 5.1, Cowork), 2026-09-02, HEAD `6e8c2e8a`.**
Roster, exactly: Fable 5.1 cold (the document only, no repository) -- `r1_fable51_cold.md`;
Codex `gpt-5.6-sol` via kibitz, file-grounded -- `r1_codex_gpt56.md`; Cursor `grok-4.6-high` via
kibitz, file-grounded -- `r1_cursor_grok46.md`; Antigravity `Gemini 3.7 Flash (High)` via kibitz,
file-grounded -- `r1_antigravity_gemini37.md`. Every claim folded below was re-read at the real
files before it was folded; the ones that were not verifiable are marked.

## 1. Where the four agree (and the files agree with them)

1. **Persist the prompt and the seed BEFORE any visual arm** (all four). Today `video.shots[]`
   with the composed prompt lives only on the wire and `_merge_with_disk`'s `TOP_PRESERVE`
   drops it (`production_ledger.py:1592-1598`); the per-clip receipt carries no prompt, seed or
   request hash. E9 is day-zero, and E14 is the same item (cut as a separate arm).
2. **Do not flip the shipping engine's still contract** (Codex, Cursor, Antigravity). Declaring
   `accepts_still = True` on `animatediff15_v3_haunted_video` pulls the lane out of
   `_NO_STILL_VIDEO_ENGINES` (`otr_provision.py:1546`), gates the 4060 on the ~11 GB Klein
   bundle, trips the preflight that asserts `subject_ownership == "prompt"`
   (`tests/test_ghost_signal_lane.py:104`, `tests/test_lane_preflight_matrix.py:1036`) and
   changes ten profiles at once. A still-in arm is a NAMED LAB PEER ENGINE with its own id and
   receipt, on the 5080 first; the 4060 gets a measured leg later, never as a gate on the arm.
3. **The still should be in-family and at the lane's canvas** (Fable A, Antigravity 4, Cursor
   optional). A 1472x832 still from a foreign model family downscaled 2.875x to a 64x36 latent
   is the wrong input; the lane can mint its own 512x288 anchor with the same checkpoint and
   VAE (adapter at 0.0 for the plate), prompted with the pack's FULL style language and the
   ledger's setting, palette and lighting. Grounded: `ADE_AnimateDiffLoaderGen1` clones the
   model (`nodes_gen1.py:85`), so the plain checkpoint MODEL stays usable for a single-frame
   branch in the same graph; the engine builds its own API sub-graph
   (`eng_ghost_signal.py:693 _build_render_request`), so this is an engine edit, not a
   canonical-graph edit.
4. **The adapter strength is the cheapest style lever and the sweep must include 0.0 and 0.25**
   (Antigravity, Cursor, Codex). `ADAPTER_V3_STRENGTH = 1.0` is unqualified, the env override
   `OTR_GHOST_HAUNTED_LORA_STRENGTH` already exists (`eng_ghost_signal_official.py:116`), and
   the anime pack's own negative bans the very grime the adapter at 1.0 re-adds. E0 becomes a
   per-style calibration (anime, storybook_engraving, sci_fi_radio at 0.0 / 0.25 / 0.5 / 1.0),
   not one global number frozen before style work.
5. **Cross-beat continuity is the real "eleven unrelated hallucinations" problem** (Fable B,
   Codex 4, Cursor 3). Every still and every clip is per beat; ContextRef only steadies windows
   inside one clip. A shared SCENE PLATE per shot (Fable A's setting plate, keyed by
   `shot_id`) is the additive answer; one timeline per shot with a prompt keyframe per beat and
   `FreeNoise` (present: `NoiseLayerType.LIST` in ADE 1.6.0 `sample_settings.py:49`) is the
   stronger answer and a `FrameContract` change (`CONTINUITY_NONE` today; the contract also
   knows `strict_first_frame` and `soft_reference`), so it is a design arc.
6. **Blind the judge** (Codex 5; lesson 40 in `docs/PRODUCTION_SPRINT_LESSONS.md:739-761`) and
   **publish two A/A nulls, not one** (Cursor S3), since a fixed seed is not bit-stable across
   the eight context windows of a 95-frame beat. Score style, face, setting and
   motion-to-speech separately before "overall better".

## 2. The finding that changes the protocol: the same-ledger A/A cannot run today (Codex 1)

The document's judging protocol assumes the same frozen ledger can be re-rendered through a
changed graph. It cannot through the canonical graph: node 1 (`OTR_LedgerScriptWriter`) always
re-executes and writes a NEW ledger with a fresh brief and cast roll, so `render_request_hash`
(brief terms + cast + beat + char) and therefore every seed moves. `preserve_ledger` is a CAST
policy (`cast_lock.py:67`), not a replay. The seed rule in section 2 of the statement is
correct about what moves the seed; it is silent about the fact that nothing in the graph can
hand the same ledger to the render phase twice. **A replay path is part of the instrument work
(item 0 below), and it is a design item** -- a canonical replay mode that injects an accepted
ledger after authorship and runs the real ShotLock-to-publish path into a distinct directory.
Until it exists, an A/B is two different episodes, and the eye compares stories, not arms.

## 3. The proposals, one by one

| proposal | from | grounded? | verdict |
|---|---|---|---|
| In-family anchor still minted by the lane at 512x288, adapter 0.0, pack language + ledger setting; figure plate per (shot, char); `mask_opt` centre cut-out on figure beats | Fable A | yes (model clone, engine-owned graph, `ADE_NoisedImageInjection.mask_opt` exists) | TAKEN as the still source for the lab peer; it dissolves the Klein gating and the 2.875x downscale in one move |
| One timeline per shot on the ledger's clock, prompt keyframe per beat, FreeNoise, slice back to beats | Fable B | yes (FreeNoise present; `ADE_PromptScheduling` present; `FrameContract` change) | TAKEN as a later design arc (item 4); not a first-wave graph tweak |
| E9/E14 before E0 | Fable, Codex, Cursor, Antigravity | yes | TAKEN (item 0) |
| E4 speech energy up (Fable) vs cut (Antigravity) vs mechanics incomplete (Codex: `ADE_ValueScheduling` emits FLOATS, the socket wants MULTIVAL) | split | yes on the mechanics | DEFERRED: engine-local only, after the instrument and the still peer are judged; the "does the picture move with the voice" criterion stays in the scorecard |
| Ledger replay mode + full `render_request_hash` (prompt, still hash, recipe, adapter, denoise) beside an honest `comparison_seed_hash` | Codex 1, 3 | yes | TAKEN (item 0); `OTR_VideoRenderBatch` is the sole writer of the actual-render trace |
| Lab peer engine instead of flipping the shipping engine | Codex 2, Cursor 2/6, Antigravity 1 | yes | TAKEN (item 2); E1's denoise < 1.0 and VAE init are recipe cells, so the peer stamps its own receipt id, never `..._haunted_static16_512x288_v1` |
| Decide per mode: figure beats consume the character still; object/signal beats stay text-only or take a prop plate; or freeze the cycle to figure on the peer | Cursor 3 | yes (`GHOST_CHARACTER_CYCLE` = figure, object, figure, signal) | TAKEN: the peer freezes the cycle to `figure` for character beats in its first build; the object/signal cut-aways are a second knob |
| Per-style checkpoint / LoRA as arm 1 (Antigravity 3) vs cut from the first build (Cursor, Codex) | split | yes (no checkpoint field in the pack schema; ~2 GB per pack) | DEFERRED behind E0-per-style: if adapter 0.0 + the pack's language + the in-family plate cannot draw anime, E11 is the next arm, named, with its own provisioning |
| Token budget for any prompt growth (20 style / 20 motif / 20 leaf / 15 law) | Antigravity 5 | yes (the composer measures 77 CLIP tokens) | TAKEN as the rule for E12 |
| Cut E13 (style-aware roll hides the defect) | Cursor, Codex | agreed | CUT |
| Cut / defer E6, E7-first, E8 | Cursor, Antigravity, Codex | agreed | E6 deferred; E7 after the peer is judged (it stays the operator's named arm, later); E8 cut from the first campaign |
| Cut the 5C research questions as a parallel programme | Cursor 7 | partly | KEPT as questions, not as a campaign: E12 is one budgeted same-seed pair per style; the rest are answered by the arms as they run |
| `free_otr_pipeline_residue` never calls `unload_all_models`; verify the image engine leaves VRAM before the haunted sampler | Cursor S6 | yes (`_otr_vram_levers.py:17`) | TAKEN as a measurement step in item 2 |
| Adapter sweep at 0.0 immediately restoring style | Antigravity Q4 | not established | the first thing item 1 measures |

## 4. The order the driver proposes (the first campaign)

0. **Instrument.** E9 (durable `video.shots[]` with the composed prompt, negative, seed, a full
   request hash beside the comparison seed; per-clip receipt gains `prompt_sha8` and
   `request_seed`) + a canonical **replay mode** + the blinded two-null scorecard. No picture
   changes. Design item (the replay), so it gets a round before code.
1. **E0 per style, including 0.0.** `OTR_GHOST_HAUNTED_LORA_STRENGTH` at 0.0 / 0.25 / 0.5 / 1.0 on
   `anime`, `storybook_engraving`, `sci_fi_radio`, the same ledger via replay, blinded. Zero code,
   zero downloads; can be rendered the same evening the replay lands (and, unblinded and on
   fresh episodes, can start tonight as a look-see).
2. **The still-in lab peer** with the in-family plate: a new engine id and receipt; scene plate
   per shot + figure plate per (shot, char) minted by the lane at 512x288 from the pack's full
   language and the ledger's setting, palette, lighting; consumed as a low-denoise init (E1) or a
   mid-point injection with a centre mask (E2), cycle frozen to `figure` for character beats;
   5080 only until measured; VRAM measured after the image phase. This is "stills in" and "style
   through the still" at once.
3. **E12, budgeted.** The pack's language back into the prompt under the 77-token budget, one
   same-seed pair per style.
4. **One timeline per shot** (Fable B): a design arc on `FrameContract` and the driver's slicing;
   after 2 is judged.
Deferred: E4 (engine-local, later), E7 SparseCtrl (after 2), E11 (if 1 + 2 fail on anime), E6.
Cut: E13, E14, E8, IP-Adapter / PIA / Lightning.

## 5. Answers to the builder questions the round produced (grounded where marked)

* The engine submits its own API sub-graph per clip (`_build_render_request`, engine-owned node
  candidates); a plate branch or a prompt schedule is an engine edit, not a canonical edit.
* `ADE_AnimateDiffLoaderGen1` clones the MODEL (`nodes_gen1.py:85`); the plain MODEL is free
  for a single-frame sample in the same graph.
* `ADE_AnimateDiffSamplingSettings.noise_type` in 1.6.0: default, constant, empty,
  repeated_context, FreeNoise; ancestral options carry `seed_override`
  (`nodes_sample.py:66-80`).
* `FrameContract.continuity`: `strict_first_frame`, `soft_reference`, `none`
  (`frame_contract.py:48-58`); a driver-side slice of one timeline into beats does not exist
  yet (the inverse of `#seg<N>` is not written).
* `scene_background_plate` is minted today only for the mesh-fodder roles
  (`otr_meta_brief_image_prompt.py:2294-2426`); the peer can reuse the kind and
  `_still_index`'s preference rule.
* Not established here: `mask_opt`'s pixel-vs-latent contract; per-clip wall time on each box
  (the video phase is not timed on disk -- E9 should add it); whether an OpenRAIL-M finetune
  meets the pack's licence bar for E11; the designated experiment bed and the obs title scheme
  for blinded candidates (a driver decision when item 0 lands).

## 6. What goes to the operator

Three options, in the order the driver recommends, each judged as radio drama by his eye after
a blinded two-null comparison: (A) build the instrument first (E9 + replay + scorecard),
(B) sweep the adapter per style, which needs no code and can start tonight on fresh episodes as
a look-see, (C) the still-in lab peer with the lane's own in-family plate. The statement and the
four reads are in this folder; nothing in the shipping recipe changes until he says which.

## 7. Cursor's planner read (the operator's other window, pasted 2026-09-02 ~16:00) -- folded

Cursor, working as a grounded planner in its own window (its file `r1_cursor_grounded.md` is
that window's to commit), agrees with the order above and adds three corrections the driver
takes: (a) the overnight headroom is **323 MB** (14,177 MB peak against the 14,500 MB target),
not the 300 MB to 1.2 GB range the statement gave -- one-clip VRAM probes precede ContextRef or
FreeInit; (b) the existing haunted ledgers carry `images.images = []`, so a same-ledger A/A on
Tectal Echo cannot consume stills that were never minted -- the still protocol is "mint once,
freeze `images[]`, then the video A/A"; (c) object and signal beats take a PROP still, figure
beats the character still (Zhang's overcoat cannot seed Boyden's amber key) -- a variant of the
mode decision in section 3, and the driver prefers it to freezing the cycle. It also names the
receipt-side E9 shape precisely: a `render_trace` stamped by `OTR_VideoRenderBatch` of what was
actually sampled (prompt, seed, adapter strength, still hash), with `prompt_sha8` and
`request_seed` on the clip receipt; ShotLock's `video.shots` stays the director's ask. Taken.

Two points the driver reads differently:
* **The adapter sweep and 0.0.** Cursor excludes 0.0 because a zero adapter under the haunted
  receipt is a lie (the engine itself logs "clean picture under a haunted receipt"). Right about
  the receipt, and the driver keeps 0.0 as a DIAGNOSTIC render under an honest label (not the
  haunted receipt id), because it is the one render that says whether the adapter is what
  kills anime. The published sweep is 0.25 / 0.5 / 0.75 / 1.0 on both `anime` and
  `storybook_engraving`.
* **"E0 on two frozen ledgers" needs the replay too.** Nothing in the canonical graph can hand
  a frozen ledger to the render phase a second time (section 2), so the frozen-ledger sweep
  waits on the replay exactly as the still arms do; an unblinded look-see on fresh episodes is
  the only thing that can start before it.

Cursor's five operator calls, with the driver's recommendation beside each: peer engine, not a
flag on the shipping lane (**peer**); native 512x288 stills, not a 1472x832 downscale
(**native**); whether E9 may start as a design arc while alpha.16 waits on the operator's
control experiment (**yes -- it is docs and ledger plumbing, and the registry control is a
separate, operator-triggered push**); one timeline per shot inside the first still week or
after (**phase 2, after the peer is judged**); the 8 GB profiles stay on the empty-latent peer
until a Klein-then-AnimateDiff episode publishes on that box (**unchanged until proven**).

## 8. Codex's second read (the operator's other window, pasted 2026-09-02 ~16:10) -- folded

Read-only, file-grounded, and it agrees with the order (instrument before pixels; E0 the first
pixel arm WITH 0.0 as a diagnostic; the still arm a named lab peer). Four specifics the driver
verified and takes:
* **The instrument is an immutable node-92 ingress bundle plus two receipts.** Freeze what node
  92 (`OTR_VideoRenderBatch`) receives -- the wire ledger, the master and per-line audio, the
  still files, the episode identity -- under a SHA-256 manifest; ShotLock owns the durable
  PLANNED `video`, VideoRenderBatch owns a bounded `render_trace[]` of what was ACTUALLY
  sampled (final positive and negative text, the seed that reached the sampler, adapter id and
  strength, engine and recipe id, model hashes, denoise / injection / context settings, still
  content hash, per-clip peak VRAM). The existing content-derived hash stays the A/B seed
  basis; a separate actual-request SHA covers everything that reached the sampler. Replay runs
  through the canonical downstream tail and counts only with `obs_publish OK` and the file on
  disk. Grounded: the existing visual smoke submits ONLY node 92
  (`scripts/otr_visual_smoke.py:8-19`), so it proves the render, not the composite-to-obs path.
* **The peer's still routing follows `ghost_prompt.mode`**: figure -> the portrait-derived
  `scene_character`; object -> an object still with no person; signal -> an environment still;
  the bookend -> `scene_open`. (Cursor's "prop still for object beats", made exact.)
* **The latent repeat node.** Core `RepeatLatentBatch` caps `amount` at 64
  (`ComfyUI/nodes.py:1310`), below this lane's 95 and 125-frame beats; the installed
  VideoHelperSuite 1.7.9 registers `VHS_DuplicateLatents` (`videohelpersuite/nodes.py:1056`,
  "Repeat Latents") with `multiply_by`, so the peer's graph is
  `IMAGE -> resize (policy stamped on the receipt) -> VAEEncode -> VHS_DuplicateLatents(U) ->
  KSampler.latent_image`, denoise 0.75 first, 0.9 only if motion is damped.
* **E2 is not the near-zero-cost twin of E1.** ADE's `perform_image_injection`
  (`animatediff/sampling.py:645-681`) decodes the whole x0 batch through the VAE, composites,
  and re-encodes it at each injection point -- on a 95-frame beat that is a full-batch decode
  and encode mid-sample. E2 moves behind E1 in the order (it already was) and gets its own
  VRAM probe before it is scheduled at all.
Also taken: state the portrait cost honestly (either `portrait required = never` for a true
one-still-per-beat trial, or count the portrait renders), and review the raw still first, then
synchronized per-beat A/A/candidate cards, with style, identity, setting, motion and flicker
scored diagnostically and the operator's blinded "overall better" decisive.

Its seven open builder questions join Cursor's five on the operator's list: the exact frozen
canonical boundary for replay; whether a named non-shipping engine/profile is acceptable for
all still experiments (the driver: yes); per-beat vs per-scene anchor for object/signal modes;
crop / pad / stretch for any resize and where it is stamped (the driver: pad, stamped on the
receipt); the frozen compact corpus (anime, engraving, sci-fi, all three modes, one 95+ frame
beat); whether an improvement must ship on both cards or may stay a 5080-only experimental
option; and where the blinded key lives so filenames and receipts do not disclose the arm.

## 9. Gemini "deep research" (pasted by the operator 2026-09-02 ~16:20) -- folded

A long restatement of the problem statement in report form; it does not touch the repository
and it does not know the two findings the grounded reads produced (the replay gap and the
peer-engine rule), so its roadmap flips `accepts_still` on the shipping engine and puts E2
ahead of E1, both of which the verified reads reject (sections 3 and 8). It also keeps E13
(cut) and gives the adapter sweep as 0.5 / 0.75 / 1.0 with a side note that non-photographic
styles "require a significantly lower strength (0.4 to 0.6)" -- consistent with sweeping
0.25 and a 0.0 diagnostic. One wrong attribution: the two-word video style cue is composed
by `compact_style_cue` in `_otr_visual_styles.py` through `ghost_signal_prompt.py`, not by
`otr_meta_brief_image_prompt.py` (that file composes the STILL prompts).

What it adds, and the driver verified on the Hub (2026-09-02): `CameraCtrl_pruned.safetensors`
is exactly 873,372,736 B (its number); `v3_sd15_sparsectrl_rgb.ckpt` is 1,988,040,333 B and
`v3_sd15_sparsectrl_scribble.ckpt` 1,992,335,697 B; `pia.ckpt` is 1,673,373,725 B; the v2 motion
LoRAs are about 77 MB each (`v2_lora_ZoomIn.ckpt` 77,474,499 B); the Hub copies of
`v3_sd15_mm.ckpt` and `v3_sd15_adapter.ckpt` match the dev box byte for byte. The statement's
"not established" size rows are now filled. Its VRAM reading of E7 -- a 1.99 GB encoder
beside a 2.13 GB checkpoint and a 1.67 GB module against 323 MB of overnight headroom -- is
the same reason E7 sits after the peer with its own probe.
