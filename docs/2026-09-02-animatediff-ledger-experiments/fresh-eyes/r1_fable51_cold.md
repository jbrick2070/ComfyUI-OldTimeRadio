# r1 -- Fable 5.1, cold (the document only, no repository access), 2026-09-02

## 1. What I would do that the document does not propose

### A. The lane mints its own anchor still: in-family, native 512x288, no Klein

**Mechanism.** A single-frame branch in the engine's graph, taken from the checkpoint MODEL before the ADE loader: `LoraLoaderModelOnly(v3 adapter at 0.0) -> KSampler(EmptyLatentImage 512x288, batch 1, the shipped 20/8.0/euler/normal) -> VAEDecode`. Per shot, a setting plate prompted from the three unread setting strings, `visual_palette`, `story_brief_terms.lighting`, `episode_canon.time_of_day` and the pack's FULL language (`scene_instruction_look`, era/positive/grade tails - a still prompt is not a motion prompt, so the 08-22 stability finding does not fence it). Per `(shot_id, char_id)`, a figure plate from the motif plus `cast[].character_description`, seeded from `hash(brief, cast, char_id)` so the same figure recurs in every shot, as `identity_seed_basis` does for portraits. Both saved to `episodes/<ep>/stills/` and stamped as `images.images[]` rows (`kind = scene_background_plate` / `scene_character`) so they are citable by `content_hash`. The plate then feeds the clip exactly as E1 or E2 do, plus one twist: `ADE_NoisedImageInjection.mask_opt` as a centre-third cut-out on figure beats, so the room is pinned and the figure keeps the shipped recipe's motion freedom.

**On screen.** Every beat of a shot starts from the same room and the same figure, and the medium arrives as a picture (5B door 1) instead of two words.

**Why it beats E1/E2/E7.** Their shared prerequisite flips `accepts_still`, pulls the lane out of `_NO_STILL_VIDEO_ENGINES`, gates the 4060 on an ~11 GB Klein bundle that has never run beside AnimateDiff, and hands the engine a 1472x832 still from a foreign model family to downscale 2.875x. This plate is the same checkpoint and VAE at the latent canvas, so a low-denoise init is in-domain and holds; the parked still prerequisite stays parked.

**Cost.** 16 GB: one to three extra single-frame samples per shot, a second or two each, batch 1 against a 16-frame window, so the 13.5-14.2 GB peak cannot move; zero downloads. 8 GB: identical, no Klein, no provisioning change.

**Risk.** Base SD1.5 with pack language is still base SD1.5 (5B fact b), so anime lands weaker than Klein would. Anchor every beat to the plate, never to the previous clip, or grime compounds. E1's breathing-still risk stands; the mask and E2's mid-point injection are the mitigations to sweep.

**Judging.** Tectal Echo, A/A null at the E0 strength, then the plate arm at denoise 0.6/0.75/0.9; no seed moves because the plate is not a seed input. The eye compares `shot_001_b1` with `shot_001_b3` (same figure, same room), with the plate PNG beside the clip in `stills/`.

### B. Render the shot as one timeline on the ledger's clock, then slice it back into beats

**Mechanism.** The driver groups `lines[]` by `shot_id` (`boundary == "shot_start"` is the cut it already writes), sums the beats' `target_frame_count` (cumulative from audio samples, so exact), and submits one timeline through the unchanged Gen1 graph. `ADE_PromptScheduling` takes one keyframe per beat at source frame `round((line.start_s - shot_start_s) * 12.5)`, whose text is that beat's existing v2 positive, untouched, duplicated one frame earlier for a hard switch. `ADE_AnimateDiffSamplingSettings.noise_type = FreeNoise` fills the empty `sample_settings` socket - the zero-cost cross-window control the document lists under sample settings and never proposes. After `VAEDecode`, slice at the same indices into per-beat clips, so hold-2, tail trim, receipts and the frame-count invariant see what they see today.

**On screen.** Room, light and figure do not reset at every line; b3 picks up where b1 left the figure; the object beat between reads as a cut-away inside one world, not a third hallucination.

**Why it beats E3 and E5.** E3 shapes one beat but the next still begins from nothing; E5 anchors a face inside one beat. This uses `beats[]` grouping and `boundary`, which section 3 lists as continuity the renderer throws away and which no arm E1-E14 touches.

**Cost.** Zero downloads. Windows are flat: Tectal shot_001 is about 67+78+68 source frames, 6+7+6 = 19 windows separately versus 18 joint, so sampling time holds; the new memory step is decoding 213 frames, chunked. Same on 8 GB.

**Risk.** A figure-to-object switch morphs across the 4-frame overlap: haunted dissolve or mush, and only the A/B says which. Longer timelines drift more; FreeNoise holds it for free, ContextRef is the paid fallback. It changes `FrameContract`, so it is a design arc, not a graph tweak.

**Judging.** One seed per shot (the first beat's `request_seed`), so say plainly that b2/b3 seeds change; judge the shot as a unit against the A/A null's three beats played in sequence, which is how the operator watches anyway.

## 2. The ranking I disagree with

- **E9/E14 -> before E0.** Section 9 makes the durable prompt/seed record a precondition of step 2, so a sweep run before it produces clips no one can tie to their text.
- **E4 -> above E3.** "Does the picture move with the voice" is one of the document's own four eye criteria, E4 is the only arm that touches it, and the engine-local route needs no schema change and no download.
- **E6 -> below E8.** Doubling or tripling the video phase on 28-37-clip overnight loops is a large price for a flicker gain that FreeNoise should be tried for at zero cost first.

## 3. What the document did not tell me

1. Does the engine submit its own API sub-graph per clip, or must every node it uses exist in `otr_canonical.json` - which decides whether a plate branch or a prompt schedule is an engine edit or a canonical-graph edit?
2. Are a clip's decoded source frames kept on disk before hold-2 and tail trim, and in what format, so a frame can be re-read and cited by hash?
3. Does `ADE_AnimateDiffLoaderGen1` clone the checkpoint MODEL, leaving the plain MODEL usable for a single-frame sample in the same graph?
4. Which `noise_type` values does the installed 1.6.0 `ADE_AnimateDiffSamplingSettings` expose, and how does its `seed_override` interact with the KSampler seed the lane sets?
5. What does `FrameContract.continuity` accept beyond `CONTINUITY_NONE`, and can the driver already slice one rendered timeline into beat clips (the inverse of `#seg<N>`)?
6. How does `ADE_PromptScheduling` express a hard switch at a frame index, and is each keyframe measured against the 77-token window?
7. What is the video-phase wall time per clip on the 5080 and on the 4060 separately, since only whole-episode and pod numbers are given?
8. What is the `mask_opt` contract on `ADE_NoisedImageInjection` (pixel or latent size, which polarity injects)?
9. Does anything mint a `scene_background_plate` row today, and can the lane's own plate reuse that kind and `_still_index`'s preference rule?
10. Is a per-machine arm acceptable to ship (Klein-fed E1 on the 5080, still-less on the 4060), or must one arm publish on both boxes?
11. Does an SD1.5 finetune under CreativeML OpenRAIL-M meet the pack's licensing bar for E11, given the dropdown label advertises Apache-2.0?
12. Which profile and episode is the designated experiment bed, and how are the A/A pair and the arm titled in `otr/obs/` so the eye can tell them apart?
