VERDICT: no — §4 prescribes removing era assertions while §5 forbids that solution; the replacement lacks a shared owner, execution contract, and downstream coverage. Reviewed repository HEAD d1ec02fa, not the anchor’s 0a1ae33e.

MUST-FIX BEFORE BUILD:

1. [§4(b), §5 questions 2–3] The proposed fix contradicts the governing combination model. §4(b) explicitly drops “period costume” and assigns garments and surroundings to story context; §5 explicitly rules out stripping era or choosing a winner. These are different implementations. Concrete fix: replace §4(b) with one specification for the combined visual result, including character clothing, setting treatment, and house-frame scope. Keep the selected pack unchanged until that specification exists. The existing source rewrite receives both candidate and style request but makes no episode-wide staging commitment: nodes/otr_meta_brief_image_prompt.py:1745–1780.

2. [§5 questions 1–2, 5] An image-node-owned combination arrives too late for video planning. In workflows/otr_canonical.json, link 255 connects ShotLock node 90 to image-prompt node 89; link 256 separately connects ShotLock to dispatcher 91. ShotLock builds creative directives and execution plans before the image prompts exist (nodes/otr_shot_lock.py:3412–3421). Moving node 89 upstream blindly also fails: its scene-source context reads beats, shots and scenes (nodes/otr_meta_brief_image_prompt.py:1668–1733).

   Concrete fix: establish the shared visual decision before ShotLock authors creative prompts; retain shot-specific image composition afterward. [ASSUMPTION — proposed design] A visual-only record established at ShotLock entry, with scene and char_id references and persisted alongside its output, is the smallest viable shared state. Specify its author, missing-input behavior, and replay behavior. A helper that merely appends the same prose cannot decide coherent clothes for two independently described subjects.

3. [§5 question 1] “Same contribution everywhere” overlooks materially different composition paths. The required execution/contract table is:

   Path | Actual composition order | Integration consequence
   ShotLock character video | Appearance/setting plus authored or fallback acting prompt → speaking anchor where applicable → full brief tail → positive_tail | Runs before image prompts and uses a different tail profile. nodes/otr_shot_lock.py:1597–1654.
   Ordinary portrait | Appearance and setting plus portrait_instruction_look and portrait_look through LLM, or deterministic appearance/setting/portrait_look → portrait-profile tail → positive_tail → image_grade_tail | Pack content can occur inside the authored description and again in finishing. nodes/otr_meta_brief_image_prompt.py:1504–1593, 2183–2257.
   Talking portrait | Appearance/setting plus portrait_look_talking and optional portrait_instruction_look → warm dramatic lighting | Explicitly skips the era/style finishing and grade branch. nodes/otr_meta_brief_image_prompt.py:2146–2152, 2222–2229.
   Ordinary scene_character, LLM success | Appearance/beat/setting plus portrait_look and scene_instruction_look → LLM → portrait-profile tail and positive_tail → appearance potentially prepended again → grade → no-text | Different profile and identity reinsertion from fallback and My Story. nodes/otr_meta_brief_image_prompt.py:1614–1652, 1895–1920.
   Ordinary scene_character, fallback | Appearance or hardcoded period-dressed subject → setting top two → wide framing → still-profile tail → positive_tail → grade → no-text | No shared clothing decision for companions; fallback itself adds period dress. nodes/_otr_story_brief_helpers.py:558–608.
   My Story scene_character | Fully composed fallback → source rewrite with raw source, scene context and style-bearing request → no-text only | Rewrite output owns the final description; appending the old appearance afterward would restore corrected contradictions. nodes/otr_meta_brief_image_prompt.py:1745–1803.
   scene_beat / scene_open | Brief-selected radio form inside open_subjects announcer/default/synthetic template → setting top two → kind-specific framing → still-profile tail → positive_tail → grade → conditional broadcast_tail → no-text | A faceless object gets the same costume-bearing positive_tail as people. nodes/_otr_story_brief_helpers.py:440–471, 569–608.
   Background plate | Setting → subject-free geometry plus plate_look → still-profile tail and positive_tail → grade → no-text | Costume vocabulary enters a surface explicitly intended to contain no subject. nodes/otr_meta_brief_image_prompt.py:2025–2051.

   Additionally, get_era_tail does not always add the pack’s era_tail: available brief atmosphere/palette/lighting replaces that fallback, with different portrait/still/full projections (nodes/_otr_story_brief_helpers.py:280–341).

   Concrete fix: define uniformity as shared staging decisions, then specify their projection into each row. Preserve intentional portrait lighting and framing differences. Route LLM success, unavailable-LLM fallback, and unresolved source correction through that same decision contract without restoring stale appearance text.

4. [§5 questions 1–2] ShotLock’s text is not the final video prompt. The render driver can shorten talking prompts, substitute engine-owned prompts, or construct bookends from motion registers. One scene branch explicitly uses style_tail=False and a 188-character budget; another uses the full tail with a 620-character budget. See nodes/_otr_video_engines/render_driver.py:3542–3592, 3618–3641, 3872–3896, 3938–3956. The shared compact style cue normally preserves only two words from positive_tail (nodes/_otr_visual_styles.py:632–653).

   Concrete fix: extend the table through final req["text_prompt"] and init_image selection. For image-conditioned motion lanes, carry staging through the originating image and preserve the motion contract; for text-generated scenes, explicitly transport the shared staging decision through their engine composer. Do not solve this by appending full wardrobe/set prose to every motion prompt.

5. [§5 question 3] Uniform period clothing would not by itself fix the announcer frame. The documented shot is announcer_visual / scene_beat, whose subject contract is a faceless radio, not a costumed dramatic person (nodes/_otr_story_brief_helpers.py:442–450, 493–515). A consistently dressed man in a consistently historical kitchen could still violate that contract.

   Concrete fix: separate dramatic staging from house-frame subject identity in the shared specification. Apply clothing to people, set treatment to environments, and object treatment to radios. This requires no forbidden-word list. [ASSUMPTION] A period-staged dinner could satisfy the combination ruling, but neither these functions nor the supplied evidence establishes it as the uniquely correct interpretation. Prompt uniformity also cannot prove pixel coherence; docs/PROD_BUG_LOG.md:14507–14511 explicitly leaves that unresolved.

6. [§5 questions 1, 4] The proposed inventory ends before additional content writers:

   - radio_form_from_meta independently chooses physical objects from brief keywords and otherwise returns “a vintage tabletop tube radio receiver” (nodes/_otr_story_brief_helpers.py:367–412).
   - _radio_face_overtness adds “subtle period-authentic dial-face” outside pack ownership (nodes/otr_meta_brief_image_prompt.py:388).
   - Ideogram’s music-card adapter injects “a period tabletop radio receiver” into the vendor object description after prompt composition (nodes/_otr_image_engines/ideogram4_local.py:570–574).
   - Ghost’s subject distillation fills missing costume buckets from a seeded pool containing coats, a uniform and a shawl (nodes/_otr_video_engines/ghost_signal_prompt.py:278–281, 546–559). This is another clothing writer, although not proof of the LA still’s cause.

   Concrete fix: include these paths or explicitly limit the repair’s claimed coverage. Route their subject/costume choices through the shared visual decision while preserving adapter geometry and specialized subject contracts. Verify which engines produced the supplied stills before attributing their pixels to any newly identified writer.

SHOULD-FIX:

1. [§5 questions 2, 5] Specify regeneration and cache identity before qualification. Dispatch hashes normalized prompt text and includes consumed portrait-reference identity; changing a portrait can consequently regenerate dependent scenes (nodes/otr_image_gen_dispatcher.py:1338–1357, 1558–1607). Adapter-only changes can leave upstream prompt text unchanged.

   Concrete fix: include shared visual state in the relevant context identity, preserve reference-image invalidation, and bump engine_version when an adapter’s effective input changes without changing the upstream request hash. Exercise both fresh rendering and cache reuse.

2. [§5 question 5, §6] “Render-inert” is not permission to ship before the wave: §6 prohibits all shipping. Pack changes also require process lifecycle planning because the registry is cached in _STYLES (nodes/_otr_visual_styles.py:552–568).

   Concrete fix: keep this arc to review/design artifacts. After the wave, deploy one coherent revision, restart resident servers, and verify the loaded pack content before qualification. Embedded visual_storybased packs retain their separate hash-validated replay contract (nodes/_otr_visual_styles.py:590–615).

OPTIONAL / NICE-TO-HAVE:

[§5 questions 1–2] Record the shared visual-decision hash beside existing prompt/source receipts so a frame can be traced without another report-only checker. Existing receipt seams are nodes/otr_meta_brief_image_prompt.py:1804–1813 and nodes/otr_image_gen_dispatcher.py:1062–1078.

CUT THESE (over-engineering):

1. [§4(a), §4(c)] Keep eligibility filtering and mandatory migration of all packs cut. Neither supplies shared episode staging; existing named packs already resolve independently through nodes/_otr_visual_styles.py:618.

2. [§5 question 2] Cut literal prompt equality across surfaces. Portraits, empty plates, radios and motion instructions have different contracts. Share visual decisions, not complete prompt strings; the distinct contracts are explicit in nodes/otr_meta_brief_image_prompt.py:391–464, 2025–2051 and nodes/_otr_video_engines/render_driver.py:3847–3853.

3. [§6] Cut any additional corrective loop or subjective publication gate. The existing My Story scene correction already has bounded attempts and retains its initial prompt when unresolved (nodes/otr_meta_brief_image_prompt.py:1767, 1796–1799). Fix the inputs and ownership of that operation.
