# OTR Cloud Engine Lanes -- voice / stills / video / (3D seam)

Status: pass00 draft for roundtable. DOCS ONLY -- no code, no workflow JSON
changes in this campaign. Build gated on operator go + free coder baton.
Date: 2026-07-02. Operator brief: per modality provide 1 CHEAPEST-WORKABLE
cloud option + 2 BEST-OF-CLASS options; video lane MUST support the
audio-reactive pipeline (episode audio drives/syncs the visuals).

## 0. Goal

Add a cloud provider lane to every generative modality (voice, music,
stills, video; 3D as a designed-but-unbuilt seam) so a full episode can
render with ZERO local GPU: local ComfyUI orchestrates story/assembly on
CPU; every heavy generation leg executes remotely, billed via the
Comfy account (Comfy Credits partner API nodes), the operator's allowed
cloud surface alongside OpenRouter.

Non-goal: running the OTR custom-node graph itself on Comfy Cloud (cloud
supports only their vetted custom-node set; the orchestration layer stays
local). Non-goal: replacing local engines -- cloud rows are opt-in
additions; byte-identical local defaults stay untouched.

## 1. Grounding -- what already exists (verified in repo 2026-07-02)

- One shared engine-registry pattern, three namespaces:
  `nodes/_otr_audio_engines/registry.py`, `nodes/_otr_video_engines/registry.py`,
  image (C1) -- all on dep-free `nodes/_otr_shared/engine_registry_base.py`.
  Adapters self-register; adding an engine = one adapter file + one import
  line. Fail-closed usability: `EngineUnusable` + 6-reason
  `EngineUsabilityReason` taxonomy incl. GATED_BY_FLAG, INCOMPATIBLE_PROFILE,
  NONCOMMERCIAL_BLOCKED. Engines carry `requires_flag`, `commercial_clean`.
- Video registry is CAPABILITY-based (C2 2026-06-30): an engine fits a role
  iff the role supplies every `required_inputs` token; tokens are
  `text_prompt | init_image | audio_ref | base_clip_ref`
  (`nodes/_otr_shared/role_compat.py`). `audio_ref` is a first-class token;
  families include `audio_driven_face` and `lipsync_overlay`. The
  audio-reactive requirement maps to existing machinery, not new concepts.
- Remote-lane pattern PROVEN twice in the LLM catalog
  (`nodes/_otr_model_catalog.py`): `provider` axis
  (`local | openrouter | comfy_credits | ollama`), virtual rows with zero
  VRAM, dropdown suffix convention (`[NOT DOWNLOADED]`-style), fail-closed
  JSON, per-run budget guard (BUG-296 lesson: reset budget per run).
- Comfy-Credits transport PROVEN (`nodes/_otr_comfy_backend.py`): auth via
  ComfyUI hidden inputs `auth_token_comfy_org` / `api_key_comfy_org`
  captured at run() -- no env keys; opt-in gate `OTR_ENABLE_COMFY_CREDITS=1`
  default-off; env-overridable endpoint (`OTR_COMFY_API_BASE`, default
  `https://api.comfy.org`); mismatch fails closed with a named error.
- UI seams are the existing engine/model dropdowns: 3a Character Voices
  (`engine`), 3b Announcer Voice (`engine`), 3c Theme Music (`engine`),
  `OTR_VideoDirector` per-role `*_video_model` + `*_image_model`,
  `OTR_ImageGenDispatcher`. Dropdowns are BUILT FROM the registries, so new
  cloud rows appear without node surgery. Adding OPTIONS to an existing
  dropdown does not change the widget COUNT -> no `widgets_values`
  positional drift in `workflows/otr_scifi_16gb_full.json` (BUG-LOCAL-097
  class). Workflow JSON is untouched until defaults change (S4, gated).
- Capability profiles exist (16gb / 8gb / cpu_floor) with a derived
  enable-set and one applier; headless honors `--profile`.
- Deliverable invariants that cloud lanes MUST NOT break: master audio is
  frozen before video; mux is LAST; in-render fallback is LOUD (log swap +
  ledger restamp, never silent); renders write straight to
  `otr\episodes\<ep>\`, finals to `otr\obs\`; sfx role and pooling are
  DELETED (2026-07-01, NO FALLBACKS) -- do not reintroduce an sfx lane.

## 2. Cloud catalog audit (verified against the live install + Comfy Cloud, 2026-07-02)

Source A: `list_api_nodes` on the running ComfyUI -- 214 hosted partner API
nodes present TODAY (billed via the logged-in Comfy account). Source B:
Comfy Cloud template catalog (cloud.comfy.org) -- open-source models
runnable on cloud GPUs (GPU-time billing), incl. ACE-Step 1.5 music and
Chatterbox TTS templates. Exact class_type names below are verbatim from
the install dump.

- VIDEO (91 nodes): Kling x21 (incl. `KlingAvatarNode` -- audio input;
  `KlingLipSyncAudioToVideoNode` -- lip-sync a rendered clip to audio;
  `KlingImageToVideoWithAudio`, `KlingTextToVideoWithAudio`), Vidu x13,
  Wan x12 (`Wan2ImageToVideoApi`, optional audio), Luma Ray x8,
  ByteDance Seedance x7 (`ByteDance2ReferenceNode` -- multimodal w/ audio
  reference, subject-identity preservation), Runway, PixVerse, Grok,
  `OpenAIVideoSora2` (video+audio), `GeminiVideoOmni` (Veo-class,
  generates its own synced audio).
- IMAGE (72 nodes): Recraft x12, BFL/Flux x10 (edit/inpaint/outpaint
  modes), Stability, Magnific, Ideogram, Gemini/Imagen + Nano Banana 2,
  Luma, OpenAI GPT-image, Grok Imagine.
- AUDIO (12 nodes): ElevenLabs x7 (`ElevenLabsTextToSpeech`,
  `ElevenLabsTextToDialogue`, `ElevenLabsTextToSoundEffects`,
  `ElevenLabsSpeechToSpeech`, `ElevenLabsSpeechToText`,
  `ElevenLabsAudioIsolation`, `ElevenLabsInstantVoiceClone`),
  Stability x3 (`StabilityTextToAudio`, `StabilityAudioToAudio`,
  `StabilityAudioInpaint`), Sonilo x2 (`SoniloTextToMusic`,
  `SoniloVideoToMusic`).
- 3D (32 nodes): Tripo x12 (text/image-to-3D, multiview), Rodin x8
  (quality-vs-cost modes), Meshy x7 (incl. rig/animate), Tencent
  Hunyuan3D x5.
- Pricing: per-call USD is NOT in the node dump (only per-second hints for
  a few). VERIFY-AT-BUILD: pull the Comfy partner-node pricing table and
  stamp each curated row with an approx_cost field before promotion.

## 3. Curated cloud rows (the operator-facing dropdown additions)

Selection rule per modality: 1 CHEAP row (cheapest that clears the quality
bar "listenable/watchable in an OTR episode") + 2 BEST rows (quality
first). Every row: provider="comfy_credits" (or "comfy_cloud_gpu" for
template-lane rows), zero local VRAM, `requires_flag` gated, cost stamped.

### 3a. VOICE (roles: char_voice, announcer_voice)

| Tier | Row | Backing | Why |
|------|-----|---------|-----|
| CHEAP | `cloud: chatterbox_cc` | Comfy Cloud template `audio-chatterbox_tts` (+`_dialog`) | Same engine as the local sidecar -> voice continuity; open-source on cloud GPU = GPU-time pricing, cheapest serious TTS; multi-speaker dialog template maps to radio scripts. |
| BEST 1 | `cloud: elevenlabs_tts` | `ElevenLabsTextToSpeech` | Industry-best naturalness; per-line delivery control. |
| BEST 2 | `cloud: elevenlabs_dialogue` | `ElevenLabsTextToDialogue` | Native multi-speaker conversation synthesis -- radio-drama-shaped; fewer stitches than per-line TTS. |

Voice-bank continuity: `ElevenLabsInstantVoiceClone` can be seeded from the
existing CC0 LibriVox reference clips so cloud voices match the local
casting concept (CastLock keeps assigning presets; cloud adapter maps
preset -> cloned-voice id). ToS/commercial-clean audit required
(verify-at-build) -- clone only the CC0 bank, never third-party voices.

### 3b. MUSIC (role: theme_music)

| Tier | Row | Backing | Why |
|------|-----|---------|-----|
| CHEAP | `cloud: ace_step_1_5` | Comfy Cloud template `audio_ace_step_1_5_*` | Open-source, renders a full song in seconds on cloud GPU; lyrics support for period jingles. |
| BEST 1 | `cloud: sonilo_music` | `SoniloTextToMusic` | Dedicated music partner; `SoniloVideoToMusic` optional later. |
| BEST 2 | `cloud: stability_audio` | `StabilityTextToAudio` | Cloud sibling of the local `stable_audio_3` default -> prompt + style continuity for the theme lane. |

### 3c. STILLS (roles: announcer/music/other_beats image)

| Tier | Row | Backing | Why |
|------|-----|---------|-----|
| CHEAP | `cloud: recraft` | Recraft image nodes | Cheap tier, strong stylization for period poster looks. |
| BEST 1 | `cloud: flux_pro` | BFL `Flux` pro-tier node | Continuity with local `flux_gen1` prompts/look; portrait quality. |
| BEST 2 | `cloud: nano_banana_2` | Gemini/Nano Banana 2 image node | Best-in-class character consistency + reference-image edit -> protects the portrait-hash / in-character invariants across shots. |

### 3d. VIDEO (roles: announcer_video, music_video, other_beats_video) -- AUDIO-REACTIVE REQUIRED

Definition (registry terms): a cloud video engine is audio-reactive iff it
declares `audio_ref` in `required_inputs` and consumes a per-beat slice of
the FROZEN master audio (existing per-beat audio machinery), OR it is a
`lipsync_overlay` family engine applied to a base clip. Engines that only
GENERATE their own audio (Veo/Sora native audio) are NOT audio-reactive
for OTR: episode audio is authored upstream and mux is LAST; their audio
track is discarded or the engine is used image-to-video-mute.

| Tier | Row | Backing | Family / audio path |
|------|-----|---------|---------------------|
| CHEAP | `cloud: wan_i2v` | `Wan2ImageToVideoApi` | image_to_video; optional audio in; cheapest workable I2V for b-roll beats. |
| BEST 1 | `cloud: kling_avatar` | `KlingAvatarNode` + `KlingLipSyncAudioToVideoNode` | audio_driven_face + lipsync_overlay; the talking-radio face driven directly by episode audio -- direct cloud upgrade of the ltx_radio_mouth concept. |
| BEST 2 | `cloud: seedance_2` | `ByteDance2ReferenceNode` (Seedance 2.0 R2V) | image_to_video w/ audio reference + subject-identity preservation -> portrait-anchored beats stay on-model. |

### 3e. 3D (SEAM ONLY this build -- no shipped engine)

Registry namespace + role tokens designed; adapters deferred. Candidate
rows recorded for the future build: CHEAP `Tripo P1` (game-ready
low-poly), BEST `Rodin Gen2.5` (quality modes), BEST `Meshy` (rig +
animate -- the path that could someday reopen the parked ARKit keystone
with REAL assets). Rationale: cloud 3D removes the local-toolchain blocker
(cu128/ninja) that parked 3D, but nothing downstream consumes meshes yet.

## 4. Architecture

- ONE new shared transport module (working name
  `nodes/_otr_shared/cloud_media_backend.py`): submit + poll + download for
  partner API-node jobs, reusing the `_otr_comfy_backend` auth capture
  (hidden inputs at run()), fail-closed named errors, retry/backoff,
  per-episode budget guard (env `OTR_CLOUD_MEDIA_BUDGET_USD`, reset per
  run -- BUG-296 pattern) and a cost ledger line per asset.
- Per-modality adapters are THIN: each cloud row is one adapter file in the
  existing registry namespace (audio/video/image), `requires_flag`-gated,
  declaring honest `required_inputs` + family. No registry surgery.
- Two transports, one adapter contract: (A) partner API nodes via the
  api.comfy.org proxy (primary; sync-ish job poll); (B) Comfy Cloud
  workflow submission for open-source template lanes (chatterbox,
  ACE-Step) -- verify headless API-key auth at build; if (B) proves
  awkward headless, the CHEAP voice/music rows fall back to lane (A)
  providers at slightly higher cost (named in plan, not silent).
- Global gate `OTR_ENABLE_COMFY_CLOUD_MEDIA=1` (default OFF) in addition to
  per-row flags; dropdown labels carry a `[CLOUD $]` suffix. Defaults
  unchanged -- byte-identical local baseline preserved.
- New capability profile `cloud` beside 16gb/8gb/cpu_floor: derived
  enable-set turns every generative role default to its curated cloud row;
  orchestration nodes (writer can already go OpenRouter/Comfy-Credits)
  need zero GPU. Acceptance: full episode on a
  `CUDA_VISIBLE_DEVICES=''`-style host with only network access.
- Ledger: every cloud asset stamps provider, model row, job id, USD cost,
  and restamps LOUDLY on fallback (existing invariant). `obs_publish`
  unchanged; assets land straight in `otr\episodes\<ep>\`.
- Commercial-clean: per-provider ToS audit recorded per row
  (`commercial_clean` + license note), release gate already enforces
  NONCOMMERCIAL_BLOCKED.

## 5. Sprints (build order, post-campaign, baton-gated)

- S0: shared cloud transport + budget guard + cost ledger + flags +
  catalog rows w/ verified pricing. Tests: transport fail-closed matrix,
  budget reset, no-network suite stays green (all cloud tests mocked).
- S1: STILLS lane (lowest risk; prompts already engine-agnostic via
  finish_visual_prompt). Acceptance: 3-image beat set on-model via cloud.
- S2: VOICE + MUSIC lanes (incl. voice-clone bank mapping). Acceptance:
  full audio episode, zero local GPU, master WAV byte-stable across mux.
- S3: VIDEO lane incl. audio-reactive paths (kling_avatar lip-sync +
  seedance audio-ref; wan_i2v mute b-roll). Acceptance: talking-radio
  beat driven by episode audio, mux-LAST intact.
- S4: `cloud` capability profile + no-GPU end-to-end acceptance render +
  cost report per episode; ONLY here may workflow JSON defaults change,
  in the same change as any code, validator + widget audit rerun.
- S5: 3D seam docs + registry tokens (no engine).

## 6. Open questions for the panel

1. Are the CHEAP picks actually the cheapest WORKABLE (Wan vs Kling-std vs
   PixVerse for video; Recraft vs Grok Imagine vs flux-dev for stills)?
2. Is ElevenLabsTextToDialogue mature enough to be BEST-2 voice, or should
   BEST-2 be MiniMax/other TTS via a different surface?
3. Audio-reactive: is discarding native-audio models (Veo/Sora) from the
   reactive set right, or is there a hybrid worth keeping (e.g. Veo for
   music-visual beats where OTR music replaces the track anyway)?
4. Transport (B) (Comfy Cloud workflow submission headless w/ API key):
   real or vapor for unattended batch? If vapor, does the CHEAP voice row
   survive as ElevenLabs-flash-tier or die?
5. Per-episode budget guard shape: hard abort vs degrade-to-local vs
   degrade-to-cheap-row at threshold? (Current lean: hard fail-closed at
   budget, LOUD ledger note; no silent degradation -- matches invariants.)
6. Anything in the 214-node catalog that beats these picks for a 1940s
   radio aesthetic that we ignored?
