# SA3 Music Quality — Problem Statement for Roundtable (2026-06-14)

**Owner:** Jeffrey A. Brick | **Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha` | **Bug:** BUG-LOCAL-408
**Goal:** make the DEFAULT music engine (Stable Audio 3 / SA3-native) produce **real,
episode-appropriate music** for an old-time-radio sci-fi show — driven by the existing
**Meta brief** — by improving the PROMPT GRAMMAR, the SA3 SAMPLER INPUTS, the TIMING
conditioning, and the SEED/determinism handling. SA3 stays the default (dependency-clean);
we are NOT reverting to musicgen. The panel hardens the *how*.

---

## 0. Operator verbatim
- "our old music gen was good i thought stable audio would be better but noooo"
- "it could be a prompt — i dunno if the prompts changed since 6/5 when musicgen had good music, or if stable audio doesn't work the same way, so need improved music based prompts"
- Decision: keep SA3, **improve prompts / inputs / seed / temp for SA3 to get real music that meets our episode usage using the Meta brief.**

## 1. Grounded findings (checked against the real code, read-only)
1. **The music prompt builder did NOT change since 6/5.** `nodes/_otr_music_prompt.py` last
   commit `ad0ac50` (2026-06-03); no `*music*` node touched after 6/5. So the "good 6/5"
   music and the "bad now" music run the **same prompt code**. The variable that changed is
   the **engine** (musicgen → SA3, default flipped 2026-06-03 in `_otr_engine_profiles.py`:
   `"music": ("stable_audio_3", "musicgen", "stable_audio_music")`).
2. **The prompts are musicgen-shaped.** `compose_music_prompt(meta, cue)` builds:
   `"<mood terms>, evokes <setting>, <period descriptor>, <cue character>, instrumental only, no dialogue, no vocals"`.
   Example: `"minor mode, unresolved tension, evokes derelict station, slow atmospheric build, introduces the scene, ends on a sustained chord, instrumental intro, instrumental only, no dialogue, no vocals"`.
   It is **abstract mood + compositional-direction language** with **NO genre, NO named
   instruments, NO tempo/BPM, NO key**. MusicGen tolerates this; Stable Audio is trained on
   tag-style prompts (genre, instrumentation, BPM, mood, production) and a **negative prompt**.
3. **SA3 sampler inputs are hardcoded** in `eng_stable_audio_3.py::generate_clip`:
   `KSampler(model, seed, steps=100, cfg=6.0, "dpmpp_3m_sde_gpu", "exponential", pos, neg, latent, denoise=1.0)`;
   **negative prompt is empty `""`**; `ConditioningStableAudio().append(pos, neg, seconds_start=0.0, seconds_total=dur)`.
   Checkpoint = `stable_audio_3_small_music.safetensors` (the **SMALL** music model); text
   encoder t5gemma; sample_rate 44100.
4. **Durations are very short:** `CUE_DURATIONS = {opening:12, closing:8, interstitial:4}` sec.
   Stable Audio's `seconds_total` timing conditioning strongly shapes structure; very short
   total-seconds (esp. 4s) is a known weak spot (clips can read as unstructured texture/noise
   rather than "music").
5. **Two music paths may diverge (verify item, not a blocker for this roundtable):** the theme
   node `OTR_StableAudioTheme` lists `_LEGACY_FIRST_FALLBACK = ("musicgen","stable_audio_music")`
   (musicgen = "legacy byte-identical default") while the engine-profile `music` slot defaults
   to SA3. The coder must confirm which engine the SAVED `otr_scifi_16gb_full.json` actually
   uses for the music slot — but regardless, the operator wants SA3 made good.
6. **Determinism contract:** per-cue seed via `_seed_to_int64(music_seed_base, slot)`, run under
   `deterministic_inference(seed, warn_only=True)`. The seed-int is the determinism carrier
   (KSampler builds its generator internally). There is **no "temperature"** in a diffusion
   sampler — the operator's "temp" maps to **cfg + sampler/scheduler + steps**. Keep per-seed
   determinism within a render.

## 2. Hard constraints (must hold)
- **Audio SPINE is FROZEN:** byte-identical master + mux-LAST; `test_audio_byte_identical` stays
  GREEN. This change is **UPSTREAM music-generation only** (prompt builder + SA3 adapter inputs).
- **100% local / offline**, no new pip deps (SA3-native is chosen precisely to avoid the
  torch/numpy Blackwell conflicts that musicgen-adjacent stacks risk). No paid services.
- **Determinism:** seed-keyed, reproducible within a render; OS-entropy cast/style unchanged.
- **UTF-8 no BOM, ASCII-only source** in `_otr_music_prompt.py` (no em-dashes), SFW.
- **Single resident heavy engine ≤ 14.5 GB** (music is light, but teardown stays in `finally`).
- **Meta-brief protocol is the source of music context** — keep reading period/setting/mood via
  `_otr_brief_reader._read_brief_field`; do NOT poke meta directly with a local template.
- Box: Windows, RTX 5080 16 GB. SA3 model on disk: `stable_audio_3_small_music.safetensors`.

## 3. The tunable surface (where a fix can land)
- **A. Prompt grammar** (`nodes/_otr_music_prompt.py::compose_music_prompt`): map the Meta brief
  into SA3-shaped text. Candidate structure: `<genre/style>, <named instruments>, <BPM/tempo>,
  <key/mode>, <mood>, <production/era descriptor>, <cue arc>` + a curated **negative prompt**.
  Period→genre/instrument mapping (e.g. 1950s sci-fi radio → "vintage orchestral sci-fi, theremin,
  brass, timpani, eerie strings, analog tape warmth").
- **B. SA3 sampler inputs** (`eng_stable_audio_3.py::generate_clip`): steps, cfg, sampler,
  scheduler, denoise, and a real **negative prompt** ("low quality, noisy, distorted, dissonant,
  off-key, muddy, vocals, speech"). Decide SA3-correct defaults (Stable Audio guidance differs
  from image-diffusion).
- **C. Timing / duration** (`CUE_DURATIONS` + `ConditioningStableAudio` seconds_total): whether to
  render a longer coherent bed and trim, vs. render exact short cues; how `seconds_total`
  conditioning should be set relative to the wanted clip length.
- **D. Model choice:** is `stable_audio_3_small_music` the quality ceiling? Is a larger/full SA3
  music checkpoint available ungated + local-viable on 16 GB? (Stay no-new-dep.)
- **E. Seed strategy:** keep determinism; optionally best-of-N seeds at author time (NOT per
  render, to preserve reproducibility) — open question for the panel.

## 4. Questions for the panel (converge on a concrete, grounded plan)
1. **Prompt grammar:** what is the highest-leverage SA3 prompt structure for short instrumental
   cinematic/period cues? Exact field order, how many descriptors, genre+instrument+BPM+key —
   and how to derive each from the Meta brief fields we already have (`music_mood_terms`,
   `story_brief_terms.setting/atmosphere`, `gen_params_initial.period_voice.descriptor`,
   `news.script_brief`). Give a concrete template + a worked example.
2. **Negative prompt:** what should SA3's negative prompt contain for clean instrumental music?
3. **Sampler inputs:** SA3-correct `steps / cfg / sampler / scheduler / denoise` for the
   ComfyUI-native Stable Audio graph (CheckpointLoaderSimple → CLIPTextEncode →
   ConditioningStableAudio → EmptyLatentAudio → KSampler → VAEDecodeAudio). Is `steps=100,
   cfg=6.0, dpmpp_3m_sde_gpu/exponential` reasonable, too high, or wrong? Recommended values.
4. **Duration/timing:** is the 4–12s range the root weakness? Should we render a longer coherent
   piece (e.g. 30–45s, SA3's sweet spot) and trim/fade to the cue length? How to set
   `seconds_total` conditioning vs the actual latent length.
5. **Model:** is `stable_audio_3_small_music` enough, or should we move to a larger SA3 music
   checkpoint (must be ungated, local, no new pip dep, 16 GB-viable)?
6. **Determinism-safe quality:** any best-of-N / seed-selection scheme that keeps within-render
   reproducibility? Or strictly single-seed?
7. **Risk check:** what could regress the frozen audio spine or determinism, and how to keep the
   change strictly upstream?

## 5. Out of scope (do NOT propose)
- Reverting to musicgen (operator chose to keep SA3).
- Touching the master mux / audio spine / `-shortest` (frozen).
- New pip dependencies or paid/cloud music services.
- Story-spine / story-pipeline / broader-audio work (PARKED).

## 6. Deliverable from the roundtable
A converged, code-grounded plan: (a) the new `compose_music_prompt` SA3 template + period→
genre/instrument map + negative prompt; (b) the SA3 `generate_clip` input values; (c) the
duration/timing decision; (d) the model decision; (e) the seed strategy; (f) a test/verify plan
(`test_audio_byte_identical` stays green; a determinism guard; an A/B listen plan). Claude grounds
every claim against the real code, discards hallucinations, and hands a coder-window prompt.
