# OTR Go-Forward Plan

**ONLY UNFINISHED WORK BELONGS HERE.** When work finishes, its receipt moves to
[HANDOFF_LOG](HANDOFF_LOG.md) or its own evidence folder and the row leaves this
page. A finished prerequisite earns **one clause inside the row that still needs
it** -- never a receipt, never a struck-through or "SHIPPED" row. The test is one
question: *does a row still in this file stop making sense without that
sentence?* No -> cut it.

Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md) first;
this file does not restate them, and does not restate the review or push rules.
**For what already happened -- commits, measurements, receipts -- read
[HANDOFF_LOG](HANDOFF_LOG.md), newest entry first.**

## Live box -- 5080 (2026-09-25, afternoon)

ComfyUI is resident on port 8000 (`scripts/_otr_soak_server_launch.cmd`).
Before any headless run, check `/queue`: if a prompt is running, it is someone's
leg -- do not reset. When the queue is empty, reset per CLAUDE.md section 4. The
tree stays clean: no unpushed patches.

## Operating order (hard)

1. **SCOPE AND DECIDE** -- more than one defensible answer? It lives here, and
   no code is written on it.
2. **CODE** -- one verifiable answer? Build it. An open fork elsewhere does not
   freeze a decided row.
3. **TEST** -- only when 1 and 2 are both empty.

A row that can only be settled by a live leg is not a row: settle it without the
leg, or cut it with the reason written in. Testing never settles a decide or a
code row (operator 2026-09-12 / 2026-09-17).

**Operator 2026-09-25: "the only thing is to regress test."** The regression
wave is open and is [TEST_WAVE](TEST_WAVE.md). The section 1 forks are his words
and do not hold it; the section 2 rows land on their own and each wave receipt
records the HEAD it ran. **Active now:** TEST_WAVE Part B on the 4060, run
against ONE pinned commit the 5080 names (the operator's rule of 2026-09-25: no
moving target), and Part C's Bug Bible run. Everything else here is open and not
this wave's.

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. *"I'm not expecting anything exact."*

Crash-class and durability-class defects are the work: an uncaught exception, a
live asset written where a sweeper can delete it, an identity that resolves
outside its episode, a machine that silently renders a configuration already
proven wrong. Aesthetic drift is closed.

## 1. SCOPE AND DECIDE

Open forks. One word from him closes a row into section 2, or cuts it.

* **Pre-push hook.** No hook exists (`.githooks/` is not in the repo). The fork:
  add one that runs `build_variants --check` plus the widget / link / doc-parity
  / registry-scan tests before a push, or keep relying on the full suite each
  window runs. Changes how both boxes push.

* **Google BYO lane: stills-only by his ruling (67332dc5).** Veo on his paid
  Tier 1 project allows 2 requests a minute and 10 a day per model, and an
  episode makes 16 Veo calls, so no Veo shape fits. The stills shape (Gemini
  image, `still_flat`, Google TTS, Lyria, Flash / Flash-Lite writers) published
  English Hamlet 1.1 in 218 s on 2026-09-19 with no 429s. Veo override:
  `OTR_GOOGLE_VEO_MODEL_ID`. **The fork:** does a Google profile ship as a
  variant (a `SHIPPING_SET` recipe, a matrix row, the README block -- all
  generated) or stay a lane preset applied by hand? If Veo ever returns: 8-second
  clips (one call per beat) or a disclosed opt-in model rotation. One word from
  him.

* **Non-English episodes admit only Kokoro, so Shakespeare-in-translation
  cannot run on the Google lane until he says which languages Google TTS
  may voice.** Measured 2026-09-19: French Hamlet on `google_veo_low_1act`
  stopped in 29 s at `cast_lock.py:62-65` -- "engine 'google_tts' is not
  admitted on a French episode (row engines: ['kokoro']). Kokoro is the
  dance leader day 1." That is his 2026-09-12 ruling working as written
  (`config/episode_languages.json` rows list the admitted engines; English
  admits everything). Gemini TTS speaks every language the switch carries,
  so admitting it is one list entry per row -- but a voice on a language is
  an ear decision, not a config edit (see the French `ff_siwis` day). The
  English Hamlet leg proves the lane on Shakespeare content meanwhile. One
  word per language, or "all", and the rows get it.

* **`google_tts` refuses a cast row with no gender; `my_story` leaves an
  unstated gender empty by design.** Measured 2026-09-19 on the second live
  leg of `google_veo_low_1act`: the canonical's bank is `roll`, the roll
  landed on `my_story`, two of three characters (Stomp, Whiskers) carried
  `gender: None` because the operator's story never states one -- and
  `_otr_my_story.py:26` says so on purpose ("a gender they did not state is
  never guessed from a name"). `cast_lock.py:1332-1336` (re-grounded
  2026-09-25) then raises `VoiceCastingError ... NO FALLBACK`
  for `google_tts`, where the Kokoro
  path takes the gender-agnostic draw and ships. Two rules that are each
  right collide only on this lane. The fork: (a) let `google_tts` take the
  same seeded gender-agnostic draw Kokoro takes when the SOURCE deliberately
  left gender empty (provider voices are gendered, so the pick is a coin
  the seed flips -- deterministic, disclosed in the report line); or (b)
  keep the refusal and have `my_story` say up front that a Google-voiced
  run needs every character's gender stated in the story. (a) ships more
  episodes; (b) never puts a voice on a character the author left open.
  The lane test moved on with `--source-bank original` (LLM-owned cast, the
  40/40/20 draw always sets a gender). One word from him picks it.

* **The Comfy key rides a V1 hidden input, and ComfyUI copies V1 inputs into
  error history.** Since the 2026-09-19 credential rip, nine nodes (writer,
  ShotLock, meta-brief prompt, stills, video, music, both voice nodes, the
  validator) declare `"hidden": {"api_key_comfy_org": "API_KEY_COMFY_ORG"}`.
  That is the V1 channel: `execution.py:230` puts the key into
  `input_data_all`, and when a node RAISES, `execution.py:630-653` serializes
  every input -- hidden included -- into `execution_error.current_inputs`,
  which lands in `/history`. Before the rip only the writer had this
  exposure; now every credit-spending host does. ComfyUI's own partner nodes
  avoid it by being V3 `io.ComfyNode` classes: the same credential arrives
  through `v3_data` (`execution.py:196-209`), which the error path never
  serializes. **Codex r2 named this as the pack-side mitigation and it is
  real.** The fork: convert the nine hosts to V3 (schema order must match the
  saved widget order byte-for-byte across the 25 shipped graphs, and the writer alone is
  ~3,500 lines), OR add one small V3 credential node that stashes the key per
  prompt and is wired into every host's `gate_in` (a canonical-graph change
  plus all variants, and it must return NaN from IS_CHANGED or a cache hit
  serves a stale key). Both are arc-sized. Interim risk, stated exactly: the
  reader needs `/history` on the server, and only a queue that both carries
  a key and raises exposes it. On the desktop boxes the server binds
  127.0.0.1, so that reader is already on the machine. **On the pod it is
  not:** [RUNPOD_INSTALL](RUNPOD_INSTALL.md) launches on `0.0.0.0` for the
  RunPod proxy, so on a pod a key-bearing queue that raises exposes the key
  to anyone the proxy admits (codex r3). Until the V3 shape
  lands, a pod run that spends Comfy credits should be treated as sharing
  its key with the proxy's audience. One word from him picks the shape.

### The registry -- his clicks and his word

* **2.3.5 published 2026-09-25** on his word (the 4060 reinstalls from the
  registry): the writer LLM folder, the queue-time node-pack gate, the
  dependency-doc fix, and everything in 2.3.4. 2.3.4 came back **Flagged**, as
  the scan replica predicted (an internal doc shipped in it); 2.3.5's tree reads
  clean, and `tests/test_registry_scan_oracle_clean.py` keeps it that way.
  Pending until Comfy-Org's scan; the 4060 installs it by picking 2.3.5 in the
  Manager's version picker.
* **Delete `v2.0-alpha`.** Unblocked: 2.1.1 is Active, the registry icon points
  at `/main/`, and the last live pin -- the RunPod bootstrap `curl` in
  RUNPOD_INSTALL -- moved to `/main/` on 2026-09-25. One click, his.
* **Flagged registry versions.** 2.1.5, 2.1.6, 2.3.0 and 2.3.1 are Flagged;
  2.3.2 and 2.3.3 are Active (read 2026-09-25). The API gives no reason. His
  Discord, not a code change.
* **Comfy Desktop's registry Install is intermittent** (PBUG-20260925-01: a
  green "installed" toast and nothing on disk, twice; then fine). Not ours to
  fix; worth telling Comfy-Org, with the 4060's timeline. His Discord.

## 2. CODE -- decided, in order

### 0. Remaining voice-route deletion (hardened 2026-09-24)

The unused `nodes/_otr_voice_route.py` module and the policy / portable-bank
fields that only compile if it exists. Spec:
[ROUTE_DELETION_PLAN](ROUTE_DELETION_PLAN.md) (re-grounded 2026-09-25).
Do not start from the 51b6c146 draft. Do not re-rip `91ad5961`.

### 0a. Windows HF_HOME -- always pin a short root (decided 2026-09-23)

When the models-adjacent pin cannot fit MAX_PATH, do not decline.
Choose a short root (registry if it fits, then models-adjacent if it
fits, then `C:\ComfyUI-Models\huggingface` if that tree exists, else
the huggingface_hub-shaped user cache). Spec:
[HF_HOME_WINDOWS_PIN](HF_HOME_WINDOWS_PIN.md) (re-grounded 2026-09-25).
`47703d7a`'s error-message half stays; its decline-to-pin does not. The
writer no longer downloads into this cache (`8f8ccebb`, the `LLM` folder);
the pin still governs Bark, MusicGen, the visual assets and the provisioner,
and the 162-character tail it sizes for was always the visual one.

### 0b. Wire the talking-face still to the audio-driven lanes (not a rip)

With Kling Avatar removed (a1e5548f) no engine implements
`wants_talking_prompt()`, so the talking still mode -- the director's
`_role_talking`, MetaBrief's `_engine_wants_talking_prompt`, the
`when_engine_talking` still-plan token, `TALKING_PORTRAIT_GEOMETRY` ("face-
forward frontal close-up bust ... the whole face and mouth clearly visible")
and the packs' `portrait_look_talking` -- is always off. The operator: "I
think it uses stills that include a face and lips; most audio-in models do."
MEASURED from the registry, and it narrows the question:
- The LTX 2.5 / MiniMax H3 / cloud LTX 2.5 "audio-in" lanes are
  `audio_conditioned_video`: the picture reacts to the sound, they do NOT
  lip-sync. Their stills are scene stills (macro open, three-quarter beat,
  medium-shot character in a wide 16:9 environment) and their portrait row
  is `never`. The matrix keeps character beats OFF the LTX audio-in lane on
  purpose (a face there would lip-sync to the ambient mix). Talking stills
  would be WRONG for them. Leave them alone.
- HuMo (all four) is the one `audio_driven_face` (true lip-sync) engine. Its
  portrait row is `always`: "three-quarter portrait, full head and face"
  (portrait) or "medium shot, head and shoulders" (16:9) -- not the frontal
  mouth-visible close-up, although its own motion prompt says "keep the
  mouth visible and the head toward camera". HuMo is in NO shipped graph.
DECIDED (operator, same day): RIP the talking mode. "I don't want just a
talking face; LTX 2.5 audio-in can handle talking faces that move and have
action; I don't want to bring it to the brand level of talking face." Remove
`wants_talking_prompt` lookups (director `_role_talking`, MetaBrief
`_engine_wants_talking_prompt` / `_effective_talking_roles`), the
`when_engine_talking` still-plan token, `TALKING_PORTRAIT_GEOMETRY` and the
talking look segment, and the packs' `portrait_look_talking` -- the packs
carry sha256 receipts, so regenerate/re-pin them and grep tests for the
hashes first. HuMo keeps its current portrait framing. Scene stills with
action stay the path for every audio lane.
CORRECTION (operator: "audio-in, we do feed clean audio"): the audio-in lane
already gets each character's CLEAN own voice on a character beat;
`render_driver._uses_ambient_master_audio` excludes character-face beats from
the master slice (2026-06-26), and only lineless announcer/music bookends use
the mix. The `otr_8gb_ltx25_native_audio_in` display text in
config/workflow_matrix.json ("CHARACTER BEATS STAY ON THE FOLEY LANE ON
PURPOSE: an audio-in lane on a character face would lip-sync to the ambient
master mix") is STALE -- correct it, and ask the operator whether that graph
should now route character beats to the audio-in lane too.

### 0b2. One lane per graph, and drop "native" from the names (operator 2026-09-25)

- RULE: an audio-in graph uses audio-in for all three roles, a foley graph
  foley, a mime graph mime. Only `workflows/otr_8gb_ltx25_native_audio_in.json`
  breaks it (character_visual is `ltx25_native_foley_16gb`); switch it to
  `ltx25_native_audio_in_16gb` and correct that row's stale display text.
  Prove character lip-sync on one leg of that graph.
- "native" meant "ComfyUI's own loaders, not GGUF"; GGUF is gone, so the word
  is noise. Rename in ONE commit: engine ids `ltx25_native_{foley,mime,
  audio_in}_{16gb,24gb}` and `ltx25_native_foley_blackwell`, and workflow files
  `otr_8gb_ltx25_native_{foley,mime,audio_in}.json` and
  `otr_24gb_native_foley.json`. Plain rename: no alias, no "renamed to"
  message, no back-compat of any kind (operator: "don't worry about back
  compat"). Shortcodes, tests, docs, preflight tables and the matrix follow.
- DECIDED (operator 2026-09-25: "let's try keeping the engines if they
  work, and we can test them"): KEEP the four engines no workflow selects
  (`ltx25_native_foley_blackwell`, `ltx25_native_mime_24gb`,
  `ltx25_native_audio_in_24gb`, `minimax_h3_audio_in`) and PROVE each one:
  a 1-act leg per engine, hand-picked on the canonical, on hardware that
  fits it. One that fails is fixed or removed in the commit that says which.
  THE BAR (operator, same day): kept only if proven to work, not GGUF, and it
  downloads itself -- "if they required extra work, no". Measured against
  `_otr_visual_assets.planned_downloads`:
  * `ltx25_native_foley_blackwell` -- auto-downloads (NVFP4 transformer + the
    shared LTX 2.5 VAEs, upscaler and Gemma encoder). Needs a big Blackwell:
    prove it on a RunPod Blackwell pod.
  * `ltx25_native_mime_24gb`, `ltx25_native_audio_in_24gb` -- auto-download
    (int8 transformer + the same shared files). Prove on a 24 GB pod; one pod
    session can prove all three.
  * `minimax_h3_video` and `minimax_h3_audio_in` (dropdown names
    `h3_low_video`, `h3_low_audio_in`; no workflow selects either) -- KEEP
    and MAKE THEM AUTO-DOWNLOAD (operator: "keep them as long as they are
    auto download"; "MiniMax 3 is popular, so people have it"). Today they
    are "manual": ~59 GB fetched only by
    `scripts/otr_fetch_lane_weights.py minimax_h3`. Measured 2026-09-25: the
    files come from `Comfy-Org/MiniMax-H3`, which is PUBLIC and NOT GATED
    (anonymous 302 on the pinned revision 4cc1d817), licence "other"; Comfy's
    own gallery ships several MiniMax H3 templates. Work: move those pinned
    WeightSpecs (revision, size, sha256 already in the fetcher) into
    `_otr_visual_assets._SOURCES`, cover both engines in `_COVERED` with their
    weight tokens so `planned_downloads` names them, drop the dropdown's
    "manual" row, and re-prove both on a leg that downloads its own weights
    (h3_low_video is already proven on 16 GB; h3_low_audio_in "fits").
- Generated docs should show a music column if they do not already.

### 0c. Portability fixes (Composer audit 2026-09-25, each claim grounded)

Found by asking "what assumes the developer's machine?" after the models
root defect (PBUG-20260925-03). Ranked by what a stranger would hit:

1. `llm_policy.py:51,76-78` rejects `cuda:N`, but the writer advertises
   `gpu:N` and `device_options.resolve_device` returns `cuda:N`: a second-GPU
   pick crashes the writer build.
2. `_otr_bark_lib.py:201` calls `torch.cuda.empty_cache()` unguarded: first
   Bark load crashes on a Mac or CPU-only torch.
3. `_otr_model_catalog.py` `effective_quant_policy` reads ROCm's "cuda" as
   NVIDIA and turns the AMD graph's `quant_policy="none"` into NF4, which
   bitsandbytes cannot run on ROCm. Use `device_options.vendor()`.
4. `prestartup_script.py:126-144` pins HF_HOME to `<comfy>/models/huggingface`
   by file depth before `_otr_hf_env` runs. Folds into item 0a above.
5. Kokoro (`eng_kokoro.py:72-77`, `_otr_kokoro_voice_prefetch.py`) joins
   `TTS/KokoroTTS` onto `models_dir`; register a `TTS` category and use
   `model_type_dir`. Move both together.
6. `_otr_model_loader.py` hardcodes CUDA device 0 (~1030, ~1289) and tells
   Accelerate the CPU has 64 GiB (~556). Thread the resolved device; size
   the CPU lane from available RAM.
7. Writes inside the pack folder: cloud cache
   (`cloud_media_backend.py:446-450`), Chatterbox/Dia stderr
   (`eng_chatterbox.py:109`, `eng_dia.py:114`, outside any try), and
   `otr_runtime.log` (`_vram_log.py`, `story_orchestrator.py`,
   `video_engine.py`). Move to the output-tree state/tmp dirs.
8. Lower: Chatterbox/Dia default venv path is Windows-only (`Scripts/`);
   `video_engine.py:2342-2346` falls back to `~/Documents/ComfyUI/output`;
   NVFP4 is preferred among installed files without a hardware check;
   `config/otr_windows_extra_model_paths.yaml` names `C:/ComfyUI-Models`.
Every fix measures the 5080 unchanged (CLAUDE.md section 0B).

### 0d. Gallery and JSON hygiene (template diff + Grok review, 2026-09-25)

Field-by-field diff of `otr_canonical.json` against the official
`video_wan2_2_14B_s2v` template, then Grok's read-only pass over all 25
graphs against the 555 official templates and frontend 1.52.7, every claim
grounded here: same litegraph 0.4 shape, same top-level keys, valid links,
groups and widget descriptors. What is left is what a stranger SEES first.
In order:
- **Stamp every node with `cnr_id` + `ver`.** Official templates carry
  `properties.cnr_id` / `properties.ver` on every node (51 of 62); the
  frontend's `getCnrIdFromNode` reads it, and it is how "Install Missing
  Nodes" finds the pack. Ours: 1 of 21 nodes has `cnr_id`, none has `ver`.
  Stamp all 21 with `comfyui-old-time-radio` and the pyproject version, in
  the canonical and in build_variants; a test pins `ver` == pyproject. Verify
  on the 4060 whether `ver` 2.3.6 makes Manager fetch the Pending version.
- **Drop `extra.info`.** It says `version: "2.0-alpha"`; nothing in nodes/,
  scripts/, tests/ or js/ reads it. pyproject is the version authority.
- **A friendly category name through `/i18n`.** Core serves each custom
  node's `locales/<lang>/main.json`; the frontend localizes the pack category
  from `templateWorkflows.category.<folder name>`. Ship `locales/en/main.json`
  with BOTH keys -- `ComfyUI-OldTimeRadio` (git clone) and
  `comfyui-old-time-radio` (registry install) -- reading "Old-Time Radio".
  Per-workflow titles are NOT localizable for custom packs in this frontend
  (they are the filename stem), so the stems stay; do not rename files.
- **Standard palette pass (cosmetic, his eye).** Measured: 1,421 of 4,635
  official nodes are coloured, almost all from five short pairs
  (`#222/#000`, `#432/#653`, `#232/#353`, `#322/#533`, `#223/#335`); 587
  groups use `#3f789e`. Ours are custom hexes on a shared `#3B3D42`. Map the
  five stage colours onto standard pairs, keeping the stages distinct.
- **Story Writer collapsed on load (his eye).** 520x1760 with 35 widgets is
  a wall; `flags.collapsed` keeps every value and position, and the Start
  here note says "expand 1 - Story Writer". Never split the node.
- **Retune `extra.ds`** (saved viewport) once the layout settles.
- **Rejected or deferred, with the reason:** an `index.json` in workflows/
  (the glob would list it as a template named "index"); renaming files for
  prettier titles (stems are coupled to matrix ids, stamps, headless runs,
  tests); replacing `seed_mode` / `request_seed` with stock seed controls
  (the deterministic request hash is a feature); a blanket `shape: 7` sweep;
  moving validator stamps out of widgets (readers and writers would all
  move); `properties.models` links; subgraphs; `widgets_values_named` (a
  second copy to keep in sync); unique UUIDs per variant (no consumer needs
  it); `localized_name` cleanup; pretty-printing variants (dev polish only);
  hand-stamping `extra.frontendVersion` (553 of 555 carry it, nothing loads
  on it, and a fixed value goes stale at once).

#### The "Start here" note

Comfy's official templates all carry an on-canvas Markdown note; ours has
none, so a stranger sees 21 boxes and no instructions. One note in the
SCRIPT / START HERE group: type a premise or pick a bank, press Queue,
where the episode lands, where to report problems. Canonical + variants
regenerated; widget/link audits as usual.

### 0e. OTR app mode -- design first (operator 2026-09-25)

ComfyUI's app view (`extra.linearMode: true`, `extra.linearData.inputs` =
`[node_id, widget]` pairs, `outputs` = node ids; supported by frontend
1.52.7) shows a graph as a simple form. The operator's input list: act
count, the models, asset cleanup, the Lemmy roll, language, story bank,
visual style; and for the writer LLMs, the cloud A/B slots as well
(operator: "don't forget the OpenRouter A/B, Comfy A/B and Google A/B").
Mapped: Story Writer (node 1) `act_count`, `creative_writing_model`,
`technical_model`, `openrouter_slot_a_model`, `openrouter_slot_b_model`,
`comfy_slot_a_model`, `comfy_slot_b_model`, `google_api_slot_a_model`,
`google_api_slot_b_model`, `asset_cleanup` (the space saver: off / partial / full), `lemmy_cameo`,
`episode_language`, `source_bank`, `visual_style`. "The models" means ALL
of them (operator: "all LLM models, video, voice TTS, music"): Video
Director (node 87) `announcer_video_model`, `music_video_model`,
`character_video_model` and the three `*_image_model` picks; Cast Lock
(node 80) `char_voice_engine`, `announcer_voice_engine`; Theme Music
(node 83) `engine`, and `music_style` for My Story; Silent Composite
(node 84) `upscale_engine` (operator: "upscaler").
LAYOUT, top to bottom (operator 2026-09-25): the story choice first --
`source_bank`, `source_ref`, `visual_style`; then the shape --
`act_count`, `num_characters` (no creativity dial: it was removed the same
day, each model samples at its maker's baseline), `episode_language`,
`lemmy_cameo`, `asset_cleanup`; then every
model picker; and AT THE BOTTOM the My Story fields, which only the My
Story bank reads -- `episode_title`, `custom_premise`, `story_characters`,
`story_plot`, `story_setting`, `story_author`, and Theme Music's
`music_style`.
THE RULE (operator: "basically almost everything in our variant matrix"):
the app shows the matrix's USER-CHOICE deltas -- `features.act_count`,
`features.num_characters`, the `llm.*_model` and cloud slot picks,
`role_overrides.*`, `slot_overrides.*` -- plus the story knobs the matrix
does not vary (cleanup, Lemmy, language, bank, visual style, upscaler).
It HIDES the matrix's machine-tuning deltas (`llm.device`,
`llm.quant_policy`, `llm.vram_ceiling_gb`, `audio.voice_device`,
`image.dtype_policy`, `video.dtype_policy`, `video.device_policy`,
`render.canvas_*`, `seed_policy.*`): the per-machine graph already set them
for that card. Generate the app's input list from the matrix plus that
extras list in `build_variants.py`, so a new matrix knob cannot be missing
from the app. Output: node 14 (Mux and Publish). Open questions for one design
round before code: canonical itself or a separate `otr_app.json`; whether
the premise/title text belongs; how variants inherit it; whether an older
frontend ignores the metadata harmlessly.

### 0f. Custom-node best practice gaps (audit 2026-09-25, docs.comfy.org + core)

Already followed: IS_CHANGED, VALIDATE_INPUTS, hidden inputs, logging over
print, WEB_DIRECTORY with a tested JS extension, soft_empty_cache and
interrupt checks on the heavy nodes, a static dependency list, no
eval/exec/runtime pip, and `.comfyignore` stripping dev files. Gaps, in
order:

1. `DESCRIPTION` on every registered node (1 of 24 today) and tooltips on
   every widget, so ComfyUI's "?" panel is not blank.
2. Live UI status from the long nodes (`PromptServer.instance.send_sync` or
   ProgressBar) so a minutes-long render does not look stalled.
3. `requires-comfyui` in `[tool.comfy]`, set to the oldest core that runs
   the canonical graph -- measured, never guessed; too high blocks installs.
4. A thumbnail for `workflows/otr_canonical.json` in the gallery. The
   audit's "rename to example_workflows/" half is rejected (Fable's call,
   OTR_STANDING_RULINGS 2026-09-25: `workflows/` stays).
5. Maybe: a GitHub Actions job running the CPU suite on push/PR (torch
   makes it heavy).
6. `Banner` in `[tool.comfy]`: the ART IS IN THE REPO (operator's own,
   2026-09-25): `assets/otr_banner.jpg`, a 21:9 crop (the registry spec)
   of `assets/otr_banner_master.webp`. Add
   `Banner = "https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/main/assets/otr_banner.jpg"`
   in the SAME commit as the next version bump -- editing pyproject.toml
   auto-publishes, so it never goes in on its own. Served live from `main`
   like the Icon. The Icon stays the animated GIF (his call), though the
   spec asks for square and at most 400x400; it has been accepted as is.
   (GitHub social preview: done 2026-09-25, the same art at 2:1, uploaded by
   the operator and confirmed live through the API.)
Parked, large: loading models through `comfy.model_management` so ComfyUI
can evict them for other packs (touches every model load; multi-box proof),
and the V3 `comfy_api` node schema (24 classes, workflow-adjacent).

## 3. TEST

Open by his word (2026-09-25). The wave -- the 5080 overnight review, the 4060
regression on the 8 GB rows, the suite and the Bug Bible -- is
[TEST_WAVE](TEST_WAVE.md). Do not freeze a head for it: each receipt records
the HEAD it ran. Standing 2026-09-25 afternoon: Part A done (the chain: 4
queued / 4 published, no failure; A3 and A4 settled by one
`otr_16gb_animatediff` leg on the 5080 that published in 28 minutes). Part B
on the 4060: B2 PASSED through the GUI on a fresh
2.3.4 install (`c2f14301`); B3's weight auto-download proof PASSED and the
leg then FAILED on the missing AnimateDiff-Evolved pack (PBUG-20260925-02,
fixed `02758478`); the operator is wiping the 4060 for a fresh start, so the
sequence there is: install, queue `otr_8gb_animatediff` WITHOUT the pack and
log the t=0 refusal (the live verify), install the pack, B3, B4. Measured
2026-09-25 from the live server's `/object_info` `python_module`: AnimateDiff-
Evolved is the ONLY third-party node pack any shipped engine needs; every
class LTX 2.5, LTX 8 GB, HuMo, MiniMax H3 and the mesh stage ask for is
ComfyUI core, so a missing one of those means an old ComfyUI, and the gate
now says "Update ComfyUI" for them. Part C: the
suite has run at every push today (12 inherited reds); the Bug Bible
regression against the pack has not.

## Parked

**Vendored Shakespeare translations -- PARKED 2026-09-19 (operator: "cut our
losses so we can ship").** 43 scenes ship (it 13, fr 12, es 6, zh 5, pt 5,
ja 2). The full reopen record -- the scanned-lane speaker-emission gap, the
per-cell holds, the markup lessons and the open cell list -- is in git at
`d167f0ab:apple/GO_FORWARD_PLAN.md` (section 1 and section 2 row 1). Rights
are not a gate (standing rulings).

Not tracked further here: unqualified installed-family combinations, H3
policy receipts, cfg promotion
comparisons, the AMD scoped pod and platform acceptance, cloud billing opt-in
routing, operator-parked casting/adaptation ideas, OTR-Lite after v2, the release
runway, the missing `device_options` test module, regenerating
`apple/MODEL_ASSET_INDEX.md`, and writer widget-label cosmetics.
