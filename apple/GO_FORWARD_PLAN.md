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

* All five forks that stood here were closed by his word on 2026-09-25 and
  are now section 2 rows 0h-0k (the pre-push hook is cut: "12 minutes of
  tests for every commit, probably not needed"; the suite at every push
  stays the guard).

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
  mouth visible and the head toward camera". HuMo is in NO shipped workflow.
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
the mix. RESOLVED via 0b2 Step 1 (2026-09-25): the stale `otr_8gb_ltx25_audio_in`
display text was corrected and the row now routes character beats to the
audio-in lane, same as announcer and music.

### 0b2. One lane per workflow, and drop "native" from the names (operator 2026-09-25, HARDENED)

**Step 1 -- DONE 2026-09-25 (commit 66272d49).** `otr_8gb_ltx25_native_audio_in`'s
`character_visual` was moved onto the same engine as `announcer_visual` and
`music_visual`, and the row's `display_name` no longer claims the audio-in
lane is unsafe for character faces (false per
`render_driver._uses_ambient_master_audio`, which already excludes
character-face beats from the ambient slice, 2026-06-26). Regenerated,
`--check` clean. STILL OWED: a live `otr_8gb_ltx25_audio_in` leg (1 act)
proving a character beat's lips track that character's own line, not the
mix -- code-complete and reviewed, not yet proven on hardware.

**Step 2 -- DONE 2026-09-25, mechanical rename, plain (no alias, no
back-compat -- a saved workflow naming an old engine id just stops
resolving).** Dropped "native" from the 7 engine ids
(`ltx25_native_foley_16gb` -> `ltx25_foley_16gb` and the other six the same
way) and the 4 workflow file stems it touched
(`otr_8gb_ltx25_native_foley.json` -> `otr_8gb_ltx25_foley.json`, `_mime`
and `_audio_in` the same way, `otr_24gb_native_foley.json` ->
`otr_24gb_foley.json`; the other 21 workflows keep their filename and only
change the engine ids they reference), plus the 3 lane-fetch weight-family
tokens in `otr_fetch_lane_weights.py` (`ltx25_native_16gb` ->
`ltx25_16gb`, `_24gb`, `_blackwell` likewise) that the plan's own
measurement grep caught but did not list by name. Shortcode VALUES
(`n16f`/`n24f`/`nbwf`/`n16m`/`n24m`/`n16a`/`n24a`) were left untouched --
only the dict keys renamed, so old published filenames keep their meaning.
`test_ltx25_native_lane_contract.py` renamed to `test_ltx25_lane_contract.py`.
Re-grep of the six search strings across the repo (excluding `otr/`, the
two append-only logs, and git history) found zero remaining hits meaning a
live engine id; every survivor is a generic English "native" (native
resolution, native audio track, `native_frame_count` the ledger field,
Python identifiers like `_native_dit`) or a reference to an
already-retired, unregistered id kept for historical accuracy. `--check`
clean, full suite green.
- DECIDED (operator 2026-09-25: "let's try keeping the engines if they
  work, and we can test them"): KEEP the four engines no workflow selects
  (`ltx25_foley_blackwell`, `ltx25_mime_24gb`,
  `ltx25_audio_in_24gb`, `minimax_h3_audio_in`) and PROVE each one:
  a 1-act leg per engine, hand-picked on the canonical, on hardware that
  fits it. One that fails is fixed or removed in the commit that says which.
  THE BAR (operator, same day): kept only if proven to work, not GGUF, and it
  downloads itself -- "if they required extra work, no". Measured against
  `_otr_visual_assets.planned_downloads`:
  * `ltx25_foley_blackwell` -- auto-downloads (NVFP4 transformer + the
    shared LTX 2.5 VAEs, upscaler and Gemma encoder). Needs a big Blackwell:
    prove it on a RunPod Blackwell pod.
  * `ltx25_mime_24gb`, `ltx25_audio_in_24gb` -- auto-download
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
- **Per-workflow music column: ALREADY EXISTS, nothing to build.** Operator
  2026-09-25: "the matrix should say what music model is being used" --
  it does. `apple/MACHINE_MATRIX.md` (generated by
  `scripts/otr_machine_matrix.py`) carries one row per workflow, all 24,
  grouped by VRAM tier, with `video | voice | music | image | status`
  columns: e.g. `otr_mac16_low | viz_camera | kokoro | stable_audio_3 |
  sd15 | shipping`. An earlier draft of this row proposed a second such
  table in `MACHINES.md`; cut, one source is enough. Two small items it
  surfaced, one commit: (1) `MACHINES.md`'s "Which graph do I open?" heading
  becomes "Which workflow do I open?" (official language; the string lives in
  `scripts/otr_dropdown_matrix.py`, the MACHINES.md writer) and gets
  one sentence pointing at that table for "what each workflow uses"; (2) the
  generator's tier headings read "N experimental profile(s), N shipping" --
  the retired word (operator: "THERE ARE NO PROFILES"); the strings are
  `scripts/otr_machine_matrix.py:510` (the VRAM-band heading) and `:531` (the
  "draft profile(s)" summary); change both to "workflow(s)" -- the word is WORKFLOW, ComfyUI's own term (operator, same day: "we have 24 workflows, can we just call them workflows"; "I want to use official language") -- and regenerate. The canonical has
  no matrix row and so no line there; its picks are its saved widgets
  (music `stable_audio_3`).
  Music engines as shipped, measured from the matrix the same day: every
  local workflow -- the four Mac ones and the AMD one included -- selects
  `stable_audio_3`; the five cloud workflows select `sonilo`. The feasibility
  table marks `stable_audio_3` **proven** on Mac 16 GB (musicgen is only
  "measured" there), so SA3 on Mac is confirmed, not assumed. The AMD cell
  is still `?`: `otr_amd_still` selects SA3 unproven, which is the
  experimental-AMD position already on record.

### 0h / 0i / 0j. The Google lane -- code DONE 2026-09-25, live legs owed

- **0h** ("yes we should have a Google JSON with stills"): row
  `otr_google_still` ships -- Gemini image stills composited by
  `still_flat`, Google TTS, Lyria, Flash / Flash-Lite writers, CPU, no
  download, key `OTR_GOOGLE_API_KEY`. Copied from the preset that
  published English Hamlet 1.1 in 218 s on 2026-09-19. README,
  CLOUD.md, VIDEO_MODELS.md and WRITERS.md name it instead of the deleted
  `google_still_*` / `google_veo_low_*` presets. The writer's Google rows
  and slot models are now listed with or without a key (as OpenRouter's
  and Comfy's always were), so the workflow opens clean on a keyless
  machine and Queue stops on the named missing-key message.
- **0i** (b2cf4251): every language row admits `google_tts` after
  `kokoro`; the 30 Gemini voices are tagged with all eight languages.
- **0j** (dca81735): a character with no stated gender is cast on
  `google_tts` by the episode seed; a stated gender it cannot serve still
  refuses by name.

OWED, one leg each on the 5080 (1 act, `otr_google_still`): English
(downloads nothing, publishes); one non-English language (0i); a
`my_story` cast with an unstated gender (0j).

### 0k. The Comfy credential rides the V3 channel: one small credential node (operator: "I dunno what is best practice" -- this is it)

Best practice, stated plainly: in the app, the user's ComfyUI login IS the
credential and nothing in the workflow or the environment should carry a key;
headless (a pod, `scripts/otr_api.py`) there is no login, so the key
travels per prompt as `extra_data.api_key_comfy_org`, which the submitter
already does. The defect is only how our nine credit-spending hosts RECEIVE
it: a V1 `"hidden": {"api_key_comfy_org": "API_KEY_COMFY_ORG"}` input,
which `execution.py:630-653` serializes into `/history` when a node raises
-- on a pod bound to `0.0.0.0` that is anyone the proxy admits. Comfy's own
partner nodes avoid it as V3 `io.ComfyNode` classes, whose `v3_data` the
error path never serializes. Work (the smaller of the two shapes): one V3
credential node, `OTR_ComfyCredential`, that reads the key from `v3_data`,
stashes it per prompt in a process-local map keyed by prompt id, and
returns a token the nine hosts take on `gate_in`; the hosts drop their V1
hidden input and fetch the key from the map by their prompt id. It returns
NaN from `IS_CHANGED` so a cache hit never serves a stale key. This is a
canonical-workflow change (one node, nine links) plus all 25 workflows
regenerated -- the section 0 three-part rule applies. Prove: queue a
credit-spending workflow with a deliberately failing host and read
`/history`: the key must not appear. Until it lands, a pod run that spends
Comfy credits is treated as sharing its key with the proxy's audience.

### 0l. Official language sweep: just "workflow" (operator 2026-09-25: "no graph, no profile, no variant, just workflow"; "I want to use official language")

In everything a person reads, the 25 shipped JSON files are workflows: "the
canonical workflow" and the "per-machine workflows"; the script-submitted
form is the "API-format workflow" (ComfyUI's code calls it an API prompt).
Measured 2026-09-25 (case-insensitive word counts: variant / profile / graph):
`README.md` 1/4/28, `apple/RUN.md` 0/0/4, `apple/INSTALL.md` 0/2/7,
`apple/MACHINES.md` 0/2/6, `apple/MACHINE_MATRIX.md` 0/21/4,
`apple/DROPDOWN_MATRIX.md` 0/5/6, `apple/LAUNCH_RECIPES.md` 1/1/29,
`apple/EXTENDING.md` 4/0/20, `apple/WRITERS.md` 2/0/17, `__init__.py`
(the boot banner and its comments) 3/0/11. Rules for the sweep:
- Generated docs (`MACHINES.md`, `MACHINE_MATRIX.md`, `DROPDOWN_MATRIX.md`,
  `LAUNCH_RECIPES.md`, `ENGINE_MATRIX.md`) are fixed IN THEIR GENERATORS
  (`scripts/otr_dropdown_matrix.py`, `scripts/otr_machine_matrix.py` --
  `:510` and `:531` print "profile(s)" -- and `scripts/build_variants.py`'s
  recipe/doc strings), then regenerated. Never hand-edit a generated doc.
- Hand-written docs and the boot banner are edited directly.
- "graph" stays only where it means the canvas itself (subgraph, "graph
  screenshot", litegraph); every hit that means a shipped file becomes
  "workflow". Read each hit; do not blanket-replace.
- Code identifiers and file names are plumbing and stay (`build_variants.py`,
  `VARIANTS_DIR`, `role_overrides`, matrix key names). Only strings a person
  reads change.
- Tests that pin doc wording follow in the same commit (a generator parity
  test will fail until the generator and doc agree -- that is the check).
Done when the same grep over those ten files finds no retired word meaning a
shipped file, the suite is green, and `build_variants.py --check` is clean.
Sonnet-safe: mechanical, one commit.

### 0c. Portability fixes (Composer audit 2026-09-25, each claim grounded)

Found by asking "what assumes the developer's machine?" after the models
root defect (PBUG-20260925-03). Ranked by what a stranger would hit. The
top 3 are HARDENED below; 4-8 stay a reviewed backlog, one commit each.

**Items 1-3 DONE 2026-09-25** (92388af7): `cuda:N` is an admitted writer
device, Bark's first load no longer calls CUDA on a Mac or CPU-only torch,
and the platform quant bakes NF4 only when the vendor is NVIDIA.

**Backlog, 4-8 (grounded, not yet hardened into steps):**
4. `prestartup_script.py:126-144` pins HF_HOME to `<comfy>/models/huggingface`
   by file depth before `_otr_hf_env` runs. Folds into item 0a above.
5. Kokoro (`eng_kokoro.py:72-77`, `_otr_kokoro_voice_prefetch.py`) joins
   `TTS/KokoroTTS` onto `models_dir`; register a `TTS` category and use
   `model_type_dir`. Move both together.
6. `_otr_model_loader.py` hardcodes CUDA device 0 (~1030, ~1289) and tells
   Accelerate the CPU has 64 GiB (~556). Thread the resolved device; size
   the CPU lane from available RAM. DEVICE HALF DONE 2026-09-25 (1c0b1dd4):
   item 1 turned a second-GPU pick from a crash into a writer silently
   loaded on GPU 0, so load_llm now probes and places on the policy's CUDA
   ordinal; c4fcae80 makes that GPU the CURRENT device for the load and
   the teardown wash. The 64 GiB CPU-lane half is still open. Known
   telemetry-only gap (Sonnet review): the VRAM_RESET / VRAM_SNAPSHOT
   lines `_run_with_timeout` writes during generation still read GPU 0's
   counters for a `cuda:N` writer. Generation itself is correct (inputs
   go to `model.device`); fix only if a two-GPU user reports it.
7. DONE 2026-09-25 (c04aad46 + 2c952de1, Cursor): the cloud media cache,
   the sidecar stderr files and `otr_runtime.log` write under the output
   tree; the old pack-folder billing ledger is copied forward once. The
   episode's telemetry line reads the live log only -- the frozen
   pack-folder log is shown by `otr_tail_logs` as history, never read as
   current (a Sonnet review reproduced a stale model name leaking in).
8. Lower: Chatterbox/Dia default venv path is Windows-only (`Scripts/`);
   `video_engine.py:2342-2346` falls back to `~/Documents/ComfyUI/output`;
   NVFP4 is preferred among installed files without a hardware check;
   `config/otr_windows_extra_model_paths.yaml` names `C:/ComfyUI-Models`.
LOW PRIORITY, operator ruling 2026-09-25 ("no biggie" / "doesn't matter
much") -- both are off-default picks only; do not re-raise:
- Gemma 4 12B on AMD bakes NF4 ON PURPOSE since a06ee44d (Cursor): the
  bitsandbytes docs list gfx1201, and a Sonnet review confirmed it. Qwen on
  AMD stays full precision (item 3) because that is what the Radeon tester's
  published episode proved. A Mac now loads Gemma full precision.
- Bark's first load on Metal reclaims no MPS pool (`_unload_bark` does).
  Kokoro is the default voice on every local lane.
Every fix measures the 5080 unchanged (CLAUDE.md section 0B): items 1-3
touch shared code with a fallback branch that never fires today on a
single-GPU NVIDIA box -- print the resolved value before/after in the
commit message, same as the models-root and per-type work earlier today.

### 0d. Gallery and JSON hygiene (template diff + Grok review, 2026-09-25)

Field-by-field diff of `otr_canonical.json` against the official
`video_wan2_2_14B_s2v` template, then Grok's read-only pass over all 25
workflows against the 555 official templates and frontend 1.52.7, every claim
grounded here: same litegraph 0.4 shape, same top-level keys, valid links,
groups and widget descriptors. What is left is what a stranger SEES first.
In order:
- **Stamp every node with `cnr_id` + `ver` (HARDENED).** Measured
  2026-09-25: canonical has 21 nodes, exactly 1 (`OTR_WorkflowValidator`)
  carries `properties.cnr_id: "comfyui-old-time-radio"`; none carries
  `ver`. Official templates carry both on most nodes; the frontend's
  `getCnrIdFromNode` reads `cnr_id` (falls back to `aux_id`), and it is how
  "Install Missing Nodes" finds the pack. Work: (1) in
  `scripts/build_variants.py::build_variant` (`:162+`), after loading the
  canonical and before stamping the validator's own widgets, add `cnr_id`
  and `ver` to every node's `properties` dict, pulled from a pinned
  `PACK_CNR_ID = "comfyui-old-time-radio"` and the LIVE pyproject version
  (parse `pyproject.toml`'s `version = "..."` line at build time, do not
  hardcode a copy that drifts -- see the DEPENDENCIES.md generator lesson
  from earlier today, PBUG-adjacent); (2) apply the same stamp to
  `otr_canonical.json` itself by hand once, then regenerate; (3)
  `scripts/build_variants.py --check` gains a check that every node in
  every committed file (canonical + 24 workflows) has `cnr_id ==
  "comfyui-old-time-radio"` and `ver == <live pyproject version>`; (4) a
  new test asserts the same against the real files. Verify on the 4060
  whether a `ver` bump makes Manager's Missing-Nodes flow fetch the exact
  Pending version rather than Active -- report only, this does not gate
  the commit.
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
  second copy to keep in sync); unique UUIDs per workflow (no consumer needs
  it); `localized_name` cleanup; pretty-printing the per-machine workflows (dev polish only);
  hand-stamping `extra.frontendVersion` (553 of 555 carry it, nothing loads
  on it, and a fixed value goes stale at once).

#### The "Start here" note

Comfy's official templates all carry an on-canvas Markdown note; ours has
none, so a stranger sees 21 boxes and no instructions. One note in the
SCRIPT / START HERE group: type a premise or pick a bank, press Queue,
where the episode lands, where to report problems. Canonical + per-machine workflows
regenerated; widget/link audits as usual.

### 0e. OTR app mode -- design first (operator 2026-09-25)

ComfyUI's app view (`extra.linearMode: true`, `extra.linearData.inputs` =
`[node_id, widget]` pairs, `outputs` = node ids; supported by frontend
1.52.7) shows a workflow as a simple form. The operator's input list: act
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
LAYOUT, top to bottom (operator 2026-09-25 evening -- REPLACES the earlier
story-choice-first order): `episode_language`, `act_count`,
`num_characters` (no creativity dial: it was removed the same day, each
model samples at its maker's baseline), `lemmy_cameo`, `source_bank`,
`source_ref`, `visual_style`; then the video lanes (announcer / music /
character video model) with `upscale_engine` beside them; the three image
models; the two voice engines; Theme Music's `engine`; the writer LLMs
(`creative_writing_model`, `technical_model`) followed by their six cloud
slots (OpenRouter / Comfy / Google, A and B); and AT THE BOTTOM the My
Story fields, which only the My Story bank reads -- `episode_title`,
`custom_premise`, `story_characters`, `story_plot`, `story_setting`,
`story_author`, and Theme Music's `music_style`; and LAST, after My Story
and not inside it, `asset_cleanup` (the space saver; operator: "default
to off ... under My Story at the end"). It already defaults to
`off (keep everything)` in code and in all 25 shipped workflows. It sits
after the My Story block rather than in it because My Story fields apply
only to the My Story bank and cleanup applies to every episode. Placement
of `source_ref`, `lemmy_cameo` and `upscale_engine` was proposed by the
driver, not named by the operator -- confirm at design time.
ORDER IS FREE (verified in frontend 1.52.7 source, not assumed): the form
renders `extra.linearData.inputs` in list order -- `appModeStore`
preserves it on load, `useResolvedSelectedInputs` maps it as stored, and
nothing sorts it; each entry is `[node, widget]` (or `[node, widget,
{height}]`), so inputs from different nodes interleave freely and a
multi-line My Story box can be given a height. The app builder also lets
a person drag to reorder.
THE RANDOMIZER the operator asked for beside bank and style ALREADY EXISTS: each
dropdown's first option is a roll (`roll (any eligible bank)`,
`roll (any style)`, `_otr_rolls.py`), so in the app it is that dropdown's
top choice, not a new switch.
THE SCROLL WORRY (operator: "worried people won't see the other
dropdowns"): about 26 dropdowns then 7 My Story text fields. My Story
LAST keeps every dropdown above the fold; the cost is that a My Story user
must scroll to the bottom. App mode CANNOT group or collapse inputs
(verified: frontend 1.52.7 `InputWidgetConfig` is `{height?, description?}`
on a flat list), so there is no fold to put anything above -- design for
scroll. The "My Story: fill in the fields at the bottom" hint belongs in
that app-only `description`, not the node tooltip, which also shows on
the canvas.
DECIDE BEFORE ANY CODE (contrarian review 2026-09-25, each claim checked
against the files):
- HOST. `apply_profile` deep-copies the whole canonical, `extra` included,
  so `extra.linearMode: true` on `otr_canonical.json` would open EVERY
  gallery card as an app (the frontend maps that boolean to its initial
  mode). Put it on the generated cards only, or on a separate
  `otr_app.json` -- never on the canonical the operator edits on the canvas.
- CARD OR FORM -- DECIDED 2026-09-25 (operator: "Both"). The per-machine
  cards open story-only (the card IS the lane); ONE separate advanced app
  shows the pickers and says an out-of-memory is the user's to accept.
  The trade-off as it was put: the list above exposes the
  video, image, voice, music and writer pickers, while the per-machine
  tuning (quant, VRAM ceiling, canvas, device) stays hidden and tuned for
  the card's own lane. So an 8 GB user can pick a lane the card cannot run
  and meet an out-of-memory -- the same outcome as changing that dropdown on
  the canvas today, but now on the stranger-facing surface. Either the card
  is the lane (the app shows story choices only) or the form is (pickers
  shown, and the out-of-memory accepted and said so).
- THE INPUT LIST IS HAND-ORDERED, NOT GENERATED. `key_indicators` in the
  matrix omits the six cloud slots and includes the hidden tuning keys, so
  "generate from the matrix plus extras" both drops what the operator named
  and adds what should hide. An explicit allow-list in the operator's order;
  a test can still check that no NEW matrix user-choice key is missing.
- `source_ref` MUST NOT SIT UNDER A BANK LEFT ON THE ROLL. The shipped
  default is `roll (any eligible bank)`, and a pinned `source_ref` with the
  roll is refused by name (`OTR_LedgerScriptWriter.py` ~2206). The frontend
  has no conditional fields, so either leave `source_ref` out of the app or
  place it where the pairing is obvious and describe it.
- UNIQUE WORKFLOW IDS become a dependency. All 25 shipped files carry the
  same `id` (09a7142b-...), and the frontend treats a same-id load as the
  same active workflow. 0d rejected unique ids ("no consumer needs it");
  app mode may be that consumer -- prove or fix before shipping apps.
THE RULE (operator: "basically almost everything in our workflow matrix"):
the app shows the matrix's USER-CHOICE deltas -- `features.act_count`,
`features.num_characters`, the `llm.*_model` and cloud slot picks,
`role_overrides.*`, `slot_overrides.*` -- plus the story knobs the matrix
does not vary (cleanup, Lemmy, language, bank, visual style, upscaler).
It HIDES the matrix's machine-tuning deltas (`llm.device`,
`llm.quant_policy`, `llm.vram_ceiling_gb`, `audio.voice_device`,
`image.dtype_policy`, `video.dtype_policy`, `video.device_policy`,
`render.canvas_*`, `seed_policy.*`): the per-machine workflow already set them
for that card. Generate the app's input list from the matrix plus that
extras list in `build_variants.py`, so a new matrix knob cannot be missing
from the app. Output: node 85 (`OTR_MasterAudioMux`, titled "14 - Mux and
Publish", which is where "node 14" came from). It returns STRINGs, and
whether the app pane plays the episode from it is UNPROVEN -- check live.
Open questions for one design
round before code: canonical itself or a separate `otr_app.json`; whether
the premise/title text belongs; how the per-machine workflows inherit it; whether an older
frontend ignores the metadata harmlessly.

### 0f. Custom-node best practice gaps (audit 2026-09-25, docs.comfy.org + core)

Already followed: IS_CHANGED, VALIDATE_INPUTS, hidden inputs, logging over
print, WEB_DIRECTORY with a tested JS extension, soft_empty_cache and
interrupt checks on the heavy nodes, a static dependency list, no
eval/exec/runtime pip, and `.comfyignore` stripping dev files. Gaps, in
order:

1. DONE 2026-09-25 (cf75a95e, agy): `DESCRIPTION` on all 24 nodes and a
   tooltip on all 184 inputs; `tests/test_node_descriptions.py` pins both.
2. DONE 2026-09-25: `nodes/_otr_shared/node_progress.py` draws ComfyUI's bar
   over the three loops that showed nothing -- the writer's per-beat compose,
   the local voice lines, the scene sequencer. Nodes already showing a
   sampler bar or the Comfy partner heartbeat get no second bar; the direct
   Google API voice gets it on the fan-out path too (Cursor review). Cancel
   lands at the next item boundary. STILL SILENT: `google_image` stills and
   `google_lyria` music (cloud-side, no heartbeat) -- same fix shape if the
   Google workflow's bar is wanted there. LIVE CHECK OWED: watch it move.
3. `requires-comfyui` in `[tool.comfy]`, set to the oldest core that runs
   the canonical workflow -- measured, never guessed; too high blocks installs.
4. DONE: `build_variants.py --all` writes a gallery thumbnail beside the
   canonical and every workflow, and `--check` fails on a missing one. The
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
