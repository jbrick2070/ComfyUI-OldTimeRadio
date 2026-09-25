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

### 0b. Remove the Kling Avatar engine (operator 2026-09-25: "remove it")

No shipped graph or matrix row selects `cloud_kling_avatar`; it was a
hand-pick cloud lane only, and its still-plan framing text had no
production reader (the last red in the suite,
`test_still_plan_layer2_parity`). Rip it fully: adapter code in
`eng_cloud_video.py`, registry row, partner-node pin row, shortcode,
slug preflight, provisioner route, generated docs, and every test that
names it. Grep for `kling` after; nothing may remain but history.

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

### 0d. "Start here" note on the canonical canvas

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
`google_api_slot_b_model`, `asset_cleanup`, `lemmy_cameo`,
`episode_language`, `source_bank`, `visual_style`; plus the video/image
picks on node 87 and voices on node 80 if "the models" includes them. Output: node 14 (Mux and Publish). Open questions for one design
round before code: canonical itself or a separate `otr_app.json`; whether
the premise/title text belongs; how variants inherit it; whether an older
frontend ignores the metadata harmlessly.

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
