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

## Live box -- 5080, do not reset (2026-09-25)

ComfyUI is on port 8000. A 1-act chain
(`C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_chain.ps1`) runs until
08:00 local on 2026-09-25 or the first failure. At handoff the mime leg
(prompt `ea876b5c`) was still rendering; AnimateDiff is next. Do not kill
python, do not interrupt the queue, and do not reboot the server. Receipts
are the 2026-09-25 RENDER entry in the handoff log; the morning review is
Part A of [TEST_WAVE](TEST_WAVE.md). Uncommitted files in the tree are a
partial red-test patch: do not commit them with other work.

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
records the HEAD it ran.

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. *"I'm not expecting anything exact."*

Crash-class and durability-class defects are the work: an uncaught exception, a
live asset written where a sweeper can delete it, an identity that resolves
outside its episode, a machine that silently renders a configuration already
proven wrong. Aesthetic drift is closed.

## 1. SCOPE AND DECIDE

Open forks. One word from him closes a row into section 2, or cuts it.

* **Pre-push hook.** `build_variants --check` plus the sibling matrix checks
  from `.githooks/pre-push`. Changes how both boxes push.

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
  stopped in 29 s at `cast_lock.py:70` -- "engine 'google_tts' is not
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
  never guessed from a name"). `cast_lock.py:1303` (citation fixed
  2026-09-24; was `:1282`) then raises `VoiceCastingError ... NO FALLBACK`
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
  saved widget order byte-for-byte across 63 graphs, and the writer alone is
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

### The registry push -- one batch, at the bottom, by ruling (operator 2026-09-19)

These three ride the same publish and are deliberately last. The floor is
moving -- graphs are still being regenerated -- and a publish is the one action
here that reaches strangers and cannot be taken back.

* **Gallery.** Comfy's scanner globs `*/workflows/*.json` -- ONE level, no
  recursion, no manifest option (`app/custom_node_manager.py`). So listing the
  24 variants means putting 24 JSONs at the top of `workflows/`, which is the
  exact shape that produced SILENT 404s when the pack briefly had two template
  folders; `tests/test_workflow_templates_single_folder.py` exists because of
  it. There is NO functional gap today: the variants ship (`.comfyignore`
  excludes only their `*.md` launch recipes and says "Never widen it to
  workflows/variants/"), they load when dragged, and `apple/MACHINES.md` opens
  with a per-machine table naming the exact file. What is missing is menu
  discovery, and the canonical in the menu already runs on every machine.
* **Delete `v2.0-alpha`.** Unblocked: 2.1.1 is Active and the registry icon
  points at `/main/`. One click, his.
* **Flagged registry versions.** 2.1.5, 2.1.6, 2.3.0 and 2.3.1 are Flagged;
  2.3.2 and 2.3.3 are Active (read 2026-09-25). The API gives no reason --
  there is no `status_reason`, no scan result and no queue position on any
  endpoint. His Discord, not a code change.

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
`47703d7a`'s error-message half stays; its decline-to-pin does not.

### 0b. Asset cleanup after publish -- `asset_cleanup` on the writer (decided 2026-09-25)

Operator: bring the space-saver back, three settings, on the first node where
the choices are made. Default `off`. Grounded against the tree on 2026-09-25;
the old one (`perfect_run_spacesaver`, 2026-05-02, inert from 2026-08-08 when
its host node was ripped, widget removed 2026-09-13) wiped the WRONG episode on
its first day (BUG-LOCAL-014: it found the ledger by an mtime walk). Everything
below that looks like paranoia is that bug.

**The widget.** `OTR_LedgerScriptWriter` gets one COMBO `asset_cleanup` =
`off` | `partial` | `full`, default `off`, appended as the LAST widget: after
`episode_language`, before the `gate_in` socket in INPUT_TYPES; `widgets_values`
slot 36 at the end of node 1; the descriptor at the end of node 1's `inputs[]`
so no link's `dst_slot` moves. Tooltip in plain words: off keeps everything;
partial deletes the sound and the pictures and keeps every text file (ledger,
canon, treatment, manifests, captions, QA); full deletes the whole episode
folder; all three leave `otr/obs` alone. Then `build_variants.py --all` and
`--check`: every matrix row inherits `off` (no row states it; a machine that
wants `partial` gets it as a row edit later, not code). Fix the order
assertion in `tests/test_episode_language_writer.py:39-40` (episode_language ->
asset_cleanup -> gate_in) and whatever pins the writer's widget count.

**The carrier.** The writer stamps `meta.asset_cleanup` beside
`delivery_intent` (`OTR_LedgerScriptWriter.py` ~3629) on BOTH the fresh path
and the replay path: it is a per-run housekeeping choice, not story content,
so `production_ledger._REPLAY_RUN_VOLATILE_META` gains `asset_cleanup` and a
replay never inherits its source's choice. `off` stamps nothing (absent key =
off, the `episode_language` convention). The mux reads the stamp off the same
in-flight ledger `_publication_decision` already loads; an unknown value reads
as `off` with one log line.

**Where it fires.** `OTRMasterAudioMux.mux`, inside the existing try, after
the delivery gate and the janitor sweep, and AFTER `_canvas_preview(final,
obs_copy)` has been computed -- it extracts its poster frame from the
archival `final`, which `full` deletes -- so: compute `ui`, clean, return.
The `ui` text line names the outcome.

**The identity guard.** The episode dir comes ONLY from
`_inflight_episode_for_stem(stem)`: the singleton's dir must be a direct child
of `otr_episodes_root()`, not `_`-prefixed, and the video stem must belong to
that id. No mtime walk, no fallback. Three more facts must agree or it
refuses: `silent_video_path` and `final` both resolve INSIDE that dir;
`obs_copy` resolves OUTSIDE it; the dir is not `otr_obs_dir()` or a parent of
it. It never enumerates the episodes root. It touches exactly one directory,
and only the one this run wrote.

**The publish proof.** Runs only when `obs_copy` is not None (a BLOCKED
episode is never cleaned; the archival final is its only copy), the file
exists and is non-empty -- on top of the stream probes `_publish_to_obs`
already did -- and `_stamp_terminal_paths` did not report failure, so a
`partial` ledger carries `obs_final_path` before anything is removed.

**What each setting does.** `partial`: walk the dir; delete files whose
extension is on a DELETE list (`.mp4 .mkv .mov .webm .wav .flac .mp3 .m4a
.png .jpg .jpeg .webp .gif .exr .npy`); keep everything else -- an unknown
extension is KEPT, because the safe mistake is a kept file; remove dirs left
empty; then stamp `meta.asset_cleanup_receipt` (mode, removed files and
bytes, kept, skipped, when) into the surviving ledger so a reader knows why
the WAVs are gone. `full`: `shutil.rmtree(episode_dir)`. Measured 2026-09-25
on 90.5 GB across 135 media-bearing episode dirs: partial keeps 0.3% of the
bytes, full keeps none. He already does `partial` by hand -- 2,291 folders in
the tree hold only the ledger and text.

**Failure is a report line, never a raise.** The episode is already
published. A locked file (a Windows handle) is skipped and named. Report:
`asset_cleanup <mode>: removed N files (M MB), kept K, skipped S`. Nothing
after it can bring the folder back: `save_ledger_safe` does not mkdir, and the
next writer run binds the singleton to a new ledger.

**Say the consequences out loud (tooltip and RUN.md).** A `partial` or
`full` episode cannot be frozen into a replay bundle afterwards
(`otr_freeze_replay_bundle.py` needs the master WAV and `stills/`,
`portraits/`): freeze first, or run it `off`. The harness and the 5-minute
rule read `otr/obs` and the leg log, never the episode dir. `apple/evidence`
cites dirs the cleanup can never reach.

**Contract docs.** `_otr_janitor.py`'s header says episode assets are never
auto-deleted and calls itself the ONE sanctioned auto-delete: amend it to name
this as the second, chosen per run by the operator, default off. Add the
ruling to [standing rulings](OTR_STANDING_RULINGS.md).

**Tests, no saved fixtures.** A pure planner
`plan_asset_cleanup(episode_dir, mode, *, episodes_root, obs_copy,
video_paths) -> (delete, keep, refusal)` exercised on a `tmp_path` tree:
partial keeps every text file; full lists the dir; off is empty; one test per
refusal (not a direct child, obs inside the dir, a video outside it, obs
missing or empty, unknown mode). A source-inspection test that the mux calls
it after the preview and after the delivery gate. The four workflow guards.

**Registry.** A widget the saved graphs need: 2.3.4 when he says.

**Review.** New capability on the canonical: one outside contrarian on this
design (the ChatGPT prompt handed over 2026-09-25), then the cursor lane plus
Sonnet on the pushed diff.

## 3. TEST

Open by his word (2026-09-25). The wave -- the 5080 overnight review, the 4060
regression on the 8 GB rows, the suite and the Bug Bible -- is
[TEST_WAVE](TEST_WAVE.md). Do not freeze a head for it: each receipt records
the HEAD it ran.

## Already scoped -- do not build

8 GB ship set (its physical wave is [TEST_WAVE](TEST_WAVE.md) Part B) · `scene_coherence_check`
stays inert · no IP-Adapter on AnimateDiff · do not ping the Radeon tester ·
`stable_audio_3` listing `cpu` (re-read the published `--cpu` leg log before
editing the capability test) · **native-language science feeds for SciFi News
Pro -- OUT OF SCOPE (operator 2026-09-19: "forget it, delete, out of scope").
The lane reads the English feed and authors natively, and that is the shipped
behaviour. Do not reopen it as a feed-selection question.**

## Constraints specific to this plan

- Full listener source, no RSS. Cast count is flexible and records requested vs
  actual; the house announcer is excluded from the dramatic cast.
- **We do not chase act count** (operator 2026-09-11), the same rule as word
  count: the value is a request and a run delivers the closest performable
  episode.
- Model checking and a fixed attempt budget only -- no separate chunker, no
  recursive loop.
- An exhausted optional correction still yields a usable ledger; no predictive
  word or duration gate.
- Byline and attribution rules differ for My Story, Original and the adaptation
  banks.
- No replay, migration or re-render project: a saved input means fresh
  generation.

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
