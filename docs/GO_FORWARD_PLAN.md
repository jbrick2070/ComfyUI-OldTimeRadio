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

## Operating order (hard)

1. **SCOPE AND DECIDE** -- more than one defensible answer? It lives here, and
   no code is written on it.
2. **CODE** -- one verifiable answer? Build it. An open fork elsewhere does not
   freeze a decided row.
3. **TEST** -- only when 1 and 2 are both empty.

A row that can only be settled by a live leg is not a row: settle it without the
leg, or cut it with the reason written in. Testing never settles a decide or a
code row (operator 2026-09-12 / 2026-09-17).

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. *"I'm not expecting anything exact."*

Crash-class and durability-class defects are the work: an uncaught exception, a
live asset written where a sweeper can delete it, an identity that resolves
outside its episode, a machine that silently renders a configuration already
proven wrong. Aesthetic drift is closed
([ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md)).

## 1. SCOPE AND DECIDE

Open forks. One word from him closes a row into section 2, or cuts it.

* **Pre-push hook.** `build_variants --check` plus the sibling matrix checks
  from `.githooks/pre-push`. Changes how both boxes push.

* **The Google BYO lane ran end to end for the first time on 2026-09-19 and
  what stops it now is his Google project's Veo quota, not code.** Five live
  legs of `google_veo_low_1act` (draft preset, `--source-bank original`),
  each one stopping on the next real wire fact and each fixed the same day:
  thinking tokens billed against `max_output_tokens` (904fd0f0), TTS MIME
  parameters (c2f688de), Veo 429 with no retry (11dac456). Two episodes are
  in `otr/obs/` -- `fire_clay_20260919_105543__anim__gveo__gimg__gtts__orig__gasa__lyra_final.mp4`
  (2 of 8 beats got Veo clips) and `brass_glass_20260919_111547__rfrc__...`
  (0 of 8) -- so Gemini writer, Google TTS, Lyria, Gemini image and the
  publish path are proven; Veo rendered four clips total today and then
  429'd through a full 75 s backoff on every shot, which a per-minute cap
  would not survive. **Afternoon, on the rotated paid key, the same
  request to every Veo model in the catalog (backoff off):**
  `veo-3.1-fast-generate-preview` ACCEPTED, `veo-3.1-generate-preview`
  ACCEPTED, `veo-3.1-lite-generate-preview` HTTP 429. So the quota is
  PER MODEL and only the lite preview -- the profile's pin -- is spent
  today; the key is fine. Every other engine passed a live one-call smoke
  on that key (writer Flash + Lite, TTS, image, Lyria, Omni). The override
  is `OTR_GOOGLE_VEO_MODEL_ID` (`_selected_model`, eng_google_veo_video.py);
  the test leg runs on fast.
  **His quota page (project ArchivalFlow, paid Tier 1) then settled it
  exactly:** every Veo model gets **2 requests per minute and 10 per day**;
  fast and lite sat at 10/10 by early afternoon, full at 2/10. An episode
  makes 16 Veo calls (8 beats x 2 jump segments), so no Veo lane fits
  Tier 1 as shaped; TTS (10/min), image (100/min) and Lyria (150/min) are
  nowhere near a limit. Tier 2 wants $250 spent over 30 days, and he is
  not begging for it. **His ruling: Google is a STILLS-ONLY lane** --
  `config/profiles/google_still_1act.json` (67332dc5): Gemini image for
  stills, `still_flat` to composite them (CPU, no models), Google TTS,
  Lyria, Flash / Flash-Lite writers, no Veo. Dry run resolves all 17
  overrides, and the first live leg -- English Hamlet 1.1 -- published
  `troubled_mind_20260919_124119__anim__stfl__gimg__gtts__sspr__gasa__lyra_final.mp4`
  in 218 s with every beat on its own Gemini still, no floors, no 429s:
  **the Google lane works every day at Tier 1 in this shape.** Cheapest
  stills for the next legs: `OTR_GOOGLE_IMAGE_MODEL_ID=gemini-3.1-flash-lite-image`
  (in the catalog, 1K only). His Ultra plan does not reach the API (it is
  Flow's consumer pool; the G-Labs tool spends it by logging in as him,
  which is a no); the honest Veo path is the Increase Requests tab on his
  quota page. If Veo ever comes back it is his call between 8-second clips
  (one call per beat, 8/day fits) and a disclosed opt-in model rotation,
  which collides with the no-fallback law as a default.
  **The fork, once video renders:** which profile ships -- `google_veo_low_1act`
  (2 chars, 100-frame clips) is the cheap candidate -- and promoting it means
  a variant + launch recipe (`SHIPPING_SET`), a tier-matrix row and the
  README block, all generated. Until then the lane stays a lane preset an
  operator applies by hand. One word from him picks the profile, and his
  dashboard says when Veo will answer again.

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
  never guessed from a name"). `cast_lock.py:1282` then raises
  `VoiceCastingError ... NO FALLBACK` for `google_tts`, where the Kokoro
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

* **Every remaining Shakespeare cell now has a named next step, and three
  held rows were held on a WRONG DIAGNOSIS.** Measured 2026-09-19. Of 73
  unvendored cells: 40 needed a URL (not OCR), 23 are page scans, 8 have
  text we could not parse, 2 are excluded. The URL hunt closed 25 of the 40
  in one evening -- 12 Italian and 2 Spanish found, 9 Hindi closed as
  ADAPTATIONS (Sitaram renames the cast), 2 Japanese still open. **OCR is
  unblocked by the 2026-09-19 ruling** (one model counts as verbatim, see
  the standing rulings), so the 23 scans are now production work rather
  than a question.
  **THE PATTERN WORTH MORE THAN ANY SCENE:** three hold notes each said its
  edition was unmarked or unparseable, and all three were wrong. Tsubouchi
  "sets stage business inline with no markup of its own" -- he marks it in
  `div.jisage_N`. pt.wikisource "no speaker pattern fits" -- every speaker
  is a bare all-caps `div.tiInherit`. Zhu Shenghao "breaks no paragraph
  before a speaker" -- every speaker opens a `<p>`. Each note then froze its
  row for weeks. **Read the markup before believing a note about the
  markup**, including one written here.
  **WHAT IS BUILT:** the Japanese and Portuguese rules ship (f446d484,
  reviewed, with a regression fixture). **WHAT IS SPECIFIED AND NOT BUILT:**
  the Chinese rule -- [the diagnosis](2026-09-19-shakespeare-vendoring/chinese_edition_diagnosis.md)
  names the markup, counts 65
  marked speeches against 0 the pipeline currently sees, and flags that
  `_marked_name`'s `len(name) < 2` guard would silently drop every
  single-character Chinese speaker (波, 衮, 蒂 are most of the cast).
  **WHAT IS BROKEN AND HELD:** the Italian set extracts at a healthy-looking
  142 speeches / 13 speakers with `ESCONO GLOC. ED EDM`, `CUCULLUS NON FACIT
  MONACHUM` and `M. O. A. I` among its "speakers" -- a direction, Feste's
  Latin joke and Malvolio's letter. Rusconi needs his own rule; the generic
  name shapes over-claim on this edition. pt/hamlet returns 25 of the 56
  speeches the page marks; es/hamlet cuts at 923 chars. Each row carries its
  measured reason. A speech count cannot see any of this.

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
  not:** `scripts/otr_pod_runtime.sh:458` launches with `--listen 0.0.0.0
  --enable-cors-header` and `docs/RUNPOD_INSTALL.md` documents reaching it
  through the RunPod proxy, so on a pod a key-bearing queue that raises
  exposes the key to anyone the proxy admits (codex r3). Until the V3 shape
  lands, a pod run that spends Comfy credits should be treated as sharing
  its key with the proxy's audience. One word from him picks the shape.

* **The canonical word counter is ASCII-only, and fixing it moves every
  episode's numbers.** `WORD_RE` in `_otr_text_metrics.py` is
  `[A-Za-z][A-Za-z0-9'...]*`, so an accented word counts as several: `révéler`
  is THREE words, `corazón` and `naïve` are two. Measured on the vendored
  corpus, French and Spanish inflate 5.1-7.8% (`king_lear_1_1` reads 2,979
  where it holds 2,748); Italian is -0.7%, because its accents sit word-final
  and merely truncate rather than split.
  **The fork is the blast radius, not the regex.** A Unicode-aware class is one
  line, but it is SHARED code and it does not stay on the translation lane:
  **69 of 157 English source files change too**, up to -2,430 words on one
  Gutenberg text, because English prose carries `naïve`, `café` and accented
  proper names.
  **Nothing is gated on it, which is why this is a decide row and not a bug
  fix.** Word targets are a REQUEST, not a gate (standing directive), and no
  validator refuses an over-cap beat -- grepped: `BEAT_WORD_HARD_MAX` is read
  only by `chunk_speech`/`beat_cost`, which share the same counter and
  therefore still agree with each other, so selection can never cost a speech
  differently from how execution cuts it. The visible symptom is three chunks
  in the whole corpus running 81 words against a cap of 80.
  So: a real defect whose only effect is that ledger word counts on accented
  text are wrong, against a change that rewrites the numbers on every English
  episode. One word from him settles it.

### The registry push -- one batch, at the bottom, by ruling (operator 2026-09-19)

These three ride the same publish and are deliberately last. The floor is
moving -- graphs are still being regenerated -- and a publish is the one action
here that reaches strangers and cannot be taken back.

* **Gallery.** Comfy's scanner globs `*/workflows/*.json` -- ONE level, no
  recursion, no manifest option (`app/custom_node_manager.py`). So listing the
  21 variants means putting 21 JSONs at the top of `workflows/`, which is the
  exact shape that produced SILENT 404s when the pack briefly had two template
  folders; `tests/test_workflow_templates_single_folder.py` exists because of
  it. There is NO functional gap today: the variants ship (`.comfyignore`
  excludes only their `*.md` launch recipes and says "Never widen it to
  workflows/variants/"), they load when dragged, and `apple/MACHINES.md` opens
  with a per-machine table naming the exact file. What is missing is menu
  discovery, and the canonical in the menu already runs on every machine.
* **Delete `v2.0-alpha`.** Unblocked: 2.1.1 is Active and the registry icon
  points at `/main/`. One click, his.
* **Flagged registry versions.** 2.1.5 and 2.1.6 are Flagged; Manager serves
  2.1.4. The API gives no reason -- there is no `status_reason`, no scan result
  and no queue position on any endpoint. His Discord, not a code change.

## 2. CODE -- decided, in order

### 1. Vendored public-domain Shakespeare translations

**RIGHTS ARE NOT A GATE (operator 2026-09-18 evening):** *"I don't want to
waste anything in rights I'm not publishing these commercially."* Nothing is
refused on a date, no rights research happens, and translator/publication
years are recorded as row DATA only. See
[standing rulings](OTR_STANDING_RULINGS.md). Fidelity is a separate axis and
still governs: a translation made from an intermediary is still refused.

The pipeline is proven end to end and **16 scenes are vendored** across es, fr,
it, ja and pt, with 95 leads hunted and every unvendored cell carrying a named
next step in `leads.json`. Adding a scene is mechanical: read the edition's own
act/scene label into `EDITION_LABELS`, add the row, run
`scripts/otr_vendor_shakespeare.py --write`.

What remains is two per-edition extractor rules, both specified and neither
built. State, diagnoses and the measured counts live in
[2026-09-19-shakespeare-vendoring](2026-09-19-shakespeare-vendoring/).

**The Chinese rule.** Speaker is 1-4 CJK characters plus U+3000 at a paragraph
head; block business is a centred div opening with a fullwidth black bracket;
headings match `^第.*[幕场場]$` and must be exempted or the scene anchors break.
The page marks 65 speeches across 14 characters in Midsummer 3.1 and
`mark_speakers` claims 0. **The trap, found independently by two models:**
`_marked_name` rejects `len(name) < 2` and most of this cast is one character
(波, 衮, 蒂), so the rule can mark every speaker correctly and still drop them
all downstream.

**The Italian Rusconi rule.** Abbreviated marks (`Orl.`, `Ber.`) followed by a
period at a paragraph head. The generic name shapes claim these correctly and
also claim stage directions and spoken text. The obvious fix is measured
harmful: anchoring on `<p>` took tempest from 142 speeches to 6. Twelve act-page
URLs are verified and recorded.

Both touch shared extractor code every edition runs through, so each needs a
blast-radius check across all 16 vendored scenes before its push.

**Then production OCR of the 23 page-scan cells**, unblocked by the 2026-09-19
ruling that one model's OCR counts as verbatim. Parallel model work, not coding.

Live traps: LiberLiber's Italian set is Raponi and still in copyright;
"A transcribir" means no text exists; Aozora's canonical text is Shift_JIS with
ruby markup; a `utm_source` parameter is proof the lead was never opened.

## 3. TEST -- only after 1 and 2 are empty

**Operator 2026-09-17:** *"TEST WAVE AFTER CODING."* When section 1 is empty and
section 2 is empty, freeze ONE hash into the `WAVE HEAD:` line of
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md) and turn four machines
loose on it at once -- the 5080, the 4060, the Mac and a RunPod box, each running
the real canonical workflow, all reporting home. What they owe is already in
[COVERAGE_OWED](2026-09-11-four-machine-test-wave/COVERAGE_OWED.md); nothing
needs planning, it needs starting. Until then do not freeze a head and do not
book a qualification leg (two heads were cut early on 2026-09-11 and both had to
be withdrawn). The morning after begins in `otr/obs/`, not the editor.

## Already scoped -- do not build

8 GB ship set (held until the physical 8 GB wave) · `scene_coherence_check`
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

[GO_FORWARD_ARCHIVE](GO_FORWARD_ARCHIVE.md) holds the tombstones: unqualified
installed-family and GGUF opt-in combinations, H3 policy receipts, cfg promotion
comparisons, the AMD scoped pod and platform acceptance, cloud billing opt-in
routing, operator-parked casting/adaptation ideas, OTR-Lite after v2, the release
runway, the missing `device_options` test module, regenerating
`docs/MODEL_ASSET_INDEX.md`, and writer widget-label cosmetics.
