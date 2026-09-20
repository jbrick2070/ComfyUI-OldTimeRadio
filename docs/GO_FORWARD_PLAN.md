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

**41 scenes are vendored: it 13, fr 12, es 6, zh 5, pt 3, ja 2** (read from the
manifest 2026-09-19 late -- this row said 39 earlier that evening, 30 an hour
before that, 16 before that and 4 before that, so read it rather than quote it).
All 95 leads are hunted. Adding an HTML scene is mechanical: read the edition's
own act/scene label into `EDITION_LABELS`, add the row, run
`scripts/otr_vendor_shakespeare.py --write`.

**THE SCANNED LANE NOW EXISTS -- `scripts/otr_vendor_scan.py`.** This row used to
say "the vendor script fetches URLs and has no PDF path at all; that seam is the
work". The seam is built and proven on two Portuguese scenes. It is a SEPARATE
script on purpose: the HTML path reads speakers off the edition's own markup,
and a PDF text layer has none, so the two share nothing at the speaker-marking
step. They do share scene boundaries, so `extract` is imported rather than
reimplemented.
* **THE SPEAKER HALF IS THE OPEN WORK, AND ITS BEHAVIOUR IS RULED, NOT
  DESIGNED.** Today every all-caps run is matched against the scene's Folger
  roster and a run that resolves to nobody is DISCARDED, its text merged into
  the previous speaker. The 2026-09-19 standing ruling (a mangled native text
  beats a clean AI translation; gates refuse MISATTRIBUTION, never IMPERFECTION)
  makes that merge indefensible: an unresolvable cue must be EMITTED as its own
  speaker under the name as printed, left unbound, costing a voice and never
  the dialogue. The Spanish editions abbreviate (`FER .`, `Fer.`, `Prós.`) and
  Clark splits an honorific across the row (`D . PED.`, 25 times in one scene),
  so a perfect `--pages` span currently returns ZERO speeches. The roster stays
  the filter for WRONG-MOUTH attribution; it stops being a filter for dropping
  lines. Grok's detector spec and Terra's runtime trace are the two open briefs.
* **A SCENE IS ADDRESSED BY ITS PAGES. `--pages START-END`, measured off
  `--probe`, is the boundary tool; `--end-label` is the fallback for a volume
  that needs it, not the story.** A line index moves whenever a rule changes,
  and a heading label does not identify a scene at all -- the Spanish volumes
  print `ESCENA PRIMERA .` five, six and ten times each, and `find_label`'s
  exact pass beats a nearer chromed match, which returned King Lear 2.1 for
  1.1 and a 78,766-character Tempest spanning two plays. Measured 2026-09-19:
  all eight Spanish windows return their scene with the COMPLETE English roster
  speaking, and the page window rescues the cell the BOOK misprints (Clark
  labels Twelfth Night 2.5 as `ESCENA II.`). A bad window fails loudly.
* **Reading order was wrong underneath all of it, and an opt-in reader fixes
  it.** `page.get_text()` emits Clark's marginal cues at the page BOTTOM and
  threw `vel!` into the witches' speech in the vendored Macbeth. Commit
  `ae792152` rebuilds rows from glyph baselines (`reading_order="coordinates"`),
  reproduces an image-verified row count exactly and conserves every word on
  1,450 pages; it stays opt-in because it re-admits joined `<folio> <title>`
  rows the furniture walk does not yet know. The hyphen weld (`horri-` +
  `BANQUO`) and the split-heading rejoin (`ESCENA` over `V .`, which fired zero
  times on either Spanish volume) are fixed and shipped; both vendored
  Portuguese scenes were regenerated and the corpus glue census is zero.
* Page furniture is removed by POSITION, never by token -- a play's title is
  usually also a character in it. `tests/test_vendor_scan_furniture.py` holds
  the thirteen shapes that earned their place, four of them neuter-proven, plus
  two recorded KNOWN LIMITS. Read it before changing that file; six of those
  rules were broken again by the fix for the next one.

**A TRANSLATOR'S OWN CHOICE IS NOT A DEFECT (operator 2026-09-19):** *"maybe
some of these foreign translators decide to create a new act or a new speech,
and it's part of their local vernacular and history and culture. Who am I to
judge"*. An unbound label costs a VOICE and never the dialogue, so such a
speaker is NAMED in `_KNOWN_UNBOUND` with its reason and the scene ships. Refuse
a row for OUR extractor failing; never for the translator's editorial hand. Snug
speaks in Rusconi and Zhu and never in Folger; Rusconi labels the
play-within-the-play by ROLE where Folger labels by ACTOR. Both ship.

**HOLD NOTES ARE DATED CLAIMS ABOUT CODE THAT KEEPS CHANGING. RE-MEASURE, DO NOT
READ.** Nine holds were examined closely on 2026-09-19 and **five described a
tool state rather than a source property** -- three of those written that same
day, hours earlier, by the window that then disproved them. The Japanese Romeo
was held for a defect the Aozora rules had already fixed; the Chinese Tempest
for a page that the content bracketing had already made readable; the Italian
Comedy of Errors for a sidecar that exists under a name I constructed instead of
looking up. `scripts/`-adjacent scratch has a re-check harness; run it after any
shared-code change, the whole set costs three minutes.

**WHAT IS LEFT, AND NONE OF IT IS BOOKKEEPING ANY MORE.**

* **27 page-scan cells. NONE is vendor-ready until the speaker half ships;
  eight have PROVEN BOUNDARIES.** This row said "12 ready to run TODAY" on
  the morning of 2026-09-19 and every one of the twelve fell to a probe that
  afternoon. The 8 es cells (king_lear 1.1, midsummer 3.1 and 3.2, much_ado
  2.3 and 3.1, tempest 3.1, twelfth_night 1.5 and 2.5) have measured page
  windows and complete casts at the boundary; they wait only on unbound
  emission above. The 2 pt Tempestade cells (1.2 and 3.1) are blocked on
  Ferdinand printing as `FERNANDO` -- eight characters, refused by the
  PREFIX test, not the length floor -- and on a four-line edge band the
  furniture walk halts on. The 2 pt Midsummer cells resolve through
  imageinfo to a real text layer and then Castilho divides the play into
  QUADROS with continuous scene numbers, the French Hugo non-alignment
  again. Corpus 41 -> up to 53 still, by a longer road.
  **START WITH `--probe`, ALWAYS.** Each volume words its headings differently:
  the Portuguese print `ACTO PRIMEIRO` / `SCENA III`, the Spanish Obras
  dramaticas spells its ordinals (`ACTO SEGUNDO`, `ESCENA PRIMERA`) and holds
  SEVERAL PLAYS in one 472-page file, so an act label alone will not find the
  right play. That volume's text layer also misreads numerals badly -- `ACTO
  11`, `ACTO 111`, `ACTO IF`, `SCENA IT` -- so do not assume clean romans.
  The remaining 15 are image-only (9 ja, 6 zh) and are the real OCR work,
  unblocked by the one-model ruling. See
  [the audit](2026-09-19-shakespeare-vendoring/scan_sources_text_layer_audit.md).
* **A LEAD URL POINTING AT A WIKIMEDIA `File:` / `Archivo:` / `Galeria:` PAGE IS
  NOT A SOURCE, IT IS A PAGE ABOUT ONE.** Nine Spanish rows pointed at
  description pages, so the vendor fetched HTML, read 18 "pages" of wiki chrome
  and reported the volume's only heading as `Actions` -- the sidebar. Resolve
  with the MediaWiki imageinfo API (`action=query&prop=imageinfo&iiprop=url`);
  the `e/ea` path segments are an md5 of the filename and CANNOT be constructed
  by hand. Keep the description page on the row as `description_page`, since
  that is where the licence tag lives.
* **4 Japanese cells have no published source.** Aozora catalogues Tsubouchi's
  Hamlet, As You Like It and Much Ado as in-progress and 404s every card. Not a
  hunt that ran out of ideas -- re-check the author page, not a search engine.
* **2 French Midsummer cells: the editions do not align.** Hugo prints EIGHT
  scenes for the play against Folger's nine and no single Hugo scene equals a
  Folger one; all four neighbours were extracted and read. `EDITION_SCENE_END`
  cannot express this -- it handles an edition FINER than Folger, and Hugo's
  scenes span ACROSS Folger boundaries. Needs a sub-scene rule or another
  edition.
* **1 Italian Tempest 3.1: Maffei prints no second scene heading**, so the act
  runs as one block and any label returns three Folger scenes with a perfect
  cast list. Deliberately has no `EDITION_LABELS` entry; the reason sits where
  the entry would go.


**THE CHECKS THAT EARNED THEIR KEEP, all three now tests and all three proven to
go red.** A label with nothing behind it (found three deleted songs). Site
chrome and zero-width padding (found 8,000 characters of privacy policy being
spoken). Two speeches in a row by one character (found a verse speech whose
label hid inside its own italic run). **Counts found none of them** -- every one
was invisible to speech and speaker totals, which is why
`prove_suffix_only`-style byte diffing against the stored file is the gate
before any push that touches shared extractor code.

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
