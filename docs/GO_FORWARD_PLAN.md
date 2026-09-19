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

**The pipeline is built, wired and proven end to end (2026-09-18).** A real
translator's words now reach a performed beat: `scripts/otr_vendor_shakespeare.py`
extracts a scene by the EDITION'S OWN markup (speakers, stage business and
footnote chrome are read off the page's own HTML, not guessed from prose),
`nodes/_otr_verbatim_corpus.py` resolves and sha256-verifies it by the ref the
shipping bank actually emits, `nodes/_otr_passage_selector.py` reads its
`NAME:` layout alongside Folger's two (measured inert on all 81 English
sources), and a manifest `speaker_map` bridges each edition label to the
existing English gender ladder -- so a voice lands on the right character and
the printed credits name the translator beside the source licence. Receipt:
[HANDOFF_LOG](HANDOFF_LOG.md).

**THE ORIGINAL DESIGN BELOW WAS SUPERSEDED, NOT COMPLETED.** This row used to
describe an automated `_otr_scene_resolver.py` + alias table + "anchor-matching
alignment by English opening/closing speaker with a confidence score." None of
that shipped. What shipped instead: the operator's own verification pass reads
each lead's page and records the edition's own act/scene label by hand
(`EDITION_LABELS` in the vendor script) -- simpler, and it cannot silently
vendor the wrong scene the way a computed alignment score could.
`alignment_confidence` in the manifest is a stamped constant, not a measured
score; nothing currently computes one. `_otr_scene_resolver.py` was RIPPED on
2026-09-19 (operator: "if it's dead code let's rip it, I approve") -- 324 lines
plus a 415-line test, zero production callers, nothing newly orphaned by its
removal.

**What is actually vendored: four scenes, three languages.**
`it/macbeth 1.3` (Rusconi), `fr/hamlet 1.1` and `fr/king_lear 1.1` (Hugo),
`es/as_you_like_it 3.2` (Marquez). Zero for ja, zh, pt, hi -- entirely
unstarted, not blocked on anything but locating and hand-verifying a lead.

**Next action: vendor the next scene.** The pipeline is proven, so this is now
mechanical per scene, not a design question: open the lead, read off its
edition's own act/scene label (`EDITION_LABELS`), add the row, run
`scripts/otr_vendor_shakespeare.py --write`. Named traps, still live:
LiberLiber's Italian set is Raponi and still in copyright; "A transcribir"
means no text exists; Aozora's canonical text is Shift_JIS with ruby markup;
a `utm_source` parameter is proof the lead was never opened.

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
