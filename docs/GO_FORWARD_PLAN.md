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

* **A1. Canonical writer on an 8 GB card.** Canonical saves Qwen 3.5 4B,
  `llm_quant_policy` `none`, ceiling `10.0`; a dropped node defaults to
  `bnb_nf4` / `14.5`. Leave canonical as the 16 GB graph and point 8 GB users
  at the variant, or retune both.
* **A5. 16 GB foley / mime without GGUF.** Official LTX 2.5 safetensors do not
  fit 16 GB (measured). `otr_16gb_video`, `otr_16gb_foley` and `otr_16gb_mime`
  still load the Q3 GGUF; cloud deluxe already ships `cloud_ltx25_foley_plus`.
  Stay on Q3, move those three to cloud, or drop them.
* **Gallery.** Comfy lists one graph; the 21 shipping variants sit in
  `workflows/variants/`, which the template scanner does not read. Promoting
  them changes what the gallery shows tomorrow.
* **Pre-push hook.** `build_variants --check` plus the sibling matrix checks
  from `.githooks/pre-push`. Changes how both boxes push.
* **Delete `v2.0-alpha`.** Unblocked: 2.1.1 is Active and the registry icon
  points at `/main/`. One click, his.
* **Flagged registry versions.** 2.1.5 and 2.1.6 are Flagged; Manager serves
  2.1.4. The API gives no reason. His Discord, not a code change.
* **Native-language science feeds for SciFi News Pro.** The lane reads the
  English feed and authors natively. Which feeds, and whether the dossier
  extraction stays English, is his call before any code.

## 2. CODE -- decided, in order

### 1. Vendored public-domain Shakespeare translations

**RIGHTS ARE NOT A GATE (operator 2026-09-18 evening):** *"I don't want to
waste anything in rights I'm not publishing these commercially."* Nothing is
refused on a date, no rights research happens, and translator/publication
years are recorded as row DATA only. See
[standing rulings](OTR_STANDING_RULINGS.md). Fidelity is a separate axis and
still governs: a translation made from an intermediary is still refused.

**Decided 2026-09-18** (*"I want the best pack available"*): a scene ships a
real translator's words -- ~~when they clear both the US test (first published
before 1931) and life+70 (translator died before 1956)~~ **(withdrawn that
evening, see above)**; verse preferred
where a public-domain verse translation exists; scene-level transcription is
allowed; no coverage requirement -- any scene without vendored words keeps the
model translation that ships today. The rule this replaces ("died before 1944")
was a conservative bound, not a legal test.

Spec, inventory and the gate's field list:
[2026-09-18-fidelity-lane-translation](2026-09-18-fidelity-lane-translation/shakespeare_corpus_spec_v2.yaml).

**The gate is built and has run** (`scripts/otr_shakespeare_corpus_gate.py`,
`nodes/_otr_verbatim_corpus.py`, leads in
`config/source_banks/shakespeare/translations/leads.json`, report in the
evidence folder).

**What it measured, 2026-09-18 -- NOT ONE LEAD URL IS A SCENE, BUT SEVEN ARE
THE RIGHT WORK.** Seven pages carry two to five act headings, i.e. the whole
play (`ACTE PREMIER` / `ACTO PRIMERO` / `ACTO PRIMEIRO`): both fr Macbeth
leads, all three es leads, pt Hamlet and hi As You Like It. One -- `Teatro
completo di Shakspeare` -- is the collected-works index with no act heading
at all. The Aozora URL is the 図書カード rather than the text, and the
archive.org URL is the details page rather than the scan.

Two corrections worth keeping, both caught in review rather than by the
author. An early, laxer gate called eight leads READY on page chrome -- the
same error the spec convicts v1 of, committed again in miniature. Then the
strict gate reported "target scene headings not found" for pages that DO
carry headings, because it read only digits and single-letter romans while
the 19th-century convention is the ordinal WORD; the plan said those pages
were heading-less landing pages, and that was wrong. Both are fixed, and the
verdicts now distinguish "wrong page" from "right work, wrong granularity".

**`nodes/_otr_scene_resolver.py` EXISTS AND IS DELIBERATELY NOT WIRED YET --
this is the row that says what it is waiting for** (2026-09-18; the repo's
"wire it in the same change or write the row" rule). It resolves `act.scene`
to a span and refuses rather than guessing, and its arrival already paid for
itself: building it against the real cached French Macbeth exposed
PBUG-20260918-07 in `_labelled`, where the bare English `act` matched inside
the French word `action` 24 times and a five-act play measured as two.

A FOURTH BLOCKER WAS ITS OWN CORRECTNESS, and this row denied it until the
Fable review of 2026-09-18 read the import line: the module did
`from nodes import _otr_verbatim_corpus`, and inside a running ComfyUI
`nodes` is COMFYUI'S OWN registry module, not this package. It resolved only
under pytest, where the test file puts the repo root on `sys.path` first, so
24 green tests proved the helper and nothing about the wiring -- the
2026-09-07 defect class exactly. Fixed to the relative import every sibling
uses, verified by importing it the way `__init__.py` does. **The sentence
that used to stand here claimed none of the blockers was the resolver's own,
and that was false.**

THREE THINGS STILL BLOCK THE WIRING, none of them the resolver's correctness:
1. **It must not hand chrome to the performance.** Its span stops at the next
   HEADING, so an end-of-act marker such as `FIN DU PREMIER ACTE.` rides
   inside the extracted text. Wired as-is that becomes a SPOKEN line -- the
   exact defect PBUG-20260918-04 already shipped once. A test asserts the gap
   on purpose and says to delete itself when the trim lands.
2. **The coordinate may not exist in the edition.** François-Victor Hugo
   numbers scenes CONTINUOUSLY with no act divisions (`SCÈNE I.` through
   `SCÈNE XXIV.` on the cached Macbeth, zero act headings). Asking that page
   for "1.3" is not a miss to fix in the resolver; it is the alias table
   below, and wiring before it exists would silently vendor a WRONG scene
   that looks right.
3. **Nothing it would feed is built.** There is no vendored tree and no
   plan-step read, so a wired resolver would have no consumer today.

So the wiring lands WITH the alias table, not before it.

**Next action: resolve a lead to its SCENE.** For the seven whole-work pages
that is a RANGE inside a text already fetched and already rights-cleared --
locate the act heading, then the scene heading under it, then the next scene
heading. Per host for the rest: a Wikisource subpage (and the `action=parse`
wikitext endpoint the spec prefers, where speaker labels are template-wrapped
and structurally detectable); Aozora's zipped Shift_JIS file rather than the
card; archive.org's `_djvu.txt` rather than the details page. Then the alias table and anchor-matching alignment
(editions renumber scenes, so a scene resolves by its English opening and
closing speaker with a confidence score, never by counting headings), then
the vendored tree `config/source_banks/shakespeare/translations/<iso>/` with
one normalised `NAME:`-labelled scene file per scene and a manifest row
(translator, death year, first publication, transcription licence, source
URL, revision id, raw sha256, verdict, confidence), then the plan-step read
that prefers a READY scene and falls back to the model translation, with the
receipt naming which and the credit naming the translator. Phases: fr + it
(weigh Carcano verse against Rusconi prose), then es, ja, zh, then pt + hi.

Named traps: LiberLiber's Italian set is Raponi and still in copyright;
"A transcribir" means no text exists; Aozora's canonical text is Shift_JIS with
ruby markup; a `utm_source` parameter is proof the row was never opened.

One contrarian on the manifest shape and the first gate output before any
ingestion.

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
editing the capability test).

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
