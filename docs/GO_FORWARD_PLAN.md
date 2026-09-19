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
* **`nodes/_otr_scene_resolver.py` -- rip it or wire it (new 2026-09-18).**
  Built for an automated act/scene extraction + confidence-scored alignment
  design that the vendored-Shakespeare row below never ended up using -- the
  shipped pipeline reads each edition's own act/scene label by hand instead
  (`EDITION_LABELS`), which cannot silently vendor the wrong scene the way a
  computed alignment score could. Zero production callers, same as before
  this note was written. Either it still has a role once vendoring scales
  past hand-verified leads (fewer manual label lookups, more leads per
  session), or it is a verified-dead symbol per the repo's rip-or-wire rule.
  His call; a grep receipt either way before acting.

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
score; nothing currently computes one. `_otr_scene_resolver.py` has zero
production callers now, same as before -- see the new fork in section 1: does
it still have a role, or is it a rip.

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
