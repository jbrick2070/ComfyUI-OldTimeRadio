# OTR Go-Forward Plan

**ONLY UNFINISHED WORK BELONGS HERE.** When work finishes its receipt moves to
[HANDOFF_LOG](HANDOFF_LOG.md) or its own evidence folder and the row leaves this
page. A finished prerequisite earns **one clause inside the row that still needs
it** -- never a receipt, never a measurement write-up, never a struck-through or
"SHIPPED" row. The test is one question: *does a row still in this file stop
making sense without that sentence?* No -> cut it.

**THIS FILE IS ARC AND CODE. TESTING IS NOT IN IT** (operator, 2026-09-12:
*"let's just do the coding and arcs first, let's not even talk testing yet"*).
Decide first, build second -- and when both are empty, section 6 at the bottom
is what you have earned. A deferred arc is deferred coding,
and a gate on evidence a leg produces is not a valid deferral either, because
the legs run last: a row that can only be settled by live evidence is settled
WITHOUT it or cut with the reason written in. **The only things that genuinely
defer** are a row blocked on an operator ruling (section 2) and a row
deliberately cut with its reason. Apply that test whenever a row claims to be
blocked.

Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md) first;
this file does not restate them. **For what has already happened -- commits,
measurements, receipts -- read [HANDOFF_LOG](HANDOFF_LOG.md), newest entry
first.**

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. Exactness is not the goal: *"I'm not expecting anything exact."*

**CRASH-CLASS AND DURABILITY-CLASS DEFECTS ARE THE WORK** -- an uncaught
exception, a live asset written where a sweeper can delete it, an identity that
silently resolves outside its episode, and **a machine that silently renders a
configuration we have already proven wrong.** Rows below use the phrase "not
crash-class" against this definition. Aesthetic drift is closed and is not work;
see [ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md).

## 1. CODE -- already decided, do these in order

Crash-class and ledger-identity first. Each row is one commit on `main`. Do
not start a row below while one above is red.

**Working gate (seconds):** `scripts/otr_working_gate.py`. **Chunk gate
(~10 min):** full `pytest tests`, once the row is green. Commands live in
[known-failures](known-failures.md). This is not the test wave.

### 1. Native character lines still come out English

A non-English `episode_language` already stamps the ledger, casts Kokoro,
and paints captions from `line.text`. The character author can still write
English, so the captions are honest English on a native episode. Author in
the selected language at the generation seam. Do not translate an English
draft. Do not add a caption-language widget.

Wire the existing row `writer_instruction` onto every production call that
can replace spoken text: per-line composer (character **system** message),
grouped exchange, cast-coverage repair, ledger-clean F2 and stage-business
repair, and the cleanup title fallback. English and `Off` stay
byte-identical. `ANNOUNCER` stays the identity key.

Plan: [2026-09-18-multilingual-native-dialogue-plan](2026-09-18-multilingual-native-dialogue-plan.md).
Verify: focused line-composer / exchange / clean-stage / writer tests;
English prompt fixtures do not drift.

### 2. A `cuda:1` voice stamp silently becomes card zero

`nodes/_otr_voice_node_common.py` `_voice_device_from_ledger` accepts only
`cuda` / `cpu` / `mps`. An ordinal stamp falls through and loads on card 0.
Round-trip `cuda:1`, or fail loud. Do not pick a card by guessing.
Verify: a voice-device test for the ordinal; existing audio wiring tests stay green.

## 2. ONE WORD -- no code until he answers

* **A1. Canonical writer on an 8 GB card.** Live canonical still saves
  Qwen 3.5 4B, `llm_quant_policy` `none`, ceiling `10.0`. A dropped node
  defaults to `bnb_nf4` / `14.5`. Same question: leave canonical as the
  16 GB graph and point 8 GB users at the variant, or retune both.
* **A5. 16 GB foley / mime without GGUF.** Official LTX 2.5 safetensors do
  not fit 16 GB (measured). Three shipping graphs still load the Q3 GGUF:
  `otr_16gb_video`, `otr_16gb_foley`, `otr_16gb_mime`. Cloud deluxe already
  ships `cloud_ltx25_foley_plus`. Stay on Q3, move those three to cloud, or
  drop them.
* **Gallery.** Comfy lists one graph. The 21 shipping variants sit in
  `workflows/variants/`, which the template scanner does not read. Promoting
  them changes what the gallery shows tomorrow. Say yes before the files move.
* **Pre-push hook.** A `build_variants --check` plus the sibling matrix
  checks, from `.githooks/pre-push`. Changes how both boxes push.
* **Delete `v2.0-alpha`.** Unblocked: 2.1.1 is Active and the registry icon
  already points at `/main/`. One click. His.
* **Flagged registry versions.** 2.1.5 and 2.1.6 are Flagged. Manager still
  serves 2.1.4. The API gives no reason. His Discord, not a code change.

## 3. Held -- do not build

* 8 GB ship set stays draft until the physical 8 GB wave. He ruled hold.
* `scene_coherence_check` stays inert. Story quality is closed.
* No IP-Adapter on AnimateDiff. He ruled hold.
* Do not ping the Radeon tester. Do not post to the ROCm thread.
* `stable_audio_3` listing `cpu`: re-read the published `--cpu` leg log
  before editing the capability test. Not a new ruling if that log already
  published.

## 4. Constraints specific to this plan

Only the ones not already in CLAUDE.md or the standing rulings.

- Full listener source, no RSS. Cast count is flexible and records requested vs
  actual; the house announcer is excluded from dramatic cast.
- **WE DO NOT CHASE ACT COUNT** (operator, 2026-09-11), the same rule as word
  count: the value is a request, and a run delivers the closest performable
  episode.
- Model checking and a fixed attempt budget only -- no separate chunker, no
  recursive loop.
- An exhausted optional correction still yields a usable ledger; no predictive
  word or duration gate.
- Byline and attribution rules differ for My Story, Original and the adaptation
  banks.
- No replay, migration or re-render project: a saved input means fresh
  generation.
- **1080 / Veo generate resolution** -- standing ruling 2026-09-17
  (do not duplicate it here).

## 5. Parked

Parked and tombstoned items are preserved in
[GO_FORWARD_ARCHIVE](GO_FORWARD_ARCHIVE.md), not here -- by this file's own rule
they are not work. That includes the unqualified installed-family and GGUF opt-in
combinations, the H3 policy receipts, the cfg promotion comparisons, the AMD
scoped pod and platform acceptance, the cloud billing opt-in routing, the
operator-parked casting/adaptation ideas, OTR-Lite after v2, the release
runway, the missing `device_options` test module, regenerating
`docs/MODEL_ASSET_INDEX.md`, and writer widget-label cosmetics.

## 6. And when all of this is done -- it is time to TEST. Hurrah.

**Operator 2026-09-17 confirmed:** *"TEST WAVE AFTER CODING."* Same gate
as 2026-09-12. Do not freeze a wave head to settle a row above.

When sections 1 and 2 are empty, the waiting is over and the fun part starts.
Freeze ONE hash, write it into the `WAVE HEAD:` line of
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md), and turn four
machines loose on it at once -- the 5080, the 4060, the Mac and a RunPod box,
each running the real canonical workflow, all reporting home.

Everything they owe is already written and waiting in
[COVERAGE_OWED](2026-09-11-four-machine-test-wave/COVERAGE_OWED.md) beside the
two lane documents. Nothing needs planning when the day comes; it needs starting.

Until then: **do not freeze a head, do not book a leg, and do not settle a row up
there by rendering something.** Two heads were cut early on 2026-09-11 and both
had to be withdrawn. An arc that can only be answered by live evidence is
answered without it, or cut with the reason written in.

**Then the next morning begins in `otr/obs/`, not in the editor** -- count what
landed against what was promised, read the four phone-homes, and triage anything
crash-class first. That is the good problem to have.
