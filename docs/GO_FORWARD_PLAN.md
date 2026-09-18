# OTR Go-Forward Plan

**ONLY UNFINISHED WORK BELONGS HERE.** When work finishes its receipt moves to
[HANDOFF_LOG](HANDOFF_LOG.md) or its own evidence folder and the row leaves this
page. A finished prerequisite earns **one clause inside the row that still needs
it** -- never a receipt, never a measurement write-up, never a struck-through or
"SHIPPED" row. The test is one question: *does a row still in this file stop
making sense without that sentence?* No -> cut it.

Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md) first;
this file does not restate them. **For what has already happened -- commits,
measurements, receipts -- read [HANDOFF_LOG](HANDOFF_LOG.md), newest entry
first.**

## Operating order (hard)

Scope the row in front of you, then code it, then -- only when the whole
queue is empty -- test.

1. **SCOPE AND DECIDE** -- if the next item has more than one answer, it
   lives here. No code on that row.
2. **CODE** -- if it already has one answer, build it. Other open forks
   do not freeze a decided bug fix.
3. **TEST** -- only when every open fork and every code row is gone.

A row with more than one defensible answer is section 1, even if it looks
like a bug. A row with one verifiable answer is section 2. A row that can
only be settled by a live leg is not a row: settle it without the leg, or
cut it with the reason written in. **Testing does not settle a decide or
a code row** (operator 2026-09-12 / 2026-09-17: coding and arcs first;
test wave last).

The only things that genuinely defer **that row** are an open operator
ruling on it and a deliberate cut with its reason.

Held, parked, and constraint lists are not work. They do not fill
section 1 and they do not block section 3.

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. Exactness is not the goal: *"I'm not expecting anything exact."*

**CRASH-CLASS AND DURABILITY-CLASS DEFECTS ARE THE WORK** -- an uncaught
exception, a live asset written where a sweeper can delete it, an identity that
silently resolves outside its episode, and **a machine that silently renders a
configuration we have already proven wrong.** Aesthetic drift is closed and is
not work; see [ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md).

## 1. SCOPE AND DECIDE

Open forks only. No code on a row in this section. One word from him closes
the row into section 2, or cuts it.

* **A1. Canonical writer on an 8 GB card.** Live canonical still saves
  Qwen 3.5 4B, `llm_quant_policy` `none`, ceiling `10.0`. A dropped node
  defaults to `bnb_nf4` / `14.5`. Leave canonical as the 16 GB graph and
  point 8 GB users at the variant, or retune both.
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

## 2. CODE -- already decided, do these in order

Crash-class and ledger-identity first. Each row is one commit on `main`. Do
not start a row below while one above is red. Do not start a row whose
answer is still sitting in section 1.

**This checkout is dirty.** Two uncommitted piles plus one unpushed
closed-row draft. Do not `git add .`. Do not rewrite a draft that is
already on disk. Composer QA then Sonnet before every push.

**Working gate (seconds):** `scripts/otr_working_gate.py` (on disk, not
on origin). **Chunk gate (~10 min):** full `pytest tests`, once the row
is green. Commands live in [known-failures](known-failures.md). Those
gates are not the test wave.

### 1. Native character lines still come out English

**Decided:** author in the selected language at the generation seam. Do not
translate an English draft. Do not add a caption-language widget. Captions
stay `line.text`. English and `Off` stay byte-identical. `ANNOUNCER` stays
the identity key.

A non-English `episode_language` already stamps the ledger, casts Kokoro,
and paints captions from the spoken text. The character author can still
write English, so the captions are honest English on a native episode.

Wire the existing row `writer_instruction` onto every production call that
can replace spoken text: per-line composer (character **system** message),
grouped exchange, cast-coverage repair, ledger-clean F2 and stage-business
repair, and the cleanup title fallback.

**Draft is already in this working tree.** Finish, QA, and push those
named files. Do not mix the Google pile into this commit. Plan (also
untracked): `docs/2026-09-18-multilingual-native-dialogue-plan.md`.
Verify: focused line-composer / exchange / clean-stage / writer tests;
English prompt fixtures do not drift.

### 2. My Story alias metrics and the working gate never reached origin

**Decided 2026-09-17 and documented as closed; the code is still only
local.** `SpokenLine` alias promotion must call `set_line_text_metrics`.
Music brief pin: `compose_music_prompt` -> `resolve_setting_terms` ->
`spoken_term(t)`. Land `scripts/otr_working_gate.py` and
`tests/test_working_gate_script.py` with that chunk. Named files only.

### 3. A `cuda:1` voice stamp silently becomes card zero

**Decided:** round-trip `cuda:1`, or fail loud. Do not pick a card by guessing.

`nodes/_otr_voice_node_common.py` `_voice_device_from_ledger` accepts only
`cuda` / `cpu` / `mps`. An ordinal stamp falls through and loads on card 0.
Verify: a voice-device test for the ordinal; existing audio wiring tests stay green.

## 3. TEST -- only after 1 and 2 are empty

**Operator 2026-09-17:** *"TEST WAVE AFTER CODING."* Same gate as 2026-09-12.
Do not freeze a wave head to settle a row above. Do not book a qualification
leg from this file.

When every **open** section-1 fork is gone and section 2 is empty, freeze ONE
hash, write it into the `WAVE HEAD:` line of
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md), and turn four
machines loose on it at once -- the 5080, the 4060, the Mac and a RunPod box,
each running the real canonical workflow, all reporting home.

Everything they owe is already written in
[COVERAGE_OWED](2026-09-11-four-machine-test-wave/COVERAGE_OWED.md). Nothing
needs planning when the day comes; it needs starting.

Until then: **do not freeze a head, do not book a leg, and do not settle a
decide or code row by rendering something.** Two heads were cut early on
2026-09-11 and both had to be withdrawn.

**Then the next morning begins in `otr/obs/`, not in the editor** -- count
what landed against what was promised, read the four phone-homes, and triage
anything crash-class first.

## Already scoped -- do not build

These are decided. They are not section 1 and they are not section 2.

* 8 GB ship set stays draft until the physical 8 GB wave. He ruled hold.
* `scene_coherence_check` stays inert. Story quality is closed.
* No IP-Adapter on AnimateDiff. He ruled hold.
* Do not ping the Radeon tester. Do not post to the ROCm thread.
* `stable_audio_3` listing `cpu`: re-read the published `--cpu` leg log
  before editing the capability test. Not a new ruling if that log already
  published.

## Constraints specific to this plan

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

## Parked

Parked and tombstoned items live in
[GO_FORWARD_ARCHIVE](GO_FORWARD_ARCHIVE.md). That includes the unqualified
installed-family and GGUF opt-in combinations, the H3 policy receipts, the
cfg promotion comparisons, the AMD scoped pod and platform acceptance, the
cloud billing opt-in routing, the operator-parked casting/adaptation ideas,
OTR-Lite after v2, the release runway, the missing `device_options` test
module, regenerating `docs/MODEL_ASSET_INDEX.md`, and writer widget-label
cosmetics.
