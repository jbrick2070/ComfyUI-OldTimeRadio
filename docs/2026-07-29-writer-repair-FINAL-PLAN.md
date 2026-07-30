# OTR WRITER REPAIR -- FINAL HARDENED PLAN

> **SUPERSEDED 2026-07-30.** This document is retained as historical analysis.
> The operator has now ruled that fictional continuity with an article or an
> abandoned draft is not required. Recoverable P0/P1/P2/P3/P5 model-output
> defects retire the candidate and start a fresh complete model-authored
> candidate until acceptance or operator cancellation; there is no fixed outer
> candidate ceiling and no deterministic summary/drop/canned-story floor.
> Complete RSS bodies are covered by overlapping P0 windows and exact rebasing.
> Therefore Section 0's open listener-floor choice, its rejection of fresh P5
> candidates, Window B's degrade-ledger design, "C-4 re-roll: CUT", "Cap
> `_p0_evidence_projection`: CUT", "Never wire `p0_source_chunks`", and the
> zero-salvage/cancellation conclusions no longer govern implementation.
> Deterministic configuration, security, provider, I/O, compiler, invariant,
> freeze, and proof failures still fail loudly. See
> `docs/2026-07-30-story-never-fails/FINAL_PLAN.md`.

Repo: ComfyUI-OldTimeRadio, branch v2.0-alpha, HEAD 47c554fa. ASCII only.
Produced by a full 4-round kibitz arc (Codex gpt-5.6-sol high + Antigravity,
plus a Claude seat in r1), every claim grounded against the real files by the
judge. 8 agent calls across the arc. This supersedes r1/r2/r3 final.md.

**READ SECTION 0 FIRST. There is one operator decision and the panel split on
it.**

---

## 0. THE ONE OPERATOR DECISION

**What should a listener hear when the writer cannot produce a line?**

This question survived all four rounds. It gates the entire degrade feature
(items D-1..D-4). It is not a technical question -- the code has no opinion --
and **the panel disagreed on it in r4**, which is why it comes to you.

Why it is unavoidable: `ScriptTextDraftLineV4.text` is `min_length=1`
(`nodes/_otr_scifi_codex.py:566`); an empty line is a validator finding
(`:2029`); TTS raises on empty prepared text
(`nodes/_otr_voice_node_common.py:511-527`); and `clean_spoken_text` can empty
a row by itself (`nodes/_otr_script_prep.py:22-30`). So "salvage a partial
draft" has no defined meaning when zero complete spoken rows survive.

**Option A -- the announcer reads a deterministic summary.** Build one or more
announcer lines in plain Python from the ALREADY-ACCEPTED P0 facts. Always
produces speakable text; the facts are guaranteed present because P3 already
gates on them. Cost: the episode is factual rather than dramatic.
*Judge's recommendation.* Note: it must NOT be bracketed text --
`_otr_script_prep.py:16` strips `\[[^\]]{1,40}\]`, so a short bracketed line
cleans to empty and hits the TTS rip, while a longer one is read aloud
brackets and all. Plain prose only.

**Option B -- drop only the failed beats.** Keep episode structure, ship fewer
beats. Cost: needs the coverage plan to tolerate a beat vanishing after
ShotLock, which is unproven.

**Option C -- still die, but only after a full receipt.** Accept that a floor
is not always reachable. Cost: does not satisfy "the writer never vetoes".

**REJECTED by the judge: Codex's r4 proposal to never assemble a partial
ledger and retry P5 indefinitely.** It has no bound, no cost ceiling and no
cancellation story -- see item 9. That is an unbounded-spend hazard, not a
fix.

---

## 1. DEFINITION OF DONE (was missing from every earlier draft)

Baseline, measured 2026-07-29: **45 legs, 34 failures, 24 of them in the
writer** (15 P0, 8 P5, 1 P3). Eleven episodes landed.

**Target: a full 19-engine 45-word pass with zero writer-caused
terminations**, every writer degrade visible in a receipt. Measure by re-running
the same campaign (`tmp\_w45_campaign.ps1`) -- it is the only apples-to-apples
baseline that exists. Stop building when that passes.

---

## 2. DO THIS FIRST -- IT MAY REORDER EVERYTHING (1 hour, no code)

**Re-classify the 15 P0 deaths against post-`47c554fa` code.** The plan's only
P0-specific fix is HTML entities, and entities were verified in just 4 logs.
Today's deterministic-repair wiring landed AFTER the campaign, so its live
effect is unmeasured. If the deterministic rung would have saved most of those
15, the priorities below are wrong. The logs are `tmp\otr_headless_*.log`.

---

## 3. WINDOW A -- INDEPENDENT FIXES (each ships value alone)

One coder window in the code at a time (`CLAUDE.md:92`) -- the earlier
"parallel-safe" section was dead text and is withdrawn. Each item = one
commit, full suite + Bug Bible + hygiene, plus a wiring test asserting the new
behaviour FIRES and the counterfactual dies without it, per
`tests/test_p0_deterministic_repair_wired.py`.

**A-1. Decode before raise.** `OTR_LedgerScriptWriter.py:957-963` raises before
`:991` decodes, so the raw completion is never available at the failure point.
Move the decode above the raise; attach prompt/output/context token counts.
Behaviour-neutral. **Do NOT put the decoded completion in the exception
string** -- it floods logs; attach it as a field.

**A-2. Measure (no code).** On the instrumented build, classify the P0 decode
failures: EOS-absent-at-cap versus prose-wrapped JSON. These need opposite
fixes. Note the 28 decode-message legs EXCEED the 24 writer deaths, so that
message fires on legs that later succeeded -- it is a retry marker, not a death
certificate. Also run one fixed-seed A/B to settle whether re-rolls differ.

**A-3. HTML entity decode -- NARROW.** Add `import html` (absent,
`_otr_scifi_codex.py:10-21`). In `_normalize_span_source_text` (`:1049-1062`),
decode **`&nbsp;` / `&#160;` / `&#xA0;` ONLY.** That is the entire verified
production failure (leg `viz_green`, MIT Genesis Mission article). A wider set
is speculative and silently widens the digest coordinate system for every
future run. Digest ordering is already correct (`:1078-1079` precedes `:1097`).
Owes a digest-stability fixture and a regression built from the real article.

**A-4. Capacity error gets a `phase`.** Add it in
`_otr_generation_budget.py`, which already owns
`GenerationContextOverflowError` (`:29`, raised by `fit_output_tokens:76,82`);
the writer re-wraps at `:847->856` as `PromptContextOverflowError`. Phases:
`prompt_no_room` | `output_limit`. Only `output_limit` may re-roll.
**There are TWO JSONDecodeError retry gates, not one:
`_otr_structured_call.py:1017` (structural rung) and `:1127-1131` (repair-syntax
rung). Patch both.** Mirror the dual relative/absolute import fallback at
`_otr_model_loader.py:48-53` or `tests/test_structured_call.py` fails at
COLLECTION -- `_otr_structured_call.py` is documented pure (`:36-45`).

**A-5. Canonicalise spoken text at acceptance.** Build a COPIED
`ScriptArtifactV4` before `_assemble_ledger` so all identity consumers read one
object: `expected` (`:2189`), `line_text_sha256` (`:2234`), `stamp_receipt`
(`:2489`), `_script_digest` (`:2496`).
**Grandfather rule: existing frozen ledgers keep their raw-text hash and are
never re-pinned; only ledgers produced after this commit use the cleaned-text
hash, and the receipt records which generation produced it.**
Note `clean_spoken_text` also strips speaker labels
(`_otr_script_prep.py:16,26`), a SEPARATE P5 rejection at `:2035` -- do not
conflate.

**A-6. Enforce "re-author, never skip" AS A FINDING, not an assert.**
`_otr_ledger_cleanup.py:253-268` silently sets `skip=True` on a voiced row that
cleans to empty, so the TTS no-fallback rip (`:511-527`) never fires. The fix
must be a P5 **finding** at `_spoken_text_finding` (`:2023`) that sends the row
back to be re-authored. **An assert would itself be a veto** and violates the
operator ruling.

**A-7. Doc supersession -- TWO commits, two repos.**
`docs/PRODUCTION_SPRINT_LESSONS.md:563-571` (lesson 34) and `:590-620`
(lesson 35) still read as capacity-terminal. `BUG_BIBLE.yaml` 11.50 (`:4045`)
and 12.68 (`:4848`) live in the SEPARATE repo
`ComfyUI\comfyui-custom-node-survival-guide` and are a mandatory gate
(`CLAUDE.md:64`). This cannot be one commit.

---

## 4. WINDOW B -- THE DEGRADE FEATURE (worthless half-built; needs Section 0)

**B-1. A cancel path, FIRST.** The writer has zero `model_management`
references; the mux has the pattern (`otr_master_audio_mux.py:255-268`). Any
retry/degrade loop without it is unbounded and uninterruptible.
**Do not copy the mux's `_poll_interrupt` verbatim -- it swallows the cancel.**
Verify the real exception type by import probe on the box first; the type is
assumed, not proven.

**B-2. Retain the rejected candidate IN THE LANE.**
`StructuredCallFailedError` carries no raw candidate (`:131-164`) and
`last_raw` is overwritten (`:955`, `:984`) -- but the lane already holds
`last_raw[0]` (`_otr_scifi_codex.py:1569`) and per-attempt sha (`:1626-1637`).
Build an immutable per-attempt record inside `invoke_codex_structured`.

**B-3. Classify, lane-locally.** `post_validator` stays binary
(`_otr_structured_call.py:722-739`); every caller depends on it.

**B-4. The degradable-exception allowlist must live INSIDE
`invoke_codex_structured`, before the re-raise.** Critical correction from r4:
that function flattens EVERY exception to `CodexPassError` (`:1705`, `:1731`),
so an allowlist at the caller can only ever see one type. Also note
`ValidationError` and `json.JSONDecodeError` are consumed at `:996` and can
never reach a degrade guard -- drop them from any allowlist.
Not degradable, ever: `CodexPackContractError`, `VoiceCastingError`,
`LedgerIncompleteError`, the interrupt exception, anything from
`model_management`.

**B-5. The guard, INSIDE `run_scifi_codex_episode`.** Not at
`OTR_LedgerScriptWriter.py:3470-3487` -- that frame sees only the lane call,
every pass is a local, and the ledger assembles at `:2485`, so a guard there
can only produce an EMPTY ledger.

**B-6. The receipt goes in ledger `meta`, not on the wire.** Correction from
r4: node 62 `OTR_LedgerFreezeCascade` never parses `script_json` -- it has zero
`json.loads`, reads `peek_ledger()` and re-serializes `led.data` at `:344`. So
stamping the receipt into `meta.scifi_codex` makes preservation automatic;
"read it off link 230" is impossible. `build_phase_telemetry`
(`_otr_freeze_cascade.py:348-373`) would need a fourth bucket to surface it.
Receipts carry schema version / pass / phase / rung / hashes -- never rejected
prose.

---

## 5. CUT AND PARKED

- **C-4 re-roll: CUT.** Blocked on a seed interface that does not exist
  (`_otr_structured_call.py:630-636`), and the GGUF `_ordinal` resets per call
  (`_otr_gguf_backend.py:1362-1375`), so even a nonce would not vary. Value
  unproven pending A-2. Record as a follow-up with the blocker named. (Node 1
  already has 34 widgets; a control would be the 35th slot.)
- **Cap `_p0_evidence_projection`: CUT.** A 48,000-byte cap already exists at
  `_otr_scifi_codex.py:1089`, and slicing would break the span coordinate
  system.
- **C-7 exact-count P5 schema: PARKED, dead end as drafted.**
  `_p5_raw_spoken_findings` runs only AFTER a successful Pydantic validation
  (call `:1846`, def `:2094`), so feeding it the count cannot fix a length
  rejection. If revived it needs `min_length == max_length` AND the `:1510`
  identity test relaxed to `issubclass` -- otherwise a derived subclass ships
  silently without `_SCRIPT_TEXT_DRAFT_ROOT_INSTRUCTION` (`:1529`), leaving a
  green suite and no contract.
- **C-3 backend-honesty: fold into B-2's commit.** One journal string at
  `:1629`; it changes no outcome alone.

---

## 6. INVARIANTS

1. Actual word count of a produced episode never rejects it. (`requested_words`
   is separately and legitimately bounded 30..900 at `:166`.)
2. No hole in the ledger: every field a weakened pass owned gets exactly one
   new owner FIRST. Known floors: `voice_preset` on every cast row
   (`cast_lock.py:423-433`); full render structure (scenes/shots/beats/routing/
   music, `:2163-2233`); `CodexTailParts` (`:2503-2513`) and
   `LedgerIncompleteError` (`_otr_ledger_cleanup.py:657-662`).
3. Fail loud, not fatal. Every receipt names what was wrong, what was done, and
   who reads it.
4. P0's fact-ID set stays authoritative even if span handling relaxes -- P3
   hard-rejects beats citing an unknown `fact_id` (`:797-803`).
5. Never wire `p0_source_chunks` / `p0_source_char_budget`.
6. Span quotes keep LITERAL identity; relocate or prune deterministically.
7. A stage-direction-only line is re-authored, never silently skipped -- and
   per A-6 this must be ENFORCED as a finding, not asserted.
8. Local and offline only.

---

## 7. CLAIMS PROVEN FALSE ACROSS THE ARC (do not let these back in)

- `_assemble_ledger` KeyErrors on `c04` / a missing cue id -- both are
  `Literal` types (`:211`, `:325`, `:377`).
- `skip=True` for a row that cleans to empty -- forbidden by the no-fallback
  rip; cleanup already does this and that is the BUG (A-6).
- "Remove Invariant 6 / use rebased windows" -- that is wiring
  `p0_source_chunks`.
- The receipt reader is `_otr_freeze_cascade.py:25-40` -- that is `__all__`/log
  setup; `build_phase_telemetry` is at `:348`.
- The receipt reader is `obs_publish` -- node 85 has no ledger input
  (`otr_master_audio_mux.py:551-588`).
- Bound the unescape output by `MAX_QUOTE_CHARS` -- that constant is 240
  (`_otr_scifi_p0_contract.py:21`) and would truncate `full_text` to 240 chars.
- Legacy hashes break `_otr_readiness.py:337-340` -- that is
  `text_for_tts_source_sha256`, recomputed live.

---

## 8. RE-VERIFY ANCHORS AT BUILD TIME

The r3 build sheet carried at least two stale line references (the TTS rip is
`:511-527` not `:494-503`; the structural gate is `:1017` not `:1016`). Line
numbers move. **Grep for the symbol, do not trust a line number from this
document.**

---

## 9. RESIDUAL RISKS AFTER FOUR ROUNDS

- **Zero-salvage is unowned and the panel split.** Section 0 is yours.
- **Nothing measures the fix** until Section 1's target is adopted and
  Section 2's re-classification is run.
- **Cancellation is unproven end to end** -- the exception type is assumed, and
  the new outer retry handler is exactly the thing most likely to swallow it.
- **The receipt schema is undefined and has no proven reader today.**
- **C-1's real entity population in live RSS is unmeasured.** One MIT `&nbsp;`
  is the entire evidence base for a change that redefines source coordinates.
- **Every number is from 45-word episodes.** 120 and 420 are unmeasured and
  could change the priority order.
