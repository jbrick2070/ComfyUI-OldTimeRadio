# original_radio reviewed against the scifi_fable2 26-roll failure classes

Date: 2026-07-11. Analysis-only, no code changed. Grounded against the real
files: `nodes/_otr_original_radio.py` (882 lines, read in full),
`nodes/OTR_LedgerScriptWriter.py` (7146 lines, targeted reads/greps),
`nodes/_otr_outline.py`, `nodes/_otr_casting.py`, `nodes/_otr_ledger_freeze.py`,
`nodes/story_packs/original_radio/original_radio_drama.json`,
`nodes/story_packs/original_radio/spark_deck.json`,
`nodes/story_rules/original_radio.json`, cross-referenced against
`docs/2026-07-10-fable2-s1b-smoke-hardening.md` (26 rolls, both addenda).

**VERDICT: no P0 latent bug found that would kill or corrupt a 720w roll.
The lane is structurally SAFER than fable2 pre-hardening on most classes
because it composes dialogue LINE-BY-LINE through the same shared
`OTR_LedgerScriptWriter.py` path every source-based lane uses (already
hardened by years of Bug Bible fixes), rather than fable2's one-shot
whole-play markup blob. The known 420w history (239-word undershoot,
key_terms 1/5, empty-cast-name) already has code-level fixes in the tree
dated 2026-07-09/10 inside `_otr_original_radio.py`. Two P1 risks are real
for a 720w step-up (undershoot risk restated below, and an unstripped
honorific-prefixed cast name); the rest are P2 nits or already-defended.**

---

## P0 -- would kill or corrupt a roll

None found.

---

## P1 -- quality / undershoot risk for the 720w roll

### P1-1. Word-budget check is WARN-only with a flat 30% band and no
absolute-word floor slack -- root cause of the 239/420 undershoot, and it
recurs at 720w unchanged

Evidence: `nodes/OTR_LedgerScriptWriter.py:204-205`
```
WORD_BUDGET_RATIO_LO = 0.7
WORD_BUDGET_RATIO_HI = 1.3
```
and `:4143-4164` -- sums only voiced (character) beats,
`ratio = beat_word_sum / max(1, resolved["target_words"])`, and on drift:
```
log.warning(
    "[OTR_LedgerScriptWriter] WORD_BUDGET_DRIFT: outline "
    "voiced beats sum to %d words, target %d (ratio=%.2f); "
    "proceeding anyway", ...)
```
No hard fail, no reroll, no widening. The pack's own
`line_composer_system` seam (`original_radio_drama.json` line 11) tells the
model "Within plus or minus 30% of the requested word count" -- a PROMPT
instruction only, never enforced by a post-validator on the composed line
text itself (`build_original_briefs`/the shared `compose_line` path has no
per-line word-count post_validator at all -- confirmed by reading
`_otr_original_radio.py` in full: its only word-count-adjacent gates are
`casting_brief`/`script_brief` minimum CHARACTER length, not word counts).

Fable2 lesson: roll 17 (WORD_BUDGET exhaustion, 54 vs 24-36 at a 30-word
target; fixed with an absolute +/-25-word floor slack because the
proportional band is too narrow at small targets). original_radio's flat
30% band has the same shape problem but INVERTED at scale: local 12B-class
models chronically UNDERWRITE prose (this is the mechanism behind
239/420 = 57%), and nothing in the pipeline ever compensates -- the
WARN-only check just logs and moves on, target_words allocation across
beats is deterministic Python (`_allocate_phase_target_words`,
`_otr_outline.py:1385-1420`, good: not LLM math), but the actual TEXT each
beat produces is free to fall well under that allocation with zero
correction.

Operator law respected: word counts are advisory-only, so the fix is NOT a
count gate. Concrete non-creative fix sketch:
- Raise the per-beat words_per_beat_range floor / widen it so under-target
  beats get a materially bigger ask, and/or bump the composer's attempt-1
  token generosity (`max_new_tokens_cap` scaling at
  `OTR_LedgerScriptWriter.py:2453-2464`, currently
  `min(cap, target_words * 4)`) -- a token ceiling that is comfortably
  above what's needed does not by itself make a small local model write
  longer, so this alone will not fully close the gap, but it removes token
  ceiling as a confound before judging the next roll.
- Strengthen `line_composer_system`'s craft language with concrete
  technique steering ("develop the beat across two clauses", "let the
  reply complicate, not just answer") rather than a bare percentage --
  percentages do not reliably steer local models; concrete instructions do.
- At minimum, promote the existing WARN to a stamped `meta.story_quality`
  field the operator eyeball can see per-episode (it is not currently
  captured anywhere outside the log line), so the 720w roll's actual
  undershoot ratio is visible without re-deriving it from the ledger.

### P1-2. No honorific-token stripping on original_radio's own cast names
(fable2 roll 19's exact shape, not yet needed here but the surface exists)

Evidence: the CONCEPT/SELECT seams ask the model for
`"a period-plausible personal name in CAPS (e.g. \"MARTHA VANE\",
\"ELI CROSS\")"` (`original_radio_drama.json` line 17) but do not forbid an
honorific prefix, and nothing downstream strips one. `_normalize_speaker`
in `_otr_outline.py:1453` (used for Stage-2 speaker routing against the
locked cast) only strips EDGE punctuation
(`_SPEAKER_EDGE_PUNCTUATION = " \t\r\n\"'\`.,;:!?*_-()[]{}<>"`,
line 1450) -- it explicitly preserves internal punctuation "(e.g. `DR.
LEMMY`)" per its own comment at line 1449. `build_original_briefs`'s cast
gates (`_otr_original_radio.py:459-474`, `501-526`) only check
non-empty/grounded, never honorific shape. If a 12B model emits
`"DR. ELI CROSS"` as a cast_sketch/select name, it will pass every gate
in this file as-is and ride through cast-membership matching literally.

Fable2 lesson: roll 19, `'DR. HARRIS'` broke repair convergence; fixed
with deterministic label normalization (strip honorific tokens, keep
surname). Concrete non-creative fix sketch: add the same deterministic
strip (small, data-driven honorific list: Dr., Mr., Mrs., Ms., Miss,
Capt., Rev., Prof., ...) as a normalization step on cast names the moment
they're accepted in `_select_gate`/`_concept_gate` in
`_otr_original_radio.py`, mirroring the precedent already set in fable2's
label-normalization fix -- this is deterministic text normalization, not
LLM rewriting, so it is legal under the "Python judges, LLM writes" law.

---

## P2 -- nits

### P2-1. `num_characters` does not scale with `target_words`
`OTR_LedgerScriptWriter.py:1157/1246` -- widget default 2, hard-clamped to
1-6, with no relationship to the word-count target. Not a code defect (it
is an explicit operator widget), but worth flagging before the 720w
bake-off: a 2-character cast carrying 720 words of dialogue with no
scaling guidance risks monotony the QA passes won't catch (QA_CLASSES has
no "cast too thin for length" class). Purely an operator dial to check
before the roll, no code change implied.

### P2-2. `coda_system` requires a terminal colon by prompt only
`original_radio_drama.json` line 13: "at most ~16 words, ending with a
colon" -- prompt instruction, no post-validator enforces it (confirmed:
`run_original_qa`/`_qa_gate` in `_otr_original_radio.py` never inspects
the coda's terminal punctuation; the QA_CLASSES enum has no coda-specific
class). Fable2 roll 15 died on exactly a colon-vs-period pivot mismatch
because ITS pivot punctuation WAS load-bearing to a downstream parser.
original_radio's coda is consumed as plain announcer speech (no structural
parser depends on the colon), so this is lower risk than fable2's case --
noted as a nit only because an un-enforced "ending with a colon" ask is
exactly the shape of instruction local models drop first under length
pressure; if a future pass adds any code that keys off the colon, revisit.

---

## Already defended -- do not re-litigate these on the next roll

- **Markdown/parenthetical/stage-direction text reaching TTS.** Two
  independent defenses already in the shared writer, both lane-agnostic
  (apply to original_radio automatically): (a) per-line generation cleanup
  strips asterisks/smart quotes at composition time
  (`OTR_LedgerScriptWriter.py:992`, "Iteratively strip ASCII + smart
  quotes, asterisks, whitespace"), and (b) a whole-ledger pre-freeze
  "dialogue scrub" regex-strips every `(...)`/`[...]` span from every
  non-skipped line (`OTR_LedgerScriptWriter.py:5990-6031`, look-QA round 4
  precedent). original_radio's `line_composer_system` seam also explicitly
  bans stage directions/parentheticals/name-colon prefixes in the prompt
  itself (`original_radio_drama.json` line 11) -- belt AND suspenders,
  stronger than fable2 had pre-hardening.

- **Chat-template consecutive-same-role turns.** original_radio's own
  `structured_call` invocations in `_otr_original_radio.py` are always
  exactly `[system, user]`, single-shot; retry/reroll folding is owned
  entirely by the shared `_otr_structured_call.py` ladder that fable2 also
  uses, so any template-folding fix there is inherited for free. No
  original_radio-local multi-turn construction exists to misfire.

- **ALL-CAPS reaching TTS / shout-leak.** The shared writer has extensive,
  already-hardened ALL-CAPS cast-name defenses (roster-caps shout
  detection/repair, `OTR_LedgerScriptWriter.py:2000-2053`, "gemma
  shout-leak" precedent) that apply to every lane including original_radio.

- **Nested-path truncation clamp (fable2 roll 10's `pitches.0.hook`
  class).** Lives in the shared `_otr_structured_call.py` ladder, not in
  fable2-local code -- original_radio's `ConceptPitches`/nested
  `cast_sketch` lists ride the same clamp automatically.

- **Casting-JSON truncation at a fixed token budget (fable2 roll 18).**
  Structurally does not apply: the shared casting pipeline
  (`nodes/_otr_casting.py`) calls the LLM ONCE PER CHARACTER SLOT
  (`llm_write_description`, line 738, "the LLM writes ONLY the prose
  description for one slot" -- gender/voice are pure Python), each call
  budgeted 250 tokens for one field. There is no whole-cast batched call
  that could truncate the way fable2's did; the per-slot design scales
  with cast size automatically. No fix needed.

- **Skip rows without `tts_skip_reason` (fable2 roll 22's freeze-cascade
  class).** `nodes/_otr_ledger_freeze.py:342-350` hard-fails any row with
  `skip=True` and a missing `tts_skip_reason`, and `:390-400` hard-fails a
  present-but-null `tts_skip_reason` -- lane-agnostic, applies to every
  lane's rows including original_radio's.

- **Announcer sentinel `char_id` exemption (fable2 roll 22's second half).**
  The shared writer stamps `cid = "announcer"` unconditionally for every
  announcer beat (`OTR_LedgerScriptWriter.py:5573-5581`), and
  `_otr_ledger_freeze.py:520` checks `if char_id == "announcer":` -- this
  is a SHARED, lane-agnostic exemption, not a fable2-local patch;
  original_radio's announcer rows get it automatically, no separate fix
  needed.

- **key_terms grounding / empty-cast-name (the lane's own 2026-07-09/10
  post-mortem fixes, already in tree).** `_otr_original_radio.py` already
  carries: (a) anchor A2 verbatim-grounding gate + a deterministic
  ungrounded-key_term pruning repair (`_prune_ungrounded_key_terms`,
  lines 528-574, explicitly documents "Live-smoke hardening 2026-07-09");
  (b) empty-cast-name gates at every stage of the creative front
  (`_concept_gate` line 425-426, `_select_gate` line 464-465). These read
  as the code-level fixes for exactly the two known 420w failures
  (key_terms 1/5, empty-cast-name); they should hold at 720w since neither
  gate is word-count-scaled.

- **Entity/number invention vs. source-only laws, spelled-number
  equivalence, honorific/label stopwords on a READ gate.** All N/A by
  design -- original_radio has no source digest, no dossier, no
  news_close_read subset gate (`news_close_brief` is hardwired `""`,
  `OriginalBriefsModel` line 307 + belt-and-suspenders override at
  `_otr_original_radio.py:598-601`). This entire fable2 failure family
  does not exist in this lane's architecture.

- **Validator strictness vs. small local models.** original_radio's gates
  (exactly-3-pitches, exactly-N-cast, 60-char minimums) are comparable in
  shape to fable2's pre-hardening gates, but the ladder around them
  (`structured_call`, 3 attempts with a dispatching deterministic-repair
  hook before any LLM repair burn) is the SAME hardened shared module
  fable2 uses today, not an earlier unhardened copy.
