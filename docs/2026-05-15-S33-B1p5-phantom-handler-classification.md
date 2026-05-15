# S33 B1.5 — Phantom handler classification per refined rule

> **Status:** B1.5 classification complete. **No MIXED classifications.** B4 cleared to proceed after B2 + B3.
> **Branch:** `s33-editor-only-cleanup` @ B1 (`8afec7b`).
> **Refined rule (Jeffrey, 2026-05-15):** "Audit calls are OK if they USE the audit to develop / edit the story. NOT OK if they just cut the pipeline (gate, halt, fail, rollback, report-only)."

---

## Refined-rule application

For each of the three phantom-related helpers, the question is: does the function (or what it powers) develop / edit the story, or does it cut the pipeline (skip / mute / report-only)?

---

## 1. `auto_remap_phantom` — **KEEP**

**File:** `nodes/_otr_ledger_reviewer.py` line 223.

**What it does.** Pure function. Takes a phantom name + cast roster, returns the closest match (case-folded substring fast-path, then Levenshtein with tie-break). Returns `None` if ambiguous or out of threshold. Docstring is explicit: "This function does NOT mutate anything."

**Where it's used.** Called by `apply_deterministic_cast_repairs` at line 575: `remap = auto_remap_phantom(v.found, full_roster)`. The remap result is then used to substitute the phantom literal in the line text — i.e., to edit the story.

**Classification reasoning.** `auto_remap_phantom` is a pure Levenshtein helper, not an audit call in its own right. The story develops through its caller (`apply_deterministic_cast_repairs` — already on the settled-KEEP list). Without `auto_remap_phantom`, `apply_deterministic_cast_repairs` loses its name-substitution mechanism. The refined rule applies transitively: the helper USES the audit signal to develop the story.

**KEEP.** Survives S33 untouched.

---

## 2. `apply_phantom_skip_fallback` — **DELETE**

**File:** `nodes/_otr_ledger_reviewer.py` line 906.

**What it does.** Mutates the ledger by setting `line["skip"] = True`, `line["text"] = ""`, `line["char_count"] = 0`, `line["word_count"] = 0`, plus a `tts_skip_reason` forensic tag. The mutation SILENCES the line; it does not rewrite the content.

**Where it's used.** Called by `review_ledger` at line 1210 between Phase 2's commit and Phase 9's audit: "Step 2.5 -- Deterministic phantom-skip fallback".

**Classification reasoning.** The function does mutate the ledger, but the mutation is a **skip/mute**, not a content edit. The line is dropped from TTS playback rather than fixed. The refined rule explicitly lists "skips/fails" as pipeline cuts. A muted line is a pipeline cut — it removes content rather than developing the story.

Additionally: per Jeffrey's phantom-ship policy ("phantoms can ship occasionally if Phase 2 produces them; rollback gate removal is the trade-off"), the rationale for muting phantoms post-Phase-2 is gone. The trade-off Jeffrey accepted is that occasional phantoms reach the audience rather than being muted to silence.

**DELETE.** Removed in B4. Its `review_ledger` dispatch site (line 1210-1212) and any tests targeting Step 2.5 phantom-skip also go.

---

## 3. `_final_phantom_check` — **DELETE**

**File:** `nodes/_otr_ledger_reviewer.py` line 961.

**What it does.** Pure read-only function. Returns `[(line_id, phantom_token), ...]` for any phantom still present in non-skipped lines after Phase 2 + Step 2.5. Does NOT mutate.

**Where it's used.** Called by `review_ledger` at line 1216: `final_phantoms = _final_phantom_check(candidate, cast_roster_upper)`. Its output feeds into `post_audit_pass` at line 1218-1222 (boolean gate combining `post_audit.pass_clean`, `post_audit.violations`, and `final_phantoms`). The gate drives the rollback branch (lines 1227-1251).

**Classification reasoning.** Pure report-only function. Its only consumer is the `post_audit_pass` rollback gate, which is on the settled-DELETE list (B2 removes it). Once `post_audit_pass` is gone, `_final_phantom_check` has no caller. The refined rule lists "report-only" as a pipeline cut.

**DELETE.** Removed in B4 after B2 has already removed the rollback gate that consumes its output. The `final_phantoms` variable assignment at line 1216 and its use at line 1221 / 1242 (`post_audit_violations` counter) also go as part of B2's rollback-gate teardown.

---

## Summary table

| Function | Line | Mutates? | Consumer | Edits or cuts? | Decision |
|---|---|---|---|---|---|
| `auto_remap_phantom` | 223 | No | `apply_deterministic_cast_repairs` (editor, KEEP) | Edits (via caller) | **KEEP** |
| `apply_phantom_skip_fallback` | 906 | Yes (skip=True, text="") | `review_ledger` Step 2.5 dispatch | Cuts (skip/mute is a pipeline cut) | **DELETE** |
| `_final_phantom_check` | 961 | No | `post_audit_pass` rollback gate (DELETE in B2) | Cuts (report-only) | **DELETE** |

No MIXED classifications. No architectural halt required. B4 proceeds with two deletions after B2 + B3 land.

---

## Tests affected (preview for B4)

Searched `tests/test_phase3_ledger_reviewer.py` for the three functions:

- `auto_remap_phantom` — KEEP. Tests at lines 121-153 + 232 (Levenshtein G8 table + edge cases). All survive.
- `apply_phantom_skip_fallback` — DELETE in B4. Need to grep for tests at file-read time.
- `_final_phantom_check` — DELETE in B4. Function is module-private (`_`-prefix); tests may or may not exercise it directly.

B4 will close these out per the deletion sweep gate.

---

## Sources

- `nodes/_otr_ledger_reviewer.py` lines 223 (auto_remap_phantom), 906 (apply_phantom_skip_fallback), 961 (_final_phantom_check), 1152 (apply_deterministic_cast_repairs call site), 1210-1212 (Step 2.5 dispatch), 1216 (final_phantoms variable).
- `docs/2026-05-15-S33-B1-cascade-auditor-inventory.md` — prior B1 inventory + halt rationale.
- Refined rule from Jeffrey 2026-05-15 (resume directive).
