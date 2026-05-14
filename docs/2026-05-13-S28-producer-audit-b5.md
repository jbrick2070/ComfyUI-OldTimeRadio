# S28 Phase 4 — Producer audit b5 (`_otr_ledger_freeze.py` ledger-shape)

Per S28 plan §Phase 4 §Step 4.1. Audits each `_otr_ledger_freeze.py`
consumer-side ledger-shape back-compat fallback. **Audio-critical
phase** — Rule F governs each deletion (one site per commit, audio-
byte-identical runs after every commit, revert + trace deeper on any
break).

| Site | Field read | Producer | Producer always populates? | Rule | Action |
|------|------------|----------|----------------------------|------|--------|
| `_otr_ledger_freeze.py:279` | `meta.outline.beats` (fallback when top-level `beats` is empty) | `OTR_LedgerScriptWriter` post-outline (always stamps `ledger["beats"]` from the outline pass). | YES — every OTR_LedgerScriptWriter code path stamps top-level beats. The meta.outline.beats walk was a back-compat fallback for "caller-shaped" ledgers from pre-D-inversion (pre-2026-05-10). | A | Drop the meta.outline.beats walk. Top-level `beats` is the only contract. Forensic comment marks the removal. |
| `_otr_ledger_freeze.py:356` | `skip=True` without `tts_skip_reason` (warn-only) | Writer per-line (phantom-skip + reviewer-skip stamps both fields). | YES — every production skip path stamps tts_skip_reason. The warn-only branch was a back-compat tolerance for "some legacy fallbacks set skip without the reason." Per Rule B (uniform shape across all line types), drop the skip flag tolerance uniformly: promote the warning to an error. | A, B | Promote `skip=True && tts_skip_reason missing` from warning to error. Producer-contract violation under v2.0. |
| `_otr_ledger_freeze.py:482` | `speaker_role` substitute (comment-only — references already-retired tts_model / speaker_role substitute) | Writer per-cast-row. | YES — this site's tolerance code was retired in voice-path-cleanbreak Gate 2. The S28 work here is comment-only: drop the "back-compat shim" framing from the forensic note. | A, B | Comment-only update. The runtime tolerance was already extinguished. |
| `_otr_ledger_freeze.py:669` | `dur_s` per sfx line — skips validation when `None` (back-compat with older ledgers) | `find_clip_durations` / upstream timing in OTR_LedgerScriptWriter outline pass — always populates `dur_s` on sfx lines from the writer's target duration. | YES — every sfx line in the outline gets a target `dur_s` (clamped to [SFX_DUR_MIN_S, SFX_DUR_MAX_S] downstream). The `dur is None` skip in G7 was a back-compat for older flat-layout ledgers; those ledgers are extinct after S26's per-episode-workspace cutover. | A, B | Drop the `if dur is None: continue` skip. A missing dur_s on an sfx line is now a hard validation error. |

## Producer fix scope

The audit finds no LIVE producer leaks in this phase — all four
producers already populate uniformly. The pre-S28 tolerance code in
`_otr_ledger_freeze.py` was defending against retired shapes, not
against current producer bugs. No `s28-p4-producer-N` commits are
required before Step 4.2; the four deletions can proceed
sequentially.

## Step 4.2 deletion sequence (one site per commit)

Each commit runs `pytest tests/test_otr_ledger_freeze.py
tests/test_audio_byte_identical.py -q` (per plan §Step 4.2). Audio-
byte-identical is the arbiter (Rule F). If byte-identity breaks on
any commit, revert and trace one producer level deeper.

  1. `s28-p4-site1` — `:279` meta.outline.beats walk in
     `_check_lines_block` valid_beat_ids construction.
  2. `s28-p4-site2` — `:356` skip=True tolerance (warn -> error
     promotion).
  3. `s28-p4-site3` — `:482` speaker_role substitute comment update.
  4. `s28-p4-site4` — `:669` dur_s None skip.

Each gets a forensic comment at the deletion site.

## Audio-byte-identical risk per site

  * Site 1 — LOW. valid_beat_ids is used for beat_id reference
    warnings only; no audio output dependency.
  * Site 2 — LOW. skip=True validation is metadata; no audio output
    dependency. If a previously-tolerated ledger now hard-errors,
    the writer either had a bug (now surfaced) or the ledger was
    invalid (now rejected) — neither degrades audio.
  * Site 3 — NONE. Comment-only change.
  * Site 4 — LOW. dur_s validation is metadata; no audio output
    dependency. Out-of-band rendered audio still byte-identical
    because the rendered dur_s post-write is the same value.

Net: byte-identity should hold at every commit boundary. Plan's
Rule F revert+trace path is the safety net.
