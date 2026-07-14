# Bakeoff observations -- 2026-07-14 (420 / 720, pinned Aion + Mistral-Nemo)

Defects seen DURING the sweep. Recorded, deliberately NOT patched mid-run.

**Why not patched:** the bank is supposed to be the ONLY variable. Anything in
the SHARED path (`_otr_story_brief.py`, `_otr_structured_call.py`, the composer,
the render driver) that changes mid-sweep means the early banks ran different
code than the late ones, and the comparison is worthless. A shared-code fix lands
AFTER the sweeps, on a clean re-baseline. Only a leg that is DEAD earns a
mid-sweep fix, and only when the fix is scoped to that one lane (see: the codex
P4 fix, which touches `_otr_scifi_codex.py` alone and is re-legged).

---

## OBS-1 -- `logline_names_no_cast_member`: the bs4 fix moved the model's aim

**Seen:** `scifi_sonnet` 420w (prompt `75f49f24`), 07:48. Non-fatal.

```
run_produced_story_summary attempt 1 failed: logline_names_no_cast_member
  {"logline": "A debate on relabeling prostate cancer and its impact; ..."}
run_produced_story_summary attempt 2 (repair) failed: logline_names_no_cast_member
[OTR_ProducedStory] ... stamping failed status
```

**Not a ledger hole.** `_produced_story_failure` (`_otr_story_brief.py:1153`) is a
designed fail-soft sentinel: it stamps `produced_story_status="failed:<reason>"`
and leaves `produced_story` ABSENT on purpose, "so no consumer prints a broken
value". The field has an explicit owner. The episode renders; it just ships
without a logline.

**ROOT CAUSE -- and it is NOT a Mistral-Nemo regression (operator, 2026-07-14:
"Mistral-Nemo was working fine for original science for many months").** He is
right, and that is the clue. The model did not change. THE INPUT DID.

Last night's `a3a48290` installed `beautifulsoup4`, which had never been present.
`_fetch_full_article` does `except ImportError: return ""`, so it had been
returning EMPTY, SILENTLY, FOREVER. Every science-sourced episode this project
ever made was written from a **~120-character RSS teaser**. Post-fix the same
feed yields **2,041-6,708 characters of real article body**.

So for months the technical slot wrote the logline from a starved teaser and
leaned on the CAST, because the cast was very nearly all it had. Hand it a dense
article about prostate-cancer relabeling and it does the natural thing -- it
summarizes THE ARTICLE, and names no character. The validator did not get
stricter. **The source got richer, and the model followed the source instead of
the story.**

The `run_produced_story_summary` seam was therefore implicitly tuned against
STARVED input, and the bs4 fix invalidated that tuning. Expect this on the other
science-sourced lanes in this sweep -- it is an input-distribution shift, not a
one-off.

**Fix (AFTER the sweeps, on a clean baseline):** the seam must say what the
logline is FOR -- a story logline names its people. Repair route belongs in the
`_repair_rules` for that pass (the codex56 unstated-contract class: a constraint
the post-validator enforces but the seam never states is invisible to the model).
Do NOT fix by loosening the validator: a logline that names nobody is a synopsis
of a news article, not a logline for a radio play, and that is exactly the
announcer-framing defect already on the books.

---

## OBS-2 -- lane structure at length (EXPECTED, not a defect)

The lanes do NOT write the same shape of show at length, and the operator has
ruled that target words are not chaseable. RECORD, never normalize. The verdict
must name this as a LANE PROPERTY, not a bank quality -- otherwise it ranks "who
writes a 144-word monologue" against "who writes 40 short exchanges" and calls
the difference a bank effect. See `tmp/bakeoff_metrics.py`.
