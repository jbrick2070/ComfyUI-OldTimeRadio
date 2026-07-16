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

## OBS-1b -- SCHEMA CAPS RAISED MID-SWEEP: the 420 rung is now CONFOUNDED

**Operator directive 2026-07-14, eyes open:** *"raise it now before we spend on
720"* / *"I know we lose our scientific rigor but time is short."* Recorded so the
verdict cannot quietly overstate what the 420 numbers mean.

Raised (evidence-based -- ONLY fields the model was observed overshooting on live
legs, never a blanket loosening):

| Field | Was | Now | Live evidence |
|---|---|---|---|
| `CastPlanRowV4.role_in_conflict` | 120 | **180** | killed a leg pre-clamp; tripped the clamp AGAIN on the very next leg ("coerced 2 over-long field(s)") |
| `StructureReviewV4.rationale` | 240 | **400** | overshot twice on one leg and killed it (f6c42c5f) |

Deliberately NOT loosened: P0's `source_spans` quotes. Those are VERBATIM slices of
the article -- trimming one would forge a citation. A cap that guards PROVENANCE is
a different animal from a cap that guards a CONTEXT BUDGET, and only the latter was
ever negotiable.

**THE CONFOUND, stated plainly.** Each leg boots a fresh server, so the caps land on
whatever leg starts next. At 420: the 7 banks that went green in the main sweep ran
at cap=120/240; the re-legged banks run at 180/400. **420 is therefore NOT a clean
bank-vs-bank comparison and must never be reported as one.** The effect is small
(these are non-spoken tag fields; the only reason they matter at all is that
`role_in_conflict` is re-serialized into P3's prompt), but it is real and it is
one-directional.

**What protects the verdict:** the judged rung is **720**, and every 720 leg runs
AFTER this change, on identical code. 720 stays uniform -- the bank remains the only
variable there, which is the whole premise of the bake-off. The 420 rung was always
a LADDER STEP (prove the lanes survive at length), not the judged artifact.

So: report 420 as a survival/structure rung, with this caveat attached. Rank on 720.

---

## OBS-1c -- the adaptation lanes were told to REWRITE the thing they adapt

**Operator, by ear, 2026-07-14:** *"shakespeare and pub domain are trying to force a
radio drama narrative onto an already proven narrative... they should mic the source
material as much as possible, but fit within the ledger scope."* Prompt-only fix,
before the 720 spend.

He is right, and the packs were arguing with themselves.

`public_domain_story` and `shakespeare` are ADAPTATION lanes: unlike `science_news`
or `media_archive` -- which are handed a NON-story (a news item, an archive record)
and must invent a drama around it -- these two are handed **a story that already
works**. Their packs know it. `faithful_radio_adaptation` and
`folger_scene_adaptation` both declare:

> "Faithfulness outranks novelty." / "Compression is allowed; replacement is not."
> forbidden: invented protagonist, changed ending, unrelated framing story

and their `line_composer_system` says "Ground every character line in the source
text ... never invent connective events, characters, or endings."

**And then `exchange_system` told the model to invent modern drama anyway:**

> "Write an EXCHANGE ... naturalistic, with subtext.
>  - Characters should not answer each other too directly.
>  - At least one line avoids the real question."

Those are GENERATIVE craft rules for original drama, and they were being applied to
Dickens and to Shakespeare. On the Shakespeare lane it is worse than generic: telling
a model to make the verse "naturalistic" and to stop characters "answering too
directly" is an instruction to rewrite rhetoric into modern evasive patter.
Shakespeare's people answer each other directly, at length, in public -- the rhetoric
IS the subtext. The pack contradicted its own law two keys later, and the craft rule
won, because the craft rule was the one talking to the model at dialogue time.

**Fix (seam-only, no Python -- per the standing law on framing defects):** rewrote
`exchange_system` in both packs from an INVENTION seam into an ADAPTATION seam --
carry the source's own words, keep its diction and rhetoric, convert narration into
speech those characters would plausibly say, and impose NO subtext the source does
not have. Added to both `tone_guardrails`: *"Put a MICROPHONE on the source; do not
re-plot it"* and *"Faithfulness outranks craft rules. A craft rule that would rewrite
a proven scene is wrong."* Added the operator's ledger-scope point to the outline
seam: the radio cast is small, so CHOOSE the source's essential speakers and fold or
drop minor figures -- never add a character to fill a scene. Also pinned the spine:
`premise` / `central_tension` must RESTATE the question the source already asks,
never a new one invented for radio.

**Checked and deliberately NOT changed:** `story_rules/{shakespeare,public_domain_story}.json`
is an anti-CLICHE lexicon ("the bard speaks", "the theme is", "the lesson is"), not a
force-subtext rule -- it helps an adaptation too, since Shakespeare never writes "I'm
so ambitious". The shared `_otr_slot_drama_contract` maps objective/obstacle/turn onto
each beat, which is legitimate scaffolding for an adaptation as much as an invention.
The forcing was in `exchange_system`, and only there.

**Confound:** shakespeare and public_domain_story ran their 420 legs on the OLD seams.
420 was already confounded (OBS-1b). Every 720 leg runs on the new seams, so the
judged rung stays uniform.

**OPERATOR RULING (2026-07-14, mid-720):** *"we can re-run 420 for them later, I just
want them to get the best overall scoring, but we do need to re-run those two."* So:
the VERDICT scores shakespeare and public_domain_story on their **720 (new-seam)**
episodes ONLY -- their old 420 episodes ("The Ink That Shows" etc.) are pre-fix and
must NOT be what a judge sees for these two banks. GATE 3 scores the 720 rung, so this
is automatic; pinned here so it cannot slip. A same-length 420 A/B on the new seams is
a POST-sweep nice-to-have, not a blocker. The 720 sweep is NOT interrupted -- shakespeare
is already live on the new seams as this was written.

---

## OBS-2 -- lane structure at length (EXPECTED, not a defect)

The lanes do NOT write the same shape of show at length, and the operator has
ruled that target words are not chaseable. RECORD, never normalize. The verdict
must name this as a LANE PROPERTY, not a bank quality -- otherwise it ranks "who
writes a 144-word monologue" against "who writes 40 short exchanges" and calls
the difference a bank effect. See `tmp/bakeoff_metrics.py`.
