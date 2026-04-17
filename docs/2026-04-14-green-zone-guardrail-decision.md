# Green Zone Guardrail Decision — v2.0-alpha Ship Prep

**Date:** 2026-04-14
**Status:** Decision made, implementation pending overnight soak confirmation
**Rationale:** Empirical — post-fix era soak data (runs 234-245) shows clear success/failure profile by parameter. Rather than fix every failure mode, guardrail to the parameters that already work. Ship v2.0-alpha in the narrow green zone, expand coverage in later releases.

---

## Decision

Restrict the soak dropdown options (and any user-facing UI defaults) to the **green zone** only. Hide or remove parameter values empirically shown to fail or timeout. Re-widen the allowed set only after the specific failure modes for each currently-excluded value are fixed and verified.

---

## Green zone (ALLOWED for v2.0-alpha ship)

| Parameter | Allowed values | Evidence |
|---|---|---|
| `words` | 350, 700, 1050, 1400 | 100%, 75%, 100%, 67% success in post-fix |
| `length` | medium (5 acts), short (3 acts) | 75% and 50% success |
| `genre` | dystopian, hard_sci_fi, first_contact, space_opera, cyberpunk, post_apocalyptic | All non-zero success in post-fix |
| `style` | space opera epic, tense claustrophobic, chaotic black-mirror, noir mystery, hard-sci-fi procedural | All non-zero success |
| `creativity` | safe & tight, balanced, wild & rough | All three cluster 50-67% |
| `profile` | Standard | 57% success; Pro shows no advantage (60%) and costs more VRAM |

**Default combo (safest known):**
- words=700, length=medium (5 acts), genre=dystopian, style=space opera epic, creativity=safe & tight, profile=Standard

---

## Red zone (BLOCKED until fixed)

| Parameter value | Reason | Fix deferred to |
|---|---|---|
| `words=2100` | 0 for 3 post-fix, all TIMEOUT at 30-min cap | v2.1 — needs operator timeout raised or LLM phase optimization |
| `length=long (7-8 acts)` | Only 50% post-fix, 3.3% in broader window | v2.1 — long arcs regress consistently |
| `style=psychological slow-burn` | 0 for 9 across all windows — writer can't close arc | v2.1 — needs dedicated prompt tuning |
| `genre=time_travel` | 0 for 9 in broader window | v2.1 — paradox prompting unresolved |
| `genre=cosmic_horror` | 0 for 9 in broader window | v2.1 — scale/tone prompting unresolved |
| `profile=Pro (Ultra Quality)` | No measurable benefit over Standard, higher VRAM | hold indefinitely; revisit when benchmarking proves a benefit |

---

## Implementation steps (execute after overnight soak confirms profile)

**Step 1 — Confirm the profile holds.**
At morning review, re-run the param-vs-result cross-tab on the overnight data. If green-zone parameters still show ≥50% success and red-zone still shows ≤20%, proceed with the guardrail. If the profile shifts materially, re-profile and revise this doc before implementing.

**Step 2 — Update soak operator dropdowns.**
File: `scripts/soak_operator.py`
Edit the PRESET_OPTIONS / GENRE_OPTIONS / STYLE_OPTIONS / etc. lists to only include green-zone values. Comment out (not delete) the red-zone values with a reference to this doc, e.g.:

```python
# RED ZONE — blocked until v2.1 per docs/2026-04-14-green-zone-guardrail-decision.md
# "words": [2100],
# "length": ["long (7-8 acts)"],
# "style": ["psychological slow-burn"],
# "genre": ["time_travel", "cosmic_horror"],
# "profile": ["Pro (Ultra Quality)"],
```

**Step 3 — Update workflow JSON defaults.**
File: `workflows/otr_scifi_16gb_full.json`
Set the default dropdown selections to the safest-known combo above. Do not modify dropdown OPTIONS (those come from node INPUT_TYPES).

**Step 4 — Update node INPUT_TYPES dropdowns (user-facing UI).**
Files: whatever OTR nodes expose these dropdowns (likely in `nodes/*.py`)
Remove red-zone values from the COMBO lists in INPUT_TYPES. This is the user-facing change — the UI dropdown will only offer green-zone values.

Critical: any INPUT_TYPES or widgets_values change triggers the Bug Bible regression suite, dropdown guardrails test, and core tests per CLAUDE.md. Run all three before commit.

**Step 5 — Add test coverage.**
File: `tests/test_green_zone_guardrail.py` (new)
Assert each red-zone value is NOT in the user-facing dropdown options. This locks the guardrail so a future change can't silently re-introduce a broken value.

**Step 6 — Document in README.**
Add a short "Known limitations" section noting the green-zone-only ship and pointing at this doc for the red-zone list.

**Step 7 — Commit.**
Subject: `Guardrail v2.0-alpha to empirical green zone`
Body: link to this doc, summarize what moved red vs green.

---

## Re-opening red-zone values

A red-zone value returns to green when:
1. A dedicated bug ticket is filed (BUG-LOCAL-NNN) identifying the root cause of its failures.
2. A fix is implemented and verified with ≥10 runs on that specific parameter value, showing ≥50% success.
3. The fix passes the Bug Bible regression suite.
4. This doc is updated to move the value from red to green, with the new evidence linked.

No value returns to green on vibes. Data or nothing.

---

## Ship criteria for v2.0-alpha

With this guardrail in place, v2.0-alpha is shippable when:
- Overnight soak on green-zone-only config shows ≥75% success over ≥30 runs
- No new error class appears (i.e. no new BUG-LOCAL entries in the overnight window)
- Audio output remains byte-identical to v1.5 baseline (C7)
- VRAM peak stays under 14.5 GB (V1)

If those four conditions hold tomorrow morning, cut the v2.0-alpha tag and merge to main.

---

## Out of scope for this decision

- Fixing any red-zone value (deferred to v2.1 scope)
- Adding HyWorld visual pipeline (v2.1 per separate spec)
- Changing the writer LLM, TTS voices, or audio pipeline
- VRAM Guardian test isolation bug (BUG-LOCAL-034 candidate — separate fix)

---

*Execute this doc's Step 1 first thing at morning review before taking any other action on the codebase.*
