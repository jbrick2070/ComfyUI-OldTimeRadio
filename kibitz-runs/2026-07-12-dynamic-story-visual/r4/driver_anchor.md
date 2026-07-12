# r4 driver anchor -- dynamic-story-visual (CONVERGENCE)

Author: Claude (Cowork), docs-only architecture owner + sole judge. Panel: codex
`gpt-5.6-sol` @ ultra, antigravity `gemini-3.5-pro`. Doc under review: rev 4.

Timing note (honest): the r4 fan-out was launched a few minutes before this anchor
was written. The substantive requirement -- that the driver's review be INDEPENDENT
of the panel's -- holds: nothing from the r4 panel had been read, or had even been
written to disk, when this was authored.

Purpose of r4: confirm CONVERGENCE. Not a fresh critique surface -- a check that
(a) every survivor of r1/r2/r3 actually landed in the doc, (b) no fold introduced a
new contradiction, and (c) nothing in rev 4 is still unbuildable. A round that only
produces new must-fixes means r2/r3 did not converge and the arc must re-loop.

---

## A. Fold audit -- did every ruled survivor land?

| Round | Ruling | In rev 4? |
|---|---|---|
| r1 | projection hashing, not raw rows | YES -- 2.5 `DirectionSourceV1` (superseded the projection, same purpose) |
| r1 | ledger-aware resolve seam | YES -- 5.3 (upgraded to a typed `ResolvedDirection` bundle in r3) |
| r1 | sentinel gate fix at the writer | YES -- 6 |
| r1 | ShotLock swallow hoist | YES -- 7.2 |
| r1 | safety-base pack split | YES -- 2.4 |
| r1 | look-authority precedence | YES -- 5.4, extended to the video lane in r2 |
| r1 | reroll cut, scenes[] cut | YES -- 2.7 |
| r2 | two typed models (draft vs artifact) | YES -- 2.1/2.2/2.3 |
| r2 | `structured_call` ladder replaces "2 attempts" | YES -- 4.5 |
| r2 | `DirectionSourceV1` closed DTO | YES -- 2.5 |
| r2 | semantic-hash preimage enumerated, no timestamps | YES -- 2.3 |
| r2 | MetaBrief + render_driver added as surfaces | YES -- 8.12, 8.14 |
| r2 | teardown barrier across three nodes | YES -- 5.5 |
| r2 | must-fit context guard | YES -- 4.3 (generalized to the provider-effective interface in r3) |
| r2 | P-A/P-B split | YES -- 4.4 |
| r2 | D2 creative (fable2 precedent) | YES -- 4.2 |
| r2 | literal `_NODE_MODULES` registration | YES -- 8.2 |
| r2 | model-diversity ladder, PROD_BUG_LOG, sprint receipt | YES -- 9.2, 9.3, 9.4 |
| r3 | no `max_length` on authored strings (the silent clamp) | YES -- 2.1, test 9.1.2 |
| r3 | per-kind consumption matrix | YES -- 7.4, test 9.1.14 |
| r3 | preflight order + worst-case repair envelope | YES -- 4.4 |
| r3 | provider-effective interface; Google has no constrained branch | YES -- 4.3, 9.2 |
| r3 | attempt-event sink on `structured_call` | YES -- 4.5, 8.7 |
| r3 | lifecycle order (teardown BEFORE seal/persist) | YES -- 5.2 |
| r3 | `freeze_unload_ok` precondition | YES -- 5.2 step 1 |
| r3 | dispatcher cache-HIT clone -> explicit digest on both paths | YES -- 7.5, test 9.1.18 |
| r3 | literal node record + `gate_in` cut | YES -- 8.1, 8.3 |
| r3 | channel-isolation test | YES -- 9.1.9 |
| r3 | quote anchors on factual evidence | YES -- 2.2, 3.3, 9.1.6 |
| r3 | `content_mutations` cut; `binding` vs `story_binding` naming | YES -- 2.3 |
| r3 | ordering edge REJECTED with a reopen trigger | YES -- 10, I4 |

Nothing ruled is missing.

## B. My own remaining concerns about rev 4 (the honest list)

1. **The doc now specifies a change to shared machinery** (`_otr_structured_call.py`
   gets an attempt-event sink, 8.7) and a NEW shared interface
   (provider-effective config, 8.8). Both are additive and backward-compatible as
   written, but they widen the blast radius beyond the visual lane. If Codex judges
   either too invasive, the fallback is a direction-local wrapper that records what
   it CAN see (raw calls + the final exception) and records `null` for the rungs it
   cannot -- an honest receipt is better than an invented one. Worth the panel's view.
2. **The still-target set is derived twice** -- once by the direction node (to know
   which lines get `shots[]` rows) and once by MetaBrief (to compose them). 2.6 says
   both call the SAME pure helper, which is correct, but that helper
   (`derive_scene_still_targets` / `_iter_beat_lines`) currently reads `lines` and
   policy inputs that the direction node may not have in identical form (still
   ASPECTS and role policy arrive via `image_policy_json` from the ImageDirector --
   which the direction node does NOT receive). If the target set depends on
   image-policy inputs, the two derivations can DIVERGE, and the 7.1 "shots set ==
   target set" gate would fire spuriously at MetaBrief. This is the sharpest
   remaining risk in the design and I want the panel on it specifically.
3. **`P-A` truncation.** 2.5 allows a deterministic head-truncation of line text in
   `DirectionSourceView_PA` when the budget demands it, and records the truncation
   point. That is honest, but it means on a long episode the look pass reasons over
   partial text while the evidence quotes must still be verbatim substrings of the
   FULL DTO text. A quote drawn from a truncated preview is still a valid substring
   of the full text, so the check holds -- but the "cite only what you saw" rule and
   the truncation interact, and I want it confirmed rather than assumed.
4. **Live-smoke feasibility.** 9.2 requires a second local family (gemma-4-E4B) and a
   cloud creative lane. The gemma family is the one that produced two wrong-depth
   PBUGs this week; there is a real chance the direction pass simply cannot be made
   to pass on it without constrained generation (I2). That is a QUALIFICATION risk,
   not a design defect -- but the doc should say what happens if a family fails
   qualification (answer: the feature ships with a DECLARED supported-model list, it
   does not silently degrade). That line is currently missing.
5. **Scope.** Section 8 now names fifteen surfaces. This is a big feature. If the
   operator wants a smaller first cut, the natural v0 is: P-A only (no `shots[]`, no
   per-kind matrix, no dispatcher provenance) -- an episode-specific PACK, nothing
   per-line. That would remove surfaces 12 (partly), 13, 15 and roughly half the test
   plan. I am NOT recommending it (the per-shot notes are the operator's stated
   product intent), but the doc should name the fallback so the choice is explicit
   rather than discovered mid-build.

## C. Questions put to the panel

1. Is there any NEW must-fix in rev 4 -- a defect introduced BY a fold, or a
   contradiction between two sections that r2/r3 did not have? Name it with file:line
   or section numbers. If there is none, say so plainly; do not manufacture findings
   to fill the template.
2. Concern B2 (the still-target set derived twice, possibly from different inputs) --
   is it real? Read `derive_scene_still_targets` / `_iter_beat_lines` / the
   `image_policy_json` plumbing and tell me whether the direction node can compute the
   SAME target set that MetaBrief will, from the frozen ledger alone.
3. Concern B3 (P-A truncation vs verbatim quotes) -- does the design hold?
4. Is the doc now BUILDABLE by Codex without further design input? If not, name the
   single biggest remaining gap.
