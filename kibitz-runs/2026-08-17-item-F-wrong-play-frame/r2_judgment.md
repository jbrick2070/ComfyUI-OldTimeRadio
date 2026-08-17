# r2 judgment -- item F coding plan

**Driver:** Claude (Cowork) -- **panelist AND sole judge.** My own r2 review
(`r2_driver_review.md`) was written and committed BEFORE either lane's output was
read, so it is an independent review rather than a reaction. **Date:** 2026-08-17.
**HEAD:** `d6ec6f53`.

**Lanes, each in its own `--topic` (the r1 collision is fixed):**
`Gemini 3.1 Pro (High)` -> `kibitz-runs/2026-08-17-item-F-r2-pro31/r2/`,
`Gemini 3.7 Flash (High)` -> `kibitz-runs/2026-08-17-item-F-r2-flash37/r2/`.
Both `agy_model_selected.txt` files confirm distinct models; 1,978 and 7,222
bytes. **r1+r2 of four. Codex joins 2026-08-19 20:31. NOT a full arc.**

---

## THE PANEL CAUGHT A BUILD-BREAKER IN THE DRIVER'S OWN DIFF PLAN

**Flash MUST-FIX 1 is CONFIRMED, and it would have crashed the majority of
episodes.** Verified at the file:

* `_otr_source_identity` is imported **only inside** the branch
  `if bool((_source_bank_row.defaults or {}).get("provenance_normalize", False)):`
  -- the `from . import _otr_source_identity as _OTRSID` and the
  `_identity = _OTRSID.identity_from_meta(meta)` assignment both live there.
  There is **no module-scope import**.
* `banks.json` has **six banks: two with `provenance_normalize=True`, four with
  `False`.**
* `SafeOpenBrief` is constructed on the bank-agnostic path, for every lane.

So r1 row 2 and **my own r2 ordered-diff step 4** -- "pass
`work_title=_otr_source_identity.identity_from_meta(meta).work_title`" -- would
raise `NameError`/`UnboundLocalError` on **four of six banks**, i.e. every
`original`, `media_archive` and news-lane episode. Not a lint issue: a dead
render on the majority of the corpus, and it would have passed any test that only
exercised an adaptation lane.

**Adopted, Flash's fix verbatim in shape:** import `_otr_source_identity` at
MODULE scope and compute `_work_title = ...identity_from_meta(meta).work_title`
**unconditionally**, once, before the first construction site; pass that local to
`OutlineRequest` and both `SafeOpenBrief(` sites. `identity_from_meta` never
raises and returns `""` on an unknown lane, so unconditional evaluation is safe
by that function's own contract.

**This is the second time in one day a panel caught an execution-order/binding
error in a driver claim** (the H-receipt round was the first). The pattern is
identical and worth naming: **I reason about where a value is COMPUTED and forget
where it is BOUND.**

## PRO CAUGHT A SHIPPING HAZARD, AND IT IS BROADER THAN EITHER LANE SAID

**Pro MUST-FIX 1 is CONFIRMED with an important correction.** The seam is
pack-routed: `_ANNOUNCER_INTRO_SYSTEM_SAFE` lives in `_otr_line_composer` and is
imported into `_otr_creative_prompt_router` as
`_MODERN_ANNOUNCER_INTRO_SAFE_SYSTEM` -- the MODERN DEFAULT, not the adaptation
text. Editing the Python constant would not change what a shakespeare episode
sees. Pro is right that the JSON is the shipping surface.

**Where BOTH lanes are wrong, and it matters:** Pro says update the seam "in all
JSON `source_banks` files"; Flash says shakespeare **and** public_domain.
**`announcer_intro_safe_system` appears in SIX files** -- `banks.json`,
`pipelines.json`, `media_archive/`, `original/`, `public_domain/`,
`shakespeare/`.

**The seam change must land ONLY on the lanes that actually receive a `WORK`
line.** Fable's sentence says *"names tonight's work from the WORK line"*. On
`original` and `media_archive`, `identity_from_meta` yields `""`, the context
tuple's `if value` filter omits `WORK` entirely, and the seam would then instruct
the announcer to name a work from a line that is not there -- **manufacturing the
exact starvation-vacuum that caused this defect, on the two lanes that never had
it.** Pro's "all JSON files" would ship that.

**Adopted:** seam wording changes on `shakespeare` and `public_domain` ONLY.
Flash MUST-FIX 2 is upheld and Pro's broader scope is REJECTED. Flash's reasoning
for including public_domain is correct and I had it too narrowly in r1 row 6:
that lane DOES get a `work_title`, so leaving its seam saying "use only the cast
names" creates a prompt that supplies `WORK:` and forbids using it.

**Also adopted:** `tests/test_story_pack_stage1.py` carries a byte-identity
section (*"(a) byte-identity + (b) exact seam set"*), so the seam edit updates
those assertions in the same commit or the suite goes red. Pro found this; I did
not have it.

## THE ONE FORK: where "a scene from" lives -- I OVERRULE MYSELF AND FLASH

Three positions, and this is a genuine three-way split:

| Who | Position |
|---|---|
| **Pro (SF1)** | Render `WORK: a scene from <title>` in Python at the composer |
| **Flash (SF2)** | Keep `WORK: <title>`; let the seam sentence carry "a scene from" |
| **Driver (my Amendment 2)** | Same as Flash -- seam, not value |

**PRO WINS, and my own r1 evidence is what defeats my position.** The seam
already contains *"Use ONLY the proper names in the cast list below; invent
none."* -- and `Verona`, `Capulet` and `Montague` are proper names. **That
instruction already forbade the shipped defect and did not hold.** So resting
"a scene from" on a seam sentence rests it on the one mechanism this item has
already proven fails on this exact prompt. Fable said the same thing from the
other side: supply material, do not instruct -- *"the title becomes a label, not
a seed."*

**Flash's objection does not apply to what Pro actually proposed.** Flash says
`a scene from` "pollutes the raw bibliographic field". It would -- if it were
written into `work_title`. Pro puts it in the COMPOSER's label rendering, leaving
`identity_from_meta(...).work_title` untouched as the clean bibliographic value.
The authority boundary I was defending is preserved: the field owns the fact, the
composer owns the phrasing.

**RULING: render `WORK: a scene from {work_title}` at the composer.** The value
stays clean. The seam still gets Fable's sentence, as belt-and-braces, but the
GUARANTEE is the rendered label, not the instruction.

## DEFECTS IN THE PANEL'S OWN PROPOSED CODE -- discarded

1. **Flash's fallback patch uses `getattr(safe_open_brief, "work_title", "")`.**
   `compose_announcer_intro` carries an explicit, hard-won prohibition:
   *"DIRECT attribute access, never getattr-with-default. The default is what hid
   this: the builder read a `hook` attribute SafeOpenBrief has never defined, so
   HOOK was silently empty on every episode."* Flash's snippet reintroduces
   exactly that failure mode on the new field. **Take the shape of Flash's
   fallback template, reject its accessor:** use `safe_open_brief.work_title`
   direct, with the dataclass default `= ""` supplying safety.
2. **Both lanes cited LINE NUMBERS** (`:4090-4124`, `:4861`, `:6084`, `:16`,
   `:302`) despite the anchor asking for symbols. They happen to be right today;
   they will rot on the first insertion, and this repo proved 8 of 21 citations
   wrong within an hour on 2026-08-17. **Every line number in both reviews is
   re-expressed as a symbol in the adopted plan below.**
3. **Pro's Row 8 "static list of known plays/locations"** is a hand-listed
   blocklist -- the shape a QA pass walked 14 of 16 ways past on 2026-08-17.
   Flash's version builds from the manifest, which is right. **Flash's wins.**

## WHAT NEITHER LANE HAD -- the driver's own build-breaker stands

`compose_announcer_intro` calls `safe_open_viability(...)` and **raises
`AnnouncerBriefStarvedError` BEFORE the model call**. Adding `work_title` to that
viability check would kill every lane without a work title. Neither lane raised
it; Flash's OPTIONAL 2 circles it (asserting a bare title cannot bypass the
check) but does not state the hazard. **`work_title` is OPTIONAL FOREVER: never
required, never validated, never starves a brief**, plus a test pinning that
`safe_open_viability`'s signature does not mention it.

Also mine and unchallenged: the `if value` filter in the context tuple means
`WORK` self-omits with no branch, and `WORK` goes FIRST so the context order
matches the seam's sentence order.

## ADOPTED r2 PLAN

| # | Change | Source |
|---|---|---|
| 1 | Module-scope `_otr_source_identity` import; compute `_work_title` ONCE, unconditionally, before the first construction site | **Flash MF1 -- build-breaker** |
| 2 | `SafeOpenBrief`: `work_title: str = ""` appended LAST; direct attribute access everywhere | driver |
| 3 | `compose_announcer_intro`: `("WORK", ...)` FIRST in the context tuple, rendered `a scene from {title}`; `if value` self-omits | driver + **Pro SF1** |
| 4 | **Never** add `work_title` to `safe_open_viability`; test pins its signature | **driver -- build-breaker** |
| 5 | Both `_OTRLC.SafeOpenBrief(` sites + `OutlineRequest(` take `_work_title` | both lanes |
| 6 | `fallback_safe_open`: Flash's template shape, direct attribute access | Flash MF3, accessor rejected |
| 7 | `OutlineRequest.work_title` rendered in `_build_macro_user_prompt` **and** `_build_user_prompt` (test harnesses use the latter) | **Flash SF1** |
| 8 | Seam wording on `shakespeare` + `public_domain` ONLY -- never the four lanes with no WORK line | Flash MF2, **Pro's scope rejected** |
| 9 | Update the byte-identity assertions in `tests/test_story_pack_stage1.py` in the same commit | **Pro MF1** |
| 10 | Cross-play leak test built FROM the manifest, not a hand-listed blocklist | Flash SF3 |
| 11 | Prompt-capture on both producers, both banks; empty-`work_title` composes-anyway pin | r1 |
| 12 | Suite + Bug Bible, **Sonnet 5 QA on the finished diff BEFORE the push**, then the live leg | standing |

## STILL TRUE

No unit result may be called "the wrong-play frame is fixed". A green suite earns
*"the title reaches both producers and no cross-play name appears in the captured
prompts"*. **"Fixed" needs the live leg**, batched into the operator's GPU
session.

## r3 SCOPE (wiring)

`workflows/otr_canonical.json` conformance, widget/`INPUT_TYPES` audit if any
node surface moves (currently none expected -- this is all internal threading),
and the re-baseline question for the byte-identity seam assertions.
