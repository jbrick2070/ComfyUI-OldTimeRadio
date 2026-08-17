# r2 -- DRIVER's own coding-plan review (panelist, written before reading either lane)

**Written:** 2026-08-17, HEAD `d6ec6f53`, with the two agy r2 lanes still in
flight and their output unread. Per the standing rule, Cowork is a code-aware
grounded PANELIST as well as the sole judge, and the panelist review is written
FIRST so it is not anchored by the panel's. Cited by symbol; every claim below
was read at the file this session.

## VERDICT: build-ready, with three amendments and one build-breaker to avoid

The r1 plan is sound. Grounding the actual edit sites turns up one hazard that
would kill renders, one insertion point that is cleaner than the plan assumed,
and one row that is not yet specified tightly enough to implement.

## BUILD-BREAKER -- do NOT add `work_title` to the starvation guard

`compose_announcer_intro` calls `safe_open_viability(setting=..., opening_status_quo=...,
cast=...)` and, on a non-empty result, **raises `AnnouncerBriefStarvedError`
before the model call**, with the comment *"Raising BEFORE the call means a
starved brief cannot become a line at all."*

**If `work_title` is added to that viability check, every `media_archive` episode
dies**, because that lane has no work title at all -- `identity_from_meta`
returns `""` for it by design. That is the one outcome the operator's rule
forbids: a render must degrade, never raise. It is an easy mistake to make,
because adding a field to a dataclass and then to its adjacent validator is the
natural motion.

**Rule for the diff: `work_title` is OPTIONAL FOREVER.** It is never required,
never validated, never starves a brief. Add a test that pins exactly this --
construct a `SafeOpenBrief` with `work_title=""` and assert
`compose_announcer_intro` still composes.

## THE INSERTION POINT IS CLEANER THAN r1 ASSUMED -- no branching needed

r1 row 1 says "render as `WORK:` when non-empty". **That conditional already
exists and is free.** `compose_announcer_intro` builds its context as:

```
context = "\n".join(
    f"{label}: {value}"
    for label, value in ( ("SETTING", ...), ("TIME", ...), ... )
    if value
)
```

The trailing `if value` was added deliberately -- the comment says a bare
`"SETTING:"` with nothing after it *"reads to the model as a form to fill in
rather than material to write from."* So adding `("WORK", clean_one_line(
safe_open_brief.work_title))` to that tuple is a **one-line change that
self-omits on every lane without a work title**, with no `if` of its own and no
media_archive branch. The plan should say so, or an implementer will write a
conditional that duplicates a guard already there.

**Where in the tuple matters.** Put `WORK` FIRST, above `SETTING`. Fable's
adopted seam line reads *"Sentence 1 names tonight's work from the WORK line and
places the listener using only the SETTING and the cast list"* -- the prompt
names WORK before SETTING, and a context block whose order contradicts the
instruction order is a small, free way to make the instruction harder to follow.

## AMENDMENT 1 -- the attribute-access doctrine applies to the new field

The same function carries a hard-won comment: *"DIRECT attribute access, never
getattr-with-default. The default is what hid this: the builder read a `hook`
attribute SafeOpenBrief has never defined, so HOOK was silently empty on every
episode."*

So the diff must read `safe_open_brief.work_title` directly, **never**
`getattr(safe_open_brief, "work_title", "")`. The dataclass default `= ""`
supplies the safety; a getattr default would re-create the exact bug that comment
memorializes -- a field silently empty on every episode, green everywhere. **This
is a real risk here** because `work_title` is being added as an optional field,
which is precisely the shape that tempts a defensive getattr.

## AMENDMENT 2 -- row 10 ("a scene from") belongs in ONE place, and the plan does not say which

Fable offered two forms and left the choice to r2: render `WORK: a scene from
Twelfth Night`, or keep `WORK: Twelfth Night` and let the seam sentence carry it.
**Take the seam, not the value.** Reasons:

* The composer's job is to state FACTS; `work_title` is a bibliographic value and
  should stay one. Baking "a scene from" into the value makes the string
  ungrammatical anywhere else it is ever rendered, and this value comes from
  `identity_from_meta`, which is the shared bibliographic authority used by the
  coda and provenance surfaces too.
* It is the same authority boundary item C settled and PBUG-20260817-01 was
  about: the field owns the fact, the prompt owns the phrasing.
* A value carrying presentation is also how the `_neg_source` lie happened
  (H-receipt) -- one value asserting two things.

**So: `WORK: <title>` verbatim, and the seam sentence carries "a scene from".**

## AMENDMENT 3 -- row 8's replacement needs its corpus named, or it cannot be written

"Assert the frame carries no proper name belonging to a DIFFERENT manifest row"
is right in principle and underspecified in practice. `play_title` alone is not
enough: the shipped defect said *"Verona ... Capulets and Montagues"*, and
**none of those three strings is a `play_title`** -- the row's title is "Romeo
and Juliet". A check that only compares titles would have MISSED the very defect
it is written for.

The manifest does carry the material: each row has a `synopsis` naming its own
places and houses ("In Capulet's garden", "a cold platform at Elsinore",
"Antipholus of Ephesus"). So the check's corpus should be **the other rows'
`play_title` PLUS the distinctive proper nouns of their synopses**, built from
the manifest at test time rather than hand-listed -- a hand-listed blocklist is
the nine-phrase blocklist that a QA pass walked 14 of 16 ways past on 2026-08-17.

**And state the known limit honestly:** this catches CROSS-PLAY leakage only. It
cannot catch a wholly invented place that belongs to no row. That residue is what
the live leg is for.

## WHAT I AGREE WITH AND WOULD NOT CHANGE

* `identity_from_meta(meta).work_title` as the single source -- one symbol, both
  adaptation lanes, never raises, `""` when degraded.
* Both producers in the same change; no dialogue parsing to recover a value we
  already hold.
* `OutlineRequest` threading, raised to MUST -- otherwise the frame names the
  right play and describes the wrong place.
* The live leg. A prompt-capture test proves the string ARRIVED, not that the
  announcer USED it, and this repo has proven twice in one day that a green gate
  is not a working fix.

## ORDERED DIFF PLAN

1. `_otr_line_composer.SafeOpenBrief`: add `work_title: str = ""` **last** in the
   dataclass -- it is positional-safe there and every existing construction site
   keeps working unchanged.
2. `compose_announcer_intro`: add `("WORK", clean_one_line(safe_open_brief.work_title))`
   as the FIRST entry of the context tuple. Direct attribute access. **Do not
   touch `safe_open_viability`.**
3. `fallback_safe_open`: name the work when non-empty, same direct access.
4. `OTR_LedgerScriptWriter`, BOTH `_OTRLC.SafeOpenBrief(` construction sites --
   the first producer and the I.4.9 rewrite -- pass
   `work_title=_otr_source_identity.identity_from_meta(meta).work_title`.
5. `_otr_outline.OutlineRequest`: add `work_title: str = ""`; render it in
   `_build_macro_user_prompt` so `_MacroShape.setting` stops minting the wrong
   place. Thread it at the writer's `OutlineRequest(` call.
6. Pack seams, shakespeare AND public_domain: Fable's two sentences, ASCII.
7. Tests: prompt-capture on both producers and both banks; the empty-`work_title`
   composes-anyway pin; the cross-play leak check built from the manifest; and a
   pin that `safe_open_viability`'s signature does NOT mention `work_title`.
8. Suite + Bug Bible, then Sonnet 5 QA on the finished diff BEFORE the push.

## THE ACCEPTANCE CLAIM I WILL NOT LET US MAKE

No unit result may be reported as "the wrong-play frame is fixed". The honest
claim after a green suite is *"the title now reaches both producers and no
cross-play name appears in the captured prompts"*. **"Fixed" needs the live leg**,
and per the operator's 2026-08-16 ruling that batches into his GPU session rather
than blocking the code.
