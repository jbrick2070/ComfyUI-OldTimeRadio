# Driver anchor — make the Shakespeare lane actually verbatim

**Driver:** Claude (Cowork, 5080). Written BEFORE fan-out, from a live episode
and the real files. **The operator has confirmed the intent: "shakespeare should
be verbatim."** This is a fidelity defect, not a quality preference — it sits in
the class the 2026-08-04 "story quality is done" directive explicitly leaves
open.

## The defect, proven on a published episode

`signal_lost_a_tapestry_of_drifting_visions_20260830_111740`, rendered with the
bank PINNED (`--source-bank shakespeare`, not rolled), so attribution is exact.

The announcer says *"A Midsummer Night's Dream, Act Three, Scene One."* The
delivered dialogue:

    BOTTOM   "I am here! I stand amidst these ancient trunks, a man of
              substance, and I command thee, spirit of the air, look upon me!"
    TITANIA  "Thou art a motley intruder, a crawling thing of clay!"
    PUCK     "My Queen, the heavy weight of sleep invites thee now..."

**Not one line is Shakespeare's.** MND 3.1 is the ass's-head scene; Titania
wakes with *"What angel wakes me from my flowery bed?"* None of the source text
appears. Right play, right act, right scene, right three characters — entirely
invented words.

## What is ALREADY wired (CONFIRMED — do not rebuild these)

1. **The Folger source is fetched and reaches the writer.** The same leg logged
   `seed_source=shakespeare_folger` plus the CC BY-NC 3.0 non-commercial
   warning, which only fires when real Folger text is in hand.
2. **`FolgerScene`** (`_otr_shakespeare_sources.py`) carries `speakers`,
   `stage_directions` and the scene `text`. `parse_folger_scene` handles
   `folger_txt`, `folger_xml` and `curated_scene_text`.
3. **`source_document` carries the complete uncapped body** and is TRANSIENT by
   design — `SourceFetchResult` documents that consumers hold it for a build and
   persist offsets plus hashes rather than the body.
4. **`_otr_source_document.SourceSpan` exists**, with `span()`, `_make_span` and
   **`verify_span()`** — i.e. there is already a way to cite an exact range of
   source text and check it later.
5. **The document already reaches the writer** —
   `OTR_LedgerScriptWriter.py:3751` reads `resolved.get("source_document")`.

## What is MISSING (CONFIRMED)

* **Nothing anywhere instructs the writer to reproduce the source.** The string
  "verbatim" does not appear on the Shakespeare path at all — only in unrelated
  contexts (voice delegation, image seeds).
* **`exchange_compose` — the mechanism the operator's 2026-08-23 ruling named as
  "not run" on this lane — does not exist in the codebase.** It was removed or
  renamed, and whatever fidelity it enforced went with it. The ruling now
  describes machinery that is gone.

**So this is not a broken wire.** The material arrives every time; nothing asks
the writer to use it, so it writes its own scene. It is doing what it is told.

## The hard constraint the plan must solve

**A Shakespeare scene is longer than an episode can speak.** The beat topology
tops out near 1,520 spoken words; MND 3.1 is well past that. So "use the source
verbatim" cannot mean "use all of it", and the plan's real content is what
happens to the remainder.

The operator's standing rules bound the answer:
* **No word-count chasing** (2026-08-03) — the target is a REQUEST, never a gate.
  A verbatim plan must not become a refusal when a scene does not fit.
* **Neither adaptation lane may invent a setting or story** (2026-08-23).
* **The author's own language is carried as written** — profanity/violence
  filtering must not creep back in on this lane.

## The driver's proposed shape — ATTACK IT

Per an earlier operator note, the scope here is *"a dialogue extractor plus one
prompt seam, not a three-leg campaign."* Concretely:

1. **Extract speaker-attributed dialogue spans** from `FolgerScene.text` into
   `SourceSpan`s — speaker, line range, offsets.
2. **Select a contiguous run that fits the budget**, rather than sampling
   across the scene, so the delivered excerpt is coherent drama.
3. **Hand those lines to the writer as THE LINES**, not as inspiration; the
   writer's remaining job is beat/shot structure, cast mapping and the
   announcer frame.
4. **Verify with `verify_span()`** — an automated check that delivered dialogue
   matches its cited source range, so this cannot silently regress again.

## Questions for the panel

1. **Is the contiguous-run selection right**, or should it be scene-summary plus
   verbatim key exchanges? The first is faithful but arbitrary in where it cuts;
   the second reintroduces invented prose.
2. **What owns cast sizing here?** MND 3.1 has more speakers than the default 2.
   The 2026-08-03 ruling says THE SCENE WINS. Should the Folger `speakers` tuple
   drive `num_characters` directly?
3. **What is the failure mode when a scene cannot fit at all?** Refusing
   violates no-word-count-chasing; truncating mid-exchange is bad drama.
4. **Should the announcer frame quote or paraphrase?** It currently invents
   ("What secrets will be unearthed...").
5. **Cheapest verification** that a rendered episode is verbatim — ideally a
   test, not a listen.
6. **Does any of this apply to `public_domain`?** That lane may PARAPHRASE by
   ruling, so the extractor may be shakespeare-only. Say so if not.

## Out of scope

Story quality on non-adaptation banks, the visual authored-path bug, PBUG-11,
the zero-touch install work, the prefs guide.
