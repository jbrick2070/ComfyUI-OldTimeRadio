# Every cast member gets a voice, or leaves the ledger

**Operator ruling 2026-08-02:** "every cast member needs a voice -- it's a radio
drama, not a mime show. It needs to either have an LLM write its lines, or
entirely remove the character from the ledger."

**Operator process ruling, same message:** "read / kibitz before doing any
coding updates -- who knows, maybe the past coding runs already did it, maybe
not." This document is that read, written before any code.

## WHAT THE READ FOUND

**This is RECURRING, not new.** `docs/2026-07-18-render-step1-blocker.md` records
the identical failure two weeks earlier:

    smoke2 | scifi_fable2 | writer script pass |
    Fable2ScriptError: pass 'script' failed after 5 attempts;
    markup ladder exhausted (BAD_LINE_SHAPE x2 + CAST_MEMBER_SILENT: ISHIKAWA,
    no fallback)

So a silent cast member has been killing legs since at least 2026-07-18 and has
never been fixed at the root -- the lane just fails loudly, burns its repair
ladder, and dies.

**No prior art implements either half of the ruling.** A grep across
`nodes/*.py` and `docs/*.md` for silent-cast handling, character removal, or
non-speaking-role logic finds only:

| site | what it is |
|---|---|
| `_otr_fable2_markup.py:165` | the `CAST_MEMBER_SILENT` defect CODE |
| `_otr_fable2_markup.py:482` | where the parser RAISES it |
| `_otr_scifi_fable2.py:2306-2317` | the fable2 assemble gate (`speaker set != cast rows`) |

All three DETECT. None repairs, and none removes. There is no
"write lines for the silent character" retry and no "drop the character from the
ledger" path anywhere in the tree.

## THE TWO LANES BEHAVE DIFFERENTLY (grounded 2026-08-02)

* **`scifi_fable2`** -- has the gate. Detects correctly, then dies:
  `UNKNOWN_SPEAKER` + `CAST_MEMBER_SILENT`, ladder exhausted, leg over at
  2.7 minutes (`wan_ti2v`).
* **`scifi_news_pro`** -- ledger meta says the pack "declares NO
  line_composer_system seam -- the lane owns its own content loop". No
  equivalent gate. The empty rows travel to the freeze gate and surface as
  `line proof coverage mismatch: extra=['shot_001_b2', ...]`, which names a row
  id when the real answer is "c03 never got dialogue" (`ltx_video`).

And the silent row names the likely cause: cast was
`c01=ANNOUNCER, c02=Elias, c03=`**`The Relay`**. The lane cast a MACHINE and
then reasonably wrote it no lines.

## WHAT THE RULING REQUIRES (the design question for the panel)

Two acceptable outcomes, never a third:
1. **Voice it** -- an LLM pass writes the missing character's lines, or
2. **Remove it** -- the character leaves the ledger ENTIRELY.

"Entirely" is the load-bearing word, and it is where this gets dangerous. A cast
member is not just a `cast[]` row: it has `char_id` references in line rows,
beat/shot rows, a minted portrait still, voice assignment, and possibly
authorship proofs already built from it. **A half-removed character is a worse
ledger than a silent one** -- that is the LEDGER-COMPLETENESS rule in CLAUDE.md,
which says a removed pass must have every field it wrote re-owned, exactly once.

## WHAT THE PANEL MUST ANSWER

1. **Which outcome, when?** Is "voice it" always preferred with "remove" as the
   fallback after N failed attempts, or does the CHOICE depend on the character
   (a machine like "The Relay" may deserve removal; a named human deserves
   lines)? Is there a signal in the ledger to tell those apart?
2. **The complete removal checklist.** Enumerate EVERY field and artifact a
   `char_id` touches -- cast rows, line rows, beat/shot rows, stills manifest,
   portrait assets, voice assignment, `speaker_role`, authorship proofs,
   caption/credit text -- and who re-owns each after removal. A field left
   pointing at a removed char_id is a broken render.
3. **Where does the gate belong?** It must fire BEFORE an authorship proof is
   minted from rows a later pass can empty, and it must be lane-agnostic, since
   `scifi_news_pro` has no composer seam to hang it on. Name the exact call site
   that both lanes pass through.
4. **What does it say?** The current failure names `shot_001_b2`. It must name
   the CHARACTER and the lane, and state which of the two outcomes it attempted.
5. **Does the repair actually fit?** The fable2 ladder already burns 4-5
   attempts. If "voice it" is another LLM call, does it fit the P0/P5 token and
   context budgets, and does it run inside the existing ladder or after it?
6. **Casting, upstream.** If `scifi_news_pro` is casting non-speaking entities,
   is the real fix at the CASTING pass -- do not cast a relay -- rather than
   repairing downstream? Which is cheaper and which is more honest?

## CONSTRAINTS

Radio drama: a cast member with no voice is a bug, never a stylistic choice.
Fail loud, never silently degrade. The ledger must stay COMPLETE for every
downstream consumer (TTS, per-beat slicing, shot direction, captions, credits,
`obs_publish`) -- a removed character must leave no dangling reference. The only
workflow JSON is `workflows/otr_canonical.json`. 100% local. **Do not launch
renders or boot a server** -- a GPU measurement leg is queued behind this.
