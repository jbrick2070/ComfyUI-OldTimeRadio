Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, near commit 8d84ec96.

YOUR LAST PASS WON ON EVERY CONTESTED POINT AND CORRECTED THE DRIVER TWICE.
Verified here before writing this:

  * The real Lear 1.1 window (positional 13659-14108) is 449 lines, 90 cue
    tokens, 30 distinct forms, with Gloucester appearing as GLÓS and Glós
    only and Edmund as EDM and Edm only. The driver's "five Gloucester
    spellings, seven Edmund spellings, 27 forms" was measured on the span
    `find_label` STEALS -- Act 2 onward -- and was reported to the operator
    as scene-level noise. Your correction stands and the claim has been
    withdrawn.
  * `ALB . Y CORN. Deteneos, señor.` is in that window, exactly as you said.
  * "Clark clean, Macpherson unusable" is refuted: Clark's own Twelfth Night
    1.5 is 30 forms against Tempest 3.1's 7, in one book. Viability is per
    SCENE. That was the driver's claim and it is withdrawn.
  * The edit-distance narrowing is abandoned on your numbers. Do not spend
    another minute on it.
  * `Min` is conceded as an inference. 0/8 stands against today's extractor.

YOUR JOB IS TO REFUTE AND TO MEASURE. Ground every claim in a number you
produced. Where you cannot ground one, say "refuted -- could not confirm".
Read-only: do not edit any tracked repository file, do not run the render
pipeline, do not run any git command that writes, do not touch the GPU.
**Write scratch outside the repository** -- the tree holds an uncommitted
change.

Python:  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe with
PYTHONUTF8=1. PDFs are cached under tmp/scan_cache.

WHY YOUR 0/8 MAY BE MEASURED ON SAND, AND WHY THAT IS THE DRIVER'S CLAIM TO
BREAK

Every cue inventory in your audit -- yours, the driver's, everyone's -- was
taken from `page.get_text()`, and that order is NOT the printed order.
Measured twice on two volumes:

  Clark page index 59, rebuilt from word coordinates, reads
      `MIR. Se me antoja / Que estás rendido.`
      `FER. No, mi noble dueña:`
  but `page.get_text()` emits both cues at the BOTTOM of the page, detached
  from their dialogue, so Prospero absorbs both speeches.

  Domingos Ramos Macbeth page index 36 -- a scene ALREADY VENDORED -- reads
  `MACBETH / ...tão horri- / vel! / BANQUO / ...` visually, while flattening
  throws `vel!` to the TOP of the page onto the `TODAS TRES` speech.

**THE DRIVER'S CLAIM, WHICH YOU SHOULD TRY TO DESTROY:** your phantom
`MIR .` at L67 -- empty body, immediately followed by `FER .` at L68, inside
Ferdinand's "Admired Miranda" speech -- is not an edition defect at all. It
is that same displacement: two marginal cues stripped from their lines and
emitted together. If that is right, then the phantoms, the empty bodies and
the "three speeches with no cue" largely dissolve once rows are rebuilt from
coordinates, and your 0/8 was measured against an artifact. If it is wrong,
say so with the page image evidence.

ALSO: YOU MEASURED AGAINST THE WRONG BAR, AND YOU SAID SO YOURSELF

You noted that the operator's 2026-09-19 ruling already contradicts gate 2
as the brief stated it, and measured against the brief anyway. That was the
right call then. The ruling is now recorded in docs/OTR_STANDING_RULINGS.md
-- read it -- and it is the bar from here:

  * A mangled native text BEATS a clean AI translation. Mangled glyphs in
    dialogue SHIP.
  * An unresolvable cue is NOT an abort. It becomes its own speaker under
    the name as printed, left unbound; an unbound label costs a VOICE and
    never the dialogue.
  * What is still refused is a speech landing in ANOTHER character's mouth.

Under that bar, `Min .` does not need to be proven. It becomes a speaker
named `Min`, unbound, and Miranda's words stay in their own turn instead of
being merged into Ferdinand's.

WHAT TO MEASURE

1. Rebuild printed rows from word coordinates -- `page.get_text("words")` or
   `("dict")`, cluster by baseline, order each row left to right. A crude
   y0-within-3-points rule is known to split rows wherever a comma sits on
   its own baseline, so tune it and SAY what you used. Validate it the cheap
   way: the multiset of words must be identical before and after, so only
   ORDER changes. Report any page where a word is lost or duplicated.

2. RE-MEASURE ALL EIGHT CELLS on the reconstructed text. For each: span,
   cue tokens, distinct forms, how many forms are wrap-period noise, how
   many are OCR corruptions of a real cue, how many speeches have no cue,
   and how many empty bodies. Put it beside your previous numbers so the
   delta is visible. **The single number that matters: how many of your
   phantom and no-cue findings survive reconstruction?**

3. THE COUNT THAT DECIDES THE BUILD. Under the real ruling above -- unbound
   speaker instead of abort, mangled glyphs shipping -- how many of the
   eight cells ship? For each that does NOT, name the defect and say whether
   it is MISATTRIBUTION (still refused) or IMPERFECTION (now allowed). That
   distinction is the whole verdict; a cell blocked only by imperfection is
   a cell that ships.

4. DOES COORDINATE ORDER FIX `D . PED.`? Your best structural find. If the
   honorific and the abbreviation sit on ONE printed row, reconstruction
   keeps them together but does not by itself make `D . PED.` parse as one
   cue. Measure whether they share a row. Then say what the cue parser needs
   -- and whether the same shape appears in Macpherson or only in Clark.

5. WHAT RECONSTRUCTION DOES NOT FIX. You have measured more of these volumes
   than anyone. Name the defects that survive correct reading order and the
   new ruling both. Those are the real remaining work, and that list is the
   most useful thing you can leave behind.

DO NOT propose changes to extract() or find_label() in
scripts/otr_vendor_shakespeare.py -- scoped out by the operator and shared
with 38 HTML-vendored scenes already shipping. A fix needing it is a finding
to report, not a recommendation.

FINISH WITH THREE LISTS, headed exactly:

    SHIPS UNDER THE REAL RULING:
      one line per cell, with any unbound labels it would carry

    STILL MISATTRIBUTION:
      cells where a speech still lands in the wrong mouth, with the evidence

    REFUTED:
      every claim in this brief you could not confirm, the driver's
      phantom-dissolution claim included if it does not survive
