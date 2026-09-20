Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, near commit 5a8afffc.

YOUR JOB IS TO REFUTE AND TO MEASURE, AND THIS TIME THE THING TO REFUTE IS
THE DRIVER'S CORRECTION OF YOUR OWN AUDIT. Ground every claim in a file, a
line, or a number you produced yourself. Where you cannot ground a claim,
say "refuted -- could not confirm" rather than agreeing with it. Read-only:
do not edit any repository file, do not run the render pipeline, do not run
any git command that writes, do not touch the GPU. You MAY write and run a
throwaway read-only probe; delete it when done.

Python:  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
with PYTHONUTF8=1. The PDFs are cached under tmp/scan_cache (filename is the
first 24 hex of a sha256 of the source URL), so nothing downloads. Use
scripts/otr_vendor_scan.py -- pdf_text(), strip_running_titles(),
_RUNNING_HEADER, _SPLIT_HEADING -- and find_label()/extract() from
scripts/otr_vendor_shakespeare.py.

WHAT YOUR AUDIT ESTABLISHED, AND WHERE IT WAS OVERRULED

Your mechanism finding was reproduced independently and holds. Measured here
on es/king_lear 1.1 (Macpherson): 27 distinct line-initial period tokens for
a 9-name scene, with five spellings of Gloucester (Glos, GLOS, Glós, Gós,
OLOS), seven of Edmund (EDM, Edm, Edu, Ezm, EOM, ED, Edy), three of Regan
(REG, Rig, Ræg), and genuine wrap-period noise -- `sé.`, `oigáis.`, `pien.`,
`Mejor.`, `Acuérdate.`, `Oid.` -- plus a mangled running head `VAKESPEARE.`.
Your gate-3 point holds too: the sidecar carries nine names and no ALBANY.
Your specific token `ALB . Y CORN` was NOT reproduced in the measured span
and is currently unconfirmed.

YOUR HEADLINE, 0/8, WAS OVERRULED TO 1/8. That is what you should attack
first. The reasoning was:

  es/tempest 3.1 (Clark), sliced to LA TEMPESTAD, act `ACTO III.`, scene
  `ESCENA PRIMERA`, end `ESCENA II` = 4822 chars, 152 lines, 24 line-initial
  period tokens, SEVEN distinct forms, and ZERO wrap-period noise -- every
  one of the 24 is a genuine cue. The two you called blockers were judged
  provable from the source and therefore allowed by the ruling's own clause
  ("measured OCR ... only where the source evidence proves that meaning"):

    line 112   `Mır .`   = M + U+0131 DOTLESS I + r, a one-codepoint
                           substitution of the cue `Mir` used elsewhere in
                           the same scene
    line 122   `Min . Mi indignidad: hacer oferta no oso`
                         = sits between Ferdinand at line 120 and Ferdinand
                           at line 133, and its text is Miranda's

1. BREAK THAT. Is `Min` provably Miranda, or is that an inference wearing a
   measurement's clothes? Consider specifically: does the FER/MIR alternation
   actually hold without exception across all 24 cues, or are there
   consecutive same-speaker cues that weaken the positional argument? Does
   any OTHER character appear in this scene who could take that line -- check
   the English sidecar for king of the scene's cast, and check whether
   Prospero's asides break the alternation. If you conclude `Min` is an
   inference, say so; the correct consequence is that the cell does not ship,
   and 0/8 stands.

2. THE VOLUME NOBODY HAS MEASURED. Two of the eight cells -- es/much_ado 2.3
   and es/much_ado 3.1 -- live in "Otelo - Mucho ruido para nada (Jaime
   Clark)", 294 pages, and its cue inventory has never been measured by
   anyone. Clark's OTHER volume is clean (7 forms, no noise) while
   Macpherson's is not (27 forms, heavy noise). Measure this one and say
   which it resembles. Report the same table you produced before: span
   length, distinct line-initial period forms with counts, how many are
   obvious corruptions, how many are wrap-period noise, and the English
   sidecar roster size.

   THE CLAIM THIS TESTS, which the driver made and which may be wrong: that
   viability is per-VOLUME rather than per-language -- Clark clean, Macpherson
   unusable. Two Clark volumes agreeing would support it; one clean and one
   dirty would destroy it. Say which happened.

3. MEASURE THE PROPOSED NARROWING, BECAUSE IT DECIDES THE DESIGN. Gate 2 as
   ruled aborts on "any unknown cue-shaped source token". With cue-shaped
   read as "line-initial letters followed by a period", it fires on `sé.` and
   `Mejor.` and nothing ships. The proposed narrowing is:

       a token is CUE-SHAPED only if it is within edit distance 1 of a
       registry member, or the generic resolver already accepts it

   Measure it on all three Spanish volumes. For each, produce two counts:

     FALSE ABORTS -- ordinary wrap-period words that land within edit
       distance 1 of a registry member anyway, so the gate still fires on
       dialogue. Name them. Short registry keys are the danger: with `COR`,
       `EDM`, `REG`, `FER`, `MIR` in the table, how many Spanish words of
       three or four letters sit one edit away?
     MISSED CATCHES -- real OCR corruptions of a cue that are MORE than one
       edit away and so slip through unflagged. `Ræg` for `REG` and `OLOS`
       for `GLOS` are the two to test first; compute the distances rather
       than eyeballing them.

   Then state plainly whether edit distance 1 is the right radius, whether
   another radius is better, or whether edit distance is the wrong instrument
   here. A narrowing that produces either false aborts or missed catches at
   scale is not worth building, and saying so is the useful answer.

4. Given (1) to (3), give the final count: of the eight cells, how many ship
   under the ruling WITH the narrowing, and how many without it. If the two
   numbers are the same, the narrowing is not worth building and you should
   say that in one sentence.

DO NOT propose changes to extract() or find_label(). The operator scoped
scripts/otr_vendor_shakespeare.py out of this ticket and it is shared with
the 38 HTML-vendored scenes already shipping; a fix needing it is a finding
to report, not a recommendation to make.

FINISH WITH THREE LISTS, headed exactly:

    SHIPS:
      cells that clear all seven gates, with the registry entries each needs

    CANNOT SHIP:
      cells that do not, with the blocking token or heading named

    REFUTED:
      every claim in this brief you could not confirm against the real files,
      including any of the driver's numbers above that you could not
      reproduce
