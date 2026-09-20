You are pressure-testing a DESIGN that has been ruled on but NOT yet written,
in a real repository at
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
on branch main, near commit 0c437238.

YOUR ONLY JOB IS TO REFUTE. Ground every claim in a file and a line number.
If you cannot ground a claim, say "refuted -- could not confirm in <file>"
rather than agreeing with it. Nothing in this brief is established: every
statement is a claim for you to check. Read-only: do not edit any file, do not
run the render pipeline, do not run any git command that writes, do not touch
the GPU.

READ
  scripts/otr_vendor_scan.py -- resolve(), speeches_from_span(), clean_label(),
      _CAPS_RUN, _LABEL_QUALIFIER, FUNCTION_NAMES, ORDINALS, PLACE_NAMES
  tests/test_vendor_scan_furniture.py
  config/source_banks/shakespeare/translations/manifest.json

THE RULING BEING IMPLEMENTED

Spanish scanned editions abbreviate their speaker labels in inconsistent case
(`FER .`, `Fer.`, `MIR`, `Prós.`, `PRÓS.`). The operator ruled: implement an
EXACT, EDITION-SCOPED speaker-label registry. Explicitly forbidden: relaxing
the generic prefix resolver, lowering its character threshold, a global
Spanish alias table, prefix inference, or first-roster-match.

MEASURED EVIDENCE FOR THE TABLE (verify what you can; the PDF is cached under
tmp/scan_cache, keyed by a sha256 of the source URL)

For `es/tempest 3.1` (Jaime Clark, "La tempestad - La noche de Reyes"), with
the volume correctly sliced to one play, the line-initial cues in the scene
span are exactly:

    FER x6, Fer x4, MIR x5, Mir x4, Prós x3, Mır x1, Min x1

Three characters, seven printed forms. Two are scanner corruptions: `Mır`
carries a DOTLESS I (U+0131), and `Min` is a misread of `Mir`.

THE DESIGN UNDER ATTACK

  * A registry keyed by the SOURCE URL, which identifies the actual scanned
    volume rather than the language or the play.
  * Each entry maps an exact printed form -> the emitted spoken name -> the
    exact English scene-sidecar roster target.
  * A cue is recognised ONLY at the start of a logical line, and only when the
    printed form is a member of the table. Supported shapes: trailing period,
    spaced period (`FER .`), a dash after the period, and an immediately
    following parenthetical qualifier (`Prós. (Aparte.)`).
  * `FERNANDO -> FERDINAND` is a dedicated proper-name alias, never an entry
    in PLACE_NAMES.

ANSWER THESE, EACH GROUNDED IN A FILE AND A LINE

1. THE KEY. Is the source URL a safe registry key? Read
   config/source_banks/shakespeare/translations/leads.json and the manifest.
   Do two DIFFERENT scenes that need DIFFERENT cue tables ever share one URL?
   Does one logical edition ever appear under two URLs? Name the rows. If the
   key is wrong, say what the right key is and what in the repo already
   carries it.

2. THE OCR ENTRIES ARE THE DANGEROUS ONES. `Min -> MIRANDA` is a guess dressed
   as a measurement: `Min` could be a misread of something else, and in
   another scene of the same volume it could be a different character
   entirely. What evidence would make that entry safe, and does this repo have
   it? If the honest answer is that a one-occurrence OCR form should ABORT the
   run rather than be mapped, argue that. Consider also that `Mır` with a
   dotless I is invisible when read on screen.

3. THE LINE-START ANCHOR. The claim is that requiring a cue at the start of a
   logical line, plus exact table membership, prevents ordinary prose becoming
   a speaker. Attack it. Find a line in one of these scanned volumes that
   BEGINS with a table member and is NOT a cue. Consider: a continuation line
   of a long speech, a line beginning with an abbreviation, and the fact that
   pymupdf's line breaking is the extractor's and not the book's -- so a "line
   start" is not a typographic fact. That last point is the one to press
   hardest.

4. THE QUALIFIER. `Prós. (Aparte.)` -- the qualifier must be removed from the
   spoken text, but parentheses INSIDE real dialogue must survive. Read
   `_LABEL_QUALIFIER`. Does it help here, or is it anchored to the wrong end
   of the token? Write the rule you would use and name the dialogue line in
   these volumes it would wrongly strip.

5. THE ROSTER TARGET. Every registry target must validate against the EXACT
   scene roster sidecar. Read roster_for() and
   nodes/_otr_roster_gender.py::load_roster_characters. Is a scene sidecar's
   roster the set of characters who SPEAK in that scene, or who are listed as
   present? If a cue maps to a character who is listed but silent, should the
   gate pass or fail? Answer with the consequence for a real scene.

6. WHAT THE TABLE CANNOT EXPRESS. Name one thing these editions do with
   speaker labels that an exact form->name table structurally cannot handle.
   Look for: a label that changes meaning by context, a collective, a label
   the edition reuses for two characters, and a character who enters mid-scene
   under a different abbreviation. If you find none, say so -- that is a
   meaningful result.

7. Having attacked it, state whether the ruled design is sound as specified.
   If you would change one thing, name exactly one, and give the strongest
   argument against your own change.

FINISH WITH TWO LISTS, headed exactly:

    MUST-FIX:
      items that would make this lane ship a speech in the wrong mouth, or
      abort a scene that should have shipped

    REFUTED:
      every claim in this brief you could not confirm against the real files
