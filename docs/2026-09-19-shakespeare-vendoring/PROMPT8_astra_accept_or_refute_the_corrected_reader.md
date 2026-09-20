Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

PRIORITY ONE. YOU REFUTED THE READER AND YOU WERE RIGHT; NOW IT IS FIXED
ON MAIN AND ONLY YOU CAN SAY WHETHER IT IS FIXED. Commit 0b35b419
implements what your audit specified: one TextPage built with
TEXTFLAGS_WORDS, both views read from it, each word's baseline the median
of ITS OWN characters by block/line/word identity, never by geometry. Your
adaptive span is in. Your fused-word sweep now crosses span boundaries and
skips rotated lines. Your furniture proposal is applied as measured.

Verified here after the change, against your numbers: `que` owns 100.3;
`gusano,` is in Prospero's row; Tempest 58-61 are 24/38/38/39; Clark 133
and 182 are 36 and 39; `ramera.M` is reported on Otelo 176; word
conservation 0 of 1,450; furniture inside both Portuguese spans 0 and 0.

None of that is the acceptance test. THE ACCEPTANCE TEST IS YOURS: does
the script's actual output now reproduce your hand reconstruction?

YOUR JOB IS TO REFUTE THE FIX. Read-only on the repo; scratch in %TEMP%;
no git writes; no GPU. Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

1. RE-RUN YOUR SCENE COMPARISON THROUGH THE ACTUAL SCRIPT. Last time:
   Lear 1.1 had 36 of 83 speech bodies differing from your hand result,
   Macbeth 1.3 had 15 of 49. Run `pdf_text(url, reading_order="coordinates")`
   through the current furniture cleanup, `extract` and `speeches_from_span`
   and diff against your saved hand reconstruction. THE NUMBER THAT DECIDES
   IT: differing speech bodies must be ZERO in both scenes. If it is not
   zero, name the first differing speech and the word that moved.

2. RE-RUN YOUR FULL-VOLUME OWNERSHIP SWEEP. Last time 14,689 words across
   1,274 pages were off by more than a point against your qualified
   word-to-character mapping. That count must now be zero. If any page
   still differs, the fix is incomplete -- name the page and the word.

3. THE FALSE JOINS YOU NAMED -- Lear `expara`, `inuma`, `Mapratico-o` --
   were downstream of reordered rows. Confirm each is gone, or say which
   remains and why.

4. THE FUSED-WORD SWEEP. You measured 32 cross-baseline tokens on 29
   pages. Run the shipped `fused_words` over the same five volumes and
   report: how many of the 32 it now names, how many rotated-table pages
   it now wrongly names (it flagged nine before), and whether the Lear
   back-matter `54-` is the only one it reports in that volume.

5. THE FURNITURE PROPOSAL AS APPLIED. It was measured in memory; it is now
   in the file. Re-run your 80-page generated check and your 20-test
   validation against the real file, not the in-memory proposal, and say
   whether any case moved. Then the one thing your report noted and did
   not test: the scene parser "can alter punctuation after furniture
   disappears" and Macbeth "retains a parenthetical qualifier `(baixo para
   Macheth)` the polluted input lost". Show both.

6. THE VERDICT THAT MATTERS TO THE CORPUS: with ownership corrected, may
   the two shipped Portuguese scenes now be RE-VENDORED under
   `reading_order="coordinates"`? That is the one decision this fix was
   for. Yes or no, with the diff that decides it -- and if yes, list every
   speech that changes speaker, because each one is a shipped
   misattribution being repaired and the commit message must name them.

DO NOT propose changes to extract() or find_label().

FINISH WITH:
    ACCEPTED / STILL REFUTED:  one word, then the deciding number
    RE-VENDOR:                 yes / no, with the speaker changes listed
    REFUTED:                   every claim above you could not confirm
