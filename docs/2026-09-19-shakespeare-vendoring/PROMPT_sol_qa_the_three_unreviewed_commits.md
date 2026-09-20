Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

THREE CODE COMMITS FROM TODAY HAVE HAD NO REVIEW AT ALL. The QA brief
written for them was handed to a lane that received a different brief
instead, so it never ran. Push-then-QA is the rule (CLAUDE.md,
2026-09-19); this is the QA. The hyphen fix (2bb45e9e) was reviewed by
Sonnet and its one finding was fixed in 2e9ff62e; the reader (ae792152)
was refuted and replaced by 0b35b419, which another lane has. These three
are yours:

  efee731a  `_SPLIT_HEADING` widened to tolerate trailing punctuation on
            the numeral line: `[ \t]*$` became `[ \t.,;:]*$`.
  76f88d8c  `--pages START-END` and `slice_pages`, slicing AFTER
            `strip_running_titles` so the furniture vote sees the whole
            volume.
  2e9ff62e  pt/king_lear_1_1.txt regenerated after `AfasKent` was found;
            80 speeches to 81; manifest row's raw_sha256 and
            speaker_labels changed.

YOUR ONLY JOB IS TO REFUTE. Find the reason each should not have shipped.
If you cannot ground an objection in a file and a line, do not raise it.
Read-only; no git writes; no render pipeline; no GPU; scratch in %TEMP%.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.
PDFs are cached under tmp\scan_cache. Read each commit with `git show`.

efee731a -- ATTACK:
  1. The widened class admits `.,;:` after the numeral. Construct a real
     two-line sequence from any of the five cached volumes where the
     first line is a bare heading word and the second is a line of
     DIALOGUE that begins with a roman numeral or a small integer or a
     spelled ordinal and then a stop -- `I.` as a pronoun-initial, `II.`
     as a chapter reference, `PRIMERO ,` opening a sentence. Does the
     rejoin weld it? The commit claims the numeral alternation keeps it
     narrow; test the claim on the real text, all five volumes.
  2. The commit says 0 rejoins on both Portuguese volumes before and
     after. Count them yourself on the RAW pages and on the
     `strip_running_titles` output separately -- a reviewer earlier
     found one raw match (`acto\niv`, Macpherson page 39) that the
     stripped count hid. Say which frame the commit's number is in.

76f88d8c -- ATTACK:
  3. The slice runs after `strip_running_titles` -- correct, the vote
     needs the whole volume. But `recurring_headings(pages)` and the
     `--end-label` refusal in `main()` are then called on the SLICED
     list. Read `main()` from the slice to the refusal. On a five-page
     window `_ECHO_FLOOR` is 3: can a scene label that is a running head
     on the whole volume FAIL to reach the floor inside the window, so
     the guard that exists to catch a fragment goes silent exactly when
     the window is small? Construct the case from a real cell.
  4. `slice_pages` accepts `"5"` as a one-page window. Is a one-page
     window ever sane, given `extract` needs a heading AND a following
     heading or the end of the list? Say what a one-page window returns
     for `es tempest 3.1` at page 58 alone.
  5. The commit message says "a bad window fails loudly." Give it the
     wrong window that still CONTAINS the label -- `--pages 244-300` for
     Lear 1.1 instead of 244-255 -- and say what is stored. If the
     answer is "a longer scene at alignment_confidence 1.0", the claim
     is only half true and the brief should say so.

2e9ff62e -- ATTACK:
  6. `git diff 2bb45e9e 2e9ff62e -- config/source_banks/shakespeare/translations/pt/king_lear_1_1.txt`.
     The commit claims one line became two and nothing else moved. Count
     the changed lines. If ANY other line differs, name it and say what
     moved it.
  7. The manifest row: raw_sha256, revision_id, speaker_labels 80->81,
     distinct_speakers 9 unchanged. Confirm all four. Then the field
     nobody checks: `speaker_map` -- did KENT's entry change, and should
     it have?

FINISH WITH, PER COMMIT:
    <hash>: HOLDS  |  DEFECT: <file:line, one sentence>
and then:
    REFUTED:  every claim above you could not confirm
