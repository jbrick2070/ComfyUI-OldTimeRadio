Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 76f88d8c.

FOUR CODE COMMITS WERE PUSHED TODAY UNDER A NEW RULE: push the green
chunk, QA it afterwards. This is that QA. The law says every code change
gets a refuting review; three of these four have had none.

YOUR ONLY JOB IS TO REFUTE. Find the reason each should not have shipped.
If you cannot ground an objection in a file and a line, do not raise it.
Read-only; no git writes; no render pipeline; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

THE FOUR, read each with `git show <hash>`:

  efee731a  _SPLIT_HEADING widened to tolerate a trailing stop on the
            numeral line. Claim: 0 -> 12 rejoins Clark, 0 -> 19 Macpherson,
            0 -> 0 on both Portuguese volumes; pt/macbeth 1.3 re-extracts
            byte for byte. ATTACK: does `[ \t.,;:]*$` admit anything that is
            not a heading? Construct the line of dialogue it would weld.

  ae792152  An opt-in coordinate reader, `reading_order="coordinates"`,
            default `flat`. Claim: rows on Tempest 3.1 = 24/38/38/39,
            zero word-conservation failures on 1,450 pages. ATTACK: is the
            default really untouched -- can any existing caller reach the
            new path? Read `pdf_text`'s signature and every call site.
            Then read `_word_baseline_from`: a word whose box holds glyphs
            from a NEIGHBOURING row -- does the median pick the wrong row?

  2e9ff62e  pt/king_lear 1.1 regenerated after `AfasKent` was found -- the
            hyphen weld hidden by a Title Case label. 80 -> 81 speeches.
            ATTACK: diff the stored file against the previous commit. Is
            the ONLY change Kent's restored line? Any other line moved?

  76f88d8c  `--pages START-END`, slicing AFTER `strip_running_titles`.
            ATTACK: the furniture vote uses `len(pages)//5` as its floor --
            after slicing, `recurring_headings` and the refusal in main()
            are called on the SLICED list. Does that change what the
            `--end-label` guard sees, and can it now fire wrongly or fail
            to fire? Trace it.

ALSO, THE THING NONE OF THE MESSAGES SAY: run the four suites --
tests/test_vendor_scan_furniture.py tests/test_verbatim_corpus.py
tests/test_shakespeare_corpus_gate.py tests/test_shakespeare_sources.py --
and report the real count. One commit message claimed 15/186 when the
truth was 16/187. Say what it is now.

FINISH WITH, PER COMMIT:
    <hash>: HOLDS  |  DEFECT: <file:line, one sentence>
and then:
    SUITE:    the real pass count
    REFUTED:  every claim above you could not confirm
