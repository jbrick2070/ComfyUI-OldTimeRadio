Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 76f88d8c.

ONE QUESTION, DOWNSTREAM OF EVERYTHING ELSE: when a vendored scene carries a
speaker label that is bound to NO English roster name, what does the
RUNTIME do with it -- exactly, file and line, from the manifest row to the
voice that reads the line?

This decides whether a standing ruling can actually be implemented. The
operator ruled on 2026-09-19 (docs/OTR_STANDING_RULINGS.md) that an
unresolvable cue in a scanned scene becomes its own speaker under the name
as printed, LEFT UNBOUND -- "an unbound label costs a VOICE and never the
dialogue." The vendor script is about to start emitting such labels for
Spanish scenes: `Min`, `Mır`, `Oliy`, `Bur`, `LENT`, `Puch`, `D . PED.`.
The claim being tested is that the runtime already handles one. If it does
not, every such scene crashes or silently drops the line, and the ruling is
a sentence rather than a behaviour.

YOUR JOB IS TO REFUTE THAT CLAIM. Ground every step in a file and a line.
Read-only; no git writes; no render pipeline; no GPU.

TRACE THE PATH, IN ORDER
  1. nodes/_otr_verbatim_corpus.py -- `speaker_bindings`, `select_scene`,
     `vendored_text`. What does a row's `speaker_map` need to contain for a
     label that binds to nobody? Is a missing key tolerated, refused, or
     does it raise? Quote the branch.
  2. nodes/_otr_passage_selector.py -- `parse_speeches` and the window
     selection around lines 470-540. A label like `D . PED.` or `Mır`: does
     the parser even READ it as a speaker? Read `_is_upper_label` and say
     which of `Min`, `Mır`, `Oliy`, `D . PED.`, `1.a FEITICEIRA` pass it.
     One that fails is not an unbound speaker -- it is dialogue glued to
     the previous line, which is the misattribution the ruling still
     refuses.
  3. nodes/OTR_LedgerScriptWriter.py around lines 4271-4279 and 4587-4590,
     and nodes/_otr_outline.py around line 1696 -- `lock_cast` and
     `_unseated`. The cast is sized to the passage's speakers. Does an
     unbound speaker get a SEAT, a voice from the roll, and a gender? Or
     does `_unseated` raise on it?
  4. tests/test_verbatim_corpus.py `_KNOWN_UNBOUND` -- the existing
     allow-list for exactly this. What does membership actually change at
     runtime, and what does a label NOT in it do to the suite?

THEN ANSWER
  * Can the runtime perform a scene with an unbound label TODAY, end to
    end, with no code change? Yes or no, with the line that decides it.
  * If no: name the smallest change, its file, and what it must NOT touch
    (the operator scoped scripts/otr_vendor_shakespeare.py out; say whether
    the change reaches it).
  * The `_is_upper_label` question is the sharp one: if the runtime parser
    refuses title-case and punctuated labels, the vendor must UPPER-CASE on
    write -- `clean_label` in scripts/otr_vendor_scan.py already does this
    for resolved labels. Say whether the same must happen for unbound ones
    and what `D . PED.` becomes.

FINISH WITH:
    RUNS TODAY:   yes / no, with the deciding line
    SMALLEST FIX: file, function, one sentence
    REFUTED:      every claim above you could not confirm
