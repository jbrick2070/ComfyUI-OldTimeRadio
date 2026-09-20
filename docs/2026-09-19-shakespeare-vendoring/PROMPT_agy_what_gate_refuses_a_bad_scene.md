You are reviewing a real repository at
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
on branch main, HEAD 21a3cd7c.

ONE QUESTION, AND IT IS NOT ABOUT THE EXTRACTOR: when a vendored Shakespeare
scene comes out WRONG, what in this repository refuses it? Trace the whole
path from the file being written to an episode being rendered from it, and
name every gate. Then say, for each gate, exactly what it would and would not
have caught.

YOUR ONLY JOB IS TO REFUTE. Ground every claim in a file and a line number.
If you cannot ground a claim, say "refuted -- could not confirm in <file>"
rather than agreeing with it. Read-only: do not edit any file, do not run the
render pipeline, do not run any git command that writes, do not touch the GPU.

THE CONCRETE BAD SCENE, MEASURED TODAY

A dry run of scripts/otr_vendor_scan.py on the Portuguese Tempest act 3
scene 1 returned 14 speeches from 2 distinct speakers. The English Folger
roster for that scene is exactly FERDINAND, MIRANDA, PROSPERO -- three
characters -- and Ferdinand opens the scene. He was absent entirely, because
the edition prints him as FERNANDO and the resolver's stem rule refuses a
name that short. His speeches were silently handed to whoever spoke last.

Nothing errored. Nothing warned. That is the scene to trace.

WHAT TO READ

  scripts/otr_vendor_scan.py -- main(), specifically the manifest row it
      writes at the end
  nodes/_otr_passage_selector.py -- the runtime consumer of the corpus
  nodes/_otr_verbatim_corpus.py
  scripts/otr_shakespeare_corpus_gate.py
  tests/test_verbatim_corpus.py, tests/test_shakespeare_corpus_gate.py,
      tests/test_shakespeare_sources.py, tests/test_verbatim_translation.py,
      tests/test_shakespeare_verbatim_executor.py
  config/source_banks/shakespeare/translations/manifest.json

SIX CLAIMS. Confirm or refute each against the code, and for each say what a
LISTENER of the finished episode would hear if it is true.

1. otr_vendor_scan.py writes its manifest row with
   "alignment_confidence": 1.0 and "verdict": "READY" as LITERAL CONSTANTS,
   not as the result of any measurement. Confirm or refute by quoting the
   lines. If true: say what a downstream reader is entitled to believe when it
   sees alignment_confidence 1.0, and whether any consumer actually reads that
   field. Grep for the field name; do not assume.

2. The same row writes speaker_map from resolve(), so the map can only ever
   contain labels that ALREADY resolved. A character who was refused cannot
   appear in it, and distinct_speakers therefore counts survivors rather than
   cast. Confirm or refute. If true, say whether any check anywhere compares
   distinct_speakers against the English roster size.

3. scripts/otr_shakespeare_corpus_gate.py verdicts LEADS -- candidate sources
   before vendoring. Does it ever run against a scene that is ALREADY
   vendored, and does anything re-verdict a stored file after it is written?
   If nothing does, say so plainly and name where such a re-check would have
   to live.

4. nodes/_otr_passage_selector.py reads the corpus at runtime. Trace what it
   does with a scene whose stored text is short and whose cast is smaller than
   the roster. Does it notice? Does it refuse? Does it pick a different
   passage? Or does it render a two-hander where the source has three people?
   Follow the actual code path, and name the function that would have to
   object.

5. THE LEDGER QUESTION. This repo has a hard rule that downstream consumers --
   TTS, per-beat audio slicing, shot direction, captions, credits,
   obs_publish -- read FIELDS, not intentions, and that every field must have
   exactly one owner. A missing character means a cast entry that never
   exists. Trace whether a scene missing a character produces a COMPLETE
   ledger with a smaller cast (which renders fine and is quietly wrong) or an
   INCOMPLETE ledger that fails somewhere. Name the file and function where
   the two paths diverge. This is the difference between a wrong episode and a
   crashed one, and the repo's bar treats those very differently.

6. Three of the 41 vendored rows were produced by an extractor rather than
   from HTML markup -- find them by transcription_method in the manifest.
   Does ANY test in tests/ assert anything about those three specifically?
   Name it, or say there is none.

FINALLY, ONE RECOMMENDATION AND ONE ONLY

Name the single cheapest check that would have caught the Ferdinand case at
vendoring time -- before the file was written, not after. Say which file it
belongs in, what it asserts, and what legitimate case it would false-positive
on. Then give the strongest argument AGAINST adding it. Do not propose a
second check; if you have two ideas, pick the one you would defend.

FINISH WITH TWO LISTS, headed exactly:

    MUST-FIX:
      places where a wrong scene reaches a listener with nothing objecting

    REFUTED:
      every claim in this brief you could not confirm against the real files,
      naming the file you looked in
