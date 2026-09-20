You are auditing SHIPPED DATA in a real repository at
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
on branch main, HEAD 21a3cd7c.

This is not a review of a proposed change. Forty-one scenes are ALREADY
VENDORED and already in the product. A defect class was found today in the
extractor that produced some of them. Your job is to find out how much of the
shipped corpus already carries it.

YOUR ONLY JOB IS TO REFUTE AND TO MEASURE. Ground every claim in a file, a
line, or a number you produced yourself. If you cannot ground a claim, say
"refuted -- could not confirm" rather than agreeing with it. Read-only: do not
edit any file, do not run the render pipeline, do not run any git command that
writes, do not touch the GPU. You MAY write and run a throwaway read-only
Python script to count things; delete it when you are done.

Python to use:
    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
with $env:PYTHONUTF8=1 set.

THE DEFECT CLASS, FOUND TODAY

scripts/otr_vendor_scan.py marks a speaker by resolving an all-caps run
against that scene's English Folger roster. Read resolve(). A translated
proper noun is matched on a SHARED STEM of at least five characters, in either
direction, and only when the two lengths differ by two or less:

    for candidate in (bare or key, key):
        if len(candidate) < 5: continue
        for fkey, name in folded.items():
            if len(fkey) < 5: continue
            if candidate.startswith(fkey) or fkey.startswith(candidate):
                if abs(len(candidate) - len(fkey)) <= 2:
                    return name

Domingos Ramos translates Ferdinand as FERNANDO. FERNANDO and FERDINAND share
only FER, so the rule refuses him. When a label is refused, it is not marked,
and speeches_from_span therefore hands his text to WHOEVER SPOKE LAST. Measured
today on a dry run: Portuguese Tempest act 3 scene 1 came back with 14 speeches
from 2 mouths, in a Folger scene whose cast is exactly FERDINAND, MIRANDA,
PROSPERO and whose lead he is.

THE POINT OF THIS AUDIT: that failure is SILENT. No error, a plausible cast, a
healthy-looking speech count, and the manifest row is written with
alignment_confidence 1.0 regardless. So a scene already in the corpus could be
missing a character and nothing would have said so.

THE CORPUS AND HOW TO READ IT

    config/source_banks/shakespeare/translations/manifest.json
        41 rows. Each carries iso, play, scene, file, speaker_labels,
        distinct_speakers, speaker_map, alignment_confidence, verdict, and
        (on the scanned ones) transcription_method and extractor.
    config/source_banks/shakespeare/translations/<iso>/<play>_<scene>.txt
        the stored text, one "LABEL: speech" per line.
    config/source_banks/shakespeare/sources/<play>__act<N>_scene<M>.txt
        the ENGLISH source. Its roster is read by
        nodes/_otr_roster_gender.py::load_roster_characters -- use that
        function, do not parse the file yourself.

WHAT I WANT MEASURED, PER SCENE, FOR ALL FORTY-ONE

1. THE MISSING-CHARACTER TABLE. For every vendored scene, load the English
   roster for its matching source stem and list every roster character that
   has ZERO speeches in the stored translation. Report it as a table: iso,
   play, scene, stored speech count, stored distinct labels, and the names of
   the absent roster characters.

   Then separate that table into three groups and say which group each row is
   in, with your reason:
     - the character genuinely does not speak in this scene (a Folger roster
       can list a character who is present and silent -- check the English
       source text, do not assume);
     - the translator's own editorial choice (this corpus explicitly accepts
       that a translator may cut, merge or rename a part -- there is a
       _KNOWN_UNBOUND mechanism, find it and say whether the row is in it);
     - THE DEFECT: the character speaks in the English source and is simply
       absent from the translation with no recorded reason.

   Only the third group is a finding. Do not report the first two as findings.

2. THE STEM-RULE BLAST RADIUS. Independently of what is stored, compute for
   every vendored scene's roster which English names a FERNANDO-class
   translation would fail to reach -- that is, find the roster names for which
   the five-character / two-length-difference rule is fragile. Name the
   specific English characters across the whole corpus most at risk and say
   why. FERDINAND is one. Find the others; do not stop at one example.

3. THE BACK-TO-BACK SIGNAL. speeches_from_span deliberately does NOT merge two
   consecutive speeches by one character, and its comment says that run is
   THE ONLY SIGNAL that a speaker was missed. Count, per stored file, how many
   times the same label appears on two consecutive lines. Rank the 41 scenes
   by that count. Say which scenes that signal is pointing at, and whether it
   agrees or disagrees with your table from question 1. Where the two disagree,
   say which you believe and why.

4. THE THREE SCANNED ROWS SPECIFICALLY. Three rows were produced by an
   extractor rather than from markup -- find them by transcription_method in
   the manifest. For each, say whether it shows the defect, and give the
   evidence. These are the rows most likely to carry it and they are already
   shipping.

5. DOES THE SUITE CATCH ANY OF THIS? Grep tests/ for anything that compares a
   vendored translation's speaker set against its English roster. Name the
   test file and function if one exists. If none exists, say so plainly -- that
   is a finding in itself, and say what the cheapest such test would assert.

RULES FOR YOUR ANSWER

  - A COUNT IS THE WEAKEST CHECK IN THIS CORPUS and has missed every defect
    found so far. Where you make a claim about a scene, quote a line from the
    stored file.
  - Do not propose a fix to resolve(). Someone else has that question. Your
    output is a measurement of what is already wrong in shipped data.
  - Do not report a scene as defective because its speech count differs from
    the English. Translations differ in length legitimately.

FINISH WITH TWO LISTS, headed exactly:

    ALREADY BROKEN:
      one line per scene you believe carries the defect, with the absent
      character named and one quoted line of evidence

    REFUTED:
      every claim in this brief you could not confirm against the real files,
      naming the file you looked in
