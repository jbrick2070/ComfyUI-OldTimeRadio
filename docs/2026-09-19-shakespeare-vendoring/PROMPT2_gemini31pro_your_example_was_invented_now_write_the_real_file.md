Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

YOUR REGISTRY DESIGN IS ADOPTED IN SHAPE. `scan_editions.json` keyed by
PDF sha256, folds scoped per scene, `unbound_speakers` as a list on the
manifest row, unbound labels omitted from `speaker_map`, and the
camel-weld assertion -- which refutes the brief's own guess that nothing
cheap catches `horriBANQUO`; the driver had already used that exact
regex as a corpus census and it found both welds and nothing else.

YOUR EXAMPLE WAS INVENTED, IN THREE PLACES, AND WOULD HAVE POISONED THE
REGISTRY IF LOADED. That is this brief. The "one real scene filled in":
  * sha256 `e2b9c7b9...` -- no cached volume has it. The Macpherson
    volume is `5c4847276297a7ffcd752535afa7b7b670f714598472eb621f3baf0e7d77aa68`.
  * URL `Obras_dram...(187-?)_Tomo_1.pdf` -- no such file. The real one
    is `...Obras_dram%C3%A1ticas_de_Guillermo_Shakespeare_-_Tomo_I_%281897%29.pdf`.
  * "translator: Jaime Clark" for Obras dramáticas -- that volume is
    Guillermo MACPHERSON. Clark is the OTHER two volumes. Your prose
    repeats it: "multi-play volumes like Jaime Clark's Obras dramáticas".
  * the fold `BUF/BUR -> FOOL`, `D . PED`, `PAJE` under king_lear 1.1 --
    those forms are Twelfth Night's Fool and Much Ado's Don Pedro and
    Boy. None occurs in Lear 1.1.
A schema whose worked example cannot be loaded is a sketch. Write the
file.

YOUR JOB IS TO PRODUCE THE REAL `scan_editions.json`, THEN ATTACK IT.
Read-only on the repo; scratch in %TEMP%; no git writes; no GPU. Every
value must come from a file or a measurement, never from memory.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

THE MEASURED DATA, ALL ON MAIN -- read it, do not recall it:
  * The five PDF sha256s and filenames under tmp\scan_cache: compute them
    yourself with hashlib; do not copy them from any brief.
  * The URLs: the `page_scan` rows of
    config\source_banks\shakespeare\translations\leads.json.
  * The eight Spanish windows and labels, in the commit message of
    76f88d8c and in docs\2026-09-19-shakespeare-vendoring\PROMPT8_flash_rerun_the_windows_under_the_corrected_reader.md.
  * The two Portuguese Tempestade cells: windows are NOT yet measured --
    a lane is measuring them now; leave `"pages": null` with a note.
  * The two already-vendored Portuguese scenes (pt/king_lear 1.1,
    pt/macbeth 1.3): their act/scene/end labels are on their manifest
    rows (`edition_label`) and in the commit messages of 2bb45e9e and
    2e9ff62e.
  * FOLD TABLES: another lane is measuring them per cell right now.
    Leave every `"fold": {}` EMPTY with a note naming what is pending.
    The one exception the operator ruled on directly: pt Tempestade gets
    `"FERNANDO": "FERDINAND"` in both scenes, because he explicitly
    allowed that alias and explicitly kept it out of PLACE_NAMES.

PRODUCE THE FILE with all five editions and all twelve scenes (8 es, 2 pt
Tempestade, 2 pt vendored), every hash computed, every URL copied, every
window and label copied from its source, every fold empty except the
ruled one. Include per scene the English `stem` and the roster size read
with nodes\_otr_roster_gender.py::load_roster_characters.

THEN ATTACK THREE OF YOUR OWN DECISIONS:
  1. ASSERTION 11, `ASSERT_NO_CONSECUTIVE_DUPLICATES`, contradicts a
     decision already written into `speeches_from_span`: "A BACK-TO-BACK
     RUN IS LEFT ALONE ON PURPOSE ... merging DESTROYED THE ONLY SIGNAL
     that says a speaker was missed." Read that comment. A back-to-back
     run is the SYMPTOM the corpus wants to SEE, not a write failure --
     and a translator may legitimately split one speech in two (the
     2026-09-19 "translator's own choice" ruling). Should 11 be an abort,
     a warning on the console, or a manifest field? Decide, and say what
     each choice costs on Much Ado 2.3 where the detector reaches 81 =
     Folger 81.
  2. `ASSERT_SPAN_CEILING` at 35,000 is a number. Midsummer 3.2 is 21,093
     and King Lear 1.1 is 13,302 -- fine. But Tempest 1.2 in the
     Portuguese, a 26,000-character scene, and the longest Folger scenes
     could approach it. Replace the constant with a ratio against the
     English source's length, or defend the constant with the longest
     scene in the corpus.
  3. `alignment_confidence: 1.0` is kept as a literal because
     `select_scene` reads it against 0.8 and `_REQUIRED_SCENE_FIELDS`
     requires it. That is a field the runtime READS and the vendor
     ASSERTS. Say in one paragraph what a reader of the manifest is
     entitled to believe when they see 1.0 on a scene with four unbound
     speakers, and whether "1.0 means every emitted line is attributed
     per the registry" is honest enough to keep, or whether the field
     should carry a second value the runtime does not read.

DO NOT propose changes to extract() or find_label().

FINISH WITH:
    FILE:        the complete scan_editions.json, loadable, with every
                 hash computed and every source cited in a `note`
    DECISIONS:   assertion 11, the ceiling, and confidence -- one
                 paragraph each
    REFUTED:     every claim above you could not confirm, INCLUDING the
                 three fabrications named above if you believe any was
                 not a fabrication
