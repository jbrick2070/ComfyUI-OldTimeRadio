# PROMPT 9 -- six lanes, six NotebookLM source segments (2026-09-19, evening)

The operator's new goal: a NotebookLM deep-dive -- working title PROVEN ODDITIES OF
HISTORICAL SHAKESPEARE TRANSLATIONS -- built only from what this corpus work measured. Each lane writes ONE source document
on ONE angle. The segment number is the identity; the lane suggestion is only
where that lane did its best work today -- any lane may take any segment.

Suggested: S1 Astra, S2 Grok, S3 Luna, S4 Terra, S5 Sol (or Composer),
S6 Gemini Flash. Outputs land in `docs/2026-09-19-shakespeare-vendoring/notebooklm/`;
upload that folder to NotebookLM as sources.

Every prompt below is paste-whole. The shared contract is repeated inside each
so a lane never needs a second message.

---

## SEGMENT 1 -- "The page fights back" (suggested: Astra)

```
SEGMENT 1 of 6: THE PAGE FIGHTS BACK -- what a 1910s printed page does to a reader that only sees text

You are writing ONE source document for a NotebookLM notebook. Working title: PROVEN ODDITIES OF HISTORICAL SHAKESPEARE TRANSLATIONS -- a thesis built only from what a small open-source radio-drama project measured while turning century-old printed translations into performable scripts. The bar is a thesis committee's: an ODDITY counts only if (a) it sits in a specific historical edition (translator, year, page), (b) the evidence is a line or page you cite, (c) you say what it reveals about how translators and printers worked, and (d) you say whether it is unique to that edition or a pattern across editions. The best oddities are the ones a mainstream Shakespeare search would never surface -- like nine Hindi 'translations' that turned out to have renamed the cast. Your segment covers one angle only. Five other lanes cover the rest; do not drift into theirs (the translator's cast list, speech counts, the performance ledger, the fidelity rulings, the Portuguese Tempest case study are all taken).

GROUND EVERYTHING. Repo (real Windows files): C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Read-only: no git writes, no edits, no GPU, no pipeline runs; your ONE permitted write is your output file. Python if you need it: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe with PYTHONUTF8=1. Cached page scans: tmp\scan_cache\*.pdf (PyMuPDF is installed; you may render a page to look at it). Read first: docs\2026-09-19-shakespeare-vendoring\README.md (the index), then YOUR SOURCES. The comments in scripts\otr_vendor_scan.py and the docstrings in tests\test_vendor_scan_furniture.py ARE the record: each rule names the volume that forced it and what a listener loses when it breaks.

THE CONTRACT. Every factual claim carries a citation in square brackets: [file:line], [commit hash], or [manifest row iso/play/scene]. A claim you cannot cite is deleted, not softened. Mark each claim's status in the register at the end: MEASURED (a number someone ran), RULING (the operator decided it), or INFERRED (your reading). Do not invent examples. Quote a translator's line only as it appears in the file, and keep quotes short. Write for a curious general listener: plain words, one concrete example per idea, every technical term explained once where it first appears. No hype, no "revolutionary", no overview of the whole project.

YOUR ANGLE. A scanned book is not a text; it is furniture wrapped around a text, and the furniture is made of the same letters as the play. Tell the listener what actually goes wrong when software reads a 1912 Portuguese Macbeth, and what rule finally held. Cases to ground (verify each in the file before using it): the play's title printed at the head of all 240 pages when the title is also the lead character, and what happened when the word was blocklisted (strip_running_titles docstring); a scene label that is ALSO the running head, so the scene finder stops at page two and stores a fragment that looks healthy (the guard comment in main()); the typesetter's hyphen that welded a broken word onto the next speaker's name -- `horri-` + `BANQUO` (the _LINE_BREAK_HYPHEN and _INTERIOR_WELD comments); why "text order" in a PDF is not reading order, and what a hanging speaker cue is (rows_from_coordinates, _word_baselines); the one OCR misprint `SOENA` that a repetition vote cannot see, which shipped inside a speech (docs\2026-09-19-shakespeare-vendoring\RESULT_codex_fold_fbfbe0d6.md, and config\source_banks\shakespeare\translations\pt\tempest_1_2.txt line 86); the rule that furniture is recognised by WHERE it sits, never by what it says.

YOUR SOURCES: scripts\otr_vendor_scan.py (docstrings and comments of strip_running_titles, recurring_headings, running_titles, _edge_folio_parts, rows_from_coordinates, _word_baselines, speeches_from_span, and the guard block in main()); tests\test_vendor_scan_furniture.py (every test docstring); docs\2026-09-19-shakespeare-vendoring\RESULT_codex_fold_fbfbe0d6.md; docs\2026-09-19-shakespeare-vendoring\REPORT_AGY_PROMPT7.md if present; the cached PDFs; git log --since=2026-09-17 -- scripts\otr_vendor_scan.py.

SHAPE (900-1500 words, UTF-8 no BOM, Markdown):
# The page fights back
## The one thing to take away (3-5 sentences)
## What we found (concrete cases, each with its number, page or line)
## Why it matters for a performance (what a voice actor or listener would hear go wrong)
## For the hosts: three hooks (one line each) and two open questions
## Claims register (table: oddity or claim | edition (translator, year) | evidence | MEASURED / RULING / INFERRED)

WRITE TO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-19-shakespeare-vendoring\notebooklm\SEGMENT_1_the_page_fights_back.md (create the folder if missing). End the file with one line: SOURCES READ: <count of files you opened>.
```

---

## SEGMENT 2 -- "The translator's cast is not Shakespeare's" (suggested: Grok)

```
SEGMENT 2 of 6: THE TRANSLATOR'S CAST IS NOT SHAKESPEARE'S -- who speaks in a translation, and who the translator added, merged or renamed

You are writing ONE source document for a NotebookLM notebook. Working title: PROVEN ODDITIES OF HISTORICAL SHAKESPEARE TRANSLATIONS -- a thesis built only from what a small open-source radio-drama project measured while turning century-old printed translations into performable scripts. The bar is a thesis committee's: an ODDITY counts only if (a) it sits in a specific historical edition (translator, year, page), (b) the evidence is a line or page you cite, (c) you say what it reveals about how translators and printers worked, and (d) you say whether it is unique to that edition or a pattern across editions. The best oddities are the ones a mainstream Shakespeare search would never surface -- like nine Hindi 'translations' that turned out to have renamed the cast. Your segment covers one angle only. Five other lanes cover the rest; do not drift into theirs (page furniture and OCR, speech counts, the performance ledger, the fidelity rulings, the Portuguese Tempest case study are all taken).

GROUND EVERYTHING. Repo (real Windows files): C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Read-only: no git writes, no edits, no GPU, no pipeline runs; your ONE permitted write is your output file. Python if you need it: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe with PYTHONUTF8=1. Cached page scans: tmp\scan_cache\*.pdf (PyMuPDF is installed; you may render a page to look at it). Read first: docs\2026-09-19-shakespeare-vendoring\README.md (the index), then YOUR SOURCES.

THE CONTRACT. Every factual claim carries a citation in square brackets: [file:line], [commit hash], or [manifest row iso/play/scene]. A claim you cannot cite is deleted, not softened. Mark each claim's status in the register at the end: MEASURED (a number someone ran), RULING (the operator decided it), or INFERRED (your reading). Do not invent examples. Quote a translator's line only as it appears in the file, and keep quotes short. Write for a curious general listener: plain words, one concrete example per idea, every technical term explained once where it first appears. No hype, no overview of the whole project.

YOUR ANGLE. A cast list is an editorial act. The measured fold tables show translators seating characters Shakespeare left silent, printing a joint speech for two lords, listing the play-within-the-play's roles as separate people, and abbreviating names so that one three-letter label means two different men in one bound volume. Ground each of these: `AJUSTADO, ebanista` speaking in a Spanish Midsummer 3.1 where the English gives Snug nothing; `TISBE` and `PIRAMO` listed as personages apart from Flauta and Borras; `ALB . Y CORN` -- "Deteneos, senor." -- one line for Albany and Cornwall together; `SEB` in the Clark volume that binds The Tempest and Twelfth Night, two Sebastians; `Per` on the Macpherson page that the photograph settled as Hermia. Then the naming layer: FERNANDO for Ferdinand carried as the translator wrote it and bound to the English roster by an explicit, recorded fold [manifest rows pt/tempest 1.2 and 3.1, key "folds"]; a version that RENAMES the cast is an adaptation, not a translation, which is why nine Hindi cells were closed on 2026-09-19 [docs\OTR_STANDING_RULINGS.md, "ONE MODEL'S OCR COUNTS AS VERBATIM"]; Castilho refused for translating from a French intermediary. Explain "unbound" in one sentence: a label that ships raw, costing a voice and never a line, because a wrong fold is worse than no fold. THEN THE IN-UNISON DILEMMA, which the operator names as the thesis's centre: a line Shakespeare gives to several people at once. Every edition marks it differently -- Folger prints ALL; the 1912 Portuguese Macbeth prints `TODAS TRES` for the three witches and the resolver maps it to ALL [scripts\otr_vendor_scan.py, FUNCTION_NAMES / ORDINALS and the `1.ª FEITICEIRA` comment; manifest row pt/macbeth 1.3]; the Spanish Lear prints `ALB . Y CORN` for Albany and Cornwall together and it ships UNBOUND; the Portuguese Tempest prints `Côro (dispersamente)` inside Ariel's song [pt	empest_1_2.txt, the ARIEL speech beginning `Desembarca`]. Show the four shapes side by side and say what a translator was deciding each time. The witches are the same problem from the other side: `1.ª FEITICEIRA` / `BRUJA 1.ª` -- an ordinal and a function word in either order, because the two languages disagree about which comes first -- so a character who is a NUMBER in one language is a WORD in another.

YOUR SOURCES: docs\2026-09-19-shakespeare-vendoring\GROK_FOLD_TABLES_measured.md (whole file, including the image-verdict banner); docs\2026-09-19-shakespeare-vendoring\PROMPT7_grok_a_wrong_fold_is_worse_than_no_fold.md; docs\OTR_STANDING_RULINGS.md (the two 2026-09-19 entries); config\source_banks\shakespeare\translations\manifest.json (every speaker_map, and the rows carrying "folds"); the resolve() function and its comments in scripts\otr_vendor_scan.py; docs\GO_FORWARD_PLAN.md section 2 row 1; the cached PDFs for the pages you cite.

SHAPE (900-1500 words, UTF-8 no BOM, Markdown):
# The translator's cast is not Shakespeare's
## The one thing to take away (3-5 sentences)
## What we found (concrete cases, each with its page, label and count)
## Why it matters for a performance (what a listener would hear go wrong)
## For the hosts: three hooks (one line each) and two open questions
## Claims register (table: oddity or claim | edition (translator, year) | evidence | MEASURED / RULING / INFERRED)

WRITE TO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-19-shakespeare-vendoring\notebooklm\SEGMENT_2_the_translators_cast.md (create the folder if missing). End the file with one line: SOURCES READ: <count of files you opened>.
```

---

## SEGMENT 3 -- "Nobody agrees how many speeches are in a scene" (suggested: Luna)

```
SEGMENT 3 of 6: NOBODY AGREES HOW MANY SPEECHES ARE IN A SCENE -- why a count is a property of the edition and the reader, not the play

You are writing ONE source document for a NotebookLM notebook. Working title: PROVEN ODDITIES OF HISTORICAL SHAKESPEARE TRANSLATIONS -- a thesis built only from what a small open-source radio-drama project measured while turning century-old printed translations into performable scripts. The bar is a thesis committee's: an ODDITY counts only if (a) it sits in a specific historical edition (translator, year, page), (b) the evidence is a line or page you cite, (c) you say what it reveals about how translators and printers worked, and (d) you say whether it is unique to that edition or a pattern across editions. The best oddities are the ones a mainstream Shakespeare search would never surface -- like nine Hindi 'translations' that turned out to have renamed the cast. Your segment covers one angle only. Five other lanes cover the rest; do not drift into theirs (page furniture and OCR, the translator's cast list, the performance ledger, the fidelity rulings, the Portuguese Tempest case study are all taken).

GROUND EVERYTHING. Repo (real Windows files): C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Read-only: no git writes, no edits, no GPU, no pipeline runs; your ONE permitted write is your output file. Python if you need it: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe with PYTHONUTF8=1. Read first: docs\2026-09-19-shakespeare-vendoring\README.md (the index), then YOUR SOURCES.

THE CONTRACT. Every factual claim carries a citation in square brackets: [file:line], [commit hash], or [manifest row iso/play/scene]. A claim you cannot cite is deleted, not softened. Mark each claim's status in the register at the end: MEASURED (a number someone ran), RULING (the operator decided it), or INFERRED (your reading). Do not invent examples or numbers. Write for a curious general listener: plain words, one concrete example per idea, every technical term explained once. No hype, no overview of the whole project.

YOUR ANGLE. Ask the simplest question -- how many speeches are in Macbeth act 1 scene 3? -- and show that the answer depends on which edition you hold and how you read it. Build the segment from real numbers: the manifest's speaker_labels for the SAME scene across languages (list every edition of macbeth 1.3, king_lear 1.1, and any other scene vendored in three or more languages, with its count); the Portuguese 49 against the French and Italian 51, and the experiment that merged back-to-back speeches to 36 and was reverted because it "DESTROYED THE ONLY SIGNAL that says a speaker was missed" (the comment at the end of speeches_from_span in scripts\otr_vendor_scan.py); Kent going from 12 to 13 speeches when one printed line was split in two, with the scene 80 to 81 and every hash matching (C:\Users\jeffr\AppData\Local\Temp\otr_three_commit_qa\QA_three_unreviewed_commits.md, if present -- cite it as a QA report, and skip it if the file is gone); the golden per-cell counts a lane measured by hand for the Spanish scans (docs\2026-09-19-shakespeare-vendoring\GROK_FOLD_TABLES_measured.md, "Expected output per cell"); and the ruling's last paragraph: count what the EDITION MARKS, never what survived, because a swallowed character never appears as a speaker to look for -- the Prince absorbed into Lady Montague's one-line part [docs\OTR_STANDING_RULINGS.md, "THE CHECK THAT SURVIVES"]. Also grep docs\ for "hold note" and "weakest" and use what you find, cited.

YOUR SOURCES: config\source_banks\shakespeare\translations\manifest.json (speaker_labels and distinct_speakers per row); scripts\otr_vendor_scan.py (speeches_from_span and its closing comment; the "only %d lines between headings" refusal in scripts\otr_vendor_shakespeare.py extract()); docs\OTR_STANDING_RULINGS.md (both 2026-09-19 entries); docs\2026-09-19-shakespeare-vendoring\GROK_FOLD_TABLES_measured.md; tests\test_verbatim_corpus.py (any test that reasons about counts or back-to-back runs); the QA report in TEMP named above.

SHAPE (900-1500 words, UTF-8 no BOM, Markdown):
# Nobody agrees how many speeches are in a scene
## The one thing to take away (3-5 sentences)
## What we found (the numbers, side by side, each cited)
## Why it matters for a performance (what a wrong count hides)
## For the hosts: three hooks (one line each) and two open questions
## Claims register (table: oddity or claim | edition (translator, year) | evidence | MEASURED / RULING / INFERRED)

WRITE TO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-19-shakespeare-vendoring\notebooklm\SEGMENT_3_nobody_agrees_on_the_count.md (create the folder if missing). End the file with one line: SOURCES READ: <count of files you opened>.
```

---

## SEGMENT 4 -- "From text to performance: the ledger" (suggested: Terra)

```
SEGMENT 4 of 6: FROM TEXT TO PERFORMANCE -- what a translation must carry before a machine can perform it

You are writing ONE source document for a NotebookLM notebook. Working title: PROVEN ODDITIES OF HISTORICAL SHAKESPEARE TRANSLATIONS -- a thesis built only from what a small open-source radio-drama project measured while turning century-old printed translations into performable scripts. The bar is a thesis committee's: an ODDITY counts only if (a) it sits in a specific historical edition (translator, year, page), (b) the evidence is a line or page you cite, (c) you say what it reveals about how translators and printers worked, and (d) you say whether it is unique to that edition or a pattern across editions. The best oddities are the ones a mainstream Shakespeare search would never surface -- like nine Hindi 'translations' that turned out to have renamed the cast. Your segment covers one angle only. Five other lanes cover the rest; do not drift into theirs (page furniture and OCR, the translator's cast list, speech counts, the fidelity rulings, the Portuguese Tempest case study are all taken).

GROUND EVERYTHING. Repo (real Windows files): C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Read-only: no git writes, no edits, no GPU, no pipeline runs; your ONE permitted write is your output file. Python if you need it: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe with PYTHONUTF8=1. Read first: docs\2026-09-19-shakespeare-vendoring\README.md (the index), then YOUR SOURCES.

THE CONTRACT. Every factual claim carries a citation in square brackets: [file:line], [commit hash], or [manifest row iso/play/scene]. A claim you cannot cite is deleted, not softened. Mark each claim's status in the register at the end: MEASURED, RULING, or INFERRED. Do not invent examples. Write for a curious general listener: plain words, one concrete example per idea, every technical term explained once. No hype, no overview of the whole project.

YOUR ANGLE. A text file of a play is not performable. To become radio it needs a ledger: every line with exactly one owner, every owner bound to a roster name, every roster name with a gender and a voice. Explain what the speaker_map is and what it is for (the contract comment near "1A STREGA" in nodes\_otr_verbatim_corpus.py; _validate_speaker_map; speaker_bindings), how the bindings reach the script writer (nodes\OTR_LedgerScriptWriter.py, the call that passes speaker_bindings to the planner), where gender comes from (nodes\_otr_roster_gender.py), and what breaks when any link is missing. Ground the failure shapes: a title-case label that parses as no speaker at all so a character's lines merge into whoever spoke before him at runtime while the file looks fine (the "UPPER-CASE ON WRITE, ALWAYS" comment in speeches_from_span, scripts\otr_vendor_scan.py); "Malvolio speaks with a woman's voice" being a correctness bug rather than a quality wish (CLAUDE.md, the STORY QUALITY IS DONE directive's exception list); an UNBOUND speaker costing a voice and never a line [docs\OTR_STANDING_RULINGS.md, 2026-09-19 "A MANGLED NATIVE TEXT..."]; and whatever the record holds on "ONE OWNER" and "_is_upper_label" (grep docs\ and nodes\ for both; cite what you find, skip what you cannot). THEN THE TWO SHAPES THAT BREAK 'ONE OWNER': (a) a SONG -- the 1914 Portuguese Tempest heads Ariel's song `Canto de Áriel` on its own line instead of labelling ARIEL, so until commit 18ffc540 the whole song sat inside a Prospero speech and would have been sung in his voice; read the `_SONG_HEADING` comment in scripts\otr_vendor_scan.py and the commit message; (b) a UNISON line -- Folger's ALL, the Portuguese `TODAS TRES`, the Spanish `ALB . Y CORN`, the chorus `Côro` -- which the ledger must still give to exactly one voice. Find what the runtime actually does with a roster name ALL (grep nodes\ for it, and the English source config\source_banks\shakespeare\sources\macbeth__act1_scene3.txt for the ALL speaker) and state it as measured: one voice, a chosen voice, or a gap. This is the operator's 'in unison dilemma' -- the collision between a literary convention and a performance ledger -- and your segment is where it is explained.

YOUR SOURCES: nodes\_otr_verbatim_corpus.py (the speaker_map contract comment, load_manifest, _validate_speaker_map, speaker_bindings, vendored_text); nodes\OTR_LedgerScriptWriter.py (the vendored-row binding site, roughly lines 3800-3850); nodes\_otr_roster_gender.py; nodes\_otr_passage_selector.py (parse_speeches and any label-shape rule); tests\test_verbatim_corpus.py; docs\OTR_STANDING_RULINGS.md (2026-09-19 entries); CLAUDE.md in the repo root (the correctness-vs-quality paragraph).

SHAPE (900-1500 words, UTF-8 no BOM, Markdown):
# From text to performance: the ledger
## The one thing to take away (3-5 sentences)
## What we found (the contract, and the concrete failure each rule prevents)
## Why it matters for a performance (what the listener hears when a link is missing)
## For the hosts: three hooks (one line each) and two open questions
## Claims register (table: oddity or claim | edition (translator, year) | evidence | MEASURED / RULING / INFERRED)

WRITE TO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-19-shakespeare-vendoring\notebooklm\SEGMENT_4_from_text_to_performance.md (create the folder if missing). End the file with one line: SOURCES READ: <count of files you opened>.
```

---

## SEGMENT 5 -- "The ruling: a mangled native text beats a clean AI translation" (suggested: Sol, or Composer)

```
SEGMENT 5 of 6: THE RULING -- the fidelity decisions this project made, and the reasons written down for each

You are writing ONE source document for a NotebookLM notebook. Working title: PROVEN ODDITIES OF HISTORICAL SHAKESPEARE TRANSLATIONS -- a thesis built only from what a small open-source radio-drama project measured while turning century-old printed translations into performable scripts. The bar is a thesis committee's: an ODDITY counts only if (a) it sits in a specific historical edition (translator, year, page), (b) the evidence is a line or page you cite, (c) you say what it reveals about how translators and printers worked, and (d) you say whether it is unique to that edition or a pattern across editions. The best oddities are the ones a mainstream Shakespeare search would never surface -- like nine Hindi 'translations' that turned out to have renamed the cast. Your segment covers one angle only. Five other lanes cover the rest; do not drift into theirs (page furniture and OCR, the translator's cast list, speech counts, the performance ledger, the Portuguese Tempest case study are all taken).

GROUND EVERYTHING. Repo (real Windows files): C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Read-only: no git writes, no edits, no GPU, no pipeline runs; your ONE permitted write is your output file. Read first: docs\2026-09-19-shakespeare-vendoring\README.md (the index), then YOUR SOURCES.

THE CONTRACT. Every factual claim carries a citation in square brackets: [file:line], [commit hash], or [manifest row iso/play/scene]. A claim you cannot cite is deleted, not softened. Mark each claim's status in the register at the end: MEASURED, RULING, or INFERRED. Quote the operator's rulings as written in the file (they are typed fast; quote them as they are, then restate plainly). Write for a curious general listener: plain words, one concrete example per idea, every technical term explained once. No hype, no overview of the whole project.

YOUR ANGLE. This project chose, in writing, what "faithful" means, and each choice has a reason attached. Lay out the decisions as a listener could argue with them: (1) a mangled native text beats a clean AI translation -- gates refuse MISATTRIBUTION and never IMPERFECTION; an unresolvable cue becomes an unbound speaker, never merged [docs\OTR_STANDING_RULINGS.md, 2026-09-19 top entry]; (2) one model's OCR of a page scan counts as verbatim, and what the same-day three-model read of Tsubouchi's Japanese Macbeth showed about which model -- thirteen labels, four wrong on a first glance [same file, second entry]; (3) a renamed cast is an adaptation and is refused; a translation made through an intermediary language is refused (Castilho from French); (4) a wrong fold is worse than no fold -- the asymmetry between losing a voice and putting a line in the wrong mouth [docs\2026-09-19-shakespeare-vendoring\PROMPT7_grok_a_wrong_fold_is_worse_than_no_fold.md, and the GROK_FOLD_TABLES banner]; (5) rights refuse nothing, and why the translator's death year and first-publication year are still REQUIRED fields (the comment block in nodes\_otr_verbatim_corpus.py load_manifest, around the words "rights refuse nothing"); (6) the two sides of the 09-19 rule that a repository comment is not a filter: generated episode content is not filtered, but the CODE and COMMENTS stay clean (CLAUDE.md). Close by naming what these rulings do NOT settle -- the open questions the record itself lists.

YOUR SOURCES: docs\OTR_STANDING_RULINGS.md (the 2026-09-19 and 2026-09-18 entries in full); docs\2026-09-19-shakespeare-vendoring\PROMPT7_grok_a_wrong_fold_is_worse_than_no_fold.md; docs\2026-09-19-shakespeare-vendoring\GROK_FOLD_TABLES_measured.md (banner and closing section); nodes\_otr_verbatim_corpus.py (load_manifest comments); CLAUDE.md in the repo root; docs\GO_FORWARD_PLAN.md section 2 row 1.

SHAPE (900-1500 words, UTF-8 no BOM, Markdown):
# The ruling: a mangled native text beats a clean AI translation
## The one thing to take away (3-5 sentences)
## The decisions (one short section per ruling: what was decided, the quoted words, the reason, one example)
## Why it matters for a performance
## For the hosts: three hooks (one line each) and two open questions
## Claims register (table: oddity or claim | edition (translator, year) | evidence | MEASURED / RULING / INFERRED)

WRITE TO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-19-shakespeare-vendoring\notebooklm\SEGMENT_5_the_ruling.md (create the folder if missing). End the file with one line: SOURCES READ: <count of files you opened>.
```

---

## SEGMENT 6 -- "One volume, end to end: A Tempestade (1914)" (suggested: Gemini Flash)

```
SEGMENT 6 of 6: ONE VOLUME, END TO END -- the 1914 Portuguese Tempest, from a Wikimedia scan to two performable scenes, including what is still wrong with them

You are writing ONE source document for a NotebookLM notebook. Working title: PROVEN ODDITIES OF HISTORICAL SHAKESPEARE TRANSLATIONS -- a thesis built only from what a small open-source radio-drama project measured while turning century-old printed translations into performable scripts. The bar is a thesis committee's: an ODDITY counts only if (a) it sits in a specific historical edition (translator, year, page), (b) the evidence is a line or page you cite, (c) you say what it reveals about how translators and printers worked, and (d) you say whether it is unique to that edition or a pattern across editions. The best oddities are the ones a mainstream Shakespeare search would never surface -- like nine Hindi 'translations' that turned out to have renamed the cast. Your segment covers one angle only. Five other lanes cover the rest; do not drift into theirs (page furniture in general, the translator's cast list in general, speech counts in general, the performance ledger, the fidelity rulings are all taken). Yours is a CASE STUDY: one book, every step, the real numbers, and the honest residue.

GROUND EVERYTHING. Repo (real Windows files): C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Read-only: no git writes, no edits, no GPU, no pipeline runs; your ONE permitted write is your output file. Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe with PYTHONUTF8=1 (PyMuPDF is installed). The volume: the manifest rows pt/tempest 1.2 and 3.1 give the source_url; the cached copy is one of tmp\scan_cache\*.pdf -- open each and match by its first page. You MAY run a dry run to reproduce the numbers (no --write): python scripts\otr_vendor_scan.py --iso pt --play tempest --scene 1.2 --stem tempest__act1_scene2 --reading-order coordinates --pages 25-62 --scene-label "SCENA II" --end-label "ACTO SEGUNDO" --fold FERNANDO=FERDINAND (and for 3.1: --scene 3.1 --stem tempest__act3_scene1 --pages 110-117 --scene-label "SCENA I" --end-label "SCENA II").

THE CONTRACT. Every factual claim carries a citation in square brackets: [file:line], [commit hash], or [manifest row iso/play/scene]. A claim you cannot cite is deleted, not softened. Mark each claim's status in the register at the end: MEASURED, RULING, or INFERRED. Quote the translation only as it appears in the file, and keep quotes short. Write for a curious general listener: plain words, every technical term explained once. No hype.

YOUR ANGLE. Walk the listener through the book as the tools met it. What the scan is (translator, year, licence, page count) [manifest rows]; what --probe showed about its headings; why the scene was addressed by PAGE WINDOW rather than by name (slice_pages docstring); why the text layer's order was wrong and what the coordinate reader changed (rows_from_coordinates and _word_baselines docstrings); the three-part running head `SCENA II A TEMPESTADE 31` and how the position rule removed it from 38 pages; the printed FERNANDO bound to FERDINAND by a recorded fold, and why the resolver could not do it alone (eight letters that fail a prefix test, not a length floor) [commit fbfbe0d6 message; manifest "folds"]; the final counts -- 137 speeches in 1.2 (PROSPERO 63, MIRANDA 34, ARIEL 24, FERDINAND 10, CALIBAN 6) and 25 in 3.1 (FERDINAND 11, MIRANDA 11, PROSPERO 3) -- matching a lane's independent hand count. THEN THE RESIDUE, stated plainly, because it is the most instructive part: after the push, a reviewer found the OCR misprint `SOENA II A TEMPESTADE 31` sitting at the end of a Prospero speech [pt\tempest_1_2.txt:86], and Ariel's song "Canto de Ariel" printed as a heading, not a speaker label, so it was stored inside Prospero's speech [pt\tempest_1_2.txt:98] -- both in docs\2026-09-19-shakespeare-vendoring\RESULT_codex_fold_fbfbe0d6.md. Say what each would sound like on air. If the repository has moved past that commit and a fix landed, cite the fixing commit and say what changed; do not assume it did.

YOUR SOURCES: config\source_banks\shakespeare\translations\manifest.json (the four pt rows); config\source_banks\shakespeare\translations\pt\tempest_1_2.txt and tempest_3_1.txt; the commit fbfbe0d6 (git show --stat and its message); docs\2026-09-19-shakespeare-vendoring\RESULT_codex_fold_fbfbe0d6.md; docs\2026-09-19-shakespeare-vendoring\PROMPT8_agy_the_portuguese_tempestade_under_the_corrected_reader.md and any REPORT that answered it; scripts\otr_vendor_scan.py (the docstrings named above, and the --fold handling in main()); the cached PDF.

SHAPE (900-1500 words, UTF-8 no BOM, Markdown):
# One volume, end to end: A Tempestade (1914)
## The one thing to take away (3-5 sentences)
## The walk (step by step, each with its number)
## The residue (what is still wrong, and what it would sound like)
## For the hosts: three hooks (one line each) and two open questions
## Claims register (table: oddity or claim | edition (translator, year) | evidence | MEASURED / RULING / INFERRED)

WRITE TO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-19-shakespeare-vendoring\notebooklm\SEGMENT_6_one_volume_end_to_end.md (create the folder if missing). End the file with one line: SOURCES READ: <count of files you opened>.
```
