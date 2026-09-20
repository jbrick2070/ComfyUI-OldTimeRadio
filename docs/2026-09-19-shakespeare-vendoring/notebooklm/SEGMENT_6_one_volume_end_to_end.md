# One volume, end to end: A Tempestade (1914)

## The one thing to take away (3-5 sentences)

Generic optical character recognition cannot turn century-old printed translations into performable radio scripts. In this 1914 Portuguese *Tempest*, raw PDF stream ordering detached cues, three-part running headers contaminated dialogue, and Ferdinand vanished because "FERNANDO" failed a prefix-matching rule [commit fbfbe0d6; PROMPT8_agy_the_portuguese_tempestade_under_the_corrected_reader.md:11-16]. Resolving these required coordinate baseline reconstruction, position-aware header stripping, and an authorized name fold [scripts/otr_vendor_scan.py:363-460, 711-739; manifest.json:1668-1670]. Yet post-commit review revealed two fatal residues—an OCR misprint and an unlabelled lyric song—that would have caused Prospero to chant book titles and sing fairy lyrics on air [RESULT_codex_fold_fbfbe0d6.md:26-42].

## The walk (step by step, each with its number)

### 1. The volume and the scan

The source is a 1914 Portuguese translation of *The Tempest* (*A Tempestade*), published in Porto by Livraria Chardron [manifest.json:1633; PDF scan p.7, 11]. The title page names Henrique Braga as translator; the scan was titled under Dr. Domingos Ramos because Ramos translated eight other Shakespeare plays in Chardron's series, while Braga translated three [PDF scan p.7, 12; manifest.json:1631, 1635]. The 216-page public domain scan is cached locally at `tmp\scan_cache\391af551b32d458ef1ce8888.pdf` [manifest.json:1634-1637; scripts/otr_vendor_scan.py:289-291].

### 2. What --probe showed about headings

The diagnostic probe (`--probe`) revealed early 20th-century Iberian printing practices [scripts/otr_vendor_scan.py:1319-1326]. Across 216 pages, `SCENA I` appeared 45 times, `SCENA II` 32 times, and `SCENA III` 4 times, while act divisions appeared once: `ACTO PRIMEIRO` (p.18), `SEGUNDO` (p.62), `TERCEIRO` (p.110), `QUARTO` (p.140), and `QUINTO` (p.162) [scripts/otr_vendor_scan.py: probe output; PROMPT8_agy_the_portuguese_tempestade_under_the_corrected_reader.md:34-37]. Printers set current scene numbers at the top of every page as running furniture, causing scene labels to repeat dozens of times [scripts/otr_vendor_scan.py:720-728].

### 3. Addressing by page window rather than scene name

Because `SCENA II` appears 32 times, requesting a scene by name bound to the book's first match—a flaw that previously returned Act 2 Scene 1 for *King Lear* 1.1 [scripts/otr_vendor_scan.py:1138-1145]. Under `slice_pages`, scenes are bounded by physical page windows: Act 1 Scene 2 to pages 25–62 (`--pages 25-62`, 38 pages) and Act 3 Scene 1 to pages 110–117 (`--pages 110-117`, 8 pages) [scripts/otr_vendor_scan.py:1135-1156; PDF scan p.25-63, 110-118]. The physical page is the only invariant coordinate; bad windows fail loudly on boundary checks, preventing silent vendoring of fragments [scripts/otr_vendor_scan.py:1138-1150].

### 4. Text-layer ordering and coordinate baselines

Standard PDF extraction via `page.get_text()` returns stream order, detaching cues or throwing syllables across pages (such as `vel!` on page 36 of the 1912 *Macbeth*, corrupting `horrível` and filing Banquo's opening speech under Macbeth) [scripts/otr_vendor_scan.py:420-430]. Bounding box clustering (`y0`) also failed: tall boxes grabbed ascenders above, creating 8.4-point errors on Clark's Spanish *Tempest* that moved words into another speaker's line [scripts/otr_vendor_scan.py:376-384]. The coordinate reader builds one PyMuPDF `TextPage` with `TEXTFLAGS_WORDS`, derives each word's median baseline from its own glyphs, and clusters rows via an adaptive baseline span anchored to row beginnings [scripts/otr_vendor_scan.py:363-385, 417-458].

### 5. Stripping the three-part running header

Recto pages print `SCENA II A TEMPESTADE 31` and verso pages `10 A TEMPESTADE ACTO I` [scripts/otr_vendor_scan.py:720-722]. Under coordinate reading, each header joined into one row [scripts/otr_vendor_scan.py:722-723]. Peeling only folios left `SCENA II A TEMPESTADE`; starting with `SCENA`, it was barred from running-title votes, leaking 38 header rows into Scene 1.2 and 8 into Scene 3.1 [scripts/otr_vendor_scan.py:723-728]. The position rule in `_edge_folio_parts` peels folios and heading tokens at margins, letting `TEMPESTADE` vote and strip cleanly across all 38 pages without erasing scene openings [scripts/otr_vendor_scan.py:711-739].

### 6. The Fernando fold and prefix matching

Miranda, Próspero, Ariel, and Cáliban resolved directly, but Ferdinand was printed `FERNANDO` [manifest.json:1644-1662; commit fbfbe0d6]. The resolver accepts translated names sharing a 5+ letter stem with length difference <= 2 [scripts/otr_vendor_scan.py:636-650]. `FERNANDO` (8 letters) and `FERDINAND` (9 letters) cleared the length floor, but failed prefix matching (`candidate.startswith(fkey) or fkey.startswith(candidate)`), diverging after `FER` [scripts/otr_vendor_scan.py:649; commit fbfbe0d6]. Discarding his cues collapsed Act 3 Scene 1 from 25 speeches to 14 [PROMPT8_agy_the_portuguese_tempestade_under_the_corrected_reader.md:11-16]. Commit `fbfbe0d6` authorized `--fold FERNANDO=FERDINAND`, a scene-scoped alias recorded in `manifest.json` [commit fbfbe0d6; manifest.json:1668-1670, 1706-1708].

### 7. Final counts

With coordinate reading, edge peeling, and the fold, commit `fbfbe0d6` vendored both scenes to match independent manual counts [commit fbfbe0d6]: Act 1 Scene 2 has 137 speeches (PROSPERO 63, MIRANDA 34, ARIEL 24, FERDINAND 10, CALIBAN 6); Act 3 Scene 1 has 25 speeches (FERDINAND 11, MIRANDA 11, PROSPERO 3) [commit fbfbe0d6; manifest.json:1641, 1687].

## The residue (what is still wrong, and what it would sound like)

Post-push review in `RESULT_codex_fold_fbfbe0d6.md` proved that passing tests did not guarantee clean performance scripts [RESULT_codex_fold_fbfbe0d6.md:1-45]. Reviewers found two severe defects in commit `fbfbe0d6`:

### 1. The OCR misprint: SOENA II A TEMPESTADE 31

On page 48 (PDF page 49, folio 31), worn type caused the scanner to read `SCENA` as `SOENA` [scripts/otr_vendor_scan.py:693-697; RESULT_codex_fold_fbfbe0d6.md:26-37]. Failing exact heading spelling, the row escaped peeling, bypassed title voting, and joined Prospero's speech at line 86 of `config/source_banks/shakespeare/translations/pt/tempest_1_2.txt` [RESULT_codex_fold_fbfbe0d6.md:26-32]. On air, Prospero instructs Ariel ("Meu gentil Ariel, escuta ao ouvido") and immediately recites book furniture in an authoritative baritone: *"Soena dois, A Tempestade, trinta e um"* [pt/tempest_1_2.txt:86; RESULT_codex_fold_fbfbe0d6.md:26-27].

### 2. Ariel's song stored inside Prospero: Canto de Áriel

On page 51 (PDF page 52), Braga titled Ariel's lyric `Canto de Áriel` without a speaker label [scripts/otr_vendor_scan.py:200-201; PDF scan p.52]. Early Portuguese printers typeset songs as musical interludes rather than dialogue turns [SCHEMA_speaker_cue_oddities.md:41]. Because `Canto de Áriel` was not in the roster, the parser appended the lyric and barking chorus into Prospero's speech at line 98 [RESULT_codex_fold_fbfbe0d6.md:39-42; pt/tempest_1_2.txt:98]. On air, Prospero dismisses Caliban, then abruptly sings Ariel's lyric and barks like a watchdog: *"Bau-au, cão de guarda ladra"* [pt/tempest_1_2.txt:98-99]. The radio performance would have the Duke of Milan barking while Ariel stays silent.

### The root-cause repair in commit 18ffc540

In commit `18ffc540`, four findings were fixed at root [commit 18ffc540]:
1. **Loose heading peeling:** Short capitalized words (3–8 letters) beside folios peel regardless of OCR spelling (`_LOOSE_HEADING_TOKEN`), removing `SOENA II` on page 48 from line 86 [scripts/otr_vendor_scan.py:692-709; commit 18ffc540].
2. **Song heading detection:** Rule `_SONG_HEADING` recognizes `Canto de [Singer]` as a cue, advancing Act 1 Scene 2 from 137 to 138 speeches (Ariel 24 to 25) [commit 18ffc540; manifest.json:1641; pt/tempest_1_2.txt:98-99].
3. **Fold validation:** Key normalization unified under `label_key()`, prioritizing exact matches [scripts/otr_vendor_scan.py:541-599; commit 18ffc540].
4. **Volume-wide voting:** Heading voting expanded volume-wide, preventing truncation [scripts/otr_vendor_scan.py:1159-1175; commit 18ffc540].

## For the hosts: three hooks (one line each) and two open questions

### Hooks

1. Ferdinand was erased from his own scene because "FERNANDO" has eight letters and shares only three with "FERDINAND," failing a prefix test despite clearing the length floor [commit fbfbe0d6].
2. A single worn piece of lead type turned "SCENA" into "SOENA," causing an automated speech engine to have Prospero chant a book title and page number thirty-one into the microphone [RESULT_codex_fold_fbfbe0d6.md:26-32].
3. Because printers set songs as musical headings rather than spoken turns, an unadjusted audio engine had Prospero bark like a guard dog during Ariel's spirit song [RESULT_codex_fold_fbfbe0d6.md:39-42].

### Open questions

1. When historical printers format theatrical lyrics as musical section titles rather than character dialogue, how can automated parsers distinguish between a character singing and a stage direction describing an offstage sound?
2. When an archival transcription engine cannot bind an ancient translated name, is it more faithful to radio drama to leave the speech unvoiced, or to allow human editors to write custom phonetic fold tables?

## Claims register

| Oddity or claim | Edition (translator, year) | Evidence | Status |
|---|---|---|---|
| Translation published in Porto, 216 pages | Henrique Braga, 1914 | Manifest `pt/tempest` [manifest.json:1630-1709]; scan [PDF p.7, 11] | MEASURED |
| Misattributed in scan filename to Ramos | Braga / Ramos, 1914 | Catalog [PDF p.12]; filename [manifest.json:1635] | MEASURED |
| Scene headings repeat across pages; act headings once | Henrique Braga, 1914 | `--probe`: `SCENA I` x45, `ACTO PRIMEIRO` x1 [scripts/otr_vendor_scan.py: probe] | MEASURED |
| Stable page windows prevent scene collision | Scan lane | `slice_pages` docstring [scripts/otr_vendor_scan.py:1135-1156] | RULING |
| Coordinate reader baselines unite printed rows | Scans | Algorithm in `_word_baselines` [scripts/otr_vendor_scan.py:363-460] | MEASURED |
| Three-part running head leaks 38 headers under folio peel | Henrique Braga, 1914 | `_edge_folio_parts` docstring [scripts/otr_vendor_scan.py:720-738] | MEASURED |
| `FERNANDO` fails prefix match against `FERDINAND` | Henrique Braga, 1914 | Commit `fbfbe0d6` message; stem logic [scripts/otr_vendor_scan.py:643-650] | MEASURED |
| Operator fold restores golden counts (1.2: 137, 3.1: 25) | Henrique Braga, 1914 | Commit `fbfbe0d6` message; manifest rows [manifest.json:1641, 1687] | MEASURED |
| Review finds `SOENA II A TEMPESTADE 31` in Prospero speech | Henrique Braga, 1914 (p.48) | Finding 5 [RESULT_codex_fold_fbfbe0d6.md:26-37; pt/tempest_1_2.txt:86] | MEASURED |
| Review finds Ariel's song inside Prospero speech | Henrique Braga, 1914 (p.51) | Finding 6 [RESULT_codex_fold_fbfbe0d6.md:39-42; pt/tempest_1_2.txt:98] | MEASURED |
| Residual defects fixed: 1.2 advances to 138 speeches | Henrique Braga, 1914 | Commit `18ffc540` message; diff on `tempest_1_2.txt` [commit 18ffc540] | MEASURED |
| Historical letterpress printers set songs as musical titles | European editions | Heading layout across editions [SCHEMA_speaker_cue_oddities.md:41] | INFERRED |

SOURCES READ: 12
