# The page fights back

## The one thing to take away

In Domingos Ramos's 1912 *Macbeth*, the same word identifies both the book and a person speaking; removing it everywhere removes the actor's cue too. [config/source_banks/shakespeare/translations/manifest.json:1576] [scripts/otr_vendor_scan.py:1030]
“Furniture” means the page's surrounding apparatus: repeated titles, act or scene headings, and page numbers; the working distinction is where these sit, not a forbidden vocabulary. [scripts/otr_vendor_scan.py:924]
The historical oddity is that the printer gives identical letters different jobs through placement, while extracted text can lose that distinction. [tests/test_vendor_scan_furniture.py:33]
My inference is that recovering the translation for performance requires recovering those spatial instructions first. [scripts/otr_vendor_scan.py:417]

## What we found

*Citation convention: code and test line numbers refer to commit [18ffc540]. Paths are repository-relative; numbers after a PDF filename identify zero-based PDF pages, starting at 0. Printed page numbers are given separately.*

**1. A title that deletes its own protagonist.** The record says blocking the token `MACBETH` left four of seventeen Macbeth speeches, with thirteen rejected cues. Here “blocking” means rejecting the word wherever it occurs. [scripts/otr_vendor_scan.py:1033]
Its further claim that the title heads “all 240 pages” needs correction: 240 is the scan's length, but printed page 11, PDF index 36, carries an act/scene heading instead. [docs/2026-09-19-shakespeare-vendoring/scan_sources_text_layer_audit.md:24] [tmp/scan_cache/8802fae93d310a78061bb06c.pdf:36]
Printed page 8, index 33, supplies the actual title-head example. [tmp/scan_cache/8802fae93d310a78061bb06c.pdf:33]

The successful rule walks inward from each page edge and stops when ordinary text begins; after taking one title, it must protect an identical speaker cue below it. [scripts/otr_vendor_scan.py:955] [scripts/otr_vendor_scan.py:1044]
This collision is a pattern across editions: Ramos's *Rei Lear* (1919), printed page 8, index 31, also has a title naming a character; extraction split that title into `REI` and `LEAR`. [config/source_banks/shakespeare/translations/manifest.json:1517] [tmp/scan_cache/67e70beaf462724dada711e6.pdf:31] [tests/test_vendor_scan_furniture.py:85]
The printer's distinction is spatial, not lexical: page identity above, speech ownership below. [scripts/otr_vendor_scan.py:786]

**2. A scene heading that pretends the scene has ended.** The guard comment describes a scene finder stopping at a repeated heading on the second page and accepting a plausible fragment. [scripts/otr_vendor_scan.py:1381]
The specific recorded Macbeth failure is more precise: within PDF indices 33–46, it stopped at index 44, printed page 19; that run stored 41 of 48 speeches with a confidence value of 1.0. Those are the recorded run's numbers, not today's scene totals. [tests/test_vendor_scan_furniture.py:615] [tmp/scan_cache/8802fae93d310a78061bb06c.pdf:44]
The page-top `SCENA III` tells a human “still in this scene”; software treated it as another boundary. [scripts/otr_vendor_scan.py:1381]

Deleting every scene heading would lose genuine endings too. The repair counts repeated headings over the whole volume and requires an explicitly supplied ending when the requested scene label also serves as a running head—a heading repeated at the page edge. [scripts/otr_vendor_scan.py:1331] [scripts/otr_vendor_scan.py:1381]
This measured truncation belongs to the 1912 edition; the broader ambiguity also occurs in Ramos's 1919 *Lear*, whose genuine act heading and running head share their wording. [tests/test_vendor_scan_furniture.py:102]
The historical lesson is that a printer can repeat location information without announcing a new dramatic event. [scripts/otr_vendor_scan.py:862]

**3. A hyphen that swallows Banquo.** On printed page 11 of *Macbeth*, `horri-` continues as `vel!` on the next printed line, before `BANQUO`. The text extraction moved `vel!` near the page top. [tmp/scan_cache/8802fae93d310a78061bb06c.pdf:36] [scripts/otr_vendor_scan.py:423]
A rule for repairing broken words then joined `horri-` to `BANQUO`, producing `horriBANQUO` and putting Banquo's opening speech under Macbeth. [scripts/otr_vendor_scan.py:246]
The typesetter had fitted one word across two lines; the extraction and repair together manufactured a new word and lost a speaker. [scripts/otr_vendor_scan.py:250] [scripts/otr_vendor_scan.py:423]

The repair joins a broken word only to a lowercase continuation, and a separate check rejects suspicious lowercase-to-capital joins inside speech text. [scripts/otr_vendor_scan.py:264] [scripts/otr_vendor_scan.py:1435]
This is demonstrably recurrent: `AfasKent` shipped from Ramos's 1919 *Lear*, printed page 8, index 31. Its mixed-case cue required a different detection shape from the capitals in `BANQUO`. [tmp/scan_cache/67e70beaf462724dada711e6.pdf:31] [scripts/otr_vendor_scan.py:218]

**4. A speaker printed beside the line but extracted after the page.** A “hanging cue” is a speaker name set in the left margin beside the first line of speech. [tests/test_vendor_scan_furniture.py:232]
In Jaime Clark's *La tempestad*, dated 1873 in the source record, printed page 50, index 59, has marginal `MIR.` and `FER.` cues that extraction delivered at the bottom; two speeches consequently fell under Prospero. [config/source_banks/shakespeare/translations/leads.json:201] [tmp/scan_cache/0fe883861a983e5a818ad82b.pdf:59] [scripts/otr_vendor_scan.py:420]
The PDF's stored text order therefore differs from the sequence visible on this page. [scripts/otr_vendor_scan.py:420]

The positional reader rebuilds rows from each word's letters and their baseline—the line on which letters sit. [scripts/otr_vendor_scan.py:431]
Using a surrounding rectangle instead accidentally borrowed letters from the row above: on Clark's page 59, `gusano,` moved from Prospero's row to Miranda's even though all words survived and the result still had 38 rows. [scripts/otr_vendor_scan.py:376]
The printing convention places speaker and verse together; preserving every word without that relationship is insufficient. [tests/test_vendor_scan_furniture.py:232] [scripts/otr_vendor_scan.py:381]
Misordered extraction is a cross-edition pattern, established here by Clark's cues and Ramos's displaced syllable, rather than a peculiar wording choice by either translator. [scripts/otr_vendor_scan.py:420]

**5. One changed letter defeats repetition.** On *A Tempestade*'s printed page 31, index 48, the image reads `SCENA II`; OCR, optical character recognition, returned `SOENA II`. [tmp/scan_cache/391af551b32d458ef1ce8888.pdf:48] [scripts/otr_vendor_scan.py:680]
Edition attribution needs an explicit qualification: the scanned title page names **Henrique Braga, 1914**, while the repository manifest names Ramos. This segment follows the page for authorship. [tmp/scan_cache/391af551b32d458ef1ce8888.pdf:6] [config/source_banks/shakespeare/translations/manifest.json:1631]

The malformed heading failed to match the normal repeated title and shipped as `SOENA II A TEMPESTADE 31` inside Prospero's speech at line 86. The review identifies that historical version; the present line is repaired. [docs/2026-09-19-shakespeare-vendoring/RESULT_codex_fold_fbfbe0d6.md:26] [fbfbe0d6] [config/source_banks/shakespeare/translations/pt/tempest_1_2.txt:86] [18ffc540]
The same one-letter error is recorded on three PDF pages, 48, 96 and 158; this exact spelling is established for this volume only. [tests/test_vendor_scan_furniture.py:518]
The repair recognises the heading beside a folio—a printed page number—and the repeated title remaining after that decoration is removed. It does not require the damaged heading word to recur often enough itself. [scripts/otr_vendor_scan.py:680]
The printer combined scene, title and page number into one navigation band; the reader must recognise their relationship. [scripts/otr_vendor_scan.py:708]

## Why it matters for a performance

Banquo's actor would lose his entrance to Macbeth's voice; Prospero would speak a page heading; a scene cut at a repeated heading would stop before its actual ending. These are consequences of the documented text defects, not claims that an audio audience heard them. [scripts/otr_vendor_scan.py:262] [docs/2026-09-19-shakespeare-vendoring/RESULT_codex_fold_fbfbe0d6.md:26] [scripts/otr_vendor_scan.py:1381]
The decisive evidence is the page-to-text relationship: even retaining every word and a plausible row count failed to retain ownership on Clark's page 59. [scripts/otr_vendor_scan.py:381]
Position is essential, but not infallible: the tests preserve the known risk that a genuine speaker repeatedly appearing at the edge could be mistaken for furniture; neither volume measured for that rule triggered it. [tests/test_vendor_scan_furniture.py:165]

## For the hosts: three hooks (one line each) and two open questions

- A cleanup rule erased Macbeth because the book kept saying “Macbeth.” [scripts/otr_vendor_scan.py:1033]
- One hyphen turned Banquo's entrance into `horriBANQUO`. [scripts/otr_vendor_scan.py:262]
- Prospero acquired a line from the page heading, complete with page number. [docs/2026-09-19-shakespeare-vendoring/RESULT_codex_fold_fbfbe0d6.md:26]

1. How could a reader distinguish a genuine cue that repeatedly lands at the edge from a title occupying the same position? [tests/test_vendor_scan_furniture.py:165]
2. What evidence would justify extending the positional reader beyond the qualified pages, given its remaining failures to join cues in Macpherson's volume? [scripts/otr_vendor_scan.py:441]

## Claims register

| Oddity or claim | Edition (translator, year) | Evidence | MEASURED / RULING / INFERRED |
|---|---|---|---|
| Title blocking leaves 4/17 speeches; 240 describes scan length, not verified title-head frequency | *Macbeth*, Ramos, 1912 | [scripts/otr_vendor_scan.py:1033]; [tmp/scan_cache/8802fae93d310a78061bb06c.pdf:36] | MEASURED (4/17 and 240); INFERRED (header-frequency correction) |
| Repeated scene head truncates the recorded run at index 44 | *Macbeth*, Ramos, 1912 | [tests/test_vendor_scan_furniture.py:615] | MEASURED |
| Hyphen joins swallow Banquo and Kent: a cross-edition recurrence | *Macbeth*, Ramos, 1912; *Rei Lear*, Ramos, 1919 | [scripts/otr_vendor_scan.py:218]; [scripts/otr_vendor_scan.py:1435] | MEASURED |
| Detached cues and wrong row assignment despite 38 surviving rows | *La tempestad*, Clark, 1873 | [scripts/otr_vendor_scan.py:376]; [scripts/otr_vendor_scan.py:420] | MEASURED |
| Braga attribution conflicts with repository metadata | *A Tempestade*, Braga, 1914 | [tmp/scan_cache/391af551b32d458ef1ce8888.pdf:6]; [config/source_banks/shakespeare/translations/manifest.json:1631] | INFERRED |
| Three SOENA pages; historical line-86 leak subsequently repaired | *A Tempestade*, Braga, 1914 | [tests/test_vendor_scan_furniture.py:518]; [18ffc540] | MEASURED |
| Spatial context carries printing instructions; blanket word removal cannot preserve them | Editions above | [scripts/otr_vendor_scan.py:1030]; [scripts/otr_vendor_scan.py:431] | INFERRED |
| Wrong voices, spoken headings and premature endings would follow these text defects | Editions above | [scripts/otr_vendor_scan.py:262]; [scripts/otr_vendor_scan.py:1381] | INFERRED |
| Repair logic has stated limits; it is not proof for every edition | Ramos volumes; Clark, 1873; Macpherson, 1897 | [tests/test_vendor_scan_furniture.py:165]; [scripts/otr_vendor_scan.py:441]; [config/source_banks/shakespeare/translations/leads.json:79] | INFERRED |

SOURCES READ: 22
