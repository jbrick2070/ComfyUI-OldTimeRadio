# Public-domain Shakespeare translations -- inventory (2026-09-18)

Research agent, web-verified where marked. Languages from
`config/episode_languages.json` (es, pt, it, fr, hi, ja, zh). Plays from
`config/source_banks/shakespeare/curated_scenes.sample.json`: 14 scenes across
Macbeth, Tempest, Midsummer, Twelfth Night, Much Ado, As You Like It, Hamlet,
King Lear, Romeo and Juliet, Comedy of Errors (Folger CC BY-NC text).

PD line used by this inventory: translator died in 1944 or earlier -- a
CONSERVATIVE bound, not a legal test. The v2 corpus spec beside this file
(`shakespeare_corpus_spec_v2.yaml`) states the real tests: US = first
published before 1931; life+70 (EU, UK, most of Latin America) = died before
1956; life+50 (CN; JP pre-2018 works) = died before 1976. Which one governs
is the operator's Q1 in that spec. Every translator marked PD here clears all
three.

| Lang | Play | Translator (died) | PD | Source | Text format | Notes |
|---|---|---|---|---|---|---|
| fr | all 10 | Francois-Victor Hugo (1873) | Yes | fr.wikisource.org, Oeuvres completes de Shakespeare / Hugo 1865-1872 (one page per play, e.g. `Macbeth_(trad._Hugo)`), "Texte valide" | `PREMIERE SORCIERE.` caps + period on its own line, speech below | Verse as dashed prose. Best single source. |
| fr | all 10 | Francois Guizot (1874) | Yes | Gutenberg plain text: Macbeth 13868, Hamlet 15032, Tempete 15071, Romeo 18143, Songe 17930, Roi Lear 18312, Jour des Rois 16128, Comedie des Meprises 15848, Beaucoup de bruit 15846, Comme il vous plaira 18162 | Play text | Translator verified on 13868/15032/15071/18143; the other six from the listing, not opened. |
| it | all 10 | Carlo Rusconi (1889) | Yes | it.wikisource.org, Teatro completo di Shakspeare (SAL 100%) | Prose; small-caps speaker + period on the SAME line (`1a Strega. In qual di...`) | Complete; needs a same-line regex. |
| it | Tempest | Diego Angeli (1937) | Yes | Gutenberg 26169 | Play text (not format-checked) | Alternative. |
| es | Macbeth, Romeo | Menendez y Pelayo (1912) | Yes | es.wikisource `Macbeth_(Menendez_y_Pelayo_tr.)`, `Romeo_y_Julieta_(...)`; Gutenberg 53207 | `BRUJA 1.a` caps on own line | Clean. |
| es | As You Like It, Comedy of Errors | Jose Arnaldo Marquez (1904) | Yes | Gutenberg 59686 (vol. 4); es.wikisource `Como_gusteis_(Marquez_tr.)` | Mixed `ORLANDO.` and `Adam.--` dash style | Mixed-case dash labels fail the caps test. |
| es | Hamlet | Leandro F. de Moratin (1828) | Yes | Gutenberg 56454 | Prose (1798) | Old but clean. |
| es | Tempest, Midsummer, Twelfth Night, Much Ado, Lear | Macpherson (1898), Marquez, Cane (1905), Clark (1875) | Yes | es.wikisource author pages | Scan/index only ("A transcribir") | No plain text verified. |
| pt | Hamlet | Luis I of Portugal (1889) | Yes | Gutenberg 25667 | `BERNARDO` caps on own line, prose below | The only clean Portuguese hit. |
| pt | Midsummer | A. F. de Castilho (1875) | Yes | archive.org `sonhodumanoited00shakgoog` | Verse; Google OCR, confidence 77 | Usable with cleanup. |
| pt | Macbeth, Tempest | Domingos Ramos / Henrique Braga (death years NOT verified) | Unverified | pt.wikisource author page (Galeria PDFs, 1912/1914) | Scan only | Attribution disagrees across sources. |
| ja | Macbeth, Midsummer, Twelfth Night | Tsubouchi Shoyo (1935) | **Yes, both tests** | Wikimedia Commons, NDL scans of 沙翁傑作集 No. 10 (1923), No. 9 (1921), No. 18 (1921) | PAGE SCAN, no text layer | **Operator 2026-09-18, and it beats this file's own row.** The PRE-1931 series clears the US test where the 1933-35 revision does not. Gate-verified HTTP 200 on all three. Needs transcription, not another source. |
| ja | Tempest | Tsubouchi (1935) | Yes, both tests | 沙翁傑作集 No. 7 (1921), Commons | scan | URL not yet located -- a constructed `NDL979376` guess 404'd, which is why a guessed URL is never a fact. |
| ja | Romeo | Tsubouchi (1935) | **NO -- US test** | aozora.gr.jp card 42773 | old kana / old kanji | The Aozora text is the 1933-35 新修 revision, first published 1933. DO NOT USE; the pre-1931 series is the approved source. |
| ja | Hamlet, Lear, Much Ado, As You Like It, Comedy of Errors | Tsubouchi | depends on volume | aozora person 264 (作業中), NDL / Commons for the pre-1931 volumes | scan or not yet transcribed | Check the 沙翁傑作集 volume year per play before using anything. |
| zh | Hamlet, Tempest, Midsummer | Zhu Shenghao (1944) | Yes | zh.wikisource (traditional; act subpages) | Hamlet: one-character abbreviated speaker + full-width space; Tempest: full-width colon | Inconsistent label style. |
| zh | the other seven | Zhu | Yes | zh.wikisource collection page | Red links, not transcribed | Gutenberg: none. |
| hi | As You Like It (1917) | Lala Sitaram (1937) | Yes | archive.org `in.ernet.dli.2015.263809` (`_djvu.txt`) | `NAME--speech` same line; OCR moderate-to-poor; names Indianised | Only PD Hindi hit. Sitaram's Hamlet/Macbeth exist in print, no scan found. |
| hi | everything else | Bachchan (2003), Rangeya Raghav (1962) | NO | archive.org, not PD | -- | hi.wikisource has no Shakespeare author page. |

## Coverage gaps

- Hindi: one play, one poor scan.
- Japanese: SUPERSEDED 2026-09-18. Three plays (five of the fourteen target
  scenes) are rights-cleared under BOTH tests and located as NDL scans of the
  pre-1931 沙翁傑作集; they need transcription, not a different source. The
  Aozora text everyone reaches for first is the 1933 revision and fails the
  US test. Old kana remains a `misaki[ja]` G2P question for whatever is
  transcribed.
- Mandarin: 3 of 10, traditional characters, two label conventions.
- Portuguese: Hamlet clean; Midsummer OCR; nothing else in plain text.
- Spanish: 5 of 10 clean, two with `Name.--` mixed-case labels.

## Loader caveat

`_otr_shakespeare_sources._speaker_from_line` recognises only ASCII `:` or
Latin ALL-CAPS. Full-width `：`, full-width space, `--` and `Nom.--` all fall
through. Either a per-language prefix rule, or a one-time normalisation to
`NAME:` at vendoring time (cheaper).

## Three easiest wins

1. French, Hugo on fr.wikisource -- all 10, validated, nearly Folger-shaped.
2. Italian, Rusconi on it.wikisource -- all 10 at SAL 100%, same-line labels.
3. Spanish, Menendez y Pelayo Macbeth + Romeo, plus Moratin's Hamlet -- three
   clean Gutenberg texts; Marquez adds two more.
