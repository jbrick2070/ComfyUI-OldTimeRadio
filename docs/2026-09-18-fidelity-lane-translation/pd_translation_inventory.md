# Public-domain Shakespeare translations -- inventory (2026-09-18)

Research agent, web-verified where marked. Languages from
`config/episode_languages.json` (es, pt, it, fr, hi, ja, zh). Plays from
`config/source_banks/shakespeare/curated_scenes.sample.json`: 14 scenes across
Macbeth, Tempest, Midsummer, Twelfth Night, Much Ado, As You Like It, Hamlet,
King Lear, Romeo and Juliet, Comedy of Errors (Folger CC BY-NC text).

PD line: translator died before 1954 (life+70); Spain is life+80, so
"worldwide" below means died in 1944 or earlier (1944 + 80 = 2024). Every
translator marked PD clears both; Zhu Shenghao (d. 1944) is the edge case
and clears.

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
| ja | Romeo | Tsubouchi Shoyo (1935) | Yes | aozora.gr.jp card 42773 (`files/42773_ruby_38390.zip`) | `NAME<full-width space>speech`, same line; old kana / old kanji | Only Aozora Shakespeare published. |
| ja | Hamlet, Lear, Macbeth, Tempest, Midsummer, Much Ado, As You Like It | Tsubouchi | Yes | aozora person 264 | In progress, not downloadable | Twelfth Night and Comedy of Errors not queued. Gutenberg: none. |
| zh | Hamlet, Tempest, Midsummer | Zhu Shenghao (1944) | Yes | zh.wikisource (traditional; act subpages) | Hamlet: one-character abbreviated speaker + full-width space; Tempest: full-width colon | Inconsistent label style. |
| zh | the other seven | Zhu | Yes | zh.wikisource collection page | Red links, not transcribed | Gutenberg: none. |
| hi | As You Like It (1917) | Lala Sitaram (1937) | Yes | archive.org `in.ernet.dli.2015.263809` (`_djvu.txt`) | `NAME--speech` same line; OCR moderate-to-poor; names Indianised | Only PD Hindi hit. Sitaram's Hamlet/Macbeth exist in print, no scan found. |
| hi | everything else | Bachchan (2003), Rangeya Raghav (1962) | NO | archive.org, not PD | -- | hi.wikisource has no Shakespeare author page. |

## Coverage gaps

- Hindi: one play, one poor scan.
- Japanese: one play, old-kana text (a `misaki[ja]` G2P risk); the rest queued.
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
