# Deep-research prompt: public-domain Shakespeare translations (paste into Gemini)

I am building an open-source radio-drama generator that performs Shakespeare
scenes VERBATIM from public-domain text. It already has the English (Folger)
scenes. I need the same scenes in seven other languages, performed from real
public-domain translations rather than machine translation. Please research
and return a sourced inventory.

## The scenes I need (14 scenes, 10 plays)

- Macbeth, Act 1 Scene 3
- The Tempest, Act 1 Scene 2 and Act 3 Scene 1
- A Midsummer Night's Dream, Act 3 Scene 1 and Act 3 Scene 2
- Twelfth Night, Act 1 Scene 5 and Act 2 Scene 5
- Much Ado About Nothing, Act 2 Scene 3 and Act 3 Scene 1
- As You Like It, Act 3 Scene 2
- Hamlet, Act 1 Scene 1
- King Lear, Act 1 Scene 1
- Romeo and Juliet, Act 2 Scene 2
- The Comedy of Errors, Act 3 Scene 1

## The languages

Spanish, Portuguese, Italian, French, Hindi, Japanese, Mandarin Chinese.

## What already exists (do not re-research; extend it)

- French: Francois-Victor Hugo (d. 1873) complete on fr.wikisource; Guizot
  (d. 1874) on Project Gutenberg. Considered solved.
- Italian: Carlo Rusconi (d. 1889) complete on it.wikisource. Considered
  solved.
- Spanish: Menendez y Pelayo (Macbeth, Romeo), Marquez (As You Like It,
  Comedy of Errors), Moratin (Hamlet) found as plain text. MISSING as plain
  text: The Tempest, Midsummer, Twelfth Night, Much Ado, King Lear.
- Portuguese: only Luis I's Hamlet (Gutenberg 25667) and Castilho's Midsummer
  (archive.org OCR). MISSING: the other eight plays.
- Mandarin: Zhu Shenghao (d. 1944) Hamlet, Tempest, Midsummer transcribed on
  zh.wikisource; the other seven plays are untranscribed red links.
- Japanese: Tsubouchi Shoyo (d. 1935) Romeo on Aozora Bunko; the rest of his
  translations are "in progress" on Aozora and not downloadable.
- Hindi: only Lala Sitaram's As You Like It (1917) as a poor OCR scan.

## Public-domain rule to apply

The translator must have died before 1944 (this clears life+70 everywhere
and life+80 in Spain). Give the translator's death year for every entry.
Flag anything where the death year is uncertain. Exclude anything by a
translator who died 1944 or later even if a site hosts it.

## What I need for EVERY (language, play) pair

1. Translator, first-publication year, translator's death year, and the PD
   status by the rule above.
2. A URL to PLAIN TEXT (UTF-8 .txt, HTML, or a Wikisource page), not a PDF
   scan, unless a scan is the only thing that exists (then say so and give
   the OCR quality if visible).
3. The speaker-label FORMAT the text uses, quoted from the actual text: for
   example `MACBETH.` on its own line, `Macbeth.--` on the same line as the
   speech, a full-width colon, a full-width space, etc. I need to write a
   parser per source, so this matters.
4. Whether the translation is verse or prose, and whether it uses archaic
   orthography (old kana / old kanji for Japanese, traditional vs simplified
   for Chinese, pre-reform spelling for Portuguese).
5. Whether the scene I need is actually present and complete (some editions
   abridge).

## Where to look (and please go beyond these)

Project Gutenberg (all languages), Wikisource in each language, Aozora Bunko
(Japanese), zh.wikisource and Chinese Text Project, archive.org / Digital
Library of India (Hindi), Biblioteca Virtual Miguel de Cervantes (Spanish),
Biblioteca Nacional de Portugal / Brasiliana (Portuguese), HathiTrust
full-view, Internet Archive full text. For Hindi and Japanese, also search in
the script itself (e.g. the play titles in Devanagari and Japanese) and check
university digital collections.

## Output format

A table with columns: language | play | translator (born-died) | first
published | PD by rule (yes/no/uncertain) | plain-text URL | speaker-label
format (quoted) | verse/prose | orthography notes | scene present?

Then, per language: the single best source for the most plays, and the
plays for which NO public-domain translation exists in any form (so I know
the gap is real and not a search failure).

Verify every URL resolves. Do not invent URLs. If a claim is unverified, say
so in the row.
