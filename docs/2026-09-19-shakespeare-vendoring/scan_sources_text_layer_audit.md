# The "page scan" cells: 12 of 31 need no OCR at all

Measured 2026-09-19 by downloading every scan source and sampling four pages
with `pypdf`. The cells were all filed `url_is: page_scan` and treated as one
undifferentiated block of OCR work. They are two different jobs.

## Result

| Language | Cells | Sources | Verdict |
|---|---|---|---|
| Spanish | 8 | 3 | **TEXT LAYER** — extract, do not transcribe |
| Portuguese | 4 | 4 | **TEXT LAYER** — extract, do not transcribe |
| Japanese | 9 | 6 | IMAGE ONLY — genuine OCR |
| Chinese | 10 | 3 + an Index page | IMAGE ONLY — genuine OCR |

## The sources that carry text

| Source | Pages | Cells |
|---|---|---|
| `Obras dramáticas de Guillermo Shakespeare - Tomo I (1897)` | 472 | es/king_lear 1.1, es/midsummer 3.1, es/midsummer 3.2 |
| `La tempestad - La noche de Reyes (Jaime Clark)` | 204 | es/tempest 3.1, es/twelfth_night 1.5, es/twelfth_night 2.5 |
| `Otelo - Mucho ruido para nada (Jaime Clark)` | 294 | es/much_ado 2.3, es/much_ado 3.1 |
| `Rei Lear (trad. Domingos Ramos, 1919)` | 240 | pt/king_lear 1.1 |
| `Macbeth (trad. Domingos Ramos, 1912)` | 240 | pt/macbeth 1.3 |
| `A Tempestade (trad. Domingos Ramos, 1914)` | 216 | pt/tempest 1.2, pt/tempest 3.1 |

The Jaime Clark volume is the one a model transcribed by hand for
`es/tempest 1.2` on the same day. That scene was vendored from a hand
transcription of a PDF **that could have been read directly**, which is the
cost of not running this check first.

## What the text layer looks like

**Portuguese is clean and keeps its speaker names on their own tokens:**

```
ACTO III SCENA IV 95 MACBETH Onde ? LENNOX Aqui ,meu bom senhor; o que é que perturba vo
```

**Spanish is dirtier — word boundaries are lost inside a line:**

```
31 Gon. ¡Quefrescayquélozanacreceestayerba!iquéverde!ANT. Enefecto,elsueloespardusco.SEB.
```

So the Spanish volumes need a de-spacing pass and the Portuguese largely do not.
Both are a different and far cheaper job than transcription, and both are
**verifiable**: the extracted characters come from the document rather than from
a model's reading of an image, so a wrong word is a bug rather than a
hallucination.

## Confirmed image-only, so genuinely OCR

Every Japanese volume in the `沙翁傑作集` series sampled zero characters across
four pages, as did the Chinese `第2輯`. That matches what an outside pass
reported independently. Those 19 cells are the real OCR work and the
2026-09-19 ruling is what governs them.

## Why this was missed

`url_is: page_scan` describes how the SOURCE IS PUBLISHED, not whether its text
is machine-readable. A digitised volume on Commons is a scan and may still carry
a full text layer from the archive that produced it. The two facts were being
read as one, and the field name invites it.

**The check is one download and four `extract_text()` calls.** Run it before
planning transcription work, not after.
