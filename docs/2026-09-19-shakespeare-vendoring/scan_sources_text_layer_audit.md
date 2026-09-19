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

## CORRECTION, measured later the same day: "extraction, not transcription" was too optimistic

The table above is right that the text is machine-readable. The inference drawn
from it -- that these cells therefore need extraction rather than transcription
-- does not follow, and a second probe shows why.

Fed through the EXISTING pipeline (strip the running header, hand the lines to
`extract` exactly as an HTML page is handed to it), `pt/macbeth 1.3` returns the
**correct scene span**: 6,853 characters, starting at `SCENA III`, ending where
it should. The boundary logic works on a PDF with no changes at all.

**The labels do not.** It found five "speakers", all of them fragments of
dialogue: `Parte`, `vosso sangue estancou-se`, `Segundo parece, os camaristas`.
Because:

```
ACTO III SCENA IV 95 MACBETH Onde ? LENNOX Aqui ,meu bom senhor; o que é que
```

**The text layer flattens the page.** A speaker name arrives INLINE, with no
colon, no period and no line break -- and every rule in `mark_speakers` and
`_LABEL_SHAPES` keys on structure that PDF text extraction discards. The name is
visually distinct on the page (its own line, centred, small caps) and none of
that survives `extract_text()`.

So segmenting these needs the CAST KNOWN IN ADVANCE, which is what the first
prototype quietly did with a hardcoded name list, and is why a speaker missing
from that list silently glued its line to the previous speaker's.

**That makes it a design question, not a wiring job.** Where does the name list
come from? The English sidecar roster is the obvious candidate and is already
loaded -- but it carries `FIRST WITCH` where the Portuguese prints
`1.ª FEITICEIRA`, and proper nouns match across languages while function names
do not. A wrong or partial list does not fail loudly; it moves a speech into
another character's mouth, which is the one failure this corpus cannot afford.

**Budget it as a design round, not as an afternoon.** The boundary half is free
and already works; the label half is the whole job.

## A prototype extractor was built and is NOT good enough to ship

Written against `pt/macbeth 1.3` (Domingos Ramos, 1912) to find out what the
lane actually costs. Scene three runs PDF pages 33 to 45, scene four starts on
46, and the speakers are all-caps names plus ordinal witches. It produced 97
lines and 8,101 characters with most speeches correctly attributed — and four
defects that make it unshippable as written, all of which the next attempt
should expect:

1. **It caught the tail of the previous scene.** Page 33 carries the end of
   scene two, so Duncan and Ross appear before the scene heading. The page range
   has to start at the heading inside the page, not at the page.
2. **A speaker missing from the name list becomes body text.** `Ross` was not in
   the list, so his line was glued to Macbeth's. A fixed name list is a
   per-play list, and omitting one silently moves a speech.
3. **It invented two speakers** — `ainda` and `Perdoae -me` — from running text.
   Same class as the Italian over-claiming.
4. **The spacing artifacts survive.** `per- dido`, `Sáem .`, `tensestado`. This
   is the half that matters for fidelity, because a mangled word is not the
   translator's word and a voice will read it aloud.

So the text-layer lane is cheaper than transcription and is still a
**per-edition extractor**, exactly like the Aozora, tiInherit and Zhu rules.
Budget it that way. The de-spacing needs a model pass, which the 2026-09-19
ruling permits and which is far safer than OCR, because the output can be
checked character-by-character against the source layer.

## Why this was missed

`url_is: page_scan` describes how the SOURCE IS PUBLISHED, not whether its text
is machine-readable. A digitised volume on Commons is a scan and may still carry
a full text layer from the archive that produced it. The two facts were being
read as one, and the field name invites it.

**The check is one download and four `extract_text()` calls.** Run it before
planning transcription work, not after.
