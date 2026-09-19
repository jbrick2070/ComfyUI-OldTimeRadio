# Job: find direct URLs for 40 Shakespeare translation scenes

I am building a corpus of public-domain Shakespeare translations. For 40
scenes I have only a **landing page or a whole-work page** — not the address
of the scene's actual transcribed text. Your job is to find that address, or
prove it does not exist.

**This is a lookup job with a verifiable answer. It is not a transcription
job and not a translation job. Do not transcribe, summarise or translate
anything.**

## The targets

The list is in `targets.json` next to this file (also `targets.md` as a
readable table). 40 rows, each with:

```json
{ "id": "it/hamlet/1.1", "iso": "it", "play": "hamlet", "scene": "1.1",
  "translator": "Carlo Rusconi", "known_url": "<landing page we already have>" }
```

Languages: Italian 12, Chinese 13, Hindi 9, Japanese 4, Spanish 2.

## What counts as a find

A URL that serves the **transcribed text of that specific scene**, in that
translator's version, as selectable text — not page images.

Good: a Wikisource scene page, an act page containing the scene, a single
page carrying the whole play as text, Aozora Bunko, Project Gutenberg,
zh.wikisource, an institutional transcription.

**Not a find:** a page-image scan with no text layer, a library catalogue
record, a bookseller listing, a modern copyrighted translation, a summary, a
different translator's version, or a PDF of photographs.

## Rules that decide whether a find is usable

1. **Same translator.** The row names one. A different translator's text is
   a different work and is worse than nothing to me, because it looks right.
   If you find another public-domain translator's transcription of the same
   scene, record it in `alternate_translator_found` — do NOT put it in `url`.
2. **Public domain.** The row gives `translator_death_date` where known.
   If you find a transcription of a translator who died after 1960, flag it
   and do not record it as the answer.
3. **Text, not pictures.** State how you verified: did you actually see the
   scene's words as text?
4. **Say the scene is there.** Quote the first 15–25 characters of the
   scene's first spoken line from the page as evidence. That single field is
   what makes your answer checkable; an answer without it is unusable to me.

## Output — write a file

Write `results.json` in this same folder (`tmp/url_hunt/`). If you cannot
write files, output the same JSON in one block and I will save it.

```json
[
  {
    "id": "it/hamlet/1.1",
    "status": "found | not_found | wrong_translator | images_only | uncertain",
    "url": "<direct URL to the scene text, or null>",
    "url_serves": "scene | act | whole_work",
    "first_line_excerpt": "<15-25 chars of the scene's first spoken line, verbatim from that page>",
    "scene_heading_on_page": "<how the page labels the act/scene, verbatim>",
    "translator_confirmed": "<how you confirmed it is this translator>",
    "notes": "<anything a later reader needs>"
  }
]
```

**Save your progress as you go** — rewrite `results.json` after every few
rows rather than holding everything to the end. A partial file is useful; a
lost session is not.

## Order to work in

1. **Italian (12)** — Rusconi. The landing page we hold is believed to list
   the whole play; the act/scene pages should be one hop away. Likely the
   fastest wins in the set.
2. **Chinese (13)** — Zhu Shenghao (朱生豪). Very widely transcribed; check
   zh.wikisource.org first.
3. **Japanese (4)** — Tsubouchi Shoyo (坪内逍遙). Aozora Bunko is the likely
   home; note that Aozora sets one work per page, so the answer may be a
   whole-work page and that is fine (`url_serves: "whole_work"`).
4. **Spanish (2)**, then **Hindi (9)**. Hindi is the hardest — a previous
   pass found translators exist but nothing transcribed. If that holds,
   `not_found` for all nine is a perfectly good and useful answer.

## What I actually want from you

Honest `not_found`s. I am deciding where to spend expensive OCR effort, so a
confident wrong URL costs me far more than an admission that the text is not
online. If you are unsure whether a page is the right translator, say
`uncertain` and explain — do not round up to `found`.

Work through as many as you can. Stop when you run out of targets or budget,
and leave `results.json` valid either way.
