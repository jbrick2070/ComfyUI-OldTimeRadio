# Diagnosis: Carlo Rusconi's Italian Shakespeare Markup (King Lear 1.1)

Act 1 Scene 1 extracts, but its speaker labels are polluted (90 speeches, 14 speakers, including stage directions like `ESCONO GLOC. ED EDM` and `A CORD`). Root cause: the existing Rusconi speaker rule lacks a paragraph anchor (`<p>`), capturing italic directions and dialogue mid-sentence; and `_ITALIC_PARENTHETICAL` fails to strip Rusconi's stage directions due to punctuation placement and nested tags.

## 1. Markup Vocabulary

- **Act and Scene Headings**: Centered `<div>` blocks styled with explicit font sizes:
  - Act: `<div class="centertext ct"...><p><span style="font-size: 150%...">ATTO PRIMO</span></p></div>`
  - Scene: `<div class="centertext ct"...><p><span style="font-size: 120%...">SCENA I.</span></p></div>`
- **Stage Directions**:
  - Setting: `<div class="centertext ct"...><p><span style="font-size: 90%...">La sala del Consiglio...</span></p></div>`
  - Entrances: Centered blocks containing `<i>Entrano</i>` and small-caps character spans:
    `<div class="centertext ct"...><p><span ...><i>Entrano</i> <span style="font-variant:small-caps">Kent, Glocester</span> <i>e</i> <span style="font-variant:small-caps">Edmondo</span>.</span></p></div>`
  - Inline Business / Exits: Italic parentheticals inline or ending speeches: `(<i>a parte</i>)`, `(<i>a Cord</i>.)`, `(<i>escono Gloc. ed Edm</i>.).`, `(<i>escono Franc. e Cord</i>.)`.
- **Speaker Labels**: Always open a paragraph (`<p>`) with italic abbreviated name and trailing period:
  `<p><i>Kent</i>. Avrei sempre creduto...`
  `<p><i>Gloc</i>. Questo pure...`
  `<p><i>Alb. e Corn</i>. Amato sire...` (joint label)
  `<p>1<sup>a</sup> <i>Strega</i>.` (Macbeth ordinal superscript)

## 2. The Two Counts

- **Edition Source Count**: **84 speeches** explicitly labeled by the edition in scene 1.1 across 10 distinct speakers (`Lear`: 24, `Kent`: 13, `Cord`: 12, `Gloc`: 8, `Gon`: 7, `Reg`: 6, `Borg`: 5, `Franc`: 5, `Edm`: 3, `Alb. e Corn`: 1).
- **Current Pipeline Count**: `V.mark_speakers` produces **91 marked labels** (84 real speeches + 2 small-caps entrance names + 5 italic directions). Downstream parsing yields **90 speeches** across **14 speakers**, adding `EDMONDO`, `ESCONO GLOC. ED EDM`, `A CORD` (3 speeches), and `ESCONO FRANC. E CORD`.

## 3. Proposed Rules

Modeled on `_AOZORA_BUSINESS` / `_AOZORA_SPEAKER`:

```python
_RUSCONI_BUSINESS = re.compile(
    r'(?is)<div[^>]*class="[^"]*centertext[^"]*"[^>]*>'
    r'(?P<body>(?:(?!<span style="font-size:\s*1[25]0%).)*?<i>Entran[oi]\b.*?)</div\s*>')

_RUSCONI_PAREN_DIR = re.compile(
    r'(?is)(?:\(\s*<i>|<i>\s*\()(?P<body>(?:(?!</?[pP]\b).)*?)(?:</i>\s*\)|\)\s*</i>)[\s.;]*')

_RUSCONI_SPEAKER = re.compile(
    r'(?is)(?P<prefix><p\b[^>]*>\s*)'
    r'(?P<ord>\d\s*<sup>\s*[ao]\s*</sup>\s*)?'
    r'<i>(?P<name>[^<]{1,40})</i>'
    r'\s*(?:\((?:<[^>]+>|[^()<])*\))?\s*\.')
```

In `mark_speakers(markup)`:
```python
body = _RUSCONI_BUSINESS.sub("\n", markup)
body = _RUSCONI_PAREN_DIR.sub(" ", body)

def _mark_rusconi_speaker(mm):
    ord_val = re.sub(r"(?s)<[^>]+>|\s+", "", mm.group("ord") or "")
    name = mm.group("name").strip()
    full = ("%s %s" % (ord_val, name)).strip()
    return "%s\n%s%s%s " % (mm.group("prefix"), SPEAKER_MARK, full, SPEAKER_MARK)

body = _RUSCONI_SPEAKER.sub(_mark_rusconi_speaker, body)
```

## 4. What Would Trip It

1. **Unanchored Italic Names**: `<i>...</i>.` captures Latin dialogue (`<i>Cucullus non facit monachum</i>.`), letters (`<i>M. O. A. I.</i>`), and parenthetical exits mid-sentence.
2. **Small-Caps Entrance Spans**: Rusconi sets entrance names in small-caps. Unstripped, `_SPEAKER_SPANS[1]` (Márquez rule) captures them as speakers.
3. **Punctuation Placement**: Periods sit inside or outside italics and parentheses: `(<i>escono...</i>.).` vs `(<i>a Cord</i>.)`. Rigid `\(\s*<i>[^<]+</i>\s*\)` misses them.
4. **Nested Tags in Directions**: Fanfares embed small-caps inside `(<i>...</i>)`, breaking `[^<]+` patterns.
5. **Headings in `div.centertext ct`**: Headings share containers with entrances. Lookahead for `font-size: 1[25]0%` preserves heading anchors.
