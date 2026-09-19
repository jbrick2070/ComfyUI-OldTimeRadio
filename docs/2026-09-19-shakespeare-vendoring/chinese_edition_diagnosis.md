# Diagnosis: Zhu Shenghao's Chinese Shakespeare Markup (Midsummer 3.1)

The recorded hold note claiming *"the edition breaks no paragraph before a speaker, so five speeches land in the wrong mouth mid-line"* is **inaccurate**. The source edition never merges speakers mid-line. Every speaker is preceded by a paragraph break (`<p>`), a block boundary (`</p>\n`), or a page break. The observed merges were entirely artifacts of the pipeline lacking markup rules for this edition and falling back to heuristic plain-text parsing.

## 1. Markup Vocabulary

- **Act and Scene Headings**: Enclosed in `<div class="center">` with explicit font sizing:
  - Act: `<div class="center">\n<span style="font-size: 144%;"><span style="letter-spacing: 1em;">第三幕</span></span></div>`
  - Scene: `<div class="center">\n<span style="font-size: 120%;">第一场　林中；蒂妲妮霞熟睡未醒</span></div>`
- **Stage Directions**:
  - Block Entrances: Centered `<div>` blocks opening with an unclosed black bracket `【` (U+3010) and closed by `</div>`:
    `<div class="center">\n【衮斯，史纳格，波顿，弗鲁脱，斯诺脱，司他巫林上。</div>`
  - Inline Business / Exits: Fullwidth parentheses inline with spoken text: `（下）`, `（众下）`, `（醒）`, `（唱）`, `（同下。）`.
- **Speaker Labels**: 1–4 CJK characters followed immediately by an ideographic space `　` (`\u3000`):
  - Standard prose dialogue: `<p>波　咱们都会齐了吗？\n</p>`
  - Across Wikisource page breaks: `<p><span><span class="pagenum ws-pagenum" ...><span ...>&#8203;</span></span></span>衮　有的，那晚上有好月亮。\n</p>`
  - Before verse blocks (`<div class="ws-poem">`): Either a standalone paragraph `<p>迫　\n</p>` or bare at a block boundary `</p>\n波　<style ...>`.

## 2. The Two Counts

- **Edition Source Count**: **65 speeches** explicitly marked by the edition in scene 3.1 across 14 characters (`波`: 22, `衮`: 13, `蒂`: 6, `斯`: 4, `弗`: 3, `迫`: 3, `史`: 2, `司`: 2, `豆`: 2, `蛛`: 3, `芥`: 2, `飞`: 1, `四仙合`: 1, `四仙`: 1).
- **Current Pipeline Count**: `V.mark_speakers` produces **0** marked speakers. Downstream heuristic parsing (`to_text` + `extract` + `normalise_labels`) produces **56 speeches**, losing 9 speeches to merges.

## 3. Proposed Rules

Modeled on `_AOZORA_BUSINESS` / `_AOZORA_SPEAKER`:

```python
_ZH_BUSINESS = re.compile(
    r'(?is)<div class="center">\s*【.*?</div\s*>')

_ZH_SPEAKER = re.compile(
    r'(?is)(?P<prefix><p\b[^>]*>(?:<span\b[^>]*>.*?</span>\s*)*|</p>\s*)'
    r'(?P<who>[\u4e00-\u9fff]{1,4})\u3000(?!\u3000)')
```

In `mark_speakers(markup)`:
```python
body = _ZH_BUSINESS.sub("\n", body)

def _mark_zh_speaker(mm):
    who = mm.group("who")
    if re.match(r"^第.*[幕场場]$", who):
        return mm.group(0)
    return "%s\n%s%s%s " % (mm.group("prefix"), SPEAKER_MARK, who, SPEAKER_MARK)

body = _ZH_SPEAKER.sub(_mark_zh_speaker, body)
```

## 4. What Would Trip It

1. **`_marked_name` Length Guard**: `otr_vendor_shakespeare.py` contains `if len(name) < 2: return ""`. This rejects all single-character Chinese speaker names (`波`, `衮`, `蒂`, etc.). It must be adjusted to allow 1-character CJK names.
2. **Headings Sharing `\u3000`**: Scene headings (`第一场　`) match the name + `\u3000` shape. The negative guard `^第.*[幕场場]$` prevents corrupting scene boundary anchors needed by `extract()`.
3. **Wikisource Page-Break Tags**: MediaWiki inserts `<span class="pagenum">...&#8203;</span>` (`\u200b` zero-width space) immediately after `<p>`, which would trip any pattern expecting the name at byte 0 of `<p>`.
4. **Verse Blocks**: Speeches before `<div class="ws-poem">` leave `\u3000` at line end, which `extract()`'s `rstrip()` strips as whitespace, reducing `波　` to bare `波`. Marking the speaker in raw HTML before tag stripping prevents this.
5. **Fullwidth Dialogue Colons**: In prose, `波　让什么人扮做墙头：` triggers greedy colon-matching in `_LABEL_SHAPES[0]`. Tagging at HTML stage bypasses heuristic plain-text shapes entirely.
