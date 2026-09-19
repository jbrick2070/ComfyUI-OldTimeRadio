### Block 1: `leads.json` updates

```json
[
  {
    "id": "fr_twelfth_night_2_5",
    "language": "fr",
    "play": "twelfth_night",
    "act": 2,
    "scene": 5,
    "url": "https://fr.wikisource.org/wiki/Le_Soir_des_Rois/Texte_entier",
    "notes": "Guizot translation. Note: Uses <span class=\"personnage\"> for speakers instead of class=\"sc\", dropping ~60 dialogue lines with current pipeline heuristics."
  },
  {
    "id": "es_macbeth_1_3",
    "language": "es",
    "play": "macbeth",
    "act": 1,
    "scene": 3,
    "url": "https://es.wikisource.org/wiki/Macbeth:_Acto_I",
    "notes": "Macpherson translation. Note: Witches are labeled 'BRUJA 1.ª' (no trailing dot), missing _LABEL_SHAPES require trailing dot."
  },
  {
    "id": "zh_tempest_1_2",
    "language": "zh",
    "play": "tempest",
    "act": 1,
    "scene": 2,
    "url": "https://zh.wikisource.org/wiki/%E6%9A%B4%E9%A2%A8%E9%9B%A8_(%E8%8E%8E%E5%A3%AB%E6%AF%94%E4%BA%9E)",
    "notes": "Zhu Shenghao translation. URL percent-encoded to prevent urllib crash. Note: find_label matches the TOC ' 第一幕' / ' 第二場' early, truncating extraction before the real scene text."
  },
  {
    "id": "ja_romeo_juliet_2_2",
    "language": "ja",
    "play": "romeo_juliet",
    "act": 2,
    "scene": 2,
    "url": "https://www.aozora.gr.jp/cards/000074/files/42171_26723.html",
    "notes": "Tsubouchi translation. Note: Served as Shift_JIS with no HTTP charset header; UTF-8 default corrupts. Scene headings use ruby tags (e.g. <ruby><rb>第</rb>...) failing string matches."
  },
  {
    "id": "it_hamlet_1_4",
    "language": "it",
    "play": "hamlet",
    "act": 1,
    "scene": 4,
    "url": "https://it.wikisource.org/wiki/Amleto_(Shakespeare_-_Rusconi)/Atto_I",
    "notes": "Rusconi translation."
  }
]
```

### Block 2: `EDITION_LABELS` updates

```python
EDITION_LABELS = {
    # ...
    "fr_twelfth_night_2_5": {
        "start": "SCÈNE V.",
        "end": "SCÈNE VI."
    },
    "es_macbeth_1_3": {
        "start": "ESCENA III.",
        "end": "ESCENA IV."
    },
    "zh_tempest_1_2": {
        "start": "第二場",
        "end": "第二幕"
    },
    "ja_romeo_juliet_2_2": {
        "start": "第二場",
        "end": "第三場"
    },
    "it_hamlet_1_4": {
        "start": "SCENA IV.",
        "end": "SCENA V."
    }
}
```

### Notes (Grounded QA Probe Findings)

#### 5 Critical Pipeline Hazards Discovered

1. **URL Encoding Crash:** Python's `urllib.request` crashes on non-ASCII characters (`é`, `’`, CJK) in URLs (like the Chinese Tempest URL). They must be percent-encoded (`%C3%A9`) in `leads.json`.
2. **French Twelfth Night (`fr_twelfth_night_2_5`):** The Wikisource page uses `<span class="personnage">` for speakers. The pipeline's `_SPEAKER_SPANS` only targets `class="sc"` (which is used for stage entrances on this specific page). This causes ~60 dialogue lines to be dropped.
3. **Spanish Macbeth (`es_macbeth_1_3`):** Witches are labeled `BRUJA 1.ª` (no trailing period). `_LABEL_SHAPES` requires a trailing dot, so the witches are missed.
4. **Chinese Tempest (`zh_tempest_1_2`):** `find_label` matches the MediaWiki Table of Contents (TOC) (` 第一幕`, ` 第二場`) at the top of the page, causing extraction to truncate at 37 characters instead of reaching the real scene text at line 216.
5. **Japanese Aozora (`ja_romeo_juliet_2_2`):** Served as `Shift_JIS` with no HTTP charset header; defaulting to UTF-8 corrupts it. Scene headings use ruby tags (`<ruby><rb>第</rb><rp>（</rp><rt>だい</rt><rp>）</rp></ruby>二<ruby><rb>場</rb><rp>（</rp><rt>ぢゃう</rt><rp>）</rp></ruby>`), which naively tag-strip to `第（だい）二場（ぢゃう）`, failing exact string matches for `第二場`.

#### Italian Anomalies Confirmed
- Maffei's *Tempest* Act I completely skips SCENA II.
- Rusconi's *Twelfth Night* Act II skips SCENA IV, making SCENA VI exactly match Folger 2.5.

#### Other Languages
- Chinese, Japanese, and Hindi Wikisource lack transcribed text layers for the remaining requested plays.
