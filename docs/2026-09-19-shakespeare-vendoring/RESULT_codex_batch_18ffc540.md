Reviewed `18ffc5404f0f1fadd203e66a19adbcebf87a3e9e` against `cb5c527a258fcab951dc98dad9fe1acc20dc0349` on the real Windows filesystem. The predecessor module was loaded from `git show 18ffc540~1:scripts/otr_vendor_scan.py` into `%TEMP%`. HEAD advanced during this review; the scan module matched the reviewed commit byte-for-byte when the measurements were taken and at the subsequent digest check. Final verification then detected another window's uncommitted rewrite of `window_truncates()` to a suffix comparison. That later edit is outside this review: all findings and code references below concern commit `18ffc540`, not the subsequently changing working tree. Python 3.12.11, PyMuPDF 1.28.2, `PYTHONUTF8=1`, bytecode writes disabled, CUDA masked. This review made no git writes or repository edits other than this report.

**1. HOLDS for the measured loose-peel blast radius.**

The cache does not contain five PDFs. It contains seven actual PDFs, one Wikimedia HTML page saved with a `.pdf` suffix, and one failed-download text file. Every cache entry was attempted. Both reading orders were measured for every readable entry; the HTML entry was also measured rather than silently excluded.

The census used the same transformations as `pdf_text()` and `strip_running_titles()`: `_LINE_BREAK_HYPHEN` at `scripts/otr_vendor_scan.py:312`, then `_SPLIT_HEADING` at line 1065, then `_page_edges()` at line 1075. For each returned edge index, both modules received the identical row. Overlapping head/foot indices were counted once per page; neither edge was omitted.

| Cache filename | Pages | Flat edge rows | Coordinate edge rows | Changed identities, flat / coordinates |
|---|---:|---:|---:|---:|
| `0fe883861a983e5a818ad82b.pdf` | 204 | 1,070 | 1,070 | 0 / 0 |
| `391af551b32d458ef1ce8888.pdf` | 216 | 1,190 | 1,181 | 0 / 4 |
| `62fa124c89695cb646178c1b.pdf` | 472 | 2,760 | 2,760 | 0 / 1 |
| `67e70beaf462724dada711e6.pdf` | 240 | 1,335 | 1,333 | 0 / 0 |
| `76b96c683e46026bc9ef9906.pdf` | 248 | 1,345 | 1,344 | 0 / 0 |
| `8802fae93d310a78061bb06c.pdf` | 240 | 1,352 | 1,351 | 0 / 0 |
| `d19dcf407d262aef5a19e763.pdf` | 294 | 1,645 | 1,644 | 0 / 0 |
| `279c2210d96e4d57ae2b035e.pdf` (HTML) | 18 reflowed pages | 108 | 108 | 0 / 0 |

That is 1,914 actual PDF pages, 21,380 PDF edge rows across both orders, and 21,596 edge rows including the HTML. `7361e51246c2cac49f6b58e2.pdf` is a 136-byte `File not found: /v1/AUTH_mw/...` response, not a PDF; PyMuPDF raises `FileDataError`. It supplies no pages to inspect.

Every changed identity follows. Page indexes are zero-based, as in `--pages`; each changed row is line 1 under coordinate order. Both versions return `folio_attached=True` for all five.

| PDF | Page index (viewer page) | Exact row | Previous identity | Patched identity | Classification |
|---|---:|---|---|---|---|
| `391af551b32d458ef1ce8888.pdf` | 24 (25) | `CENA I A TEMPESTADE 7` | `CENA I A TEMPESTADE` | `TEMPESTADE` | Furniture: scene/title/folio header; the next row is the separate `GONZALO` cue. |
| `391af551b32d458ef1ce8888.pdf` | 48 (49) | `SOENA II A TEMPESTADE 31` | `SOENA II A TEMPESTADE` | `TEMPESTADE` | Furniture: scene/title/folio header; the next row is the separate `ÁRIEL` cue. |
| `391af551b32d458ef1ce8888.pdf` | 96 (97) | `SOENA II TEMPESTADE 79` | `SOENA II TEMPESTADE` | `TEMPESTADE` | Furniture: scene/title/folio header, followed by continuing dialogue beginning `acesa, para engolfar-me nas trevas`. |
| `391af551b32d458ef1ce8888.pdf` | 158 (159) | `SOENA I A TEMPESTADE 141` | `SOENA I A TEMPESTADE` | `TEMPESTADE` | Furniture: scene/title/folio header, followed by continuing dialogue beginning `vestido em paga`. |
| `62fa124c89695cb646178c1b.pdf` | 257 (258) | `16 DRAMAS DE SHAKE PEARL .` | `DRAMAS DE SHAKE PEARL` | `DRAMAS DE SHAKE` | Furniture: collection-title/folio header, followed by continuing dialogue beginning `mildad. Ven á verme`. |

The fifth row exposes over-peeling inside a word: `PEARL` can be parsed as `PEAR` plus Roman `L` because line 693 permits zero spaces. It remains a furniture row, so it does not meet the question's speaker/dialogue refutation criterion. A separate comparison of complete `strip_running_titles()` outputs found exactly four changed output rows: the four Tempestade headers above become blank. The Macpherson row's output is unchanged. No changed speaker or dialogue row was found.

**2. HOLDS: reproduced byte identity.**

The direct reproduction used `main()` with its unmodified `pdf_text()` reader and these arguments, without `--write`:

```text
--iso pt --play tempest --scene 3.1 --stem tempest__act3_scene1
--reading-order coordinates --pages 110-117
--scene-label "SCENA I" --end-label "SCENA II" --fold FERNANDO=FERDINAND
```

The only observation wrapper captured the returned pairs from `speeches_from_span()` without changing them. They were serialized exactly as `scripts/otr_vendor_scan.py:1465` and line 1469 specify: `LABEL: speech`, LF between rows, final LF, UTF-8. `main()` returned 0; 25 speeches produced 5,332 bytes. The generated bytes equal the committed `pt/tempest_3_1.txt` bytes, its predecessor bytes, and the working file bytes. Generated and committed SHA-256 are both:

```text
ecf2c6f35b10f99d4e325cb702b7f08fcb111aca05be1464124b7e608963a1b1
```

Additional replays through `main()` used the complete-volume text collected in question 1 as a cache-only reader. Their generated bytes also equal the committed bytes:

| File | Speeches | Bytes | Generated = committed SHA-256 |
|---|---:|---:|---|
| `pt/macbeth_1_3.txt` | 49 | 8,163 | `21da1d5ee3898b9aafe87259ca9c44c17bb609264eaa0832707986aaf55ddbe0` |
| `pt/king_lear_1_1.txt` | 83 | 18,380 | `0a5cf6c35664e25ca8976eb6d25ebebf0ed2781239c3d0708ca72d52242f2829` |
| `pt/tempest_1_2.txt` | 138 | 26,266 | `1323d5f02b7e5ba7547c960e923925063beb8417219de85824093be6dc52bc9b` |

Macbeth used `--act-label "ACTO PRIMEIRO" --scene-label "SCENA III" --end-label "SCENA IV"`; Lear used `--act-label "ACTO PRIMEIRO" --scene-label "SCENA I" --end-label "SCENA II"`; both used coordinate order and no page window. Tempest 1.2 used coordinate order, pages `25-62`, scene `SCENA II`, end `ACTO SEGUNDO`, and `FERNANDO=FERDINAND`. The first two also equal the predecessor files. Tempest 1.2 matches its new manifest hash at `config/source_banks/shakespeare/translations/manifest.json:1637`, with ARIEL increasing to 25 speeches.

**3. REFUTED: `window_truncates()` falsely refuses a scene that a heading closed.**

`scripts/otr_vendor_scan.py:1168` searches for the last equal line anywhere in the window. It does not identify the occurrence that ended the extracted scene. This reproduced through the actual shared extractor:

```python
lines = [
    "SCENA I", "MIRANDA", "First answer.", "PROSPERO", "Sim.",
    "SCENA II", "MIRANDA", "Sim.",
]
body, reason = S.V.extract(lines, None, None, "SCENA I")
assert reason == ""
assert body == "SCENA I\nMIRANDA\nFirst answer.\nPROSPERO\nSim."
refusal = S.window_truncates(body, lines, False)
```

The scene closes at index 5, `SCENA II`, under `scripts/otr_vendor_shakespeare.py:1006`. The final body line occurs at indexes 4 and 7. The guard selects 7 and returns:

```text
the scene runs to the last line of the page window with no heading to close it -- the window's edge is not evidence the scene ended; widen --pages or pass --end-label
```

This is a false refusal. Passing `end_label="SCENA II"` to `extract()` returns the same body and the same false refusal, so the suggested `--end-label` does not cure it. `main()` unconditionally applies this guard at `scripts/otr_vendor_scan.py:1410` and returns 1 at line 1413. The guard needs the selected scene's boundary position or termination reason; equality with an unrelated later line cannot establish that boundary.

**4. REFUTED: folds can still redirect already-bound names through every remaining resolution path.**

`parse_folds()` only rejects membership in `{fold(name) for name in roster}` at `scripts/otr_vendor_scan.py:568` and line 579. `resolve()` consumes a declared fold at line 610, before the title-prefix, place, stem, and ordinal/function branches.

These are actual calls using `roster_for("king_lear__act1_scene1")` and `roster_for("macbeth__act1_scene3")`, not invented rosters. For each row I cleared `FOLDS`, called `resolve()`, called `parse_folds([declaration], roster)`, installed its returned map, and called `resolve()` again. Every declaration below returned an empty error string and a nonempty map.

| Bypassed branch | Declaration accepted | Resolution without fold | Resolution with fold | Branch location |
|---|---|---|---|---|
| Title-prefix exact `bare` | `REI LEAR=CORDELIA` | `LEAR` | `CORDELIA` | `scripts/otr_vendor_scan.py:615`, line 617 |
| Title-prefix place lookup | `DUQUE DE BORGONHA=LEAR` | `BURGUNDY` | `LEAR` | `scripts/otr_vendor_scan.py:619` |
| Direct `PLACE_NAMES` | `BORGONHA=LEAR` | `BURGUNDY` | `LEAR` | `scripts/otr_vendor_scan.py:621` |
| Stem match | `EDMUNDO=LEAR` | `EDMUND` | `LEAR` | `scripts/otr_vendor_scan.py:631`, line 637 |
| Ordinal plus function | `1.ª FEITICEIRA=MACBETH` | `FIRST WITCH` | `MACBETH` | `scripts/otr_vendor_scan.py:647`, line 650 |
| Function without ordinal | `TODAS TRES=MACBETH` | `ALL` | `MACBETH` | `scripts/otr_vendor_scan.py:648`, line 652 |

For example:

```python
roster = S.roster_for("king_lear__act1_scene1")
S.FOLDS.clear()
assert S.resolve("REI LEAR", roster) == "LEAR"
folds, why = S.parse_folds(["REI LEAR=CORDELIA"], roster)
assert (folds, why) == ({"REI LEAR": "CORDELIA"}, "")
S.FOLDS.update(folds)
assert S.resolve("REI LEAR", roster) == "CORDELIA"
S.FOLDS.clear()
```

The direct exact-name branch at line 599 is protected. The broader claim that folds only supply forms the resolver cannot reach is false. Preventing that redirection requires checking the existing resolution without declared folds, including these downstream branches.

**5. HOLDS for the complete song-cue condition on the measured corpus; the regex alone does match dialogue.**

Scanned all 43 vendored scene files, totaling 3,431 lines, and all seven actual PDFs in both orders: 64,270 raw flat lines plus 50,896 raw coordinate lines. Total: 118,597 line observations. Also scanned the HTML cache entry's 1,095 lines across both orders, bringing the attempted readable-cache total to 119,692. Both raw text and the reader's hyphen-welded text were searched. The invalid 136-byte cache entry has no PDF text layer.

There were seven raw regex-hit observations, listed exhaustively below. Name resolution was checked against all 15 actual English scene rosters. PDF page indexes are zero-based; line numbers here are one-based within the raw text layer, before hyphen welding.

| Location | Matching line | Classification and cue result |
|---|---|---|
| `0fe883861a983e5a818ad82b.pdf`, flat, page 33, line 8 | `CANCION DE ARIEL.` | Actual song heading, between Ariel's singing entrance and `Venid á hollar...`; resolves to ARIEL. |
| Same PDF, coordinates, page 33, line 7 | `CANCION DE ARIEL. .` | Same song heading; resolves to ARIEL after label normalization. |
| Same PDF, flat, page 126, line 37 | `cancion de vida ejemplar ?` | Dialogue continuing the Fool's preceding question. Regex matches, but `resolve("vida ejemplar ?", roster)` is `None` for every roster. No song cue fires. |
| Same PDF, coordinates, page 126, line 27 | `cancion de vida ejemplar ?` | Same dialogue and same rejected name. |
| `391af551b32d458ef1ce8888.pdf`, flat, page 51, line 14 | `Canto de Áriel` | Actual song heading, after Ariel's singing entrance and before `Desembarca...`; resolves to ARIEL. |
| Same PDF, coordinates, page 51, line 12 | `Canto de Áriel` | Same song heading; resolves to ARIEL. |
| `config/source_banks/shakespeare/translations/es/tempest_1_2.txt:102` | `CANCION DE ARIEL.: Ladra el mastin:` | Serialized label plus dialogue, not an own-line heading. It is 35 characters, outside the 34-character caller limit, and the captured name `ARIEL.: Ladra el mastin` resolves to nobody. |

Thus the raw regex is not exclusive to headings, but no real false positive passes the full condition at `scripts/otr_vendor_scan.py:1198` and line 1206. Claiming zero non-heading regex matches would be false; claiming zero observed erroneous song cues is supported by these measurements.

**6. REFUTED: the newly added `d'` alternative misses an attached singer name.**

`scripts/otr_vendor_scan.py:210` puts mandatory whitespace after every preposition, including `d'`:

```python
r"(?:de|di|d'|of)\s+(?P<who>[^\n]{2,30}?)\s*[.:]?$"
```

Reproduced with the new matcher and the actual speech parser:

```python
roster = {"PROSPERO", "ARIEL", "MIRANDA"}
assert S.resolve("Ariel", roster) == "ARIEL"
assert S._SONG_HEADING.match("Chanson d'Ariel") is None
assert S._SONG_HEADING.match("Chanson d' Ariel") is not None
span = "PROSPERO\nCome here.\nChanson d'Ariel\nDance softly now.\nMIRANDA\nI hear you."
S.speeches_from_span(span, roster)
# [('PROSPERO', "Come here. Chanson d'Ariel Dance softly now"),
#  ('MIRANDA', 'I hear you')]
```

The attached-apostrophe form leaves the heading and song with PROSPERO despite a resolvable ARIEL. Adding the artificial space produces three speeches with ARIEL owning `Dance softly now`. This is a constructed false-negative test of newly added code, not a claim that a cached scene currently contains that French heading. Question 5's corpus search for false positives does not cover it.

The permitted test run was `python -B -m pytest -q -p no:cacheprovider` on `tests/test_vendor_scan_furniture.py` and `tests/test_verbatim_corpus.py`, with pytest scratch under `%TEMP%` and plugin autoload disabled: **165 passed in 2.92 seconds**. The sole warning was an unknown asyncio configuration option with that plugin disabled. Those tests do not exercise the repeated-tail counterexample, downstream fold redirections, or attached `d'Ariel` form above. Measurement script and JSON receipts remain under `%TEMP%\otr_review_18ffc540.py` and `%TEMP%\otr_review_18ffc540\`.

VERDICT: MUST-FIX: Closed scenes can be falsely refused, accepted folds still redirect already-bound speakers, and the new song matcher misses attached d' names.
