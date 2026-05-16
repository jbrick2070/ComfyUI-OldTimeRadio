# UNEXPECTED FINDING -- workflow JSON byte-count divergence between main and sprint-c-story-brief-v2

**Status:** CAPTURE-ONLY. Operator directive 2026-05-16: halt investigation, capture findings, await operator review. No file modifications, no reverts, no cherry-picks, no pushes.

**Captured by:** retrospective-triage Cowork window (Deliverable 1 verification side-finding).

## File

`workflows/otr_scifi_16gb_full.json` (full path: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json`)

## Byte counts (git blob, authoritative)

| Branch | Commit (tip) | Blob SHA | Size (bytes) |
|---|---|---|---|
| `main` | `0aa6d6e` | `0f8cfba81c6701bf8fd5f3f2b7b69ce40ae8e576` | **22314** |
| `sprint-c-story-brief-v2` | `a125a35` | `212a38161ff7d79e9407d328329ccccd1acbb05a` | **44049** |
| Working copy (bash sandbox view) | -- | -- | 44049 (with first NUL byte observed at offset 22314 in sandbox read; see "Sandbox vs git" below) |

Delta: sprint-c-story-brief-v2 blob is **21735 bytes larger** than the main blob.

## xxd tail -20 -- main blob (`0f8cfba...`, 22314 bytes)

```
000055f0: 666f 223a 207b 0a20 2020 2020 2020 2020  fo": {.
00005600: 2020 2022 6e61 6d65 223a 2022 4f6c 642d     "name": "Old-
00005610: 5469 6d65 2052 6164 696f 2053 6369 2d46  Time Radio Sci-F
00005620: 6920 416e 7468 6f6c 6f67 7922 2c0a 2020  i Anthology",.
00005630: 2020 2020 2020 2020 2020 2261 7574 686f            "autho
00005640: 7222 3a20 224a 6566 6672 6579 2042 7269  r": "Jeffrey Bri
00005650: 636b 222c 0a20 2020 2020 2020 2020 2020  ck",.
00005660: 2022 6465 7363 7269 7074 696f 6e22 3a20   "description":
00005670: 2246 756c 6c20 7069 7065 6c69 6e65 3a20  "Full pipeline:
00005680: 4c4c 4d20 7363 7269 7074 7320 2b20 4261  LLM scripts + Ba
00005690: 726b 2054 5453 202b 2053 7061 7469 616c  rk TTS + Spatial
000056a0: 2041 7564 696f 202b 2043 5254 2056 6964   Audio + CRT Vid
000056b0: 656f 2e20 7632 2e30 2056 6973 7561 6c20  eo. v2.0 Visual
000056c0: 4472 616d 6120 456e 6769 6e65 2070 6c61  Drama Engine pla
000056d0: 6365 686f 6c64 6572 7320 696e 636c 7564  ceholders includ
000056e0: 6564 2e22 2c0a 2020 2020 2020 2020 2020  ed.",.
000056f0: 2020 2276 6572 7369 6f6e 223a 2022 312e    "version": "1.
00005700: 352e 3022 0a20 2020 2020 2020 207d 0a20  5.0".        }.
00005710: 2020 207d 2c0a 2020 2020 2276 6572 7369     },.    "versi
00005720: 6f6e 223a 2030 2e34 0a7d                 on": 0.4.}
```

Final byte = `0x7d` (`}`). Clean JSON close. No NUL bytes anywhere in this blob.

## xxd tail -20 -- sprint-c-story-brief-v2 blob (`212a381...`, 44049 bytes)

```
0000aae0: 0a20 2020 2020 205d 0a20 2020 207d 2c0a  .      ].    },.
0000aaf0: 2020 2020 2269 6e66 6f22 3a20 7b0a 2020      "info": {.
0000ab00: 2020 2020 226e 616d 6522 3a20 224f 6c64      "name": "Old
0000ab10: 2d54 696d 6520 5261 6469 6f20 5363 692d  -Time Radio Sci-
0000ab20: 4669 2041 6e74 686f 6c6f 6779 222c 0a20  Fi Anthology",.
0000ab30: 2020 2020 2022 6175 7468 6f72 223a 2022       "author": "
0000ab40: 4a65 6666 7265 7920 4272 6963 6b22 2c0a  Jeffrey Brick",.
0000ab50: 2020 2020 2020 2264 6573 6372 6970 7469        "descripti
0000ab60: 6f6e 223a 2022 4675 6c6c 2070 6970 656c  on": "Full pipel
0000ab70: 696e 653a 204c 4c4d 2073 6372 6970 7473  ine: LLM scripts
0000ab80: 202b 2042 6172 6b20 5454 5320 2b20 5370   + Bark TTS + Sp
0000ab90: 6174 6961 6c20 4175 6469 6f20 2b20 4352  atial Audio + CR
0000aba0: 5420 5669 6465 6f2e 2076 322e 3020 5669  T Video. v2.0 Vi
0000abb0: 7375 616c 2044 7261 6d61 2045 6e67 696e  sual Drama Engin
0000abc0: 6520 706c 6163 6568 6f6c 6465 7273 2069  e placeholders i
0000abd0: 6e63 6c75 6465 642e 222c 0a20 2020 2020  ncluded.",.
0000abe0: 2022 7665 7273 696f 6e22 3a20 2231 2e35   "version": "1.5
0000abf0: 2e30 220a 2020 2020 7d0a 2020 7d2c 0a20  .0".    }.  },.
0000ac00: 2022 7665 7273 696f 6e22 3a20 302e 340a   "version": 0.4.
0000ac10: 7d                                       }
```

Final byte = `0x7d` (`}`). Clean JSON close. **No NUL bytes in this blob either.** Indentation is 2-space (vs main's 12-space at the same logical depth) -- the blobs encode the same logical workflow but with different formatting/whitespace conventions, and sprint-c-story-brief-v2 has additional content (e.g. node 15, node 13/14 widgets fully populated per BUG-LOCAL-031/032 canonicalization).

## Sandbox vs git blob (separate concern, not the size jump)

The Linux bash sandbox mounted at `/sessions/.../mnt/ComfyUI-OldTimeRadio/` read the working copy as 44049 bytes with the **first NUL byte at offset 22314** and the remaining 21735 bytes all NULs -- which contradicted the git blob's content (44049 bytes of clean text, no NULs). This is most likely a **sandbox/mount read artifact**, not a real on-disk file defect: Windows-side `git status --short` reported no modifications, and the Desktop Commander cmd shell read the same blob clean via `git cat-file -p`. Capture-only -- do not act on this discrepancy without a Windows-side independent read confirming it.

## Commit history of the file on sprint-c-story-brief-v2 (`git log --oneline -- workflows/otr_scifi_16gb_full.json`)

121 commits touched the file across the full history reachable from `sprint-c-story-brief-v2`. The 17-commit Sprint C window contributed:

```
851f54e C2a: era literal cleanbreak -- visual layer (FLUX portrait fallback/widget/sig + _DEFAULT_STYLE_TAIL + workflow JSON via string-based replace)
```

(C2a is the only Sprint C commit that touched the workflow JSON. C0a/C0b/C1/C2b/C3/C3b/C4/C5a1/C5a2/C5b/C5c/C5d/C5e/C5f/C5g/C-final did not touch this file.)

Older commits that touched it, in walk order back to v1.4 era, include (truncated -- full list available via the git log command):

```
ee67d9c fix(v2.0-alpha): TITLE_STUCK + WordExtend NameError + fatal-streak halt
dabcebd fix(workflow): canonicalize widgets_values shapes across nodes (BUG-LOCAL-032)
7ba2ffd fix(workflow): Node 13 widgets_values add speed=0.95 (BUG-LOCAL-031)
bf362ce fix(workflow): Node 11 widgets_values temperature=0.7 (BUG-LOCAL-030)
787e944 Rename target_minutes to target_words (INT spinner, 350-10000, 140 wpm pacing)
...
```

## Blob-size walk (capture pass, descending from sprint-c-story-brief-v2 HEAD)

Highlights -- not exhaustive, but enough to localize the size jump:

| Commit | Size (bytes) | Note |
|---|---|---|
| `851f54e` | 44049 | C2a -- Sprint C touch, current sprint-c-story-brief-v2 size |
| `bc883b1` | 44052 | pre-Sprint-C (B6 writer/workflow residuals) |
| `9e6b27b` | 43160 | cleanbreak(s26-A4b): workflow fixture textual scrub -- script_json widget '[]' -> '{}' |
| `af4e655` | 28735 | bug_bible_regression 24 passed 2 xfailed |
| `ee67d9c` | 22314 | fix(v2.0-alpha): TITLE_STUCK + WordExtend NameError + fatal-streak halt -- **same size as main** |
| `dabcebd` | 22368 | fix(workflow): canonicalize widgets_values shapes (BUG-LOCAL-032) |
| `068bf54` | 28751 | Visual pipeline tests: 256/256 passed -- **size jumps from 20241 to 28751** |
| `a0893d3` | 20241 | Workflow JSON cleanup: strip emoji, stage-numbered titles (cosmetic only) |
| `a5274fd` | 20037 | Workflow JSON: remove STAGE 4 VISUAL SIDECAR group rectangle |

The file size oscillates across the history as nodes were added, removed, and re-canonicalized. Main's 22314 byte version corresponds approximately to the `ee67d9c` blob shape; sprint-c-story-brief-v2's 44049 byte version is the cumulative growth from `068bf54` (visual pipeline tests, +8514 bytes vs the immediately prior commit) through `af4e655` (+16021 bytes vs `068bf54`) and onward to its current form.

## Candidate commit that introduced the size jump

**Two candidates surfaced in the 2-minute capture window:**

1. `068bf54` -- "Visual pipeline tests: 256/256 passed.)" -- jumped the blob from ~20K to ~28K (+8.5K bytes). This commit's subject suggests test-suite work, not workflow JSON content, so the size growth here may be hidden additions worth a closer read.
2. `af4e655` -- "bug_bible_regression 24 passed 2 xfailed. AST OK.)" -- jumped from ~28K to ~46K (+16K bytes). Same pattern: regression-pass subject, but a 16K JSON growth landed in the same commit.

**Candidate commit isolated:** **partially.** Both `068bf54` and `af4e655` are size-jump suspects. **Full root cause not isolated in capture pass** -- a deeper diff of those two commits' workflow JSON deltas is required to determine whether they added legitimate new content (more nodes, more links, more widgets) or accidentally bloated the file via a paste-merge or formatting regression.

## Operator decision items (NOT executed by this capture)

1. Confirm sandbox NUL observation is a sandbox/mount artifact -- a Windows-side `Get-Content -Encoding Byte | Measure-Object -Sum` or `certutil -encodehex` on the working copy will resolve.
2. Decide whether the 21735-byte delta between main and sprint-c-story-brief-v2 is intentional (legitimate node/link/widget additions during the Sprint A/B/C/D... lineage) or hidden bloat from paste-merge artifacts.
3. If intentional: no action; capture this finding for future merge-coordination.
4. If bloat: a Sprint A or Sprint G cleanup commit could pretty-print the JSON to canonical form and re-baseline.

Out of scope for this retrospective-triage. Waiting on operator review.
