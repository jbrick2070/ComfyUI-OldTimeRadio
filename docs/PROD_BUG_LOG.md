# Production Bug Log (staging pre-Bible)

**Contract (operator, 2026-07-10):** Claude appends entries here AUTONOMOUSLY, but
ONLY for bugs that actually failed in a live/prod run (live render, headless lane,
soak, published episode). Dev/audit/review catches get fixed, never logged. NO entry
here touches the Bug Bible directly -- at ship time the operator triggers a BUG
FAN-OUT over this log, which promotes approved entries into the survival-guide
Bible in bulk under the Three-File Contract (YAML + README count + regression
test, one commit). Promoted entries get marked `PROMOTED <bible-id>`; rejected
ones get marked `REJECTED` and stay for the record. Append-only, newest last.
Running tests/bug_bible_regression.py after every code change stays automatic
and is unrelated to this log.

Entry format:

```
## PBUG-YYYYMMDD-NN -- short title
- surfaced: <which live run: smoke/soak/episode + date>
- symptom: <one line, what the operator/log saw>
- root cause: <one line>
- fix: <commit sha + one line>
- verify idea: <candidate machine check for the future bible test>
- bible-worthy: <yes/no guess + why -- operator decides at fan-out>
- status: OPEN | PROMOTED <id> | REJECTED
```

---

(no entries yet -- log opened 2026-07-10)
