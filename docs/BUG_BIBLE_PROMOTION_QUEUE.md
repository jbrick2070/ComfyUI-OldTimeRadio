# Bug Bible Promotion Queue

**Updated:** 2026-07-12

**Production staging log:** `docs/PROD_BUG_LOG.md`

**Portable rules:** `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\BUG_BIBLE.yaml`

This is a thin approval queue, not a second bug log. Full symptoms, evidence,
fixes, and verification ideas remain in `PROD_BUG_LOG.md`. A row appears here
only after a real production artifact/run proves the failure, the root cause is
known, the rule is generalizable, overlap with the Bible is checked, and the
operator approves promotion.

## Pending approved candidates

**None.**

Audit at survival-guide `72e1cd3`: every explicit `promotion: BUG-*` mapping and
every `PROMOTED BUG-*` status in `PROD_BUG_LOG.md` exists in
`BUG_BIBLE.yaml`. No already-promoted incident is duplicated here.

## Not eligible yet

| Production record | Why it is not queued |
|---|---|
| `PBUG-20260712-17` | Live failure, but root cause and reusable law are still open |
| `PBUG-20260702-01` | Live VRAM/zero-clip incident without an isolated root cause or fix |
| `PBUG-20260703-01` | Environmental Ollama outage; no production-proven launcher preflight fix |
| `PBUG-20260711-18` | Predicted by analysis, not observed in a live artifact; fails the admission rule |
| `PBUG-20260711-15` | Superseded by `PBUG-20260711-16`, already represented by `BUG-12.51` |

## Candidate row contract

When a candidate becomes eligible, add only:

| PBUG | Production evidence | Generalized law | Automatable verify | Overlap check | Proposed phase/area | Approval |
|---|---|---|---|---|---|---|

After the Three-File Contract lands in the survival-guide repo (YAML, README
count, regression coverage), remove the pending row and stamp the mapping in the
append-only production log. Static review findings and invented fixtures never
enter this file.
