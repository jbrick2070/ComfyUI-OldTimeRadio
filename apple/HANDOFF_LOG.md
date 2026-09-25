## 2026-09-25 -- No quant pack, no Wan, no harnesses; LTX 2.5 downloads itself

Branch `claude/practical-hamilton-ftvd2j` (draft PR #5). Operator directives:
no quant-format mentions outside the Bug Bible, no harnesses, no rigs, no saved
fixtures, delete all history (every dated `docs/2026-*` folder, `kibitz-runs/`,
this log's older entries and `GO_FORWARD_ARCHIVE.md` -- all recoverable from git
history), and "less friction for the end user, auto download things to work".

What the pack is now: the canonical graph plus 24 variants generated from
`config/workflow_matrix.json` rows -- the only workflow definitions. The LTX 2.5
lanes are native safetensors on stock loaders and fetch their own weights at
queue time from ungated mirrors (about 25 GB for the 16 GB lanes). Removed:
`wan_ti2v`, `fastwan_8gb`, `ltx_video`, `ltx_audio_in`, the quant LTX 2.5 lanes,
the DMD sampler node (24 nodes now), 98 harness scripts, 72 rigs,
`tests/fixtures/`. Needed reference docs moved to `apple/`.

Verification: full suite on the Linux cloud box (no GPU, no Windows paths) --
zero new failures against the 8ffb09d baseline (289 baseline reds, 225 now).
`build_variants --check`: 24 variants, 0 failures. Sonnet refute-QA ran on the
pushed diffs; findings folded in. NOT yet proven on hardware: the LTX 2.5
queue-time download and the 16 GB silent lane on its new weights. Bug Bible not
run from here -- its repo is not reachable in the cloud session.
