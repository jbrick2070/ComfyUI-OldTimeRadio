<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

NO -- build-blocking items remain.

BUILD-BLOCKING items:
1. [ARCHITECTURE / I/O CONTRACTS / M0-M2] LTX-AV graph is not coder-actionable: M2 says “winning-lane graph,” ARCH says node gate names missing classes, and TESTING requires mocked `NODE_CLASS_MAPPINGS`, but no node-class candidate list, loader widget names, terminal node id, or graph inputs are specified. Existing `eng_ltx_video.py` has this explicitly in `_node_candidates()` / `_build_graph()`; `eng_ltx_av.py` cannot be built to the same standard from this doc. Exact fix: add an M0-required artifact in `M0_RESULTS.md`: exact LTX-AV graph topology, node class candidates, widget/input names, terminal node, loader artifact names, and which fields differ for talk vs music; M1/M2 must consume that artifact.

2. [ARCHITECTURE / HARDWARE / Additive touch list] Weight gate requires checking three artifacts — transformer, encoder, video VAE — but only `OTR_LTX_AV_CKPT` and `OTR_LTX_AV_TEXT_ENCODER` are named. There is no resolver/env/default for the video VAE path, yet ARCH requires `realpath EXISTS + size >= ... video VAE >= 1 GiB`. Exact fix: define `OTR_LTX_AV_VAE` or an explicit default VAE lookup rule/name in M0 output and in `eng_ltx_av.py`; update `scripts/download_ltx_2_3.ps1` to pull/document it.

3. [I/O CONTRACTS / NEW `nodes/_otr_shared/av_dims.py`] `av_dims` cannot be implemented/tested as written: it must “RAISE with nearest-valid,” but the valid dimension rule is not defined here: stride, min/max, aspect buckets, or exact allowed tuples are absent. Existing `eng_ltx_video.py` only floors to multiples of 32; this plan implies a stricter LTX-AV rule. Exact fix: add the valid-dim contract before build: either exact allowed tuples from M0 or a deterministic rule `(stride, min_w/h, max_w/h, aspect constraints)` plus nearest-valid tie-break.

4. [HARDWARE / M0 PROBE] Gate “quality >= 2B A/B” is not evaluable: no definition of “2B,” comparison set, rater count, pass rubric, or artifact path. Exact fix: define the rubric in `M0_RESULTS.md` template, e.g. named reference clips + scoring labels + minimum acceptable classification, or move this to non-blocking look-QA.

5. [WIRING / Additive touch list / registry.py grounding] Plan says “Flag-off = RENDER-TIME degrade (ShotLock never asserts; registry docstring corrected),” but grounded `registry.py` docstring currently says `OTR_ShotLock` calls `assert_usable` to fail closed. Touch list edits only registry docstrings, not ShotLock. VERIFY-AT-BUILD: if ShotLock actually calls video `assert_usable` during lock, flag-off dark lanes will fail before render-time fallback. Exact fix: either add the ShotLock behavior change to the touch list or explicitly confirm no ShotLock assert path exists for these engines; docstring-only edit is insufficient if behavior exists.

SHOULD-CONSIDER:
- [M4 GATES] “obs playable AAC only” should be rewritten as an ffprobe command/check; current wording is ambiguous.
- [M1/M3] Storm-line tests are split confusingly: M1 says all 3 test files green, M3 says storm-line emission tests. Keep storm tests in the ticket that implements run_episode summary/storm lines.

Ticket-cut proposal:
1. Ticket A — M0 probe artifacts only. Done: `M0_RESULTS.md` contains graph topology/node candidates/widgets, artifact paths+hashes+sizes incl. VAE, valid-dim rule/table, max-frame value, encoder mode, NVML rows, and quality rubric result.
2. Ticket B — CPU contracts + dark registration. Done: `eng_ltx_av.py` metadata/assert gates only, `av_dims.py`, schemas family map, role_compat music audio_ref, guarded import, registry docs; unit tests for dims/schema/assert gates pass; no render path required.
3. Ticket C — Driver wiring + CPU tests. Done: render_driver deltas a-i, prompt gates, slice-cache key, force guard, alias, summary/storm emission tests; full CPU suite green and untouched byte-identical checks green.
4. Ticket D — Graph/render lane + live gates. Done: M2 graph/preflight/lease/phasing/silent encode/trim-pad/max-frames from M0; M4 full suite + Bug Bible + forced-lane smoke + ffprobe/manifest/grep/NVML gates pass.