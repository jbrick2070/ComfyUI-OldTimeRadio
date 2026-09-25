## 2026-09-25 -- HEAD 9b9e0766 +handoff (main) -- RENDER (16 GB 1-act rotation, registry 2.3.3)

Did: Stills descriptions now say z_image_turbo. Lumina is not a default in any
  profile JSON. Mac stays sd15. Cloud and Google rows were not touched.
  Commits 248bf651 (descriptions) and 9b9e0766 (pyproject 2.3.2 -> 2.3.3).
  Registry publish Action 36103232501 succeeded. GET
  /nodes/comfyui-old-time-radio/versions shows version 2.3.3 status
  NodeVersionStatusPending (created 2026-09-25T06:31:46Z). Zip contents were
  not checked. Active waits on Comfy-Org's scan.
  Live 1-acts on this 5080, canonical graph, act_count forced to 1. Published
  to C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\:
  covenant_ink_20260924_232810__arch__stmo__zimg__koko__pubd__g412__sa3_final.mp4
  (otr_16gb_still),
  notched_key_20260924_234023__scif__l25v__zimg__koko__orig__g412__sa3_final.mp4
  (otr_16gb_video),
  black_fog_20260925_011016__shst__n16f__zimg__koko__sspr__g412__sa3_final.mp4
  (otr_16gb_foley). All three filenames say zimg.
  Mime prompt ea876b5c-3646-4b37-8603-a92fe79a4657 was still running at
  handoff (server log on beat shot_001_b15, ltx25_native_mime_16gb).
  AnimateDiff (otr_16gb_animatediff) is next. Chain script
  C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_chain.ps1 repeats stills,
  video, foley, mime, animatediff until 2026-09-25 08:00 local or the first
  failure. Server log: C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_server.log.
  Do not kill python and do not free port 8000.
  Full pytest suite and Bug Bible were not run. A render occupied the box.
  Dirty and uncommitted, do not git add with other work: a partial red-test
  patch (worktree commit 40aef8ab, branch
  worktree-agent-aaf3612b6248a9d9e). The EXPECTED_RED ledger edit and the
  google client quota_state rename were rejected and are not in the tree.
  The rest is unstaged: apple/evidence/video_evidence_manifest.json,
  nodes/story_orchestrator.py (LLM-slot comments only), and the test files
  named in git status. apple/MACHINES.md shows modified from CRLF only.
  Operator asked for a Stable Audio 3 house/techno check of prompt, inputs,
  temperature, and seed. Not done. Rendering was the priority. The
  music-is-done ruling still covers cue-wording chases.
Current step: mime 1-act still on the GPU. AnimateDiff has not started.
Next: leave the chain alone until 08:00 or a RESULT FAIL. Then take the
  next GO_FORWARD row. Do not reset the box while :8000 is serving this chain.
Models: no panel. Description strings plus the version bump. No Composer or
  Sonnet QA on that diff.
Commits: 248bf651, 9b9e0766. The sha above is the second-to-last on the
  branch; the last is this handoff commit.

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
