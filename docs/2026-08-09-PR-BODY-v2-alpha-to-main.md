# PR: `v2.0-alpha` -> `main`

**Title:**

    v2.0-alpha -> main: v2 release promotion

---

## Merge facts (measured 2026-08-09, not estimated)

| | |
|---|---|
| Source | `v2.0-alpha` @ `e16e9a63` |
| Target | `main` @ `0aa6d6e1` |
| Merge base | `178dd840` (2026-04-15, "v1.7: README update") |
| Commits ahead | **3445** |
| Commits behind | 11 -- all April 2026 v1.6/v1.7 commits |
| Content main has that the branch lacks | **NONE** (`git diff v2.0-alpha...main` is empty) |
| Merge conflicts (dry run) | **0** |

No fork: this is a branch-to-branch PR inside `jbrick2070/ComfyUI-OldTimeRadio`.
The "11 behind" is not real divergence -- those commits' content is already
present on the branch, which is why the three-dot diff is empty and
`git merge-tree` finds nothing to reconcile.

## Green-on-CI note

A local working tree may show 3 failures
(`test_engine_matrix_doc::test_the_doc_matches_the_live_registry` and two in
`test_ltx_8gb_canonical_canvas`). They come from an **uncommitted**
`render_canvas = (832, 480)` edit to `nodes/_otr_video_engines/eng_wan_i2v.py`
that never reached origin. A fresh clone of `v2.0-alpha` does not have it.
That edit is a real VRAM fix and should land in its own commit, together with a
regenerated `docs/ENGINE_MATRIX.md` and updates to those two canvas tests.

## What landed in the final session before this PR (2026-08-09)

Thirteen commits, `36d695f6..e16e9a63`. Every one pushed and lockstep-verified;
`workflows/otr_canonical.json` is byte-identical across the whole session.

**Production defects fixed**

* `088dabc8` -- the composite's ComfyUI cache key hardcoded one engine id and
  one checkpoint filename, and resolved the model differently from the loader.
  **This was live on the render server, not latent:** the headless config maps
  `upscale_models` at a directory holding no `.pth`, so the checkpoint is
  reachable only through the repo-relative fallback -- the exact lookup the old
  fingerprint never consulted. Swapping weights would not have invalidated the
  composite on the box that publishes episodes.
* `5fdf93f1` -- `scripts/validate_canonical_workflow.py` could **never** run its
  contract check. The package dir is `ComfyUI-OldTimeRadio` (hyphen) and ComfyUI
  loads it by path, so `import ComfyUI_OldTimeRadio` was permanently
  unsatisfiable; the error path returned "no problems" and the script exited 0.
  The item-8 receipt "clean (23 nodes, 56 links)" was the skip path.
* `e16e9a63` -- `visual_style_receipt["attempts"]` always reported 1. The
  shared `on_attempt_complete` contract that three callers depend on is now
  pinned against the real ladder for the first time.
* `22012263` -- a `~latest` model alias left no provenance on writer-only runs
  (the resolved-model record lived only on the video path), and
  `workflows/otr_story_only.json` could not be submitted at all due to a stale
  size-suffixed widget value.

**Durability**

* `262dfa8f` -- every shipped creative slug is now a `~family-latest` pointer
  rather than a version pin. Proven live: the alias resolved to
  `anthropic/claude-opus-5` while the pin it replaced still said
  `claude-opus-4.8`.
* `15f23044` -- corrected a false "blocked on an API key" claim that had been
  carried in the plan for a day; the key was in the User environment all along.

**Live receipts**

* Item 8 chip 4 DISCHARGED: `spandrel_esrgan` proven on `cuda:0`, 7/7 segments
  through the model path, `Prompt executed in 00:41:04`, `obs_publish OK`.
* A `wan_ti2v` VRAM defect found and documented with a matched `ltx_video`
  control arm proving it is engine-specific:
  `docs/2026-08-09-PROBLEM-STATEMENT-wan-ti2v-inter-shot-vram-retention.md`.
  **Not fixed here -- a separate window owns that engine.**

**Companion repo** (`comfyui-custom-node-survival-guide`, `905e85c` +
`656c36e`): BUG-12.87 promoted for the gate-false-green class, and
`BUG_BIBLE.yaml` now actually `yaml.safe_load`s -- it never had, while the
README called it machine-readable, because every structural check counted
entries by regex.

## Known-open, carried forward

* `wan_ti2v` inter-shot VRAM retention (documented, owned elsewhere).
* `SpandrelEsrgan._resolve_model` robustness pair -- an unreadable non-winning
  candidate aborts the search; a double-stat TOCTOU window. **Requires a kibitz
  panel before any code** (third touch of that logic, two-strikes rule).
* `_otr_structured_call.py:1142-1153` skips `notify_attempt` on the
  deterministic-repair branch; unreachable from the fixed caller, reachable from
  `_otr_scifi_codex.py`.
* `otr_upscaled_dir()` is dead code since the RTXUpscale rip.
* Lemmy Phases 2-4 never shipped (`bec0ca79` was Phase 1), which still holds the
  SF#1 chips and the 20-clip accept-rate measurement.
